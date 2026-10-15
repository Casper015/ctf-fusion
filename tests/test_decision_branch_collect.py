"""Shortlist logs and branch collection over them (task T0036, training/decision_branch_collect.py).

A committed recorded match gets synthetic shortlist logs on its designated seat; play is an injected fake with
synthetic scores. Nothing here plays a game: the hook's C++ side is tested in bots/p_only/tests/test_decision_hook.py.
"""
from __future__ import annotations

import copy
import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fusion import matchlog
from fusion.labels import pairwise_outcome
from fusion.paths import REPO
from training import decision_branch_collect as B
from training import decision_dataset as D

CARRIER = "c" * 64
MATCH = REPO / "roster" / "report" / "results" / "matches" / "n04" / "n04-s1-000003.json.gz"


def deviation(order: str, k: int, skip: int = 0) -> str:
    tok = order.split()
    tok[2 * k] = [m for m in "SUDLR" if m != tok[2 * k]][skip]
    return " ".join(tok)


def log_line(turn: int, order: str, block: str, stage="refine", entries=None, root=None) -> str:
    entries = entries or [(order, "1.5", "2.5", "beam"), (deviation(order, 0), "1", "3", "beam"),
                          (deviation(order, 1), "2", "1", "beam"), (deviation(order, 2), "0.5", "na", "beam")]
    head = f"{turn} {order} # dsl1 stage={stage} k={len(entries)} inc=0 root={root or B.root_digest(block)} surv=-1 checked=0"
    return head + "".join(f" | {o};{b};{r};{s}" for o, b, r, s in entries)


def fixture_match(logged_turns=range(6, 60)) -> dict:
    m = copy.deepcopy(matchlog.load(MATCH))
    m["seats"][0] = dict(m["seats"][0], name="@candidate", sha256=CARRIER)
    for t in logged_turns:
        line = m["replies"][t - 1][0]
        if line is None:
            continue
        m["replies"][t - 1][0] = log_line(t, B.order_of(line), m["blocks"][t - 1])
    return m


def fixture_frozen(match: dict) -> dict:
    opp = {s["name"]: {"binary": "bin/x", "binary_sha256": "b" * 64, "source_sha256": s["sha256"], "family": s["family"]}
           for s in match["seats"][1:]}
    return {"identity": "f" * 64, "designated": {"binary": "bin/carrier", "binary_sha256": CARRIER}, "opponents": opp,
            "engine": {"sha256": "e" * 64, "commit": "test"}}


def fake_play(engine, match, seat, turn, line, commands, limits):
    rng = np.random.default_rng(abs(hash((line, turn))) % (2 ** 32))
    score = rng.integers(0, 6, match["teams"]).tolist()
    return {"line": line, "score": score, "outcome": float(pairwise_outcome(score)[seat]), "prefix_mismatch": 0,
            "prefix_mismatch_raw": 0, "prefix_examples": [], "prefix_faults": 0, "faults": [],
            "root_block": match["blocks"][turn - 1], "next_block": match["blocks"][turn]}


class Logs(unittest.TestCase):
    def setUp(self):
        self.m = matchlog.load(MATCH)
        self.t = 20
        self.order = B.order_of(self.m["replies"][self.t - 1][0])
        self.block = self.m["blocks"][self.t - 1]

    def test_root_digest(self):
        self.assertEqual(B.root_digest(self.block), B.root_digest(self.block))
        self.assertNotEqual(B.root_digest(self.block), B.root_digest(self.m["blocks"][self.t]))
        self.assertRegex(B.root_digest(self.block), r"^[0-9a-f]{16}$")

    def test_parse_and_reject(self):
        line = log_line(self.t, self.order, self.block)
        log = B.parse_log(line, self.t)
        self.assertEqual((log["stage"], log["k"], log["entries"][0]["order"]), ("refine", 4, self.order))
        self.assertIsNone(log["entries"][3]["refined"])
        self.assertIsNone(B.parse_log(f"{self.t} {self.order}", self.t))                   # no log at all
        bad = {
            "reply not first": log_line(self.t, self.order, self.block,
                                        entries=[(deviation(self.order, 0), "1", "1", "beam"), (self.order, "1", "1", "beam")]),
            "duplicate": log_line(self.t, self.order, self.block, entries=[(self.order, "1", "1", "beam")] * 2),
            "k": line.replace("k=4", "k=5"),
            "number": line.replace(";1.5;", ";inf;"),
            "invalid order": line.replace(f" | {deviation(self.order, 0)};", " | S S S S S S;"),
        }
        for name, text in bad.items():
            with self.subTest(name), self.assertRaises((ValueError, KeyError)):
                B.parse_log(text, self.t)

    def test_deciding_scores_by_stage(self):
        refine, _ = B.deciding_scores(B.parse_log(log_line(self.t, self.order, self.block), self.t))
        self.assertEqual(refine["scale"], "refined")
        self.assertEqual([e["score"] for e in refine["entries"]], [2.5, 3.0, 1.0])            # "na" refined is left out
        beam, _ = B.deciding_scores(B.parse_log(log_line(self.t, self.order, self.block, stage="beam"), self.t))
        self.assertEqual([e["score"] for e in beam["entries"]], [1.5, 1.0, 2.0, 0.5])
        surv = [(self.order, "1", "na", "survivor"), (deviation(self.order, 0), "5", "na", "search"),
                (deviation(self.order, 1), "0.5", "na", "survivor")]
        search, _ = B.deciding_scores(B.parse_log(log_line(self.t, self.order, self.block, "search", surv), self.t))
        self.assertEqual(len(search["entries"]), 2)                                            # survivors only
        for stage in ("cut", "v_cut", "duel", "search+v"):
            got, why = B.deciding_scores(B.parse_log(log_line(self.t, self.order, self.block, stage), self.t))
            self.assertIsNone(got)
            self.assertTrue(why)
        unscored = [(self.order, "na", "na", "incumbent"), (deviation(self.order, 0), "1", "1", "beam")]
        got, why = B.deciding_scores(B.parse_log(log_line(self.t, self.order, self.block, "beam", unscored), self.t))
        self.assertIsNone(got)
        self.assertIn("own order", why)

    def test_select_candidates(self):
        scored, _ = B.deciding_scores(B.parse_log(log_line(self.t, self.order, self.block, stage="beam"), self.t))
        picked = B.select_candidates(scored, 2)
        self.assertEqual(picked[0]["order"], self.order)
        self.assertEqual([e["score"] for e in picked[1:]], [2.0, 1.0])

    def test_pick_root_counts_reasons(self):
        m = fixture_match(range(6, 30))
        m["replies"][9][0] = log_line(10, B.order_of(m["replies"][9][0]), m["blocks"][9], root="0" * 16)
        root, reasons = B.pick_root(m, 0)
        self.assertIsNotNone(root)
        self.assertEqual(reasons["root digest differs from the recorded block"], 1)
        self.assertGreater(reasons["no log"], 0)
        self.assertEqual(B.pick_root(m, 0)[0]["turn"], root["turn"])                         # deterministic


class Collect(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.match = fixture_match()
        self.mdir = self.tmp / "matches"
        matchlog.save(self.mdir / "n04" / f"{self.match['match_id']}.json.gz", self.match)
        self.frozen = fixture_frozen(self.match)
        self.fpath = self.tmp / "FROZEN.json"
        self.fpath.write_text(json.dumps(self.frozen))
        self.out = self.tmp / "branches"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_collect(self, **kw):
        args = dict(play_fn=fake_play, reproduce_fn=lambda e, m, t: True, verify_fn=lambda f, r: [])
        return B.collect(self.fpath, self.mdir, self.out, None, **(args | kw))

    def rows(self):
        out = []
        for f in sorted(self.out.rglob("*.json.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                out.append(json.load(fh))
        return out

    def test_rows_repeats_and_dataset_ingestion(self):
        s = self.run_collect()
        self.assertEqual((s["roots"], s["played"]), (1, 1))
        row = self.rows()[0]
        self.assertEqual(row["candidates"][0], D.F.normalize_line(self.match["replies"][row["turn"] - 1][0]))
        self.assertEqual(len(row["candidates"]), 3)                  # refine: the reply + 2 refined alternatives
        self.assertEqual(row["repeats"], 4)
        self.assertEqual(len(row["branches"]), 12)
        self.assertEqual(len({b["repeat_key"] for b in row["branches"]}), 12)
        for b in row["branches"]:
            self.assertAlmostEqual(b["outcome"], float(pairwise_outcome(b["score"])[0]))
        self.assertTrue(row["bot_state_mechanism"].startswith("engine_forced"))
        self.assertEqual(row["carrier_scores"]["orders"], row["candidates"])
        built, stats = D.build(sorted(self.out.rglob("*.json.gz")), self.mdir, log=lambda *_: None)
        self.assertEqual(stats["roots_with_carrier_scores"], 1)
        d = next(D.to_arrays(built["decision"][s_]) for s_ in D.SPLITS if built["decision"][s_]["root_match"])
        np.testing.assert_allclose(d["cand_carrier_score"], row["carrier_scores"]["scores"])
        self.assertEqual(list(d["cand_n"]), [4, 4, 4])

    def test_resume_and_identity(self):
        self.run_collect()
        self.assertIn("input_binding", self.rows()[0])
        s = self.run_collect()
        self.assertEqual((s["complete_before"], s["played"], s["stale"]), (1, 0, {}))
        with self.assertRaises(SystemExit):                          # another configuration, same output
            self.run_collect(repeats=2)

    # Review of PR #33 [P1]: a stored row is bound to its exact input, not only to the collector identity.

    def save(self, m: dict) -> None:
        matchlog.save(self.mdir / "n04" / f"{self.match['match_id']}.json.gz", m)

    def relog(self, m: dict, t: int, **kw) -> None:
        m["replies"][t - 1][0] = log_line(t, B.order_of(m["replies"][t - 1][0]), m["blocks"][t - 1], **kw)

    def mutations(self, t: int) -> dict:
        order = B.order_of(self.match["replies"][t - 1][0])

        def score(m):                                                # the review's reproduction: 2.5 -> 123.5
            self.relog(m, t, entries=[(order, "1.5", "123.5", "beam"), (deviation(order, 0), "1", "3", "beam"),
                                      (deviation(order, 1), "2", "1", "beam"), (deviation(order, 2), "0.5", "na", "beam")])

        def orders(m):
            self.relog(m, t, entries=[(order, "1.5", "2.5", "beam"), (deviation(order, 0, 1), "1", "3", "beam"),
                                      (deviation(order, 1), "2", "1", "beam"), (deviation(order, 2), "0.5", "na", "beam")])

        def root(m):                                                 # the same turn stays the root (log re-bound)
            lines = m["blocks"][t - 1].split("\n")
            i = next(k for k, x in enumerate(lines) if x.startswith("U ") and x.split()[5] == "100")
            u = lines[i].split()
            u[5] = "99"
            lines[i] = " ".join(u)
            m["blocks"][t - 1] = "\n".join(lines)
            self.relog(m, t)

        def prefix(m):                                               # an opponent's recorded reply before the root
            p = t - 3
            m["replies"][p - 1][1] = f"{p} " + deviation(B.order_of(m["replies"][p - 1][1]), 0)
        return {"scores": score, "candidates": orders, "root_block_sha256": root, "prefix_sha256": prefix}

    def test_changed_input_is_replayed_not_reused(self):
        self.run_collect()
        t = self.rows()[0]["turn"]
        for field, mutate in self.mutations(t).items():
            with self.subTest(field):
                shutil.rmtree(self.out)
                self.save(self.match)
                self.run_collect()
                m = copy.deepcopy(self.match)
                mutate(m)
                self.save(m)
                s = self.run_collect()
                self.assertEqual((s["complete_before"], s["played"]), (0, 1))
                (why, n), = s["stale"].items()
                self.assertEqual(n, 1)
                self.assertIn(field, why)
                self.assertIn("match_sha256", why)
                rows = self.rows()
                self.assertEqual(len(rows), 1)                       # the stale row is set aside, not ingested
                self.assertEqual(rows[0]["turn"], t)
                self.assertEqual(len(list((self.out / "stale").iterdir())), 1)
                if field == "scores":
                    self.assertEqual(rows[0]["carrier_scores"]["scores"][0], 123.5)
                again = self.run_collect()                           # now bound to the new input: reused
                self.assertEqual((again["complete_before"], again["played"], again["stale"]), (1, 0, {}))

    def test_root_turn_change_sets_the_old_row_aside(self):
        self.run_collect()
        t = self.rows()[0]["turn"]
        m = copy.deepcopy(self.match)
        self.relog(m, t, stage="cut")                                # that turn no longer has deciding scores
        self.save(m)
        s = self.run_collect()
        self.assertEqual((s["played"], s["stale"]), (1, {"root turn changed": 1}))
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0]["turn"], t)

    def test_no_root_any_more_sets_the_old_row_aside(self):
        self.run_collect()
        plain = copy.deepcopy(self.match)
        plain["replies"] = matchlog.load(MATCH)["replies"]
        self.save(plain)
        s = self.run_collect()
        self.assertEqual((s["roots"], s["stale"]), (0, {"root turn changed": 1}))
        self.assertEqual(self.rows(), [])

    def test_dropped_rows_are_bound_too(self):
        self.run_collect(reproduce_fn=lambda e, m, t: False)
        t = self.rows()[0]["turn"]
        self.assertEqual(self.run_collect(reproduce_fn=lambda e, m, t: False)["complete_before"], 1)
        m = copy.deepcopy(self.match)
        self.mutations(t)["scores"](m)
        self.save(m)
        s = self.run_collect()
        self.assertEqual((s["complete_before"], s["played"]), (0, 1))
        self.assertNotIn("dropped", self.rows()[0])

    def test_unbound_row_is_not_reused(self):
        self.run_collect()
        rp = next(self.out.rglob("*.json.gz"))
        with gzip.open(rp, "rt", encoding="utf-8") as fh:
            row = json.load(fh)
        del row["input_binding"]
        B.write_gz(rp, row)
        s = self.run_collect()
        self.assertEqual((s["played"], s["stale"]), (1, {"input changed: unbound": 1}))

    def test_source_mismatch_refused(self):
        frozen = copy.deepcopy(self.frozen)
        frozen["designated"]["binary_sha256"] = "d" * 64
        self.fpath.write_text(json.dumps(frozen))
        with self.assertRaises(SystemExit):
            self.run_collect()
        frozen = copy.deepcopy(self.frozen)
        next(iter(frozen["opponents"].values()))["source_sha256"] = "0" * 64
        self.fpath.write_text(json.dumps(frozen))
        with self.assertRaises(SystemExit):
            self.run_collect()

    def test_frozen_identities_must_verify(self):
        with self.assertRaises(SystemExit):
            self.run_collect(verify_fn=lambda f, r: ["binary of @candidate changed or missing"])

    def test_dropped_and_missing_roots_are_reported(self):
        s = self.run_collect(reproduce_fn=lambda e, m, t: False)
        self.assertEqual(s["dropped"], {"recorded blocks not reproduced": 1})
        self.assertIn("dropped", self.rows()[0])
        plain = copy.deepcopy(self.match)
        plain["replies"] = matchlog.load(MATCH)["replies"]
        shutil.rmtree(self.out)
        matchlog.save(self.mdir / "n04" / f"{self.match['match_id']}.json.gz", plain)
        s = self.run_collect()
        self.assertEqual(s["no_root"], {"no usable shortlist in any turn": 1})
        self.assertGreater(s["turn_reasons"]["no log"], 0)

    def test_continuation_tag(self):
        with self.assertRaises(SystemExit):
            self.run_collect(continuation="fast")
        self.run_collect(continuation="fixed_work_diagnostic")
        self.assertEqual(self.rows()[0]["continuation_policy"], "fixed_work_diagnostic")


class _Game:
    """A four-turn stand-in for the engine: decode checks the turn and token count as engine.py does."""
    TURNS = 4

    def __init__(self, teams, map_seed, flag_seed=None):
        self.turn, self.done, self.applied = 0, False, {}
        _Game.last = self

    def encode(self, i):
        return f"TURN {self.turn + 1}\nEND\n"

    def decode(self, i, line):
        tok = line.partition("#")[0].split()
        if not tok or tok[0] != str(self.turn + 1) or len(tok) != 7:
            raise ValueError("not a reply for this turn")
        return {"order": " ".join(tok[1:])}

    def step(self, orders):
        self.turn += 1
        self.applied[self.turn] = dict(orders)
        self.done = self.turn >= self.TURNS

    def result(self):
        return {"score": [1, 0, 0]}


class _Bot:
    """Scripted bots: SCRIPT[(name, turn)] is 'timeout', 'malformed' or 'dead'; otherwise a valid reply."""
    SCRIPT: dict = {}

    def __init__(self, cmd, name, limits):
        self.name, self.turn_limit, self.strikes, self.dead, self.faults = name, limits[1], 0, None, []

    def send(self, text):
        return None if self.dead else 0.0

    def receive(self, started, turn):
        what = self.SCRIPT.get((self.name, turn))
        if self.dead:
            return None
        if what == "timeout":
            return self.strike(turn, "timeout (>50 ms)")
        if what == "dead":
            self.dead = "process exited"
            return None
        return "garbage" if what == "malformed" else f"{turn} S - S - S -"

    def ok(self):
        self.strikes = 0

    def strike(self, turn, reason):
        self.strikes += 1
        self.faults.append((turn, reason))

    def close(self):
        pass


class RootTurnFaults(unittest.TestCase):
    """Review of PR #33 [P2]: faults of the computed root-turn replies are kept although the engine plays the
    recorded and forced orders there."""

    def setUp(self):
        self.match = {"teams": 3, "map_seed": 1, "flag_seed": 2,
                      "seats": [{"name": "@candidate"}, {"name": "a"}, {"name": "b"}],
                      "replies": [[f"{t} S - S - S -", f"{t} U - U - U -", f"{t} D - D - D -"] for t in range(1, 5)]}
        self.forced = "3 L - L - L -"
        self.engine = SimpleNamespace(Game=_Game)

    def play(self, script):
        _Bot.SCRIPT = script
        with patch("fusion.runner.Bot", _Bot):
            return B.play_branch(self.engine, self.match, 0, 3, self.forced, [["x"]] * 3, B.LIMITS)

    def test_root_timeout_and_malformed_reply_are_recorded(self):
        r = self.play({("a", 3): "timeout", ("b", 3): "malformed"})
        self.assertEqual(r["discarded_root_computed_faults"],
                         [[3, 1, "timeout (>50 ms)"], [3, 2, "malformed output: not a reply for this turn"]])
        self.assertEqual(r["root_computed_replies"], ["3 S - S - S -", None, "garbage"])
        self.assertEqual(r["fault_counts"], {"prefix": 0, "root_discarded": 2, "continuation": 0})
        self.assertEqual((r["prefix_faults"], r["faults"]), (0, []))
        root = _Game.last.applied[3]                                 # the engine root action stays forced/recorded
        self.assertEqual(root[0], {"order": "L - L - L -"})
        self.assertEqual(root[1], {"order": "U - U - U -"})
        self.assertEqual(root[2], {"order": "D - D - D -"})

    def test_phases_are_counted_apart(self):
        r = self.play({("a", 2): "timeout", ("b", 3): "dead"})
        self.assertEqual(r["fault_counts"], {"prefix": 1, "root_discarded": 1, "continuation": 0})
        self.assertEqual(r["discarded_root_computed_faults"], [[3, 2, "no reply (process exited)"]])
        self.assertEqual(_Game.last.applied[3][2], {"order": "D - D - D -"})     # recorded order still applied
        r = self.play({("a", 4): "malformed"})
        self.assertEqual(r["fault_counts"], {"prefix": 0, "root_discarded": 0, "continuation": 1})
        self.assertEqual(r["faults"], [[4, 1, "malformed output: not a reply for this turn"]])

    def test_clean_root_records_nothing(self):
        r = self.play({})
        self.assertEqual((r["discarded_root_computed_faults"], r["fault_counts"]["root_discarded"]), ([], 0))
        self.assertEqual(len(r["root_computed_replies"]), 3)


if __name__ == "__main__":
    unittest.main()
