"""Unit tests of the branch diagnostics (task T0085): schemas, bindings, keys, selection, analysis, P labels, estimator.

    python -m unittest discover -s tools/search_diagnostics/tests -t .

No compiler, arena or host is needed here; tests/test_integration.py runs the native programs, the engine and real
bot processes.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.search_diagnostics import analyze, estimator, extract, p_eval, seeds
from tools.search_diagnostics.collect import binding, row_path, validate_row
from tools.search_diagnostics.schemas import (
    PROTOCOL, SCHEMA_MANIFEST, SCHEMA_ROOT, SCHEMA_ROW, ValidationError, digest, sha256_bytes, sha256_text,
    validate_branch, validate_root, write_gz,
)

TEAMS, SEED = 5, 7


def block(turn: int, units: list[tuple], score="0 0") -> str:
    """A minimal TURN block: units are (id, team, x, y, hp, flag, respawn)."""
    lines = [f"TURN {turn}", f"SCORE {score}", f"UNITS {len(units)}"] + [f"U {' '.join(map(str, u))}" for u in units]
    return "\n".join(lines + ["FLAGS 0", "EVENTS 0", "END"]) + "\n"


def fake_match(turns: int = 40) -> dict:
    blocks, replies = [], []
    for t in range(1, turns + 1):
        near = t % 3 == 0                     # contact every third turn
        carry = 1 if t % 5 == 0 else -1       # our unit 0 carries every fifth turn
        units = [(0, 0, 1, 1, 3, carry, -1), (1, 0, 2, 1, 3, -1, -1), (2, 0, 3, 1, 3, -1, -1),
                 (3, 1, 2 if near else 9, 2 if near else 9, 3, -1, -1), (4, 1, 9, 8, 3, -1, -1), (5, 1, 8, 9, 3, -1, -1)]
        blocks.append(block(t, units))
        replies.append([f"{t} S - S - S -", None if t == 12 else f"{t} S - S - S -"])
    return {"match_id": "m1", "teams": 2, "init": "INIT 2 0 15 400 3 1 10 20 20\n", "blocks": blocks, "replies": replies,
            "seats": [{"name": "@candidate", "sha256": "a"}, {"name": "Fable", "sha256": "b"}]}


def fake_root(root_id="r1", match="m1", k=8, v=None, stratum="unconditioned") -> dict:
    stream = "INIT ...\nTURN 1\nEND\n"
    cands = [{"index": i, "joint": i, "prior": 100 - i, "order": f"50 {'SUDLR'[i % 5]} - {'SUDLR'[i // 5 % 5]} - S {i}",
              "v_values": [float(v[i]) if v else 0.1 * i] * 4, "v_terminal": [1] * 4, "leaf_rows": [-1] * 4,
              "leaf_feature_fnv": ["0"] * 4} for i in range(k)]
    return {
        "schema": SCHEMA_ROOT, "root_id": root_id, "turn": 50,
        "source": {"match_file": f"{match}.json.gz", "match_sha256": "x", "match_id": match, "teams": TEAMS,
                   "map_seed": 1, "flag_seed": 2, "seat": 0, "seat_sha256": "a", "engine_sha256": "e"},
        "selection": {"rule": "test", "stratum": stratum, "weight": 1.0},
        "history": {"seat_stream": stream, "seat_stream_sha256": sha256_text(stream), "prefix_digest": "p",
                    "root_block_sha256": "b"},
        "tool": {"package": "p2general", "binary_sha256": "t", "bundle_sha256": "u", "tree": [24, 4]},
        "model": {k: "m" for k in ("carrier", "actor", "actor_digest", "predictor_kind", "predictor", "critic", "resolver")},
        "policy": {"seed": "s", "state_key": "k", "parent_key": "p", "policy_feature_fnv": "f", "mask": "1" * 125,
                   "tree_T": "t", "sampled_joint": 0},
        "leaf_features": {"file": f"{root_id}.leaves.f32", "sha256": sha256_bytes(b""), "rows": 0, "dim": 416},
        "baseline": {"order": cands[0]["order"], "joint": 0, "verified": True},
        "candidates": cands, "candidate_sets": {"primary": list(range(min(k, 8))), "extended": list(range(k))},
    }


def score_for(u: float) -> list[int]:
    """A five-team score vector in which seat 0's pairwise utility is u (a multiple of 1/8)."""
    half = round(u * 8)
    out = [5]
    for _ in range(4):
        step = min(2, half)
        half -= step
        out.append({2: 4, 1: 5, 0: 6}[step])
    return out


def branch(c, r, u, keys, status="complete"):
    k = keys[r]
    if status != "complete":
        return {"candidate": c, "repeat": r, "env_key": k["env_key"], "actor_key": k["actor_key"], "status": "refused",
                "reason": "seat 1 diverged: x"}
    return {"candidate": c, "repeat": r, "env_key": k["env_key"], "actor_key": k["actor_key"], "status": "complete",
            "protocol": PROTOCOL, "final_score": score_for(u), "utility": u, "final_turn": 400, "terminal": True,
            "faults": {"prefix": [], "root": [], "continuation": []}, "wall_s": 1.0}


MANIFEST = {"schema": SCHEMA_MANIFEST, "candidate_set": "primary", "repeats": 4, "scenario_seed": SEED, "evidence": False}
IDENT = digest(MANIFEST)


def fake_row(root, table, cand_set="primary"):
    """table[c][r] = utility."""
    reps = len(table[0])
    keys = seeds.repeat_keys(SEED, root["root_id"], reps)
    return {"schema": SCHEMA_ROW, "identity": IDENT, "root_id": root["root_id"], "binding": binding(root, cand_set, reps),
            "keys": keys, "branches": [branch(c, r, table[c][r], keys) for c in range(len(table)) for r in range(reps)]}


def collection(roots_rows, repeats=4) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "rows").mkdir()
    man = {**MANIFEST, "repeats": repeats}
    (d / "rows" / "MANIFEST.json").write_text(json.dumps({**man, "identity_digest": digest(man)}))
    for root, row in roots_rows:
        write_gz(d / "roots" / f"{root['root_id']}.json.gz", root)
        (d / "roots" / root["leaf_features"]["file"]).write_bytes(b"")
        if row is not None:
            row["identity"] = digest(man)
            write_gz(row_path(d / "rows", root), row)
    return d


class Keys(unittest.TestCase):
    def test_keys_are_common_per_repeat_and_independent_across(self):
        a, b = seeds.repeat_keys(5, "r1", 4), seeds.repeat_keys(5, "r1", 4)
        self.assertEqual(a, b)
        self.assertEqual(len({k["env_key"] for k in a}), 4)
        self.assertNotEqual(a[0]["env_key"], a[0]["actor_key"])
        self.assertNotEqual(seeds.repeat_keys(6, "r1", 1), a[:1])
        self.assertTrue(all(0 < k["env_key"] < 2 ** 63 for k in a))
        with self.assertRaises(ValidationError):
            seeds.check_unique(a + a)

    def test_seed_must_be_reserved_unused_and_unique_to_its_ledger(self):
        d = Path(tempfile.mkdtemp())
        (d / "A_SEEDS.md").write_text("| 20262200 | T0086 primary | Reserved |\n")
        (d / "B_SEEDS.md").write_text("| 20262201 | other | Reserved |\n")
        self.assertEqual(seeds.check_seed(20262200, d / "A_SEEDS.md", d)["seed"], 20262200)
        for text in ("| 20262200 | T0086 |\n", "| 20262200 | T0086 | Reserved | Used |\n", "| 2026220 | Reserved |\n"):
            with self.subTest(text=text):
                (d / "A_SEEDS.md").write_text(text)
                with self.assertRaises(ValidationError):
                    seeds.check_seed(20262200, d / "A_SEEDS.md", d)
        (d / "A_SEEDS.md").write_text("| 20262200 | T0086 | Reserved |\n")
        (d / "C_SEEDS.md").write_text("reused 20262200 here\n")
        with self.assertRaises(ValidationError):
            seeds.check_seed(20262200, d / "A_SEEDS.md", d)


class Schemas(unittest.TestCase):
    def test_a_good_root_passes(self):
        validate_root(fake_root())

    def test_broken_roots_are_refused(self):
        def stream_changed(r):
            r["history"]["seat_stream"] += "x"
        cases = {
            "unverified baseline": lambda r: r["baseline"].update(verified=False),
            "candidate 0 not the baseline": lambda r: r["baseline"].update(order="50 U - U - U -"),
            "candidate 0 not the sampled joint": lambda r: r["policy"].update(sampled_joint=3),
            "duplicate order": lambda r: r["candidates"][2].update(order=r["candidates"][1]["order"]),
            "stream hash": stream_changed,
            "set not a prefix": lambda r: r["candidate_sets"].update(primary=[0, 2, 3]),
            "nan value": lambda r: r["candidates"][1].update(v_values=[float("nan")] * 4),
            "empty values": lambda r: r["candidates"][1].update(v_values=[]),
            "values for another tree": lambda r: r["candidates"][1].update(v_values=[0.1] * 3),
            "leaf row on a terminal leaf": lambda r: r["candidates"][1].update(leaf_rows=[0, -1, -1, -1]),
            "missing leaf rows": lambda r: r["leaf_features"].update(rows=2),
            "missing model": lambda r: r.pop("model"),
            "missing policy field": lambda r: r["policy"].pop("mask"),
            "bool turn": lambda r: r.update(turn=True),
            "missing source": lambda r: r.pop("source"),
            "zero weight": lambda r: r["selection"].update(weight=0),
        }
        for why, f in cases.items():
            with self.subTest(why=why):
                r = fake_root()
                f(r)
                with self.assertRaises(ValidationError):
                    validate_root(r)

    def test_rows_must_bind_to_their_root_identity_and_keys(self):
        root = fake_root(k=2)
        good = fake_row(root, [[0.5] * 4, [1.0] * 4])
        validate_row(copy.deepcopy(good), IDENT, root, "primary", 4, SEED)
        other_root = copy.deepcopy(root)
        other_root["turn"] = 51
        cases = {
            "other identity": (lambda row: row.update(identity="other"), root),
            "missing identity": (lambda row: row.pop("identity"), root),
            "another root's row": (lambda row: None, other_root),
            "duplicate branch": (lambda row: row["branches"].append(copy.deepcopy(row["branches"][0])), root),
            "missing branch": (lambda row: row["branches"].pop(), root),
            "one candidate's env key changed": (lambda row: row["branches"][5].update(env_key=123), root),
            "keys list changed": (lambda row: row["keys"][0].update(actor_key=9), root),
            "wrong protocol": (lambda row: row["branches"][0].update(protocol="other"), root),
            "utility out of range": (lambda row: row["branches"][0].update(utility=1.5), root),
            "utility not the score's": (lambda row: row["branches"][0].update(utility=0.125), root),
            "empty score": (lambda row: row["branches"][0].update(final_score=[]), root),
            "bool utility": (lambda row: row["branches"][0].update(utility=True), root),
            "not terminal": (lambda row: row["branches"][0].update(final_turn=399), root),
            "missing faults": (lambda row: row["branches"][0].pop("faults"), root),
            "root fault in a complete branch": (lambda row: row["branches"][0]["faults"]["root"].append([50, 1, "x"]), root),
            "unknown status": (lambda row: row["branches"][0].update(status="ok"), root),
        }
        for why, (f, r) in cases.items():
            with self.subTest(why=why):
                row = copy.deepcopy(good)
                f(row)
                with self.assertRaises(ValidationError):
                    validate_row(row, IDENT, r, "primary", 4, SEED)
        with self.assertRaises(ValidationError):
            validate_row(copy.deepcopy(good), IDENT, root, "primary", 4, SEED + 1)
        validate_branch(branch(0, 0, 0, seeds.repeat_keys(SEED, "r1", 1), status="refused"), "refused")


class Selection(unittest.TestCase):
    def test_strata_are_disjoint_weighted_and_outcome_blind(self):
        m = fake_match()
        plan = [("unconditioned", 2), ("contact", 2), ("carrier", 1)]
        rows = extract.select_roots(m, 0, plan, "salt")
        turns = [r["turn"] for r in rows]
        self.assertEqual(len(turns), len(set(turns)))
        for r in rows:
            if r["stratum"] == "contact":
                self.assertEqual(r["turn"] % 3, 0)
            if r["stratum"] == "carrier":
                self.assertEqual(r["turn"] % 5, 0)
            self.assertAlmostEqual(r["weight"], r["eligible"] / dict(plan)[r["stratum"]])
            self.assertTrue(extract.MARGIN + 2 <= r["turn"] <= 400 - extract.MARGIN)
        changed = copy.deepcopy(m)                   # scores, events and the result are never read
        changed["result"] = {"score": [9, 0]}
        changed["blocks"] = [b.replace("SCORE 0 0", f"SCORE {i} 0").replace("EVENTS 0", "EVENTS 1\nE return 0")
                             for i, b in enumerate(m["blocks"])]
        self.assertEqual(extract.select_roots(changed, 0, plan, "salt"), rows)
        self.assertNotEqual(extract.select_roots(m, 0, plan, "other"), rows)

    def test_the_override_stratum_needs_the_gate(self):
        m = fake_match()
        rows = extract.select_roots(m, 0, [("override", 2)], "s", {9, 21, 33})
        self.assertTrue(all(r["turn"] in {9, 21, 33} for r in rows))
        with self.assertRaises(ValidationError):
            extract.select_roots(m, 0, [("override", 1)], "s")

    def test_turns_without_a_recorded_reply_are_not_roots(self):
        m = fake_match()
        m["replies"][19][0] = None
        self.assertNotIn(20, [r["turn"] for r in extract.select_roots(m, 0, [("unconditioned", 30)], "s")])

    def test_reply_tokens_keep_the_turn(self):
        self.assertEqual(extract.tokens("7 S - U 3 R P # note"), ["7", "S", "-", "U", "3", "R", "P"])
        self.assertNotEqual(extract.tokens("999 S - S - S -"), extract.tokens("7 S - S - S -"))
        self.assertIsNone(extract.tokens(None))


class Analysis(unittest.TestCase):
    def test_gains_regret_and_cross_fit_are_computed_from_rows(self):
        root = fake_root(k=3, v=[0.0, 0.5, 0.125])
        # baseline 0.5; candidate 1 helps on even repeats only; candidate 2 always +0.25
        table = [[0.5, 0.5, 0.5, 0.5], [1.0, 0.5, 1.0, 0.5], [0.75, 0.75, 0.75, 0.75]]
        m = analyze.root_metrics(root, fake_row(root, table), tau=0.002)
        self.assertEqual(m["gain"], [0.0, 0.25, 0.25])
        self.assertEqual(m["in_sample_best_gain"], 0.25)
        self.assertEqual(m["shortlist_regret"], [0.25, 0.0, 0.0])
        self.assertEqual(m["ranks"], [1.0, 2.5, 2.5])
        # fold A (repeats 0, 2) picks candidate 1 (+0.5), worth 0 on fold B; fold B picks 2 (+0.25), worth +0.25 on A
        self.assertEqual(m["cross_fit_gain"], 0.125)
        self.assertEqual(m["v_gate_pick"], 1)                  # the critic's 0.5 beats the baseline's 0 by > tau
        self.assertEqual(m["v_gate_gain"], 0.25)
        self.assertAlmostEqual(m["gain_repeat_sd"], (math.sqrt(1 / 12) + 0.0) / 2)
        self.assertIsNone(analyze.root_metrics(root, fake_row(root, [r[:2] for r in table]), 0.002)["cross_fit_gain"])

    def test_cross_fit_may_keep_the_baseline(self):
        root = fake_root(k=2)
        m = analyze.root_metrics(root, fake_row(root, [[0.5] * 4, [0.0, 0.5, 0.0, 0.5]]), 0.002)
        self.assertEqual(m["cross_fit_gain"], 0.0)           # both folds keep the baseline (ties go to it)

    def test_the_frozen_gate_sees_only_the_first_eight_roots(self):
        v = [0.0] * 8 + [1.0] + [0.0] * 15
        root = fake_root(k=24, v=v)
        table = [[0.5] * 4 for _ in range(24)]
        m = analyze.root_metrics(root, fake_row(root, table, "extended"), 0.002)
        self.assertEqual(m["v_gate_pick"], 0)
        self.assertEqual(m["v_gate24_pick"], 8)

    def test_gate_and_rank_helpers(self):
        self.assertEqual(analyze.gated_choice([0.0, 0.002, 0.001], 0.002), 0)
        self.assertEqual(analyze.gated_choice([0.0, 0.0021, 0.0021], 0.002), 1)
        self.assertEqual(analyze.gated_choice([0.0, float("nan"), 1.0], 0.0), 0)
        self.assertIsNone(analyze.spearman([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(analyze.spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_aggregation_clusters_by_map_and_never_invents_an_error(self):
        rows = []
        for i, (match, root_id) in enumerate((("mA", "a1"), ("mA", "a2"), ("mB", "b1"))):
            root = fake_root(root_id=root_id, match=match, k=2)
            rows.append((root, fake_row(root, [[0.5] * 4, [0.5 + 0.125 * (i + 1)] * 4])))
        d = collection(rows[:2])
        one = analyze.analyze(d / "roots", d / "rows", counts=[2, 5])["groups"]["unsplit/unconditioned"]["in_sample_best_gain"]
        self.assertIsNone(one["se"])
        self.assertEqual(one["per_count"]["5"]["maps"], 1)
        self.assertEqual(one["missing_counts"], [2])
        d = collection(rows)
        two = analyze.analyze(d / "roots", d / "rows", counts=[5])["groups"]["unsplit/unconditioned"]["in_sample_best_gain"]
        self.assertEqual(two["per_count"]["5"], {"maps": 2, "roots": 3, "mean": (0.1875 + 0.375) / 2,
                                                 "se": math.sqrt(((0.1875 - 0.28125) ** 2 * 2) / 1 / 2)})
        self.assertIsNotNone(two["ci95_normal"])

    def test_refused_and_missing_rows_are_excluded_not_imputed(self):
        root, other = fake_root(k=2), fake_root(root_id="r2", k=2)
        row = fake_row(root, [[0.5] * 4, [1.0] * 4])
        row["branches"][3] = branch(0, 3, 0, row["keys"], status="refused")
        d = collection([(root, row), (other, None)])
        out = analyze.analyze(d / "roots", d / "rows")
        self.assertEqual(out["analysed"], 0)
        self.assertEqual(out["excluded"], {"seat 1 diverged": 1, "not collected": 1})

    def test_unbound_rows_and_missing_leaves_are_refused(self):
        root = fake_root(k=2)
        cases = {
            "row of another root hash": lambda d: write_gz(row_path(d / "rows", root),
                                                           {**fake_row(root, [[0.5] * 4, [1.0] * 4]), "binding": {}}),
            "row without a root": lambda d: (d / "roots" / "r1.json.gz").unlink(),
            "duplicate root": lambda d: write_gz(d / "roots" / "copy.json.gz", root),
            "leaf sidecar changed": lambda d: (d / "roots" / "r1.leaves.f32").write_bytes(b"\0" * 4),
            "manifest changed": lambda d: (d / "rows" / "MANIFEST.json").write_text(
                (d / "rows" / "MANIFEST.json").read_text().replace('"repeats": 4', '"repeats": 3')),
        }
        for why, f in cases.items():
            with self.subTest(why=why):
                d = collection([(root, fake_row(root, [[0.5] * 4, [1.0] * 4]))])
                f(d)
                with self.assertRaises(ValidationError):
                    analyze.analyze(d / "roots", d / "rows")


class PLabels(unittest.TestCase):
    """Feasibility-projected request labels, as fusion/labels.py label_match builds P's training labels."""

    def setUp(self):
        units = [(0, 0, 1, 1, 3, -1, -1), (1, 0, 2, 1, 3, -1, -1), (2, 0, 3, 1, 3, -1, -1),
                 (3, 1, 2, 2, 3, -1, -1), (4, 1, 9, 9, 3, -1, -1), (5, 1, 14, 14, 3, -1, -1)]
        self.match = {"match_id": "m", "teams": 2, "init": "INIT 2 0 15 400 3 1 10 20 20\n", "blocks": [block(1, units)],
                      "replies": [["1 S - S - S -", None]]}
        unit = {"uid": 3, "move": [0.2] * 5, "legal": "11110", "act_legal": ["1001"] * 5,
                "act": [[0.25] * 4] * 5, "target": [[0.5, 0.5] + [0.0] * 6] * 5, "cand": [0, 1, -1, -1, -1, -1, -1, -1]}
        self.rows = [{"schema": "x", "predictor": "p", "turn": 1, "units": [unit]}]

    def score(self, reply):
        self.match["replies"][0][1] = reply
        tallies = {e: p_eval.Tally() for e in p_eval.EVENTS}
        out = p_eval.score_rows(copy.deepcopy(self.rows), self.match, tallies)
        return tallies, out

    def test_an_infeasible_move_is_stay_and_an_effectless_action_wait(self):
        t, _ = self.score("1 R P S - S -")             # R is infeasible (legal[4] = 0); pickup has no effect
        self.assertEqual(t["move"].relabelled["infeasible move projected to stay"], 1)
        self.assertEqual(t["act"].relabelled["effectless action scored as wait"], 1)
        self.assertEqual((t["move"].n, t["act"].n, t["target"].n), (1, 1, 0))

    def test_attack_targets(self):
        t, _ = self.score("1 S 1 S - S -")              # a target in slot 1
        self.assertEqual((t["act"].n, t["target"].n), (1, 1))
        t, _ = self.score("1 S 4 S - S -")              # an ally: not hittable, so a wait
        self.assertEqual(t["act"].relabelled["attack without a valid target scored as wait"], 1)
        self.assertEqual(t["target"].n, 0)

    def test_refused_turns_are_counted(self):
        self.rows.append({"turn": 2, "refused": "malformed block"})
        self.match["blocks"].append(self.match["blocks"][0])
        self.match["replies"].append(["2 S - S - S -", "2 S - S - S -"])
        _, out = self.score("1 S - S - S -")
        self.assertEqual(out["refused_turns"], [[2, "malformed block"]])

    def test_distributions_are_checked(self):
        p_eval.check_distribution([0.25] * 4, "ok")
        for bad in ([2, -1, 0, 0, 0], [0.5, 0.6], [float("nan"), 1.0], [True, 0.0]):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                p_eval.check_distribution(bad, "bad")

    def test_tally(self):
        t = p_eval.Tally()
        t.add([0.5, 0.5], 0)
        t.add([0.9, 0.1], 0)
        r = t.report()
        self.assertEqual(r["count"], 2)
        self.assertAlmostEqual(r["nll"], -(math.log(0.5) + math.log(0.9)) / 2)
        self.assertEqual(r["top1_accuracy"], 1.0)
        t.add([1.0, 0.0], 1)
        self.assertIsNone(t.report()["nll"])
        self.assertEqual(p_eval.Tally().report()["count"], 0)


class Estimate(unittest.TestCase):
    def test_t0086_design_and_unmeasured_time(self):
        e = estimator.estimate()
        self.assertEqual((e["roots"], e["primary_episodes"], e["subset_episodes"], e["episodes"], e["source_games"]),
                         (336, 10752, 1792, 12544, 112))
        self.assertIsNone(e["wall_hours"])
        part = estimator.estimate(measured={3: {"branches": 5, "mean_wall_s": 10.0}})
        self.assertIsNone(part["wall_hours"])
        self.assertEqual(part["per_count"]["3"]["cpu_seconds"], (768 + 128 + 8) * 10.0)
        full = estimator.estimate(measured={n: {"branches": 1, "mean_wall_s": 1.0} for n in range(2, 16)})
        self.assertAlmostEqual(full["wall_hours"], (12544 + 112) / 3600 / 8)
        self.assertIn("source games costed at branch time", full["time_status"])
        with self.assertRaises(ValidationError):
            estimator.estimate({"subset_candidates": 4})

    def test_the_linear_teams_model_is_labelled_and_needs_two_counts(self):
        with self.assertRaises(ValidationError):
            estimator.estimate(measured={3: {"branches": 5, "mean_wall_s": 10.0}}, model="linear-teams")
        e = estimator.estimate(measured={2: {"branches": 4, "mean_wall_s": 2.0}, 10: {"branches": 4, "mean_wall_s": 10.0}},
                               model="linear-teams")
        self.assertAlmostEqual(e["per_count"]["5"]["branch_wall_s"], 5.0)
        self.assertEqual(e["per_count"]["5"]["time_status"], "modelled, not measured (linear-teams)")
        self.assertEqual(e["per_count"]["2"]["time_status"], "measured")
        self.assertTrue(e["time_status"].startswith("partly modelled"))


if __name__ == "__main__":
    unittest.main()
