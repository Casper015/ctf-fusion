"""Unit tests of the branch diagnostics (task T0085): schemas, keys, selection, analysis, P scoring, estimator.

    python -m unittest discover -s tools/search_diagnostics/tests -t .

No compiler, arena or host is needed here; tests/test_integration.py runs the native programs, the engine and real
bot processes.
"""
from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

from tools.search_diagnostics import analyze, estimator, extract, p_eval, seeds
from tools.search_diagnostics.collect import binding, row_path, validate_row
from tools.search_diagnostics.schemas import (
    SCHEMA_ROOT, SCHEMA_ROW, ValidationError, sha256_text, validate_branch, validate_root, write_gz,
)


def block(turn: int, units: list[tuple]) -> str:
    """A minimal TURN block: units are (id, team, x, y, hp, flag, respawn)."""
    lines = [f"TURN {turn}", "SCORE 0 0", f"UNITS {len(units)}"] + [f"U {' '.join(map(str, u))}" for u in units]
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


def fake_root(root_id="r1", teams=3, match="m1", k=8, v=None, stratum="unconditioned") -> dict:
    stream = "INIT ...\nTURN 1\nEND\n"
    cands = [{"index": i, "joint": i, "prior": 100 - i, "order": f"50 {'SUDLR'[i % 5]} - {'SUDLR'[i // 5 % 5]} - S {i}",
              "v_values": [float(v[i]) if v else 0.1 * i] * 4, "v_terminal": [0] * 4, "leaf_feature_fnv": ["0"] * 4}
             for i in range(k)]
    return {
        "schema": SCHEMA_ROOT, "root_id": root_id, "turn": 50,
        "source": {"match_file": f"{match}.json.gz", "match_sha256": "x", "match_id": match, "teams": teams,
                   "map_seed": 1, "flag_seed": 2, "seat": 0, "seat_sha256": "a", "engine_sha256": "e"},
        "selection": {"rule": "test", "stratum": stratum, "weight": 1.0},
        "history": {"seat_stream": stream, "seat_stream_sha256": sha256_text(stream), "prefix_digest": "p",
                    "root_block_sha256": "b"},
        "tool": {"package": "p2general", "binary_sha256": "t", "bundle_sha256": "u", "tree": [24, 4]},
        "baseline": {"order": cands[0]["order"], "joint": 0, "verified": True},
        "candidates": cands, "candidate_sets": {"primary": list(range(k))},
    }


def branch(c, r, u, status="complete"):
    if status != "complete":
        return {"candidate": c, "repeat": r, "env_key": 1, "actor_key": 2, "status": "refused", "reason": "seat 1 diverged: x"}
    return {"candidate": c, "repeat": r, "env_key": 1 + r, "actor_key": 2 + r, "status": "complete", "final_score": [1, 0, 0],
            "utility": u, "final_turn": 400, "terminal": True, "faults": {"prefix": [], "root": [], "continuation": []},
            "wall_s": 1.0}


def fake_row(root, table, ident="id"):
    """table[c][r] = utility."""
    reps = len(table[0])
    return {"schema": SCHEMA_ROW, "identity": ident, "root_id": root["root_id"],
            "binding": binding(root, "primary", reps),
            "branches": [branch(c, r, table[c][r]) for c in range(len(table)) for r in range(reps)]}


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

    def test_seed_must_be_reserved_and_unused_elsewhere(self):
        d = Path(tempfile.mkdtemp())
        (d / "A_SEEDS.md").write_text("| 20262200 | T0086 |\n")
        (d / "B_SEEDS.md").write_text("| 20262201 | other |\n")
        self.assertEqual(seeds.check_seed(20262200, d / "A_SEEDS.md", d)["seed"], 20262200)
        with self.assertRaises(ValidationError):
            seeds.check_seed(2026220, d / "A_SEEDS.md", d)          # a prefix of a reserved seed is not reserved
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
            "duplicate order": lambda r: r["candidates"][2].update(order=r["candidates"][1]["order"]),
            "stream hash": stream_changed,
            "set not a prefix": lambda r: r["candidate_sets"].update(primary=[0, 2, 3]),
            "nan value": lambda r: r["candidates"][1].update(v_values=[float("nan")] * 4),
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

    def test_branch_rows_fail_closed(self):
        root = fake_root(k=2)
        good = fake_row(root, [[0.5] * 4, [1.0] * 4])
        validate_row(copy.deepcopy(good), "id")
        cases = {
            "other identity": (lambda row: None, "other"),
            "duplicate branch": (lambda row: row["branches"].append(copy.deepcopy(row["branches"][0])), "id"),
            "missing branch": (lambda row: row["branches"].pop(), "id"),
            "utility out of range": (lambda row: row["branches"][0].update(utility=1.5), "id"),
            "bool utility": (lambda row: row["branches"][0].update(utility=True), "id"),
            "not terminal": (lambda row: row["branches"][0].update(final_turn=399), "id"),
            "missing faults": (lambda row: row["branches"][0].pop("faults"), "id"),
            "root fault in a complete branch": (lambda row: row["branches"][0]["faults"]["root"].append([50, 1, "x"]), "id"),
            "unknown status": (lambda row: row["branches"][0].update(status="ok"), "id"),
        }
        for why, (f, ident) in cases.items():
            with self.subTest(why=why):
                row = copy.deepcopy(good)
                f(row)
                with self.assertRaises(ValidationError):
                    validate_row(row, ident)
        validate_branch(branch(0, 0, 0, status="refused"), "refused")


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

    def test_turns_without_a_recorded_reply_are_not_roots(self):
        m = fake_match()
        m["replies"][19][0] = None
        self.assertNotIn(20, [r["turn"] for r in extract.select_roots(m, 0, [("unconditioned", 30)], "s")])

    def test_order_of_strips_turn_and_comment(self):
        self.assertEqual(extract.order_of("7 S - U 3 R P # note"), "S - U 3 R P")
        self.assertIsNone(extract.order_of(None))


class Analysis(unittest.TestCase):
    def write(self, roots_rows):
        d = Path(tempfile.mkdtemp())
        for root, row in roots_rows:
            write_gz(d / "roots" / f"{root['root_id']}.json.gz", root)
            if row is not None:
                write_gz(row_path(d / "rows", root), row)
        return d

    def test_gains_regret_and_cross_fit_are_computed_from_rows(self):
        root = fake_root(k=3, v=[0.0, 0.5, 0.1])
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
        self.assertIsNone(analyze.root_metrics(root, fake_row(root, [r[:2] for r in table]), 0.002)["cross_fit_gain"])

    def test_cross_fit_may_keep_the_baseline(self):
        root = fake_root(k=2)
        m = analyze.root_metrics(root, fake_row(root, [[0.5] * 4, [0.0, 0.5, 0.0, 0.5]]), 0.002)
        self.assertEqual(m["cross_fit_gain"], 0.0)           # both folds keep the baseline (ties go to it)

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
        d = self.write(rows[:2])
        one = analyze.analyze(d / "roots", d / "rows")["groups"]["unsplit/unconditioned"]["in_sample_best_gain"]
        self.assertIsNone(one["se"])
        self.assertEqual(one["per_count"]["3"]["maps"], 1)
        d = self.write(rows)
        two = analyze.analyze(d / "roots", d / "rows")["groups"]["unsplit/unconditioned"]["in_sample_best_gain"]
        self.assertEqual(two["per_count"]["3"], {"maps": 2, "roots": 3, "mean": (0.1875 + 0.375) / 2,
                                                 "se": math.sqrt(((0.1875 - 0.28125) ** 2 * 2) / 1 / 2)})
        self.assertIsNotNone(two["ci95_normal"])

    def test_refused_and_missing_rows_are_excluded_not_imputed(self):
        root, other = fake_root(k=2), fake_root(root_id="r2", k=2)
        row = fake_row(root, [[0.5] * 4, [1.0] * 4])
        row["branches"][3] = branch(0, 3, 0, status="refused")
        d = self.write([(root, row), (other, None)])
        out = analyze.analyze(d / "roots", d / "rows")
        self.assertEqual(out["analysed"], 0)
        self.assertEqual(out["excluded"], {"seat 1 diverged": 1, "not collected": 1})

    def test_rows_without_roots_and_duplicate_roots_are_refused(self):
        root = fake_root(k=2)
        d = self.write([(root, fake_row(root, [[0.5] * 4, [1.0] * 4]))])
        (d / "roots" / "r1.json.gz").unlink()
        write_gz(d / "roots" / "x.json.gz", fake_root(root_id="other", k=2))
        with self.assertRaises(ValidationError):
            analyze.analyze(d / "roots", d / "rows")
        d = self.write([(root, None)])
        write_gz(d / "roots" / "copy.json.gz", root)
        with self.assertRaises(ValidationError):
            analyze.analyze(d / "roots", d / "rows")


class PScoring(unittest.TestCase):
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
