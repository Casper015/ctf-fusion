"""Regression tests for tools/screens/p3_search_host/diagnose.py (task T0090).

    python -m unittest discover -s tools/screens/p3_search_host/tests

They live beside the tool because tests/ is reserved by another task. No compiler, arena or host is needed.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import diagnose  # noqa: E402

REPO = HERE.parents[3]
COMMITTED = REPO / "reports" / "host" / "agent_v2_p3_search"


def counters(turns, used, fallback):
    return {"turns": turns, "refused": 0, "inconsistent": 0, "search_used": used, "search_fallback": fallback,
            "leaves": 0, "trajectories": 0, "refuse_setup": 0, "refuse_trajectory": fallback, "refuse_expand": 0,
            "refuse_critic": 0, "refuse_late": 0, "max_setup_us": 0, "max_traj_us": 0, "max_expand_us": 0,
            "max_critic_us": 0, "max_search_us": 0, "branches": 0, "critic": "fp32"}


def tail(c, overrides, kept):
    return ("AGENTV2 " + " ".join(f"{k} {c[k]}" for k in diagnose.LINE_FIELDS)
            + f" critic {c['critic']} overrides {overrides} gate_kept {kept}")


def game(variant, teams, i, used, fallback, overrides, kept, turns=100):
    c = counters(turns, used, fallback)
    return {"variant": variant, "fixture_id": f"s1-n{teams:02d}-m{i:05d}", "teams": teams, "turns_played": turns,
            "counters": c, "stderr_tail": [tail(c, overrides, kept)]}


def synthetic() -> dict:
    return {
        "schema": diagnose.PREFLIGHT_SCHEMA, "seed": 1, "counts": [2, 5], "maps_per_count": 2,
        "variants": {"s": {"search": True}, "d": {"search": False}},
        "games": [game("s", 2, 0, 100, 0, 30, 10), game("s", 2, 1, 100, 0, 20, 0),
                  game("s", 5, 0, 40, 60, 4, 4), game("s", 5, 1, 0, 100, 0, 0),
                  {"variant": "d", "fixture_id": "s1-n02-m00000", "teams": 2, "turns_played": 100, "counters": None}],
    }


def numbers(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers(v)


class DiagnoseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def put(self, data) -> Path:
        p = self.tmp / "PREFLIGHT.json"
        p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return p

    def refused(self, argv) -> str:
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(diagnose.main(argv), 2)
        return err.getvalue()

    def test_missing_input_is_refused_and_writes_nothing(self):
        out = self.tmp / "out"
        self.assertIn("not found", self.refused(["--preflight", str(self.tmp / "absent.json"), "--out", str(out)]))
        self.assertFalse(out.exists())

    def test_changed_input_is_refused(self):
        p = self.put(synthetic())
        self.assertIn("expected", self.refused(["--preflight", str(p), "--expect-sha256", "0" * 64,
                                                "--out", str(self.tmp / "out")]))
        good = hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(diagnose.main(["--preflight", str(p), "--expect-sha256", good, "--out", str(self.tmp / "o")]), 0)

    def test_corrupt_or_foreign_input_is_refused(self):
        no_coverage, no_seed, null_game = synthetic(), synthetic(), synthetic()
        del no_coverage["maps_per_count"]
        del no_seed["seed"]
        null_game["games"].append(None)
        for bad in ("{not json", json.dumps({"schema": "other", "games": [1]}),
                    json.dumps({"schema": diagnose.PREFLIGHT_SCHEMA, "games": []}), json.dumps([1, 2]),
                    json.dumps(no_coverage), json.dumps(no_seed), json.dumps(null_game)):
            with self.subTest(bad=bad[:40]):
                self.assertIn("refused", self.refused(["--preflight", str(self.put(bad)), "--out", str(self.tmp / "o")]))

    def test_a_valid_change_in_the_input_changes_the_result(self):
        data = synthetic()
        before = diagnose.diagnose(self.put(data))["dimensions"]["dim5_compute_allocation"]["rows"]["s n05"]
        data["games"][3] = game("s", 5, 1, 50, 50, 10, 5)
        after = diagnose.diagnose(self.put(data))["dimensions"]["dim5_compute_allocation"]["rows"]["s n05"]
        self.assertEqual((before["completed_share"], after["completed_share"]), (0.2, 0.45))
        self.assertEqual((before["overrides"], after["overrides"]), (4, 14))

    def test_known_counts(self):
        rows = diagnose.diagnose(self.put(synthetic()))["dimensions"]["dim5_compute_allocation"]["rows"]
        self.assertEqual(set(rows), {"s n02", "s n05"})
        self.assertEqual(rows["s n02"]["completed_share"], 1.0)
        self.assertEqual(rows["s n02"]["overrides"], 50)
        self.assertEqual(rows["s n02"]["override_share_of_completed"], 0.25)
        self.assertEqual(rows["s n05"]["completed_share"], 0.2)
        self.assertEqual(rows["s n05"]["search_fallback"], 160)
        self.assertEqual(rows["s n05"]["override_share_of_turns"], 0.02)

    def test_unreadable_games_never_count_as_measured(self):
        def consistent(c, overrides=4, kept=4):
            return lambda g: (g.update(counters=c), g.update(stderr_tail=[tail(c, overrides, kept)]))

        cases = {
            "no counters": lambda g: g.update(counters=None),
            "row disagrees with stderr": lambda g: g["counters"].update(search_used=41),
            "negative row counter": lambda g: g["counters"].update(search_fallback=-1),
            "float row counter": lambda g: g["counters"].update(search_used=40.0),
            "critic differs": lambda g: g.update(stderr_tail=[g["stderr_tail"][0].replace("critic fp32", "critic junk")]),
            "negative in both": lambda g: (g["counters"].update(search_fallback=-1),
                                           g.update(stderr_tail=[g["stderr_tail"][0].replace(
                                               "search_fallback 60", "search_fallback -1")])),
            "refused turn counted as search": consistent(counters(100, 40, 60) | {"refused": 1}),
            "zero turns": lambda g: (consistent(counters(0, 0, 0))(g), g.update(turns_played=0)),
            "fixture from another count": lambda g: g.update(fixture_id="s1-n02-m00000"),
            "fixture from another seed": lambda g: g.update(fixture_id="s2-n05-m00000"),
            "no override line": lambda g: g.update(stderr_tail=["AGENTV2 turns 100 refused 0"]),
            "truncated line": lambda g: g.update(stderr_tail=[g["stderr_tail"][0].rsplit(" gate_kept", 1)[0]]),
            "stale line": consistent(counters(99, 40, 59)),
            "impossible overrides": consistent(counters(100, 40, 60), overrides=41, kept=0),
            "used + fallback != turns": consistent(counters(100, 40, 59) | {"turns": 100}),
            "refusals != fallback": consistent(counters(100, 40, 60) | {"refuse_setup": 999999}),
            "duplicate fixture": lambda g: g.update(fixture_id="s1-n05-m00001"),
        }
        for why, break_it in cases.items():
            with self.subTest(why=why):
                data = synthetic()
                break_it(data["games"][2])
                alloc = diagnose.diagnose(self.put(data))["dimensions"]["dim5_compute_allocation"]
                self.assertEqual(alloc["status"], "partial")
                self.assertEqual(alloc["rows"]["s n05"]["status"], "unavailable")
                self.assertNotIn("completed_share", alloc["rows"]["s n05"])
                self.assertEqual(alloc["rows"]["s n02"]["status"], "computed")

    def test_missing_or_extra_coverage_is_never_computed(self):
        data = synthetic()
        data["games"] = [g for g in data["games"] if not (g["variant"] == "s" and g["teams"] == 5)]
        alloc = diagnose.diagnose(self.put(data))["dimensions"]["dim5_compute_allocation"]
        self.assertEqual((alloc["status"], alloc["rows"]["s n05"]["status"]), ("partial", "unavailable"))
        data = synthetic()
        data["games"].append(game("s", 5, 2, 0, 100, 0, 0))
        alloc = diagnose.diagnose(self.put(data))["dimensions"]["dim5_compute_allocation"]
        self.assertEqual(alloc["rows"]["s n05"]["status"], "unavailable")
        data = synthetic()
        data["games"].append(game("s", 9, 0, 0, 100, 0, 0))
        self.assertEqual(diagnose.diagnose(self.put(data))["dimensions"]["dim5_compute_allocation"]["status"], "partial")

    def test_no_dimension_reports_a_number_it_did_not_compute(self):
        report = diagnose.diagnose(self.put(synthetic()))
        for name, dim in report["dimensions"].items():
            if name == "dim5_compute_allocation":
                continue
            self.assertEqual(dim["status"], "unavailable", name)
            self.assertEqual(list(numbers(dim)), [], name)
        self.assertNotIn("primary", json.dumps(report).lower())

    def test_markdown_numbers_come_from_the_json(self):
        p = self.put(synthetic())
        diagnose.write(diagnose.diagnose(p), self.tmp / "a")
        diagnose.write(diagnose.diagnose(p), self.tmp / "b")
        for f in ("SEARCH_FAILURE_DIAGNOSIS.json", "SEARCH_FAILURE_DIAGNOSIS.md"):
            self.assertEqual((self.tmp / "a" / f).read_bytes(), (self.tmp / "b" / f).read_bytes(), f)
        js = (self.tmp / "a" / "SEARCH_FAILURE_DIAGNOSIS.json").read_text(encoding="utf-8")
        table = (self.tmp / "a" / "SEARCH_FAILURE_DIAGNOSIS.md").read_text(encoding="utf-8").split("## Search work")[1]
        allowed = {f"{v:.4f}" if isinstance(v, float) else str(v) for v in numbers(json.loads(js))}
        for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", table):
            self.assertIn(tok, allowed)

    @unittest.skipUnless((COMMITTED / "PREFLIGHT.json").is_file(), "no committed T0081 preflight")
    def test_committed_diagnostics_are_regenerated_not_edited(self):
        report = diagnose.diagnose(COMMITTED / "PREFLIGHT.json")
        self.assertEqual(report["dimensions"]["dim5_compute_allocation"]["status"], "computed")
        diagnose.write(report, self.tmp / "regen")
        for f in ("SEARCH_FAILURE_DIAGNOSIS.json", "SEARCH_FAILURE_DIAGNOSIS.md"):
            self.assertEqual((COMMITTED / "diagnostics" / f).read_bytes(), (self.tmp / "regen" / f).read_bytes(), f)
        rows = report["dimensions"]["dim5_compute_allocation"]["rows"]
        self.assertEqual(rows["agent_v2_p3exp_p3_gw8x4_fp32 n05"]["completed_share"], 0.548)
        self.assertEqual(rows["agent_v2_p3exp_p2_gw8x4_fp32 n05"]["completed_share"], 0.9722)


if __name__ == "__main__":
    unittest.main()
