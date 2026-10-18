"""Tests for tools/search_diagnostics/summarize_pilot.py (task T0092).

    python -m unittest discover -s tools/search_diagnostics/tests -t .

The gate rules on synthetic estimates, missing inputs and metrics, and the committed REPORT pair against the receipts.
"""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tools.search_diagnostics import summarize_pilot as S


def est(mean, ci):
    return {"status": "computed", "counts": 14, "mean": mean, "ci95": ci, "se_status": "computed" if ci else "unavailable",
            "missing_counts": [], "missing_roots": 0}


PARITY = {"status": "passed", "differences": 0, "short_of_minimum": {}, "steps_compared": 1600}
COMPLETION = {"status": "computed", "arms": {"x_search": {"below_half": {"9": 0.2}, "incomplete_counts": []}}}


class Gates(unittest.TestCase):
    def gates(self, cross, spearman, parity=PARITY, completion=COMPLETION):
        return S.gates({"cross_fit_gain": cross, "v_spearman": spearman}, parity, completion)

    def test_an_opportunity_that_survives_cross_fitting_permits_the_scorer(self):
        g = self.gates(est(0.03, [0.01, 0.05]), est(0.0, [-0.1, 0.1]))
        self.assertEqual((g["A"]["decision"], g["A"]["follow_up"], g["B"]["decision"]), ("permitted", "T0088", "does not apply"))

    def test_in_sample_headroom_alone_does_not(self):
        for cross in (est(0.008, [-0.009, 0.024]), est(0.008, None), {"status": "unavailable", "missing_counts": []}):
            with self.subTest(cross=cross.get("ci95")):
                g = self.gates(cross, est(0.0, [-0.1, 0.1]))
                self.assertEqual((g["A"]["decision"], g["B"]["decision"]), ("not permitted", "applies"))
                self.assertIsNone(g["A"]["follow_up"])
                self.assertIn("uncertainty", g["B"]["reading"])

    def test_a_ranking_v_withholds_the_scorer(self):
        g = self.gates(est(0.03, [0.01, 0.05]), est(0.4, [0.2, 0.6]))
        self.assertEqual(g["A"]["decision"], "not permitted")

    def test_parity_problems_trigger_c(self):
        for change in ({"differences": 2}, {"status": "failed"}, {"short_of_minimum": {"capture": 0}}):
            with self.subTest(change=change):
                g = self.gates(est(0, None), est(0, None), parity=PARITY | change)
                self.assertEqual(g["C"]["decision"], "triggered")
        self.assertEqual(self.gates(est(0, None), est(0, None))["C"]["decision"], "not triggered")

    def test_d_follows_the_measured_completion(self):
        self.assertEqual(self.gates(est(0, None), est(0, None))["D"]["decision"], "permitted")
        none_below = {"status": "computed", "arms": {"x_search": {"below_half": {}, "incomplete_counts": []}}}
        self.assertEqual(self.gates(est(0, None), est(0, None), completion=none_below)["D"]["decision"], "not permitted")
        absent = {"status": "unavailable", "reason": "no search_telemetry in the audit"}
        self.assertEqual(self.gates(est(0, None), est(0, None), completion=absent)["D"]["decision"], "unavailable")


class Sections(unittest.TestCase):
    def test_completion_skips_direct_arms_and_incomplete_counts(self):
        def d(used, fallback, complete=True):
            return {"search_used": used, "search_fallback": fallback, "complete": complete,
                    "completion": used / (used + fallback) if complete and used + fallback else None}
        audit = {"schema": "x", "search_telemetry": {"status": "computed", "per_arm": {
            "direct": {"2": d(0, 0)}, "search": {"2": d(90, 10), "9": d(10, 90), "15": d(1, 99, complete=False)}}}}
        out = S.completion_section(audit)
        self.assertEqual(list(out["arms"]), ["search"])
        self.assertEqual(out["arms"]["search"], {"below_half": {"9": 0.1}, "incomplete_counts": ["15"]})

    def test_p_fidelity_uses_only_counts_both_models_have(self):
        def p(nlls):
            return {"scope": "factual", "games": 2, "event_definitions": {"move": "m"},
                    "by_count": {c: {"move": {"count": 10, "nll": v, "top1_accuracy": 0.5}} for c, v in nlls.items()}}
        out = S.p_fidelity(p({"2": 1.0, "3": None}), p({"2": 0.8, "3": 0.7}))["events"]["move"]
        self.assertEqual((out["counts"], out["missing_counts"], out["labels"]), (1, ["3"], 10))
        self.assertAlmostEqual(out["p3_minus_p2_nll"], -0.2)
        none = S.p_fidelity(p({"2": None}), p({"2": 0.8}))["events"]["move"]
        self.assertEqual((none["counts"], none["p2_nll"], none["p3_minus_p2_nll"]), (0, None, None))   # not 0.0

    def test_exclusions_are_grouped_by_opponent(self):
        out = S.exclusions({"excluded": {"seat 1 (Astra) diverged from the record at turn 13": 2,
                                         "seat 4 (Astra) diverged from the record at turn 9": 1,
                                         "seat 2 (Sol) diverged from the record at turn 3": 1, "other": 1}})
        self.assertEqual(out, {"Astra diverged from its recorded game on replay": 3,
                               "Sol diverged from its recorded game on replay": 1, "other": 1})

    def test_signed_metrics_carry_verdicts_and_the_rest_do_not(self):
        block = S.with_verdicts({"cross_fit_gain": est(0.01, [-0.01, 0.03]), "in_sample_best_gain": est(0.09, [0.07, 0.1])})
        self.assertEqual(block["cross_fit_gain"]["verdict"], "not shown")
        self.assertNotIn("verdict", block["in_sample_best_gain"])            # >= 0 by construction


class Receipts(unittest.TestCase):
    def test_committed_report_is_current(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(S.main(["--check"]), 0, err.getvalue())

    def test_missing_inputs_refuse_and_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                S.main(["--dir", tmp])
            self.assertIn("ANALYSIS_PRIMARY.json", str(cm.exception))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_the_report_renders_from_its_json_alone(self):
        doc = json.loads((S.DIR / "REPORT.json").read_text(encoding="utf-8"))
        self.assertEqual(S.render(copy.deepcopy(doc)), (S.DIR / "REPORT.md").read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], S.SCHEMA)


if __name__ == "__main__":
    unittest.main()
