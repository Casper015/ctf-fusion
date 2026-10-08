"""Synthetic arithmetic/contract tests. These are not model-strength results."""
import itertools
import math
import unittest


from scorecard import evaluate, utility


def plan(counts=(2,), weights=None):
    return {
        "schema": "ctf-scorecard-plan-1", "team_counts": list(counts),
        "stratum_weights_by_suite": {"main": weights or {"elite": 1.0}},
        "bootstrap_reps": 500, "confidence": 0.95, "bootstrap_seed": 8,
        "minimum_independent_groups_total": 2, "minimum_independent_groups_per_count": 2,
        "comparisons": {"p_nll_gain": ["p_baseline", "p_candidate"],
                        "v_mse_gain": ["v_baseline", "v_candidate"]},
    }


def row(name, scores=(0, 1), candidate=(1, 0), stratum="elite", cluster=None):
    return {"fixture_id": name, "cluster_id": cluster or name, "suite": "main",
            "teams": len(scores), "stratum": stratum, "ego": 0,
            "baseline_scores": list(scores), "candidate_scores": list(candidate),
            "metrics": {"p_baseline": [math.log(4), 1], "p_candidate": [math.log(2), 1],
                        "v_baseline": [0.09, 1], "v_candidate": [0.01, 1]}}


class ScorecardTests(unittest.TestCase):
    def test_pairwise_enumeration_and_conservation(self):
        # Independent enumeration cross-check; total pairwise utility is N/2.
        for n in (2, 3, 4):
            for scores in itertools.product(range(3), repeat=n):
                values = []
                for ego in range(n):
                    wins = sum(scores[ego] > scores[j] for j in range(n) if j != ego)
                    ties = sum(scores[ego] == scores[j] for j in range(n) if j != ego)
                    actual = utility(list(scores), ego)
                    self.assertEqual(actual, (2*wins + ties)/(2*(n-1)))
                    values.append(actual)
                self.assertAlmostEqual(sum(values), n/2)
        self.assertEqual(utility([5, 6, 4], 0), 0.5)
        self.assertEqual(utility([5, 6, 5], 0), 0.25)

    def test_hand_calculated_losses(self):
        out = evaluate([row("a"), row("b")], plan())
        overall = out["suites"]["main"]["overall_equal_count"]
        self.assertAlmostEqual(overall["p_nll_gain"]["mean"], math.log(2))
        self.assertAlmostEqual(overall["v_mse_gain"]["mean"], 0.08)
        self.assertEqual(overall["play_gain"]["mean"], 1.0)
        self.assertTrue(out["confirmation_sample_complete"])
        self.assertFalse(out["deployment_authorized"])

    def test_strata_and_team_counts_equal_weight(self):
        records = []
        for suffix in ("a", "b"):
            records += [row("2e"+suffix), row("2m"+suffix, (1, 0), (0, 1), "mixed"),
                        row("3e"+suffix, (1, 0, 0), (0, 1, 1)),
                        row("3m"+suffix, (0, 1, 1), (1, 1, 1), "mixed")]
        # Extra independent 2-team samples must not change the 2 vs 3 count weights.
        records += [row("2eextra"), row("2mextra", (1, 0), (0, 1), "mixed")]
        out = evaluate(records, plan((2, 3), {"elite": 0.8, "mixed": 0.2}))
        report = out["suites"]["main"]
        self.assertAlmostEqual(report["by_count"]["2"]["metrics"]["play_gain"]["mean"], 0.6)
        self.assertAlmostEqual(report["by_count"]["3"]["metrics"]["play_gain"]["mean"], -0.7)
        self.assertAlmostEqual(report["overall_equal_count"]["play_gain"]["mean"], -0.05)

    def test_map_repeats_not_independent_or_extra_gameplay_weight(self):
        records = [row("loss", (1, 0), (0, 1), cluster="map_loss")]
        records += [row(f"win{i}", cluster="map_win") for i in range(100)]
        out = evaluate(records, plan())
        result = out["suites"]["main"]["overall_equal_count"]["play_gain"]
        self.assertEqual(out["independent_main_play_groups"], 2)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["mean"], 0)
        self.assertLess(result["ci"][0], 0)
        self.assertGreater(result["ci"][1], 0)

    def test_missing_scope_not_renormalized(self):
        out = evaluate([row("a"), row("b")], plan((2, 3)))
        self.assertIsNone(out["suites"]["main"]["overall_equal_count"]["play_gain"]["mean"])
        self.assertFalse(out["confirmation_sample_complete"])
        out = evaluate([row("a"), row("b")], plan(weights={"elite": 0.7, "mixed": 0.3}))
        self.assertIsNone(out["suites"]["main"]["by_count"]["2"]["metrics"]["play_gain"]["mean"])

    def test_missing_metric_not_silently_dropped(self):
        records = [row("a"), row("b")]
        records[1]["metrics"] = {}
        out = evaluate(records, plan())
        self.assertIsNone(out["suites"]["main"]["overall_equal_count"]["p_nll_gain"]["mean"])
        self.assertIsNotNone(out["suites"]["main"]["overall_equal_count"]["play_gain"]["mean"])
        p = plan()
        p["comparisons"]["regret_gain"] = ["regret_baseline", "regret_candidate"]
        missing = evaluate(records, p)["suites"]["main"]["overall_equal_count"]["regret_gain"]
        self.assertIsNone(missing["mean"])
        self.assertFalse(missing["complete"])

    def test_no_play_records_not_counted_as_match_confirmation(self):
        records = [row("a"), row("b")]
        for r in records:
            del r["baseline_scores"], r["candidate_scores"]
        out = evaluate(records, plan())
        self.assertEqual(out["independent_main_play_groups"], 0)
        self.assertFalse(out["confirmation_sample_complete"])

    def test_merged_raw_timing_and_deadline_boundary(self):
        records = [row("a"), row("b")]
        for r, turns in zip(records, ([10, 20], [30, 60])):
            r["timing"] = {"candidate": {"first_ms": 2000, "turn_ms": turns, "faults": 0},
                           "baseline": {"first_ms": 100, "turn_ms": [50, 50], "faults": 0}}
        result = evaluate(records, plan())["suites"]["main"]["by_count"]["2"]["runtime"]
        self.assertAlmostEqual(result["candidate"]["post_startup_quantiles_ms"]["p99"], 59.1)
        self.assertEqual(result["candidate"]["first_late"], 0)
        self.assertEqual(result["candidate"]["post_startup_late_rate"], 0.25)
        self.assertEqual(result["baseline"]["post_startup_late"], 0)
        self.assertAlmostEqual(result["baseline"]["zero_late_cluster_upper95"], 1-math.sqrt(0.05))

    def test_bad_input_rejected(self):
        cases = []
        same = [row("a"), row("a")]
        cases.append(same)
        unequal = [row("a"), row("b")]
        unequal[0]["metrics"]["p_candidate"][1] = 2
        cases.append(unequal)
        bad = [row("a"), row("b")]
        bad[0]["metrics"]["p_baseline"][0] = float("nan")
        cases.append(bad)
        crossing = [row("a", cluster="same"), row("b", stratum="mixed", cluster="same")]
        cases.append(crossing)
        for records in cases:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    evaluate(records, plan(weights={"elite": 0.5, "mixed": 0.5}))

    def test_a_a_zero_gain(self):
        records = [row("a"), row("b")]
        for r in records:
            r["candidate_scores"] = r["baseline_scores"][:]
            r["metrics"]["p_candidate"] = r["metrics"]["p_baseline"][:]
            r["metrics"]["v_candidate"] = r["metrics"]["v_baseline"][:]
        out = evaluate(records, plan())["suites"]["main"]["overall_equal_count"]
        for key in ("play_gain", "p_nll_gain", "v_mse_gain"):
            self.assertEqual(out[key]["mean"], 0)
            self.assertEqual(out[key]["ci"], [0, 0])
            self.assertTrue(out[key]["degenerate_bootstrap"])


if __name__ == "__main__":
    unittest.main()
