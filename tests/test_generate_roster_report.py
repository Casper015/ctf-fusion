"""Unit test for tools/reporting/generate_roster_report.py."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    from tools.reporting.generate_roster_report import generate_report
except ModuleNotFoundError as e:                   # matplotlib is a report dependency: requirements-dev.txt
    raise unittest.SkipTest(f"{e.name} is not installed (pip install -r requirements-dev.txt)")


class TestGenerateRosterReport(unittest.TestCase):
    def test_report_generation_from_synthetic_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = tmp_path / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            fig_dir = tmp_path / "figures"
            out_report = tmp_path / "REPORT.md"
            
            # Synthetic suite
            suite = {
                "schema": "ctf-probe-suite-1",
                "per_count": 2,
                "probes": [
                    {"probe_id": "n02-p01", "teams": 2},
                    {"probe_id": "n02-p02", "teams": 2},
                ]
            }
            (results_dir / "probe_suite.json").write_text(json.dumps(suite), encoding="utf-8")
            
            # Synthetic distance
            ids = [
                "ctf-arena/Player7/judge/ctf/bots/gpt-6-astra.cpp",
                "ctf-arena/Player7/judge/ctf/bots/opus-5.5.cpp",
                "New_AI_Test/Player2/workspace/experiments/cand-002.cpp"
            ]
            overall_excess = [
                [0.0, 0.25, 0.35],
                [0.25, 0.0, 0.30],
                [0.35, 0.30, 0.0]
            ]
            dist = {
                "schema": "ctf-behavior-distance-2",
                "suite_identity": "test-suite-sha-12345",
                "created": "2026-10-08T20:00:00",
                "params": {
                    "margin": 0.02,
                    "joint_margin": 0.05,
                    "context_margin": 0.05,
                    "z": 1.645,
                    "min_probes": 2,
                    "stochastic_noise": 0.01
                },
                "ids": ids,
                "overall": overall_excess,
                "overall_excess": overall_excess,
                "maps_per_count": {"2": 2},
                "per_count": {"2": overall_excess},
                "excess_per_count": {"2": overall_excess},
                "state_per_count": {"2": [
                    [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0]
                ]},
                "unmeasured_per_count": {"2": []},
                "clusters": [[ids[0]], [ids[1]], [ids[2]]],
                "cluster_of": {ids[0]: 0, ids[1]: 1, ids[2]: 2},
                "cluster_diameter": {"0": 0.0, "1": 0.0, "2": 0.0},
                "noise": {ids[0]: 0.0, ids[1]: 0.0, ids[2]: 0.0},
                "stochastic": [],
                "established_distinct_counts": {ids[0]: [2], ids[1]: [2], ids[2]: [2]},
                "nearest": {
                    ids[0]: {"id": ids[1], "excess": 0.25, "state": "different"},
                    ids[1]: {"id": ids[0], "excess": 0.25, "state": "different"},
                    ids[2]: {"id": ids[1], "excess": 0.30, "state": "different"},
                },
                "distinctive_tags": {
                    ids[0]: [{"tag": "contact", "excess": 0.25}],
                    ids[1]: [{"tag": "contact", "excess": 0.25}],
                    ids[2]: [{"tag": "flag", "excess": 0.30}],
                },
                "coverage": {
                    ids[0]: {"2": {"valid_share": 1.0}},
                    ids[1]: {"2": {"valid_share": 1.0}},
                    ids[2]: {"2": {"valid_share": 1.0}},
                },
                "stale_runs": {}
            }
            (results_dir / "behavior_distance.json").write_text(json.dumps(dist), encoding="utf-8")
            
            # Synthetic env_check
            env_info = {
                "host": "TEST-HOST",
                "cpu": "Synthetic CPU",
                "logical_cpus": 16,
                "compiler_version": "g++ 15.2.0"
            }
            (results_dir / "env_check_TEST-HOST.json").write_text(json.dumps(env_info), encoding="utf-8")
            
            # Run generator
            report = generate_report(results_dir, out_report, fig_dir, run_info={
                "ctf_commit": "abcdef123456",
                "ai_test_commit": "7890abcdef12"
            })
            
            # Assertions on Markdown report
            self.assertTrue(out_report.is_file())
            self.assertIn("## 1. Run Identity & Environment", report)
            self.assertIn("## 2. Reliability & Execution Diagnostics", report)
            self.assertIn("## 3. Evidence States Per Team Count", report)
            self.assertIn("## 4. Behavior Groups & Distinctiveness", report)
            self.assertIn("## 5. Behavioral Distance Visualizations", report)
            self.assertIn("## 6. Nearest Neighbors & Distinctive Context Tags", report)
            self.assertIn("## 7. Measured Live Play Outcomes", report)
            self.assertIn("## 8. Theoretical & Empirical Limitations", report)
            self.assertIn("test-suite-sha-12345", report)
            self.assertIn("TEST-HOST", report)
            self.assertIn("GPT-6-Astra", report)
            self.assertIn("Opus-5.5", report)
            
            # Assertions on generated figure images
            self.assertTrue((fig_dir / "judge_behavior_matrix.png").is_file())
            self.assertTrue((fig_dir / "all_clusters_behavior_matrix.png").is_file())
            self.assertTrue((fig_dir / "style_space_2d.png").is_file())


if __name__ == "__main__":
    unittest.main()
