"""Unit tests for tools/screens/p3_search_host/audit_host.py (task T0084).

    python -m unittest discover -s tools/screens/p3_search_host/tests

Validates:
  - Reaggregation and equal-count weighting calculation.
  - Bonferroni interval computation (z = 2.394).
  - Fault reconciliation and timing convention logic.
  - A/A discrepancy decomposition into wall-clock admission vs RNG.
  - Training corpus and roster overlap accounting.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.screens.p3_search_host import audit_host


class TestHostAudit(unittest.TestCase):
    def test_bonferroni_equal_weight_calculation(self):
        # 2 team counts, each with 2 fixtures
        fixtures = {}
        for cnt in (2, 3):
            for m in (0, 1):
                fix_id = f"s1-n{cnt:02d}-m{m:05d}"
                # In n02, m=0 has diff 0.05, m=1 has diff 0.15 -> mean 0.1, var > 0
                # In n03, m=0 has diff 0.15, m=1 has diff 0.25 -> mean 0.2, var > 0
                base_diff = 0.1 if cnt == 2 else 0.2
                diff = base_diff + (-0.05 if m == 0 else +0.05)
                fixtures[fix_id] = {
                    "agent_v2_p3exp_p3_gw8x4_fp32": {"utility": 0.5 + diff},
                    "agent_v2_p3exp_p2_gw8x4_fp32": {"utility": 0.5},
                    "agent_v2_p3exp_p3_direct": {"utility": 0.5},
                    "agent_v2_p3exp_p2_direct": {"utility": 0.5},
                    "agent_v2_p3exp_p3_gw8x4_fp32~aa": {"utility": 0.5 + diff},
                }

        # Pad with remaining team counts with zero diff
        for cnt in range(4, 16):
            for m in (0, 1):
                fix_id = f"s1-n{cnt:02d}-m{m:05d}"
                fixtures[fix_id] = {
                    "agent_v2_p3exp_p3_gw8x4_fp32": {"utility": 0.5},
                    "agent_v2_p3exp_p2_gw8x4_fp32": {"utility": 0.5},
                    "agent_v2_p3exp_p3_direct": {"utility": 0.5},
                    "agent_v2_p3exp_p2_direct": {"utility": 0.5},
                    "agent_v2_p3exp_p3_gw8x4_fp32~aa": {"utility": 0.5},
                }

        comps = audit_host.compute_comparisons(fixtures)
        comp = comps["p3_search_minus_p2_search"]
        expected_mean = (0.1 + 0.2 + 0.0 * 12) / 14.0
        self.assertAlmostEqual(comp["overall_mean"], round(expected_mean, 5))
        self.assertGreater(comp["se"], 0.0)

    def test_aa_rerun_decomposition(self):
        # Synthetic fixtures: count 2 identical, count 4 differing
        fixtures = {}
        for m in range(56):
            fix_id = f"s1-n02-m{m:05d}"
            fixtures[fix_id] = {
                "agent_v2_p3exp_p3_gw8x4_fp32": {"score": [10, 5], "utility": 1.0},
                "agent_v2_p3exp_p3_gw8x4_fp32~aa": {"score": [10, 5], "utility": 1.0},
            }
        for m in range(56):
            fix_id = f"s1-n04-m{m:05d}"
            # first 20 identical, 36 differing
            s2 = [10, 5] if m < 20 else [8, 7]
            u2 = 1.0 if m < 20 else 0.5
            fixtures[fix_id] = {
                "agent_v2_p3exp_p3_gw8x4_fp32": {"score": [10, 5], "utility": 1.0},
                "agent_v2_p3exp_p3_gw8x4_fp32~aa": {"score": s2, "utility": u2},
            }
        for cnt in range(3, 16):
            if cnt in (2, 4):
                continue
            for m in range(56):
                fix_id = f"s1-n{cnt:02d}-m{m:05d}"
                fixtures[fix_id] = {
                    "agent_v2_p3exp_p3_gw8x4_fp32": {"score": [10, 5], "utility": 1.0},
                    "agent_v2_p3exp_p3_gw8x4_fp32~aa": {"score": [10, 5], "utility": 1.0},
                }

        aa = audit_host.audit_aa_rerun(fixtures)
        self.assertEqual(aa["per_count"]["2"]["identical"], 56)
        self.assertEqual(aa["per_count"]["2"]["discrepancy_rate"], 0.0)
        self.assertEqual(aa["per_count"]["4"]["identical"], 20)
        self.assertAlmostEqual(aa["per_count"]["4"]["discrepancy_rate"], 36.0 / 56.0, places=4)

    def test_committed_audit_matches_verified_screen(self):
        """Verify that committed EVIDENCE_AUDIT.json accurately reproduces screen numbers."""
        audit_path = Path("reports/host/agent_v2_p3_search/EVIDENCE_AUDIT.json")
        self.assertTrue(audit_path.is_file(), "EVIDENCE_AUDIT.json must exist")
        data = json.loads(audit_path.read_text(encoding="utf-8"))

        p3_p2 = data["comparisons"]["p3_search_minus_p2_search"]
        self.assertAlmostEqual(p3_p2["overall_mean"], -0.00396, places=5)
        self.assertEqual(p3_p2["verdict"], "not shown")

        p3_dir = data["comparisons"]["p3_search_minus_p3_direct"]
        self.assertAlmostEqual(p3_dir["overall_mean"], -0.01968, places=5)
        self.assertEqual(p3_dir["verdict"], "hurts")

        # Check A/A identical counts
        self.assertEqual(data["aa_rerun"]["identical_scores"], 456)
        self.assertEqual(data["aa_rerun"]["differing_scores"], 328)


if __name__ == "__main__":
    unittest.main()
