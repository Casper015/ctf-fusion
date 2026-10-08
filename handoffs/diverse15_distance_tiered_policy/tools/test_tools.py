"""Synthetic utility tests, not evidence about real bots."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from inventory_bots import inventory
from select_roster import select


def catalog(n=16):
    bots = []
    for i in range(n):
        by_count = {str(k): 714 for k in range(2, 16)}
        by_count["2"] = 718
        bots.append({"id": f"synthetic_bot_{i:02d}", "source_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                     "lineage": f"lineage_{i}", "behavior_cluster": f"cluster_{i}", "eligible": True,
                     "independent_maps": 10000, "maps_by_count": by_count, "quality_lcb": 0.9-i*0.01,
                     "skills": [f"synthetic_skill_{i}"], "unique_skill_evidence": "Synthetic fixture for testing only"})
    return {"schema": "ctf-roster-evidence-1", "evidence_manifest_sha256": "0"*64,
            "bots": bots, "distance_ids": [b["id"] for b in bots],
            "distances": [[0.0 if i == j else 0.5 for j in range(n)] for i in range(n)]}


class ToolsTests(unittest.TestCase):
    def test_inventory_alias_and_missing_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/"bots").mkdir()
            (root/"bots/a.cpp").write_text("int main(){return 0;}\n")
            (root/"bots/b.cpp").write_text("int main(){return 0;}\n")
            result = inventory(root, ["bots", "bots", "missing"])
            self.assertEqual(result["source_candidates"], 2)
            self.assertEqual(result["distinct_source_hashes"], 1)
            self.assertEqual(result["bots"][1]["byte_identical_source_of"], "bots/a.cpp")
            self.assertEqual(result["missing_dirs"], ["missing"])
            self.assertTrue(all(b["assessment_status"] == "unmeasured" for b in result["bots"]))
            with self.assertRaises(ValueError):
                inventory(root, ["../"])

    def test_selects_exactly_15_with_evidence(self):
        result = select(catalog())
        self.assertEqual(len(result["selected"]), 15)
        self.assertIn("pending", result["status"])
        self.assertEqual(result["minimum_selected_distance"], 0.5)
        self.assertEqual(len({b["behavior_cluster"] for b in result["selected"]}), 15)

    def test_does_not_pad_duplicate_behavior(self):
        c = catalog(16)
        for b in c["bots"]:
            b["behavior_cluster"] = "same_behavior"
        result = select(c)
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["status"], "no_feasible_roster_found_by_greedy")

    def test_low_distance_and_lineage_limits(self):
        c = catalog(16)
        for row in c["distances"]:
            row[:] = [0.0]*16
        self.assertEqual(select(c)["selected"], [])
        c = catalog(16)
        for b in c["bots"]:
            b["lineage"] = "same_parent"
        self.assertEqual(select(c)["selected"], [])

    def test_insufficient_measurement_is_excluded(self):
        c = catalog(15)
        c["bots"][0]["independent_maps"] = 9999
        result = select(c)
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["status"], "insufficient_qualified_candidates")
        c = catalog(15)
        del c["bots"][0]["maps_by_count"]["15"]
        self.assertEqual(select(c)["selected"], [])

    def test_invalid_evidence_matrix_is_rejected(self):
        for change in ("matrix", "hash", "quality"):
            c = catalog()
            if change == "matrix":
                c["distances"][0][1] = 0.7
            elif change == "hash":
                c["evidence_manifest_sha256"] = "unmeasured"
            else:
                c["bots"][0]["quality_lcb"] = float("nan")
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    select(c)

    def test_id_reordering_preserves_selection(self):
        c = catalog()
        original = select(c)
        reordered = copy.deepcopy(c)
        reordered["bots"].reverse()
        actual = select(reordered)
        self.assertEqual([b["id"] for b in original["selected"]], [b["id"] for b in actual["selected"]])


if __name__ == "__main__":
    unittest.main()
