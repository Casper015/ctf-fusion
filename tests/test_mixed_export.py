"""Mixed model sets (training/export_mixed.py): P from one tier, V from another, copied exactly."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion.paths import REPO, sha256
from training import export_mixed

SETS = {"models_pxl_vm": ("xl", "m"), "models_pxxl_vm": ("xxl", "m")}


class MixedExport(unittest.TestCase):
    def test_arrays_are_exact_copies_of_their_sources(self):
        for name, (p_tier, v_tier) in SETS.items():
            out = REPO / "inference" / name
            manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
            src_p = REPO / "inference" / export_mixed.TIER_DIR[p_tier]
            src_v = REPO / "inference" / export_mixed.TIER_DIR[v_tier]
            for model, m in manifest.items():
                with self.subTest(set=name, model=model):
                    self.assertEqual(sha256(out / m["header"]), m["header_sha256"])
                    self.assertEqual(sha256(out / f"weights_{model}.npz"), m["npz_sha256"])
                    self.assertEqual(m["sources"]["policy"]["npz_sha256"], sha256(src_p / f"weights_{model}.npz"))
                    self.assertEqual(m["sources"]["value"]["npz_sha256"], sha256(src_v / f"weights_{model}.npz"))
                    self.assertFalse(m["value_approved"])
                    with np.load(out / f"weights_{model}.npz") as mixed, \
                         np.load(src_p / f"weights_{model}.npz") as zp, \
                         np.load(src_v / f"weights_{model}.npz") as zv:
                        for k in mixed.files:
                            ref = zv[k] if k.startswith("value.") else zp[k]
                            self.assertTrue(np.array_equal(mixed[k], ref) and mixed[k].dtype == ref.dtype, k)
                        self.assertEqual(sorted(mixed.files), sorted([k for k in zp.files if not k.startswith("value.")] +
                                                                    [k for k in zv.files if k.startswith("value.")]))
            registry = (out / "registry.hpp").read_text(encoding="utf-8")
            for model, m in manifest.items():
                self.assertIn(f'#include "{m["header"]}"', registry)

    def test_regeneration_reproduces_the_committed_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, (p_tier, v_tier) in SETS.items():
                out = export_mixed.export(p_tier, v_tier, Path(tmp) / name)
                for f in sorted((REPO / "inference" / name).glob("weights_*.hpp")):
                    ours = (out / f.name).read_text(encoding="utf-8").split("\n", 1)[1]      # skip the dated first line
                    self.assertEqual(ours, f.read_text(encoding="utf-8").split("\n", 1)[1], f"{name}/{f.name}")
                for f in sorted((REPO / "inference" / name).glob("weights_*.npz")):
                    with np.load(f) as a, np.load(out / f.name) as b:
                        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a.files), f"{name}/{f.name}")


if __name__ == "__main__":
    unittest.main()
