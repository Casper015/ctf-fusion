"""tools/repo/release_bots.py: bundles are packaged deterministically, hashed, and listed so the host can verify them."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "repo"))
import release_bots  # noqa: E402
from fusion.runner import find_compiler  # noqa: E402

SMALL = ["v12_control", "c08_control"]          # 0.05 and 0.16 MB: no model weights inlined


class TestReleaseBots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants = [v for v in release_bots.load_variants(release_bots.VARIANTS) if v["name"] in SMALL]
        assert len(cls.variants) == 2, "variants.json must keep v12_control and c08_control"

    def test_package_hashes_and_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = release_bots.package(self.variants, out, check=find_compiler() is not None)
            self.assertEqual(sorted(manifest["variants"]), sorted(SMALL))
            for name, e in manifest["variants"].items():
                text = gzip.decompress((out / e["file"]).read_bytes())
                self.assertEqual(hashlib.sha256(text).hexdigest(), e["sha256"])
                self.assertEqual(len(text), e["bytes"])
                self.assertNotIn(b"PONLY_STATS 1\n", text.split(b"\n")[0])        # deployed form: no instrumentation define
                if find_compiler():
                    self.assertEqual(e["syntax_check"], "ok")
            listed = dict(line.split("  ")[::-1] for line in (out / "SHA256SUMS").read_text().strip().split("\n"))
            self.assertEqual(set(listed), {e["file"] for e in manifest["variants"].values()} | {"MANIFEST.json"})
            for fname, digest in listed.items():
                self.assertEqual(hashlib.sha256((out / fname).read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads((out / "MANIFEST.json").read_text())["variants"].keys(), manifest["variants"].keys())

    def test_gzip_is_reproducible(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ma = release_bots.package(self.variants, Path(a))
            mb = release_bots.package(self.variants, Path(b))
            for name in SMALL:
                fa, fb = ma["variants"][name]["file"], mb["variants"][name]["file"]
                self.assertEqual((Path(a) / fa).read_bytes(), (Path(b) / fb).read_bytes())

    def test_bundle_matches_what_build_py_hashes(self):
        """A screen hashes '#define PONLY_STATS 1' + the bundle: the release hash is the same bundle, one line shorter."""
        sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
        from bundle import bundle, sha256_text
        v = self.variants[0]
        text, _ = bundle(REPO / v["source"])
        with tempfile.TemporaryDirectory() as tmp:
            m = release_bots.package([v], Path(tmp))
        self.assertEqual(m["variants"][v["name"]]["sha256"], sha256_text(text))
        self.assertNotEqual(sha256_text("#define PONLY_STATS 1\n" + text), sha256_text(text))


if __name__ == "__main__":
    unittest.main()
