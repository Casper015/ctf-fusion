"""Tests of the P-only package (no games; the compile and contract tests need a C++ compiler).

    python -m unittest discover -s p_only/tests -v
"""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "p_only" / "tools"))

import analyze  # noqa: E402
import bundle  # noqa: E402
import size_rule  # noqa: E402
from fusion.runner import find_compiler  # noqa: E402

BOTS = REPO / "p_only" / "bots"
HEADERS = [BOTS / "c08_ponly.hpp", BOTS / "v12_ponly.hpp", REPO / "p_only" / "include" / "ponly.hpp"]
CXX = find_compiler()

# Identifiers of the hand-written opponent model and of the legacy mixture: none
# may remain in the P-only code (comments are stripped first).
FORBIDDEN = ["mixedMarginal", "mixedJointProb", "LegacyForecast", "drawTeam", "alpha", "MOD_FUSION_P",
             "attentionWeight", "priority(", "crossStay", "crossMove", "otherFocus", "lasttarget", "W_other_attention",
             "scriptedStep", "rivalsFirstStep", "rivalsNextStep", "plyAttention", "plyDiscount", "plies",
             "rivalAttacks", "rivalWants", "rivalDrops", "groundPickupProbability", "groundFlagAt", "closestGoal",
             "enemyDropProbability", "observeDrops", "forecastDrops", "dropCounts", "shootChance", "dropChance",
             "e.attention", "tw("]


def code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


class HandWrittenPredictionRemoved(unittest.TestCase):
    def test_no_legacy_identifiers(self):
        for h in HEADERS:
            text = code_only(h)
            for name in FORBIDDEN:
                with self.subTest(header=h.name, identifier=name):
                    self.assertNotIn(name, text)

    def test_every_opponent_prediction_reads_the_network_view(self):
        c08 = code_only(BOTS / "c08_ponly.hpp")
        for needle in ["void forecast(const Game& g){", "void rivalOrders(const Game& g,int model", "void predict(const Game& g) {"]:
            body = c08.split(needle, 1)[1][:1500]
            with self.subTest(function=needle):
                self.assertIn("fSession.cur", body)
        v12 = code_only(BOTS / "v12_ponly.hpp")
        self.assertIn("outcomes(fSession.view", v12)
        self.assertIn("rivalOrders(g,fSession.view", v12)
        self.assertIn("rivalOrders(g,branchView", v12)

    def test_derived_headers_match_their_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            for script, name in [("make_c08.py", "c08_ponly.hpp"), ("make_v12.py", "v12_ponly.hpp"),
                                 ("make_c08_control.py", "c08_control.hpp")]:
                subprocess.run([sys.executable, str(REPO / "p_only" / "tools" / "derive" / script)], check=True,
                               env={**os.environ, "PONLY_DERIVE_DIR": tmp}, capture_output=True)
                with self.subTest(header=name):
                    self.assertEqual((Path(tmp) / name).read_text(encoding="utf-8"), (BOTS / name).read_text(encoding="utf-8"))


class Bundle(unittest.TestCase):
    def test_conditions(self):
        m = {"FUSION_TIER_M": "1", "MOD_FUSION_P": "1", "MOD_FUSION_V": "0", "GUARD": ""}
        self.assertTrue(bundle.evaluate("defined(FUSION_TIER_M)", m))
        self.assertFalse(bundle.evaluate("defined(FUSION_TIER_L)", m))
        self.assertTrue(bundle.evaluate("MOD_FUSION_P || MOD_FUSION_V", m))
        self.assertFalse(bundle.evaluate("MOD_FUSION_V && MOD_FUSION_P", m))
        self.assertTrue(bundle.evaluate("!defined(FUSION_TIER_L) && MOD_FUSION_P == 1", m))
        with self.assertRaises(bundle.Unknown):
            bundle.evaluate("defined(__AVX2__)", m)

    def test_variants_bundle_into_one_file(self):
        variants = json.loads((REPO / "p_only" / "profiles" / "variants.json").read_text(encoding="utf-8"))["variants"]
        for v in variants:
            with self.subTest(variant=v["name"]):
                text, files = bundle.bundle(REPO / v["source"])
                active = [line for line in text.splitlines() if bundle.INCLUDE.match(line)]
                self.assertEqual(active, [], "a local include was left in the bundle")
                if v["tier"]:
                    self.assertTrue(any(f"models_{v['tier'].lower()}" in str(f) or (v["tier"] == "S" and "/models/" in str(f))
                                        for f in files), "the bundle lacks its tier's weights")


class SizeRule(unittest.TestCase):
    def arm(self, p99, faults=0, late=0):
        return {"games": 15, "faults": faults, "late_replies": late, "p99_ms_worst": p99, "max_ms": p99 + 3}

    def test_largest_c08_size_within_40_ms(self):
        timing = {"arms": {"c08_ponly_m": self.arm(25), "c08_ponly_l": self.arm(33), "c08_ponly_xl": self.arm(39.9),
                           "c08_ponly_xxl": self.arm(44), "v12_ponly_xl": self.arm(8), "v12_ponly_xxl": self.arm(12, late=1)}}
        d = size_rule.decide(timing)
        self.assertEqual(d["c08"], "c08_ponly_xl")
        self.assertEqual(d["v12"], ["v12_ponly_xl"])
        arms, comp = size_rule.screen_arms(d["c08"], d["v12"])
        self.assertIn("c08_control_2ply", arms)
        self.assertIn(["c08_ponly_xl", "c08_control_2ply"], comp)

    def test_no_c08_size_qualifies(self):
        d = size_rule.decide({"arms": {"c08_ponly_m": self.arm(41), "c08_ponly_l": self.arm(30, faults=1)}})
        self.assertIsNone(d["c08"])


class Analyze(unittest.TestCase):
    def test_paired_difference(self):
        fixtures = {f"f{i}": {"teams": 2 if i < 4 else 5} for i in range(8)}
        a = {f: {"utility": 1.0, "score": [1]} for f in fixtures}
        b = {f: {"utility": 0.5 if i % 2 else 1.0, "score": [1] if i % 2 == 0 else [0]} for i, f in enumerate(fixtures)}
        r = analyze.paired(a, b, fixtures, [2, 5])
        self.assertAlmostEqual(r["overall"]["mean"], 0.25)
        self.assertEqual((r["improved"], r["unchanged"], r["degraded"]), (4, 4, 0))
        self.assertEqual(r["identical_scores"], 4)
        self.assertEqual(r["bands"]["2"]["fixtures"], 4)
        self.assertIsNone(r["bands"]["6-10"])


@unittest.skipUnless(CXX, "no C++ compiler")
class Contract(unittest.TestCase):
    """History, distributions, orders, coherent states and the engine comparison on recorded games (contract_test.cpp)."""

    def run_contract(self, carrier: str):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / f"contract_{carrier}"
            subprocess.run([CXX, "-std=c++20", "-O2", f"-I{REPO / 'inference'}", f"-I{REPO / 'p_only' / 'include'}",
                            f"-I{BOTS}", "-DFUSION_TIER_M", f"-DCARRIER_{carrier}", str(REPO / "p_only" / "tests" / "contract_test.cpp"),
                            "-o", str(exe)], check=True, capture_output=True)
            for stream in sorted((REPO / "p_only" / "tests" / "fixtures").glob("stream_n*.txt.gz")):
                orders = Path(tmp) / stream.name.replace("stream_", "orders_").removesuffix(".gz")
                orders.write_bytes(gzip.decompress((stream.parent / stream.name.replace("stream_", "orders_")).read_bytes()))
                r = subprocess.run([str(exe), str(orders)], input=gzip.decompress(stream.read_bytes()), capture_output=True, timeout=600)
                out = r.stdout.decode()
                with self.subTest(carrier=carrier, stream=stream.name):
                    self.assertEqual(r.returncode, 0, out + r.stderr.decode()[-2000:])
                    self.assertIn(" 0 failures", out)
                    self.assertIn(" 0 inconsistent", out)
                    steps = int(re.search(r"engine steps (\d+)", out).group(1))
                    self.assertGreater(steps, 300, "too few turns compared with the engine")
                    if "n04" in stream.name:
                        branches = int(re.search(r"(\d+) branch views", out).group(1))
                        self.assertGreater(branches, 0, "the stream never exercised a simulated branch")

    def test_c08(self):
        self.run_contract("C08")

    def test_v12(self):
        self.run_contract("V12")


if __name__ == "__main__":
    unittest.main()
