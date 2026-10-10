"""Feature-family carriers (bots/p_only/tools/derive/make_families.py, docs/FEATURE_FAMILIES.md).

  * the family headers are exactly what their generator writes;
  * with a p3/v3 model set (random weights, exported by the real exporter) both carriers keep every P-only
    contract on recorded games: the root's history, branch parents (a branch never reaches the real root),
    distributions, legal orders, coherent states and the engine comparison;
  * the bounded V route: with PONLY_V_MAX_CALLS=0 every V decision takes its choice without V, so the replies
    equal the V-off build's on a recorded game, and the counters say so;
  * a carrier compiled for one family refuses a model set of another (static_assert).
Not timing or strength evidence.
"""
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fusion import families  # noqa: E402
from fusion.capacity import buffer_width  # noqa: E402
from fusion.models import PolicyNet, ValueNet  # noqa: E402
from fusion.native import schema  # noqa: E402
from fusion.schema import layout  # noqa: E402
from training import export_weights  # noqa: E402

BOTS = REPO / "bots" / "p_only" / "bots"
TESTS = REPO / "bots" / "p_only" / "tests"
CXX = shutil.which(os.environ.get("CXX", "")) or shutil.which("g++") or shutil.which("clang++")


def model_set(out: Path, pfam: str, vfam: str, vhidden=(64, 16), seed=0) -> Path:
    """A complete model set (headers, MANIFEST.json, registry.hpp) of random weights in the given families."""
    L = layout()
    rng = np.random.default_rng(seed)
    policy = PolicyNet(L, rng, {"mover": [16], "squad": [32, 16], "act": [16], "target": [8]}, pfam)
    value = ValueNet(L, rng, vhidden, vfam)
    nets = export_weights.quantize_model(policy, value, 1.0, L, export_weights.model_width(policy, value))
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights_general.hpp").write_text(export_weights.header("general", nets, {}, {"test": "random"},
                                                                    policy_family=pfam, value_family=vfam), encoding="utf-8")
    entry = {"header": "weights_general.hpp", "required_buffer_width": buffer_width([export_weights.widest(nets)])}
    if (pfam, vfam) != ("p2", "v2"):
        entry["features"] = {"policy": families.record(schema(), pfam), "value": families.record(schema(), vfam)}
    export_weights.registry(out, {"general": entry})
    return out / "registry.hpp"


def compile_harness(src: str, exe: Path, defines: list[str], registry: Path | None) -> subprocess.CompletedProcess:
    pre = ["-include", str(registry), "-DPONLY_FAMILIES"] if registry else []
    return subprocess.run([CXX, "-std=c++20", "-O2", f"-I{REPO / 'inference'}", f"-I{REPO / 'bots' / 'p_only' / 'include'}",
                           f"-I{BOTS}", *pre, *defines, str(TESTS / src), "-o", str(exe)], capture_output=True, text=True)


class Derived(unittest.TestCase):
    def test_family_headers_match_their_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(REPO / "bots" / "p_only" / "tools" / "derive" / "make_families.py")], check=True,
                           env={**os.environ, "PONLY_DERIVE_DIR": tmp}, capture_output=True)
            for rel in ("include/ponly_p3.hpp", "bots/v12_ponly_f.hpp", "bots/c08_ponly_f.hpp"):
                with self.subTest(header=rel):
                    self.assertEqual((Path(tmp) / Path(rel).name).read_text(encoding="utf-8"),
                                     (REPO / "bots" / "p_only" / rel).read_text(encoding="utf-8"))


@unittest.skipUnless(CXX, "no C++ compiler")
class FamilyCarriers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ponly-families-"))
        cls.p3v3 = model_set(cls.tmp / "p3v3", "p3", "v3")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_contracts_hold_with_p3_v3(self):
        defines = ["-DPONLY_POLICY_FAMILY=3", "-DPONLY_VALUE_FAMILY=3", "-DMOD_FUSION_V=1", "-DPONLY_FIXED_SAMPLES=32"]
        for carrier in ("C08", "V12"):
            exe = self.tmp / f"contract_{carrier}"
            r = compile_harness("contract_test.cpp", exe, defines + [f"-DCARRIER_{carrier}"], self.p3v3)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            for name in ("n04", "n15"):
                orders = self.tmp / f"orders_{name}.txt"
                orders.write_bytes(gzip.decompress((TESTS / "fixtures" / f"orders_{name}.txt.gz").read_bytes()))
                run = subprocess.run([str(exe), str(orders)], input=gzip.decompress((TESTS / "fixtures" / f"stream_{name}.txt.gz").read_bytes()),
                                     capture_output=True, timeout=900)
                out = run.stdout.decode()
                with self.subTest(carrier=carrier, game=name):
                    self.assertEqual(run.returncode, 0, out + run.stderr.decode()[-2000:])
                    self.assertIn(" 0 failures", out)
                    self.assertIn(" 0 inconsistent", out)

    def test_v_bound_falls_back_to_the_choice_without_v(self):
        stream = gzip.decompress((TESTS / "fixtures" / "stream_n04.txt.gz").read_bytes())
        base = ["-DCARRIER_V12", "-DPONLY_POLICY_FAMILY=3", "-DPONLY_VALUE_FAMILY=3", "-DPONLY_STATS=1"]
        runs = {}
        for name, extra in (("off", []), ("cut", ["-DMOD_FUSION_V=1", "-DPONLY_V_MAX_CALLS=0"]), ("on", ["-DMOD_FUSION_V=1"])):
            exe = self.tmp / f"replay_{name}"
            r = compile_harness("replay_test.cpp", exe, base + extra, self.p3v3)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            run = subprocess.run([str(exe)], input=stream, capture_output=True, timeout=900, check=True)
            runs[name] = (run.stdout.decode().splitlines()[:-1], run.stderr.decode())
        self.assertEqual(len(runs["off"][0]), 400)
        self.assertEqual(runs["off"][0], runs["cut"][0], "a V route cut to zero calls must reply as the V-off build")
        cut = runs["cut"][1]
        self.assertIn("rollout_calls 0 ", cut)
        skipped = int(re.search(r"skipped_calls (\d+)", cut).group(1))
        fallbacks = int(re.search(r"fallback_decisions (\d+)", cut).group(1))
        self.assertGreater(skipped, 0)
        decisions = int(re.search(r"rollout_decisions (\d+)", cut).group(1))
        # Every decision that needed the network fell back; near the end a rollout can reach the last turn,
        # where the exact pairwise utility replaces V and nothing is cut.
        self.assertLessEqual(fallbacks, decisions)
        self.assertGreater(fallbacks, 0.9 * decisions)
        calls = int(re.search(r"rollout_calls (\d+)", runs["on"][1]).group(1))
        self.assertGreater(calls, 0)
        self.assertIn("fallback_decisions 0", runs["on"][1])

    def test_crossed_families_do_not_compile(self):
        p2v2 = model_set(self.tmp / "p2v2", "p2", "v2")
        for registry, defines in ((p2v2, ["-DPONLY_POLICY_FAMILY=3"]), (self.p3v3, ["-DPONLY_POLICY_FAMILY=2", "-DPONLY_VALUE_FAMILY=3"]),
                                  (self.p3v3, ["-DPONLY_POLICY_FAMILY=3", "-DPONLY_VALUE_FAMILY=2"])):
            r = compile_harness("replay_test.cpp", self.tmp / "crossed", ["-DCARRIER_V12", *defines], registry)
            with self.subTest(registry=registry.parent.name, defines=defines):
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("model set is not", r.stderr)


if __name__ == "__main__":
    unittest.main()


class FamilyVariants(unittest.TestCase):
    """bots/p_only/profiles/variants_families.json: every wrapper exists, selects the families it declares, includes
    its model set's registry first and the family carrier after it; sets that exist bundle into one file."""

    def test_wrappers_match_their_declaration(self):
        sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
        import build  # noqa: PLC0415
        import bundle  # noqa: PLC0415
        variants = build.load_variants(REPO / "bots" / "p_only" / "profiles" / "variants_families.json")
        names = [v["name"] for v in variants]
        self.assertEqual(len(names), len(set(names)))
        own = [v for v in variants if v.get("role") == "family"]
        self.assertGreaterEqual(len(own), 9)
        for v in own:
            text = (REPO / v["source"]).read_text(encoding="utf-8")
            with self.subTest(variant=v["name"]):
                self.assertIn(f"#define PONLY_POLICY_FAMILY {v['policy_family'][1:]}", text)
                self.assertIn(f"#define PONLY_VALUE_FAMILY {v['value_family'][1:]}", text)
                self.assertIn(f"#define MOD_FUSION_V {int(v['value_on'])}", text)
                reg = text.index(f'#include "{Path(v["models"]).name}/registry.hpp"')
                self.assertLess(reg, text.index(f'#include "{v["carrier"].lower()}_ponly_f.hpp"'))
                self.assertIn(f"--out {v['models']}", v["export"])
                if (REPO / v["models"] / "registry.hpp").exists():
                    out, _ = bundle.bundle(REPO / v["source"])
                    self.assertNotIn('#include "', out)
