"""Precision routing in the family carriers (task T0030, docs/P_MIXED_PRECISION.md).

  * the precision-aware views are exactly what make_families.py writes;
  * PONLY_PRECISION=1 with routing off (the default selector: Reference16 everywhere) replies exactly as the
    default build on a recorded game, for C08 (with and without its ply-2 view cache) and v12;
  * with low-bit models registered (uniform8 and mixed4_8, exported by training/export_precision.py from random
    weights) and routing on, C08 keeps every P-only contract on recorded games: root history, branch parents,
    distributions, legal orders, coherent states and the engine comparison.
Not timing or strength evidence; nothing here changes a default.
"""
import gzip
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion import precision  # noqa: E402
from fusion.models import PolicyNet  # noqa: E402
from fusion.schema import layout  # noqa: E402
from test_families import CXX, TESTS, compile_harness, model_set  # noqa: E402
from training import export_precision  # noqa: E402

HIDDEN = {"mover": [16], "squad": [32, 16], "act": [16], "target": [8]}


def lp_header(out: Path, profile: str, family: str = "p2") -> Path:
    L = layout()
    policy = PolicyNet(L, np.random.default_rng(7), HIDDEN, family)
    qp = precision.quantize_policy(policy, L, profile)
    path = out / f"lp_{profile}.hpp"
    path.write_text(export_precision.header(qp, f"t_{profile}", precision.manifest_identity(qp)), encoding="utf-8")
    return path


@unittest.skipIf(CXX is None, "no C++ compiler")
class PrecisionCarriers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.p2v2 = model_set(cls.tmp / "p2v2", "p2", "v2")
        cls.stream = gzip.decompress((TESTS / "fixtures" / "stream_n04.txt.gz").read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_precision_views_match_their_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(REPO / "bots" / "p_only" / "tools" / "derive" / "make_families.py")], check=True,
                           env={**os.environ, "PONLY_DERIVE_DIR": tmp}, capture_output=True)
            for rel in ("include/ponly_precision.hpp", "include/ponly_p3_precision.hpp"):
                with self.subTest(header=rel):
                    self.assertEqual((Path(tmp) / Path(rel).name).read_text(encoding="utf-8"),
                                     (REPO / "bots" / "p_only" / rel).read_text(encoding="utf-8"))

    def replies(self, name: str, defines: list[str], includes=()) -> list[str]:
        exe = self.tmp / f"replay_{name}"
        pre = [x for h in includes for x in ("-include", str(h))]
        r = compile_harness("replay_test.cpp", exe, pre + defines, self.p2v2)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        run = subprocess.run([str(exe)], input=self.stream, capture_output=True, timeout=900, check=True)
        return run.stdout.decode().splitlines()[:-1]

    def test_disabled_routing_replies_as_before(self):
        for carrier, extra in (("C08", []), ("C08", ["-DPONLY_PLY2_MODE=1"]), ("V12", [])):
            base = [f"-DCARRIER_{carrier}", "-DPONLY_FIXED_SAMPLES=32", *extra]
            with self.subTest(carrier=carrier, extra=extra):
                off = self.replies(f"{carrier}_off{len(extra)}", base)
                on = self.replies(f"{carrier}_on{len(extra)}", base + ["-DPONLY_PRECISION=1"])
                self.assertEqual(len(off), 400)
                self.assertEqual(off, on, "PONLY_PRECISION=1 with routing off must reply exactly as the default build")

    def test_routed_low_bit_successors_keep_the_contracts(self):
        lp8, lp48 = lp_header(self.tmp, "uniform8"), lp_header(self.tmp, "mixed4_8")
        register = "-DPONLY_PRECISION_REGISTER(b)=((b).uniform8=&fusion::lpmodels::t_uniform8::model,(b).mixed4_8=&fusion::lpmodels::t_mixed4_8::model)"
        defines = ["-DCARRIER_C08", "-DPONLY_FIXED_SAMPLES=32", "-DPONLY_PLY2_MODE=1", "-DPONLY_PRECISION=1",
                   "-DPONLY_PRECISION_ROUTE=1", register, "-include", str(lp8), "-include", str(lp48)]
        exe = self.tmp / "contract_routed"
        r = compile_harness("contract_test.cpp", exe, defines, self.p2v2)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        for game in ("n04", "n15"):
            orders = self.tmp / f"orders_{game}.txt"
            orders.write_bytes(gzip.decompress((TESTS / "fixtures" / f"orders_{game}.txt.gz").read_bytes()))
            run = subprocess.run([str(exe), str(orders)], input=gzip.decompress((TESTS / "fixtures" / f"stream_{game}.txt.gz").read_bytes()),
                                 capture_output=True, timeout=900)
            out = run.stdout.decode()
            with self.subTest(game=game):
                self.assertEqual(run.returncode, 0, out + run.stderr.decode()[-2000:])
                self.assertIn(" 0 failures", out)
                self.assertIn(" 0 inconsistent", out)


if __name__ == "__main__":
    unittest.main()
