"""Deployment contracts that need the C++ tools but not AI_Test (docs/REVIEW.md section 7).

  * raw extractor records are byte-exact and aligned (Windows binary stdout);
  * feature spec v2 separates score distributions that v1 confused;
  * inference/adapter.hpp gives exactly the probabilities fusion/mixture.py
    scores, and its sampler draws from them;
  * layer widths beyond the C++ buffers are refused before training and export;
  * an export accepts only a gate report made for the same artifact.
"""
import json
import math
import unittest

import numpy as np

from fusion import metrics, native, quant
from fusion.mixture import ADAPTERS, adapter_table
from fusion.nn import MLP
from fusion.schema import layout
from training import export_weights, train_policy, train_value

SIZE, TURNS = 23, 30


def synthetic_stream(scores: dict[int, list[int]]) -> str:
    """A legal 3-team seat stream on an open 23 x 23 map: units stand still, scores follow `scores[turn]`."""
    bases = [(3, 3), (19, 3), (11, 19)]
    lines = [f"INIT 3 0 {SIZE} {TURNS} 100 34 10 15 8", "MAP", *["." * SIZE] * SIZE, "BASES 3",
             *[f"{x} {y}" for x, y in bases], "SPOTS 2", "11 11", "5 15"]
    score = [0, 0, 0]
    for t in range(1, TURNS + 1):
        score = scores.get(t, score)
        lines += [f"TURN {t}", "SCORE " + " ".join(map(str, score)), "UNITS 9"]
        for uid in range(9):
            bx, by = bases[uid // 3]
            lines.append(f"U {uid} {uid // 3} {bx + uid % 3 - 1} {by + 2} 100 -1 -1")
        lines += ["FLAGS 2", "F 0 home 11 11 -1 -1", "F 1 home 5 15 -1 -1", "EVENTS 0", "END"]
    return "\n".join(lines) + "\n"


class TestRecords(unittest.TestCase):
    def test_raw_records_are_byte_exact(self):
        """Turns 10, 13 and 26 put 0x0A, 0x0D and 0x1A bytes in the stream: text-mode stdout would corrupt them."""
        stream = synthetic_stream({})
        r = native.run_tool(native.extractor(), stream.encode())
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        width = native.schema()["record_width"]
        self.assertEqual(len(r.stdout), TURNS * 3 * width * 2)
        for byte in (b"\x0a\x00", b"\x0d\x00", b"\x1a\x00"):
            self.assertIn(byte, r.stdout)
        rec = native.extract(stream)  # alignment-checked
        np.testing.assert_array_equal(rec[:, 0], np.repeat(np.arange(1, TURNS + 1), 3))
        with self.assertRaises(RuntimeError):  # a shifted stream must be caught, not decoded
            native.check_records(np.frombuffer(r.stdout[2:] + b"\0\0", "<i2").reshape(-1, width), 3)

    def test_value_features_separate_score_distributions(self):
        """For team 0, [5,6,4] and [5,6,5] share own score, best rival, rank and margin; v2 tells them apart."""
        rec = native.extract(synthetic_stream({10: [5, 6, 4], 20: [5, 6, 5]}))
        L = layout()
        v = L.get(rec, "v")
        names = L.names("global") + [f"u{k}_{n}" for k in range(3) for n in L.names("value_unit")] + L.names("value_agg")
        col = {n: i for i, n in enumerate(names)}
        a, b = v[(10 - 1) * 3], v[(20 - 1) * 3]  # team 0 at turns 10 and 20
        for n in ("own_score", "best_enemy_score", "own_rank", "margin"):
            self.assertEqual(a[col[n]], b[col[n]], n)
        self.assertEqual((a[col["rivals_tied"]], b[col["rivals_tied"]]), (0, 1))
        self.assertEqual((a[col["pairwise_now"]], b[col["pairwise_now"]]), (50, 25))
        self.assertEqual((a[col["rivals_1_2_ahead"]], a[col["rivals_1_2_behind"]]), (1, 1))


class TestAdapterNative(unittest.TestCase):
    def run_case(self, adapter: str, alpha: float, qh: np.ndarray, nn: np.ndarray, draws=400_000, seed=11):
        num = lambda xs: " ".join(f"{float(x):.17g}" for x in xs)
        text = f"{ADAPTERS[adapter]} {alpha:.17g} {draws} {seed}\n{num(qh.reshape(-1))}\n{num(nn)}\n"
        r = native.run_tool(native.adapter_check(), text.encode())
        self.assertEqual(r.returncode, 0, r.stderr.decode())
        exact, counts, marg = r.stdout.decode().split("\n")[:3]
        return np.array(exact.split(), float), np.array(counts.split(), int), np.array(marg.split(), float).reshape(3, 5)

    def test_cpp_adapter_matches_the_gate(self):
        rng = np.random.default_rng(7)
        for adapter in ADAPTERS:
            for alpha in (0.0, 0.3, 1.0):
                legal = rng.random((3, 5)) < 0.7
                legal[:, 0] = True
                qh = np.where(legal, rng.random((3, 5)), 0)
                qh /= qh.sum(1, keepdims=True)
                mask = (legal[0][:, None, None] & legal[1][None, :, None] & legal[2][None, None, :]).reshape(-1)
                nn = np.where(mask, rng.gamma(0.3, size=125), 0)
                nn /= nn.sum()
                exact, counts, marg = self.run_case(adapter, alpha, qh, nn)
                want = adapter_table(adapter, alpha, qh, nn)
                np.testing.assert_allclose(exact, want, rtol=1e-12, atol=1e-15)
                self.assertAlmostEqual(exact.sum(), 1.0, places=12)
                nn_marg = [nn.reshape(5, 5, 5).sum(a) for a in ((1, 2), (0, 2), (0, 1))]
                np.testing.assert_allclose(marg, [(1 - alpha) * qh[k] + alpha * nn_marg[k] for k in range(3)], rtol=1e-12)
                n = counts.sum()
                sd = np.sqrt(n * want * (1 - want)) + 1e-9
                self.assertTrue((np.abs(counts - n * want) < 5 * sd + 1).all(), f"{adapter} alpha {alpha}: sampler off")
                self.assertTrue((counts[want == 0] == 0).all())

    def test_marginal_adapter_drops_coordination(self):
        """The review's example: the network only knows that units 0 and 1 move together."""
        qh = np.zeros((3, 5))
        qh[:2, :2] = 0.5
        qh[2, 0] = 1.0
        nn = np.zeros((5, 5, 5))
        nn[0, 0, 0] = nn[1, 1, 0] = 0.5
        nn = nn.reshape(-1)
        labels = [0, 30]  # (0,0,0) and (1,1,0)
        for alpha in (0.0, 0.5, 1.0):
            v1 = adapter_table("marginal-v1", alpha, qh, nn)
            np.testing.assert_allclose(v1, adapter_table("marginal-v1", 0.0, qh, nn))  # identical for every alpha
            v2 = adapter_table("joint-v2", alpha, qh, nn)
            self.assertAlmostEqual(-np.log(v2[labels]).mean(), -np.log((1 - alpha) * 0.25 + alpha * 0.5))
        self.assertAlmostEqual(np.log(adapter_table("joint-v2", 1.0, qh, nn)[0] / adapter_table("marginal-v1", 1.0, qh, nn)[0]),
                               math.log(2))


class TestWidths(unittest.TestCase):
    def test_limit_matches_cpp(self):
        text = (native.REPO / "inference" / "mlp.hpp").read_text(encoding="utf-8")
        self.assertIn(f"#define FUSION_MAX_WIDTH {quant.MAX_WIDTH}", text)
        self.assertIn("inline constexpr int kMaxWidth = FUSION_MAX_WIDTH;", text)

    def test_wide_layers_are_refused(self):
        net = MLP([4, quant.MAX_WIDTH + 1, 1], np.random.default_rng(0), dtype=np.float64)
        q = quant.quantize(net, np.ones(4), np.zeros(4, np.int64), np.ones(4, np.int64))
        with self.assertRaises(ValueError):
            quant.check_widths(q, "wide")
        quant.check_widths(q, "wide", limit=2 * quant.MAX_WIDTH)  # allowed for a -DFUSION_MAX_WIDTH build
        for trainer, flag in ((train_value, "--hidden"), (train_policy, "--hidden-squad")):
            with self.assertRaises(SystemExit):  # before any data is read
                trainer.main(["--model", "general", "--out", "unused", flag, f"{quant.MAX_WIDTH * 2},16"])

    def test_header_asserts_widths_and_spec(self):
        from fusion.models import PolicyNet, ValueNet  # noqa: PLC0415
        L = layout()
        rng = np.random.default_rng(0)
        nets = export_weights.quantize_model(PolicyNet(L, rng), ValueNet(L, rng), 1.0, L)
        text = export_weights.header("probe", nets, {}, {})
        self.assertIn(f"static_assert(fusion::kSpecVersion == {L.schema['spec_version']}", text)
        self.assertEqual(text.count('exceeds kMaxWidth");'), sum(len(q["layers"]) for q in nets.values()))
        self.assertIn("inline constexpr int kAdapter = 2;", text)
        self.assertIn("inline constexpr bool kValueApproved = false;", text)


class TestKernels(unittest.TestCase):
    """The SIMD multiply-add kernels give the original scalar loop's bits on every path this machine runs."""

    def test_native_paths(self):
        ok, summary = native.run_kernel_check(200)
        self.assertTrue(ok, summary)
        self.assertIn("mismatches 0", summary)

    def test_forced_scalar_build(self):
        r = native.run_tool(native.kernel_check(simd=False), args=(50,))
        self.assertEqual(r.returncode, 0, r.stdout.decode())
        self.assertIn("active scalar", r.stdout.decode())

    def test_x86_paths_under_rosetta(self):
        """On Apple silicon, the x86-64 build (SSE2; AVX2 where the CPU has it) runs under Rosetta when installed."""
        import platform  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        if platform.system() != "Darwin" or platform.machine() != "arm64" or not shutil.which("clang++"):
            self.skipTest("only on Apple silicon with clang")
        if subprocess.run(["arch", "-x86_64", "/usr/bin/true"], capture_output=True).returncode != 0:
            self.skipTest("Rosetta is not installed")
        out = native.BUILD / "kernel_check_x86_64"
        native.BUILD.mkdir(parents=True, exist_ok=True)
        src = native.REPO / "tools" / "kernel_check.cpp"
        subprocess.run(["clang++", "-arch", "x86_64", "-std=c++20", "-O2", "-DFUSION_MAX_WIDTH=1024", str(src), "-o", str(out)],
                       check=True, capture_output=True)
        r = subprocess.run(["arch", "-x86_64", str(out), "100"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("sse2", r.stdout)


class TestCapacityBench(unittest.TestCase):
    """tools/capacity_bench.py: correctness and arithmetic only. Its timing runs belong on the target host."""

    @classmethod
    def setUpClass(cls):
        import sys  # noqa: PLC0415
        sys.path.insert(0, str(native.REPO / "tools"))
        import capacity_bench  # noqa: PLC0415
        cls.cb = capacity_bench

    def test_wide_tiers_compute_exactly(self):
        tiers = {"S": self.cb.TIERS["S"], "W": self.cb.parse_tier("W squad=300,64 value=520,16")[1]}
        self.assertEqual(self.cb.verify(tiers, [3]), 0)  # incremental == full value, buffers of 576

    def test_narrow_build_refuses_wide_tiers(self):
        binary = native.build(native.REPO / "tools" / "size_bench.cpp", "size_bench_w256", defines=["FUSION_MAX_WIDTH=256"])
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmp:
            tiers = Path(tmp) / "tiers.txt"
            tiers.write_text(self.cb.tier_line("L", self.cb.TIERS["L"]) + "\n", encoding="utf-8")
            from fusion import synthetic  # noqa: PLC0415
            r = native.run_tool(binary, synthetic.stream(3, turns=4).encode(), args=("--verify", 1, 4, tiers))
        self.assertEqual(r.returncode, 2)
        self.assertIn("FUSION_MAX_WIDTH", r.stderr.decode())

    def test_plan_arithmetic(self):
        seat = lambda p99, v99, node: {"turns": 400, "policy_turn_ms": {"p99": p99, "max": p99 * 2},
                                       "policy_joint_ms": {"p99": p99 / 2}, "value_turn_ms": {"p99": v99},
                                       "value_node_us": {"encode": node / 2, "incremental": node / 2, "full": node}}
        p = self.cb.plan([seat(1.0, 0.5, 20.0), seat(2.0, 0.4, 20.0)], budget_ms=5.0, safety=1.5)
        self.assertAlmostEqual(p["need_ms"], 1.5 * (2.0 + 0.5))  # the worst seat's p99s
        self.assertTrue(p["fits"])
        self.assertEqual(p["value_nodes_affordable"], int((5.0 / 1.5 - 2.0) * 1000 / 20.0))
        self.assertFalse(self.cb.plan([seat(3.0, 0.5, 20.0)], budget_ms=5.0, safety=1.5)["fits"])

    def test_report_compares_kernels_without_timing(self):
        """The timing path end to end, with size_bench replaced by synthetic lines (nothing is timed here)."""
        import tempfile  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        def line(tier, kernel, p99):
            return {"tier": tier, "kernel": kernel, "teams": 3, "map_size": 31, "turns": 100, "value_calls": 24,
                    "policy_turn_ms": {"p50": p99 / 2, "p95": p99, "p99": p99, "max": p99 * 1.2},
                    "policy_joint_ms": {"p50": p99 / 4, "p99": p99 / 2},
                    "value_node_us": {"encode": 1.0, "incremental": 2.0 if kernel == "scalar" else 1.0, "full": 4.0},
                    "value_turn_ms": {"p50": 0.05, "p99": 0.1, "max": 0.2}}

        def fake_run(binary, stream_file, tiers_file, seats, repeat, value_calls):
            scalar = "scalar" in Path(binary).name
            return [[line("S", "scalar" if scalar else "neon", 0.8 if scalar else 0.2),
                     line("M", "scalar" if scalar else "neon", 1.6 if scalar else 0.4)] for _ in range(seats)]

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(self.cb, "run_seats", fake_run), \
                mock.patch.object(self.cb, "recorded_stream", lambda *a: ("INIT\n", "synthetic")):
            self.cb.main(["--teams", "3", "--tiers", "S", "M", "--out", tmp])
            report = json.loads(next(Path(tmp).glob("capacity_*.json")).read_text(encoding="utf-8"))
            md = next(Path(tmp).glob("capacity_*.md")).read_text(encoding="utf-8")
        c = report["counts"][0]
        self.assertEqual(report["kernels"], ["simd", "scalar"])
        self.assertEqual(report["simd_path"], "neon")
        self.assertEqual(c["tiers"], c["kernels"]["simd"])               # verdicts use what the bot runs
        self.assertAlmostEqual(c["kernels"]["scalar"]["S"]["policy_p99_ms"] / c["kernels"]["simd"]["S"]["policy_p99_ms"], 4.0)
        self.assertIn("## Kernels: scalar loop vs SIMD", md)
        self.assertIn("| 3 | S | 0.800 / 0.200 | 4.00× |", md)
        self.assertIn("mismatches 0", report["kernel_check"])
        self.assertTrue(report["host_profile"]["cpu"])

    def test_tier_sizes(self):
        s = self.cb.tier_static(self.cb.TIERS["S"], layout())
        self.assertEqual(s["macs_per_opponent_team"], 30096)  # today's default architecture
        self.assertEqual(s["max_width"], quant.MAX_WIDTH)
        self.assertEqual(self.cb.tier_static(self.cb.TIERS["L"], layout())["max_width"], 512)


class TestMultithreadingReport(unittest.TestCase):
    """tools/generate_multithreading_report.py computes every statement from the committed capacity reports."""

    def test_report_is_computed(self):
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        sys.path.insert(0, str(native.REPO / "tools"))
        import generate_multithreading_report as g  # noqa: PLC0415
        self.assertEqual(g.ranges([2, 3, 4, 7, 9, 10]), "2–4, 7, 9–10")
        self.assertEqual(g.ranges([]), "none")
        multi = native.REPO / "reports" / "capacity_DESKTOP-G60OJUD.json"
        single = native.REPO / "reports" / "capacity_1seat_DESKTOP-G60OJUD.json"
        if not (multi.exists() and single.exists()):
            self.skipTest("committed capacity reports not present")
        r = g.compute(json.loads(multi.read_text(encoding="utf-8")), json.loads(single.read_text(encoding="utf-8")))
        self.assertEqual((r["seats_multi"], r["seats_single"]), (16, 1))
        self.assertEqual(r["ceiling_multi"], "XL")
        self.assertEqual(r["per_tier"]["XXL"]["fits_single"], list(range(2, 11)))
        self.assertEqual(r["per_tier"]["XXL"]["fits_multi"], [])
        text = g.markdown(r, [15])
        self.assertIn("XXL** fits 2–10 teams with 1 seat but none with 16", text)
        self.assertNotIn("5.4 GHz", text)


class TestGateIdentity(unittest.TestCase):
    IDENTITY = {"model": "n03", "policy_sha256": "a", "value_sha256": "b", "weights_digest": "c", "spec_version": 2,
                "schema_digest": "d", "adapter": "joint-v2"}

    def test_matching_report_is_accepted(self):
        report = {"identity": self.IDENTITY | {"dataset": {}, "baseline": {}}, "alpha": {"3": {"alpha": 0.4, "use": True}}}
        self.assertEqual(export_weights.verified_gates(report, self.IDENTITY), {3: {"alpha": 0.4, "use": True}})

    def test_other_artifacts_fail_closed(self):
        for key in self.IDENTITY:
            other = self.IDENTITY | {key: "changed"}
            with self.assertRaises(SystemExit):
                export_weights.verified_gates({"identity": other, "alpha": {}}, self.IDENTITY)
        with self.assertRaises(SystemExit):
            export_weights.verified_gates({"alpha": {"3": {"alpha": 0.4, "use": True}}}, self.IDENTITY)

    def test_weight_digest_ignores_file_timestamps(self):
        flat = {"m.0.w": np.arange(6, dtype=np.int16).reshape(2, 3), "m.out_exp": np.array(9)}
        self.assertEqual(quant.digest(flat), quant.digest({k: v.copy() for k, v in reversed(flat.items())}))
        self.assertNotEqual(quant.digest(flat), quant.digest(flat | {"m.out_exp": np.array(10)}))


class TestClusteredMean(unittest.TestCase):
    def test_matches_manual_formula(self):
        rng = np.random.default_rng(3)
        v, w = rng.normal(size=200), rng.uniform(0.5, 2, 200)
        c = rng.integers(0, 25, 200)
        m, se, k = metrics.clustered_mean(v, w, c)
        self.assertAlmostEqual(m, (w * v).sum() / w.sum())
        s = np.array([(w[c == j] * (v[c == j] - m)).sum() for j in np.unique(c)])
        self.assertAlmostEqual(se, np.sqrt((s ** 2).sum() * k / (k - 1)) / w.sum())
        self.assertEqual(k, len(np.unique(c)))

    def test_one_cluster_has_no_error_bar(self):
        self.assertEqual(metrics.clustered_mean(np.ones(5), None, np.zeros(5))[1], float("inf"))


if __name__ == "__main__":
    unittest.main()
