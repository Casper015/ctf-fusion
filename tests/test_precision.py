"""Experimental mixed-precision P (task T0030, docs/P_MIXED_PRECISION.md): formats, kernels, calibration, distillation.

  * profiles: built-ins equal profiles/p_precision_code.json; per-layer bits; deterministic identities and digests
  * packing: int4 round trips with padding; flatten/unflatten round trips; storage measured from the arrays
  * numerics: proven bounds hold at the input extremes; requantization rounds half up and clamps to 0..128;
    importance weighting lowers the weighted error it optimizes
  * compatibility: reference16 is exactly the INT16 export; p2 weights never meet p3 features (Python and C++)
  * native: every kernel path this CPU runs (scalar, sse2, avx2, neon, active) equals the independent reference
    and the NumPy emulation, on odd widths, negative inputs and boundary activations; whole p2/p3 policies give
    the emulation's joint logits bit for bit and normalized masked distributions; routing contracts hold
  * calibration refuses test rows; distillation gradients, shard identity checks and a toy pipeline (NumPy)
The Torch QAT contracts run in tests/test_torch_backend.py (the CI job that installs Torch).
Nothing here trains a real model or measures time.
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import native, policy_distill as D, precision as P, precision_calib as C
from fusion.capacity import buffer_width
from fusion.models import PolicyNet, policy_batch
from fusion.nn import MLP, masked_log_softmax
from fusion.runner import find_compiler
from fusion.schema import layout as feature_layout
from training import distill_policy, export_precision, export_weights

REPO = Path(__file__).resolve().parents[1]
SMALL = {"mover": [16], "squad": [64, 40], "act": [16], "target": [8]}


def rows(n: int, seed: int = 0) -> dict:
    return distill_policy.synthetic_rows(feature_layout(), n, seed)


def random_net(rng, sizes, bits, lo=-30, hi=99, scale=7.0) -> tuple[dict, np.ndarray, np.ndarray]:
    m = MLP(sizes, rng)
    lo_v, hi_v = np.full(sizes[0], lo), np.full(sizes[0], hi)
    return P.quantize_mlp(m, np.full(sizes[0], scale), lo_v, hi_v, "squad", {"default": bits, "rules": []}), lo_v, hi_v


def case_header(nets: dict, policy_header: str = "", family: int = 0) -> str:
    out = ["#pragma once"]
    if family:
        out.append(f"#define PCASE_POLICY_FAMILY {family}")
    out.append('#include "policy_precision/lowbit.hpp"')
    if policy_header:
        out.append(policy_header)
    out.append("namespace pcase {")
    out += export_precision.emit_nets(nets)
    out.append("inline const fusion::lowbit::Net* nets[] = {" + ", ".join(f"&{k}" for k in nets) + "};")
    if family:
        out.append("inline const fusion::PolicyModelLP* policy = &fusion::lpmodels::case_model::model;")
    out.append("}")
    return "\n".join(out) + "\n"


class Formats(unittest.TestCase):
    def test_profiles_match_the_declared_json(self):
        self.assertEqual(P.load_profiles(REPO / "profiles" / "p_precision_code.json"), P.PROFILES)

    def test_layer_bits(self):
        mixed = P.PROFILES["mixed4_8"]
        self.assertEqual([P.layer_bits(mixed, "squad", i, 3) for i in range(3)], [8, 4, 8])
        self.assertEqual([P.layer_bits(mixed, "mover", i, 2) for i in range(2)], [8, 8])
        self.assertEqual([P.layer_bits(P.PROFILES["uniform8"], "squad", i, 3) for i in range(3)], [8, 8, 8])

    def test_identities_are_deterministic_and_distinct(self):
        ids = {name: P.profile_identity(name, prof) for name, prof in P.PROFILES.items()}
        self.assertEqual(len(set(ids.values())), 3)
        self.assertEqual(ids, {name: P.profile_identity(name, prof) for name, prof in P.PROFILES.items()})
        L = feature_layout()
        pol = PolicyNet(L, np.random.default_rng(1), SMALL)
        digests = [P.manifest_identity(P.quantize_policy(pol, L, p))["weights_digest"] for p in ("uniform8", "mixed4_8", "uniform8")]
        self.assertEqual(digests[0], digests[2])
        self.assertNotEqual(digests[0], digests[1])

    def test_int4_packing_round_trips(self):
        rng = np.random.default_rng(2)
        for n_in in (1, 15, 16, 17, 31, 32, 33, 100):
            q = rng.integers(-8, 8, (5, n_in))
            packed = P.pack_int4(q)
            self.assertEqual(packed.shape, (5, -(-n_in // 32) * 16))
            np.testing.assert_array_equal(P.unpack_int4(packed, n_in), q)
            full = P.unpack_int4(packed, packed.shape[1] * 2)
            self.assertTrue((full[:, n_in:] == 0).all(), "padding columns decode to zero weights")

    def test_flatten_round_trip(self):
        rng = np.random.default_rng(3)
        for bits in (4, 8, 16):
            net, lo, hi = random_net(rng, [45, 70, 33, 5], bits)
            x = rng.integers(lo, hi + 1, (50, 45))
            back = P.unflatten(P.flatten(net, "h"), "h")
            np.testing.assert_array_equal(P.forward_acc(net, x), P.forward_acc(back, x))
            self.assertEqual(P.digest(P.flatten(net, "h")), P.digest(P.flatten(back, "h")))

    def test_bounds_hold_at_the_input_extremes(self):
        rng = np.random.default_rng(4)
        for bits in (4, 8, 16):
            net, lo, hi = random_net(rng, [40, 64, 9], bits)
            for L in net["layers"]:
                self.assertLessEqual(L["bound"], P.ACC_LIMIT)
            first = net["layers"][0]
            w = first["d"].astype(np.int64)[:, np.arange(40) // 32] * first["q"].astype(np.int64)
            x = np.where(w[:1] >= 0, hi, lo)            # pushes the first row's sum to its signed extreme
            P.forward_acc(net, np.vstack([x, -x.clip(lo, hi), np.tile(lo, (1, 1)), np.tile(hi, (1, 1))]))

    def test_requantization_rounds_half_up_and_clamps(self):
        layer = {"bits": 8, "in": 1, "out": 3, "in_pad": 32, "block": 32, "hidden": True, "q": np.array([[1], [1], [1]]),
                 "d": np.ones((3, 1), np.uint8), "b": np.array([0, 0, 0], np.int32), "mult": np.array([1, 1, 1], np.int32),
                 "rshift": 1}
        last = {"bits": 8, "in": 3, "out": 3, "in_pad": 32, "block": 32, "hidden": False, "q": np.eye(3, dtype=np.int64),
                "d": np.ones((3, 1), np.uint8), "b": np.zeros(3, np.int32), "out_scale": np.ones(3)}
        net = {"layers": [layer, last]}
        # h = clamp((x + 1) >> 1, 0, 128): 3 -> 2 (half up), -3 -> 0 (clamp), 300 -> 128 (clamp)
        for x, want in ((3, 2), (2, 1), (-3, 0), (300, 128)):
            np.testing.assert_array_equal(P.forward_acc(net, np.array([[x]]))[0], [want] * 3)

    def test_importance_lowers_the_weighted_error(self):
        rng = np.random.default_rng(5)
        W = rng.normal(0, 0.3, (24, 96))
        imp = np.exp(rng.normal(0, 1.5, 96))
        for bits in (4, 8):
            plain = P.quantize_layer(W, np.zeros(24), bits, False, False, np.full(96, 128))
            aware = P.quantize_layer(W, np.zeros(24), bits, False, False, np.full(96, 128), imp)
            err = lambda L: (((W - L["row_scale"][:, None] * L["d"].astype(float)[:, np.arange(96) // 32] * L["q"]) ** 2) * imp).sum()  # noqa: E731
            self.assertLessEqual(err(aware), err(plain) + 1e-12)

    def test_storage_is_measured_from_the_arrays(self):
        L = feature_layout()
        pol = PolicyNet(L, np.random.default_rng(6), SMALL)
        mixed = P.quantize_policy(pol, L, "mixed4_8")
        st = P.policy_storage(mixed)
        for h, net in mixed["nets"].items():
            arrays = [P.layer_arrays(x) for x in net["layers"]]
            self.assertEqual(st[h]["weights"], sum(a["w"].nbytes for a in arrays))
            self.assertEqual(st[h]["resident_copies"], 1)
        hidden = mixed["nets"]["squad"]["layers"][1]
        self.assertEqual(hidden["bits"], 4)
        self.assertEqual(st["squad"]["layers"][1]["weights"], hidden["out"] * hidden["in_pad"] // 2)
        ref = P.policy_storage(P.quantize_policy(pol, L, "reference16"))
        self.assertGreater(ref["squad"]["weights"], st["squad"]["weights"])

    def test_reference16_is_the_int16_export(self):
        from fusion.models import ValueNet  # noqa: PLC0415

        L = feature_layout()
        rng = np.random.default_rng(7)
        pol, val = PolicyNet(L, rng, SMALL), ValueNet(L, rng, (32, 8))
        legacy = export_weights.quantize_model(pol, val, 1.0, L)
        ref = P.quantize_policy(pol, L, "reference16")
        self.assertEqual(P.digest(P.policy_flat(ref)), P.digest(export_weights.flat_arrays({h: legacy[h] for h in P.HEADS})))

    def test_family_mismatch_is_refused(self):
        L = feature_layout()
        qp = P.quantize_policy(PolicyNet(L, np.random.default_rng(8), SMALL, "p2"), L, "uniform8")
        qp["family"] = "p3"
        with self.assertRaises(ValueError):
            export_precision.check_family(qp, L)

    def test_reference16_is_not_reexported(self):
        L = feature_layout()
        with self.assertRaises(SystemExit):
            export_precision.export(PolicyNet(L, np.random.default_rng(9), SMALL), L, "reference16", Path("unused"), "x")


@unittest.skipIf(find_compiler() is None, "no C++ compiler")
class Native(unittest.TestCase):
    def run_tool(self, exe: Path, lines: list[str]) -> list[str]:
        r = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return r.stdout.splitlines()

    def test_every_kernel_path_matches_the_emulation(self):
        rng = np.random.default_rng(10)
        shapes = [(4, [1, 3, 2]), (4, [33, 64, 7]), (4, [100, 96, 125]), (8, [5, 40, 3]), (8, [31, 32, 1]),
                  (8, [222, 70, 33, 9]), (16, [47, 31, 2]), (16, [64, 65, 4])]
        nets, specs = {}, []
        for i, (bits, sizes) in enumerate(shapes):
            net, lo, hi = random_net(rng, sizes, bits, lo=-40)
            nets[f"n{i}"] = net
            specs.append((sizes[0], lo, hi))
        exe = native.precision_check(case_header(nets))
        lines, cases = [], []
        for i, (n, lo, hi) in enumerate(specs):
            for k in range(12):
                x = rng.integers(lo, hi + 1)
                if k == 0:
                    x = lo.copy()
                elif k == 1:
                    x = hi.copy()
                elif k == 2:
                    x = np.zeros(n, np.int64)
                cases.append((i, x))
                lines.append(f"net {i} {n} " + " ".join(map(str, x)))
        out = self.run_tool(exe, lines)
        tested = out[0].split()[3:]
        self.assertIn("reference", tested)
        self.assertIn("active", tested)
        per = len(tested)
        for j, (i, x) in enumerate(cases):
            want = P.forward_acc(nets[f"n{i}"], x[None])[0].tolist()
            for line in out[1 + j * per: 1 + (j + 1) * per]:
                name, *vals = line.split()
                with self.subTest(net=i, case=j, path=name):
                    self.assertEqual(list(map(int, vals)), want)

    def policy_case(self, family: str, profile: str):
        L = feature_layout()
        pol = PolicyNet(L, np.random.default_rng(11), SMALL, family)
        qp = P.quantize_policy(pol, L, profile)
        model = export_precision.header(qp, "case_model", P.manifest_identity(qp)).replace("#pragma once\n", "")
        header = case_header({}, model, 3 if family == "p3" else 2)
        width = buffer_width(max(L["in_pad"], L["out"]) for net in qp["nets"].values() for L in net["layers"])
        exe = native.precision_check(header, defines=[f"FUSION_MAX_WIDTH={width}"])
        d = rows(24, seed=12)
        lines = []
        for r in d["rec"]:
            parts = [L.get(r[None], b).reshape(-1) for b in ("g", "u", "m", "legal", "act_legal", "cand", "cand_f")]
            if family == "p3":
                parts.append(L.get(r[None], "pc").reshape(-1))
            lines.append("policy " + " ".join(str(int(v)) for v in np.concatenate(parts)))
        out = self.run_tool(exe, lines)[1:]
        Xr = policy_batch(d["rec"], L, False, family)
        joint = P.joint_logits(qp, Xr)
        return out, joint, Xr, qp

    def test_policy_heads_match_the_emulation(self):
        for family in ("p2", "p3"):
            for profile in ("uniform8", "mixed4_8"):
                out, joint, Xr, qp = self.policy_case(family, profile)
                row = -1
                for line in out:
                    kind, *vals = line.split()
                    if kind == "joint":
                        row += 1
                        got = np.array([float.fromhex(v) if v != "x" else -np.inf for v in vals])
                        want = np.where(Xr.joint_mask[row], joint[row], -np.inf)
                        with self.subTest(family=family, profile=profile, row=row):
                            np.testing.assert_array_equal(got, want)
                    else:
                        k, m = int(vals[0]), int(vals[1])
                        prob = np.array([float.fromhex(v) for v in vals[2:]])
                        if kind == "act":
                            z = P.head_forward(qp, "act", Xr.mover_in[row:row + 1, k, m])[0]
                            mask = Xr.act_legal[row, k, m]
                        else:
                            z = P.head_forward(qp, "target", Xr.target_in(np.full((len(Xr.legal), 3), m))[row, k])[:, 0]
                            mask = Xr.cand_valid[row, k]
                        want = np.exp(masked_log_softmax(z[None], mask[None]))[0] if mask.any() else np.zeros_like(prob)
                        with self.subTest(family=family, profile=profile, head=kind, row=row, unit=k, move=m):
                            np.testing.assert_allclose(prob, np.where(mask, want, 0.0), rtol=0, atol=1e-12)
                            if mask.any():
                                self.assertAlmostEqual(prob.sum(), 1.0, places=12)

    def test_p2_weights_do_not_compile_for_p3(self):
        L = feature_layout()
        qp = P.quantize_policy(PolicyNet(L, np.random.default_rng(13), SMALL, "p2"), L, "uniform8")
        qp["family"] = "p3"
        model = export_precision.header(qp, "case_model", P.manifest_identity(qp)).replace("#pragma once\n", "")
        with self.assertRaises(RuntimeError) as err:
            native.precision_check(case_header({}, model, 3))
        self.assertIn("is not family p3", str(err.exception))

    def test_routing_contracts(self):
        r = subprocess.run([str(native.precision_routing_check())], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("all ok", r.stdout)
        self.assertNotIn("FAIL", r.stdout)

    def test_bench_tool_builds(self):
        L = feature_layout()
        qp = P.quantize_policy(PolicyNet(L, np.random.default_rng(14), SMALL), L, "mixed4_8")
        with tempfile.TemporaryDirectory() as tmp:
            hdr = Path(tmp) / "policy_lp.hpp"
            hdr.write_text(export_precision.header(qp, "bench_case", P.manifest_identity(qp)), encoding="utf-8")
            self.assertTrue(export_precision.bench_binary(hdr, "fusion::lpmodels::bench_case").is_file())


class Calibration(unittest.TestCase):
    def test_test_rows_are_refused(self):
        L = feature_layout()
        pol = PolicyNet(L, np.random.default_rng(15), SMALL)
        for split in ("test", "confirm", "ood_holdout"):
            with self.assertRaises(ValueError):
                C.ActivationStats(pol, split)
            with self.assertRaises(ValueError):
                C.report(pol, L, rows(4)["rec"], rows(4), split)

    def test_statistics_report_and_sensitivity(self):
        L = feature_layout()
        d = rows(80, seed=16)
        for family in ("p2", "p3"):
            pol = PolicyNet(L, np.random.default_rng(17), SMALL, family)
            stats = C.collect(pol, L, d["rec"], d | {"successor_move": d["y_move"]}, "train", batch=32, successors=d["rec"][:16])
            self.assertEqual(stats.rows, {"recorded": 80, "successor": 16})
            imp = stats.importance()
            for h in P.HEADS:
                self.assertEqual([len(v) for v in imp[h]], [w.shape[1] for w in getattr(pol, h).W])
            rep = C.report(pol, L, d["rec"], d, "val", importance=imp)
            for prof in ("uniform8", "mixed4_8"):
                dist = rep["profiles"][prof]["distributions"]
                self.assertEqual(set(dist), set(C.CONTEXTS))
                self.assertEqual(dist["all"]["rows"], 80)
                self.assertGreaterEqual(dist["all"]["joint_kl"], 0.0)
            sens = C.sensitivity(pol, L, d["rec"][:40], {k: v[:40] for k, v in d.items()}, "train")
            self.assertEqual(len(sens), sum(len(getattr(pol, h).W) for h in P.HEADS))


class Distillation(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        L = feature_layout()
        d = rows(30, seed=18)
        X = policy_batch(d["rec"], L, True, "p2")
        X.mover_in, X.squad_in, X.cand_f = (a.astype(np.float64) for a in (X.mover_in, X.squad_in, X.cand_f))
        st, te = PolicyNet(L, np.random.default_rng(1), SMALL), PolicyNet(L, np.random.default_rng(2), SMALL)
        for net in (*st.nets.values(), *te.nets.values()):
            net.W = [w.astype(np.float64) for w in net.W]
            net.b = [b.astype(np.float64) for b in net.b]
            net.gW = [np.zeros_like(w) for w in net.W]
            net.gb = [np.zeros_like(b) for b in net.b]
        mv = d["y_move"].astype(np.int64)
        tp = {k: v.astype(np.float64) for k, v in D.teacher_probs(te, X, mv, 2.0).items()}
        rng = np.random.default_rng(0)
        for alpha in (0.0, 0.5, 1.0):
            cfg = D.DistillConfig(alpha=alpha, tau=2.0, act_weight=0.7, target_weight=0.4)
            st.zero_grad()
            D.distill_loss(st, X, d, d["weight"], tp, cfg, True)
            for name in ("squad", "mover", "act", "target"):
                W, g = getattr(st, name).W[0], getattr(st, name).gW[0]
                i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
                old, eps = W[i, j], 1e-6
                W[i, j] = old + eps
                lp = D.distill_loss(st, X, d, d["weight"], tp, cfg, False)["total"]
                W[i, j] = old - eps
                lm = D.distill_loss(st, X, d, d["weight"], tp, cfg, False)["total"]
                W[i, j] = old
                with self.subTest(alpha=alpha, net=name):
                    self.assertAlmostEqual((lp - lm) / (2 * eps), g[i, j], delta=1e-6)

    def test_teacher_distributions_are_masked_and_conditional(self):
        L = feature_layout()
        d = rows(20, seed=19)
        X = policy_batch(d["rec"], L, True, "p2")
        te = PolicyNet(L, np.random.default_rng(3), SMALL)
        mv = d["y_move"].astype(np.int64)
        tp = D.teacher_probs(te, X, mv, 1.5)
        self.assertTrue((tp["joint"][~X.joint_mask] == 0).all())
        np.testing.assert_allclose(tp["joint"].sum(-1), 1.0, atol=1e-5)
        self.assertTrue((tp["act"][~X.act_mask(mv)] == 0).all())
        # act targets are at the recorded move: another move's context gives another distribution
        other = (mv + 1) % 5
        self.assertFalse(np.allclose(tp["act"], D.teacher_probs(te, X, other, 1.5)["act"]))

    def test_shard_identity_checks(self):
        L = feature_layout()
        d = rows(32, seed=20)
        te = PolicyNet(L, np.random.default_rng(4), SMALL)
        mv = d["y_move"].astype(np.int64)
        ids = np.arange(500, 532)
        with tempfile.TemporaryDirectory() as tmp:
            w = D.TeacherShardWriter(Path(tmp), 32, 8, {"teacher": D.state_digest(te), "family": "p2", "split": "train"})
            for a in (0, 16):
                X = policy_batch(d["rec"][a:a + 16], L, True, "p2")
                w.write(ids[a:a + 16], X, mv[a:a + 16], D.teacher_probs(te, X, mv[a:a + 16], 1.0))
            w.close()
            with self.assertRaises(ValueError):
                D.TeacherShard(Path(tmp), {"family": "p3"})
            sh = D.TeacherShard(Path(tmp), {"family": "p2"})
            X = policy_batch(d["rec"][:16], L, True, "p2")
            sh.batch(0, ids[:16], X, mv[:16])
            with self.assertRaises(ValueError):
                sh.batch(0, ids[:16] + 1, X, mv[:16])
            with self.assertRaises(ValueError):
                sh.batch(0, ids[:16], X, (mv[:16] + 1) % 5)
            with self.assertRaises(ValueError):
                sh.batch(0, ids[:16], policy_batch(d["rec"][16:32], L, True, "p2"), mv[:16])
        with tempfile.TemporaryDirectory() as tmp:
            w = D.TeacherShardWriter(Path(tmp), 32, 8, {"family": "p2"})
            with self.assertRaises(ValueError):
                w.close()                                            # incomplete shard

    def test_toy_pipeline_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = distill_policy.toy(Path(tmp), with_torch=False)
        self.assertEqual(res["warmup"]["config"]["loss"]["alpha"], 1.0)
        self.assertEqual(res["distill"]["config"]["loss"]["alpha"], 0.5)
        self.assertEqual(res["distill"]["config"]["teacher_shard"]["rows"], 96)
        self.assertGreater(res["distill"]["receipt"]["steps"], 0)

    def test_teacher_profile_estimates(self):
        L = feature_layout()
        est = D.teacher_estimates(L, "p2")
        heads = est["heads"]
        self.assertEqual(heads["squad"]["dims"], [222, 4096, 1024, 125])
        self.assertEqual(heads["squad"]["params"], 222 * 4096 + 4096 + 4096 * 1024 + 1024 + 1024 * 125 + 125)
        self.assertEqual(est["params"], sum(h["params"] for h in heads.values()))
        self.assertEqual(D.teacher_estimates(L, "p2", D.STUDENT_XXL)["heads"]["squad"]["dims"], [222, 1024, 256, 125])


if __name__ == "__main__":
    unittest.main()
