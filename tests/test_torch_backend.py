"""Parity of the optional PyTorch training backend (fusion/torch_backend.py) with the NumPy reference.

Skipped when PyTorch is not installed. Runs on the CPU build; on a CUDA host
`FUSION_TEST_DEVICE=cuda python -m unittest tests.test_torch_backend` checks the GPU path with the same
tolerances (float summation order differs between BLAS and CUDA; exported integer parity is separate and exact).
"""
from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # optional dependency
    torch = None

from fusion import nn
from fusion.models import PolicyNet, ValueNet, policy_batch, value_inputs
from fusion.schema import layout as feature_layout

DEVICE = os.environ.get("FUSION_TEST_DEVICE", "cpu")


def records(n: int, seed: int = 0):
    """Random extractor-shaped records and labels that exercise every mask case."""
    L = feature_layout()
    rng = np.random.default_rng(seed)
    rec = np.zeros((n, L.width), np.int16)

    def put(name, values):
        off, shape = L.blocks[name]
        rec[:, off:off + int(np.prod(shape))] = values.reshape(n, -1)

    for name in ("g", "u", "m", "cand_f", "v"):
        _, shape = L.blocks[name]
        put(name, rng.integers(0, 120, (n, *shape)))
    legal = rng.random((n, 3, 5)) < 0.6
    legal[:, :, 0] = True                                  # staying is always legal
    put("legal", legal.astype(np.int16))
    act = rng.random((n, 3, 5, 4)) < 0.5
    act[:, :, :, 0] = True
    act[0, 0, :, :] = False                                # an empty action mask (row 0, unit 0)
    put("act_legal", act.astype(np.int16))
    cand = rng.integers(-1, 30, (n, 3, 8))
    cand[1, 2, :] = -1                                     # a unit without attack candidates
    put("cand", cand)
    move = np.stack([[rng.choice(np.nonzero(legal[b, k])[0]) for k in range(3)] for b in range(n)])
    y_act = np.array([[rng.choice(np.nonzero(act[b, k, move[b, k]])[0]) if act[b, k, move[b, k]].any() else -1
                       for k in range(3)] for b in range(n)])
    y_act[2, 1] = -1                                       # an ignored label
    if not act[3, 0, move[3, 0], 3]:
        y_act[3, 0] = 3                                    # a label outside its mask (gradient w (p - target) there)
    valid = cand >= 0
    y_tgt = np.array([[rng.choice(np.nonzero(valid[b, k])[0]) if valid[b, k].any() and rng.random() < 0.7 else -1
                       for k in range(3)] for b in range(n)])
    extra = np.random.default_rng(seed + 1000)              # p3 / v3 blocks, from their own stream
    for name in ("pc", "vx"):
        _, shape = L.blocks[name]
        put(name, extra.integers(0, 20, (n, *shape)))
    return {"rec": rec, "y_move": move.astype(np.int8), "y_joint": (move[:, 0] * 25 + move[:, 1] * 5 + move[:, 2]).astype(np.int16),
            "y_act": y_act.astype(np.int8), "y_tgt": y_tgt.astype(np.int8),
            "weight": rng.uniform(0.5, 2.0, n).astype(np.float32),
            "v_y": np.stack([rng.random(n), rng.random(n) < 0.3, rng.random(n) < 0.2], 1).astype(np.float32)}


def args(**kw):
    base = {"batch": 64, "lr": 1e-3, "weight_decay": 1e-5, "gpu_resident_gb": 6.0, "gpu_window_rows": 1_000_000,
            "epochs": 2, "lr_schedule": "cosine", "min_lr": 1e-4, "patience": 2, "plateau_decay": 1}
    return types.SimpleNamespace(**(base | kw))


CFG = {"smoothing": 0.01, "act_weight": 0.5, "target_weight": 0.5}
# Output biases of the mover and target nets add one constant to every legal choice of a unit (all five moves,
# all candidates): the softmax ignores them, their exact gradient is zero, and both backends move them only by
# rounding noise that Adam scales to a full step. They cannot change any prediction, so parity skips them
# (test_invariant_biases_do_not_change_predictions checks that).
INVARIANT = ("mover.b1", "target.b1")
LABELS = ("y_move", "y_joint", "y_act", "y_tgt")


@unittest.skipIf(torch is None, "PyTorch not installed (optional backend)")
class TestTorchBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fusion import torch_backend as tb  # noqa: PLC0415
        cls.tb, cls.dev = tb, tb.pick_device(DEVICE)
        tb.configure(cls.dev, tf32=False)
        cls.layout = feature_layout()

    def test_clipped_relu_gradient_excludes_the_endpoints(self):
        z = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5], requires_grad=True)
        self.tb._ClippedReLU.apply(z).sum().backward()
        self.assertEqual(z.grad.tolist(), [0.0, 0.0, 1.0, 0.0, 0.0])    # numpy: (z > 0) & (z < 1)

    def test_cross_entropy_matches_numpy_on_every_mask_case(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(6, 5)).astype(np.float32)
        mask = rng.random((6, 5)) < 0.6
        mask[0] = False                                                     # empty mask
        mask[1, 2] = False
        label = np.array([3, 2, -1, 0, 4, 1])                               # row 1: label outside the mask
        mask[3, 0] = mask[4, 4] = mask[5, 1] = True
        w = rng.uniform(0.5, 2, 6).astype(np.float32)
        for smoothing in (0.0, 0.05):
            loss_np, grad_np = nn.softmax_xent(logits, mask, label, w, smoothing)
            z = torch.tensor(logits, requires_grad=True)
            loss = self.tb._SoftmaxXent.apply(z, torch.tensor(mask), torch.tensor(label), torch.tensor(w), smoothing)
            loss.backward()
            self.assertAlmostEqual(float(loss), loss_np, places=4)
            np.testing.assert_allclose(z.grad.numpy(), grad_np, rtol=1e-5, atol=1e-6)
            self.assertTrue(np.isfinite(z.grad.numpy()).all())

    def policy_pair(self, seed=3):
        model_np = PolicyNet(self.layout, np.random.default_rng(seed))
        model_t = PolicyNet(self.layout, np.random.default_rng(seed))
        return model_np, model_t

    def test_one_policy_step_matches_numpy(self):
        d = records(96)
        model_np, model_t = self.policy_pair()
        X = policy_batch(d["rec"], self.layout)
        model_np.zero_grad()
        out_np = model_np.loss(X, {k: d[k] for k in LABELS}, d["weight"], CFG, backward=True)
        opt = nn.Adam(model_np.params, lr=1e-3, weight_decay=1e-5)
        opt.step(model_np.grads)
        tr = self.tb.PolicyTrainer(model_t, self.layout, d, d, CFG, args(batch=96), self.dev)
        rec = torch.from_numpy(d["rec"]).to(self.dev)
        Xt = self.tb.DevicePolicyBatch(rec, tr.dl)
        np.testing.assert_allclose(Xt.mover_in.cpu().numpy(), X.mover_in, rtol=0, atol=0)   # inputs are exact
        y = {k: torch.from_numpy(d[k].astype(np.int64)).to(self.dev) for k in LABELS}
        tr.opt.zero_grad()
        loss, out = tr.net.loss(Xt, y, torch.from_numpy(d["weight"]).to(self.dev), CFG)
        loss.backward()
        tr.opt.step()
        for k in ("joint", "act", "target", "total"):
            self.assertAlmostEqual(float(out[k]), out_np[k], places=5, msg=k)
        grads_t = {f"{name}.{kind}{i}": t.grad.cpu().numpy() for name, net in tr.net.nets.items()
                   for kind, ts in (("W", net.W), ("b", net.b)) for i, t in enumerate(ts)}
        grads_np = {f"{name}.{kind}{i}": g for name, net in model_np.nets.items()
                    for kind, gs in (("W", net.gW), ("b", net.gb)) for i, g in enumerate(gs)}
        for k, g in grads_np.items():
            np.testing.assert_allclose(grads_t[k], g, rtol=1e-4, atol=1e-6 * max(1.0, np.abs(g).max()), err_msg=k)
        st = tr.state()
        for k, v in model_np.state().items():
            if k not in INVARIANT:
                np.testing.assert_allclose(st[k], v, rtol=1e-5, atol=2e-6, err_msg=k)

    def test_policy_epochs_match_numpy_and_streaming_equals_resident(self):
        from training.common import batches  # noqa: PLC0415
        d = records(700, seed=5)
        model_np, model_t = self.policy_pair(7)
        opt = nn.Adam(model_np.params, lr=1e-3, weight_decay=1e-5)
        rng_np = np.random.default_rng(11)
        losses_np = []
        for _ in range(2):
            total = wsum = 0.0
            for idx in batches(len(d["rec"]), 64, rng_np):
                model_np.zero_grad()
                out = model_np.loss(policy_batch(d["rec"][idx], self.layout), {k: d[k][idx] for k in LABELS},
                                    d["weight"][idx], CFG, backward=True)
                opt.step(model_np.grads)
                total += out["total"] * float(d["weight"][idx].sum())
                wsum += float(d["weight"][idx].sum())
            losses_np.append(total / wsum)
        results = {}
        for mode, a in (("resident", args()), ("streamed", args(gpu_resident_gb=0.0, gpu_window_rows=150))):
            _, m = self.policy_pair(7)
            tr = self.tb.PolicyTrainer(m, self.layout, d, d, CFG, a, self.dev)
            self.assertEqual(tr.train.resident, mode == "resident")
            rng = np.random.default_rng(11)
            losses = [tr.train_epoch(rng) for _ in range(2)]
            results[mode] = (losses, tr.state())
        np.testing.assert_allclose(results["resident"][0], losses_np, rtol=2e-4)
        for k, v in model_np.state().items():
            np.testing.assert_array_equal(results["streamed"][1][k], results["resident"][1][k], err_msg=k)
            if k in INVARIANT:
                continue
            self.assertLess(np.abs(results["resident"][1][k] - v).max(), 5e-3, k)       # Adam can amplify rounding
            self.assertLess(np.abs(results["resident"][1][k] - v).mean(), 1e-4, k)
        self.assertEqual(results["streamed"][0], results["resident"][0])

    def test_invariant_biases_do_not_change_predictions(self):
        from training.train_policy import evaluate  # noqa: PLC0415
        d = records(200, seed=17)
        model, _ = self.policy_pair(19)
        before = evaluate(model, self.layout, d, CFG)[0]
        model.mover.b[-1] += 0.37
        model.target.b[-1] -= 0.81
        self.assertAlmostEqual(evaluate(model, self.layout, d, CFG)[0], before, places=5)

    def test_validation_loss_matches_numpy(self):
        from training.train_policy import evaluate  # noqa: PLC0415
        d = records(300, seed=9)
        model_np, model_t = self.policy_pair(13)
        tr = self.tb.PolicyTrainer(model_t, self.layout, d, d, CFG, args(), self.dev)
        self.assertAlmostEqual(tr.evaluate(), evaluate(model_np, self.layout, d, CFG)[0], places=5)

    def test_value_epochs_match_numpy(self):
        from training.common import batches  # noqa: PLC0415
        d = records(900, seed=21)
        cfg = {"aux_weight": 0.25}
        model_np = ValueNet(self.layout, np.random.default_rng(4))
        opt = nn.Adam(model_np.params, lr=1e-3, weight_decay=1e-5)
        rng_np = np.random.default_rng(2)
        losses_np = []
        for _ in range(2):
            total = wsum = 0.0
            for idx in batches(len(d["rec"]), 128, rng_np):
                model_np.zero_grad()
                out = model_np.loss(value_inputs(d["rec"][idx], self.layout), d["v_y"][idx], d["weight"][idx], cfg, backward=True)
                opt.step(model_np.grads)
                total += out["outcome"] * float(d["weight"][idx].sum())
                wsum += float(d["weight"][idx].sum())
            losses_np.append(total / wsum)
        tr = self.tb.ValueTrainer(ValueNet(self.layout, np.random.default_rng(4)), self.layout, d, d, cfg, args(batch=128), self.dev)
        rng = np.random.default_rng(2)
        losses = [tr.train_epoch(rng) for _ in range(2)]
        np.testing.assert_allclose(losses, losses_np, rtol=2e-4)
        st = tr.state()
        for k, v in model_np.state().items():
            self.assertLess(np.abs(st[k] - v).mean(), 1e-4, k)

    def test_p3_batch_matches_numpy(self):
        d = records(64, seed=41)
        dl = self.tb.DeviceLayout(self.layout, self.dev)
        X = policy_batch(d["rec"], self.layout, family="p3")
        Xt = self.tb.DevicePolicyBatch(torch.from_numpy(d["rec"]).to(self.dev), dl, "p3")
        np.testing.assert_allclose(Xt.mover_in.cpu().numpy(), X.mover_in, rtol=1e-6)
        np.testing.assert_allclose(Xt.squad_in.cpu().numpy(), X.squad_in, rtol=1e-6)
        self.assertEqual(X.squad_in.shape[1], 447)

    def test_value_v3_with_ranking_matches_numpy(self):
        """The ranking loss (training.train_value --rank): same pairs, same order, same updates on both backends."""
        from fusion.models import value_columns  # noqa: PLC0415
        from training.common import batches  # noqa: PLC0415
        d = records(700, seed=51)
        cfg = {"aux_weight": 0.25, "rank_weight": 0.5}
        prng = np.random.default_rng(9)
        cand = records(60, seed=52)["rec"]
        a, b = prng.integers(0, 60, 300), prng.integers(0, 60, 300)
        rank = {"cand_v": value_columns(cand, self.layout, "v3"), "a": a, "b": b, "y": (prng.random(300) < 0.5).astype(np.float32),
                "w": prng.random(300).astype(np.float32)}
        rx = (rank["cand_v"] / self.layout.value_scale("v3")).astype(np.float32)
        model_np = ValueNet(self.layout, np.random.default_rng(4), (64, 16), "v3")
        opt = nn.Adam(model_np.params, lr=1e-3, weight_decay=1e-5)
        rng_np, cursor, losses_np = np.random.default_rng(2), {"perm": None, "pos": 0}, []
        for _ in range(2):
            total = wsum = 0.0
            for idx in batches(len(d["rec"]), 128, rng_np):
                model_np.zero_grad()
                out = model_np.loss(model_np.inputs(d["rec"][idx], self.layout), d["v_y"][idx], d["weight"][idx], cfg, backward=True)
                if cursor["perm"] is None or cursor["pos"] >= 300:
                    cursor["perm"], cursor["pos"] = rng_np.permutation(300), 0
                p = cursor["perm"][cursor["pos"]:cursor["pos"] + 64]
                cursor["pos"] += 64
                model_np.rank_loss(rx[a[p]], rx[b[p]], rank["y"][p], rank["w"][p], cfg, backward=True)
                opt.step(model_np.grads)
                total += out["outcome"] * float(d["weight"][idx].sum())
                wsum += float(d["weight"][idx].sum())
            losses_np.append(total / wsum)
        tr = self.tb.ValueTrainer(ValueNet(self.layout, np.random.default_rng(4), (64, 16), "v3"), self.layout, d, d, cfg,
                                  args(batch=128, rank_batch=64), self.dev, rank)
        rng = np.random.default_rng(2)
        losses = [tr.train_epoch(rng) for _ in range(2)]
        np.testing.assert_allclose(losses, losses_np, rtol=2e-4)
        st = tr.state()
        for k, v in model_np.state().items():
            self.assertLess(np.abs(st[k] - v).mean(), 1e-4, k)

    def test_value_sweep_records_each_batch(self):
        """training.value_sweep's bookkeeping on a tiny cache (a plumbing check; its timings mean nothing here)."""
        import json  # noqa: PLC0415

        from fusion import families  # noqa: PLC0415
        from fusion.models import value_columns  # noqa: PLC0415
        from training import value_sweep  # noqa: PLC0415
        d = records(300, seed=61)
        with tempfile.TemporaryDirectory() as tmp:
            split = os.path.join(tmp, "train")
            os.makedirs(split)
            arrays = {"v": value_columns(d["rec"], self.layout, "v3"), "v_y": d["v_y"], "weight": d["weight"],
                      "teams": np.full(300, 4, np.int8), "turn": np.arange(300, dtype=np.int16), "match": np.zeros(300, np.int32),
                      "map": np.zeros(300, np.int64), "tags": np.zeros(300, np.uint8)}
            for k, v in arrays.items():
                np.save(os.path.join(split, f"{k}.npy"), v)
            os.makedirs(os.path.join(tmp, "val"))
            for k, v in arrays.items():
                np.save(os.path.join(tmp, "val", f"{k}.npy"), np.ones_like(v[:80]) if k == "map" else v[:80])
            with open(os.path.join(tmp, "meta.json"), "w") as f:
                json.dump({"features": families.record(self.layout.schema, "v3")}, f)
            out = os.path.join(tmp, "sweep.json")
            value_sweep.main(["--cache", tmp, "--value-family", "v3", "--hidden", "32,8", "--batches", "64", "128",
                              "--steps", "2", "--warmup", "1", "--device", DEVICE, "--out", out])
            rep = json.load(open(out))
            self.assertEqual([r["batch"] for r in rep["results"]], [64, 128])
            self.assertTrue(all(r["timed_steps"] == 2 and r["rows_per_second"] for r in rep["results"]))
            self.assertTrue(all(r["validation_rows"] == 80 and r["val_batch"] == r["batch"] for r in rep["results"]))

    def test_distill_teacher_probabilities_match_numpy(self):
        from fusion.models import value_columns  # noqa: PLC0415
        from training.distill_value import teacher_probs  # noqa: PLC0415
        d = records(200, seed=71)
        teacher = ValueNet(self.layout, np.random.default_rng(6), (64, 16), "v3")
        v = value_columns(d["rec"], self.layout, "v3")
        np.testing.assert_allclose(teacher_probs(teacher, 1.3, v, self.layout, DEVICE), teacher_probs(teacher, 1.3, v, self.layout, "numpy"),
                                   rtol=1e-5, atol=1e-6)

    def test_fit_loop_keeps_the_best_state_in_the_numpy_model(self):
        from training.common import Run, torch_fit  # noqa: PLC0415
        d = records(400, seed=31)
        model = PolicyNet(self.layout, np.random.default_rng(5))
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(tmp, {"kind": "test"})
            a = args(device=DEVICE, tf32=False)
            best = torch_fit(lambda tb, dev: tb.PolicyTrainer(model, self.layout, d, d, CFG, a, dev), a, run, np.random.default_rng(1))
            self.assertTrue(np.isfinite(best))
            self.assertEqual(run.config["backend"]["backend"], "torch")
            from training.train_policy import evaluate  # noqa: PLC0415
            self.assertAlmostEqual(evaluate(model, self.layout, d, CFG)[0], best, places=4)   # the numpy model holds the best weights


@unittest.skipIf(torch is None, "PyTorch not installed (optional backend)")
class TestQuantizationAwareTraining(unittest.TestCase):
    """fusion/torch_qat.py (task T0030): fake quantization is the deployment's arithmetic, not an approximation."""

    HIDDEN = {"mover": [16], "squad": [64, 40], "act": [16], "target": [8]}

    @classmethod
    def setUpClass(cls):
        from fusion import policy_distill, precision, torch_qat  # noqa: PLC0415
        from training.distill_policy import synthetic_rows  # noqa: PLC0415
        cls.P, cls.Q, cls.D = precision, torch_qat, policy_distill
        cls.layout = feature_layout()
        cls.dev = torch.device(DEVICE)
        cls.d = synthetic_rows(cls.layout, 48, seed=31)

    def models(self, family):
        return PolicyNet(self.layout, np.random.default_rng(3), self.HIDDEN, family)

    def test_quantizer_matches_numpy(self):
        for family in ("p2", "p3"):
            pol = self.models(family)
            for profile in ("uniform8", "mixed4_8"):
                qp = self.P.quantize_policy(pol, self.layout, profile)
                qat = self.Q.QATPolicy(pol, self.layout, profile, self.dev)
                for h in self.P.HEADS:
                    for t, n in zip(qat.heads[h].quantized(), qp["nets"][h]["layers"]):
                        with self.subTest(family=family, profile=profile, head=h):
                            np.testing.assert_array_equal(t["q"].cpu().numpy(), n["q"].astype(np.int64))
                            np.testing.assert_array_equal(t["d"].cpu().numpy(), n["d"].astype(np.int64))
                            np.testing.assert_array_equal(t["B"].cpu().numpy(), n["b"].astype(np.int64))
                            if n["hidden"]:
                                np.testing.assert_array_equal(t["mult"].cpu().numpy(), n["mult"].astype(np.int64))
                                self.assertEqual(t["rshift"], n["rshift"])
                            else:
                                np.testing.assert_array_equal(t["out_scale"].cpu().numpy(), n["out_scale"])

    def test_forward_is_the_deployed_arithmetic(self):
        for family in ("p2", "p3"):
            pol = self.models(family)
            X = policy_batch(self.d["rec"], self.layout, normalize=False, family=family)
            mv = self.d["y_move"].astype(np.int64)
            for profile in ("uniform8", "mixed4_8"):
                qp = self.P.quantize_policy(pol, self.layout, profile)
                qat = self.Q.QATPolicy(pol, self.layout, profile, self.dev)
                for h, x in (("mover", X.mover_in), ("squad", X.squad_in), ("act", X.act_in(mv)), ("target", X.target_in(mv))):
                    exact = qat.heads[h].forward(torch.tensor(np.asarray(x, np.int64), device=self.dev), exact_only=True)
                    with self.subTest(family=family, profile=profile, head=h):
                        np.testing.assert_array_equal(exact.cpu().numpy(), self.P.head_forward(qp, h, x))
                zj, _, _ = qat.logits(self.Q.tensors(X, self.dev), torch.tensor(mv, device=self.dev))
                np.testing.assert_allclose(zj.detach().cpu().numpy(), self.P.joint_logits(qp, X).astype(np.float32), rtol=0, atol=1e-5)

    def test_gradients_flow_through_the_surrogate(self):
        pol = self.models("p2")
        qat = self.Q.QATPolicy(pol, self.layout, "mixed4_8", self.dev)
        X = self.Q.tensors(policy_batch(self.d["rec"], self.layout, normalize=False), self.dev)
        y = {k: torch.tensor(self.d[k].astype(np.int64), device=self.dev) for k in LABELS}
        cfg = self.D.DistillConfig(alpha=1.0)
        loss, _ = self.Q.qat_loss(qat, X, y, torch.tensor(self.d["weight"], device=self.dev), None, cfg)
        loss.backward()
        for h in self.P.HEADS:
            for w in qat.heads[h].t.W:
                with self.subTest(head=h):
                    self.assertTrue(torch.isfinite(w.grad).all())
                    self.assertGreater(float(w.grad.abs().sum()), 0.0)

    def test_teacher_kl_matches_numpy(self):
        rng = np.random.default_rng(5)
        z = rng.normal(0, 2, (16, 9))
        mask = rng.random((16, 9)) < 0.7
        mask[3] = False                                        # an empty mask contributes nothing
        logits = np.where(mask, rng.normal(0, 1, (16, 9)), -np.inf)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p = np.where(mask, p, 0.0)
        p = p / np.maximum(p.sum(-1, keepdims=True), 1e-300)
        w = rng.uniform(0.5, 2.0, 16)
        ref, grad = self.D.kl_term(z, p, mask, w, 2.0)
        zt = torch.tensor(z, requires_grad=True)
        val = self.Q.kl_term(zt, torch.tensor(p), torch.tensor(mask), torch.tensor(w), 2.0)
        val.backward()
        self.assertAlmostEqual(float(val), ref, places=9)
        np.testing.assert_allclose(zt.grad.numpy(), grad, rtol=0, atol=1e-12)

    def test_float_distill_loss_and_gradients_match_numpy(self):
        """--backend torch warmup/distill (FloatPolicy on device-built batches) == fusion.policy_distill.distill_loss."""
        from fusion.torch_backend import DeviceLayout  # noqa: PLC0415
        dl = DeviceLayout(self.layout, self.dev)
        mv = self.d["y_move"].astype(np.int64)
        y = {k: torch.tensor(self.d[k].astype(np.int64), device=self.dev) for k in LABELS}
        w = torch.tensor(self.d["weight"], device=self.dev)
        for family in ("p2", "p3"):
            Xn = policy_batch(self.d["rec"], self.layout, True, family)
            teacher = PolicyNet(self.layout, np.random.default_rng(9), self.HIDDEN, family)
            tp = self.D.teacher_probs(teacher, Xn, mv, 2.0)
            for alpha, tps in ((1.0, None), (0.5, tp), (0.0, tp)):
                cfg = self.D.DistillConfig(alpha=alpha, tau=2.0, act_weight=0.5, target_weight=0.5, smoothing=0.01)
                student = self.models(family)
                student.zero_grad()
                ref = self.D.distill_loss(student, Xn, self.d, self.d["weight"], tps, cfg, backward=True)
                fp = self.Q.FloatPolicy(student, self.dev)
                T = self.Q.device_batch(torch.tensor(self.d["rec"], device=self.dev), dl, family)
                tt = None if tps is None else {k: torch.tensor(v, device=self.dev) for k, v in tps.items()}
                loss, parts = self.Q.qat_loss(fp, T, y, w, tt, cfg)
                loss.backward()
                with self.subTest(family=family, alpha=alpha):
                    for k in ("joint", "act", "target", "total"):
                        self.assertAlmostEqual(float(parts[k]), float(ref[k]), delta=1e-5 * max(1.0, abs(float(ref[k]))))
                    for g_t, g_n in zip((p.grad for p in fp.params), student.grads):
                        np.testing.assert_allclose(g_t.cpu().numpy(), g_n, rtol=1e-4, atol=1e-6 * max(1.0, np.abs(g_n).max()))

    def test_torch_stages_train_and_bind_the_teacher_shard_to_its_cache(self):
        import argparse  # noqa: PLC0415

        from training import distill_policy as T  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            T.write_cache(out / "a", self.d, np.arange(100, 148))
            T.write_cache(out / "b", self.d, np.arange(200, 248))
            ca, cb = T.RowCache(out / "a"), T.RowCache(out / "b")
            teacher = PolicyNet(self.layout, np.random.default_rng(2), self.HIDDEN)
            T.teacher_targets(teacher, self.layout, ca, "train", out / "t", tau=2.0, batch=16)
            args = argparse.Namespace(alpha=0.5, tau=2.0, act_weight=0.5, target_weight=0.5, smoothing=0.01, seed=0, epochs=1,
                                      batch=16, lr=1e-3, weight_decay=0.0, init=None, profile="mixed4_8", device=DEVICE,
                                      optimizer="adam", backend="torch")
            student = T.load_student(self.layout, "s", "p2", None, 0)
            with self.D.TeacherShard(out / "t") as sh:
                for stage in ("warmup", "distill"):
                    res = T.train(stage, student, self.layout, ca, args, None if stage == "warmup" else sh, out / stage)
                    self.assertEqual(res["config"]["backend"], "torch")
                    self.assertEqual(res["receipt"]["steps"], 3)
                    self.assertTrue(np.isfinite(res["receipt"]["final"]["total"]))
                with self.assertRaises(SystemExit):              # a shard written for another cache
                    T.train("distill", student, self.layout, cb, args, sh, out / "x")
            ca.close()
            cb.close()

    def test_scheduled_torch_stages_keep_the_best_state(self):
        """--val-cache on the device: the kept state is the best epoch's, for the float student and for QAT."""
        import argparse  # noqa: PLC0415

        from training import distill_policy as T  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            T.write_cache(out / "tr", {k: v[:32] for k, v in self.d.items()}, np.arange(32))
            T.write_cache(out / "va", {k: v[32:] for k, v in self.d.items()}, np.arange(32, 48))
            tr, va = T.RowCache(out / "tr"), T.RowCache(out / "va")
            args = argparse.Namespace(alpha=0.5, tau=1.0, act_weight=0.5, target_weight=0.5, smoothing=0.01, seed=0, epochs=3,
                                      batch=8, lr=3e-2, min_lr=1e-3, lr_schedule="cosine", patience=1, plateau_decay=1,
                                      weight_decay=0.0, init=None, profile="mixed4_8", device=DEVICE, optimizer="adam",
                                      backend="torch", count_weighting="none")
            sup = self.D.DistillConfig(alpha=1.0, smoothing=0.01)
            for stage in ("warmup", "qat"):
                student = T.load_student(self.layout, "s", "p2", None, 0)
                if stage == "qat":
                    teacher = PolicyNet(self.layout, np.random.default_rng(2), self.HIDDEN)
                    T.teacher_targets(teacher, self.layout, tr, "train", out / "t", tau=1.0, batch=16)
                    with self.D.TeacherShard(out / "t") as sh:
                        res = T.train(stage, student, self.layout, tr, args, sh, out / stage, va)
                else:
                    res = T.train(stage, student, self.layout, tr, args, None, out / stage, va)
                log = [json.loads(x) for x in (out / stage / "log.jsonl").read_text().splitlines()]
                with self.subTest(stage=stage):
                    self.assertAlmostEqual(res["receipt"]["best_val"], min(r["val"] for r in log), places=6)
                    self.assertEqual(res["receipt"]["calibrated"], stage != "qat")
                    saved = dict(np.load(out / stage / "model.npz"))
                    best = dict(np.load(out / stage / "best.npz"))
                    for k in best:
                        if not k.startswith("beta"):
                            np.testing.assert_array_equal(saved[k], best[k], err_msg=k)
                    if stage == "warmup":      # the kept float state reproduces the best validation loss
                        student.calibrated = False
                        self.assertAlmostEqual(T.val_loss(student, self.layout, va, args, sup), res["receipt"]["best_val"], places=4)
            tr.close()
            va.close()

    def test_toy_pipeline_runs_every_stage(self):
        from training import distill_policy  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmp:
            res = distill_policy.toy(Path(tmp), with_torch=True)
        self.assertEqual(res["qat"]["config"]["profile"], "mixed4_8")
        self.assertGreater(res["qat"]["receipt"]["steps"], 0)
        self.assertTrue(np.isfinite(res["qat"]["receipt"]["final"]["total"]))


if __name__ == "__main__":
    unittest.main()
