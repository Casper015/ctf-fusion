"""Pure-math tests: gradients, quantization, mixture and temperature fits, KL fusion, labels helpers."""
import unittest

import numpy as np

from fusion import quant
from fusion.labels import _window_any, pairwise_outcome
from fusion.mixture import fit_alpha, fit_temperature, kl_fusion, mixture_loss
from fusion.nn import MLP, bce_with_logits, softmax_xent


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        hi = f()
        x[i] = old - eps
        lo = f()
        x[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


class TestGradients(unittest.TestCase):
    def test_mlp_softmax_gradients(self):
        rng = np.random.default_rng(0)
        net = MLP([6, 5, 4], rng, dtype=np.float64)
        x = rng.uniform(0, 1, (7, 6))
        mask = rng.random((7, 4)) < 0.8
        mask[:, 0] = True
        label = np.array([0, 1, 2, 3, 0, -1, 1])
        label = np.where((label >= 0) & ~mask[np.arange(7), np.maximum(label, 0)], 0, label)
        w = rng.uniform(0.5, 2, 7)

        def loss():
            return softmax_xent(net.forward(x, keep=False), mask, label, w, 0.05)[0]

        net.zero_grad()
        _, g = softmax_xent(net.forward(x), mask, label, w, 0.05)
        dx = net.backward(g)
        for p, gp in zip(net.params, net.grads):
            np.testing.assert_allclose(gp, numeric_grad(loss, p), rtol=1e-4, atol=1e-7)
        x_saved = x.copy()

        def loss_x():
            return softmax_xent(net.forward(x, keep=False), mask, label, w, 0.05)[0]

        np.testing.assert_allclose(dx, numeric_grad(loss_x, x), rtol=1e-4, atol=1e-7)
        np.testing.assert_array_equal(x, x_saved)

    def test_bce_gradient(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=(5, 3))
        y = rng.uniform(0, 1, (5, 3))
        w = rng.uniform(0.5, 2, (5, 3))
        _, g = bce_with_logits(z, y, w)
        np.testing.assert_allclose(g, numeric_grad(lambda: bce_with_logits(z, y, w)[0], z), rtol=1e-5, atol=1e-8)


class TestQuant(unittest.TestCase):
    def test_quantized_close_to_float_and_bounded(self):
        rng = np.random.default_rng(2)
        net = MLP([5, 16, 8, 3], rng, dtype=np.float64)
        scale = np.array([1, 10, 100, 400, 2], float)
        lo = np.array([0, -10, 0, 0, -1])
        hi = np.array([1, 10, 100, 400, 1])
        q = quant.quantize(net, scale, lo, hi, out_mult=1.7)
        x = np.stack([rng.integers(l, h + 1, 2000) for l, h in zip(lo, hi)], 1)
        ref = 1.7 * net.forward(x / scale, keep=False)
        got = quant.to_real(q, quant.forward_raw(q, x))
        self.assertLess(np.abs(got - ref).max(), 0.05)
        for layer in q["layers"]:
            self.assertLessEqual(np.abs(layer["w"].astype(np.int64)).max(), quant.W_LIMIT)
            self.assertLessEqual(layer["bound"], quant.ACC_LIMIT)
        extreme = np.stack([np.where(rng.random(5) < 0.5, lo, hi) for _ in range(200)])
        quant.forward_raw(q, extreme)  # the proven bound must hold at the corners

    def test_round_shift_matches_cpp_semantics(self):
        v = np.array([-5, -4, -3, -1, 0, 1, 3, 4, 5], np.int64)
        # C++20: (v + 2) >> 2 with arithmetic shift (floor division by 4)
        np.testing.assert_array_equal(quant._round_shift(v, 2), np.floor((v + 2) / 4).astype(np.int64))


class TestMixture(unittest.TestCase):
    def test_alpha_matches_grid(self):
        rng = np.random.default_rng(3)
        qh = rng.uniform(0.01, 1, 500)
        qt = np.clip(qh * rng.uniform(0.5, 3, 500), 0.001, 1)
        w = rng.uniform(0.5, 2, 500)
        a, loss = fit_alpha(qh, qt, w)
        grid = np.linspace(0, 1, 2001)
        best = grid[np.argmin([mixture_loss(g, qh, qt, w) for g in grid])]
        self.assertAlmostEqual(a, best, delta=1e-3)
        self.assertLessEqual(loss, min(mixture_loss(g, qh, qt, w) for g in grid) + 1e-9)

    def test_alpha_endpoints(self):
        q = np.full(10, 0.5)
        self.assertEqual(fit_alpha(q, q * 0.5)[0], 0.0)
        self.assertEqual(fit_alpha(q * 0.5, q)[0], 1.0)

    def test_temperature_matches_grid(self):
        rng = np.random.default_rng(4)
        logits = rng.normal(size=(400, 6)) * 3
        mask = rng.random((400, 6)) < 0.8
        mask[:, 0] = True
        true = np.where(mask, logits * 0.4 + rng.gumbel(size=(400, 6)), -np.inf).argmax(1)

        def nll(b):
            z = np.where(mask, b * logits, -np.inf)
            z = z - z.max(1, keepdims=True)
            lse = np.log(np.where(mask, np.exp(z), 0).sum(1))
            return float((lse - z[np.arange(400), true]).sum())

        beta = fit_temperature(logits, mask, true)
        grid = np.linspace(0.05, 3, 3000)
        self.assertAlmostEqual(beta, grid[np.argmin([nll(b) for b in grid])], delta=2e-3)

    def test_kl_fusion_is_the_regularized_argmax(self):
        rng = np.random.default_rng(5)
        p0 = np.array([0.5, 0.3, 0.2, 0.0])
        qv = rng.normal(size=4)
        tau = 0.7
        log_p0 = np.log(p0, where=p0 > 0, out=np.full(4, -np.inf))
        mu = kl_fusion(log_p0, qv, tau)
        self.assertEqual(mu[3], 0.0)

        def objective(m):
            s = m > 0
            return float((m * qv).sum() - tau * (m[s] * np.log(m[s] / p0[s])).sum())

        best = objective(mu)
        for _ in range(2000):
            m = rng.dirichlet(np.ones(3))
            self.assertLessEqual(objective(np.append(m, 0.0)), best + 1e-12)


class TestLabelsMath(unittest.TestCase):
    def test_pairwise_outcome(self):
        np.testing.assert_allclose(pairwise_outcome([3, 1, 1, 0]), [1.0, 0.5, 0.5, 0.0])
        np.testing.assert_allclose(pairwise_outcome([2, 2]), [0.5, 0.5])

    def test_window_any(self):
        flags = np.zeros((6, 2), bool)
        flags[3, 0] = True
        np.testing.assert_array_equal(_window_any(flags, 2)[:, 0], [False, False, True, True, False, False])


if __name__ == "__main__":
    unittest.main()
