"""Meowmind PPO trainer contracts (task T0054, training/meow_rl): rewards and GAE, the masked joint policy, the pool,
storage and resume, the integer export, identities, the PPO update and the CLI.

NumPy parts run everywhere. The PyTorch parts (actor parity, PPO update, checkpoint resume, the CLI) are skipped
without PyTorch; CI's CPU-torch job runs them. Synthetic data only: no engine, no compiler.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # optional dependency
    torch = None

from fusion import runner
from roster.strength import utility as roster_utility
from training.meow_rl import gae as G
from training.meow_rl import identity as I
from training.meow_rl import policy as P
from training.meow_rl import quant as Q
from training.meow_rl.pool import Pool
from training.meow_rl.store import RunStore, atomic_json


def random_actor(seed=0, scale=0.3) -> dict:
    rng = np.random.default_rng(seed)
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in P.ACTOR_SHAPES.items()}


def random_obs(n, seed=0):
    """Feature tensors in [-4, 4] and masks with the bridge's invariants (all-stay feasible, some joints masked)."""
    rng = np.random.default_rng(seed)
    feats = np.clip(rng.standard_normal((n, 3, 5, 112)).astype(np.float32), -4, 4)
    mask = rng.random((n, 125)) < 0.5
    mask[:, 0] = True
    return feats, mask


class Rewards(unittest.TestCase):
    def test_utility_matches_the_roster_definition_with_ties(self):
        rng = np.random.default_rng(1)
        for teams in range(2, 16):
            for _ in range(20):
                scores = rng.integers(0, 4, teams).tolist()                 # many ties
                seat = int(rng.integers(teams))
                self.assertAlmostEqual(G.utility(scores, seat), roster_utility(scores, seat))
                self.assertAlmostEqual(G.terminal_reward(scores, seat), 2 * roster_utility(scores, seat) - 1)
        self.assertEqual(G.terminal_reward([5, 5, 5], 1), 0.0)
        self.assertEqual(G.terminal_reward([9, 1], 0), 1.0)
        self.assertEqual(G.terminal_reward([1, 9, 1], 0), -0.5)

    def test_gae_by_hand(self):
        r, v = np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.2, 0.3])
        adv, ret = G.gae(r, v, terminated=True, bootstrap=0.0, gamma=1.0, lam=0.95)
        d = [0.2 - 0.1, 0.3 - 0.2, 1.0 - 0.3]                               # terminal bootstraps 0
        a2 = d[2]
        a1 = d[1] + 0.95 * a2
        a0 = d[0] + 0.95 * a1
        np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-6)
        np.testing.assert_allclose(ret, adv + v, rtol=1e-6)
        adv1, _ = G.gae(r, v, True, 0.0, lam=1.0)                            # lambda 1: Monte Carlo
        np.testing.assert_allclose(adv1, [1 - 0.1, 1 - 0.2, 1 - 0.3], rtol=1e-6)
        adv0, _ = G.gae(r, v, True, 0.0, lam=0.0)                            # lambda 0: one-step TD
        np.testing.assert_allclose(adv0, d, rtol=1e-6)

    def test_truncation_bootstraps_the_next_value(self):
        adv, _ = G.gae(np.zeros(2), np.array([0.5, 0.25]), terminated=False, bootstrap=0.75, lam=0.95)
        np.testing.assert_allclose(adv, [0.25 - 0.5 + 0.95 * 0.5, 0.75 - 0.25], rtol=1e-6)
        with self.assertRaises(ValueError):
            G.gae(np.zeros(2), np.zeros(2), terminated=True, bootstrap=0.3)
        with self.assertRaises(ValueError):
            G.gae(np.array([np.nan]), np.zeros(1), terminated=True, bootstrap=0.0)


class JointPolicy(unittest.TestCase):
    def test_masked_softmax_against_an_independent_oracle(self):
        w = random_actor(2, scale=0.5)
        feats, mask = random_obs(20, 3)
        lp = P.joint_log_probs(w, feats, mask)
        s = P.unit_scores(w, feats).astype(np.float64) / I.TEMPERATURE
        for n in range(20):
            z = np.array([s[n, 0, a] + s[n, 1, b] + s[n, 2, c] if mask[n, 25 * a + 5 * b + c] else -np.inf
                          for a in range(5) for b in range(5) for c in range(5)])
            oracle = z - np.log(np.exp(z[np.isfinite(z)]).sum())
            np.testing.assert_allclose(lp[n][mask[n]], oracle[mask[n]], atol=1e-4)
            self.assertTrue(np.isneginf(lp[n][~mask[n]]).all())
            self.assertAlmostEqual(float(np.exp(lp[n][mask[n]].astype(np.float64)).sum()), 1.0, places=5)

    def test_sampling_never_returns_an_infeasible_joint(self):
        w = random_actor(4)
        feats, mask = random_obs(30, 5)
        for f, m in zip(feats, mask):
            lp = P.joint_log_probs(w, f, m)
            for u in (0.0, 1e-12, 0.5, 1 - 1e-12, np.nextafter(1.0, 0)):
                self.assertTrue(m[P.sample(lp, u)])
        one = np.zeros(125, bool)
        one[0] = True
        self.assertEqual(P.sample(P.joint_log_probs(w, feats[0], one), 0.999), 0)
        self.assertEqual(P.entropy(P.joint_log_probs(w, feats[0], one)), 0.0)

    def test_sampling_follows_the_probabilities(self):
        logp = np.full(125, -np.inf, np.float32)
        logp[[3, 40, 77]] = np.log([0.2, 0.5, 0.3])
        rng = np.random.default_rng(0)
        draws = np.bincount([P.sample(logp, rng.random()) for _ in range(20000)], minlength=125)
        np.testing.assert_allclose(draws[[3, 40, 77]] / 20000, [0.2, 0.5, 0.3], atol=0.015)

    def test_a_very_unlikely_feasible_move_keeps_positive_probability(self):
        """No policy-dependent pruning (the reference's 0.00005 threshold is not part of the mask)."""
        w = random_actor(6)
        w["b2"][:] = 0
        feats = np.zeros((3, 5, 112), np.float32)
        mask = np.ones(125, bool)
        w["w2"][:] = 0
        w["w0"][:] = 0
        w["b0"][:] = 0
        w["b0"][0] = 1.0
        w["w1"][:] = 0
        w["w1"][0, 0] = 1.0
        w["w2"][0, 0] = 0.0
        feats[0, 4, :] = 4.0
        w["w0"][1, :] = 1.0                                                   # unit 0 move 4 scores far above
        w["w1"][1, 1] = 1.0
        w["w2"][0, 1] = 1.0
        lp = P.joint_log_probs(w, feats, mask)
        self.assertTrue(np.isfinite(lp).all())
        self.assertGreater(float(np.exp(lp.astype(np.float64)).min()), 0.0)


class Pools(unittest.TestCase):
    def spec(self, families=8, cap=0.15):
        return {"schema": "ctf-meow-rl-pool-1", "family_cap": cap,
                "bots": [{"name": f"B{f}{m}", "family": f"f{f}", "path": f"x/{f}{m}.cpp"} for f in range(families) for m in range(2)]}

    def test_family_cap_is_enforced(self):
        Pool(self.spec(8))
        with self.assertRaises(ValueError):
            Pool(self.spec(6))                                               # 1/6 > 0.15
        s = self.spec(8)
        s["family_shares"] = {f"f{f}": (3.0 if f == 0 else 1.0) for f in range(8)}
        with self.assertRaises(ValueError):
            Pool(s)

    def test_schedule_is_deterministic_and_covers_counts_and_seats(self):
        pool = Pool(self.spec(8))
        counts = list(range(2, 16))
        specs = [pool.episode(424242, i, counts) for i in range(4000)]
        self.assertEqual(specs[17], pool.episode(424242, 17, counts))
        self.assertNotEqual(specs[17], pool.episode(424243, 17, counts))
        seen = np.bincount([s.teams for s in specs], minlength=16)[2:]
        self.assertTrue((abs(seen / 4000 - 1 / 14) < 0.02).all())
        fam = np.bincount([int(pool.family_of(o)[1:]) for s in specs for o in s.opponents], minlength=8)
        self.assertTrue((fam / fam.sum() < 0.15).all())
        for s in specs[:200]:
            self.assertEqual(len(s.opponents), s.teams - 1)
            self.assertEqual(s.seats()[s.seat], "@learner")
            self.assertTrue(0 <= s.seat < s.teams)


class Storage(unittest.TestCase):
    def test_recover_discards_uncommitted_episodes_and_never_reuses_them(self):
        out = Path(tempfile.mkdtemp(prefix="meow-rl-store-"))
        st = RunStore(out)
        st.log("started", indices=[0, 1, 2])
        st.log("committed", update=1, indices=[0, 1, 2])
        st.log("started", indices=[3, 4])
        st.write_batch(2, {"x": np.zeros(3)}, {"partial": True})          # an uncommitted update's files
        rec = st.recover(committed_next=3, committed_update=1)
        self.assertEqual(rec["discarded"], [3, 4])
        self.assertEqual(rec["next_index"], 5)
        self.assertTrue((out / "discarded" / "update_000002.npz").is_file())
        self.assertEqual(st.journal()[-1]["event"], "discarded")
        (out / "batches" / "update_000001.npz.tmp77").write_bytes(b"partial")   # a crash inside atomic_write
        with open(out / "journal.jsonl", "a") as f:
            f.write('{"event": "started", "ind')                               # a torn append
        rec = st.recover(committed_next=3, committed_update=1)
        self.assertIn("update_000001.npz.tmp77", rec["moved"])
        self.assertIsNotNone(rec["torn_journal_line"])
        atomic_json(out / "x.json", {"a": 1})
        self.assertEqual(json.loads((out / "x.json").read_text()), {"a": 1})
        self.assertEqual([p.name for p in out.iterdir() if ".tmp" in p.name], [])


class Ready:
    def __init__(self, value):
        self.value = value

    def ready(self):
        return True

    def get(self):
        return self.value


class Collection(unittest.TestCase):
    def test_collection_stops_after_a_fault_and_counts_complete_episodes(self):
        from training.meow_rl.train import drive  # noqa: PLC0415  (torch-free)
        pool = Pool({"schema": "ctf-meow-rl-pool-1", "family_cap": 1.0, "bots": [{"name": "X", "family": "x", "path": "x"}]})
        nxt = iter(range(100))
        started = []

        def submit(spec):
            ok = spec.index != 1
            return Ready({"status": "ok" if ok else "environment_fault", "index": spec.index,
                          "arrays": {"action": np.zeros(400 if ok else 0)}})

        res, _ = drive(submit, lambda: pool.episode(5, next(nxt), [2]), 10_000, 3, started.extend)
        self.assertEqual(started, [0, 1, 2])                                  # nothing new after the fault
        self.assertEqual([r["index"] for r in res], [0, 1, 2])
        nxt = iter(range(100))
        res, _ = drive(lambda s: Ready({"status": "ok", "index": s.index, "arrays": {"action": np.zeros(400)}}),
                       lambda: pool.episode(5, next(nxt), [2]), 1000, 2, lambda i: None)
        self.assertEqual(len(res), 3)                                         # 1,200 >= 1,000, every start returned


class Export(unittest.TestCase):
    def test_folding_keeps_the_function_and_the_integer_net_tracks_it(self):
        w = random_actor(7, scale=0.2)
        feats, mask = random_obs(200, 8)
        acts = Q.activations(w, feats)
        folded = Q.fold(w, [a.max(axis=0) + 1e-3 for a in acts])
        x = feats.reshape(-1, 112).astype(np.float64)
        h = x
        for i, (W, b) in enumerate(zip(folded.W, folded.b)):
            h = h @ W.T + b
            if i < 2:
                self.assertLessEqual(float(h.max()), 1.0 + 1e-9)                 # inside the format's [0, 1] range
                h = np.maximum(h, 0)
        np.testing.assert_allclose(h[:, 0], P.unit_scores(w, feats).reshape(-1), atol=1e-4)
        q = Q.quantize(w, feats[:100])
        report = Q.agreement(w, q, feats[100:], mask[100:])
        self.assertGreater(report["greedy_agreement"], 0.8)
        self.assertEqual(q["net"]["layers"][0]["w"].dtype, np.int16)
        for layer in q["net"]["layers"]:
            self.assertLessEqual(layer["bound"], 1 << 30)                       # fusion.quant's proved int32 bound
        self.assertEqual(Q.quantize(w, feats[:100])["digest"], q["digest"])
        self.assertIn("fusion::Net kActor", Q.header(q, "t", {"a": 1}))

    def test_greedy_ties_go_to_the_lowest_index(self):
        w = random_actor(9)
        for k in w:
            w[k][:] = 0
        q = Q.quantize(w, np.zeros((4, 112), np.float32))
        mask = np.zeros(125, bool)
        mask[[30, 7, 99]] = True
        self.assertEqual(Q.greedy(q, np.zeros((3, 5, 112), np.float32), mask), 7)


class Identities(unittest.TestCase):
    def test_profile_and_parameter_counts(self):
        p = I.load_profile()
        self.assertEqual(I.critic_parameters(p["critic"]["hidden"]), 2_001_921)
        self.assertEqual(I.critic_parameters(p["small_critic"]["hidden"]), 28_801)
        bad = copy.deepcopy(p)
        bad["actor"]["hidden"] = [64, 64]
        tmp = Path(tempfile.mkdtemp()) / "p.json"
        tmp.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            I.load_profile(tmp)

    def test_sources_bind_the_reference_and_the_code(self):
        s = I.sources()
        self.assertEqual(s["reference_sha256"], I.REFERENCE_SHA256)
        self.assertIn("tools/cpp/meow_rl/bridge.cpp", s["files"])
        self.assertIn("training/meow_rl/ppo.py", s["files"])
        self.assertNotEqual(I.identity("large_v")["critic"], I.identity("small_v")["critic"])
        with self.assertRaises(ValueError):
            I.identity("huge_v")


def synthetic_batch(w, n=600, seed=0):
    """A batch shaped like a collected one: actions sampled from the NumPy actor, terminal rewards per 100 rows."""
    feats, mask = random_obs(n, seed)
    rng = np.random.default_rng(seed + 1)
    lp = P.joint_log_probs(w, feats, mask)
    action = np.array([P.sample(l, rng.random()) for l in lp])
    old = lp[np.arange(n), action]
    reward = np.zeros(n, np.float32)
    reward[99::100] = rng.choice([-1.0, 0.0, 1.0], n // 100)
    vf = rng.standard_normal((n, 416)).astype(np.float32)
    return feats, mask, action, old, vf, reward


@unittest.skipUnless(torch is not None, "PyTorch not installed")
class TorchLearner(unittest.TestCase):
    def setUp(self):
        from training.meow_rl.nets import Actor, Critic  # noqa: PLC0415
        from training.meow_rl.ppo import Learner, PPOConfig  # noqa: PLC0415
        self.Actor, self.Critic, self.Learner, self.PPOConfig = Actor, Critic, Learner, PPOConfig
        torch.manual_seed(0)
        self.w = random_actor(11, scale=0.3)

    def learner(self, hidden=(64, 32), **cfg):
        torch.manual_seed(0)
        c = self.PPOConfig.from_profile(I.load_profile()["ppo"])
        c.minibatch = 128
        for k, v in cfg.items():
            setattr(c, k, v)
        return self.Learner(self.Actor().load_numpy(self.w), self.Critic(list(hidden)), c, torch.device("cpu"))

    def batch(self, learner, seed=0):
        feats, mask, action, old, vf, reward = synthetic_batch(self.w, seed=seed)
        values = learner.values(vf)
        adv, ret = np.zeros(len(action), np.float32), np.zeros(len(action), np.float32)
        for s in range(0, len(action), 100):
            adv[s:s + 100], ret[s:s + 100] = G.gae(reward[s:s + 100], values[s:s + 100], True, 0.0)
        return {"feats": feats, "mask": mask, "action": action, "old_logp": old, "value_feats": vf, "adv": adv, "ret": ret}

    def test_numpy_and_torch_actors_agree_and_the_first_ratio_is_one(self):
        lr = self.learner()
        feats, mask = random_obs(50, 12)
        np.testing.assert_allclose(lr.log_probs(feats, mask)[mask], P.joint_log_probs(self.w, feats, mask)[mask], atol=1e-5)
        b = self.batch(lr)
        again = lr.log_probs(b["feats"], b["mask"])[np.arange(len(b["action"])), b["action"]]
        np.testing.assert_allclose(np.exp(again - b["old_logp"]), 1.0, atol=1e-5)
        np.testing.assert_array_equal(lr.actor.numpy()["w0"], self.w["w0"])

    def test_critic_sizes(self):
        p = I.load_profile()
        self.assertEqual(self.Critic(p["critic"]["hidden"]).parameters_count(), 2_001_921)
        self.assertEqual(self.Critic(p["small_critic"]["hidden"]).parameters_count(), 28_801)

    def test_update_is_finite_and_moves_both_networks(self):
        lr = self.learner()
        before = {k: v.clone() for k, v in lr.actor.state_dict().items()}
        cb = {k: v.clone() for k, v in lr.critic.state_dict().items()}
        stats = lr.update(self.batch(lr), np.random.default_rng(0))
        self.assertGreater(stats["actor_steps"], 0)
        self.assertEqual(stats["critic_steps"], 4 * 5)                       # 4 epochs x ceil(600 / 128)
        self.assertGreater(stats["max_actor_grad_norm"], 0)
        self.assertGreater(stats["max_critic_grad_norm"], 0)
        self.assertTrue(np.isfinite([stats["pg_loss"], stats["v_loss"], stats["entropy"]]).all())
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in lr.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(cb[k], v) for k, v in lr.critic.state_dict().items()))

    def test_kl_stop_freezes_the_actor_but_not_the_critic(self):
        lr = self.learner(target_kl=1e-12, actor_lr=1e-2)
        before = {k: v.clone() for k, v in lr.actor.state_dict().items()}
        stats = lr.update(self.batch(lr), np.random.default_rng(0))
        self.assertIsNotNone(stats["kl_stop"])
        self.assertLess(stats["actor_steps"], 20)
        self.assertEqual(stats["critic_steps"], 20)
        if stats["actor_steps"] == 0:
            for k, v in lr.actor.state_dict().items():
                self.assertTrue(torch.equal(before[k], v))

    def test_an_action_outside_its_mask_is_refused(self):
        lr = self.learner()
        b = self.batch(lr)
        b["mask"] = b["mask"].copy()
        b["mask"][0, b["action"][0]] = False
        with self.assertRaises(FloatingPointError):
            lr.update(b, np.random.default_rng(0))

    def test_resume_reproduces_the_next_update_exactly(self):
        a = self.learner()
        rng_a = np.random.default_rng(5)
        a.update(self.batch(a, 0), rng_a)
        saved = {"learner": copy.deepcopy(a.state_dict()), "rng": copy.deepcopy(rng_a.bit_generator.state)}
        batch2 = self.batch(a, 1)
        a.update(batch2, rng_a)
        b = self.learner()
        b.load_state_dict(saved["learner"])
        rng_b = np.random.default_rng()
        rng_b.bit_generator.state = saved["rng"]
        b.update(batch2, rng_b)
        for (k, x), y in zip(a.state_dict()["actor"].items(), b.state_dict()["actor"].values()):
            self.assertTrue(torch.equal(x, y), k)
        for x, y in zip(a.state_dict()["critic"].values(), b.state_dict()["critic"].values()):
            self.assertTrue(torch.equal(x, y))


class SyntheticCollector:
    """Stands in for the process collector: complete 100-decision episodes sampled from the given actor snapshot."""

    def __init__(self, fail_at=None):
        self.fail_at, self.calls = fail_at, 0

    def run(self, next_spec, actor, version, target, on_start):
        self.calls += 1
        if self.fail_at == self.calls:
            spec = next_spec()
            on_start([spec.index])
            raise KeyboardInterrupt("simulated crash during collection")
        out, n = [], 0
        while n < target:
            spec = next_spec()
            on_start([spec.index])
            feats, mask, action, old, vf, reward = synthetic_batch(actor, 100, seed=spec.index)
            arrays = {"feats": feats, "mask": mask, "value_feats": vf, "action": action.astype(np.int16),
                      "resolved": np.full((100, 3), -1, np.int16), "logp": old,
                      "turn": np.arange(1, 101, dtype=np.int16), "valid": mask.sum(1).astype(np.int16), "reward": reward}
            rec = {"index": spec.index, "teams": spec.teams, "families": ["f" for _ in spec.opponents],
                   "utility": 0.5, "faults": []}
            out.append({"status": "ok", "index": spec.index, "record": rec, "arrays": arrays})
            n += 100
        return out

    def close(self):
        pass


@unittest.skipUnless(torch is not None and runner.find_compiler(), "needs PyTorch and a C++ compiler (the bridge's warm start)")
class Cli(unittest.TestCase):
    def args(self, out, pool, mode="smoke", decisions=400, seed=777, extra=()):
        return ["--mode", mode, "--arm", "small_v", "--out", str(out), "--pool", str(pool), "--seed-base", str(seed),
                "--dev", "--device", "cpu", "--decisions", str(decisions), "--rollout-decisions", "200",
                "--minibatch", "64", *extra]

    def setUp(self):
        from training.meow_rl import train as T  # noqa: PLC0415
        self.T = T
        self.dir = Path(tempfile.mkdtemp(prefix="meow-rl-cli-"))
        self.pool = self.dir / "pool.json"
        self.pool.write_text(json.dumps({"schema": "ctf-meow-rl-pool-1", "family_cap": 1.0,
                                         "bots": [{"name": "Fake", "family": "f", "path": "x.cpp"}]}))

    def test_smoke_then_full_resume(self):
        out = self.dir / "run"
        r = self.T.train(self.args(out, self.pool), collector=SyntheticCollector(), engine={"sha256": "test"})
        self.assertEqual(r["decisions"], 400)
        checks = json.loads((out / "smoke_report.json").read_text())["checks"]
        self.assertTrue(checks["nonzero_actor_grad"] and checks["nonzero_critic_grad"] and checks["state_round_trip"])
        self.assertLess(checks["max_logp_recompute_gap"], 1e-4)
        self.assertTrue((out / "checkpoints" / "smoke.pt").is_file())
        r = self.T.train(self.args(out, self.pool, "full", 800, extra=["--resume"]), collector=SyntheticCollector(),
                         engine={"sha256": "test"})
        self.assertEqual(r["decisions"], 800)
        self.assertTrue((out / "checkpoints" / "final.pt").is_file())
        committed = [i for e in json.loads("[" + ",".join((out / "journal.jsonl").read_text().split("\n")[:-1]) + "]")
                     if e["event"] == "committed" for i in e["indices"]]
        self.assertEqual(committed, list(range(8)))                          # no episode index used twice

    def test_resume_refuses_a_different_run_and_requires_a_ledger(self):
        out = self.dir / "run2"
        self.T.train(self.args(out, self.pool), collector=SyntheticCollector(), engine={"sha256": "test"})
        with self.assertRaises(SystemExit):
            self.T.train(self.args(out, self.pool, "full", 800, seed=778, extra=["--resume"]), collector=SyntheticCollector(),
                         engine={"sha256": "test"})
        with self.assertRaises(SystemExit):                                  # a run directory is never reused silently
            self.T.train(self.args(out, self.pool), collector=SyntheticCollector(), engine={"sha256": "test"})
        ledger = self.dir / "SEEDS.md"
        ledger.write_text("| 20991231 | test |\n")
        argv = [a for a in self.args(self.dir / "run4", self.pool, seed=20991230) if a != "--dev"]
        with self.assertRaises(SystemExit):
            self.T.train(argv + ["--seed-ledger", str(ledger)], collector=SyntheticCollector(), engine={})

    def test_an_interrupted_collection_is_discarded_on_resume(self):
        out = self.dir / "run5"
        with self.assertRaises(KeyboardInterrupt):
            self.T.train(self.args(out, self.pool, decisions=600), collector=SyntheticCollector(fail_at=2),
                         engine={"sha256": "test"})
        r = self.T.train(self.args(out, self.pool, decisions=600, extra=["--resume"]), collector=SyntheticCollector(),
                         engine={"sha256": "test"})
        self.assertEqual(r["decisions"], 600)
        events = [json.loads(line) for line in (out / "journal.jsonl").read_text().splitlines()]
        discarded = [i for e in events if e["event"] == "discarded" for i in e["indices"]]
        self.assertEqual(discarded, [2])
        committed = [i for e in events if e["event"] == "committed" for i in e["indices"]]
        self.assertNotIn(2, committed)
        self.assertEqual(len(committed), len(set(committed)))


if __name__ == "__main__":
    unittest.main()
