"""The agent_v2 PPO adapter (task T0065, docs/AGENT_V2.md): the learning interface runs the deployed decision path.

Needs a C++ compiler; the PyTorch parts (a synthetic smoke update, resume) are skipped without torch. No engine: a
recorded seat stream stands in for the game (tests/test_meow_rl_bridge.py's ReplayGame and FakeBot).
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import runner
from training.meow_rl import bridge as MB
from training.meow_rl import identity as MI
from training.meow_rl import policy as P

try:
    import torch
except ImportError:
    torch = None

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
CXX = runner.find_compiler()


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


@unittest.skipUnless(CXX, "no C++ compiler")
class Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.agent_v2 import bridge as AB  # noqa: PLC0415
        cls.AB = AB
        cls.libpath = AB.build()
        cls.lib = AB.load(cls.libpath)
        cls.tmp = Path(tempfile.mkdtemp(prefix="agent-v2-train-"))
        sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
        from bundle import bundle  # noqa: PLC0415
        text, _ = bundle(REPO / "bots" / "agent_v2" / "variants" / "agent_v2_p2xxl_direct.cpp")
        src = cls.tmp / "bot.cpp"
        src.write_text(text, encoding="utf-8")
        cls.bot = cls.tmp / f"bot{runner.EXE}"
        subprocess.run([CXX, "-std=c++20", "-O2", "-w", *(["-static"] if runner.WINDOWS else []), str(src), "-o", str(cls.bot)],
                       check=True, capture_output=True)

    def test_the_library_names_its_carrier_and_action_schema(self):
        import ctypes  # noqa: PLC0415
        self.lib.mr_action_schema.restype = ctypes.c_char_p
        self.assertEqual(self.lib.mr_action_schema(), b"agent_v2_actions_v1")
        self.assertEqual(self.lib.mr_policy_features(), 112)
        self.assertEqual(self.lib.mr_value_features(), 416)

    def test_frozen_feature_schemas_equal_the_original_encoder(self):
        ref = MB.load()                               # the Meowmind reference bridge: the code the actors were trained on
        for g in ("n02", "n04", "n15"):
            with MB.Session(self.lib) as a, MB.Session(ref) as r:
                for block in MB.split_stream(stream(g)):
                    oa, orr = a.observe(block), r.observe(block)
                    for k in ("feats", "joint_mask", "unit_legal", "value_feats"):
                        np.testing.assert_array_equal(getattr(oa, k), getattr(orr, k), err_msg=f"{g} turn {oa.turn} {k}")
                    a.resolve(0), r.resolve(0)

    def test_training_executes_exactly_the_deployed_decision_path(self):
        for g in ("n02", "n04", "n15"):
            text = stream(g)
            bot = subprocess.run([str(self.bot)], input=text, capture_output=True, text=True, check=True).stdout.splitlines()
            with MB.Session(self.lib) as s:
                for block, line in zip(MB.split_stream(text), bot):
                    o = s.observe(block)
                    t = line.split()
                    j = "SUDLR".index(t[1]) * 25 + "SUDLR".index(t[3]) * 5 + "SUDLR".index(t[5])
                    self.assertTrue(o.joint_mask[j])
                    self.assertEqual(s.resolve(j)[0], line, (g, o.turn))

    def test_an_episode_stores_the_probability_of_the_executed_joint(self):
        sys.path.insert(0, str(REPO))
        from tests.test_meow_rl_bridge import FakeBot, ReplayGame  # noqa: PLC0415
        from training.meow_rl import env as E  # noqa: PLC0415
        from training.meow_rl.pool import EpisodeSpec  # noqa: PLC0415
        text = "\n".join(MB.split_stream(stream("n02"))[:40])
        z = MB.reference_weights(MB.load(), "policy")
        w = P.check_actor(z)
        spec = EpisodeSpec(7, 2, 0, 1, 2, 99, ["Fake"])
        ep = E.run_episode(spec, w, 0, {"Fake": ["x"]}, {"Fake": {"family": "fake", "sha256": "0"}}, (2.0, 0.05),
                           lib_path=self.libpath, engine_factory=lambda n, m, f: ReplayGame(text, []),
                           bot_factory=lambda cmd, name, limits: FakeBot([], 1))
        self.assertEqual(ep.decisions, 40)
        a = ep.action.astype(int)
        self.assertTrue(ep.mask[np.arange(40), a].all())
        logp = P.joint_log_probs(w, ep.feats, ep.mask)
        np.testing.assert_allclose(logp[np.arange(40), a], ep.logp, atol=1e-6)
        again = E.run_episode(spec, w, 0, {"Fake": ["x"]}, {"Fake": {"family": "fake", "sha256": "0"}}, (2.0, 0.05),
                              lib_path=self.libpath, engine_factory=lambda n, m, f: ReplayGame(text, []),
                              bot_factory=lambda cmd, name, limits: FakeBot([], 1))
        np.testing.assert_array_equal(again.action, ep.action)
        np.testing.assert_array_equal(again.resolved, ep.resolved)                # resolved orders are deterministic


class Identity(unittest.TestCase):
    def test_a_new_carrier_identity_with_unchanged_model_contracts(self):
        from training.agent_v2 import identity as AI  # noqa: PLC0415
        a, m = AI.identity("large_v"), MI.identity("large_v")
        self.assertNotEqual(a, m)
        for k in ("actor", "critic", "policy_features", "value_features", "reward_units", "temperature"):
            self.assertEqual(a[k], m[k], k)                                       # old weights can initialize it
        self.assertEqual(a["carrier"], "agent_v2_v1")
        self.assertEqual(a["action_schema"], "agent_v2_actions_v1")
        self.assertEqual(a["resolver"]["rule"], "assign_v1")
        self.assertEqual(a["predictor"]["kind"], "P")
        self.assertTrue(a["predictor"]["models"])
        self.assertIn("inference/agent_v2/resolver.hpp", AI.sources()["files"])

    def test_the_trainer_substitutions_are_explicit_and_undone(self):
        from training.agent_v2 import train as AT  # noqa: PLC0415
        from training.meow_rl import train as T  # noqa: PLC0415
        tmp = Path(tempfile.mkdtemp(prefix="agent-v2-id-"))
        npz = tmp / "actor.npz"
        rng = np.random.default_rng(0)
        np.savez(npz, **{k: rng.standard_normal(s).astype(np.float32) for k, s in P.ACTOR_SHAPES.items()})
        saved = (T.B.build, T.B.reference_weights, T.run_config)
        with AT.substituted(npz, lib=tmp / "fake.dylib") as built:
            self.assertEqual(T.B.build(), built)
            self.assertEqual(set(T.B.reference_weights(None)), set(P.ACTOR_SHAPES))
            base = lambda args, *rest, **kw: {"task": "T0055", "identity": MI.identity(args.arm)}  # noqa: E731
            cfg = AT.agent_v2_config(base, npz)(type("A", (), {"arm": "large_v"}))
            self.assertEqual(cfg["task"], "T0065")
            self.assertEqual(cfg["identity"]["carrier"], "agent_v2_v1")
            self.assertNotEqual(cfg["identity"], MI.identity("large_v"))          # --resume of an old run is refused
        self.assertEqual((T.B.build, T.B.reference_weights, T.run_config), saved)


@unittest.skipUnless(torch is not None and CXX, "needs PyTorch and a C++ compiler")
class Trainer(unittest.TestCase):
    def setUp(self):
        from tests.test_meow_rl import SyntheticCollector  # noqa: PLC0415
        from training.agent_v2 import bridge as AB  # noqa: PLC0415
        from training.agent_v2 import train as AT  # noqa: PLC0415
        from training.meow_rl import train as T  # noqa: PLC0415
        self.Collector, self.AT, self.T, self.lib = SyntheticCollector, AT, T, AB.build()
        self.dir = Path(tempfile.mkdtemp(prefix="agent-v2-cli-"))
        self.pool = self.dir / "pool.json"
        self.pool.write_text(json.dumps({"schema": "ctf-meow-rl-pool-1", "family_cap": 1.0,
                                         "bots": [{"name": "Fake", "family": "f", "path": "x.cpp"}]}))
        self.npz = self.dir / "actor.npz"
        np.savez(self.npz, **MB.reference_weights(MB.load(), "policy"))

    def args(self, out, mode="smoke", decisions=400, extra=()):
        return ["--init-actor", str(self.npz), "--mode", mode, "--arm", "small_v", "--out", str(out), "--pool", str(self.pool),
                "--seed-base", "777", "--dev", "--device", "cpu", "--decisions", str(decisions), "--rollout-decisions", "200",
                "--minibatch", "64", *extra]

    def test_smoke_update_and_resume_on_the_new_identity(self):
        out = self.dir / "run"
        r = self.AT.train(self.args(out), collector=self.Collector(), engine={"sha256": "test"}, lib=self.lib)
        self.assertEqual(r["decisions"], 400)
        checks = json.loads((out / "smoke_report.json").read_text())["checks"]
        self.assertTrue(checks["nonzero_actor_grad"] and checks["state_round_trip"])
        self.assertLess(checks["max_logp_recompute_gap"], 1e-4)
        self.assertEqual(json.loads((out / "config.json").read_text())["identity"]["carrier"], "agent_v2_v1")
        r = self.AT.train(self.args(out, "full", 800, ["--resume"]), collector=self.Collector(), engine={"sha256": "test"}, lib=self.lib)
        self.assertEqual(r["decisions"], 800)

    def test_an_old_carrier_run_is_never_resumed(self):
        out = self.dir / "old"
        self.T.train(self.args(out)[2:], collector=self.Collector(), engine={"sha256": "test"})
        with self.assertRaises(SystemExit):
            self.AT.train(self.args(out, "full", 800, ["--resume"]), collector=self.Collector(), engine={"sha256": "test"}, lib=self.lib)


if __name__ == "__main__":
    unittest.main()
