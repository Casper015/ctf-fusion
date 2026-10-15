"""The native Meowmind bridge and the live adapter (task T0054, training/meow_rl).

Needs a C++ compiler (skipped otherwise). Recorded seat streams (bots/p_only/tests/fixtures/stream_n{02,04,15}) drive
the bridge exactly as the engine does. The live-engine tests also need AI_TEST_ROOT; they play one plumbing game,
never evidence.
"""
from __future__ import annotations

import gzip
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import runner
from training.meow_rl import bridge as B
from training.meow_rl import env as E
from training.meow_rl import policy as P
from training.meow_rl import quant as Q
from training.meow_rl.identity import REFERENCE, REPO
from training.meow_rl.pool import EpisodeSpec, Pool

CXX = runner.find_compiler()
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
GAMES = ("n02", "n04", "n15")


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


def observe_game(lib, game: str, choose=None):
    """[(obs, reply)] over a recorded stream; `choose(obs, turn_index)` picks the joint (default all stay)."""
    out = []
    with B.Session(lib) as s:
        for t, block in enumerate(B.split_stream(stream(game))):
            obs = s.observe(block)
            j = choose(obs, t) if choose else 0
            out.append((obs, s.resolve(j)[0]))
    return out


@unittest.skipUnless(CXX, "no C++ compiler")
class Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = B.load()
        cls.w = P.check_actor(B.reference_weights(cls.lib, "policy"))
        cls.tmp = Path(tempfile.mkdtemp(prefix="meow-rl-"))

    def test_warm_start_actor_matches_the_reference(self):
        for game in GAMES:
            for obs, _ in observe_game(self.lib, game):
                np.testing.assert_allclose(P.unit_probs(self.w, obs.feats, obs.unit_legal), obs.ref_prob, atol=2e-5)

    def test_batched_features_equal_the_scalar_reference(self):
        with B.Session(self.lib) as s:
            for block in B.split_stream(stream("n15"))[:120]:
                obs = s.observe(block)
                np.testing.assert_array_equal(s.scalar_features(), obs.feats)
                s.resolve(0)

    def test_resolver_and_reply_equal_the_original_bot(self):
        """The original compiled without search plays its top joint with its own resolver: handing the bridge that
        joint must give the identical reply on every turn; the greedy warm-start actor picks the same joint except
        where the original's probability pruning removed a move."""
        exe = self.tmp / f"meowmind_nosearch{runner.EXE}"
        if not exe.is_file():
            subprocess.run([CXX, "-std=c++20", "-O2", "-DNN_SEARCH=0", *(["-static"] if runner.WINDOWS else []),
                            str(REFERENCE), "-o", str(exe)], check=True, capture_output=True)
        for game in GAMES:
            text = stream(game)
            ref = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True).stdout.splitlines()
            same = greedy_same = 0
            with B.Session(self.lib) as s:
                for block, line in zip(B.split_stream(text), ref):
                    obs = s.observe(block)
                    _, moves, _ = B.parse_reply(line)
                    j = B.joint_index(moves)
                    self.assertTrue(obs.joint_mask[j], f"{game}: the original played an infeasible joint {line}")
                    greedy_same += P.greedy(P.joint_log_probs(self.w, obs.feats, obs.joint_mask)) == j
                    same += s.resolve(j)[0] == line
            self.assertEqual(same, len(ref), game)
            self.assertGreaterEqual(greedy_same, len(ref) - 3, game)

    def test_masks(self):
        for game in GAMES:
            for obs, _ in observe_game(self.lib, game):
                self.assertTrue(obs.joint_mask[0])                              # all stay: the safe fallback
                self.assertEqual(obs.valid, int(obs.joint_mask.sum()))
                for j in np.flatnonzero(obs.joint_mask):
                    for k, m in enumerate(B.joint_moves(j)):
                        self.assertTrue(obs.unit_legal[k, m])
                for k in range(3):
                    self.assertTrue(obs.unit_legal[k, 0])
                    if not obs.alive >> k & 1:
                        self.assertEqual(obs.unit_legal[k].tolist(), [True, False, False, False, False])
                        self.assertEqual(obs.feats[k, 0, 0], -1)
                    else:                                                       # a wall bump is infeasible
                        bump = obs.feats[k, 1:, 7] > 0.5
                        self.assertEqual((~obs.unit_legal[k, 1:]).tolist(), bump.tolist())
                lp = P.joint_log_probs(self.w, obs.feats, obs.joint_mask)
                self.assertTrue(np.isneginf(lp[~obs.joint_mask]).all())
                self.assertAlmostEqual(float(np.exp(lp[obs.joint_mask].astype(np.float64)).sum()), 1.0, places=5)

    def test_observations_do_not_depend_on_the_learners_choices(self):
        """Causality: the next observation comes from the engine's next block only. Two sessions answering every
        turn differently see identical features, masks and value features on the same recorded stream."""
        rng = np.random.default_rng(3)
        a = observe_game(self.lib, "n04")
        b = observe_game(self.lib, "n04", lambda o, t: int(rng.choice(np.flatnonzero(o.joint_mask))))
        for (x, _), (y, _) in zip(a, b):
            np.testing.assert_array_equal(x.feats, y.feats)
            np.testing.assert_array_equal(x.joint_mask, y.joint_mask)
            np.testing.assert_array_equal(x.value_feats, y.value_feats)

    def test_reset_isolation(self):
        fresh = observe_game(self.lib, "n02")
        observe_game(self.lib, "n15")                                           # another episode in the process
        again = observe_game(self.lib, "n02")
        for (x, r1), (y, r2) in zip(fresh, again):
            np.testing.assert_array_equal(x.feats, y.feats)
            self.assertEqual(r1, r2)

    def test_protocol_errors(self):
        blocks = B.split_stream(stream("n02"))
        with B.Session(self.lib) as s:
            with self.assertRaises(B.BridgeError):
                s.observe(blocks[1])                                            # no INIT
        with B.Session(self.lib) as s:
            obs = s.observe(blocks[0])
            with self.assertRaises(B.BridgeError):
                s.observe(blocks[1])                                            # turn 1 not answered
            bad = int(np.flatnonzero(~obs.joint_mask)[0])
            with self.assertRaises(B.BridgeError):
                s.resolve(bad)                                                  # outside the mask: no repair
            s.resolve(0)
            with self.assertRaises(B.BridgeError):
                s.resolve(0)                                                    # answered twice
            with self.assertRaises(B.BridgeError):
                s.observe(blocks[0].split("TURN")[0] + "TURN 1\n")              # truncated block
        for line in ("1 S P S P", "1 Q P S P S P", "x S P S P S P", "1 S Z S P S P"):
            with self.assertRaises(ValueError):
                B.parse_reply(line)

    def test_carrier_hp_reaches_the_features(self):
        """An enemy carrier's HP enters the enemy slots as hp/100 (32 -> 0.32, 100 -> 1.0) with the carrier flag."""
        block = B.split_stream(stream("n02"))[0]

        def with_carrier(hp):
            lines = block.splitlines()
            for i, line in enumerate(lines):
                tok = line.split()
                if tok[:2] == ["U", "3"]:
                    tok[5], tok[6] = str(hp), "0"
                    lines[i] = " ".join(tok)
                if tok[:2] == ["F", "0"]:
                    lines[i] = f"F 0 carried -1 -1 3 -1"
            return "\n".join(lines) + "\n"

        def enemy_slot(obs):
            f = obs.feats[0, 0]
            for j in range(4):
                at = 52 + 9 * j
                if f[at + 4] == 1:                                              # the carrier
                    return f[at + 3]
            self.fail("no carrier among the nearest enemies")

        hp = {}
        for value in (100, 32):
            with B.Session(self.lib) as s:
                hp[value] = enemy_slot(s.observe(with_carrier(value)))
        self.assertAlmostEqual(float(hp[100]), 1.0, places=6)
        self.assertAlmostEqual(float(hp[32]), 0.32, places=6)

    def test_integer_actor_is_bit_exact_in_cpp(self):
        obs = [o for o, _ in observe_game(self.lib, "n15")]
        feats = np.array([o.feats for o in obs])
        masks = np.array([o.joint_mask for o in obs])
        q = Q.quantize(self.w, feats[:200])
        report = Q.agreement(self.w, q, feats[200:], masks[200:])
        self.assertGreater(report["greedy_agreement"], 0.9)
        head = self.tmp / "actor_model.hpp"
        head.write_text(Q.header(q, "test", {"test": True}), encoding="utf-8")
        exe = self.tmp / f"actor_int_check{runner.EXE}"
        subprocess.run([CXX, "-std=c++20", "-O2", f"-I{REPO / 'inference'}", f"-I{REPO / 'tools/cpp/meow_rl'}", "-include",
                        str(head), str(REPO / "tools/cpp/meow_rl/actor_int_check.cpp"), "-o", str(exe)],
                       check=True, capture_output=True)
        sub, sub_masks = feats[200:320], masks[200:320]
        rows = sub.reshape(-1, 112)
        text = [f"{len(rows)} {len(sub)}"]
        text += [" ".join(format(int(v), "x") for v in r.view(np.uint32)) for r in rows]
        text += ["".join("1" if x else "0" for x in m) for m in sub_masks]
        out = subprocess.run([str(exe)], input="\n".join(text) + "\n", capture_output=True, text=True, check=True).stdout.split()
        got = np.array([int(x) for x in out[:len(rows)]])
        np.testing.assert_array_equal(got, Q.forward_int(q, rows))
        greedy = [int(x) for x in out[len(rows):] if x != "G"]
        self.assertEqual(greedy, [Q.greedy(q, f, m) for f, m in zip(sub, sub_masks)])


class FakeBot:
    def __init__(self, log, seat):
        self.log, self.seat, self.faults = log, seat, []

    def send(self, text):
        self.turn = int(text.split("TURN ")[1].split()[0])
        self.log.append(("send", self.seat, self.turn))
        return 0.0

    def receive(self, started, turn):
        self.log.append(("receive", self.seat, turn))
        return f"{turn} S - S - S -"

    def ok(self):
        pass

    def strike(self, turn, reason):
        self.faults.append((turn, reason))

    def report(self):
        return {"name": f"fake{self.seat}", "dead": None, "faults": len(self.faults), "late": 0, "first_ms": 0.0,
                "p99_ms": None, "max_ms": None}

    def close(self):
        pass


class ReplayGame:
    """A recorded seat-0 stream as an engine: the same blocks whatever the orders (2 teams, learner seat 0)."""

    def __init__(self, text, log, score=(3, 1)):
        self.blocks, self.log, self.turn, self.score = B.split_stream(text), log, 0, list(score)

    @property
    def done(self):
        return self.turn >= len(self.blocks)

    def encode(self, i):
        return self.blocks[self.turn]

    def decode(self, i, line):
        turn, moves, acts = B.parse_reply(line)
        if turn != self.turn + 1:
            raise ValueError("wrong turn")
        return {"moves": moves, "acts": acts}

    def step(self, orders):
        self.log.append(("step", None, self.turn + 1))
        self.turn += 1

    def result(self):
        return {"score": self.score, "stats": []}


@unittest.skipUnless(CXX, "no C++ compiler")
class Adapter(unittest.TestCase):
    def test_learner_acts_before_opponent_replies_and_records_the_episode(self):
        text = "\n".join(B.split_stream(stream("n02"))[:40])            # a 40-turn prefix as a short episode
        log = []
        orig = B.Session.observe

        def observe(self_, block):
            o = orig(self_, block)
            log.append(("observe", None, o.turn))
            return o

        B.Session.observe = observe
        try:
            spec = EpisodeSpec(7, 2, 0, 1, 2, 99, ["Fake"])
            w = P.check_actor(B.reference_weights(B.load(), "policy"))
            ep = E.run_episode(spec, w, 0, {"Fake": ["x"]}, {"Fake": {"family": "fake", "sha256": "0"}}, (2.0, 0.05),
                               engine_factory=lambda n, m, f: ReplayGame(text, log),
                               bot_factory=lambda cmd, name, limits: FakeBot(log, 1))
        finally:
            B.Session.observe = orig
        for turn in range(1, 41):
            events = [e for e, _, t in log if t == turn]
            self.assertEqual(events, ["send", "observe", "receive", "step"], turn)
        self.assertEqual(ep.decisions, 40)
        self.assertEqual(ep.reward.tolist(), [0.0] * 39 + [1.0])            # score 3 vs 1: U = 1, reward 2U - 1
        logp = P.joint_log_probs(w, ep.feats, ep.mask)
        np.testing.assert_allclose(logp[np.arange(40), ep.action.astype(int)], ep.logp, atol=1e-6)
        self.assertTrue(ep.mask[np.arange(40), ep.action.astype(int)].all())
        self.assertEqual(ep.record["decisions"], 40)
        self.assertEqual(ep.record["status"], "ok")
        again = E.run_episode(spec, w, 0, {"Fake": ["x"]}, {"Fake": {"family": "fake", "sha256": "0"}}, (2.0, 0.05),
                              engine_factory=lambda n, m, f: ReplayGame(text, []),
                              bot_factory=lambda cmd, name, limits: FakeBot([], 1))
        np.testing.assert_array_equal(again.action, ep.action)                # the episode's own sampling seed

    def test_a_learner_reply_the_engine_refuses_is_a_fault(self):
        text = "\n".join(B.split_stream(stream("n02"))[:5])

        class Refusing(ReplayGame):
            def decode(self, i, line):
                if i == 0:
                    raise ValueError("refused")
                return super().decode(i, line)

        spec = EpisodeSpec(0, 2, 0, 1, 2, 3, ["Fake"])
        w = P.check_actor(B.reference_weights(B.load(), "policy"))
        with self.assertRaises(E.LearnerFault):
            E.run_episode(spec, w, 0, {"Fake": ["x"]}, {"Fake": {"family": "fake", "sha256": "0"}}, (2.0, 0.05),
                          engine_factory=lambda n, m, f: Refusing(text, []), bot_factory=lambda c, n, l: FakeBot([], 1))


    def test_an_opponent_that_cannot_start_is_an_environment_fault(self):
        text = "\n".join(B.split_stream(stream("n02"))[:5])

        class Dead(FakeBot):
            dead = "cannot start: no such file"

        spec = EpisodeSpec(0, 2, 0, 1, 2, 3, ["Fake"])
        w = P.check_actor(B.reference_weights(B.load(), "policy"))
        with self.assertRaises(E.EnvironmentFault):
            E.run_episode(spec, w, 0, {"Fake": ["x"]}, {"Fake": {"family": "fake", "sha256": "0"}}, (2.0, 0.05),
                          engine_factory=lambda n, m, f: ReplayGame(text, []), bot_factory=lambda c, n, l: Dead([], 1))


@unittest.skipUnless(CXX and os.environ.get("AI_TEST_ROOT"), "needs a C++ compiler and AI_TEST_ROOT (an arena checkout)")
class LiveEngine(unittest.TestCase):
    """One plumbing episode through the official engine with real opponent processes (not evidence)."""

    def test_one_live_episode(self):
        from fusion.paths import ai_test_root  # noqa: PLC0415
        root = ai_test_root()
        pool = Pool({"schema": "ctf-meow-rl-pool-1", "family_cap": 1.0,
                     "bots": [{"name": "Baseline", "family": "baseline", "path": "ctf/workspace/baseline.cpp"}]})
        if not (root / "ctf/workspace/baseline.cpp").is_file():
            self.skipTest("this arena layout has no ctf/workspace/baseline.cpp")
        tmp = Path(tempfile.mkdtemp(prefix="meow-rl-live-"))
        commands, info = pool.commands(root, tmp)
        spec = pool.episode(20990101, 0, [3])
        w = P.check_actor(B.reference_weights(B.load(), "policy"))
        res = E.collect((spec, w, 0, commands, info, (2.0, 0.05), str(root), None))
        self.assertEqual(res["status"], "ok", res.get("error"))
        rec, arr = res["record"], res["arrays"]
        self.assertEqual(rec["teams"], 3)
        self.assertEqual(rec["decisions"], len(arr["action"]))
        self.assertEqual(arr["reward"][-1], rec["reward"])
        self.assertAlmostEqual(rec["reward"], 2 * rec["utility"] - 1)
        self.assertTrue((arr["reward"][:-1] == 0).all())
        self.assertEqual(sorted(rec["opponents"]), sorted(str(i) for i in range(3) if i != spec.seat))


if __name__ == "__main__":
    unittest.main()
