"""End-to-end pipeline tests against the official engine (needs AI_TEST_ROOT).

Plays a few matches between seeded random bots (each takes about a second),
then checks:
  * the C++ features against the engine's own geometry (walls, can_attack,
    spawn order) and against what actually happened next turn (respawns);
  * requested-action labels against the reply lines and the next positions;
  * training, export and bit-exact C++/Python parity of the integer models;
  * the legacy forecast baseline;
  * that the policy network can learn a deterministic labeling.
"""
import os
import shutil
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from fusion import matchlog, native
from fusion.models import PolicyNet, QuantizedTier, policy_batch
from fusion.nn import Adam
from fusion.paths import ai_test_root, load_engine
from fusion.protocol import DELTAS, parse_block, parse_reply
from fusion.schema import layout
from training import collect, dataset, evaluate_offline, export_weights, heuristic_baseline, train_policy, train_value

TEAMS = (2, 3, 5)


@unittest.skipUnless(os.environ.get("AI_TEST_ROOT"), "set AI_TEST_ROOT to the AI_Test checkout")
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fusion-test-"))
        cls.raw, cls.shards = cls.tmp / "raw", cls.tmp / "shards"
        collect.main(["--teams", *map(str, TEAMS), "--matches", "6", "--bots", "Random", "--out", str(cls.raw)])
        dataset.main(["--raw", str(cls.raw), "--out", str(cls.shards), "--teams", *map(str, TEAMS),
                      "--stride", "1", "--keep", "quiet=1,respawn=1"])
        cls.engine, _ = load_engine(ai_test_root())
        cls.layout = layout()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def matches(self):
        for path in matchlog.iter_matches(self.raw):
            m = matchlog.load(path)
            yield m, native.extract(matchlog.seat_stream(m, 0))

    def test_stream_reconstruction(self):
        for m, rec in self.matches():
            self.assertEqual(len(rec), len(m["blocks"]) * m["teams"])
            for me in range(m["teams"]):
                self.assertTrue(matchlog.seat_stream(m, me).startswith(f"INIT {m['teams']} {me} "))

    def test_features_match_engine(self):
        L = self.layout
        unit_col = {n: i for i, n in enumerate(L.names("unit"))}
        move_col = {n: i for i, n in enumerate(L.names("move"))}
        checked = 0
        for m, rec in self.matches():
            n = m["teams"]
            game = self.engine.Game(n, m["map_seed"], flag_seed=m["flag_seed"])
            size = game.size
            home = [bfs(game, base) for base in game.bases]
            blocks = [parse_block(b) for b in m["blocks"]] + [parse_block(m["final_block"])]
            origin_rec, legal_rec = L.get(rec, "origin"), L.get(rec, "legal")
            u_rec, m_rec, cand_rec = L.get(rec, "u"), L.get(rec, "m"), L.get(rec, "cand")
            for t in range(1, len(m["blocks"]) + 1, 7):
                block = blocks[t - 1]
                origin, immune = engine_origins(game, block, t)
                respawned = {e[1]: (e[2], e[3]) for e in blocks[t].events if e[0] == "respawn"}
                for uid, pos in respawned.items():
                    self.assertEqual(origin[uid], pos, f"spawn prediction, turn {t} unit {uid}")
                for team in range(n):
                    r = (t - 1) * n + team
                    for k in range(3):
                        uid = 3 * team + k
                        o = origin[uid]
                        cell = -1 if o is None else o[1] * size + o[0]
                        self.assertEqual(origin_rec[r, k], cell)
                        if o is None:
                            continue
                        self.assertEqual(u_rec[r, k, unit_col["home_dist"]], min(home[team][o], 99))
                        enemies = [u for u in block.units if u[1] != team and u[2] >= 0]
                        threat = 0 if immune[uid] else sum(
                            1 for e in enemies if e[5] < 0 and game.can_attack((e[2], e[3]), o))
                        self.assertEqual(u_rec[r, k, unit_col["threatened_by"]], min(threat, 15))
                        near = sorted((abs(e[2] - o[0]) + abs(e[3] - o[1]), e[0]) for e in enemies)
                        expect = [e for d, e in near if d <= 4][:8]
                        self.assertEqual(list(cand_rec[r, k][:len(expect)]), expect)
                        self.assertTrue((cand_rec[r, k][len(expect):] == -1).all())
                        for mv, (dx, dy) in enumerate(DELTAS):
                            q = (o[0] + dx, o[1] + dy)
                            legal = q not in game.walls
                            self.assertEqual(bool(legal_rec[r, k, mv]), legal)
                            if not legal:
                                continue
                            shooters = 0 if immune[uid] else sum(
                                1 for e in enemies if e[5] < 0 and game.can_attack((e[2], e[3]), q))
                            targets = sum(1 for e in enemies if game.can_attack(q, (e[2], e[3])))
                            self.assertEqual(m_rec[r, k, mv, move_col["shooters"]], min(shooters, 15))
                            self.assertEqual(m_rec[r, k, mv, move_col["targets"]], min(targets, 15))
                            checked += 1
        self.assertGreater(checked, 1000)

    def test_labels(self):
        L = self.layout
        for m, rec in self.matches():
            from fusion.labels import label_match
            lab = label_match(m, rec, L)
            n = m["teams"]
            legal = L.get(rec, "legal")
            for r in range(0, len(rec), 5):
                t, team = int(rec[r, 0]), int(rec[r, 1])
                reqs = parse_reply(m["replies"][t - 1][team], t)
                self.assertEqual(lab["valid"][r], reqs is not None)
                if reqs is None:
                    continue
                for k, req in enumerate(reqs):
                    want = req.move if legal[r, k, req.move] else 0
                    self.assertEqual(lab["y_move"][r, k], want)
            nxt_ok = 0
            for r in np.nonzero(lab["blocked"].any(1))[0]:
                t, team = int(rec[r, 0]), int(rec[r, 1])
                nxt = parse_block(m["blocks"][t] if t < len(m["blocks"]) else m["final_block"])
                for k in np.nonzero(lab["blocked"][r])[0]:
                    u = nxt.units[3 * team + k]
                    o = L.get(rec, "origin")[r, k]
                    self.assertEqual(u[3] * int(m["init"].split()[3]) + u[2], o)
                    nxt_ok += 1
            self.assertEqual(len(rec), len(m["blocks"]) * n)

    def test_train_export_parity(self):
        runs = self.tmp / "runs"
        for tier in ("m23", "m31"):
            pol, val = runs / f"policy_{tier}", runs / f"value_{tier}"
            train_policy.main(["--shards", str(self.shards), "--tier", tier, "--out", str(pol), "--epochs", "1", "--batch", "512",
                              "--val-split", "train"])
            train_value.main(["--shards", str(self.shards), "--tier", tier, "--out", str(val), "--epochs", "1", "--batch", "512",
                              "--val-split", "train"])
            models = self.tmp / "models"
            export_weights.main(["--tier", tier, "--policy", str(pol), "--value", str(val), "--out", str(models)])
            binary = native.infer_check(models / f"{tier}.hpp", f"fusion::models::{tier}")
            q = QuantizedTier(dict(np.load(models / f"{tier}.npz")))
            for m, rec in self.matches():
                if (m["teams"] == 2) != (tier == "m23"):
                    continue
                stream = matchlog.seat_stream(m, 0)
                cpp = native.run_infer_check(binary, stream).reshape(len(rec), -1)
                np.testing.assert_array_equal(cpp, q.raw(rec, self.layout))
            evaluate_offline.main(["--shards", str(self.shards), "--tier", tier, "--policy", str(pol), "--value", str(val),
                                   "--quantized", str(models / f"{tier}.npz"), "--split", "train", "--fit-split", "train",
                                   "--out", str(self.tmp / "reports" / tier)])
            self.assertTrue((self.tmp / "reports" / f"{tier}.md").is_file())

    def test_heuristic_baseline(self):
        heuristic_baseline.main(["--shards", str(self.shards), "--tier", "m31", "--split", "train"])
        d = dataset.load(self.shards, "m31", "train")
        qh = d["qh"]
        ok = ~np.isnan(qh).any(-1)
        self.assertGreater(ok.mean(), 0.99)
        np.testing.assert_allclose(qh[ok].sum(-1), 1.0, atol=1e-5)
        legal = self.layout.get(d["rec"], "legal").astype(bool)
        self.assertTrue((qh[ok][~legal[ok]] == 0).all(), "forecast mass on a wall")

    def test_policy_learns_deterministic_labels(self):
        """Label every unit with 'continue last displacement' (stay if blocked by a wall): learnable exactly."""
        L = self.layout
        d = dataset.load(self.shards, "m31", "train")
        rec = d["rec"][:4000]
        cont = evaluate_offline.cont_move(rec, L)
        legal = L.get(rec, "legal").astype(bool)
        r = np.arange(len(rec))[:, None]
        y_move = np.where(legal[r, np.arange(3)[None, :], cont], cont, 0)
        y = {"y_move": y_move, "y_joint": y_move[:, 0] * 25 + y_move[:, 1] * 5 + y_move[:, 2],
             "y_act": np.full((len(rec), 3), -1), "y_tgt": np.full((len(rec), 3), -1)}
        model = PolicyNet(L, np.random.default_rng(0))
        opt = Adam(model.params, lr=3e-3)
        X = policy_batch(rec, L)
        w = np.ones(len(rec), np.float32)
        first = model.loss(X, y, w, {}, backward=False)["joint"]
        for _ in range(150):
            idx = np.random.default_rng(_).choice(len(rec), 512, replace=False)
            Xb = policy_batch(rec[idx], L)
            model.zero_grad()
            model.loss(Xb, {k: v[idx] for k, v in y.items()}, w[idx], {}, backward=True)
            opt.step(model.grads)
        last = model.loss(X, y, w, {}, backward=False)["joint"]
        self.assertLess(last, 0.25 * first, f"joint loss {first:.3f} -> {last:.3f}")


def bfs(game, sources):
    dist = {tuple(s): 0 for s in sources}
    q = deque(dist)
    while q:
        x, y = q.popleft()
        for dx, dy in DELTAS[1:]:
            p = (x + dx, y + dy)
            if p not in game.walls and p not in dist and 0 <= p[0] < game.size and 0 <= p[1] < game.size:
                dist[p] = dist[(x, y)] + 1
                q.append(p)
    return dist


def engine_origins(game, block, turn):
    """Where each unit moves from this turn, following engine Game.step phase 1."""
    origin = {u[0]: ((u[2], u[3]) if u[2] >= 0 else None) for u in block.units}
    immune = {u[0]: False for u in block.units}
    occupied = {p for p in origin.values() if p is not None}
    for u in block.units:
        if u[2] < 0 and u[6] >= 0 and turn >= u[6]:
            for p in game.spawn_order[u[1]]:
                if p not in occupied:
                    origin[u[0]], immune[u[0]] = p, True
                    occupied.add(p)
                    break
    return origin, immune


if __name__ == "__main__":
    unittest.main()
