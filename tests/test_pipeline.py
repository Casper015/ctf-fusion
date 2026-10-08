"""End-to-end pipeline tests against the official engine (needs AI_TEST_ROOT).

Plays a few matches between seeded test bots (each takes about a second),
then checks:
  * the C++ features against the engine's own geometry (walls, can_attack,
    spawn order) and against what actually happened next turn (respawns);
  * the standalone simulator (inference/sim.hpp) against every recorded turn;
  * requested-action labels against the reply lines and the next positions;
  * per-count shards and phase-2 appends;
  * the model matrix: general model, a dedicated model warm-started from it,
    export, the dispatch registry, bit-exact C++/Python parity, and the
    runtime contract (no per-turn allocation, exact incremental value);
  * the legacy forecast baseline;
  * that the policy network can learn a deterministic labeling.
"""
import json
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
        cls.engine = load_engine(ai_test_root())
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

    def test_simulator_matches_engine(self):
        for m, _ in self.matches():
            exact, summary = native.sim_check_match(m, self.tmp)
            self.assertTrue(exact, f"{m['match_id']}: {summary}")
            self.assertIn("skipped 0", summary)
            self.assertEqual(int(summary.split("compared ")[1].split()[0]), len(m["blocks"]))  # positive coverage

    def test_append_keeps_existing_shards(self):
        out = self.tmp / "append"
        dataset.main(["--raw", str(self.raw), "--out", str(out), "--teams", "2", "--stride", "1", "--keep", "quiet=1,respawn=1"])
        before = sorted(p.relative_to(out) for p in out.rglob("shard_*.npz"))
        dataset.main(["--raw", str(self.raw), "--out", str(out), "--teams", "2", "3", "--stride", "1",
                      "--keep", "quiet=1,respawn=1", "--append"])
        after = sorted(p.relative_to(out) for p in out.rglob("shard_*.npz"))
        self.assertTrue(set(before) <= set(after))
        full = dataset.load_index(self.shards)["counts"]
        index = dataset.load_index(out)
        self.assertEqual({k: v for k, v in index["counts"].items() if k[:3] in ("n02", "n03")},
                         {k: v for k, v in full.items() if k[:3] in ("n02", "n03")})
        self.assertEqual(len(index["history"]), 2)
        self.assertEqual(index["manifest"]["matches_by_count"], {"n02": 6, "n03": 6})
        with self.assertRaises(SystemExit):  # different sampling options cannot extend the dataset
            dataset.main(["--raw", str(self.raw), "--out", str(out), "--teams", "5", "--stride", "2", "--append"])

    def test_fixture_holdout_reserves_whole_matches(self):
        raw = self.tmp / "noise_raw"
        if not raw.exists():
            collect.main(["--teams", "3", "--matches", "1", "--bots", "Random,Noise,Random", "--out", str(raw)])
        for mode in ("fixture", "actor-row"):
            out = self.tmp / f"holdout_{mode}"
            dataset.main(["--raw", str(raw), "--out", str(out), "--teams", "3", "--stride", "1", "--keep", "quiet=1,respawn=1",
                          "--holdout-family", "noise", "--holdout-mode", mode])
            noise_team = dataset.load_index(out)["matches"][0]["families"].index("noise")
            ood = dataset.load(out, [3], "ood")
            self.assertEqual(set(np.unique(ood["team"])), {0, 1, 2} if mode == "fixture" else {noise_team})
            seen = [s for s in ("train", "val", "test") if (out / "n03" / s).is_dir()]
            if mode == "fixture":
                self.assertEqual(seen, [])  # no row of a held-out fixture reaches any other split
                self.assertEqual(dataset.load_index(out)["matches"][0]["split"], "ood")

    def test_noise_rows_are_value_data_not_policy_labels(self):
        from training.common import policy_rows
        raw, shards = self.tmp / "noise_raw", self.tmp / "noise_shards"
        if not raw.exists():
            collect.main(["--teams", "3", "--matches", "1", "--bots", "Random,Noise,Random", "--out", str(raw)])
        dataset.main(["--raw", str(raw), "--out", str(shards), "--teams", "3", "--stride", "1", "--keep", "quiet=1,respawn=1"])
        families = dataset.load_index(shards)["families"]
        rows = {k: np.concatenate([dataset.load(shards, [3], s)[k] for s in ("train", "val", "test")
                                   if (shards / "n03" / s).is_dir()]) for k in ("valid", "family")}
        noise = rows["family"] == families.index("noise")
        self.assertTrue(noise.any())
        keep = policy_rows(rows, shards)
        self.assertFalse((keep & noise).any())
        self.assertTrue(keep[~noise].all())
        self.assertTrue(policy_rows(rows, shards, keep_unlabeled=True)[noise].all())

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

    def test_model_matrix_export_parity(self):
        runs, models = self.tmp / "runs", self.tmp / "models"
        common = ["--shards", str(self.shards), "--epochs", "1", "--batch", "512", "--val-split", "train"]
        for kind, trainer in (("policy", train_policy), ("value", train_value)):
            trainer.main([*common, "--model", "general", "--out", str(runs / f"{kind}_general")])
            trainer.main([*common, "--model", "n03", "--init", str(runs / f"{kind}_general"), "--out", str(runs / f"{kind}_n03")])
        for model, counts in (("general", ["2", "3", "5"]), ("n03", ["3"])):
            pol, val = runs / f"policy_{model}", runs / f"value_{model}"
            report = self.tmp / "reports" / model
            evaluate_offline.main(["--shards", str(self.shards), "--model", model, "--teams", *counts, "--policy", str(pol),
                                   "--value", str(val), "--split", "train", "--fit-split", "train", "--out", str(report)])
            rep = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(rep["identity"]["model"], model)
            self.assertEqual(rep["identity"]["adapter"], "joint-v2")
            for gate in rep["alpha"].values():  # a few test matches can never open the gate
                self.assertFalse(gate["use"])
                self.assertIn("maps <", gate["reason"])
            export_weights.main(["--model", model, "--policy", str(pol), "--value", str(val),
                                 "--report", str(report.with_suffix(".json")), "--out", str(models)])
            header = models / f"weights_{model}.hpp"
            q = QuantizedTier(dict(np.load(models / f"weights_{model}.npz")))
            self.assertEqual(json.loads((models / "MANIFEST.json").read_text(encoding="utf-8"))[model]["identity"]["weights_digest"],
                             rep["identity"]["weights_digest"])
            parity = native.infer_check(header, f"fusion::models::{model}")
            runtime = native.model_tool("runtime_check", header, f"fusion::models::{model}")
            for m, rec in self.matches():
                stream = matchlog.seat_stream(m, 0)
                np.testing.assert_array_equal(native.run_infer_check(parity, stream, len(rec)), q.raw(rec, self.layout))
                r = native.run_tool(runtime, stream.encode())
                self.assertEqual(r.returncode, 0, r.stdout.decode() + r.stderr.decode())
                self.assertIn("allocating_turns 0", r.stdout.decode())
        with self.assertRaises(SystemExit):  # the general report cannot approve the n03 artifact
            export_weights.main(["--model", "n03", "--policy", str(runs / "policy_n03"), "--value", str(runs / "value_n03"),
                                 "--report", str(self.tmp / "reports" / "general.json"), "--out", str(self.tmp / "rejected")])
        with self.assertRaises(SystemExit):  # nor a different adapter than the one it was fitted for
            export_weights.main(["--model", "n03", "--policy", str(runs / "policy_n03"), "--value", str(runs / "value_n03"),
                                 "--report", str(self.tmp / "reports" / "n03.json"), "--adapter", "marginal-v1",
                                 "--out", str(self.tmp / "rejected")])
        config = __import__("json").loads((runs / "policy_n03" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["init"], str(runs / "policy_general"))
        self.assertEqual(config["data"]["team_counts"], [3])
        out = native.run_tool(native.registry_check(models)).stdout.decode().split("\n")
        served = dict(line.split()[:2] for line in out if line)
        self.assertEqual(served["3"], "n03")      # dedicated model exported
        self.assertEqual(served["2"], "general")  # n02 not exported: falls back to general
        self.assertEqual(served["9"], "general")  # no dedicated model by design
        self.assertTrue(all(line.split()[3] == "2" for line in out if line and line.split()[1] != "-"))  # joint-v2

    def test_heuristic_baseline(self):
        heuristic_baseline.main(["--shards", str(self.shards), "--teams", "3", "5", "--split", "train"])
        meta = dataset.load_meta(self.shards, [3, 5], "train")
        self.assertEqual((meta["qh_shards"], len(meta["baselines"])), (meta["shards"], 1))
        ok, reason, _ = evaluate_offline.baseline_status(self.shards, [2, 3], ["train"])
        self.assertFalse(ok)  # n02 has no q_H: the legacy gate stays closed
        self.assertIn("missing", reason)
        d = dataset.load(self.shards, [3, 5], "train")
        qh = d["qh"]
        ok = ~np.isnan(qh).any(-1)
        self.assertGreater(ok.mean(), 0.99)
        np.testing.assert_allclose(qh[ok].sum(-1), 1.0, atol=1e-5)
        legal = self.layout.get(d["rec"], "legal").astype(bool)
        self.assertTrue((qh[ok][~legal[ok]] == 0).all(), "forecast mass on a wall")

    def test_policy_learns_deterministic_labels(self):
        """Label every unit with 'continue last displacement' (stay if blocked by a wall): learnable exactly."""
        L = self.layout
        d = dataset.load(self.shards, [3, 5], "train")
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
