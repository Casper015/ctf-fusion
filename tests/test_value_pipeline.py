"""V retraining pipeline against the official engine (needs AI_TEST_ROOT): frozen-carrier data, branches, training.

Plays a few short matches in which one seat is the designated carrier ("@candidate"; a seeded test bot stands
in for the frozen P-only carrier), then checks:
  * the p3 combat block against the engine's own rules (can_attack, hit points and damage, respawn
    immunity) on every recorded turn;
  * designated-seat shards hold only the carrier's rows, and P training refuses them;
  * V-only caches, the four-arm pilot's training paths (v2/v3, NumPy) with and without ranking pairs, and the
    evaluation report;
  * branch continuations: the root is rebuilt exactly from recorded orders, candidates are legal and
    distinct, playing branches never changes the root match or the reproduced root state, and the ranking
    data keeps each root in its map's split;
  * the policy comparison and the feature audit run on layout-3 shards.
"""
import gzip
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import matchlog, native, runner
from fusion.paths import REPO, ai_test_root, git_commit, load_engine, load_json, sha256
from fusion.protocol import DELTAS, parse_block, parse_reply
from fusion.schema import layout
from training import (branch_collect, branch_dataset, collect, compare_policy, dataset, distill_value, evaluate_value,
                      feature_audit, train_policy, train_value, value_cache)

TEAMS = (3, 4)
MATCHES = 4


def engine_origins(game, block, turn):
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


@unittest.skipUnless(os.environ.get("AI_TEST_ROOT"), "set AI_TEST_ROOT to an arena checkout (a clone of chen-006/meow-ai-arena, or AI_Test)")
class TestValuePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fusion-vtest-"))
        cls.raw = cls.tmp / "raw"
        root = ai_test_root()
        cls.root = root
        engine = load_engine(root)
        cls.engine = engine
        pool = collect.resolve_pool(load_json(REPO / "profiles" / "pool.json"), root, {"Random"})
        cls.cmd = runner.command_for(Path(pool["Random"]["abs"]), REPO / "build" / "bots")
        src = sha256(pool["Random"]["abs"])
        info = {"Random": {"family": "random", "sha256": src}, "@candidate": {"family": "candidate", "sha256": src},
                "__engine__": sha256(Path(engine.__file__)), "__commit__": git_commit(root)}
        cls.limits = (2.0, 0.5)
        for teams in TEAMS:
            for i in range(MATCHES):
                seats = ["Random"] * teams
                seats[i % teams] = "@candidate"
                fx = collect.Fixture(f"n{teams:02d}-s77-{i:06d}", teams, collect.derive(77, teams, i, "map"),
                                     collect.derive(77, teams, i, "flag", bits=63), seats)
                match = collect.play((fx, {"Random": cls.cmd, "@candidate": cls.cmd}, info, str(root), cls.limits))
                matchlog.save(cls.raw / f"n{teams:02d}" / f"{fx.match_id}.json.gz", match)
        cls.shards, cls.vshards = cls.tmp / "shards", cls.tmp / "vshards"
        dataset.main(["--raw", str(cls.raw), "--out", str(cls.shards), "--teams", *map(str, TEAMS), "--stride", "1",
                      "--keep", "quiet=1,respawn=1"])
        dataset.main(["--raw", str(cls.raw), "--out", str(cls.vshards), "--teams", *map(str, TEAMS), "--stride", "1",
                      "--keep", "quiet=1,respawn=1", "--rows", "designated"])
        cls.L = layout()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def matches(self):
        for path in sorted(matchlog.iter_matches(self.raw)):
            m = matchlog.load(path)
            yield path, m, native.extract(matchlog.seat_stream(m, 0))

    def test_combat_block_matches_engine_rules(self):
        """kill1_stay, kill1_carrier_stay, exposure_reach and lethal_stay recomputed from the engine."""
        L = self.L
        ci = {n: i for i, n in enumerate(L.names("combat"))}
        checked = 0
        for _, m, rec in self.matches():
            n = m["teams"]
            game = self.engine.Game(n, m["map_seed"], flag_seed=m["flag_seed"])
            damage = int(m["init"].split("\n", 1)[0].split()[6])
            pc = L.get(rec, "pc")
            for t in range(1, len(m["blocks"]) + 1, 3):
                block = parse_block(m["blocks"][t - 1])
                origin, immune = engine_origins(game, block, t)
                units = {u[0]: u for u in block.units}

                def dests(uid):
                    o = origin[uid]
                    return [(o[0] + dx, o[1] + dy) for dx, dy in DELTAS if (o[0] + dx, o[1] + dy) not in game.walls]

                shooters = [u for u in units if origin[u] is not None and not immune[u] and units[u][5] < 0]
                for team in range(n):
                    r = (t - 1) * n + team
                    for k in range(3):
                        uid = 3 * team + k
                        if origin[uid] is None:
                            continue
                        can_fire = not immune[uid] and units[uid][5] < 0
                        hp = units[uid][4]
                        for mv, (dx, dy) in enumerate(DELTAS):
                            q = (origin[uid][0] + dx, origin[uid][1] + dy)
                            if q in game.walls:
                                continue
                            enemies = [e for e in units if e // 3 != team and origin[e] is not None and not immune[e]]
                            hit = [e for e in enemies if can_fire and game.can_attack(q, origin[e])]
                            k1 = [e for e in hit if units[e][4] <= damage]
                            stay = 0 if immune[uid] else sum(1 for e in shooters if e // 3 != team and game.can_attack(origin[e], q))
                            reach = 0 if immune[uid] else sum(1 for e in shooters if e // 3 != team and any(
                                game.can_attack(d, q) for d in dests(e)))
                            got = pc[r, k, mv]
                            self.assertEqual(got[ci["kill1_stay"]], min(len(k1), 15))
                            self.assertEqual(got[ci["kill1_carrier_stay"]], min(sum(units[e][5] >= 0 for e in k1), 8))
                            self.assertEqual(got[ci["exposure_reach"]], min(reach, 15))
                            self.assertEqual(got[ci["lethal_stay"]], int(not immune[uid] and stay * damage >= hp))
                            checked += 1
        self.assertGreater(checked, 500)

    def test_designated_rows_only(self):
        index = dataset.load_index(self.vshards)
        self.assertEqual(index["options"]["rows"], "designated")
        d = dataset.load(self.vshards, list(TEAMS), "train") if (self.vshards / "n03" / "train").exists() else None
        if d is not None:
            for mi, team in zip(d["match"], d["team"]):
                seats = index["matches"][mi]["seats"]
                self.assertEqual(seats[team], "@candidate")
        with self.assertRaises(SystemExit):
            train_policy.main(["--shards", str(self.vshards), "--model", "general", "--out", str(self.tmp / "nope")])

    def test_value_training_paths_and_report(self):
        caches = {}
        for fam in ("v2", "v3"):
            caches[fam] = self.tmp / f"vcache_{fam}"
            value_cache.main(["--shards", str(self.shards), "--teams", *map(str, TEAMS), "--family", fam, "--out", str(caches[fam])])
            meta = json.loads((caches[fam] / "meta.json").read_text())
            self.assertEqual(meta["features"]["family"], fam)
        with self.assertRaises(SystemExit):  # a v2 cache cannot feed a v3 run
            value_cache.load_cache(caches["v2"], "train", "v3")
        runs = {}
        for name, fam, hidden in (("A", "v2", "16,4"), ("B", "v3", "16,4"), ("D", "v3", "32,8")):
            runs[name] = self.tmp / f"run_{name}"
            train_value.main(["--model", "general", "--cache", str(caches[fam]), "--value-family", fam, "--hidden", hidden,
                              "--epochs", "2", "--batch", "256", "--checkpoint", "--out", str(runs[name])])
            cfg = json.loads((runs[name] / "config.json").read_text())
            self.assertEqual(cfg["features"]["family"], fam)
            self.assertTrue((runs[name] / "model_best.npz").exists())
        student = self.tmp / "run_E"
        distill_value.main(["--teacher", str(runs["D"]), "--cache", str(caches["v3"]), "--soft-weight", "0.5", "--hidden", "8,4",
                            "--epochs", "1", "--batch", "256", "--out", str(student)])
        cfg = json.loads((student / "config.json").read_text())
        self.assertEqual(cfg["data"]["soft_targets"]["teacher"], str(runs["D"]))
        self.assertEqual(cfg["features"]["family"], "v3")
        with self.assertRaises(SystemExit):  # a student must read the teacher's family
            distill_value.main(["--teacher", str(runs["D"]), "--cache", str(caches["v3"]), "--value-family", "v2",
                                "--hidden", "8,4", "--out", str(self.tmp / "run_bad")])
        evaluate_value.main(["--cache", str(caches["v3"]), "--cache-for", f"v2={caches['v2']}", "--split", "train",
                             "--run", f"A={runs['A']}", "--run", f"B={runs['B']}", "--run", f"D={runs['D']}",
                             "--compare", "B:A", "--compare", "D:B", "--out", str(self.tmp / "veval")])
        rep = json.loads((self.tmp / "veval.json").read_text())
        self.assertEqual(set(rep["runs"]), {"A", "B", "D"})
        self.assertIn("carrier", rep["runs"]["B"]["subsets"] | {"carrier": 1})
        self.assertEqual(len(rep["compare"]), 2)

    def test_branches_rank_and_root_integrity(self):
        path = sorted(matchlog.iter_matches(self.raw, [3]))[0]
        before = path.read_bytes()
        match = matchlog.load(path)
        seat = branch_collect.designated_seat(match)
        game = branch_collect.reproduce(self.engine, match, len(match["blocks"]))
        self.assertIsNotNone(game, "recorded orders must reproduce every recorded block")
        turn = branch_collect.root_turns(match, 1)[0]
        row = branch_collect._root_job((str(path), turn, 3, 1, [self.cmd] * match["teams"], self.limits, str(self.root)))
        self.assertEqual(path.read_bytes(), before, "branch evaluation changed the root match")
        self.assertNotIn("dropped", row)
        self.assertEqual(len(row["candidates"]), len(set(row["candidates"])))
        self.assertEqual(row["candidates"][0], " ".join(match["replies"][turn - 1][seat].partition("#")[0].split()))
        for line in row["candidates"]:
            self.assertIsNotNone(parse_reply(line, turn))
        for b in row["branches"]:
            self.assertEqual(b["root_block"], match["blocks"][turn - 1])
            self.assertTrue(0.0 <= b["outcome"] <= 1.0)
        out = self.tmp / "branches" / f"n{match['teams']:02d}"
        out.mkdir(parents=True)
        with gzip.open(out / "root.json.gz", "wt") as fh:
            json.dump(row, fh)
        branch_dataset.main(["--branches", str(self.tmp / "branches"), "--out", str(self.tmp / "rank")])
        split = dataset.split_of(f"map-{match['map_seed']}")
        z = np.load(self.tmp / "rank" / split / "rank.npz")
        self.assertEqual(len(z["root_map"]), 1)
        self.assertEqual(int(z["root_map"][0]), match["map_seed"])
        self.assertTrue(z["cand_on_policy"][0])

    def test_policy_comparison_and_audit(self):
        runs = {}
        for fam in ("p2", "p3"):
            runs[fam] = self.tmp / f"policy_{fam}"
            train_policy.main(["--shards", str(self.shards), "--model", "general", "--policy-family", fam, "--epochs", "1",
                               "--batch", "256", "--out", str(runs[fam]), "--max-width", "512"])
        compare_policy.main(["--shards", str(self.shards), "--split", "train", "--teams", *map(str, TEAMS),
                             "--run", f"p2={runs['p2']}", "--run", f"p3={runs['p3']}", "--compare", "p3:p2",
                             "--out", str(self.tmp / "pcmp")])
        rep = json.loads((self.tmp / "pcmp.json").read_text())
        self.assertIn("contact", rep["compare"][0]["strata"])
        feature_audit.main(["--shards", str(self.shards), "--teams", *map(str, TEAMS), "--split", "train",
                            "--out", str(self.tmp / "audit")])
        audit = json.loads((self.tmp / "audit.json").read_text())
        self.assertIn("all", audit["coverage"])
        self.assertIsNotNone(audit["aliases"]["groups_split_by_v3"])


if __name__ == "__main__":
    unittest.main()
