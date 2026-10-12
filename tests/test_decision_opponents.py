"""Expanded decision opponent registry and designated-carrier collector (task T0035).

No game, compilation or engine is used: neural entries are resolved by bundling only, builds and play are injected
fakes, and outcomes are synthetic.
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fusion import matchlog
from training import decision_collect as C
from training import decision_opponents as O

UNUSED_SEED = 20269991   # not in the DIGEST ledger; only for these in-memory schedules


class Registry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = O.build_registry()
        cls.by = {m["name"]: m for m in cls.reg["members"]}

    def test_counts_and_states(self):
        c = self.reg["counts"]
        self.assertEqual((c["reference:enabled"], c["addition:enabled"]), (12, 6))
        self.assertEqual((c["neural:enabled"], c["neural:disabled"], c["neural:not_registered"]), (34, 5, 5))
        for x in "ABCDE":
            m = self.by[f"v12_ponly_pxxl_vp{x}_v"]
            self.assertEqual(m["status"], "disabled")
            self.assertIn("models_pxxl_vp" + x, m["reason"])
        mix = O.mixture(self.reg)
        self.assertFalse(any(u["name"].startswith("v12_ponly_pxxl_vp") for u in mix["units"].values()))
        for name in ("c08_control", "c08_control_2ply", "v12_control", "c08_hybrid_m", "v12_hybrid_m"):
            self.assertEqual(self.by[name]["status"], "not_registered")
        self.assertEqual(self.by["c08_ponly_s"]["model_registry"], ["inference/models/registry.hpp"])
        self.assertTrue(self.by["c08_ponly_pxxl_vm_v"]["value_on"])

    def test_sources_resolve_by_hash(self):
        for name in ("ctf-arena_Player5_v12_rollout_volley", "ctf-arena_Player5_fortress", "ctf-arena_Player5_sparring_raider"):
            r = self.by[name]["resolution"]
            self.assertEqual((r["how"], r["substituted"]), ("repository", False))
        for name in ("cand-012", "C04_G_PLAYOUT_4ply", "bot_2", "Astra", "Baseline"):
            r = self.by[name]["resolution"]
            self.assertEqual((r["how"], r["substituted"]), ("vendored", True))   # without an arena checkout: recorded
            self.assertEqual(O.sha256_file(r["path"]), self.by[name]["source_sha256"])
        bad = O.resolve_source(["ctf-arena/missing.cpp"], "0" * 64, None)
        self.assertEqual(bad["status"], "unresolved")

    def test_lineages_from_the_inventory(self):
        self.assertEqual(self.by["bot_2"]["lineage"], "lineage:Player5")
        self.assertEqual(self.by["C04_G_PLAYOUT_4ply"]["lineage"], "lineage:Control_Hybrid")
        self.assertEqual(self.by["c08_ponly_xxl"]["lineage"], "lineage:Control_Hybrid")   # C08's origin source
        self.assertEqual(self.by["v12_ponly_xxl"]["lineage"], "lineage:Player5")
        self.assertEqual(self.by["Player5"]["lineage"], "lineage:Player5")

    def test_designated_incumbent(self):
        d = self.reg["designated"]
        self.assertEqual((d["variant"], d["value"], d["status"]), ("v12_ponly_xxl", "off", "admitted"))
        self.assertTrue(d["reference"]["same_inputs"])     # the T0010 incumbent's build inputs, file by file
        self.assertTrue(self.by["v12_ponly_xxl"].get("same_policy_as_designated"))

    def test_exclusion_and_quarantine_change_the_pool(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = O.load_json(O.OPPONENTS)
            cfg["additions"]["bots"].append({"name": "AI_Arena_CTF_Player02_bot", "pool_name": "AI_Arena_CTF_Player02_bot",
                                             "source_id": "@repo/roster/new_bots/independent/AI_Arena_CTF_Player02_bot.cpp",
                                             "source_sha256": "ef8719390bcdb0ff95c757bf56c73d1138064068b02b318eb1adf6623bf268ab"})
            cfg["quarantine"] = {"Kimi": "runtime faults on the host (test)"}
            p = tmp / "opponents.json"
            p.write_text(json.dumps(cfg))
            reg = O.build_registry(opponents=p, resolve_neural=False)
            by = {m["name"]: m for m in reg["members"]}
            self.assertEqual(by["AI_Arena_CTF_Player02_bot"]["status"], "excluded")
            self.assertEqual(by["Kimi"]["status"], "quarantined")
            base = O.build_registry(resolve_neural=False)
            self.assertNotEqual(reg["pool_identity"], base["pool_identity"])
            names = {u["name"] for u in O.mixture(reg)["units"].values()}
            self.assertFalse({"AI_Arena_CTF_Player02_bot", "Kimi"} & names)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_addition_must_match_the_selection_csv(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = O.load_json(O.OPPONENTS)
            cfg["additions"]["bots"][0]["source_sha256"] = "0" * 64
            p = tmp / "opponents.json"
            p.write_text(json.dumps(cfg))
            with self.assertRaises(SystemExit):
                O.build_registry(opponents=p, resolve_neural=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Catalogs(unittest.TestCase):
    def test_merge(self):
        merged = O.merge_catalogs(O.load_json(O.VARIANTS)["catalogs"])
        names = [v["name"] for v in merged]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 44)
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "b.json").write_text(json.dumps({"variants": [{"name": "x", "role": "ponly", "source": "s.cpp"}]}))
            (tmp / "a.json").write_text(json.dumps({"include": [str(tmp / "b.json")],
                                                    "variants": [{"name": "x", "role": "ponly", "source": "s.cpp"}]}))
            self.assertEqual(len(O.merge_catalogs([str(tmp / "a.json")])), 1)       # identical: merged
            (tmp / "a.json").write_text(json.dumps({"include": [str(tmp / "b.json")],
                                                    "variants": [{"name": "x", "role": "ponly_v", "source": "s.cpp"}]}))
            with self.assertRaises(SystemExit):
                O.merge_catalogs([str(tmp / "a.json")])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def synthetic_registry(cap=None):
    def m(name, comp, group, lineage, unit, aliases=None):
        return {"name": name, "component": comp, "group": group, "lineage": lineage, "status": "enabled", "unit": unit,
                "aliases": aliases or [name], "representative": (aliases or [name])[0] == name}
    members = [m("r1", "reference", "r1", "L:a", "u1"), m("r2", "reference", "r2", "L:b", "u2"),
               m("a1", "addition", "a1", "L:p5", "u3"),
               m("c1", "neural", "C08", "L:c", "u4"), m("c2", "neural", "C08", "L:c", "u5", ["c2", "c2x"]),
               m("c2x", "neural", "C08", "L:c", "u5", ["c2", "c2x"]), m("v1", "neural", "v12", "L:p5", "u6")]
    return {"members": members, "composition": {"reference": 0.6, "addition": 0.2, "neural": 0.2}, "lineage_cap": cap}


class Mixture(unittest.TestCase):
    def test_groups_before_versions_and_alias_budget(self):
        mix = O.mixture(synthetic_registry())
        rho = mix["rho"]
        self.assertAlmostEqual(rho["u1"], 0.3)
        self.assertAlmostEqual(rho["u3"], 0.2)
        self.assertAlmostEqual(rho["u4"], 0.05)     # C08 group 0.1 over two units
        self.assertAlmostEqual(rho["u5"], 0.05)     # the alias pair shares one unit's exposure
        self.assertAlmostEqual(rho["u6"], 0.1)      # v12 group 0.1 with one unit
        self.assertNotIn("c2x", {u["name"] for u in mix["units"].values()})

    def test_lineage_cap_reported(self):
        mix = O.mixture(synthetic_registry(cap=0.25))
        self.assertTrue(mix["cap"]["binding"])                    # L:a holds 0.3, L:p5 0.3
        self.assertLessEqual(max(mix["cap"]["post_cap"].values()), 0.25 + 1e-9)
        self.assertNotAlmostEqual(mix["composition_after_cap"]["reference"], 0.6)
        tight = O.mixture(synthetic_registry(cap=0.1))           # 4 lineages: 0.1 is infeasible
        self.assertTrue(tight["cap"]["relaxed"])
        self.assertAlmostEqual(tight["cap"]["effective_cap"], 0.25)


class Schedule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = O.build_registry()

    def test_deterministic_fixtures_and_designated_seat(self):
        a = O.make_schedule(self.reg, UNUSED_SEED, 3, [2, 5, 15])
        b = O.make_schedule(self.reg, UNUSED_SEED, 3, [2, 5, 15])
        self.assertEqual(a, b)
        self.assertNotEqual(a["fixtures"], O.make_schedule(self.reg, UNUSED_SEED + 1, 3, [2, 5, 15])["fixtures"])
        enabled = {u["name"] for u in O.mixture(self.reg)["units"].values()}
        for fx in a["fixtures"]:
            seats, comps = C.fixture_seats(fx)
            self.assertEqual(seats.count(C.CANDIDATE), 1)
            self.assertEqual(seats[fx["seat"]], C.CANDIDATE)
            self.assertEqual(comps[fx["seat"]], "designated")
            self.assertEqual(len(seats), fx["teams"])
            self.assertTrue(set(fx["opponents"]) <= enabled)
            self.assertEqual(len(set(fx["units"])), fx["teams"] - 1)

    def test_consumed_seed_refused(self):
        self.assertIn(20261201, O.consumed_seeds())
        self.assertIn(20261340, O.consumed_seeds())     # inside the showcase range
        with self.assertRaises(SystemExit):
            O.make_schedule(self.reg, 20261201, 1, [2])

    def test_simulated_shares(self):
        sim = O.simulate(self.reg, games=300, teams=[3, 15])
        for n in ("3", "15"):
            shares = sim["per_count"][n]["component_share"]
            self.assertAlmostEqual(sum(shares.values()), 1.0, places=3)
        self.assertFalse(sim["cap"]["relaxed"])

    def test_bad_fixture_refused(self):
        fx = O.make_schedule(self.reg, UNUSED_SEED, 1, [4])["fixtures"][0]
        dup = dict(fx, opponents=[fx["opponents"][0]] * 3)
        with self.assertRaises(SystemExit):
            C.fixture_seats(dup)


def fake_match(job):
    fixture, commands, info, root, limits = job
    n = fixture.teams
    score = [(i * 7 + fixture.map_seed) % 5 for i in range(n)]
    return {"schema": matchlog.SCHEMA, "match_id": fixture.match_id, "teams": n, "map_seed": fixture.map_seed,
            "flag_seed": fixture.flag_seed, "seats": [{"name": s, "family": info[s]["family"], "sha256": info[s]["sha256"]}
                                                       for s in fixture.seats],
            "engine_sha256": info["__engine__"], "ai_test_commit": info["__commit__"],
            "limits_ms": [1000 * limits[0], 1000 * limits[1]], "init": "", "blocks": [], "final_block": "", "replies": [],
            "faults": [], "result": {"score": score, "ranks": [], "stats": []},
            "bots": [{"name": s, "faults": 0, "late": 0, "dead": None} for s in fixture.seats],
            "sampling": {"regime": fixture.regime, "components": fixture.components, "registry": fixture.registry}}


class FreezeAndResume(unittest.TestCase):
    ENGINE = {"sha256": "e" * 64, "commit": "test"}
    COMPILER = {"path": "c++", "version": "test", "flags": ["-O2"]}

    @classmethod
    def setUpClass(cls):
        cls.reg = O.build_registry()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        sched = O.make_schedule(self.reg, UNUSED_SEED, 1, [2, 4])
        self.sched = self.tmp / "SCHEDULE.json"
        self.sched.write_text(json.dumps(sched))

        def variant(name, out, stats):
            p = Path(out) / f"{name}.bin"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"binary {name} {stats}")
            return {"binary": str(p), "binary_sha256": O.sha256_file(p), "bundle_sha256": "b", "inputs": [], "stats": stats}

        def source(path, out):
            p = Path(out) / (Path(path).stem + ".bin")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"binary {path}")
            return {"binary": str(p), "binary_sha256": O.sha256_file(p)}
        self.builders = (variant, source)
        self.frozen = C.prepare(self.sched, self.tmp / "run", None, self.builders, self.ENGINE, self.COMPILER, self.reg)
        self.fpath = self.tmp / "run" / "FROZEN.json"
        self.out = self.tmp / "out"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_collect(self, play=fake_match, **kw):
        return C.collect(self.fpath, self.out, None, play_fn=play, engine=self.ENGINE, registry=self.reg, **kw)

    def test_collect_and_resume(self):
        r = self.run_collect()
        self.assertEqual((r["played"], r["errors"]), (2, 0))
        rows = sorted((self.out / "rows").glob("*.json"))
        self.assertEqual(len(rows), 2)
        row = json.loads(rows[0].read_text())
        self.assertEqual(row["identity"], self.frozen["identity"])
        self.assertEqual(row["seats"].count(C.CANDIDATE), 1)
        m = matchlog.load(C.match_path(self.out, {"teams": row["teams"], "fixture_id": row["fixture_id"]}))
        self.assertEqual(m["seats"][row["seat"]]["family"], "candidate")
        self.assertEqual(m["limits_ms"], [2000, 50])
        self.assertEqual(self.run_collect()["todo"], [])          # resume: nothing left

    def test_stale_identities_refused(self):
        self.run_collect()
        bin_path = Path(self.frozen["designated"]["binary"])
        bin_path.write_text("rebuilt")
        with self.assertRaises(SystemExit):
            self.run_collect()
        frozen = json.loads(self.fpath.read_text())
        frozen["limits_ms"] = [2000, 60]
        self.fpath.write_text(json.dumps(frozen))
        self.assertIn("FROZEN.json was edited", C.verify(frozen, None, self.ENGINE, self.reg)[0])
        self.assertTrue(C.verify(self.frozen, None, {"sha256": "x", "commit": "y"}, self.reg))

    def test_rows_of_another_identity_refused(self):
        self.run_collect()
        row_p = sorted((self.out / "rows").glob("*.json"))[0]
        row = json.loads(row_p.read_text())
        row["identity"] = "0" * 64
        row_p.write_text(json.dumps(row))
        with self.assertRaises(SystemExit):
            self.run_collect()

    def test_partial_and_errors_not_hidden(self):
        calls = {"n": 0}

        def flaky(job):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("bot crashed (test)")
            return fake_match(job)
        r = self.run_collect(flaky)
        self.assertEqual((r["played"], r["errors"]), (1, 1))
        self.assertEqual(len(list((self.out / "errors").glob("*.json"))), 1)
        row_p = sorted((self.out / "rows").glob("*.json"))[0]
        row_p.unlink()                                             # a match without its row is not complete
        r = self.run_collect()
        self.assertEqual(r["played"], 2)
        self.assertEqual(len(list((self.out / "partial").glob("*.json.gz"))), 1)

    def test_prepare_refuses_while_collecting(self):
        C.lock_path(self.tmp / "run").write_text("{}")
        with self.assertRaises(SystemExit):
            C.prepare(self.sched, self.tmp / "run", None, self.builders, self.ENGINE, self.COMPILER, self.reg)

    def test_schedule_from_another_pool_refused(self):
        sched = json.loads(self.sched.read_text())
        sched["pool_identity"] = "0" * 64
        other = self.tmp / "other.json"
        other.write_text(json.dumps(sched))
        with self.assertRaises(SystemExit):
            C.prepare(other, self.tmp / "run2", None, self.builders, self.ENGINE, self.COMPILER, self.reg)

    def test_frozen_records_every_identity(self):
        f = self.frozen
        for key in ("registry", "pool_identity", "schedule", "designated", "opponents", "compiler", "engine", "runner",
                    "limits_ms"):
            self.assertIn(key, f)
        self.assertEqual(f["limits_ms"], [2000, 50])
        self.assertEqual(f["designated"]["variant"], "v12_ponly_xxl")
        opp = next(iter(f["opponents"].values()))
        self.assertIn(opp["component"], O.COMPONENTS)
        self.assertTrue(copy.deepcopy(f["runner"]))


if __name__ == "__main__":
    unittest.main()
