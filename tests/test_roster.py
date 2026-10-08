"""Phase A roster tools: inventory, probes, fingerprints, distances, strength, catalog, pool.

Unit tests need only a C++ compiler. TestReplayEndToEnd needs AI_TEST_ROOT: it
records a few test-bot matches, freezes probes, and replays them to bots.
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion.paths import REPO, TEAM_COUNTS
from roster import archive, assess, catalog, distance, inventory, pool, registry, sampling, strength
from roster.common import TAGS, load_json

BOT = """#include <bits/stdc++.h>
int main() {{ std::string line; long n = 0;
  while (std::getline(std::cin, line)) {{ if (line.rfind("END", 0) == 0) {{ ++n; std::cout << n << " {move} - S - S -" << std::endl; }} }}
  return 0; }}
"""


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


class TestInventory(unittest.TestCase):
    def test_aliases_lineage_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "ctf-arena/Player7/workspace/engine.py", "")
            d = "ctf-stress-pack/bots/"
            write(root, d + "a.cpp", BOT.format(move="S"))
            write(root, d + "b.cpp", BOT.format(move="S"))                               # byte alias of a
            write(root, d + "c.cpp", "// a renamed copy\n" + BOT.format(move="S"))       # same code, other bytes
            write(root, d + "d.cpp", "#include <bits/stdc++.h>\nint main() { int x; while (std::cin >> x) "
                                     "std::cout << x * 3 + 1 << '\\n'; for (int i = 0; i < 3; i++) std::puts(\"z\"); }\n")
            write(root, d + "lib.cpp", "int helper(int x) { return x + 1; }\n")
            write(root, d + "tests/probe_driver.cpp", BOT.format(move="U"))
            write(root, d + "bad.cpp", "int main() { this is not C++; }\n")
            write(root, d + "inc.cpp", '#include "inc_helper.hpp"\n' + BOT.format(move="D"))
            write(root, d + "inc_helper.hpp", "inline int tag() { return 1; }\n")
            out = root / "inv.json"
            inv = inventory.main(["--ai-test", str(root), "--out", str(out), "--jobs", "2"])
            by = {b["id"].removeprefix(d): b for b in inv["bots"]}
            self.assertEqual(by["b.cpp"]["byte_alias_of"], d + "a.cpp")
            self.assertIn("byte-identical", by["b.cpp"]["excluded"])
            self.assertEqual(by["b.cpp"]["build"]["inherited_from"], d + "a.cpp")
            self.assertIsNone(by["c.cpp"]["byte_alias_of"])
            self.assertEqual(by["c.cpp"]["code_alias_of"], d + "a.cpp")  # comments do not change the compiled policy
            self.assertIn("comments and formatting", by["c.cpp"]["excluded"])
            self.assertEqual(by["c.cpp"]["representative"], d + "a.cpp")
            self.assertEqual(by["c.cpp"]["lineage_group"], by["a.cpp"]["lineage_group"])
            self.assertEqual(by["c.cpp"]["nearest_relative"]["similarity"], 1.0)
            self.assertNotEqual(by["d.cpp"]["lineage_group"], by["a.cpp"]["lineage_group"])
            self.assertIn("no main()", by["lib.cpp"]["excluded"])
            self.assertIn("driver", by["tests/probe_driver.cpp"]["excluded"])
            self.assertIn("build failed", by["bad.cpp"]["excluded"])
            self.assertEqual(by["inc.cpp"]["local_includes"], [d + "inc_helper.hpp"])
            self.assertNotEqual(by["inc.cpp"]["dependency_sha256"], by["inc.cpp"]["source_sha256"])
            self.assertEqual(by["inc.cpp"]["build"]["status"], "ok")
            self.assertTrue(by["a.cpp"]["build"]["binary_sha256"])
            self.assertIsNone(by["a.cpp"]["excluded"])


def synthetic_run(bot, probe, rep, move, act=None, target=None, valid=None):
    T = len(move)
    return {"bot": {"id": bot}, "probe_id": probe, "rep": rep, "valid": valid or [1] * T, "move": move,
            "act": act or [[0, 0, 0]] * T, "target": target or [[-1, -1, -1]] * T}


PROBES = ("p3", "p3b", "p5", "p5b")


class TestDistance(unittest.TestCase):
    def suite(self, n_quiet=15, n_contact=5):
        """Two probes (maps) at each of 3 and 5 teams."""
        bit = {t: 1 << i for i, t in enumerate(TAGS)}
        tags = [bit["quiet"]] * n_quiet + [bit["contact"]] * n_contact
        return {"probes": [{"probe_id": p, "teams": int(p[1]), "tags": tags, "active": [7] * len(tags)} for p in PROBES]}

    def test_rare_events_weigh_like_quiet_ones_and_clusters_respect_noise(self):
        suite = self.suite()
        T = 20
        base = [[0, 0, 0]] * T
        on_contact = [[0, 0, 0]] * 15 + [[1, 1, 1]] * 5        # differs only on the 5 contact turns
        runs = {}
        for p in PROBES:
            runs.setdefault("A", []).extend([synthetic_run("A", p, 0, base), synthetic_run("A", p, 1, base)])
            runs.setdefault("Acopy", []).extend([synthetic_run("Acopy", p, 0, base), synthetic_run("Acopy", p, 1, base)])
            runs.setdefault("B", []).extend([synthetic_run("B", p, 0, on_contact), synthetic_run("B", p, 1, on_contact)])
        out = distance.compute(suite, runs, min_tag_turns=5)
        i = {b: k for k, b in enumerate(out["ids"])}
        # contact tag: move rate 1, action rate 0 -> (.5 * 1 + .3 * 0) / .8; quiet tag 0; probe = mean of the two tags
        self.assertAlmostEqual(out["overall"][i["A"]][i["B"]], 0.625 / 2, places=5)
        self.assertEqual(out["overall"][i["A"]][i["Acopy"]], 0.0)
        self.assertEqual(out["noise"]["A"], 0.0)
        self.assertIn(["A", "Acopy"], out["clusters"])
        self.assertIn(["B"], out["clusters"])
        self.assertEqual(out["distinctive_tags"]["B"][0]["tag"], "contact")
        # 95 quiet turns would not dilute it: the tags weigh equally
        long = self.suite(n_quiet=95)
        runs_long = {b: [synthetic_run(b, p, r, ([[0, 0, 0]] * 95 + [[1, 1, 1]] * 5) if b == "B" else [[0, 0, 0]] * 100)
                         for p in PROBES for r in (0, 1)] for b in ("A", "B")}
        self.assertAlmostEqual(distance.compute(long, runs_long)["overall"][0][1], 0.3125, places=5)

    def test_targets(self):
        suite = self.suite(n_quiet=10, n_contact=10)
        T = 20
        attack = [[3, 3, 3]] * T
        tgt_a = [[5, 6, 7]] * T
        tgt_b = [[5, 6, 8]] * T                                  # one unit of three picks another target
        runs = {"A": [synthetic_run("A", p, r, [[0, 0, 0]] * T, attack, tgt_a) for p in PROBES for r in (0, 1)],
                "B": [synthetic_run("B", p, r, [[0, 0, 0]] * T, attack, tgt_b) for p in PROBES for r in (0, 1)]}
        out = distance.compute(suite, runs)
        self.assertAlmostEqual(out["components"]["target"][0][1], 1 / 3, places=5)
        self.assertAlmostEqual(out["overall"][0][1], 0.2 * (1 / 3) / 1.0, places=5)
        self.assertEqual(out["joint_per_count"]["3"][0][1], 1.0)  # the team's request differs on every turn

    def random_runs(self, bot, choose, T=400, reps=4, seed=0):
        """reps runs per probe; choose(rng) -> one turn's three moves."""
        rng = np.random.default_rng(seed)
        return [synthetic_run(bot, p, r, [choose(rng) for _ in range(T)]) for p in PROBES for r in range(reps)]

    def test_noise_is_not_equivalence(self):
        """Review example: deterministic left and right, and a 50/50 mixture of them. The old rule
        (distance within the larger noise plus a margin) chained all three; the excess distance does not."""
        suite = self.suite(n_quiet=200, n_contact=200)
        runs = {"L": self.random_runs("L", lambda r: [1, 1, 1]), "R": self.random_runs("R", lambda r: [2, 2, 2]),
                "Mix": self.random_runs("Mix", lambda r: list(r.integers(1, 3, 3)), seed=1),
                "Mix2": self.random_runs("Mix2", lambda r: list(r.integers(1, 3, 3)), seed=2)}
        out = distance.compute(suite, runs)
        i = {b: k for k, b in enumerate(out["ids"])}
        self.assertAlmostEqual(out["noise"]["Mix"], 0.3125, delta=0.02)          # (.5 * .5 + .3 * 0) / .8
        self.assertAlmostEqual(out["excess_per_count"]["3"][i["L"]][i["Mix"]], 0.3125 / 2, delta=0.02)
        self.assertIn(["Mix", "Mix2"], out["clusters"])                           # two draws of one random policy
        self.assertIn(["L"], out["clusters"])
        self.assertIn(["R"], out["clusters"])
        self.assertEqual(out["stochastic"], ["Mix", "Mix2"])

    def test_joint_coordination_separates_equal_marginals(self):
        """Same per-unit 50/50 marginals: one team moves together, the other independently."""
        suite = self.suite(n_quiet=200, n_contact=200)

        def together(r):
            m = int(r.integers(1, 3))
            return [m, m, m]

        runs = {"Team": self.random_runs("Team", together, seed=3),
                "Solo": self.random_runs("Solo", lambda r: list(r.integers(1, 3, 3)), seed=4)}
        out = distance.compute(suite, runs)
        self.assertLess(abs(out["excess_per_count"]["3"][0][1]), 0.02)            # per-unit: indistinguishable
        self.assertAlmostEqual(out["joint_excess_per_count"]["3"][0][1], 3 / 16, delta=0.03)
        self.assertEqual(out["clusters"], [["Solo"], ["Team"]])

    def test_complete_linkage_does_not_chain(self):
        """Review example: AB = BC = 0.02, AC = 0.04 at margin 0.02; connected components merged all three."""
        ids = ["A", "B", "C"]
        score = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], float)                  # distance / margin
        groups = distance.complete_clusters(ids, score)
        self.assertEqual(len(groups), 2)
        self.assertNotIn(["A", "B", "C"], groups)
        self.assertEqual(distance.complete_clusters(ids, np.where(score > 1, np.nan, score)), [["A", "B"], ["C"]])

    def test_a_duel_specialist_is_not_averaged_away(self):
        """Different only at 3 teams by .2: the equal-count mean (.1 here, .014 over 14 counts) hid it."""
        suite = self.suite(n_quiet=10, n_contact=10)
        T = 20
        duel = [[0, 0, 0]] * 12 + [[1, 1, 1]] * 8                 # move differs on 8 of 20 turns
        runs = {"Ref": [synthetic_run("Ref", p, r, [[0, 0, 0]] * T) for p in PROBES for r in (0, 1)],
                "Duel": [synthetic_run("Duel", p, r, duel if p.startswith("p3") else [[0, 0, 0]] * T) for p in PROBES for r in (0, 1)]}
        out = distance.compute(suite, runs)
        self.assertEqual(out["clusters_per_count"]["5"], [["Duel", "Ref"]])
        self.assertEqual(out["clusters_per_count"]["3"], [["Duel"], ["Ref"]])
        self.assertEqual(out["clusters"], [["Duel"], ["Ref"]])
        self.assertEqual(out["distinct_counts"]["Duel"], [3])

    def test_one_probe_is_not_enough_for_equivalence(self):
        suite = {"probes": self.suite()["probes"][:1]}
        runs = {b: [synthetic_run(b, "p3", r, [[0, 0, 0]] * 20) for r in (0, 1)] for b in ("A", "Acopy")}
        self.assertEqual(distance.compute(suite, runs)["clusters"], [["A"], ["Acopy"]])
        self.assertEqual(distance.compute(suite, runs, min_probes=1)["clusters"], [["A", "Acopy"]])

    def test_missing_replies_are_not_agreement(self):
        suite = self.suite(n_quiet=10, n_contact=10)
        T = 20
        silent = [0] * T
        runs = {"A": [synthetic_run("A", p, 0, [[0, 0, 0]] * T) for p in PROBES],
                "S": [synthetic_run("S", p, 0, [[0, 0, 0]] * T, valid=silent) for p in PROBES]}
        out = distance.compute(suite, runs)
        self.assertIsNone(out["overall"][0][1])


def strength_rows(cand, utilities_by_count, sha="0" * 64):
    rows = []
    for n, us in utilities_by_count.items():
        for m, u in enumerate(us):
            rows.append({"candidate": cand, "source_sha256": sha, "fixture_id": f"n{n}-m{m}", "teams": n, "map_seed": m,
                         "seat": 0, "score": [], "utility": u, "faults": 0, "dead": None, "late": 0,
                         "first_ms": 5.0, "p99_ms": 1.0, "max_ms": 2.0})
    return rows


class TestStrength(unittest.TestCase):
    def test_aggregate_per_count_and_levels(self):
        rows = strength_rows("X", {n: [0.5, 0.7] * 4 for n in TEAM_COUNTS})
        rows += rows[:3]                                          # a rerun of a fixture is not a new map
        out = strength.aggregate(rows)["candidates"]["X"]
        self.assertEqual(out["maps"], 8 * 14)
        self.assertAlmostEqual(out["utility"], 0.6)
        se = np.std([0.5, 0.7] * 4, ddof=1) / np.sqrt(8)
        self.assertAlmostEqual(out["per_count"]["2"]["se"], se)
        self.assertAlmostEqual(out["se"], np.sqrt(14 * se ** 2) / 14)
        self.assertEqual(out["evidence"], "screen")
        big = strength.aggregate(strength_rows("Y", {n: [0.6] * 715 for n in TEAM_COUNTS}))["candidates"]["Y"]
        self.assertEqual((big["maps"], big["evidence"]), (10010, "confirmation"))

    def test_schedule_is_shared_and_deterministic(self):
        root = os.environ.get("AI_TEST_ROOT")
        if not root:
            self.skipTest("needs AI_TEST_ROOT for the reference bots")
        a = strength.schedule(3, [2, 7, 15], strength.REFERENCE, 1, Path(root))
        b = strength.schedule(3, [2, 7, 15], strength.REFERENCE, 1, Path(root))
        self.assertEqual(a["fixtures"], b["fixtures"])
        self.assertEqual(len(a["fixtures"]), 9)
        for fx in a["fixtures"]:
            self.assertEqual(len(fx["opponents"]), fx["teams"] - 1)
            self.assertLess(fx["seat"], fx["teams"])
        self.assertEqual(len({fx["map_seed"] for fx in a["fixtures"]}), 9)


def load_selector():
    path = REPO / "handoffs" / "diverse15_distance_tiered_policy" / "tools" / "select_roster.py"
    spec = importlib.util.spec_from_file_location("select_roster", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCatalogAndPool(unittest.TestCase):
    def test_catalog_feeds_the_selector_and_the_pool_locks_hashes(self):
        ids = [f"bots/b{i:02d}.cpp" for i in range(17)]
        rng = np.random.default_rng(0)
        d = rng.uniform(0.2, 0.9, (17, 17))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        dist = {"ids": ids, "overall": d.tolist(), "cluster_of": {b: f"behavior:{b}" for b in ids},
                "distinctive_tags": {b: [] for b in ids}, "noise": {b: 0.0 for b in ids}}
        dist["overall"][16] = [None] * 17                                  # no fingerprints on some probes
        dist["overall"][16][16] = 0.0
        for row in dist["overall"][:16]:
            row[16] = None
        sha = {b: f"{i:064x}" for i, b in enumerate(ids)}
        assessments = {"bots": [{"id": b, "excluded": None, "source_sha256": sha[b], "lineage_group": f"lineage:{i % 6}"}
                                for i, b in enumerate(ids)]}
        strength_json = {"candidates": {b: {"per_count": {str(n): {"maps": 715} for n in TEAM_COUNTS}, "lcb": 0.4 + 0.01 * i,
                                            "utility": 0.5, "evidence": "confirmation"} for i, b in enumerate(ids)}}
        review = {"bots": {b: {"eligible": True, "skills": [f"skill {i % 5}"], "unique_skill_evidence": "probe report"}
                           for i, b in enumerate(ids) if i != 3}}
        cat = catalog.build(assessments, dist, strength_json, review, "a" * 64)
        self.assertIn(ids[16], cat["left_out"])
        self.assertEqual(len(cat["bots"]), 16)
        self.assertFalse(next(b for b in cat["bots"] if b["id"] == ids[3])["eligible"])  # unreviewed
        selected = load_selector().select(cat)
        self.assertEqual(selected["status"], "selected_pending_manual_and_downstream_review")
        self.assertEqual(len(selected["selected"]), 15)
        self.assertNotIn(ids[3], [s["id"] for s in selected["selected"]])
        inv = {"bots": [{"id": b, "path": b, "source_sha256": sha[b], "pool_name": None} for b in ids]}
        p = pool.build(selected, inv)
        self.assertEqual(len(p["bots"]), 15)                               # Noise only on request
        self.assertTrue(all(b.get("source_sha256") for b in p["bots"]))
        self.assertEqual(len(pool.build(selected, inv, noise=True)["bots"]), 16)
        with self.assertRaises(SystemExit):
            pool.build(selected | {"selected": [selected["selected"][0] | {"source_sha256": "f" * 64}]}, inv)

    def test_review_template_starts_closed(self):
        rows = {"bots": [{"id": "a", "excluded": None, "evidence_status": "screened", "lineage_group": "lineage:a",
                          "distinctive_tags": "contact x1.4", "utility_lcb": 0.4},
                         {"id": "b", "excluded": "byte-identical to a", "evidence_status": "alias"}]}
        t = catalog.review_template(rows)
        self.assertEqual(list(t["bots"]), ["a"])
        self.assertFalse(t["bots"]["a"]["eligible"])
        self.assertEqual(t["bots"]["a"]["hints"]["distinctive_tags"], "contact x1.4")

    def test_collection_refuses_a_changed_bot(self):
        from training.collect import resolve_pool  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmp:
            bot = write(Path(tmp), "bots/x.cpp", BOT.format(move="S"))
            entry = {"name": "X", "path": "bots/x.cpp", "family": "x", "source_sha256": "0" * 64}
            with self.assertRaises(SystemExit):
                resolve_pool({"bots": [entry]}, Path(tmp))
            from fusion.paths import sha256  # noqa: PLC0415
            self.assertIn("X", resolve_pool({"bots": [entry | {"source_sha256": sha256(bot)}]}, Path(tmp)))


def member(name, group_by_count, lineage, roles=("style",), counts=(3, 5, 15), **kw):
    return {"name": name, "lineage": lineage, "roles": list(roles), "counts": list(counts),
            "groups": {str(n): f"n{n}:{group_by_count.get(n, name)}" for n in counts}, **kw}


class TestSampling(unittest.TestCase):
    def test_groups_share_mass_and_lineages_are_capped(self):
        ms = [member("a1", {3: "a"}, "L1"), member("a2", {3: "a"}, "L1"), member("a3", {3: "a"}, "L1"), member("b", {}, "L2")]
        rho = sampling.group_balanced(ms, 3, 1.0)
        self.assertAlmostEqual(rho["b"], 0.5)                     # one group of 3 versions, one of 1: equal group mass
        self.assertAlmostEqual(rho["a1"], 1 / 6)
        ms = [member(f"x{i}", {}, "X") for i in range(3)] + [member("y", {}, "Y")]
        self.assertAlmostEqual(sampling.group_balanced(ms, 3, 1.0)["y"], 0.25)
        capped = sampling.group_balanced(ms, 3, 0.5)              # lineage X held 3/4 of the mass
        self.assertAlmostEqual(capped["y"], 0.5)
        self.assertAlmostEqual(sum(capped[f"x{i}"] for i in range(3)), 0.5)
        self.assertAlmostEqual(sampling.group_balanced(ms, 3, 0.1)["y"], 0.5)   # infeasible cap relaxes to 1/lineages

    def test_empty_components_renormalize(self):
        ms = [member("f", {}, "F", roles=("style", "field")), member("s", {}, "S")]
        mix = sampling.mixture(ms, 3)
        self.assertEqual(set(mix["lambdas"]), {"field", "style"})
        self.assertAlmostEqual(mix["lambdas"]["field"], 0.6 / 0.9)
        self.assertAlmostEqual(sum(mix["rho"].values()), 1.0)
        hard = [member("h", {}, "H", roles=("style", "hard"), hard_counts=[5])] + ms
        self.assertNotIn("hard", sampling.mixture(hard, 3)["lambdas"])      # verified only at 5 teams
        self.assertIn("hard", sampling.mixture(hard, 5)["lambdas"])

    def test_composition_and_inclusion(self):
        fifteen = [member(f"b{i:02d}", {}, f"L{i % 5}", counts=(15,)) for i in range(15)]
        rep = sampling.inclusion(sampling.mixture(fifteen, 15), games=200)
        self.assertTrue(rep["fixed_composition"])                 # 15 bots, 15 seats, no repeats: always everyone
        self.assertEqual(set(rep["in_game"].values()), {1.0})
        wide = fifteen + [member(f"c{i:02d}", {}, f"L{i % 5}", counts=(15,)) for i in range(10)]
        rep = sampling.inclusion(sampling.mixture(wide, 15), games=2000)
        self.assertFalse(rep["fixed_composition"])
        self.assertAlmostEqual(sum(rep["in_game"].values()), 15.0, places=6)
        self.assertEqual(rep["forced_repeat_seats_per_game"], 0.0)
        g = sampling.draw(__import__("random").Random(1), sampling.mixture(fifteen[:3], 15))
        self.assertEqual(len(g["seats"]), 15)                     # too few policies: repeats, logged
        self.assertEqual(g["components"].count("repeat"), 12)

    def test_augmentation_and_focus_regimes(self):
        ms = [member(f"p{i}", {}, f"L{i}", counts=(5,)) for i in range(8)] + [member("Noise", {}, "N", roles=("augmentation",), counts=(5,))]
        mix = sampling.mixture(ms, 5, {"augmentation": 1.0})
        self.assertNotIn("Noise", mix["rho"])
        rep = sampling.inclusion(mix, games=500)
        self.assertEqual(rep["in_game"]["Noise"], 1.0)
        self.assertAlmostEqual(rep["component_seat_share"]["augmentation"], 0.2)
        rep = sampling.inclusion(sampling.mixture(ms, 5, {"focus": 1.0, "focus_fraction": 0.6}), games=500)
        self.assertEqual(rep["regimes"], {"focus": 1.0})
        self.assertAlmostEqual(rep["component_seat_share"]["focus"], 0.6)
        with self.assertRaises(ValueError):
            sampling.settings({"lambdas": {"field": 0, "style": 0, "hard": 0}})


class TestRegistry(unittest.TestCase):
    ids = ["A", "B", "C", "D", "Dalias", "E", "Flaky", "Noise", "Ood"]

    def fixtures(self):
        n = len(self.ids)
        eq = {}
        for teams in (3, 5):
            m = np.full((n, n), 5.0)
            np.fill_diagonal(m, 0.0)
            i = {b: k for k, b in enumerate(self.ids)}
            m[i["A"], i["B"]] = m[i["B"], i["A"]] = 0.5 if teams == 3 else 3.0    # A, B equivalent at 3 teams only
            m[i["A"], i["C"]] = m[i["C"], i["A"]] = 0.5                            # C equivalent to A, but not to B at 5
            eq[str(teams)] = m.tolist()
        clusters = {str(t): [[b] for b in self.ids if not (t == 5 and b == "E")] for t in (3, 5)}   # E unprobed at 5
        dist = {"ids": self.ids, "params": {"margin": 0.02}, "equivalence_score_per_count": eq, "clusters_per_count": clusters,
                "noise": {b: 0.0 for b in self.ids}, "stochastic": ["Noise"], "distinct_counts": {b: [] for b in self.ids},
                "nearest_per_count": {b: {"3": {"id": "A", "excess": 0.1}} for b in self.ids}}
        rows = []
        for b in self.ids:
            rows.append({"id": b, "excluded": None, "evidence_status": "fingerprinted", "lineage_group": "lineage:ood" if b == "Ood" else f"lineage:{b[0]}",
                         "source_sha256": b.ljust(64, "0"), "valid_reply_share": 0.9 if b == "Flaky" else 1.0, "probe_dead_runs": 0,
                         "inherited_from": "D" if b == "Dalias" else None})
        rows[self.ids.index("Dalias")]["evidence_status"] = "alias"
        rows[0] |= {"strength_evidence": "confirmation", "utility": 0.6, "utility_lcb": 0.58, "maps": 10010}
        inv = {"bots": [{"id": b, "path": f"bots/{b}.cpp", "source_sha256": b.ljust(64, "0"), "pool_name": None, "excluded": None,
                         "lineage_group": r["lineage_group"]} for b, r in zip(self.ids, rows)]}
        spec = {"admit": True, "contribution": "probe-measured difference", "counts": [3, 5]}
        admission = {"revision": "t1", "reserved_ood_lineages": ["lineage:ood"],
                     "bots": {b: dict(spec) for b in self.ids} | {"A": spec | {"roles": ["style", "field"]},
                                                                   "B": spec | {"roles": ["style", "hard"], "hard_counts": [5]},
                                                                   "C": spec | {"contribution": " "},
                                                                   "Noise": spec | {"roles": ["augmentation"]}}}
        return admission, {"bots": rows}, dist, inv

    def test_admission_gates_and_groups(self):
        admission, rows, dist, inv = self.fixtures()
        reg = registry.build(admission, rows, dist, inv, games=500)
        names = {m["id"] for m in reg["bots"]}
        self.assertEqual(names, {"A", "B", "D", "E", "Noise"})       # no size cap; every gate below is a reason
        self.assertIn("alias", reg["not_admitted"]["Dalias"])
        self.assertIn("reserved", reg["not_admitted"]["Ood"])
        self.assertIn("contribution", reg["not_admitted"]["C"])
        self.assertIn("reliability", reg["not_admitted"]["Flaky"])
        self.assertEqual(reg["counts_without_behavior_evidence"], {"E": [5]})
        by = {m["id"]: m for m in reg["bots"]}
        self.assertEqual(by["A"]["evidence_level"], "confirmed")
        self.assertEqual(by["B"]["evidence_level"], "provisional")
        self.assertEqual(by["A"]["groups"]["3"], by["B"]["groups"]["3"])   # equivalent at 3 teams: they share mass
        self.assertNotEqual(by["A"]["groups"]["5"], by["B"]["groups"]["5"])
        self.assertTrue(by["Noise"]["family"].startswith("augmentation:"))
        self.assertFalse(by["Noise"]["policy_labels"])
        self.assertNotIn("groups", by["Noise"])
        c3 = reg["per_count"]["3"]
        self.assertEqual((c3["members"], c3["groups"]), (4, 3))
        # style: groups {A, B}, {D}, {E} at 1/3 each; field: A alone; weights .6 and .3 renormalized (no hard at 3)
        self.assertAlmostEqual(c3["rho"][by["D"]["name"]], (1 / 3) * (1 / 3), places=4)
        self.assertAlmostEqual(c3["rho"][by["B"]["name"]], (1 / 3) * (1 / 6), places=4)
        self.assertAlmostEqual(c3["rho"][by["A"]["name"]], 2 / 3 + (1 / 3) * (1 / 6), places=4)
        self.assertNotIn("hard", c3["lambdas"])                     # B is a verified hard case only at 5 teams
        self.assertIn("hard", reg["per_count"]["5"]["lambdas"])
        self.assertTrue(any(w.startswith("5 teams: fewer policy copies than seats") for w in reg["warnings"]))
        again = registry.build(admission, rows, dist, inv, games=500)
        self.assertEqual(reg["identity"], again["identity"])
        changed = registry.build(admission | {"sampling": {"lambdas": {"field": 0.5, "style": 0.5, "hard": 0}}}, rows, dist, inv, games=500)
        self.assertNotEqual(reg["identity"], changed["identity"])

    def test_template_admits_nothing(self):
        admission, rows, dist, _ = self.fixtures()
        t = registry.template(rows, dist)
        self.assertNotIn("Dalias", t["bots"])
        self.assertFalse(any(b["admit"] for b in t["bots"].values()))
        self.assertEqual(t["bots"]["B"]["hints"]["counts_beyond_margin_of_nearest"], [3])

    def test_collection_uses_the_sampler_and_freeze_eval(self):
        from fusion.paths import is_unlabeled  # noqa: PLC0415
        from training import collect  # noqa: PLC0415
        admission, rows, dist, inv = self.fixtures()
        reg = registry.build(admission, rows, dist, inv, games=200)
        pool_entries = {m["name"]: m for m in reg["bots"]}
        fx = collect.schedule(3, 40, 0, 7, {"collect": {"pool": "default"}}, pool_entries, registry=reg)
        self.assertEqual(fx, collect.schedule(3, 40, 0, 7, {"collect": {"pool": "default"}}, pool_entries, registry=reg))
        self.assertTrue(all(f.registry == reg["identity"] and len(f.components) == 3 for f in fx))
        self.assertTrue(all(len(set(f.seats)) == 3 for f in fx))  # four policies at 3 teams: no repeats
        self.assertTrue(is_unlabeled(pool_entries["Noise"]["family"], set()))
        ev = registry.freeze_eval(inv, ["A", "Ood"], reg, "core", 1, 500, [3, 5])
        self.assertEqual([(b["in_training"], b["ood_lineage"]) for b in ev["bots"]], [(True, False), (False, True)])
        self.assertEqual(ev["training_registry"], reg["identity"])


class TestArchive(unittest.TestCase):
    def test_save_check_and_tamper(self):
        from roster.common import sha256_file  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "ai", Path(tmp) / "archive"
            a = write(root, "bots/a.cpp", "int main(){}\r\n")         # CRLF kept byte for byte
            write(root, "bots/copy.cpp", "int main(){}\r\n")
            entry = lambda rel, p: {"id": rel, "path": rel, "language": "cpp", "source_sha256": sha256_file(p), "local_includes": []}  # noqa: E731
            inv = {"ai_test_commit": "c0ffee", "bots": [entry("bots/a.cpp", a), entry("bots/copy.cpp", root / "bots/copy.cpp"),
                                                         {"id": "@repo/tools/noise_bot.py", "path": "@repo/tools/noise_bot.py",
                                                          "language": "py", "source_sha256": "f" * 64}]}
            r = archive.save(inv, root, out)
            self.assertEqual((r["candidates"], r["stored_files"]), (3, 1))   # one content, stored once
            stored = out / archive.stored_name(sha256_file(a), "a.cpp")
            self.assertEqual(stored.read_bytes(), a.read_bytes())
            pool_file = Path(tmp) / "pool.json"
            pool_file.write_text(json.dumps({"bots": [{"name": "A", "source_sha256": sha256_file(a)},
                                                      {"name": "Gone", "source_sha256": "0" * 64}, {"name": "Legacy"}]}))
            res = archive.check(out, [str(pool_file)], root)
            self.assertEqual(len(res["errors"]), 1)                        # Gone is not archived
            self.assertEqual(len(res["untraced"]), 1)                      # Legacy has no hash
            a.write_text("int main(){return 1;}")
            self.assertEqual(archive.check(out, [], root)["changed_in_ai_test"], ["bots/a.cpp"])
            stored.write_text("edited")
            self.assertTrue(any("no longer matches" in e for e in archive.check(out)["errors"]))
            with self.assertRaises(SystemExit):                            # a source changed since the inventory
                archive.save(inv, root, Path(tmp) / "other")


@unittest.skipUnless(os.environ.get("AI_TEST_ROOT"), "set AI_TEST_ROOT to the AI_Test checkout")
class TestReplayEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from roster import fingerprint, probes  # noqa: PLC0415
        from training import collect  # noqa: PLC0415
        cls.tmp = Path(tempfile.mkdtemp(prefix="roster-test-"))
        raw = cls.tmp / "raw"
        collect.main(["--teams", "3", "4", "--matches", "2", "--bots", "Random", "--out", str(raw)])
        cls.suite_path = cls.tmp / "suite.json"
        probes.main(["--raw", str(raw), "--teams", "3", "4", "--per-count", "2", "--turns", "60", "--out", str(cls.suite_path)])
        copy = cls.tmp / "random_copy.py"
        shutil.copy(REPO / "tools" / "random_bot.py", copy)
        cls.fp = cls.tmp / "fp"
        fingerprint.main(["--suite", str(cls.suite_path), "--bot", "Random=@repo/tools/random_bot.py",
                          "--bot", "Noise=@repo/tools/noise_bot.py", "--bot", f"RandomCopy={copy}",
                          "--repeat", "2", "--workers", "1", "--out", str(cls.fp)])
        cls.runs = fingerprint.load_all(cls.fp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_suite_is_frozen_and_tagged(self):
        suite = load_json(self.suite_path)
        self.assertEqual(len(suite["probes"]), 4)
        for p in suite["probes"]:
            self.assertEqual(p["turns"], 60)
            self.assertEqual(len(p["tags"]), 60)
            strata = sum(p["tag_counts"][t] for t in ("quiet", "contact", "flag", "respawn"))
            self.assertEqual(strata, 60)                          # the four strata partition the turns

    def test_replay_reproduces_a_deterministic_bot(self):
        """Random played every seat of the recorded matches: replaying its history must give its recorded replies."""
        from fusion import matchlog  # noqa: PLC0415
        suite = {p["probe_id"]: p for p in load_json(self.suite_path)["probes"]}
        checked = 0
        for run in self.runs["Random"]:
            p = suite[run["probe_id"]]
            recorded = [row[p["seat"]] for row in matchlog.load(p["file"])["replies"][:p["turns"]]]
            self.assertEqual(run["lines"], recorded)
            checked += 1
        self.assertEqual(checked, 8)

    def test_distances_separate_behaviors(self):
        out = distance.compute(load_json(self.suite_path), self.runs)
        i = {b: k for k, b in enumerate(out["ids"])}
        self.assertEqual(out["overall"][i["Random"]][i["RandomCopy"]], 0.0)
        self.assertEqual(out["noise"]["Random"], 0.0)
        self.assertGreater(out["overall"][i["Random"]][i["Noise"]], 0.2)
        self.assertIn(["Random", "RandomCopy"], out["clusters"])
        self.assertEqual(set(out["clusters_per_count"]), {"3", "4"})
        inv = {"bots": [{"id": b, "pool_name": None, "excluded": None, "representative": b, "byte_alias_of": None,
                         "binary_alias_of": None, "lineage_group": "lineage:test", "nearest_relative": None,
                         "build": {"status": "interpreted"}, "source_sha256": "0" * 64} for b in ("Random", "Noise", "RandomCopy")]}
        rows = {r["id"]: r for r in assess.assess(inv, self.runs, out, None)}
        self.assertEqual(rows["Random"]["evidence_status"], "fingerprinted")
        self.assertEqual(rows["Random"]["valid_reply_share"], 1.0)
        self.assertEqual(rows["Random"]["behavior_cluster"], rows["RandomCopy"]["behavior_cluster"])
        self.assertNotEqual(rows["Random"]["behavior_cluster"], rows["Noise"]["behavior_cluster"])


if __name__ == "__main__":
    unittest.main()
