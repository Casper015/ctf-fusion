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
from roster import assess, catalog, distance, inventory, pool, strength
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


class TestDistance(unittest.TestCase):
    def suite(self, n_quiet=15, n_contact=5):
        bit = {t: 1 << i for i, t in enumerate(TAGS)}
        tags = [bit["quiet"]] * n_quiet + [bit["contact"]] * n_contact
        return {"probes": [{"probe_id": "p3", "teams": 3, "tags": tags, "active": [7] * len(tags)},
                           {"probe_id": "p5", "teams": 5, "tags": tags, "active": [7] * len(tags)}]}

    def test_rare_events_weigh_like_quiet_ones_and_clusters_respect_noise(self):
        suite = self.suite()
        T = 20
        base = [[0, 0, 0]] * T
        on_contact = [[0, 0, 0]] * 15 + [[1, 1, 1]] * 5        # differs only on the 5 contact turns
        runs = {}
        for p in ("p3", "p5"):
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
                         for p in ("p3", "p5") for r in (0, 1)] for b in ("A", "B")}
        self.assertAlmostEqual(distance.compute(long, runs_long)["overall"][0][1], 0.3125, places=5)

    def test_targets_and_noise_threshold(self):
        suite = self.suite(n_quiet=10, n_contact=10)
        T = 20
        attack = [[3, 3, 3]] * T
        tgt_a = [[5, 6, 7]] * T
        tgt_b = [[5, 6, 8]] * T                                  # one unit of three picks another target
        runs = {"A": [synthetic_run("A", p, r, [[0, 0, 0]] * T, attack, tgt_a) for p in ("p3", "p5") for r in (0, 1)],
                "B": [synthetic_run("B", p, r, [[0, 0, 0]] * T, attack, tgt_b) for p in ("p3", "p5") for r in (0, 1)]}
        out = distance.compute(suite, runs)
        self.assertAlmostEqual(out["components"]["target"][0][1], 1 / 3, places=5)
        self.assertAlmostEqual(out["overall"][0][1], 0.2 * (1 / 3) / 1.0, places=5)
        # A noisy bot whose own repeats disagree as much as it differs from A joins A's cluster
        noisy = [synthetic_run("N", p, r, [[0, 0, 0]] * T, attack, tgt_a if r == 0 else tgt_b) for p in ("p3", "p5") for r in (0, 1)]
        out = distance.compute(suite, runs | {"N": noisy})
        self.assertGreater(out["noise"]["N"], 0.0)
        self.assertTrue(any({"A", "N"} <= set(c) for c in out["clusters"]))

    def test_missing_replies_are_not_agreement(self):
        suite = self.suite(n_quiet=10, n_contact=10)
        T = 20
        silent = [0] * T
        runs = {"A": [synthetic_run("A", p, 0, [[0, 0, 0]] * T) for p in ("p3", "p5")],
                "S": [synthetic_run("S", p, 0, [[0, 0, 0]] * T, valid=silent) for p in ("p3", "p5")]}
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
        self.assertEqual(len(p["bots"]), 16)                               # 15 + Noise
        self.assertTrue(all(b.get("source_sha256") for b in p["bots"][:15]))
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
