"""bots/p_only/tools/branch_portfolio.py and branch_regret.py against the official engine (needs AI_TEST_ROOT).

Short matches with a seeded test bot standing in for two screen arms: only the declared arm's matches are used,
another arm's are skipped and counted, a changed carrier binary stops the run; candidates are distinct complete
orders with their sources, the recorded order first, each decoded by the engine; another arm's reply at the same
root joins them; the root is rebuilt exactly and never changed; the summary and the regret report add up.
"""
import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from fusion import matchlog, runner  # noqa: E402
from fusion.paths import ai_test_root, git_commit, load_engine, load_json, sha256  # noqa: E402
from fusion.protocol import parse_reply  # noqa: E402

TEAMS = 3


@unittest.skipUnless(os.environ.get("AI_TEST_ROOT"), "set AI_TEST_ROOT to the AI_Test checkout")
class BranchPortfolio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import branch_portfolio as bp  # noqa: PLC0415
        from training import collect  # noqa: PLC0415
        cls.bp = bp
        cls.tmp = Path(tempfile.mkdtemp(prefix="ponly-branches-"))
        cls.root = ai_test_root()
        cls.engine = load_engine(cls.root)
        pool = collect.resolve_pool(load_json(REPO / "profiles" / "pool.json"), cls.root, {"Random"})
        bot = Path(pool["Random"]["abs"])
        cls.cmd = runner.command_for(bot, REPO / "build" / "bots")
        src = sha256(bot)
        cls.manifest = {a: {"binary": str(bot.relative_to(REPO)), "binary_sha256": src} for a in ("armA", "armB")}
        info = {"Random": {"family": "random", "sha256": src}, "@candidate": {"family": "candidate", "sha256": src},
                "__engine__": sha256(Path(cls.engine.__file__)), "__commit__": git_commit(cls.root)}
        cls.limits = (2.0, 0.5)
        for arm in ("armA", "armB"):
            for i in range(2):
                seats = ["Random"] * TEAMS
                seats[i % TEAMS] = "@candidate"
                fx = collect.Fixture(f"n03-s91-{i:06d}-{bp.arm_key(arm)}", TEAMS, collect.derive(91, TEAMS, i, "map"),
                                     collect.derive(91, TEAMS, i, "flag", bits=63), seats)
                match = collect.play((fx, {"Random": cls.cmd, "@candidate": cls.cmd}, info, str(cls.root), cls.limits))
                matchlog.save(cls.tmp / "matches" / "n03" / f"{fx.match_id}.json.gz", match)
        cls.files = sorted(matchlog.iter_matches(cls.tmp / "matches"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_arm_selection(self):
        selected, counts = self.bp.select(self.files, self.manifest, "armA", ["armB"])
        self.assertEqual(counts["selected"], 2)
        self.assertEqual(counts["skipped_other_arms"], 2)
        self.assertTrue(all(m["match_id"].endswith(self.bp.arm_key("armA")) for _, m in selected))
        changed = {k: dict(v) for k, v in self.manifest.items()}
        changed["armA"]["binary_sha256"] = "0" * 64
        with self.assertRaises(SystemExit) as ctx:
            self.bp.select(self.files, changed, "armA", ["armB"])
        self.assertIn("rebuilt", str(ctx.exception))

    def test_candidates_branches_and_reports(self):
        selected, counts = self.bp.select(self.files, self.manifest, "armA", ["armB"])
        path, match = selected[0]
        before = Path(path).read_bytes()
        turn = self.bp.root_turns(match, 1)[0]
        seat = self.bp.designated_seat(match)
        row = self.bp._root_job((str(path), turn, 6, 1, [self.cmd] * TEAMS, self.cmd, self.limits, str(self.root), "armA", "armB"))
        self.assertEqual(Path(path).read_bytes(), before, "branch play changed the root match")
        self.assertNotIn("dropped", row)
        orders = row["candidates"]
        self.assertEqual(len(orders), len(set(orders)))
        self.assertEqual(row["sources"][0], "recorded")
        self.assertEqual(orders[0], " ".join(match["replies"][turn - 1][seat].partition("#")[0].split()))
        self.assertTrue(set(row["sources"]) <= {"recorded", "arm:armB", *self.bp.SOURCES})
        self.assertGreaterEqual(len(set(row["sources"])), 3, "the budget should reach several kinds of alternative")
        game = self.bp.reproduce(self.engine, match, turn)
        for order in orders:
            self.assertIsNotNone(parse_reply(order, turn))
            game.decode(seat, order)  # raises if the engine would reject the order
        self.assertEqual(len(row["branches"]), len(orders))
        for b in row["branches"]:
            self.assertEqual(b["root_block"], match["blocks"][turn - 1])
            self.assertTrue(0.0 <= b["outcome"] <= 1.0)
        out = self.tmp / "branches" / "n03"
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / "root.json.gz", "wt") as fh:
            json.dump(row, fh)
        summary = self.bp.summarize([row], counts)
        self.assertEqual(summary["roots"], 1)
        self.assertEqual(sum(summary["candidates_by_source"].values()), len(orders))
        k = len(orders)
        self.assertEqual(summary["pairs"]["resolved"] + summary["pairs"]["unresolved"], k * (k - 1) // 2)
        import branch_regret  # noqa: PLC0415
        if row["compare_reply"] is not None:
            branch_regret.main(["--branches", str(self.tmp / "branches"), "--arm", "armA", "--compare-arm", "armB",
                                "--out", str(self.tmp / "regret")])
            rep = json.loads((self.tmp / "regret.json").read_text())
            self.assertEqual(rep["roots"], 1)


class SearchDiag(BranchPortfolio):
    """bots/p_only/tools/search_diag.py on the recorded test games: the root's predictions are identical in every
    configuration, and the diagnostic mode replies exactly as the current search."""

    def test_fixed_work_diagnostics(self):
        import search_diag  # noqa: PLC0415
        out = self.tmp / "diag" / "report"
        search_diag.main(["--matches", str(self.tmp / "matches"), "--arm", "armA", "--games", "1", "--fixed-samples", "8",
                          "--configs", "c08_A", "c08_diag", "c08_D", "v12_A", "v12_B", "--out", str(out)])
        rep = json.loads(out.with_suffix(".json").read_text())
        for name, c in rep["configs"].items():
            self.assertEqual(c["root_digest_mismatches"], 0, name)
            self.assertEqual(c["truncated"], 0, name)
        self.assertEqual(rep["configs"]["c08_diag"]["changed_share"], 0.0)
        self.assertGreater(rep["configs"]["c08_D"]["counters"]["ply2.predictions"], rep["configs"]["c08_A"]["counters"]["ply2.predictions"])

    test_arm_selection = None
    test_candidates_branches_and_reports = None


if __name__ == "__main__":
    unittest.main()
