"""Comparison-reply validity in bots/p_only/tools/branch_portfolio.py and branch_regret.py (task T0020).

A comparison arm's reply counts as the incumbent's order only when its canonical complete order equals the
incumbent's; a malformed reply, a timeout, a reply without a candidate or a candidate without branches is an
exclusion, reported with the valid-root denominator, never a zero difference. Synthetic root records, no engine.
"""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

import branch_portfolio as bp  # noqa: E402
import branch_regret  # noqa: E402

TURN = 12
INCUMBENT = "12 S P U 3 D -"
OTHER = "12 L - U 3 D -"
THIRD = "12 S P R 3 D -"


def root(reply, candidates, outcomes, map_seed=7, teams=4):
    """A branch_portfolio root record: candidate i played once with outcome outcomes[i] (None: no branch)."""
    branches = [{"candidate": i, "repeat": 0, "outcome": y} for i, y in enumerate(outcomes) if y is not None]
    return {"arm": "inc", "compare_arm": "cmp", "turn": TURN, "map_seed": map_seed, "teams": teams, "candidates": candidates,
            "sources": ["recorded"] + ["move1"] * (len(candidates) - 1), "branches": branches,
            "compare_reply": None if reply is None else " ".join(reply.partition("#")[0].split())}


class Canonical(unittest.TestCase):
    def test_equivalent_orders_share_one_canonical_form(self):
        self.assertEqual(bp.canonical("12  S P   U 3 D -   # note", TURN), INCUMBENT)
        self.assertEqual(bp.canonical(INCUMBENT, TURN), INCUMBENT)
        self.assertIsNone(bp.canonical("12 S P U", TURN))           # too short
        self.assertIsNone(bp.canonical("11 S P U 3 D -", TURN))     # another turn
        self.assertIsNone(bp.canonical("12 Q P U 3 D -", TURN))     # not a move
        self.assertIsNone(bp.canonical(None, TURN))
        self.assertNotEqual(bp.canonical(OTHER, TURN), INCUMBENT)

    def test_candidates_deduplicate_by_canonical_order(self):
        class Game:  # alternatives() needs only these for an order whose units stand off the board
            size, walls = 9, set()
        block = "TURN 12\nSCORE 0 0\nUNITS 6\n" + "".join(f"U {i} {i // 3} -1 -1 0 -1 20\n" for i in range(6)) + \
                "FLAGS 0\nEVENTS 0\nEND"
        cands = bp.candidates(Game(), block, INCUMBENT + " # note", 0, TURN, 4, "k", [("12 S P   U 3 D -", "arm:cmp")])
        self.assertEqual([c for c, _ in cands], [INCUMBENT])  # the identical reply is the incumbent, not a second candidate


class Classify(unittest.TestCase):
    def test_statuses(self):
        orders = [INCUMBENT, OTHER]
        self.assertEqual(bp.classify_compare("12 S P U 3 D -  # same", TURN, orders, {0, 1}), ("identical", INCUMBENT, 0))
        self.assertEqual(bp.classify_compare(OTHER, TURN, orders, {0, 1}), ("different", OTHER, 1))
        self.assertEqual(bp.classify_compare("garbage", TURN, orders, {0, 1})[0], "malformed")
        self.assertEqual(bp.classify_compare(None, TURN, orders, {0, 1})[0], "timeout")
        self.assertEqual(bp.classify_compare(THIRD, TURN, orders, {0, 1})[0], "missing_candidate")
        self.assertEqual(bp.classify_compare(OTHER, TURN, orders, {0})[0], "missing_branch")
        self.assertEqual(bp.classify_compare(None, TURN, orders, {0, 1}, has_compare=False)[0], "none")


class RootStats(unittest.TestCase):
    def test_identical_reply_has_zero_difference(self):
        status, s = branch_regret.root_stats(root("12 S P U 3 D -   # x", [INCUMBENT, OTHER], [0.5, 1.0]), "cmp")
        self.assertEqual(status, "identical")
        self.assertTrue(s["same"])
        self.assertEqual(s["difference"], 0.0)
        self.assertEqual(s["regret_incumbent"], 0.5)

    def test_different_reply_uses_its_own_outcome(self):
        status, s = branch_regret.root_stats(root(THIRD, [INCUMBENT, OTHER, THIRD], [0.5, 1.0, 0.25]), "cmp")
        self.assertEqual(status, "different")
        self.assertFalse(s["same"])
        self.assertAlmostEqual(s["difference"], -0.25)
        self.assertEqual(s["rank_compare"], 3)

    def test_malformed_reply_is_never_identical(self):
        status, s = branch_regret.root_stats(root("12 S P U", [INCUMBENT, OTHER], [0.5, 1.0]), "cmp")
        self.assertEqual((status, s), ("malformed", None))

    def test_timeout_is_excluded(self):
        self.assertEqual(branch_regret.root_stats(root(None, [INCUMBENT, OTHER], [0.5, 1.0]), "cmp"), ("timeout", None))

    def test_valid_different_reply_without_its_branch_is_excluded(self):
        self.assertEqual(branch_regret.root_stats(root(OTHER, [INCUMBENT, OTHER], [0.5, None]), "cmp"), ("missing_branch", None))
        self.assertEqual(branch_regret.root_stats(root(THIRD, [INCUMBENT, OTHER], [0.5, 1.0]), "cmp"), ("missing_candidate", None))

    def test_report_counts_exclusions_and_valid_roots(self):
        rows = [root(INCUMBENT, [INCUMBENT, OTHER], [0.5, 1.0], map_seed=1),
                root(OTHER, [INCUMBENT, OTHER], [0.5, 1.0], map_seed=2),
                root("12 S P U", [INCUMBENT, OTHER], [0.5, 1.0], map_seed=3),
                root(None, [INCUMBENT, OTHER], [0.5, 1.0], map_seed=4)]
        with tempfile.TemporaryDirectory() as d:
            for i, r in enumerate(rows):
                with gzip.open(Path(d) / f"r{i}.json.gz", "wt") as fh:
                    json.dump(r, fh)
            branch_regret.main(["--branches", d, "--arm", "inc", "--compare-arm", "cmp", "--out", str(Path(d) / "rep")])
            rep = json.loads((Path(d) / "rep.json").read_text())
        self.assertEqual(rep["roots"], 2)                       # the valid-root denominator
        self.assertEqual(rep["excluded"], {"malformed": 1, "timeout": 1})
        self.assertEqual(rep["differs_share"], 0.5)
        self.assertAlmostEqual(rep["difference"]["mean"], 0.25)  # (0 + 0.5) / 2: the malformed root is not a zero


if __name__ == "__main__":
    unittest.main()
