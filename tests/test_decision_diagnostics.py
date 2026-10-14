"""Decision-data diagnostics and provenance (task T0042): one noise population, repeat ids, faults by phase.

Synthetic outcomes on real recorded prefixes (tests/test_decision_learning.py fixtures) and hand-made arrays. Nothing
here plays a game, trains a model or says anything about real noise levels.
"""
from __future__ import annotations

import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_decision_learning import build_fixture, handmade
from training import decision_dataset as D
from training import decision_noise as N
from training import evaluate_decisions as E

USABLE = [(2, 1, [("a", [0.0, 1.0, 0.5], True), ("b", [1.0, 1.0, 0.0], True), ("c", [0.5, 0.5, 1.0], True)]),
          (3, 2, [("a", [0.25, 0.75], True), ("b", [1.0, 0.5], True)])]
# An incumbent and unsupported alternatives with large repeat variance: no supported alternative, no noise root.
UNSUPPORTED_ROOT = (2, 3, [("a", [0.0, 1.0, 0.0, 1.0], True), ("x", [1.0, 0.0, 1.0, 0.0], False),
                           ("y", [0.0, 1.0, 1.0, 0.0], False)])


class NoisePopulation(unittest.TestCase):
    def test_unsupported_root_changes_nothing(self):
        """T0038 counterexample: adding an unsupported root must leave raw, within, net and paired statistics alone."""
        base, more = N.repeat_stats(handmade(USABLE)), N.repeat_stats(handmade(USABLE + [UNSUPPORTED_ROOT]))
        for key in ("between_action_variance_raw", "within_action_variance", "between_action_variance_net",
                    "noise_roots", "noise_candidates"):
            self.assertEqual(base[key], more[key], key)
        self.assertEqual(base["paired"], more["paired"])
        # the all-candidate within variance does change: that pooled population is what the old correction used
        self.assertNotAlmostEqual(base["within_action_variance_all_candidates"], more["within_action_variance_all_candidates"])

    def test_unlabeled_and_single_candidate_roots_change_nothing(self):
        extra = [(2, 4, [("a", [0.0, 1.0], True), ("b", [None, None], True)]),     # alternative without an outcome
                 (2, 5, [("a", [0.0, 1.0], True)])]                                  # the incumbent alone
        self.assertEqual(N.repeat_stats(handmade(USABLE))["between_action_variance_net"],
                         N.repeat_stats(handmade(USABLE + extra))["between_action_variance_net"])

    def test_net_uses_the_matched_within_variance(self):
        d = handmade(USABLE)
        st = N.repeat_stats(d)
        pop = N.noise_population(d)
        raw = np.mean([np.var(d["cand_y"][c], ddof=1) for _, c in pop])
        corr = st["within_action_variance"] * np.mean([np.mean(1.0 / d["cand_n"][c]) for _, c in pop])
        self.assertAlmostEqual(st["between_action_variance_net"], raw - corr)
        self.assertEqual(len(st["assumptions"]), len(N.NOISE_ASSUMPTIONS))

    def test_paired_gap_variance_and_covariance(self):
        same = handmade([(2, 1, [("a", [0.0, 1.0, 0.5], True), ("b", [0.25, 1.25, 0.75], True)])])
        p = N.repeat_stats(same)["paired"]
        self.assertAlmostEqual(p["paired_gap_variance"], 0.0)       # the gap is constant across repeat ids
        self.assertAlmostEqual(p["implied_covariance"], p["candidate_variance"])
        shifted = handmade([(2, 1, [("a", [0.0, 1.0, 0.5], True), ("b", [None, 0.25, 1.25, 0.75], True)])])
        p2 = N.repeat_stats(shifted)["paired"]
        self.assertEqual(p2["pairs"], 1)                            # paired on ids 1 and 2 only, not packed
        self.assertNotAlmostEqual(p2["paired_gap_variance"], 0.0)
        one = handmade([(2, 1, [("a", [0.0], True), ("b", [1.0], True)])])
        self.assertIsNone(N.repeat_stats(one)["paired"]["paired_gap_variance"])

    def test_headroom_partitions_by_repeat_id(self):
        # repeat 0 of "b" is missing: even ids select, odd ids evaluate, with nothing shifted into the other side
        d = handmade([(2, 1, [("a", [0.5, 0.5, 0.5, 0.5], True), ("b", [None, 0.0, 1.0, 0.0], True)]),
                      (2, 2, [("a", [0.5, 0.5, 0.5, 0.5], True), ("b", [None, 0.0, 1.0, 0.0], True)])])
        h = E.headroom(d, 0.002)
        self.assertAlmostEqual(h["headroom"]["mean"], -0.5)          # selected on id 2 (1.0), evaluated on ids 1, 3 (0.0)
        lopsided = handmade([(2, 1, [("a", [0.5, 0.5, 0.5], True), ("b", [1.0, None, 0.0], True)]),
                             (2, 2, [("a", [0.5, 0.5, 0.5], True), ("b", [1.0, None, 0.0], True)])])
        self.assertTrue(E.headroom(lopsided, 0.002)["status"].startswith("unavailable"))   # "b" has no odd-id outcome

    def test_failed_repeats_are_kept_and_counted(self):
        d = handmade([(2, 1, [("a", [0.0, "fail", 1.0], True), ("b", [1.0, 1.0, "fail"], False)])])
        self.assertEqual(int(d["cand_n"][0]), 2)
        self.assertEqual(N.coverage(d)["failed_repeats"], 2)
        self.assertEqual(N.coverage(d)["unsupported"], 1)
        self.assertEqual(N.faults(d)["failed_repeats"], 2)


class RepeatIdsAndPhases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, name, mutate):
        bdir, mdir = build_fixture(self.tmp / name, reps=3, mutate=mutate)
        built, stats = D.build(D.branch_listing(bdir), mdir, log=lambda *_: None)
        return {sp: D.to_arrays(built["decision"][sp]) for sp in D.SPLITS}, stats

    def test_missing_and_failed_repeats_keep_their_ids(self):
        def mutate(row):
            br = [b for b in row["branches"] if not (b["candidate"] == 1 and b["repeat"] == 0)]   # repeat 0 never ran
            for b in br:
                if b["candidate"] == 2 and b["repeat"] == 1:
                    b.update(score=None, outcome=None, error="engine crashed")             # a failed continuation
            row["branches"] = br
        arrs, stats = self.build("ids", mutate)
        d = arrs["train"]
        r0 = 0
        c1, c2 = d["root_first"][r0] + 1, d["root_first"][r0] + 2
        np.testing.assert_array_equal(d["cand_repeat_state"][c1], [D.REPEAT_NONE, D.REPEAT_OUTCOME, D.REPEAT_OUTCOME])
        self.assertTrue(np.isnan(d["cand_outcome"][c1][0]) and np.isfinite(d["cand_outcome"][c1][1:]).all())
        np.testing.assert_array_equal(d["cand_repeat_state"][c2], [D.REPEAT_OUTCOME, D.REPEAT_FAILED, D.REPEAT_OUTCOME])
        self.assertEqual(int(d["cand_n"][c1]), 2)
        self.assertEqual(int(d["cand_n"][c2]), 2)
        self.assertAlmostEqual(d["cand_y"][c2], np.nanmean(d["cand_outcome"][c2]))
        self.assertEqual(d["cand_faults"][c1][0], D.FAULT_NO_BRANCH)
        reasons = {f["reason"] for f in stats["failed"]}
        self.assertEqual(reasons, {"engine crashed"})

    def test_outcome_without_score_refused(self):
        def mutate(row):
            row["branches"][1].update(score=None)
        with self.assertRaises(D.LinkError):
            self.build("noscore", mutate)

    def test_repeat_id_must_be_an_integer(self):
        def mutate(row):
            row["branches"][0]["repeat"] = "r0"
        with self.assertRaises(D.LinkError):
            self.build("badid", mutate)

    def test_fault_phases(self):
        def mutate(row):
            for b in row["branches"]:
                if b["candidate"] == 0 and b["repeat"] == 0:
                    b.update(prefix_faults=2, faults=[[20, 1, "timeout"]],
                             discarded_root_computed_faults=[[12, 0, "no reply"], [12, 1, "timeout"]])
                if b["candidate"] == 0 and b["repeat"] == 1:
                    b.pop("prefix_faults")
        arrs, _ = self.build("phases", mutate)
        d = arrs["train"]
        inc = d["root_incumbent"][0]
        self.assertEqual(int(d["cand_faults_prefix"][inc][0]), 2)
        self.assertEqual(int(d["cand_faults_root"][inc][0]), 2)
        self.assertEqual(int(d["cand_faults"][inc][0]), 1)                         # continuation only
        self.assertEqual(int(d["cand_faults_prefix"][inc][1]), D.FAULT_UNRECORDED)
        self.assertEqual(int(d["cand_faults_root"][inc][1]), D.FAULT_UNRECORDED)   # the fixture records no root phase
        f, roots = N.faults(d), len(d["root_first"])                                # every root got the same mutation
        self.assertEqual(f["prefix"]["faults"], 2 * roots)
        self.assertEqual(f["prefix"]["repeats_unrecorded"], roots)
        self.assertEqual(f["root"]["faults"], 2 * roots)
        self.assertEqual(f["continuation"]["faults"], roots)

    def test_index_lists_failed_repeats_and_schema(self):
        def mutate(row):
            row["branches"][-1].update(score=None, outcome=None)
        bdir, mdir = build_fixture(self.tmp / "index", reps=2, mutate=mutate)
        out = self.tmp / "index" / "data"
        D.main(["--branches", str(bdir), "--matches", str(mdir), "--out", str(out)])
        index = json.loads((out / "index.json").read_text())
        self.assertEqual(index["schema"], "ctf-fusion-decision-2")
        self.assertGreater(index["failed_repeats"]["count"], 0)
        self.assertEqual(index["failed_repeats"]["repeats"][0]["reason"], "no final score recorded")

    def test_old_schema_refused(self):
        bdir, mdir = build_fixture(self.tmp / "old", reps=1)
        out = self.tmp / "old" / "data"
        D.main(["--branches", str(bdir), "--matches", str(mdir), "--out", str(out)])
        index = json.loads((out / "index.json").read_text())
        index["schema"] = "ctf-fusion-decision-1"
        (out / "index.json").write_text(json.dumps(index))
        with self.assertRaises(SystemExit):
            D.index_of(out)


if __name__ == "__main__":
    unittest.main()
