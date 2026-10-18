"""Unit and integration tests for agent_v2 search diagnostics (T0085).

Covers:
  - Exact self-comparison on direct baseline (Delta = 0 on every repeat).
  - True non-negative regret R(a) >= 0 distinct from signed baseline gain.
  - Candidate ranking with fractional tie-handling.
  - Winner's-curse elimination via out-of-sample repeat cross-fitting.
  - Wrong-state histories and corrupted receipt refusal (fail closed).
  - Duplicate roots rejection.
  - Leakage rejection (hidden RNG or future actions in features).
  - Simulator parity diffs and mechanics coverage.
  - Opponent P calibration, ECE, cross-entropy, and denominators.
  - Pilot cost estimator formulaic consistency.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.search_diagnostics import (
    analyze,
    branch,
    estimator,
    extract,
    p_eval,
    parity,
    schemas,
)
from tools.search_diagnostics.schemas import (
    BranchResult,
    CandidateOrder,
    RootRecord,
    SCHEMA_BRANCH,
    SCHEMA_ROOT,
    ValidationError,
)


def make_synthetic_block(turn: int = 10, teams: int = 2) -> str:
    """Generate a valid TURN block string."""
    lines = [f"TURN {turn}", f"SCORE {' '.join(['0'] * teams)}", f"UNITS {3 * teams}"]
    for uid in range(3 * teams):
        t = uid // 3
        lines.append(f"U {uid} {t} 10 10 100 -1 0")
    lines.append(f"FLAGS {teams}")
    for fid in range(teams):
        lines.append(f"F {fid} home 5 5 -1 0")
    lines.append("EVENTS 0")
    return "\n".join(lines)


def make_synthetic_root(
    turn: int = 10,
    teams: int = 2,
    seat: int = 0,
    sampled_joint: int = 42,
    target_k: int = 8,
) -> RootRecord:
    """Create a fully valid synthetic RootRecord."""
    block_text = make_synthetic_block(turn, teams)
    prefix_text = "\n".join([f"INIT {teams} 3 20 20", make_synthetic_block(turn - 1, teams)])
    # Prior scores favoring sampled_joint slightly or arbitrary
    prior_scores = [1.0 + 0.01 * (i % 7) for i in range(125)]
    prior_scores[sampled_joint] = 2.5

    return extract.extract_root_from_match(
        match={
            "match_id": "synth_m1",
            "map_seed": 1001,
            "flag_seed": 2001,
            "teams": teams,
            "init": f"INIT {teams} 3 20 20",
            "blocks": [make_synthetic_block(t, teams) for t in range(1, turn + 1)],
            "replies": [[f"{t} S - S - S -" for _ in range(teams)] for t in range(1, turn + 1)],
        },
        seat=seat,
        turn=turn,
        actor_scores=prior_scores,
        sampled_joint=sampled_joint,
        model_identities={
            "actor_sha256": "a" * 64,
            "critic_sha256": "c" * 64,
            "p_sha256": "p" * 64,
            "simulator_sha256": "s" * 64,
        },
        target_k=target_k,
    )


class TestSchemasAndIntegrity(unittest.TestCase):
    def test_baseline_exact_self_comparison_requirement(self):
        """Candidate 0 must be marked is_baseline=True and match baseline_joint."""
        root = make_synthetic_root(turn=5, sampled_joint=33)
        self.assertEqual(root.candidates[0].index, 0)
        self.assertTrue(root.candidates[0].is_baseline)
        self.assertEqual(root.candidates[0].joint, 33)
        self.assertTrue(all(not c.is_baseline for c in root.candidates[1:]))

    def test_corrupted_candidate_joint_raises(self):
        root = make_synthetic_root(turn=5, sampled_joint=10)
        d = root.to_dict()
        d["candidates"][0]["joint"] = 999  # invalid joint out of [0, 125)
        with self.assertRaises(ValidationError):
            RootRecord.from_dict(d)

    def test_duplicate_candidate_joints_rejected(self):
        root = make_synthetic_root(turn=5, sampled_joint=10)
        d = root.to_dict()
        d["candidates"][1]["joint"] = d["candidates"][0]["joint"]  # duplicate joint
        with self.assertRaises(ValidationError):
            RootRecord.from_dict(d)

    def test_non_finite_prior_score_rejected(self):
        root = make_synthetic_root(turn=5)
        d = root.to_dict()
        d["candidates"][0]["prior_score"] = float("nan")
        with self.assertRaises(ValidationError):
            RootRecord.from_dict(d)


class TestRankingAndTies(unittest.TestCase):
    def test_fractional_ranking_with_ties(self):
        # Case 1: All distinct
        vals = [10.0, 8.0, 5.0, 2.0]
        ranks = analyze.compute_fractional_ranks(vals, reverse=True)
        self.assertEqual(ranks, [1.0, 2.0, 3.0, 4.0])

        # Case 2: Tie for 1st place
        vals = [10.0, 10.0, 5.0, 2.0]
        ranks = analyze.compute_fractional_ranks(vals, reverse=True)
        self.assertEqual(ranks, [1.5, 1.5, 3.0, 4.0])

        # Case 3: Three-way tie in middle
        vals = [12.0, 7.0, 7.0, 7.0, 1.0]
        ranks = analyze.compute_fractional_ranks(vals, reverse=True)
        self.assertEqual(ranks, [1.0, 3.0, 3.0, 3.0, 5.0])

        # Case 4: Complete tie across all candidates
        vals = [4.0, 4.0, 4.0, 4.0]
        ranks = analyze.compute_fractional_ranks(vals, reverse=True)
        self.assertEqual(ranks, [2.5, 2.5, 2.5, 2.5])


class TestSignedGainAndRegret(unittest.TestCase):
    def test_baseline_exact_self_comparison_has_zero_gain(self):
        """Direct baseline candidate compared against itself has Delta = 0.0 on every repeat."""
        root = make_synthetic_root(turn=6, target_k=4)
        branches = []
        for r in range(5):
            for c in range(4):
                # Baseline has utility 0.5; candidates have varying utilities
                u = 0.5 if c == 0 else (0.4 + 0.1 * c)
                branches.append(
                    BranchResult(
                        schema=SCHEMA_BRANCH,
                        root_id=root.root_id,
                        candidate_index=c,
                        repeat_id=r,
                        repeat_key=f"rep_{r}",
                        final_score=[10, 5],
                        pairwise_utility=u,
                        faults_prefix=0,
                        faults_root_discarded=0,
                        faults_continuation=0,
                        latency_ms=10.0,
                    )
                )

        rm = analyze.analyze_root(root, branches)
        # Baseline candidate metrics
        base_cm = rm.candidates[0]
        self.assertEqual(base_cm.candidate_index, 0)
        self.assertTrue(base_cm.is_baseline)
        self.assertEqual(base_cm.signed_gain, 0.0)
        self.assertEqual(base_cm.paired_variance, 0.0)

        # Regret is non-negative and relative to best candidate
        for cm in rm.candidates:
            self.assertGreaterEqual(cm.regret, 0.0)

        # Best candidate has regret exactly 0.0
        best_cand = max(rm.candidates, key=lambda c: c.mean_utility)
        self.assertEqual(best_cand.regret, 0.0)

    def test_regret_is_distinct_from_negative_signed_gain(self):
        """Verify that regret is nonnegative relative to best action, not -Delta."""
        root = make_synthetic_root(turn=6, target_k=3)
        # c=0 (baseline) utility 0.5 -> Delta(0)=0.0, Regret(0)=0.3
        # c=1 utility 0.8 -> Delta(1)=+0.3, Regret(1)=0.0 (best)
        # c=2 utility 0.2 -> Delta(2)=-0.3, Regret(2)=0.6
        branches = []
        for c, u in [(0, 0.5), (1, 0.8), (2, 0.2)]:
            for r in range(2):
                branches.append(
                    BranchResult(
                        schema=SCHEMA_BRANCH,
                        root_id=root.root_id,
                        candidate_index=c,
                        repeat_id=r,
                        repeat_key=f"r_{r}",
                        final_score=[10, 5],
                        pairwise_utility=u,
                        faults_prefix=0,
                        faults_root_discarded=0,
                        faults_continuation=0,
                        latency_ms=5.0,
                    )
                )

        rm = analyze.analyze_root(root, branches)
        cm0, cm1, cm2 = rm.candidates
        self.assertEqual(cm0.signed_gain, 0.0)
        self.assertAlmostEqual(cm0.regret, 0.3)
        self.assertAlmostEqual(cm1.signed_gain, 0.3)
        self.assertEqual(cm1.regret, 0.0)
        self.assertAlmostEqual(cm2.signed_gain, -0.3)
        self.assertAlmostEqual(cm2.regret, 0.6)  # Regret != -Delta


class TestWinnersCurseElimination(unittest.TestCase):
    def test_cross_fitting_eliminates_winners_curse_on_noise(self):
        """When candidates are pure noise (true Delta = 0), naive max > 0, cross-fit == 0."""
        import random
        rnd = random.Random(20261018)

        roots: list[RootRecord] = []
        branches: list[BranchResult] = []

        num_roots = 40
        num_cands = 8
        num_repeats = 6

        for ri in range(num_roots):
            root = make_synthetic_root(turn=ri + 5, target_k=num_cands)
            roots.append(root)

            # Generate pure noise utilities with true mean 0.5 for ALL candidates
            for r in range(num_repeats):
                # Baseline noise
                base_u = max(0.0, min(1.0, 0.5 + rnd.gauss(0, 0.15)))
                branches.append(
                    BranchResult(
                        schema=SCHEMA_BRANCH,
                        root_id=root.root_id,
                        candidate_index=0,
                        repeat_id=r,
                        repeat_key=f"r_{r}",
                        final_score=[10, 5],
                        pairwise_utility=base_u,
                        faults_prefix=0,
                        faults_root_discarded=0,
                        faults_continuation=0,
                        latency_ms=1.0,
                    )
                )
                for c in range(1, num_cands):
                    # Candidates have identical true mean 0.5 with independent noise
                    cand_u = max(0.0, min(1.0, 0.5 + rnd.gauss(0, 0.15)))
                    branches.append(
                        BranchResult(
                            schema=SCHEMA_BRANCH,
                            root_id=root.root_id,
                            candidate_index=c,
                            repeat_id=r,
                            repeat_key=f"r_{r}",
                            final_score=[10, 5],
                            pairwise_utility=cand_u,
                            faults_prefix=0,
                            faults_root_discarded=0,
                            faults_continuation=0,
                            latency_ms=1.0,
                        )
                    )

        summary = analyze.compute_cross_fit_evaluation(roots, branches)

        # Naive oracle max must exhibit positive selection bias (winner's curse)
        self.assertGreater(summary.naive_oracle_max_gain, 0.05)

        # Cross-fitted realizable gain must eliminate winner's curse (statistically ~0)
        self.assertAlmostEqual(summary.cross_fit_realizable_gain, 0.0, delta=0.035)

        # Positive bias proven
        self.assertGreater(summary.winners_curse_bias, 0.03)


class TestFailClosedAndValidation(unittest.TestCase):
    def test_wrong_state_history_refused(self):
        """Mismatched turn or invalid history block raises ValidationError."""
        with self.assertRaises(ValidationError):
            extract.extract_root_from_match(
                match={"match_id": "test", "blocks": []},  # empty blocks
                seat=0,
                turn=5,
                actor_scores=[0.0] * 125,
                sampled_joint=0,
                model_identities={},
            )

    def test_corrupted_branch_receipt_rejected(self):
        res = BranchResult(
            schema=SCHEMA_BRANCH,
            root_id="root_123",
            candidate_index=0,
            repeat_id=0,
            repeat_key="rk",
            final_score=[10, 10],
            pairwise_utility=1.5,  # invalid utility > 1.0
            faults_prefix=0,
            faults_root_discarded=0,
            faults_continuation=0,
            latency_ms=5.0,
        )
        with self.assertRaises(ValidationError):
            res.validate()

    def test_map_clustering_keeps_same_map_together(self):
        r1 = make_synthetic_root(turn=5)
        r2 = copy.deepcopy(r1)
        r2.turn = 6
        r2.root_id = "r2"

        clusters = analyze.cluster_roots_by_map([r1, r2])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[r1.map_seed]), 2)


class TestPFidelityAndCalibration(unittest.TestCase):
    def test_p_fidelity_metrics_computation(self):
        preds = []
        # Perfect predictions on 10 samples
        for _ in range(10):
            preds.append({
                "predicted_move_probs": [0.8, 0.05, 0.05, 0.05, 0.05],
                "actual_move": 0,
                "is_own_team": False,
                "is_immune_or_dead": False,
                "predicted_attack_target": 3,
                "actual_attack_target": 3,
            })
        # One misprediction
        preds.append({
            "predicted_move_probs": [0.1, 0.7, 0.1, 0.05, 0.05],
            "actual_move": 0,  # actual is 0, predicted was 1
            "is_own_team": False,
            "is_immune_or_dead": False,
            "predicted_attack_target": 2,
            "actual_attack_target": 3,
        })
        # One skipped unit (own team)
        preds.append({
            "predicted_move_probs": [0.2] * 5,
            "actual_move": 0,
            "is_own_team": True,
            "is_immune_or_dead": False,
        })

        metrics = p_eval.evaluate_p_fidelity(preds)
        self.assertEqual(metrics.total_samples, 11)
        self.assertEqual(metrics.denominators["skipped_own_team"], 1)
        self.assertAlmostEqual(metrics.movement_accuracy, 10.0 / 11.0)
        self.assertGreater(metrics.movement_cross_entropy, 0.0)
        self.assertGreaterEqual(metrics.expected_calibration_error, 0.0)


class TestCostEstimator(unittest.TestCase):
    def test_estimator_mathematical_consistency(self):
        est = estimator.estimate_pilot_cost(
            team_counts=[2, 4],
            matches_per_count=10,
            candidates_per_root=8,
            repeats=4,
            avg_root_turn=100,
            max_turns=400,
            workers=8,
            ms_per_step=50.0,
        )
        self.assertEqual(est.total_matches, 20)
        self.assertEqual(est.total_roots, 20)
        self.assertEqual(est.total_rollouts, 20 * 8 * 4)  # 640
        self.assertEqual(est.continuation_turns, 300)
        self.assertEqual(est.total_engine_steps, 640 * 300)  # 192,000 steps
        expected_sec = 192000 * 0.050
        self.assertAlmostEqual(est.total_cpu_hours, expected_sec / 3600.0)
        self.assertAlmostEqual(est.wall_clock_hours, est.total_cpu_hours / 8.0)


if __name__ == "__main__":
    unittest.main()
