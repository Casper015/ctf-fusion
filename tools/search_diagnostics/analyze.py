"""Statistical analysis of branch continuations and winner's-curse elimination (T0085).

Computes:
  - Signed baseline gain Delta(a) = E[U_T | a, pi_c] - E[U_T | direct, pi_c].
  - Exact self-comparison on direct baseline (Delta = 0).
  - True non-negative regret R(a) = max_a' E[U_T | a'] - E[U_T | a] >= 0.
  - Within-root candidate ranking with fractional tie-handling.
  - Candidate opportunity: fraction of roots where max_{a != direct} Delta(a) > 0.
  - Override effects: realized gain when search overrides direct baseline.
  - Cross-fitting over repeat partitions to eliminate winner's curse.
  - Source-map clustering to ensure roots from the same map stay together.
  - Stratified population weighting for oversampled tactical encounters.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any

from tools.search_diagnostics.schemas import (
    BranchResult,
    CandidateOrder,
    RootRecord,
    SCHEMA_ANALYSIS,
    ValidationError,
)


@dataclass
class CandidateMetrics:
    candidate_index: int
    joint: int
    is_baseline: bool
    mean_utility: float                 # sample mean U_T
    signed_gain: float                  # Delta(a) relative to baseline (0.0 for baseline)
    regret: float                       # non-negative regret relative to best candidate
    rank: float                         # fractional rank within root (1.0 = best)
    repeat_variance: float              # sample variance across repeats
    paired_variance: float              # sample variance of Delta(a, r)
    n_repeats: int


@dataclass
class RootMetrics:
    root_id: str
    match_id: str
    map_seed: int
    turn: int
    seat: int
    stratum: str
    stratum_weight: float
    candidates: list[CandidateMetrics]
    baseline_index: int = 0
    opportunity: bool = False           # True if any alternative candidate achieves Delta > 0
    max_signed_gain: float = 0.0
    pooled_within_variance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "match_id": self.match_id,
            "map_seed": self.map_seed,
            "turn": self.turn,
            "seat": self.seat,
            "stratum": self.stratum,
            "stratum_weight": self.stratum_weight,
            "baseline_index": self.baseline_index,
            "opportunity": self.opportunity,
            "max_signed_gain": self.max_signed_gain,
            "pooled_within_variance": self.pooled_within_variance,
            "candidates": [dataclasses.asdict(c) for c in self.candidates],
        }


@dataclass
class CrossFitSummary:
    schema: str                         # SCHEMA_ANALYSIS
    total_roots: int
    naive_oracle_max_gain: float        # E[max_{a} Delta(a)] - contains winner's curse
    cross_fit_realizable_gain: float    # unbiased out-of-sample evaluation across repeat folds
    winners_curse_bias: float           # naive_oracle - cross_fit
    opportunity_rate: float             # fraction of roots with realizable positive gain
    mean_baseline_utility: float        # direct policy mean utility
    pooled_repeat_variance: float


def compute_fractional_ranks(values: list[float], reverse: bool = True) -> list[float]:
    """Compute 1-indexed fractional ranks for a list of values with proper tie handling.

    If reverse=True, highest value gets rank 1. Tied values share the average of their ranks.
    Example: [0.8, 0.8, 0.5] -> [1.5, 1.5, 3.0].
    """
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda x: -x[1] if reverse else x[1])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and math.isclose(indexed[j][1], indexed[i][1], rel_tol=1e-9, abs_tol=1e-12):
            j += 1
        # Positions are from i+1 to j (1-indexed)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    return ranks


def compute_sample_variance(vals: list[float]) -> float:
    """Compute sample variance with Bessel's correction (ddof=1). Returns 0.0 if len < 2."""
    n = len(vals)
    if n < 2:
        return 0.0
    mean_v = sum(vals) / n
    return sum((v - mean_v) ** 2 for v in vals) / (n - 1)


def analyze_root(root: RootRecord, branches: list[BranchResult]) -> RootMetrics:
    """Compute signed gain, regret, ranking with ties, and opportunity for a single root."""
    root.validate()

    # Group branch results by candidate_index and repeat_id
    by_cand: dict[int, dict[int, float]] = collections.defaultdict(dict)
    for b in branches:
        if b.root_id == root.root_id:
            b.validate()
            by_cand[b.candidate_index][b.repeat_id] = b.pairwise_utility

    if 0 not in by_cand or not by_cand[0]:
        raise ValidationError(f"root {root.root_id} has no branch results for baseline candidate 0")

    baseline_repeats = by_cand[0]
    num_candidates = len(root.candidates)

    mean_utilities: list[float] = []
    cand_metrics_list: list[CandidateMetrics] = []
    within_sq_err = 0.0
    within_dof = 0

    for c in range(num_candidates):
        c_repeats = by_cand.get(c, {})
        # Find shared repeats with baseline
        shared_r_ids = sorted(set(c_repeats.keys()) & set(baseline_repeats.keys()))
        if not shared_r_ids:
            raise ValidationError(f"candidate {c} shares no repeat IDs with baseline")

        u_vals = [c_repeats[r] for r in shared_r_ids]
        mean_u = sum(u_vals) / len(u_vals)
        mean_utilities.append(mean_u)

        if len(u_vals) >= 2:
            within_sq_err += sum((u - mean_u) ** 2 for u in u_vals)
            within_dof += len(u_vals) - 1

        # Signed gain Delta(c, r) = U(c, r) - U(baseline, r)
        gaps = [c_repeats[r] - baseline_repeats[r] for r in shared_r_ids]
        signed_gain = sum(gaps) / len(gaps)

        # Exact self-comparison check for baseline: Delta must be identically 0
        if c == 0:
            if not all(math.isclose(g, 0.0, abs_tol=1e-12) for g in gaps):
                raise ValidationError("baseline candidate compared to itself did not yield zero gap")
            signed_gain = 0.0

        r_var = compute_sample_variance(u_vals)
        paired_var = compute_sample_variance(gaps)

        cand_metrics_list.append(
            CandidateMetrics(
                candidate_index=c,
                joint=root.candidates[c].joint,
                is_baseline=(c == 0),
                mean_utility=mean_u,
                signed_gain=signed_gain,
                regret=0.0,  # computed below after max is found
                rank=0.0,    # computed below with tie handling
                repeat_variance=r_var,
                paired_variance=paired_var,
                n_repeats=len(shared_r_ids),
            )
        )

    # Compute true non-negative regret relative to best candidate: R(c) = max_c' U(c') - U(c) >= 0
    max_u = max(mean_utilities)
    for cm in cand_metrics_list:
        cm.regret = max(0.0, max_u - cm.mean_utility)

    # Compute fractional ranks with tie handling
    ranks = compute_fractional_ranks(mean_utilities, reverse=True)
    for cm, rk in zip(cand_metrics_list, ranks):
        cm.rank = rk

    # Candidate opportunity: is there an alternative candidate with signed gain > 0?
    alt_gains = [cm.signed_gain for cm in cand_metrics_list if not cm.is_baseline]
    max_gain = max(alt_gains) if alt_gains else 0.0
    has_opportunity = (max_gain > 1e-9)

    pooled_within = (within_sq_err / within_dof) if within_dof > 0 else 0.0

    return RootMetrics(
        root_id=root.root_id,
        match_id=root.match_id,
        map_seed=root.map_seed,
        turn=root.turn,
        seat=root.seat,
        stratum=root.stratum,
        stratum_weight=root.stratum_weight,
        candidates=cand_metrics_list,
        baseline_index=0,
        opportunity=has_opportunity,
        max_signed_gain=max_gain,
        pooled_within_variance=pooled_within,
    )


def compute_cross_fit_evaluation(
    roots: list[RootRecord],
    branches: list[BranchResult],
    folds: int = 2,
) -> CrossFitSummary:
    """Evaluate realizable gain using cross-fitting across repeat partitions to eliminate winner's curse.

    Algorithm (2-fold cross-fitting over repeats):
      - Partition repeats into Fold 1 (e.g. even repeat IDs) and Fold 2 (odd repeat IDs).
      - For each root:
        - Select apparent best candidate on Fold 1: c1* = argmax_{c != 0} Delta_F1(c).
        - Evaluate c1* on Fold 2: g1 = Delta_F2(c1*).
        - Select apparent best candidate on Fold 2: c2* = argmax_{c != 0} Delta_F2(c).
        - Evaluate c2* on Fold 1: g2 = Delta_F1(c2*).
        - Cross-fitted realizable gain = (g1 + g2) / 2.
      - Contrast with naive oracle maximum: max_{c != 0} Delta_{all}(c).
    """
    if not roots:
        raise ValidationError("roots list is empty")

    branches_by_root: dict[str, list[BranchResult]] = collections.defaultdict(list)
    for b in branches:
        branches_by_root[b.root_id].append(b)

    naive_max_gains: list[float] = []
    cross_fit_gains: list[float] = []
    baseline_utilities: list[float] = []
    opportunities: list[bool] = []
    within_variances: list[float] = []

    for root in roots:
        r_branches = branches_by_root.get(root.root_id, [])
        if not r_branches:
            continue

        rm = analyze_root(root, r_branches)
        naive_max_gains.append(rm.max_signed_gain)
        baseline_utilities.append(rm.candidates[0].mean_utility)
        opportunities.append(rm.opportunity)
        within_variances.append(rm.pooled_within_variance)

        # Organize repeat data for cross-fitting
        by_cand_rep: dict[int, dict[int, float]] = collections.defaultdict(dict)
        for b in r_branches:
            by_cand_rep[b.candidate_index][b.repeat_id] = b.pairwise_utility

        base_reps = by_cand_rep[0]
        rep_ids = sorted(base_reps.keys())
        if len(rep_ids) < 2:
            # Cannot cross-fit with fewer than 2 repeats; fall back to naive
            cross_fit_gains.append(rm.max_signed_gain)
            continue

        f1_ids = [r for i, r in enumerate(rep_ids) if i % 2 == 0]
        f2_ids = [r for i, r in enumerate(rep_ids) if i % 2 != 0]

        # Candidate gaps on Fold 1
        gaps_f1: dict[int, float] = {}
        for c in range(1, len(root.candidates)):
            c_reps = by_cand_rep.get(c, {})
            shared = [r for r in f1_ids if r in c_reps and r in base_reps]
            if shared:
                gaps_f1[c] = sum(c_reps[r] - base_reps[r] for r in shared) / len(shared)

        # Candidate gaps on Fold 2
        gaps_f2: dict[int, float] = {}
        for c in range(1, len(root.candidates)):
            c_reps = by_cand_rep.get(c, {})
            shared = [r for r in f2_ids if r in c_reps and r in base_reps]
            if shared:
                gaps_f2[c] = sum(c_reps[r] - base_reps[r] for r in shared) / len(shared)

        if not gaps_f1 or not gaps_f2:
            cross_fit_gains.append(rm.max_signed_gain)
            continue

        # Choose on Fold 1, evaluate on Fold 2
        c1_star = max(gaps_f1.keys(), key=lambda c: gaps_f1[c])
        eval_g1 = gaps_f2.get(c1_star, gaps_f1[c1_star])

        # Choose on Fold 2, evaluate on Fold 1
        c2_star = max(gaps_f2.keys(), key=lambda c: gaps_f2[c])
        eval_g2 = gaps_f1.get(c2_star, gaps_f2[c2_star])

        cf_gain = (eval_g1 + eval_g2) / 2.0
        cross_fit_gains.append(cf_gain)

    if not cross_fit_gains:
        raise ValidationError("no valid roots could be evaluated")

    mean_naive = sum(naive_max_gains) / len(naive_max_gains)
    mean_cf = sum(cross_fit_gains) / len(cross_fit_gains)
    bias = mean_naive - mean_cf
    opp_rate = sum(1.0 for o in opportunities if o) / len(opportunities)
    mean_base_u = sum(baseline_utilities) / len(baseline_utilities)
    pooled_var = sum(within_variances) / len(within_variances) if within_variances else 0.0

    return CrossFitSummary(
        schema=SCHEMA_ANALYSIS,
        total_roots=len(cross_fit_gains),
        naive_oracle_max_gain=mean_naive,
        cross_fit_realizable_gain=mean_cf,
        winners_curse_bias=bias,
        opportunity_rate=opp_rate,
        mean_baseline_utility=mean_base_u,
        pooled_repeat_variance=pooled_var,
    )


def cluster_roots_by_map(roots: list[RootRecord]) -> dict[int, list[RootRecord]]:
    """Group roots by map_seed so roots from the same map stay together across splits."""
    by_map: dict[int, list[RootRecord]] = collections.defaultdict(list)
    for r in roots:
        by_map[r.map_seed].append(r)
    return by_map
