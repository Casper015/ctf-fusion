"""Opponent P fidelity audit against actual causal transitions (T0085).

Evaluates P predictions against actual observed opponent actions from factual games:
  - Movement distribution calibration (ECE - Expected Calibration Error across bins).
  - Cross-entropy loss (log-loss).
  - Top-1 move prediction accuracy.
  - Tactical attack prediction accuracy.
  - Denominator accounting (eligible units, excluding own team, respawning, and immune units).
Separates opponent-model error from rules mechanics parity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from tools.search_diagnostics.schemas import (
    SCHEMA_P_FIDELITY,
    ValidationError,
)


@dataclass
class CalibrationBin:
    bin_index: int
    bin_min: float
    bin_max: float
    count: int
    avg_confidence: float
    empirical_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "range": [self.bin_min, self.bin_max],
            "count": self.count,
            "avg_confidence": self.avg_confidence,
            "empirical_accuracy": self.empirical_accuracy,
        }


@dataclass
class PFidelityMetrics:
    schema: str                         # SCHEMA_P_FIDELITY
    total_samples: int
    movement_cross_entropy: float       # mean negative log-likelihood
    movement_accuracy: float            # top-1 move agreement
    expected_calibration_error: float   # ECE
    calibration_bins: list[CalibrationBin]
    attack_samples: int
    attack_accuracy: float
    denominators: dict[str, int]        # eligible_units, skipped_own_team, skipped_immune, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "total_samples": self.total_samples,
            "movement_cross_entropy": self.movement_cross_entropy,
            "movement_accuracy": self.movement_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "calibration_bins": [b.to_dict() for b in self.calibration_bins],
            "attack_samples": self.attack_samples,
            "attack_accuracy": self.attack_accuracy,
            "denominators": self.denominators,
        }


def compute_calibration(
    confidences: list[float],
    accuracies: list[bool],
    num_bins: int = 10,
) -> tuple[float, list[CalibrationBin]]:
    """Compute Expected Calibration Error (ECE) and per-bin diagnostics."""
    if not confidences:
        return 0.0, []
    if len(confidences) != len(accuracies):
        raise ValidationError("confidences and accuracies must have identical length")

    bins: list[CalibrationBin] = []
    total_n = len(confidences)
    weighted_ece = 0.0

    for bi in range(num_bins):
        b_min = bi / num_bins
        b_max = (bi + 1) / num_bins
        in_bin = [
            (c, a)
            for c, a in zip(confidences, accuracies)
            if (b_min <= c < b_max or (bi == num_bins - 1 and c == 1.0))
        ]
        cnt = len(in_bin)
        if cnt > 0:
            avg_conf = sum(c for c, _ in in_bin) / cnt
            emp_acc = sum(1.0 for _, a in in_bin if a) / cnt
            weighted_ece += (cnt / total_n) * abs(avg_conf - emp_acc)
        else:
            avg_conf = (b_min + b_max) / 2.0
            emp_acc = 0.0

        bins.append(
            CalibrationBin(
                bin_index=bi,
                bin_min=b_min,
                bin_max=b_max,
                count=cnt,
                avg_confidence=avg_conf,
                empirical_accuracy=emp_acc,
            )
        )

    return weighted_ece, bins


def evaluate_p_fidelity(
    predictions: list[dict[str, Any]],
    eps: float = 1e-12,
) -> PFidelityMetrics:
    """Evaluate opponent P model against factual observations.

    Each prediction dict should contain:
      - 'predicted_move_probs': list of 5 floats summing to 1.0 (moves 0..4: S, U, D, L, R)
      - 'actual_move': int in 0..4
      - 'is_own_team': bool
      - 'is_immune_or_dead': bool
      - 'predicted_attack_target': optional int
      - 'actual_attack_target': optional int
    """
    confidences: list[float] = []
    accuracies: list[bool] = []
    log_losses: list[float] = []

    attack_total = 0
    attack_correct = 0

    denominators = {
        "total_rows": len(predictions),
        "eligible_units": 0,
        "skipped_own_team": 0,
        "skipped_immune_or_dead": 0,
    }

    for p in predictions:
        if p.get("is_own_team", False):
            denominators["skipped_own_team"] += 1
            continue
        if p.get("is_immune_or_dead", False):
            denominators["skipped_immune_or_dead"] += 1
            continue

        denominators["eligible_units"] += 1
        probs = p["predicted_move_probs"]
        if len(probs) != 5:
            raise ValidationError(f"expected 5 move probabilities, got {len(probs)}")
        sum_p = sum(probs)
        if abs(sum_p - 1.0) > 1e-4:
            raise ValidationError(f"move probabilities must sum to 1.0, got {sum_p}")

        actual_m = int(p["actual_move"])
        if not (0 <= actual_m < 5):
            raise ValidationError(f"invalid actual move: {actual_m}")

        p_actual = max(probs[actual_m], eps)
        log_losses.append(-math.log(p_actual))

        best_m = max(range(5), key=lambda m: probs[m])
        is_acc = (best_m == actual_m)
        accuracies.append(is_acc)
        confidences.append(probs[best_m])

        # Attack evaluation
        if "actual_attack_target" in p and p["actual_attack_target"] is not None:
            act_tg = p["actual_attack_target"]
            pred_tg = p.get("predicted_attack_target")
            attack_total += 1
            if pred_tg == act_tg:
                attack_correct += 1

    if not accuracies:
        raise ValidationError("no eligible opponent predictions found to evaluate")

    mean_loss = sum(log_losses) / len(log_losses)
    top1_acc = sum(1.0 for a in accuracies if a) / len(accuracies)
    ece, cal_bins = compute_calibration(confidences, accuracies, num_bins=10)
    att_acc = (attack_correct / attack_total) if attack_total > 0 else 0.0

    metrics = PFidelityMetrics(
        schema=SCHEMA_P_FIDELITY,
        total_samples=len(accuracies),
        movement_cross_entropy=mean_loss,
        movement_accuracy=top1_acc,
        expected_calibration_error=ece,
        calibration_bins=cal_bins,
        attack_samples=attack_total,
        attack_accuracy=att_acc,
        denominators=denominators,
    )
    return metrics
