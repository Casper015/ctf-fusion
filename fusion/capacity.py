"""Shared architecture presets for training and CPU capacity checks.

Presets describe shapes, not trained quality or deadline approval. S is the default; XXL is the largest. P and V can
choose capacities independently. Returned lists are copies, so CLI overrides
cannot mutate a subsequent experiment's preset.
"""
from __future__ import annotations

from .quant import MAX_WIDTH

POLICY_HEADS = ("mover", "squad", "act", "target")
NETS = (*POLICY_HEADS, "value")
TIERS = {
    "S": {"mover": [16], "squad": [64, 32], "act": [16], "target": [8], "value": [128, 16]},
    "M": {"mover": [32], "squad": [128, 64], "act": [32], "target": [16], "value": [256, 32]},
    "L": {"mover": [64], "squad": [256, 64], "act": [64], "target": [16], "value": [512, 64]},
    "XL": {"mover": [128], "squad": [512, 128], "act": [128], "target": [32], "value": [1024, 64]},
    "XXL": {"mover": [256], "squad": [1024, 256], "act": [256], "target": [32], "value": [2048, 128]},
}
# No preset wider than XXL: on the host, the 3XL/4XL shapes needed 23 ms and 191 ms of inference per turn under a
# 16-process load against a 5 ms budget (reports/host/capacity_extreme_scaling_16seats.json); XXL needed 2.7 ms. Wider
# custom shapes remain possible with explicit --hidden-* and --max-width.


def preset(name: str) -> dict[str, list[int]]:
    return {head: list(widths) for head, widths in TIERS[name].items()}


def buffer_width(dimensions) -> int:
    """At least the native default, rounded to SIMD-friendly blocks of 64."""
    widths = list(dimensions)
    if not widths or any(type(w) is not int or w < 1 for w in widths):
        raise ValueError("buffer dimensions must be positive integers")
    return max(MAX_WIDTH, 64 * ((max(widths) + 63) // 64))


def hidden_buffer_width(hidden: dict[str, list[int]]) -> int:
    return buffer_width(w for widths in hidden.values() for w in widths)
