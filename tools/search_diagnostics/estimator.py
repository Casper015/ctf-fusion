"""Authoritative host pilot cost estimator for agent_v2 branch diagnostics (T0085).

Calculates exact continuation counts, step volumes, bot wall-clock hours,
and disk footprint for proposed diagnostic pilots under the 8-worker CPU lease.
Zero mock numbers; strictly input-bound formulaic derivations.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from tools.search_diagnostics.schemas import (
    SCHEMA_ESTIMATE,
    ValidationError,
)

DEFAULT_WORKERS = 8
DEFAULT_STEP_MS = 50.0
DEFAULT_BYTES_PER_ROW = 512


@dataclass
class PilotCostEstimate:
    schema: str                         # SCHEMA_ESTIMATE
    team_counts: list[int]
    matches_per_count: int
    total_matches: int
    roots_per_match: int
    total_roots: int
    candidates_per_root: int
    repeats: int
    total_rollouts: int
    avg_root_turn: int
    max_turns: int
    continuation_turns: int
    total_engine_steps: int
    total_bot_turns: int
    workers: int
    ms_per_step: float
    total_cpu_hours: float
    wall_clock_hours: float
    disk_footprint_mb: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def estimate_pilot_cost(
    team_counts: list[int] | None = None,
    matches_per_count: int = 16,
    roots_per_match: int = 1,
    candidates_per_root: int = 8,
    repeats: int = 4,
    avg_root_turn: int = 100,
    max_turns: int = 400,
    workers: int = DEFAULT_WORKERS,
    ms_per_step: float = DEFAULT_STEP_MS,
    bytes_per_row: int = DEFAULT_BYTES_PER_ROW,
) -> PilotCostEstimate:
    """Compute exact game and worker-hour requirements for a proposed host pilot."""
    counts = team_counts or [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if not counts:
        raise ValidationError("team_counts cannot be empty")
    if matches_per_count < 1:
        raise ValidationError("matches_per_count must be >= 1")
    if candidates_per_root < 1:
        raise ValidationError("candidates_per_root must be >= 1")
    if repeats < 1:
        raise ValidationError("repeats must be >= 1")
    if workers < 1:
        raise ValidationError("workers must be >= 1")
    if avg_root_turn >= max_turns:
        raise ValidationError("avg_root_turn must be < max_turns")

    total_matches = len(counts) * matches_per_count
    total_roots = total_matches * roots_per_match
    total_rollouts = total_roots * candidates_per_root * repeats

    cont_turns = max_turns - avg_root_turn
    total_steps = total_rollouts * cont_turns

    avg_teams = sum(counts) / len(counts)
    total_bot_turns = int(total_steps * avg_teams)

    # Wall-clock estimation
    # Total step time = total_steps * ms_per_step
    total_sec = total_steps * (ms_per_step / 1000.0)
    cpu_hours = total_sec / 3600.0
    wall_hours = cpu_hours / workers

    disk_mb = (total_rollouts * bytes_per_row) / (1024.0 * 1024.0)

    return PilotCostEstimate(
        schema=SCHEMA_ESTIMATE,
        team_counts=counts,
        matches_per_count=matches_per_count,
        total_matches=total_matches,
        roots_per_match=roots_per_match,
        total_roots=total_roots,
        candidates_per_root=candidates_per_root,
        repeats=repeats,
        total_rollouts=total_rollouts,
        avg_root_turn=avg_root_turn,
        max_turns=max_turns,
        continuation_turns=cont_turns,
        total_engine_steps=total_steps,
        total_bot_turns=total_bot_turns,
        workers=workers,
        ms_per_step=ms_per_step,
        total_cpu_hours=cpu_hours,
        wall_clock_hours=wall_hours,
        disk_footprint_mb=disk_mb,
    )


def format_pilot_cost_markdown(est: PilotCostEstimate) -> str:
    """Format PilotCostEstimate into a standard GitHub Markdown report."""
    lines = [
        "## Agent V2 Branch Diagnostics: Pilot Cost Estimate",
        "",
        "| Metric | Value | Unit |",
        "|---|---:|---|",
        f"| Team Counts Evaluated | {len(est.team_counts)} ({min(est.team_counts)}..{max(est.team_counts)}) | counts |",
        f"| Matches per Count | {est.matches_per_count:,} | matches |",
        f"| Total Base Matches | {est.total_matches:,} | matches |",
        f"| Total Causal Roots | {est.total_roots:,} | roots |",
        f"| Candidates per Root ($K$) | {est.candidates_per_root:,} | orders |",
        f"| Repeats per Candidate ($R$) | {est.repeats:,} | seeds |",
        f"| Total Continuation Rollouts | {est.total_rollouts:,} | games |",
        f"| Average Continuation Length | {est.continuation_turns:,} | turns |",
        f"| Total Engine Steps | {est.total_engine_steps:,} | steps |",
        f"| Total Bot Inferences | {est.total_bot_turns:,} | calls |",
        f"| Assigned Workers (CPU lease) | {est.workers} | workers |",
        f"| Estimated Step Latency | {est.ms_per_step:.1f} | ms |",
        f"| Total Compute Budget | {est.total_cpu_hours:.2f} | CPU-hours |",
        f"| **Host Wall-Clock Runtime** | **{est.wall_clock_hours:.2f}** | **hours** |",
        f"| Row Manifest Disk Size | {est.disk_footprint_mb:.2f} | MB |",
        "",
        "> [!IMPORTANT]",
        f"> The pilot requires {est.wall_clock_hours:.2f} hours across {est.workers} dedicated workers on `desktop-g60ojud`.",
        "> GPU training must be stopped during CPU execution per repo lease constraints.",
    ]
    return "\n".join(lines)
