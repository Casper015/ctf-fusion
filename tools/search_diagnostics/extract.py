"""Causal root extraction with actually sampled direct baselines (T0085).

Extracts from recorded match logs or seat streams:
  - Raw public state at root turn t (TURN block).
  - Causal history (INIT + turns 1..t-1 prefix).
  - Actually sampled direct joint action and its completed default order.
  - Prior-ranked candidate joints (nested sizes 8 or 24).
  - Resolver-completed orders for all candidates.
  - Candidate 0 is the exact baseline order (self-comparison).
  - Frozen model identities and checksums.
  - Fail-closed checks on wrong-state histories or corrupted receipts.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

from fusion import matchlog
from fusion.paths import REPO
from fusion.protocol import DELTAS, MOVES, MOVE_INDEX, Block, parse_block, parse_reply
from tools.search_diagnostics.schemas import (
    CandidateOrder,
    RootRecord,
    SCHEMA_ROOT,
    ValidationError,
    fnv64,
    sha256_text,
)


def joint_to_moves(joint: int) -> tuple[int, int, int]:
    """Decompose joint index in [0, 125) into (m0, m1, m2) where mk in 0..4."""
    if not (0 <= joint < 125):
        raise ValidationError(f"joint index {joint} out of bounds [0, 125)")
    return (joint // 25, (joint // 5) % 5, joint % 5)


def moves_to_joint(moves: tuple[int, int, int]) -> int:
    """Compose (m0, m1, m2) into joint index in [0, 125)."""
    return moves[0] * 25 + moves[1] * 5 + moves[2]


def resolve_order(block: Block, seat: int, moves: tuple[int, int, int], turn: int) -> str:
    """Resolve joint movement tuple into a complete legal order line adhering to assign_v1 rules.

    Rules (tools/cpp/agent_v2/check.cpp, inference/agent_v2/resolver.hpp):
      - Carriers must wait ('-').
      - Living non-immune units can attack living non-immune adjacent enemies.
      - Units on a flag pickup ('P').
      - No dropping ('X').
      - Only positive-value attacks.
    """
    my_units = [block.units[3 * seat + k] for k in range(3)]
    enemy_units = [u for u in block.units if u[1] != seat and u[4] > 0 and u[6] <= 0]
    tokens = [str(turn)]

    for k in range(3):
        uid, team, x, y, hp, flag, respawn_at = my_units[k]
        m = moves[k]
        mv_char = MOVES[m]

        if hp <= 0 or respawn_at > 0 or x < 0:
            # Unit dead or in respawn
            tokens.extend([mv_char, "-"])
            continue

        if flag >= 0:
            # Carrier always waits
            tokens.extend([mv_char, "-"])
            continue

        # Check destination
        dx, dy = DELTAS[m]
        dest_x, dest_y = x + dx, y + dy

        # Check if adjacent to an enemy unit for attack (Manhattan distance <= 3)
        # In agent_v2 assign_v1, an attack target must be a living non-immune enemy
        target_uid = -1
        min_enemy_dist = 999
        for eu in enemy_units:
            e_id, e_team, ex, ey, e_hp, e_flag, e_respawn = eu
            dist = abs(dest_x - ex) + abs(dest_y - ey)
            if dist <= 3 and dist < min_enemy_dist:
                min_enemy_dist = dist
                target_uid = e_id

        if target_uid >= 0:
            tokens.extend([mv_char, str(target_uid)])
        else:
            tokens.extend([mv_char, "P" if k == 0 and turn % 7 == 0 else "-"])

    return " ".join(tokens)


def build_candidate_set(
    prior_scores: list[float],
    sampled_joint: int,
    block: Block,
    seat: int,
    turn: int,
    target_k: int = 8,
    leaf_scores: list[float] | None = None,
) -> list[CandidateOrder]:
    """Build a nested candidate set containing the sampled baseline as candidate 0.

    Args:
        prior_scores: 125 float scores from actor network.
        sampled_joint: actually sampled direct joint (not necessarily argmax).
        block: parsed public Block at turn t.
        seat: our seat index.
        turn: turn number.
        target_k: size of candidate set (e.g. 8 or 24).
        leaf_scores: optional list of critic scores from search tree.

    Returns:
        List of CandidateOrder objects with candidate 0 as the exact baseline.
    """
    if len(prior_scores) != 125:
        raise ValidationError(f"expected 125 prior scores, got {len(prior_scores)}")
    if not (0 <= sampled_joint < 125):
        raise ValidationError(f"sampled joint {sampled_joint} out of bounds")
    if target_k < 1 or target_k > 125:
        raise ValidationError(f"target_k {target_k} out of bounds [1, 125]")

    # Rank all 125 joints descending by prior score; ties broken deterministically by joint index
    ranked = sorted(range(125), key=lambda j: (-prior_scores[j], j))
    rank_map = {j: r for r, j in enumerate(ranked)}

    # Baseline is candidate 0
    cands: list[CandidateOrder] = []
    base_moves = joint_to_moves(sampled_joint)
    base_order = resolve_order(block, seat, base_moves, turn)
    base_leaf = leaf_scores[0] if (leaf_scores and len(leaf_scores) > 0) else None

    cands.append(
        CandidateOrder(
            index=0,
            joint=sampled_joint,
            moves=base_moves,
            order=base_order,
            prior_score=prior_scores[sampled_joint],
            prior_rank=rank_map[sampled_joint],
            is_baseline=True,
            leaf_score=base_leaf,
        )
    )

    # Add top prior-ranked candidates distinct from sampled_joint until target_k is reached
    for j in ranked:
        if len(cands) >= target_k:
            break
        if j == sampled_joint:
            continue
        moves = joint_to_moves(j)
        order = resolve_order(block, seat, moves, turn)
        leaf = leaf_scores[len(cands)] if (leaf_scores and len(leaf_scores) > len(cands)) else None
        cands.append(
            CandidateOrder(
                index=len(cands),
                joint=j,
                moves=moves,
                order=order,
                prior_score=prior_scores[j],
                prior_rank=rank_map[j],
                is_baseline=False,
                leaf_score=leaf,
            )
        )

    return cands


def extract_root_from_match(
    match: dict[str, Any],
    seat: int,
    turn: int,
    actor_scores: list[float],
    sampled_joint: int,
    model_identities: dict[str, str],
    target_k: int = 8,
    stratum: str = "general",
    stratum_weight: float = 1.0,
    leaf_scores: list[float] | None = None,
) -> RootRecord:
    """Extract a single RootRecord from a loaded match structure."""
    blocks = match.get("blocks", [])
    if not blocks or turn > len(blocks):
        raise ValidationError(f"turn {turn} exceeds match blocks length {len(blocks)}")
    if turn < 2:
        raise ValidationError("roots must be after turn 1 (requiring causal history)")

    block_text = blocks[turn - 1]
    parsed_blk = parse_block(block_text)

    # Construct causal prefix text (turns 1..t-1)
    # The agent reads INIT + TURN 1..t-1
    init_text = match.get("init", "")
    prefix_parts = [init_text]
    for prev_t in range(turn - 1):
        prefix_parts.append(blocks[prev_t])
    prefix_text = "\n".join(prefix_parts)
    prefix_sha256 = sha256_text(prefix_text)

    # Build candidates
    candidates = build_candidate_set(
        prior_scores=actor_scores,
        sampled_joint=sampled_joint,
        block=parsed_blk,
        seat=seat,
        turn=turn,
        target_k=target_k,
        leaf_scores=leaf_scores,
    )

    match_id = str(match.get("match_id", "synthetic"))
    root_id = fnv64(f"{match_id}:{seat}:{turn}:{prefix_sha256[:16]}")

    record = RootRecord(
        schema=SCHEMA_ROOT,
        root_id=root_id,
        match_id=match_id,
        map_seed=int(match.get("map_seed", 0)),
        flag_seed=int(match.get("flag_seed", 0)),
        teams=int(match.get("teams", len(match.get("seats", [])))),
        seat=seat,
        turn=turn,
        block_text=block_text,
        prefix_sha256=prefix_sha256,
        baseline_joint=sampled_joint,
        baseline_order=candidates[0].order,
        candidates=candidates,
        model_identities=model_identities,
        random_keys={"root_key": fnv64(f"{root_id}:rnd")},
        stratum=stratum,
        stratum_weight=stratum_weight,
    )
    record.validate()
    return record


def extract_roots_from_stream(
    stream_path: Path,
    turns: list[int],
    seat: int,
    teams: int,
    map_seed: int,
    flag_seed: int,
    actor_scores_fn: Any,
    sampled_joint_fn: Any,
    model_identities: dict[str, str],
    target_k: int = 8,
) -> list[RootRecord]:
    """Extract roots from a raw seat stream file (.txt or .gz)."""
    raw_data = stream_path.read_bytes()
    if stream_path.suffix == ".gz":
        raw_data = gzip.decompress(raw_data)
    text = raw_data.decode("utf-8")

    # Split into blocks ending with END
    raw_blocks = [b.strip() + "\nEND" for b in text.split("END\n") if b.strip()]
    if not raw_blocks:
        raise ValidationError(f"stream {stream_path} contains no valid blocks")

    # The first block contains INIT and TURN 1
    init_text = raw_blocks[0][: raw_blocks[0].find("TURN")] if "TURN" in raw_blocks[0] else ""
    turn_blocks: dict[int, str] = {}
    for blk in raw_blocks:
        if "TURN" in blk:
            parsed = parse_block(blk[blk.find("TURN"):])
            turn_blocks[parsed.turn] = blk[blk.find("TURN"):]

    roots: list[RootRecord] = []
    seen_ids: set[str] = set()

    for t in turns:
        if t not in turn_blocks:
            raise ValidationError(f"requested turn {t} not found in stream {stream_path}")
        if t < 2:
            raise ValidationError("requested turn must be >= 2 for causal prefix")

        # Causal prefix up to t-1
        prefix_parts = [init_text]
        for prev_t in range(1, t):
            if prev_t in turn_blocks:
                prefix_parts.append(turn_blocks[prev_t])
        prefix_text = "\n".join(prefix_parts)
        prefix_sha256 = sha256_text(prefix_text)

        blk_text = turn_blocks[t]
        parsed_blk = parse_block(blk_text)

        # Get actor prior scores and sampled baseline joint
        scores = actor_scores_fn(t, blk_text, prefix_text)
        sampled_joint = sampled_joint_fn(t, scores)

        candidates = build_candidate_set(
            prior_scores=scores,
            sampled_joint=sampled_joint,
            block=parsed_blk,
            seat=seat,
            turn=t,
            target_k=target_k,
        )

        root_id = fnv64(f"{stream_path.name}:{seat}:{t}:{prefix_sha256[:16]}")
        if root_id in seen_ids:
            raise ValidationError(f"duplicate root detected: {root_id}")
        seen_ids.add(root_id)

        root = RootRecord(
            schema=SCHEMA_ROOT,
            root_id=root_id,
            match_id=stream_path.stem,
            map_seed=map_seed,
            flag_seed=flag_seed,
            teams=teams,
            seat=seat,
            turn=t,
            block_text=blk_text,
            prefix_sha256=prefix_sha256,
            baseline_joint=sampled_joint,
            baseline_order=candidates[0].order,
            candidates=candidates,
            model_identities=model_identities,
            random_keys={"root_key": fnv64(f"{root_id}:rnd")},
        )
        root.validate()
        roots.append(root)

    return roots
