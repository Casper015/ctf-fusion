"""Faithful branch restoration and continuation harness (T0085).

Preserves:
  - Official engine.py Game state INCLUDING hidden RNG and map/flag seeds for evaluator use.
  - Full causal prefix replay into live bot processes (reconstructing persistent state, memory, caches).
  - Submitted opponent commands at root turn t are held COMMON across all candidate branches.
  - From turn t+1 to terminal, all bots are live and reactive to the counterfactual branch state.
  - Factual future opponent replies are NEVER reused as counterfactual ground truth.
  - Complete episode rollouts to terminal pairwise rank utility U_T.
  - Fault tracking by phase: prefix, root-turn discarded, and continuation.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

from fusion import matchlog
from fusion.labels import pairwise_outcome
from fusion.paths import REPO
from fusion.protocol import parse_block, parse_reply
from tools.search_diagnostics.schemas import (
    BranchResult,
    CandidateOrder,
    RootRecord,
    SCHEMA_BRANCH,
    ValidationError,
    fnv64,
)

LIMITS = (2.0, 0.05)
PREFIX_LIMIT = 5.0


class PrefixDivergenceError(ValidationError):
    """Raised when replayed prefix diverges from recorded public state or legal orders."""


def recorded_orders(game: Any, match: dict[str, Any], turn: int) -> dict[int, Any]:
    """Decode recorded orders for a given turn from match replies."""
    orders: dict[int, Any] = {}
    replies = match.get("replies", [])
    if turn - 1 >= len(replies):
        return orders
    turn_replies = replies[turn - 1]
    for i, line in enumerate(turn_replies):
        if line is not None:
            try:
                orders[i] = game.decode(i, line)
            except (ValueError, KeyError):
                orders[i] = {}
        else:
            orders[i] = {}
    return orders


def play_branch(
    engine: Any,
    match: dict[str, Any],
    seat: int,
    turn: int,
    candidate_order: str,
    bot_commands: list[list[str]],
    bot_names: list[str] | None = None,
    limits: tuple[float, float] = LIMITS,
    prefix_limit: float = PREFIX_LIMIT,
    shared_opponent_replies: list[str | None] | None = None,
    bot_factory: Callable[..., Any] | None = None,
) -> tuple[BranchResult, list[str | None]]:
    """Execute a faithful branch continuation rollout from root turn t to terminal.

    Args:
        engine: official engine module (from fusion.paths.load_engine).
        match: source match dict (must contain teams, map_seed, flag_seed, blocks, replies).
        seat: our seat index.
        turn: root turn t.
        candidate_order: candidate order line to execute for our seat at turn t.
        bot_commands: list of command arg-lists for each seat.
        bot_names: optional names for each seat.
        limits: (first_limit_s, turn_limit_s).
        prefix_limit: limit during prefix replay.
        shared_opponent_replies: opponent replies held common at root turn t.
        bot_factory: optional factory to construct bots (defaults to fusion.runner.Bot).

    Returns:
        (BranchResult, turn_t_opponent_replies)
    """
    from fusion import runner  # noqa: PLC0415

    teams = int(match.get("teams", len(bot_commands)))
    map_seed = int(match["map_seed"])
    flag_seed = int(match.get("flag_seed", 0))

    # 1. Restore official engine including hidden state
    game = engine.Game(teams, map_seed, flag_seed=flag_seed)

    names = bot_names or [s.get("name", f"seat{i}") for i, s in enumerate(match.get("seats", []))]
    if len(names) < teams:
        names = [f"seat{i}" for i in range(teams)]

    make_bot = bot_factory or (lambda cmd, name, lim: runner.Bot(cmd, name, lim))
    bots = [make_bot(cmd, name, (limits[0], prefix_limit)) for cmd, name in zip(bot_commands, names)]

    t0_start = time.perf_counter()
    prefix_faults = 0
    continuation_faults = 0
    prefix_mismatches = 0
    root_faults: list[tuple[int, int, str]] = []
    root_computed_replies: list[str | None] = []
    effective_root_replies: list[str | None] = []

    try:
        while not game.done:
            t = game.turn + 1
            inputs = [game.encode(i) for i in range(teams)]

            if t == turn:
                # Switch bots to official continuation limit (50 ms)
                for b in bots:
                    b.turn_limit = limits[1]
                    b.strikes = 0

            # Query all bots
            starts = [b.send(text) for b, text in zip(bots, inputs)]
            lines = [b.receive(t0, t) for b, t0 in zip(bots, starts)]

            if t < turn:
                # Causal prefix replay phase
                rec_lines = match.get("replies", [])[t - 1] if t - 1 < len(match.get("replies", [])) else []
                for i, (got, rec) in enumerate(zip(lines, rec_lines)):
                    if got != rec:
                        prefix_mismatches += 1
                for b in bots:
                    if getattr(b, "faults", None):
                        prefix_faults += sum(1 for ft, _ in b.faults if ft == t)
                # Engine steps with the factual recorded orders to ensure faithful state progression
                orders = recorded_orders(game, match, t)
                game.step(orders)

            elif t == turn:
                # Root turn branching phase
                root_computed_replies = list(lines)
                for i, (b, got) in enumerate(zip(bots, lines)):
                    reasons = [r for tt, r in getattr(b, "faults", []) if tt == t]
                    if got is None and not reasons:
                        reasons.append("no reply (dead)" if getattr(b, "dead", False) else "no reply")
                    if got is not None:
                        try:
                            game.decode(i, got)
                        except (ValueError, KeyError) as exc:
                            reasons.append(f"malformed output: {exc}")
                    for r in reasons:
                        root_faults.append((t, i, r))

                # Opponent commands at turn t: held common across all candidate branches!
                if shared_opponent_replies is not None:
                    branch_lines = list(shared_opponent_replies)
                else:
                    # If not externally supplied, hold what the opponent bots just computed
                    branch_lines = list(lines)

                # Ensure learner seat plays the counterfactual candidate order
                branch_lines[seat] = candidate_order
                effective_root_replies = list(branch_lines)

                orders = {}
                for i, (b, ln) in enumerate(zip(bots, branch_lines)):
                    if ln is None:
                        orders[i] = {}
                        continue
                    try:
                        orders[i] = game.decode(i, ln)
                        if hasattr(b, "ok"):
                            b.ok()
                    except (ValueError, KeyError) as exc:
                        orders[i] = {}
                        if hasattr(b, "strike"):
                            b.strike(t, f"malformed root output: {exc}")

                game.step(orders)

            else:
                # Live continuation phase (t > turn)
                # All bots observe and react to the counterfactual branch state
                orders = {}
                for i, (b, ln) in enumerate(zip(bots, lines)):
                    if ln is None:
                        orders[i] = {}
                        continue
                    try:
                        orders[i] = game.decode(i, ln)
                        if hasattr(b, "ok"):
                            b.ok()
                    except (ValueError, KeyError) as exc:
                        orders[i] = {}
                        if hasattr(b, "strike"):
                            b.strike(t, f"malformed continuation output: {exc}")
                        continuation_faults += 1

                for b in bots:
                    if getattr(b, "faults", None):
                        continuation_faults += sum(1 for ft, _ in b.faults if ft == t)

                game.step(orders)

        raw_score = list(game.result()["score"])
        utility = float(pairwise_outcome(raw_score)[seat])

    finally:
        for b in bots:
            if hasattr(b, "close"):
                b.close()

    elapsed_ms = (time.perf_counter() - t0_start) * 1000.0
    match_id = str(match.get("match_id", "synthetic"))
    root_id = fnv64(f"{match_id}:{seat}:{turn}")
    repeat_key = fnv64(f"{root_id}:{time.time_ns()}")

    result = BranchResult(
        schema=SCHEMA_BRANCH,
        root_id=root_id,
        candidate_index=0,  # caller sets candidate_index
        repeat_id=0,        # caller sets repeat_id
        repeat_key=repeat_key,
        final_score=raw_score,
        pairwise_utility=utility,
        faults_prefix=prefix_faults,
        faults_root_discarded=len(root_faults),
        faults_continuation=continuation_faults,
        latency_ms=elapsed_ms,
        prefix_mismatch_count=prefix_mismatches,
    )
    return result, effective_root_replies


def run_faithful_root_rollouts(
    engine: Any,
    match: dict[str, Any],
    root: RootRecord,
    bot_commands: list[list[str]],
    repeats: int = 4,
    bot_factory: Callable[..., Any] | None = None,
) -> list[BranchResult]:
    """Execute complete rollouts for all candidates of a root across shared repeats.

    Guarantees:
      - Candidate 0 (baseline) and candidates 1..K-1 share the exact same root turn t opponent commands on repeat r.
      - Each repeat r evaluates the baseline and all candidates.
      - All results are validated and input-bound.
    """
    all_results: list[BranchResult] = []

    for r in range(repeats):
        # 1. Roll out Candidate 0 (the direct baseline order) first
        base_cand = root.candidates[0]
        base_res, shared_opp = play_branch(
            engine=engine,
            match=match,
            seat=root.seat,
            turn=root.turn,
            candidate_order=base_cand.order,
            bot_commands=bot_commands,
            shared_opponent_replies=None,
            bot_factory=bot_factory,
        )
        base_res.candidate_index = 0
        base_res.repeat_id = r
        base_res.root_id = root.root_id
        base_res.validate()
        all_results.append(base_res)

        # 2. Roll out alternative candidates holding opponent replies common at turn t
        for c_idx in range(1, len(root.candidates)):
            cand = root.candidates[c_idx]
            cand_res, _ = play_branch(
                engine=engine,
                match=match,
                seat=root.seat,
                turn=root.turn,
                candidate_order=cand.order,
                bot_commands=bot_commands,
                shared_opponent_replies=shared_opp,
                bot_factory=bot_factory,
            )
            cand_res.candidate_index = c_idx
            cand_res.repeat_id = r
            cand_res.root_id = root.root_id
            cand_res.validate()
            all_results.append(cand_res)

    return all_results
