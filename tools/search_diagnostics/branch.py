"""Faithful branch continuations from a root (task T0085, protocol faithful-replay-v1).

One branch plays candidate c of root (match, seat, t) in repeat k to the end of the game:

  1. the official engine is rebuilt from the match's map and flag seeds; every bot of the match is a fresh process
     (our seat: the package's continuation bot with the repeat's actor key from turn t + 1; the opponents: their
     recorded programs, whose hashes must equal the match's);
  2. turns 1 .. t - 1 replay the record: every reproduced TURN block must equal the recorded one, every bot is fed
     exactly what it saw, every bot's reply must equal its recorded reply (by order), and the engine steps the
     recorded orders. Any difference refuses the branch with the reason: a process whose replies diverge has a
     different private state than in the source game, and no branch from it is called faithful;
  3. turn t: the block must again equal the record; every bot answers under the official limit, and every answer
     must again equal the record (our seat's is the baseline order). The engine's flag-refresh RNG is reseeded with
     the repeat's environment key, then it steps the opponents' recorded orders and our candidate order. The
     opponents' submitted commands are therefore common to every candidate, and every candidate of a repeat meets
     the same refresh draws;
  4. turns t + 1 .. 400 are live under the official limits: every bot observes the branch and answers; nothing
     factual is reused. The outcome is our seat's terminal pairwise utility (fusion.labels.pairwise_outcome).

Our continuation process forced to play candidate c at t keeps no state of its own computed order (agent_v2's
History holds observed states only; inference/agent_v2/core.hpp), so its memory after t is the branch's. Opponent
processes answered the root exactly as recorded and played it, so theirs is too. Opponent randomness after t is
their own and is not controlled; each row says so.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from fusion import matchlog, runner
from fusion.labels import pairwise_outcome
from tools.search_diagnostics.extract import order_of
from tools.search_diagnostics.schemas import PROTOCOL

LIMITS = (2.0, 0.05)
PREFIX_LIMIT = 5.0     # seconds per reply while a bot replays the recorded prefix (no timing is measured there)


class BranchRefused(Exception):
    """The branch cannot be played faithfully; the message is the reason."""


def _decode(game, i: int, line: str | None) -> dict:
    if line is None:
        return {}
    try:
        return game.decode(i, line)
    except ValueError:
        return {}


def play_branch(engine, match: dict, root: dict, candidate: int, env_key: int, actor_key: int,
                commands: list[list[str]], limits: tuple[float, float] = LIMITS,
                bot_factory: Callable[..., object] = runner.Bot) -> dict:
    """One branch result (status complete), or BranchRefused."""
    seat, turn = root["source"]["seat"], root["turn"]
    order = root["candidates"][candidate]["order"]
    if order_of(order) is None or not order.startswith(f"{turn} "):
        raise BranchRefused(f"candidate {candidate} is not an order for turn {turn}")
    cmds = [list(c) for c in commands]
    cmds[seat] = cmds[seat] + [str(actor_key), str(turn + 1)]
    names = [s["name"] for s in match["seats"]]
    game = engine.Game(match["teams"], match["map_seed"], flag_seed=match["flag_seed"])
    bots = [bot_factory(c, n, (limits[0], PREFIX_LIMIT)) for c, n in zip(cmds, names)]
    faults: dict[str, list] = {"prefix": [], "root": [], "continuation": []}
    started = time.perf_counter()
    try:
        while not game.done:
            t = game.turn + 1
            inputs = [game.encode(i) for i in range(match["teams"])]
            if t <= turn:
                block = matchlog.split_init(inputs[0])[1] if t == 1 else inputs[0]
                if block != match["blocks"][t - 1]:
                    raise BranchRefused(f"the engine's turn-{t} block differs from the recorded one")
            if t == turn:
                for b in bots:
                    b.turn_limit = limits[1]
                    b.strikes = 0
            sent = [b.send(x) for b, x in zip(bots, inputs)]
            lines = [b.receive(s, t) for b, s in zip(bots, sent)]
            phase = "prefix" if t < turn else "root" if t == turn else "continuation"
            faults[phase] += [[t, i, r] for i, b in enumerate(bots) for tt, r in b.faults if tt == t]
            if t <= turn:
                if faults[phase]:
                    raise BranchRefused(f"{phase} fault at turn {t}: {faults[phase][0]}")
                for i, (got, want) in enumerate(zip(lines, match["replies"][t - 1])):
                    if want is None:
                        raise BranchRefused(f"seat {i}'s recorded reply at turn {t} is missing (a recorded fault)")
                    if order_of(got) != order_of(want):
                        raise BranchRefused(f"seat {i} ({names[i]}) diverged from the record at turn {t}"
                                            f"{' (root)' if t == turn else ''}: {want!r} recorded, {got!r} replayed")
                lines = list(match["replies"][t - 1])
                if t == turn:
                    lines[seat] = order
                    game.flag_rng.seed(env_key)
            orders = {}
            for i, (b, line) in enumerate(zip(bots, lines)):
                orders[i] = _decode(game, i, line)
                if t > turn and line is not None:
                    if orders[i]:
                        b.ok()
                    else:
                        b.strike(t, "malformed output")
                        faults["continuation"].append([t, i, "malformed output"])
            game.step(orders)
        result = game.result()
    finally:
        for b in bots:
            b.close()
    score = list(result["score"])
    return {
        "candidate": candidate, "status": "complete", "protocol": PROTOCOL, "env_key": env_key, "actor_key": actor_key,
        "final_score": score, "utility": float(pairwise_outcome(score)[seat]), "final_turn": result["turns"],
        "terminal": bool(game.done), "faults": faults, "wall_s": round(time.perf_counter() - started, 3),
        "opponent_randomness": "uncontrolled (unmodified opponent processes)",
    }


def play_root(engine, match: dict, root: dict, candidates: list[int], keys: list[dict], commands: list[list[str]],
              limits: tuple[float, float] = LIMITS, bot_factory: Callable[..., object] = runner.Bot) -> list[dict]:
    """Every (candidate, repeat) branch of a root; refused branches are kept with their reasons."""
    out = []
    for k in keys:
        for c in candidates:
            base = {"candidate": c, "repeat": k["repeat"], "env_key": k["env_key"], "actor_key": k["actor_key"]}
            try:
                b = play_branch(engine, match, root, c, k["env_key"], k["actor_key"], commands, limits, bot_factory)
                out.append({**b, **base})
            except BranchRefused as e:
                out.append({**base, "status": "refused", "reason": str(e)})
    return out


def commands_for(match: dict, seat: int, continuation_bot: Path, ai_test: Path, build_dir: Path) -> list[list[str]]:
    """Every seat's command: our seat the continuation bot, each opponent its pool program, whose SHA-256 must equal
    the one the match recorded (a changed opponent is refused, never substituted)."""
    from fusion.paths import bot_path, load_pool, sha256  # noqa: PLC0415

    pool = {e["name"]: e for e in load_pool()["bots"]}
    out = []
    for i, s in enumerate(match["seats"]):
        if i == seat:
            out.append([str(continuation_bot)])
            continue
        if s["name"] not in pool:
            raise BranchRefused(f"seat {i} ({s['name']}) is not in the pool")
        path = bot_path(pool[s["name"]], ai_test)
        if sha256(path) != s["sha256"]:
            raise BranchRefused(f"seat {i} ({s['name']}) changed since the match was played")
        out.append(runner.command_for(path, build_dir))
    return out
