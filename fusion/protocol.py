"""Light parser for TURN blocks and reply lines.

Labels only need scores, unit positions and events, so this stays small.
Model features are never computed here; they come from inference/features.hpp
through the extractor, the same code the bot runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MOVES = "SUDLR"                     # index order of inference/state.hpp
MOVE_INDEX = {c: i for i, c in enumerate(MOVES)}
DELTAS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
WAIT, PICKUP, DROP, ATTACK = range(4)


@dataclass
class Block:
    turn: int
    score: list[int]
    units: list[tuple[int, int, int, int, int, int, int]]   # id, team, x, y, hp, flag, respawn_at
    events: list[tuple] = field(default_factory=list)


def parse_block(text: str) -> Block:
    lines = text.split("\n")
    turn = int(lines[0].split()[1])
    score = [int(v) for v in lines[1].split()[1:]]
    n_units = int(lines[2].split()[1])
    units = [tuple(int(v) for v in line.split()[1:8]) for line in lines[3:3 + n_units]]
    i = 3 + n_units
    n_flags = int(lines[i].split()[1])
    i += 1 + n_flags
    n_events = int(lines[i].split()[1])
    events = []
    for line in lines[i + 1:i + 1 + n_events]:
        tok = line.split()
        kind = tok[1]
        if kind == "death":
            events.append(("death", int(tok[2])))
        elif kind == "capture":
            events.append(("capture", int(tok[2]), int(tok[3])))
        elif kind == "attack":
            events.append(("attack", int(tok[2]), int(tok[3])))
        else:
            events.append((kind, *[int(v) for v in tok[2:]]))
    return Block(turn, score, units, events)


@dataclass
class Request:
    move: int      # index into MOVES
    act: int       # WAIT, PICKUP, DROP or ATTACK
    target: int    # unit id for ATTACK, else -1


def parse_reply(line: str | None, turn: int) -> list[Request] | None:
    """Mirror of engine Game.decode. None when the engine would reject the line."""
    if line is None:
        return None
    body = line.partition("#")[0].split()
    if len(body) != 7 or body[0] != str(turn):
        return None
    out = []
    for k in range(3):
        mv, act = body[1 + 2 * k], body[2 + 2 * k]
        if mv not in MOVE_INDEX:
            return None
        if act == "-":
            out.append(Request(MOVE_INDEX[mv], WAIT, -1))
        elif act == "P":
            out.append(Request(MOVE_INDEX[mv], PICKUP, -1))
        elif act == "X":
            out.append(Request(MOVE_INDEX[mv], DROP, -1))
        elif act.isdigit():
            try:                      # the engine: str.isdigit, then int(); "²" passes isdigit but int() rejects it
                out.append(Request(MOVE_INDEX[mv], ATTACK, int(act)))
            except ValueError:
                return None
        else:
            return None
    return out
