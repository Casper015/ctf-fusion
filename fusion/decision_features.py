"""Our own complete orders at a decision root: canonical identity, support audit and scorer inputs (task T0034).

P predicts the opponents; the own-action scorer (training/train_decision_ranker.py) ranks complete orders of
our own team at a root. Everything here is a function of the *root record* (the extractor row of our seat at
turn t, extracted from the recorded prefix INIT + TURN 1..t, docs/RL_PLAN.md section 3) and of a candidate's
reply line. Nothing reads an outcome, a later block, an opponent's turn-t reply or a realized successor: the
signatures take no such argument, so a score cannot depend on information unknown at decision time.

Canonical orders follow fusion.labels.label_match (the convention P's labels use), with every change from the
requested order kept as a flag instead of silently dropped:

  move    a requested move that the root's `legal` mask forbids is the stay it settles as (REQ_MOVE_ILLEGAL);
  action  an action `act_legal` forbids after that move is a wait (ACT_CANONICALIZED);
  attack  a target in the eight candidate slots keeps its slot; a target hittable only beyond the slots is an
          attack *without* target features (TARGET_OUT_OF_CAP): unavailable to the scorer, never a zero-feature
          success; an unhittable target is a wait (TARGET_INVALID);
  unit    a unit that is off the board (origin < 0) has no action; its tokens do not distinguish candidates.

Two requested orders with the same canonical key are the same effective action (DUPLICATE_EFFECTIVE).
Candidate support for scoring = parsed, every target representable, and the first of its effective
duplicates (the incumbent first). FEATURE_VERSION and feature_identity() bind a scorer to exactly the groups
and shapes it reads, in the manner of fusion/families.py.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .labels import _hittable_beyond_cap
from .protocol import ATTACK, MOVES, WAIT, parse_reply
from .schema import Layout

FEATURE_VERSION = "own-action-1"
KIND = "own_action_scorer"      # run kind of every model trained on these features; never P, never V

# cand_flags bits
PARSE_FAIL = 1
REQ_MOVE_ILLEGAL = 2
ACT_CANONICALIZED = 4
TARGET_OUT_OF_CAP = 8
TARGET_INVALID = 16
DUPLICATE_EFFECTIVE = 32
CANON_CHANGED = 64
FLAG_NAMES = {PARSE_FAIL: "parse_fail", REQ_MOVE_ILLEGAL: "requested_move_illegal", ACT_CANONICALIZED: "action_canonicalized",
              TARGET_OUT_OF_CAP: "target_out_of_cap", TARGET_INVALID: "target_invalid",
              DUPLICATE_EFFECTIVE: "duplicate_effective", CANON_CHANGED: "canonical_differs_from_request"}
UNSUPPORTED = PARSE_FAIL | TARGET_OUT_OF_CAP | DUPLICATE_EFFECTIVE

NO_ACTION = -1   # canon action of a unit off the board
NO_TARGET = -1   # canon target: none
_ACT_CODE = "-PXA"

_GROUPS = ("global", "unit", "move", "combat", "cand_f", "value_unit", "value_agg", "value_ext")
_BLOCKS = ("u", "m", "legal", "act_legal", "origin", "cand", "cand_f", "pc", "v", "vx")
PER_UNIT = 5 + 4 + 11 + 15 + 11 + 2   # move, action, move features, combat, target features, flags


def require_layout(layout: Layout) -> None:
    missing = [b for b in _BLOCKS if not layout.has(b)] + [g for g in _GROUPS if g not in layout.specs]
    if missing:
        raise ValueError(f"the extractor layout lacks {missing}: own-action features need layout 3 (p3 combat, v3)")


def feature_identity(layout: Layout) -> dict:
    """What a scorer reads: the version, group specs and block shapes (not offsets), as a digest."""
    require_layout(layout)
    d = {"version": FEATURE_VERSION, "specs": {g: layout.specs[g] for g in _GROUPS},
         "blocks": {b: list(layout.blocks[b][1]) for b in _BLOCKS}, "per_unit": PER_UNIT}
    digest = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    # "family" names no P or V family, so fusion.families refuses it wherever a P or V family is expected.
    return {"family": FEATURE_VERSION, "version": FEATURE_VERSION, "digest": digest, "layout_version": layout.layout_version}


def normalize_line(line: str | None) -> str | None:
    """A reply line without comment and with single spaces (how branch files store candidates)."""
    if line is None:
        return None
    return " ".join(line.partition("#")[0].split())


def check_root_record(rec: np.ndarray, turn: int, seat: int) -> None:
    if rec.ndim != 1 or int(rec[0]) != turn or int(rec[1]) != seat:
        raise ValueError(f"root record is for turn {int(rec[0])} team {int(rec[1])}, expected turn {turn} team {seat}")


def canonicalize(rec: np.ndarray, layout: Layout, line: str | None, turn: int, seat: int, units, size: int) -> dict:
    """Canonical complete order of a candidate reply at the root.

    units: the root TURN block's unit tuples (fusion.protocol.parse_block(...).units), for the out-of-cap test;
    size: the board size. Returns {key, move (3,), act (3,), target (3,) unit id, slot (3,), flags}.
    """
    check_root_record(rec, turn, seat)
    move = np.zeros(3, np.int16)
    act = np.full(3, NO_ACTION, np.int16)
    target = np.full(3, NO_TARGET, np.int16)
    slot = np.full(3, NO_TARGET, np.int16)
    reqs = parse_reply(line, turn)
    if reqs is None:
        return {"key": "invalid", "move": move, "act": act, "target": target, "slot": slot, "flags": PARSE_FAIL}
    legal = layout.get(rec, "legal").astype(bool)
    act_legal = layout.get(rec, "act_legal").astype(bool)
    origin = layout.get(rec, "origin").astype(np.int64)
    cand = layout.get(rec, "cand").astype(np.int64)
    flags = 0
    parts = []
    for k, req in enumerate(reqs):
        if origin[k] < 0:      # off the board: no action, whatever was written
            parts.append("x")
            continue
        m = req.move if legal[k, req.move] else 0
        if m != req.move:
            flags |= REQ_MOVE_ILLEGAL
        a = req.act if act_legal[k, m, req.act] else WAIT
        if a != req.act:
            flags |= ACT_CANONICALIZED
        t = NO_TARGET
        if a == ATTACK:
            slots = list(cand[k])
            if req.target in slots:
                slot[k] = slots.index(req.target)
                t = req.target
            elif _hittable_beyond_cap(units, req.target, seat, int(origin[k]), size):
                flags |= TARGET_OUT_OF_CAP
                t = req.target
            else:
                flags |= TARGET_INVALID
                a = WAIT
        move[k], act[k], target[k] = m, a, t
        parts.append(f"{MOVES[m]}{_ACT_CODE[a]}{t if a == ATTACK else ''}")
    if flags & (REQ_MOVE_ILLEGAL | ACT_CANONICALIZED | TARGET_INVALID):
        flags |= CANON_CHANGED
    return {"key": "|".join(parts), "move": move, "act": act, "target": target, "slot": slot, "flags": flags}


def mark_duplicates(keys: list[str], flags: list[int], incumbent: int = 0) -> list[int]:
    """dup_of per candidate (-1 unique or first); the incumbent is always the representative of its key."""
    first: dict[str, int] = {}
    order = [incumbent] + [i for i in range(len(keys)) if i != incumbent]
    dup = [-1] * len(keys)
    for i in order:
        if flags[i] & PARSE_FAIL:
            continue
        if keys[i] in first:
            dup[i] = first[keys[i]]
            flags[i] |= DUPLICATE_EFFECTIVE
        else:
            first[keys[i]] = i
    return dup


def supported(flags: np.ndarray) -> np.ndarray:
    return (np.asarray(flags) & UNSUPPORTED) == 0


def root_features(rec: np.ndarray, layout: Layout) -> np.ndarray:
    """State part of the scorer input: our three units (unit group), then the v3 value inputs, scaled."""
    require_layout(layout)
    u = layout.get(rec, "u").reshape(*rec.shape[:-1], -1) / np.tile(layout.scale("unit"), 3)
    v = np.concatenate([layout.get(rec, "v"), layout.get(rec, "vx")], axis=-1) / layout.value_scale("v3")
    return np.concatenate([u, v], axis=-1).astype(np.float32)


def candidate_features(rec: np.ndarray, layout: Layout, move: np.ndarray, act: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """Action part of the scorer input for one candidate: per unit, one-hot move and action, the move's
    features and p3 combat block, the chosen target slot's features, and (target present, unit active)."""
    require_layout(layout)
    m_f = layout.get(rec, "m") / layout.scale("move")
    pc = layout.get(rec, "pc") / layout.scale("combat")
    cf = layout.get(rec, "cand_f") / layout.scale("cand_f")
    out = np.zeros((3, PER_UNIT), np.float32)
    for k in range(3):
        if act[k] == NO_ACTION:
            continue
        mv, a = int(move[k]), int(act[k])
        row = out[k]
        row[mv] = 1.0
        row[5 + a] = 1.0
        row[9:20] = m_f[k, mv]
        row[20:35] = pc[k, mv]
        if slot[k] >= 0:
            row[35:46] = cf[k, int(slot[k])]
            row[46] = 1.0
        row[47] = 1.0
    return out.reshape(-1)


def root_dim(layout: Layout) -> int:
    return 3 * len(layout.specs["unit"]) + len(layout.value_scale("v3"))


CAND_DIM = 3 * PER_UNIT
