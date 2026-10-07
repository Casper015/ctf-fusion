"""Labels for one match, aligned row for row with its extractor records.

Policy labels are the *requested* actions from the reply lines, never inferred
from position changes, so a move stopped by a collision keeps its requested
direction (`blocked` marks those rows for diagnostics). Requests the engine
would turn into no-ops are canonicalized to their effect: a move into a wall
is a stay, a pickup off a flag or a drop without a flag is a wait, and an
attack on a unit that cannot be hit this turn is a wait. A missing or
malformed reply (timeout, parse error) gives no policy label at all.

Value targets come from the end of the match: the pairwise outcome against
every other team (win 1, draw .5, loss 0, averaged), the target the
Bradley-Terry rating rewards; plus two auxiliary horizons.
"""
from __future__ import annotations

import numpy as np

from .protocol import ATTACK, WAIT, parse_block, parse_reply
from .schema import Layout

STRATA = ("flag", "contact", "respawn", "quiet")
CAPTURE_HORIZON = 20
DEATH_HORIZON = 10


def pairwise_outcome(score: list[int]) -> np.ndarray:
    s = np.asarray(score, dtype=np.float64)
    n = len(s)
    wins = (s[:, None] > s[None, :]).sum(1) + 0.5 * ((s[:, None] == s[None, :]).sum(1) - 1)
    return wins / (n - 1)


def _window_any(flags: np.ndarray, horizon: int) -> np.ndarray:
    """flags[s, team] per settlement s (0-based = turn s+1); any() over s..s+horizon-1."""
    c = np.vstack([np.zeros((1, flags.shape[1]), np.int64), np.cumsum(flags, 0)])
    s = np.arange(flags.shape[0])
    end = np.minimum(s + horizon, flags.shape[0])
    return (c[end] - c[s]) > 0


def label_match(match: dict, rec: np.ndarray, layout: Layout) -> dict[str, np.ndarray]:
    n = match["teams"]
    turns = len(match["blocks"])
    rows = rec.shape[0]
    if rows != turns * n:
        raise ValueError(f"{match['match_id']}: {rows} records for {turns} turns x {n} teams")
    blocks = [parse_block(b) for b in match["blocks"]] + [parse_block(match["final_block"])]
    size = int(match["init"].split("\n", 1)[0].split()[3])
    turn_col, team_col = rec[:, 0].astype(np.int64), rec[:, 1].astype(np.int64)
    if not (np.array_equal(turn_col, np.repeat(np.arange(1, turns + 1), n)) and
            np.array_equal(team_col, np.tile(np.arange(n), turns))):
        raise ValueError(f"{match['match_id']}: records are not turn-major, team-minor")

    legal = layout.get(rec, "legal").astype(bool)
    act_legal = layout.get(rec, "act_legal").astype(bool)
    origin = layout.get(rec, "origin").astype(np.int64)
    cand = layout.get(rec, "cand").astype(np.int64)

    y_move = np.zeros((rows, 3), np.int8)
    y_act = np.full((rows, 3), -1, np.int8)
    y_tgt = np.full((rows, 3), -1, np.int8)
    blocked = np.zeros((rows, 3), bool)
    valid = np.zeros(rows, bool)
    for r in range(rows):
        t, team = int(turn_col[r]), int(team_col[r])
        reqs = parse_reply(match["replies"][t - 1][team], t)
        if reqs is None:
            continue
        valid[r] = True
        units = blocks[t - 1].units
        nxt = blocks[t].units
        for k, req in enumerate(reqs):
            m = req.move if legal[r, k, req.move] else 0
            y_move[r, k] = m
            o = origin[r, k]
            if o < 0:
                continue
            uid = 3 * team + k
            if m and nxt[uid][2] >= 0 and nxt[uid][3] * size + nxt[uid][2] == o:
                blocked[r, k] = True
            a = req.act if act_legal[r, k, m, req.act] else WAIT
            if a == ATTACK:
                slots = list(cand[r, k])
                if req.target in slots:
                    y_tgt[r, k] = slots.index(req.target)
                elif not _hittable_beyond_cap(units, req.target, team, o, size):
                    a = WAIT
            y_act[r, k] = a
    y_joint = np.where(valid, y_move[:, 0].astype(np.int16) * 25 + y_move[:, 1] * 5 + y_move[:, 2], -1).astype(np.int16)

    final = blocks[-1].score
    outcome = pairwise_outcome(final)
    captures = np.zeros((turns, n), bool)
    deaths = np.zeros((turns, n), bool)
    for s in range(turns):
        for e in blocks[s + 1].events:
            if e[0] == "capture":
                captures[s, e[2]] = True
            elif e[0] == "death":
                deaths[s, e[1] // 3] = True
    cap_h = _window_any(captures, CAPTURE_HORIZON).reshape(-1)
    death_h = _window_any(deaths, DEATH_HORIZON).reshape(-1)
    v_y = np.stack([np.tile(outcome, turns), cap_h, death_h], 1).astype(np.float32)

    return {"y_move": y_move, "y_joint": y_joint, "y_act": y_act, "y_tgt": y_tgt, "blocked": blocked,
            "valid": valid, "v_y": v_y, "stratum": strata(rec, layout)}


def _hittable_beyond_cap(units, target: int, team: int, origin: int, size: int) -> bool:
    """A target missing from the candidate slots only because the slots were full."""
    if not 0 <= target < len(units):
        return False
    _, t_team, x, y, *_ = units[target]
    if t_team == team or x < 0:
        return False
    return abs(x - origin % size) + abs(y - origin // size) <= 4


def strata(rec: np.ndarray, layout: Layout) -> np.ndarray:
    u = layout.get(rec, "u")
    names = layout.names("unit")
    col = {name: i for i, name in enumerate(names)}
    v = layout.get(rec, "v")
    agg = {name: len(v[0]) - len(layout.names("value_agg")) + i for i, name in enumerate(layout.names("value_agg"))}
    active = u[..., col["active"]] > 0
    flag = ((u[..., col["carrying"]] > 0).any(1) | (active & (u[..., col["flag_dist"]] <= 2)).any(1)
            | (v[:, agg["escort_threat"]] > 0) | (v[:, agg["carrier_shots"]] > 0))
    contact = (active & (u[..., col["enemies_r3"]] > 0)).any(1)
    respawn = (~active | (u[..., col["immune"]] > 0)).any(1)
    out = np.full(len(rec), 3, np.int8)
    out[respawn] = 2
    out[contact] = 1
    out[flag] = 0
    return out
