"""Freeze a probe suite: recorded seat streams every candidate bot replays (roster.fingerprint).

    python -m roster.probes --raw data/raw --per-count 3 --out data/roster/probe_suite.json

A probe is one seat of one recorded match: the bot receives the exact stream
that seat received, INIT through turn T, so it always sees a valid, complete
history, never an isolated state. Every bot answers the same probes, so their
requested moves, actions and targets are directly comparable state by state.
(A probe history was produced by other bots' orders, not the replaying bot's;
that is the price of identical states. On-policy behavior comes from the
strength screen's live matches.)

Each probe turn carries pre-registered event tags (roster.common.TAGS),
computed from the shared C++ features of the probing team: the four strata
(quiet, contact, flag, respawn) plus pickup options, own carriers, escorts,
interceptions, focus fire, overkill risk, collision risk, flag-return risk and
the endgame. roster.distance weighs tags equally, so quiet turns cannot
dominate a fingerprint, and reports each tag's coverage.

Selection is deterministic: per team count, the matches with the most
distinct seat families (ties by a hash of the match id), distinct maps, and
in each the seat with the most tagged (non-quiet) turns. The suite records
every source file's sha256; replays refuse a changed file.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from fusion import matchlog, native
from fusion.labels import strata
from fusion.paths import REPO, TEAM_COUNTS
from fusion.schema import layout as feature_layout
from roster.common import SCHEMA_SUITE, TAGS, save_json, sha256_file

KILL_HP = 34  # one hit (engine damage); a candidate at or below it dies to any single shot
TAG_RULES = "tags-1"  # version of turn_tags(); a change re-freezes the suite (it enters the suite identity)


def truncate(match: dict, turns: int) -> dict:
    """The match as if it ended after `turns` turns: blocks[:turns], the next block as the final one."""
    if turns >= len(match["blocks"]):
        return match
    return match | {"blocks": match["blocks"][:turns], "final_block": match["blocks"][turns],
                    "replies": match["replies"][:turns]}


def turn_tags(rec: np.ndarray, layout) -> np.ndarray:
    """Tag bitmask per record row (TAGS order) from the extractor's features."""
    bit = {t: 1 << i for i, t in enumerate(TAGS)}
    g, u, m = layout.get(rec, "g"), layout.get(rec, "u"), layout.get(rec, "m")
    cf, cand, act = layout.get(rec, "cand_f"), layout.get(rec, "cand"), layout.get(rec, "act_legal")
    v = layout.get(rec, "v")
    gc = {n: i for i, n in enumerate(layout.names("global"))}
    uc = {n: i for i, n in enumerate(layout.names("unit"))}
    mc = {n: i for i, n in enumerate(layout.names("move"))}
    cc = {n: i for i, n in enumerate(layout.names("cand_f"))}
    agg0 = len(layout.names("global")) + 3 * len(layout.names("value_unit"))
    ac = {n: agg0 + i for i, n in enumerate(layout.names("value_agg"))}
    out = np.zeros(len(rec), np.int64)
    out |= np.array([bit[t] for t in ("flag", "contact", "respawn", "quiet")])[strata(rec, layout)]
    valid_cand = cand >= 0
    tests = {
        "pickup_option": act[..., 1].any((1, 2)),
        "carrier_own": (u[..., uc["carrying"]] > 0).any(1),
        "escort": v[:, ac["escort_threat"]] > 0,
        "interception": (v[:, ac["carrier_shots"]] > 0) | (m[..., mc["carrier_targets"]] > 0).any((1, 2)),
        "focus_fire": (valid_cand & (cf[..., cc["support"]] >= 1)).any((1, 2)),
        "overkill_risk": (valid_cand & (cf[..., cc["support"]] >= 1) & (cf[..., cc["hp"]] <= KILL_HP)).any((1, 2)),
        "collision_risk": ((m[..., mc["legal"]] > 0) & (m[..., mc["occupied"]] > 0)).any((1, 2)),
        "return_risk": g[:, gc["flags_dropped"]] > 0,
        "endgame": g[:, gc["remaining"]] <= 40,
    }
    for name, hit in tests.items():
        out[hit] |= bit[name]
    return out


def active_mask(rec: np.ndarray, layout) -> np.ndarray:
    """3-bit mask per row: units on the board this turn (only they can disagree)."""
    origin = layout.get(rec, "origin")
    return ((origin >= 0) * np.array([1, 2, 4])).sum(1)


def probe_records(match: dict, turns: int) -> tuple[dict, np.ndarray]:
    """(the match truncated to the probe, every record row of it: turn-major, team-minor)."""
    m = truncate(match, turns)
    return m, native.extract(matchlog.seat_stream(m, 0))


def describe(path: Path, match: dict, seat: int, rec: np.ndarray, layout) -> dict:
    rows = rec[rec[:, 1] == seat]
    tags = turn_tags(rows, layout)
    return {"probe_id": f"{match['match_id']}-s{seat:02d}", "file": path.as_posix(), "match_sha256": sha256_file(path),
            "match_id": match["match_id"], "teams": match["teams"], "map_seed": match["map_seed"],
            "flag_seed": match["flag_seed"], "seat": seat, "turns": int(len(rows)),
            "history_seats": [s["name"] for s in match["seats"]],
            "tags": [int(t) for t in tags], "active": [int(x) for x in active_mask(rows, layout)],
            "tag_counts": {t: int(((tags >> i) & 1).sum()) for i, t in enumerate(TAGS)}}


def choose(raw: Path, teams: int, per_count: int, turns: int, layout) -> list[dict]:
    files = list(matchlog.iter_matches(raw, [teams]))
    ranked = []
    for f in files:
        m = matchlog.load(f)
        if m.get("faults") and len(m["faults"]) > 3 * teams:
            continue  # a history full of missing replies is a poor probe
        families = {s["family"] for s in m["seats"]}
        ranked.append((-len(families), hashlib.sha256(m["match_id"].encode()).hexdigest(), f, m))
    ranked.sort(key=lambda r: r[:2])
    probes, maps = [], set()
    for _, _, f, m in ranked:
        if len(probes) == per_count:
            break
        if m["map_seed"] in maps:
            continue
        maps.add(m["map_seed"])
        best = None
        _, rec = probe_records(m, turns)
        for seat in range(teams):
            p = describe(f.resolve(), m, seat, rec, layout)
            score = sum(p["tag_counts"][t] for t in TAGS[4:])
            if best is None or score > best[0]:
                best = (score, p)
        probes.append(best[1])
    return probes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="recorded matches (data/raw layout: nXX/<match_id>.json.gz)")
    ap.add_argument("--teams", type=int, nargs="+", default=list(TEAM_COUNTS))
    ap.add_argument("--per-count", type=int, default=6,
                    help="probe streams per team count (distinct maps; 6-12 for discovery, see the fingerprint review)")
    ap.add_argument("--turns", type=int, default=400, help="turns per probe (from turn 1; the full game by default)")
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "probe_suite.json"))
    a = ap.parse_args(argv)
    layout = feature_layout()
    probes, short = [], {}
    for teams in a.teams:
        got = choose(Path(a.raw), teams, a.per_count, a.turns, layout)
        probes += got
        if len(got) < a.per_count:
            short[teams] = len(got)
        print(f"[probes] {teams} teams: {len(got)} probes", flush=True)
    totals = {t: sum(p["tag_counts"][t] for p in probes) for t in TAGS}
    suite = {"schema": SCHEMA_SUITE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": str(Path(a.raw).resolve()),
             "spec_version": layout.schema["spec_version"], "tag_rules": TAG_RULES, "turns": a.turns, "per_count": a.per_count,
             "tags": list(TAGS), "tag_totals": totals, "short_counts": short, "probes": probes}
    save_json(a.out, suite, indent=None)
    print(f"[probes] {len(probes)} probes, tag turns {totals} -> {a.out}")
    if short:
        print(f"[probes] fewer matches than requested for team counts {short}: collect more before freezing")


if __name__ == "__main__":
    main()
