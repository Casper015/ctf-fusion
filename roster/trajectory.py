"""Live outcomes per seat from full match logs (reports/FINGERPRINT_EFFECTIVENESS_REVIEW.md, P1 live backfill).

    python -m roster.trajectory --raw data/raw_screen --inventory data/roster/bot_inventory.json \\
        --out data/roster/live_profiles.json

Fingerprints measure what a bot requests on shared histories; this measures
what happens when it actually controls its units. Every number has an
explicit denominator, and an unfinished episode is censored, not a failure.

Event alignment: the events of settlement t arrive in the TURN t+1 block;
settlement 400's are in the match's final_block, so every event is counted
exactly once. Engine conventions are kept: a kill is credited 1/k to each of
the k units whose hits killed the victim in that settlement (several teams
can share it); an "attack" event is a hit, not a request; within a
settlement, movement precedes attacks, then pickup, voluntary drop and
capture, so a carrier cannot drop and shoot in the same turn.

Per seat (seat_outcomes):
  exposure      alive, carrier, shoot-capable (alive, not carrying) and
                threatened unit-turns at decision time;
  combat        attack requests (valid replies), hits, kills and carrier
                kills (fractional credit), deaths;
  flags         pickups, captures, voluntary drops, forced drops (deaths
                while carrying);
  episodes      carrier episodes keyed by (unit, flag, pickup turn), ended
                by delivery, voluntary drop or death, else censored at the end;
  drop chains   after a voluntary drop: the next pickup of that flag by our
                team or an enemy, or its return, within the flag-return
                window (else censored), and whether our units hit anything
                in the next 3 settlements;
  escape        survival over 5 and 10 settlements from a threatened
                decision (an enemy non-carrier within Manhattan reach 3:
                one move plus attack range 2, walls ignored; a declared
                proxy, not a death probability), censored near the end;
  engine        the engine's own totals (result.stats, absent keys are 0),
                and whether the recomputed totals match them.

aggregate() turns seat outcomes into profile entries per count: metric,
numerator, denominator, unique maps, estimate and a map-clustered interval
(ratio estimator, linearized standard error over maps), censored counts and
the evidence identity. High survival, many kills or frequent drops are
tendencies and outcomes, not strength: keep utility alongside.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

from fusion import matchlog
from fusion.paths import REPO
from fusion.protocol import ATTACK, parse_reply
from roster.common import load_json, save_json

ENGINE_KEYS = ("kills", "carrier_kills", "deaths", "pickups", "captures")
THREAT_REACH = 3
HORIZONS = (5, 10)
DROP_WINDOW = 15            # the engine's flag_return; read from INIT when present
HIT_FOLLOWUP = 3
SCHEMA_LIVE = "ctf-live-profile-1"


def parse_detailed(text: str) -> dict:
    """A TURN block with every event field the engine writes (fusion.protocol.parse_block keeps its smaller contract)."""
    lines = text.split("\n")
    i = next(k for k, line in enumerate(lines) if line.startswith("TURN "))
    turn = int(lines[i].split()[1])
    n_units = int(lines[i + 2].split()[1])
    units = {}
    for line in lines[i + 3:i + 3 + n_units]:
        uid, team, x, y, hp, flag, respawn = (int(v) for v in line.split()[1:8])
        units[uid] = {"team": team, "pos": None if x < 0 else (x, y), "hp": hp, "flag": None if flag < 0 else flag}
    j = i + 3 + n_units
    j += 1 + int(lines[j].split()[1])
    events = []
    for line in lines[j + 1:j + 1 + int(lines[j].split()[1])]:
        tok = line.split()
        kind, v = tok[1], [int(x) for x in tok[2:]]
        if kind == "death":
            events.append({"type": "death", "unit": v[0], "pos": (v[1], v[2]), "flag": None if v[3] < 0 else v[3],
                           "by": v[5:5 + v[4]]})
        elif kind == "attack":
            events.append({"type": "attack", "unit": v[0], "target": v[1]})
        elif kind == "capture":
            events.append({"type": "capture", "unit": v[0], "team": v[1], "flag": v[2]})
        elif kind in ("pickup", "drop"):
            events.append({"type": kind, "unit": v[0], "flag": v[1]})
        elif kind == "respawn":
            events.append({"type": "respawn", "unit": v[0], "pos": (v[1], v[2])})
        elif kind == "return":
            events.append({"type": "return", "flag": v[0]})
    return {"turn": turn, "units": units, "events": events}


def rules(match: dict) -> dict:
    tok = match["init"].split("\n", 1)[0].split()
    return {"turns": int(tok[4]), "flag_return": int(tok[8])} if len(tok) >= 9 else {"turns": 400, "flag_return": DROP_WINDOW}


def seat_outcomes(match: dict, seat: int) -> dict:
    """Outcomes of team `seat` over one full match (see the module docstring)."""
    blocks = [parse_detailed(b) for b in match["blocks"]] + [parse_detailed(match["final_block"])]
    T = len(match["blocks"])
    window = rules(match)["flag_return"]
    own = lambda uid: uid // 3 == seat  # noqa: E731
    settle = [blocks[t]["events"] for t in range(1, T + 1)]           # settle[t - 1]: settlement t's events
    o = defaultdict(float)
    episodes = {"started": 0, "delivered": 0, "voluntary_drop": 0, "forced_drop": 0, "censored": 0, "carry_turns": 0}
    chain = {"drops": 0, "own_pickup": 0, "enemy_pickup": 0, "returned": 0, "censored": 0, "own_hit_within_3": 0}
    escape = {f"{k}{h}": {"eligible": 0, "survived": 0, "censored": 0} for h in HORIZONS for k in ("", "carrier_")}
    deaths_at = defaultdict(set)                                       # settlement -> our units that died
    open_eps = {}
    for t, ev in enumerate(settle, 1):
        for e in ev:
            if e["type"] == "death":
                if own(e["unit"]):
                    o["deaths"] += 1
                    deaths_at[t].add(e["unit"])
                    if e["flag"] is not None:
                        o["forced_drops"] += 1
                elif e["by"]:
                    share = sum(own(u) for u in e["by"]) / len(e["by"])
                    o["kills"] += share
                    o["carrier_kills"] += share if e["flag"] is not None else 0.0
            elif e["type"] == "attack" and own(e["unit"]):
                o["hits"] += 1
            elif e["type"] == "pickup" and own(e["unit"]):
                o["pickups"] += 1
            elif e["type"] == "drop" and own(e["unit"]):
                o["voluntary_drops"] += 1
            elif e["type"] == "capture" and own(e["unit"]):
                o["captures"] += 1
        # carrier episodes, in the engine's event order (deaths, pickups, drops, captures): a unit can pick up and
        # capture in the same settlement, which is a delivery of zero carried turns
        for e in ev:
            key = e.get("unit")
            if e["type"] == "pickup" and own(key):
                episodes["started"] += 1
                open_eps[key] = t
            elif key in open_eps and e["type"] in ("capture", "drop", "death"):
                start = open_eps.pop(key)
                episodes["carry_turns"] += t - start
                episodes[{"capture": "delivered", "drop": "voluntary_drop", "death": "forced_drop"}[e["type"]]] += 1
        # what happened after each voluntary drop
        for e in ev:
            if e["type"] == "drop" and own(e["unit"]):
                chain["drops"] += 1
                outcome = None
                for t2 in range(t + 1, min(T, t + window) + 1):
                    for e2 in settle[t2 - 1]:
                        if e2["type"] == "pickup" and e2["flag"] == e["flag"]:
                            outcome = "own_pickup" if own(e2["unit"]) else "enemy_pickup"
                        elif e2["type"] == "return" and e2["flag"] == e["flag"]:
                            outcome = "returned"
                        if outcome:
                            break
                    if outcome:
                        break
                chain[outcome or ("censored" if t + window > T else "returned")] += 1
                chain["own_hit_within_3"] += any(e2["type"] == "attack" and own(e2["unit"])
                                                  for t2 in range(t + 1, min(T, t + HIT_FOLLOWUP) + 1) for e2 in settle[t2 - 1])
    for start in open_eps.values():
        episodes["censored"] += 1
        episodes["carry_turns"] += T - start
    # decisions: exposure, requests, escape
    for t in range(1, T + 1):
        state = blocks[t - 1]["units"]
        line = match["replies"][t - 1][seat] if t - 1 < len(match["replies"]) else None
        reqs = parse_reply(line, t)
        enemies = [u["pos"] for uid, u in state.items() if not own(uid) and u["pos"] is not None and u["flag"] is None]
        for k in range(3):
            uid = 3 * seat + k
            u = state.get(uid)
            if u is None or u["pos"] is None:
                continue
            o["alive_unit_turns"] += 1
            carrying = u["flag"] is not None
            o["carrier_unit_turns"] += carrying
            o["shoot_capable_unit_turns"] += not carrying
            if reqs is not None and reqs[k].act == ATTACK:
                o["attack_requests"] += 1
            x, y = u["pos"]
            if not any(abs(x - ex) + abs(y - ey) <= THREAT_REACH for ex, ey in enemies):
                continue
            o["threatened_unit_turns"] += 1
            for h in HORIZONS:
                for key in (f"{h}", f"carrier_{h}") if carrying else (f"{h}",):
                    if t + h - 1 > T:
                        escape[key]["censored"] += 1
                        continue
                    escape[key]["eligible"] += 1
                    escape[key]["survived"] += not any(uid in deaths_at[s] for s in range(t, t + h))
    stats = (match.get("result", {}).get("stats") or [None] * (seat + 1))[seat]
    engine = None if stats is None else {k: float(stats.get(k, 0)) for k in ENGINE_KEYS}
    recomputed = {k: round(o[k], 2) for k in ENGINE_KEYS}
    return {"turns": T, **{k: round(v, 4) for k, v in o.items()}, "episodes": episodes, "drop_chain": chain,
            "escape": escape, "engine": engine,
            "engine_match": None if engine is None else all(abs(recomputed[k] - engine[k]) < 0.011 for k in ENGINE_KEYS)}


# metric -> (numerator, denominator) over seat outcomes; per-game metrics use the game count
METRICS = {
    "kills_per_game": (lambda r: r.get("kills", 0), lambda r: 1),
    "carrier_kills_per_game": (lambda r: r.get("carrier_kills", 0), lambda r: 1),
    "deaths_per_game": (lambda r: r.get("deaths", 0), lambda r: 1),
    "captures_per_game": (lambda r: r.get("captures", 0), lambda r: 1),
    "death_hazard_per_100": (lambda r: 100 * r.get("deaths", 0), lambda r: r.get("alive_unit_turns", 0)),
    "hit_per_attack_request": (lambda r: r.get("hits", 0), lambda r: r.get("attack_requests", 0)),
    "attack_request_rate": (lambda r: r.get("attack_requests", 0), lambda r: r.get("shoot_capable_unit_turns", 0)),
    "carrier_share": (lambda r: r.get("carrier_unit_turns", 0), lambda r: r.get("alive_unit_turns", 0)),
    "delivery_rate": (lambda r: r["episodes"]["delivered"], lambda r: r["episodes"]["started"] - r["episodes"]["censored"]),
    "forced_drop_rate": (lambda r: r["episodes"]["forced_drop"], lambda r: r["episodes"]["started"] - r["episodes"]["censored"]),
    "voluntary_drop_rate": (lambda r: r["episodes"]["voluntary_drop"], lambda r: r["episodes"]["started"] - r["episodes"]["censored"]),
    "drop_then_own_pickup": (lambda r: r["drop_chain"]["own_pickup"], lambda r: r["drop_chain"]["drops"] - r["drop_chain"]["censored"]),
    "drop_then_hit_within_3": (lambda r: r["drop_chain"]["own_hit_within_3"], lambda r: r["drop_chain"]["drops"]),
    **{f"threatened_survival_{k}{h}": (lambda r, k=k, h=h: r["escape"][f"{k}{h}"]["survived"],
                                        lambda r, k=k, h=h: r["escape"][f"{k}{h}"]["eligible"])
       for h in HORIZONS for k in ("", "carrier_")},
}
CENSORED = {"delivery_rate": ("episodes", "censored"), "forced_drop_rate": ("episodes", "censored"),
            "voluntary_drop_rate": ("episodes", "censored"), "drop_then_own_pickup": ("drop_chain", "censored")}


def ratio(rows: list[dict], num, den, z: float = 1.96) -> dict:
    """Ratio estimate sum(num) / sum(den) with a map-clustered (linearized) interval; rows carry a map key."""
    by_map = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by_map[r["map"]][0] += num(r["live"])
        by_map[r["map"]][1] += den(r["live"])
    n, d = sum(v[0] for v in by_map.values()), sum(v[1] for v in by_map.values())
    m = len(by_map)
    if d <= 0:
        return {"numerator": n, "denominator": d, "unique_maps": m, "estimate": None, "interval": None}
    est = n / d
    if m > 1:
        resid = [v[0] - est * v[1] for v in by_map.values()]
        se = math.sqrt(sum(x * x for x in resid) / (m * (m - 1))) / (d / m)
        interval = [round(est - z * se, 5), round(est + z * se, 5)]
    else:
        interval = None
    return {"numerator": round(n, 4), "denominator": round(d, 4), "unique_maps": m, "estimate": round(est, 5), "interval": interval}


def aggregate(rows: list[dict], identity: dict | None = None) -> dict:
    """rows: {"teams", "map", "live"} per game for one bot. Profile entries per count and over all counts."""
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[str(r["teams"])].append(r)
    for count, rs in sorted(groups.items(), key=lambda kv: int(kv[0])):
        entries = {}
        for metric, (num, den) in METRICS.items():
            e = ratio(rs, num, den) | {"metric": metric, "count": int(count), "mode": "live", "games": len(rs)}
            if metric in CENSORED:
                a, b = CENSORED[metric]
                e["censored"] = sum(r["live"][a][b] for r in rs)
            entries[metric] = e
        entries["_engine_mismatch_games"] = sum(r["live"]["engine_match"] is False for r in rs)
        out[count] = entries
    return {"per_count": out, "games": len(rows), "identity": identity or {}}


def backfill(raw: Path, inventory: dict) -> dict:
    """Seat outcomes for every seat whose recorded source hash is an inventoried representative, from retained logs."""
    rep_of = {}
    for b in inventory["bots"]:
        rep_of.setdefault(b["source_sha256"], b["representative"])
    rows = defaultdict(list)
    files, skipped = [], 0
    for f in sorted(Path(raw).rglob("*.json.gz")):
        try:
            m = matchlog.load(f)
        except ValueError:                                         # not a match log (e.g. fingerprint runs nearby)
            skipped += 1
            continue
        files.append(f)
        for seat, s in enumerate(m["seats"]):
            bot = rep_of.get(s.get("sha256"))
            if bot:
                rows[bot].append({"teams": m["teams"], "map": m["map_seed"], "live": seat_outcomes(m, seat),
                                  "source_sha256": s["sha256"], "engine_sha256": m.get("engine_sha256")})
    profiles = {}
    for bot, rs in sorted(rows.items()):
        shas = sorted({r["source_sha256"] for r in rs})
        profiles[bot] = aggregate(rs, {"source_sha256": shas, "engine_sha256": sorted({r["engine_sha256"] or "" for r in rs})})
    return {"schema": SCHEMA_LIVE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": str(Path(raw).resolve()),
            "matches": len(files), "skipped_files": skipped, "bots": profiles,
            "conventions": {"threat_reach": THREAT_REACH, "horizons": list(HORIZONS), "hit_followup": HIT_FOLLOWUP,
                            "kill_credit": "1/k per killing hit, as the engine", "attack_event": "a hit, not a request"}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="retained full match logs (strength --keep-matches, or collection)")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "live_profiles.json"))
    a = ap.parse_args(argv)
    result = backfill(Path(a.raw), load_json(a.inventory))
    save_json(a.out, result)
    bad = sum(p["per_count"][c]["_engine_mismatch_games"] for p in result["bots"].values() for c in p["per_count"])
    print(f"[trajectory] {result['matches']} matches, {len(result['bots'])} bots profiled, "
          f"{bad} seat-games whose recomputed totals differ from the engine -> {a.out}")


if __name__ == "__main__":
    main()
