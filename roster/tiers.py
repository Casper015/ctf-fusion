"""Performance tiers from a (light) strength screen, and the screened admission for the open training pool.

    python -m roster.tiers --strength roster/report/screen_light/strength.json \\
        --shortlist roster/report/selection/shortlist.json --admission roster/report/selection/admission_draft.json \\
        --out roster/report/screen_light

Every candidate played the same frozen fixtures against the reference roster
(roster.strength), so utilities are paired across candidates. Utility is
pairwise (beat 1, tie 0.5, per other team), team counts weigh equally, and
the interval is utility +- z SE over maps (z from the screen, 1.96).

Tiers, pre-registered against parity with the reference field (0.5):

  above   lower bound > 0.5: beats the reference field on average;
  below   upper bound < 0.5;
  level   the interval contains 0.5 (not separated by this screen).

Reliability under the official limits (2 s first turn, 50 ms later): a bot
with any dead game, more than --max-faults faults per game or more than
--max-late late replies per game is flagged unreliable. In the screened
admission an unreliable bot is not admitted (its missed replies are no P
labels and distort V), a reliable "above" bot gains the field role (strong
competition), judge bots keep the field role, and every contribution
sentence gains its screen result. Tiers describe strength against this
reference roster on these maps only; a light screen separates large gaps,
not neighbors.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

from fusion.paths import REPO
from roster.common import load_json, save_json

BANDS = {"2": [2], "3-5": [3, 4, 5], "6-10": [6, 7, 8, 9, 10], "11-15": [11, 12, 13, 14, 15]}


def short(bot: str) -> str:
    p = Path(bot)
    return p.name if p.parent.name in ("", ".") else f"{p.parent.name}/{p.name}"


def classify(strength: dict, max_faults=0.01, max_late=0.1) -> list[dict]:
    z = strength.get("z", 1.96)
    rows = []
    for bot, c in strength["candidates"].items():
        u, se = c["utility"], c["se"]
        lcb, ucb = (None, None) if u is None or se is None else (u - z * se, u + z * se)   # no spread: not separated
        tier = "level" if lcb is None else "above" if lcb > 0.5 else "below" if ucb < 0.5 else "level"
        games = max(1, c["games"])
        why = []
        if c["dead_games"]:
            why.append(f"{c['dead_games']} dead games")
        if c["faults_per_game"] > max_faults:
            why.append(f"{c['faults_per_game']:.3f} faults per game")
        if c["late_replies"] / games > max_late:
            why.append(f"{c['late_replies'] / games:.2f} late replies per game")
        bands = {}
        for name, ns in BANDS.items():
            vals = [c["per_count"][str(n)]["utility"] for n in ns if str(n) in c["per_count"]]
            bands[name] = None if not vals else round(sum(vals) / len(vals), 3)
        rows.append({"bot": bot, "utility": None if u is None else round(u, 4), "lcb": None if lcb is None else round(lcb, 4),
                     "ucb": None if ucb is None else round(ucb, 4), "maps": c["maps"], "counts": c["team_counts_covered"],
                     "tier": tier, "reliable": not why, "unreliable_because": why, "bands": bands,
                     "p99_ms_median": c.get("p99_ms_median"), "max_ms": c.get("max_ms"), "first_ms_max": c.get("first_ms_max"),
                     "faults_per_game": round(c["faults_per_game"], 4), "late_replies": c["late_replies"]})
    rows.sort(key=lambda r: -(r["utility"] if r["utility"] is not None else -1))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def screened_admission(admission: dict, rows: list[dict], revision: str) -> dict:
    by = {r["bot"]: r for r in rows}
    out = copy.deepcopy(admission)
    out["revision"] = revision
    out["_comment"] = ("DRAFT updated by roster.tiers with light-screen results: unreliable bots not admitted, reliable bots above "
                       "the reference field added to the field role. A person reviews before training.")
    for bot, spec in out["bots"].items():
        r = by.get(bot)
        if r is None:
            spec["contribution"] += " Not in the light screen."
            continue
        interval = "no interval" if r["lcb"] is None else f"[{r['lcb']:.3f}, {r['ucb']:.3f}]"
        spec["contribution"] += (f" Light screen: utility {r['utility']:.3f} {interval} over {r['maps']} maps "
                                 f"({r['tier']} the reference field).")
        if not r["reliable"]:
            spec["admit"] = False
            spec["contribution"] += f" Not admitted: unreliable at the official limits ({'; '.join(r['unreliable_because'])})."
        elif r["tier"] == "above" and "augmentation" not in spec["roles"] and "field" not in spec["roles"]:
            spec["roles"] = sorted(set(spec["roles"]) | {"field"})
    return out


def report(rows: list[dict], strength: dict, shortlist: dict | None) -> str:
    fam = {}
    for f in (shortlist or {}).get("families", []):
        fam[f["representative"]] = f
    tiers = {t: sum(r["tier"] == t for r in rows) for t in ("above", "level", "below")}
    unreliable = [r for r in rows if not r["reliable"]]
    maps = sorted({r["maps"] for r in rows})
    out = [
        "# Light strength screen: performance tiers",
        "",
        "<!-- Generated by `python -m roster.tiers` from strength.json. Do not edit by hand. -->",
        "",
        f"{len(rows)} candidates, {maps[0] if maps else 0}-{maps[-1] if maps else 0} maps each over "
        f"{max((r['counts'] for r in rows), default=0)} team counts, the same fixtures for every candidate against the frozen "
        f"reference roster, official limits. Interval: utility +- {strength.get('z', 1.96)} SE over maps. "
        f"Tiers against parity 0.5: **{tiers['above']} above**, **{tiers['level']} level**, **{tiers['below']} below**; "
        f"{len(unreliable)} unreliable at the official limits. Stale rows ignored: {sum((strength.get('stale_rows') or {}).values())}.",
        "",
        "A light screen separates large gaps; neighbors inside one tier are not ranked by it. Utility is against this reference "
        "roster on these maps only. Per-band columns are mean utilities over the team counts of the band.",
        "",
        "| Rank | Bot | Tier | Utility [95% interval] | Maps | 2 | 3-5 | 6-10 | 11-15 | Shortlist tier | p99 ms | Reliable |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in rows:
        b = r["bands"]
        f = fam.get(r["bot"], {})
        cell = ("n/a" if r["utility"] is None else f"{r['utility']:.3f}" if r["lcb"] is None
                else f"{r['utility']:.3f} [{r['lcb']:.3f}, {r['ucb']:.3f}]")
        out.append(f"| {r['rank']} | `{short(r['bot'])}` | {r['tier']} | {cell} | {r['maps']} | "
                   + " | ".join("" if b[k] is None else f"{b[k]:.2f}" for k in BANDS)
                   + f" | {f.get('tier', '')} | {'' if r['p99_ms_median'] is None else round(r['p99_ms_median'], 1)} | "
                   f"{'yes' if r['reliable'] else 'no: ' + '; '.join(r['unreliable_because'])} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strength", required=True)
    ap.add_argument("--shortlist", default=str(REPO / "roster" / "report" / "selection" / "shortlist.json"))
    ap.add_argument("--admission", default=str(REPO / "roster" / "report" / "selection" / "admission_draft.json"))
    ap.add_argument("--max-faults", type=float, default=0.01, help="faults per game above which a bot is unreliable")
    ap.add_argument("--max-late", type=float, default=0.1, help="late replies per game above which a bot is unreliable")
    ap.add_argument("--revision", default="r2-draft-screened")
    ap.add_argument("--out", default=str(REPO / "roster" / "report" / "screen_light"))
    a = ap.parse_args(argv)
    strength = load_json(a.strength)
    rows = classify(strength, a.max_faults, a.max_late)
    shortlist = load_json(a.shortlist) if Path(a.shortlist).exists() else None
    out = Path(a.out)
    save_json(out / "tiers.json", {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "params": {"parity": 0.5, "max_faults": a.max_faults,
                                   "max_late": a.max_late, "z": strength.get("z", 1.96)}, "bots": rows})
    (out / "TIERS.md").write_text(report(rows, strength, shortlist), encoding="utf-8")
    if Path(a.admission).exists():
        save_json(out / "admission_screened.json", screened_admission(load_json(a.admission), rows, a.revision))
    t = {k: sum(r["tier"] == k for r in rows) for k in ("above", "level", "below")}
    print(f"[tiers] {len(rows)} bots {t}, {sum(not r['reliable'] for r in rows)} unreliable -> {out}")


if __name__ == "__main__":
    main()
