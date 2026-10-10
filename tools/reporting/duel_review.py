"""Review of AI_Test ctf-elite-duels-v3: 31 bots x 2,000 frozen duels against six elite opponents.

    python tools/pipeline/fetch_benchmark.py
    python tools/reporting/duel_review.py        # docs/benchmark/DUEL_TABLES.md, docs/benchmark/duel_review.json

Duel play is deterministic: bots sharing the duel code path produce identical
results on every fixture, so the 31 bots collapse into a few distinct duel
behaviors. Each behavior is compared with the Control_Hybrid behavior fixture
by fixture (pairwise score: win 1, tie 0.5, loss 0). Every fixture is its own
map, so fixtures are independent and the paired standard error is the plain
one. Because there is no run-to-run noise, the uncertainty is only about how
well 2,000 fixtures represent other maps.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "data" / "benchmark_raw" / "ctf-elite-duels-v3" / "results" / "games_elite_duels.jsonl"
CONTROL = "C01_Control_Hybrid"
OPPONENTS = ["Player5", "GPT-6 Astra", "GPT-6.1 Sol", "Fable 5.1", "Sonnet 5.5", "Opus 5.5"]
LABELS = {  # one representative per behavior: what differs from Control_Hybrid in duels
    "C01_Control_Hybrid": "Control_Hybrid duel planner",
    "C07_Player5_Omega": "Omega (Hybrid + endgame decay)",
    "C12_R3_CFL_conditional": "CFL (proximity-tiered prediction)",
    "C17_R3_EXPECTIMAX_anytime": "R3 expectimax (24-sample duel refinement)",
    "Astra_Habits": "Astra habits",
    "C20_R3_HABITS_adaptive": "R3 habits (ours)",
    "C16_R3_ESCORT_mech": "ESCORT",
    "C21_R3_HEALTHY_pickup": "HEALTHY",
    "C15_R3_ESCORT_healthy": "ESCORT + HEALTHY",
}


def load():
    g = defaultdict(dict)
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            g[x["candidate"]][x["fixture_id"]] = x
    return g


def behaviors(g):
    fx = sorted(g[CONTROL])
    groups = defaultdict(list)
    for c, d in g.items():
        groups[tuple((d[f]["cand_score"], d[f]["opp_score"]) for f in fx)].append(c)
    out = {}
    for members in groups.values():
        rep = next((m for m in members if m in LABELS), sorted(members)[0])
        out[rep] = sorted(members)
    return out


def paired(a, b, fixtures):
    d = [a[f]["pair_win"] - b[f]["pair_win"] for f in fixtures]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return {"delta": m, "se": sd / math.sqrt(n), "n": n, "gained": sum(x > 0 for x in d), "lost": sum(x < 0 for x in d)}


def mark(c):
    if c["se"] == 0:
        return ""
    z = c["delta"] / c["se"]
    return "++" if z > 3 else "+" if z > 1.96 else "--" if z < -3 else "-" if z < -1.96 else ""


def main():
    g = load()
    groups = behaviors(g)
    ctl = g[CONTROL]
    fixtures = sorted(ctl)
    by_opp = defaultdict(list)
    for f in fixtures:
        by_opp[ctl[f]["opponent"]].append(f)
    rows = []
    for rep, members in sorted(groups.items(), key=lambda kv: -sum(g[kv[0]][f]["pair_win"] for f in fixtures)):
        d = g[rep]
        row = {"behavior": LABELS.get(rep, rep), "representative": rep, "members": members,
               "pair_win": sum(d[f]["pair_win"] for f in fixtures) / len(fixtures),
               "margin": sum(d[f]["cand_score"] - d[f]["opp_score"] for f in fixtures) / len(fixtures),
               "vs_control": paired(d, ctl, fixtures) if rep != CONTROL else None,
               "by_opponent": {o: {"pair_win": sum(d[f]["pair_win"] for f in by_opp[o]) / len(by_opp[o]),
                                   "vs_control": paired(d, ctl, by_opp[o]) if rep != CONTROL else None} for o in OPPONENTS},
               "by_seat": {s: sum(d[f]["pair_win"] for f in fixtures if d[f]["cand_seat"] == s) /
                              max(1, sum(1 for f in fixtures if d[f]["cand_seat"] == s)) for s in (0, 1)}}
        rows.append(row)
    # Do the two duel improvements win different fixtures? (an upper bound for combining them)
    cfl, exp = g["C12_R3_CFL_conditional"], g["C17_R3_EXPECTIMAX_anytime"]
    gain = lambda d: {f for f in fixtures if d[f]["pair_win"] > ctl[f]["pair_win"]}
    loss = lambda d: {f for f in fixtures if d[f]["pair_win"] < ctl[f]["pair_win"]}
    overlap = {"cfl_gained": len(gain(cfl)), "cfl_lost": len(loss(cfl)), "expectimax_gained": len(gain(exp)),
               "expectimax_lost": len(loss(exp)), "gained_by_both": len(gain(cfl) & gain(exp)),
               "lost_by_both": len(loss(cfl) & loss(exp)),
               "oracle_best_of_two": sum(max(cfl[f]["pair_win"], exp[f]["pair_win"]) for f in fixtures) / len(fixtures),
               "control_losses": sum(ctl[f]["pair_win"] == 0 for f in fixtures),
               "control_losses_by_opponent": {o: sum(ctl[f]["pair_win"] == 0 for f in by_opp[o]) for o in OPPONENTS}}
    (REPO / "docs" / "benchmark" / "duel_review.json").write_text(json.dumps({"rows": rows, "overlap": overlap}, indent=1), encoding="utf-8")
    pct = lambda x: f"{100 * x:.1f}"
    dp = lambda c: "" if c is None else f"{100 * c['delta']:+.1f}{mark(c)}"
    out = ["# Elite duels: per-behavior tables (generated by tools/reporting/duel_review.py)", "",
           "31 bots x 2,000 frozen duels (one map each, seats alternate, about 333 per opponent). Bots with identical duel "
           "results on every fixture are one behavior. Deltas: pairwise points vs the Control_Hybrid behavior, same fixtures; "
           "`+`/`-` 95% interval excludes 0, `++`/`--` |z| > 3.", "",
           "| Behavior | Win % | Δ vs Hybrid | Fixtures gained / lost | Avg margin | " + " | ".join(OPPONENTS) + " | Bots |",
           "|---|---:|---:|---:|---:|" + "---:|" * len(OPPONENTS) + "---|"]
    for r in rows:
        v = r["vs_control"]
        opp = " | ".join(f"{pct(r['by_opponent'][o]['pair_win'])} {dp(r['by_opponent'][o]['vs_control'])}".strip() for o in OPPONENTS)
        out.append(f"| {r['behavior']} | {pct(r['pair_win'])} | {dp(v) or '—'} | "
                   f"{'' if v is None else str(v['gained']) + ' / ' + str(v['lost'])} | {r['margin']:+.2f} | {opp} | "
                   f"{', '.join(r['members'])} |")
    out += ["", "## Do CFL and R3 expectimax fix the same duels?", ""] + [f"- {k}: {v}" for k, v in overlap.items()] + [""]
    (REPO / "docs" / "benchmark" / "DUEL_TABLES.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
