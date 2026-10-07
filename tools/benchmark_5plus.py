"""Per-team-size review of the AI_Test benchmarks for 5 or more teams.

    python tools/fetch_benchmark.py          # raw logs from GitHub into data/benchmark_raw/
    python tools/benchmark_5plus.py          # docs/BENCHMARK_5PLUS.md, docs/benchmark_5plus.json,
                                             # benchmarks/games_5plus.csv.gz (committed)

Every size is analyzed on its own (no multi-team average). Requested sizes
are 5, 6, 7, 8, 10, 12, 13, 14, 15; the benchmark tracks only played 5, 6, 8,
10 and 15 of them, and the others are reported as not played.

Paired deltas use the same method as tools/benchmark_review.py: variant minus
control on identical fixtures, pairwise score in percentage points, standard
errors clustered by map. Event rates come from the per-game counters the
benchmark logs carry (captures, pickups, carrier kills and deaths, drops,
timeouts); the logs have no per-turn latency and no overall kill/death counts.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from benchmark_review import paired, verdict  # noqa: E402

RAW = REPO / "data" / "benchmark_raw" / "ctf-benchmark-v3"
LOGS = {"T1": RAW / "track1_10k_suite" / "results" / "games.jsonl",
        "C": RAW / "track2_2k_c_series" / "results" / "games_c_2000.jsonl"}
REQUESTED = (5, 6, 7, 8, 10, 12, 13, 14, 15)
TURNS = 400

KEY = [  # (track, label, variant, control)
    ("T1", "Omega vs Control_Hybrid", "Control_Omega", "B0_Control_Hybrid"),
    ("C", "Omega vs Control_Hybrid", "C07_Player5_Omega", "C01_Control_Hybrid"),
    ("C", "RISK gene vs Omega", "C05_G_RISK_entropic", "C07_Player5_Omega"),
    ("T1", "Astra habits + carrier DP vs Omega", "Astra_Full", "Control_Omega"),
    ("T1", "Astra habits + carrier DP vs Player2 v005", "Astra_Full", "B2_Player2_v005"),
    ("C", "Expectimax vs foundation (copy 1)", "C17_R3_EXPECTIMAX_anytime", "C02_Control_NG_foundation"),
    ("C", "Expectimax vs foundation (copy 2)", "C18_R3_EXPECTIMAX_sampled", "C02_Control_NG_foundation"),
    ("C", "Expectimax vs foundation (both copies)", "EXPECTIMAX_MEAN", "C02_Control_NG_foundation"),
    ("T1", "Player2 v005 vs Control_Hybrid", "B2_Player2_v005", "B0_Control_Hybrid"),
    ("T1", "Omega vs Player2 v005", "Control_Omega", "B2_Player2_v005"),
]
RANKED = [  # single factors, for the ranking-shift table: (track, name, variant, control)
    ("C", "Tuned weights (foundation)", "C22_R3_WEIGHTS_tuned", "C02_Control_NG_foundation"),
    ("C", "Expectimax (both copies)", "EXPECTIMAX_MEAN", "C02_Control_NG_foundation"),
    ("C", "CFL", "C12_R3_CFL_conditional", "C02_Control_NG_foundation"),
    ("C", "CDP", "C11_R3_CDP_horizon", "C02_Control_NG_foundation"),
    ("C", "TERMINAL", "C13_R3_CL_terminal", "C02_Control_NG_foundation"),
    ("C", "DROP", "C14_R3_DROP_scored", "C02_Control_NG_foundation"),
    ("C", "ESCORT", "C16_R3_ESCORT_mech", "C02_Control_NG_foundation"),
    ("C", "HEALTHY", "C21_R3_HEALTHY_pickup", "C02_Control_NG_foundation"),
    ("C", "HABITS (ours)", "C20_R3_HABITS_adaptive", "C02_Control_NG_foundation"),
    ("C", "FLAGS", "C19_R3_FLAGS_joint", "C02_Control_NG_foundation"),
    ("C", "Tuned weights (Omega)", "C07_Player5_Omega", "C09_Player5_Omega_defaultWeights"),
    ("C", "Hybrid macro (Omega)", "C10_Player5_Omega_hybridMacro", "C07_Player5_Omega"),
    ("C", "Gene RISK", "C05_G_RISK_entropic", "C07_Player5_Omega"),
    ("C", "Gene ASYM", "C03_G_ASYM_plans", "C07_Player5_Omega"),
    ("C", "Gene SH", "C06_G_SH_halving", "C07_Player5_Omega"),
    ("C", "Gene PLAYOUT", "C04_G_PLAYOUT_4ply", "C07_Player5_Omega"),
    ("C", "Genes LIT (all four)", "C08_Player5_Omega_LIT", "C07_Player5_Omega"),
    ("T1", "Astra joint sampler", "Astra_Joint", "Control_Omega"),
    ("T1", "Astra habits", "Astra_Habits", "Astra_Joint"),
    ("T1", "Astra carrier DP", "Astra_Carrier", "Astra_Joint"),
    ("T1", "Astra habits + DP", "Astra_Full", "Astra_Joint"),
]
EVENT_BOTS = [("T1", "B0_Control_Hybrid"), ("T1", "Control_Omega"), ("T1", "B2_Player2_v005"), ("T1", "Astra_Habits"),
              ("T1", "Astra_Full"), ("C", "C02_Control_NG_foundation"), ("C", "C17_R3_EXPECTIMAX_anytime"),
              ("C", "C07_Player5_Omega"), ("C", "C05_G_RISK_entropic"), ("C", "C08_Player5_Omega_LIT")]
FIELDS = ["track", "game_id", "fixture_id", "map", "seat", "size", "candidate", "pair_win", "cand_score", "rank", "is_win",
          "captures", "pickups", "carrier_kills", "carrier_deaths", "drops", "timeouts", "disqualified"]


def load(path: Path) -> dict[str, dict[str, dict]]:
    out = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            out[g["candidate"]][g["fixture_id"]] = g
    return out


def add_expectimax_mean(c: dict) -> None:
    """Average of the two identical expectimax builds per fixture: halves their timing noise."""
    a, b = c["C17_R3_EXPECTIMAX_anytime"], c["C18_R3_EXPECTIMAX_sampled"]
    avg = ("pair_win", "cand_score", "captures", "pickups", "carrier_kills", "carrier_deaths", "drops", "timeouts")
    c["EXPECTIMAX_MEAN"] = {f: {**a[f], **{k: (float(a[f][k]) + float(b[f][k])) / 2 for k in avg}} for f in a if f in b}


def cell(g, variant, control, size, key="pair_win"):
    r = paired(g, variant, control, size, key)
    return None if r is None else {"delta": r[0], "se": r[1], "n": r[2], "verdict": verdict(r[0], r[1])}


def events(fx: dict, size: int) -> dict | None:
    games = [x for x in fx.values() if x["size"] == size]
    if not games:
        return None
    n = len(games)
    tot = {k: sum(float(x[k]) for x in games) for k in ("captures", "pickups", "carrier_kills", "carrier_deaths", "drops", "timeouts")}
    ck = sorted(float(x["carrier_kills"]) for x in games)
    cd = sorted(float(x["carrier_deaths"]) for x in games)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    return {
        "games": n, "pair_win": sum(x["pair_win"] for x in games) / n,
        "pickups": tot["pickups"] / n, "captures": tot["captures"] / n,
        "conversion": tot["captures"] / tot["pickups"] if tot["pickups"] else math.nan,
        "carrier_deaths": tot["carrier_deaths"] / n,
        "carrier_survival": 1 - tot["carrier_deaths"] / tot["pickups"] if tot["pickups"] else math.nan,
        "carrier_kills": tot["carrier_kills"] / n,
        "kill_death": tot["carrier_kills"] / tot["carrier_deaths"] if tot["carrier_deaths"] else math.nan,
        "carrier_kills_p10_p50_p90": [q(ck, .1), q(ck, .5), q(ck, .9)],
        "carrier_deaths_p10_p50_p90": [q(cd, .1), q(cd, .5), q(cd, .9)],
        "drops": tot["drops"] / n, "timeouts_per_100_games": 100 * tot["timeouts"] / n,
        "timeouts_per_1000_turns": 1000 * tot["timeouts"] / (n * TURNS),
        "games_with_timeout": sum(x["timeouts"] > 0 for x in games) / n,
        "disqualified": sum(bool(x["disqualified"]) for x in games),
    }


def write_clean(tracks, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for track, g in tracks.items():
            for cand, fx in g.items():
                if cand == "EXPECTIMAX_MEAN":
                    continue
                for fid, x in sorted(fx.items()):
                    if x["size"] < 5:
                        continue
                    m, seat = fid.rsplit("-", 1)
                    w.writerow([track, x["game_id"], fid, m, seat, x["size"], cand, x["pair_win"], x["cand_score"], x["rank"],
                                int(x["is_win"]), x["captures"], x["pickups"], x["carrier_kills"], x["carrier_deaths"],
                                x["drops"], x["timeouts"], int(bool(x["disqualified"]))])
                    rows += 1
    return rows


def pp(c, scale=100.0, digits=1):
    if not c:
        return "n/p"
    mark = c["verdict"].replace("0", "")
    return f"{scale * c['delta']:+.{digits}f}{mark}"


def main():
    tracks = {t: load(p) for t, p in LOGS.items()}
    add_expectimax_mean(tracks["C"])
    played = sorted({x["size"] for g in tracks.values() for fx in g.values() for x in fx.values() if x["size"] >= 5})
    sizes = [s for s in REQUESTED if s in played]
    missing = [s for s in REQUESTED if s not in played]

    key = [{"track": t, "label": l, "variant": v, "control": c,
            "pair": {s: cell(tracks[t], v, c, s) for s in sizes},
            "score": {s: cell(tracks[t], v, c, s, "cand_score") for s in sizes},
            "timeouts": {s: cell(tracks[t], v, c, s, "timeouts") for s in sizes},
            "carrier_deaths": {s: cell(tracks[t], v, c, s, "carrier_deaths") for s in sizes},
            "captures": {s: cell(tracks[t], v, c, s, "captures") for s in sizes}} for t, l, v, c in KEY]
    ranked = [{"track": t, "name": n, "variant": v, "control": c, "pair": {s: cell(tracks[t], v, c, s) for s in sizes}}
              for t, n, v, c in RANKED]
    ev = {f"{t}:{b}": {s: events(tracks[t][b], s) for s in sizes} for t, b in EVENT_BOTS}
    timeouts_all = {f"{t}:{b}": {s: events(fx, s)["timeouts_per_100_games"] for s in sizes}
                    for t, g in tracks.items() for b, fx in g.items() if b != "EXPECTIMAX_MEAN"}
    rows = write_clean(tracks, REPO / "benchmarks" / "games_5plus.csv.gz")
    manifest = json.loads((REPO / "data" / "benchmark_raw" / "MANIFEST.json").read_text(encoding="utf-8"))
    summary = {"source_commit": manifest["commit"], "sizes_played": sizes, "sizes_not_played": missing, "clean_rows": rows,
               "key": key, "ranked": ranked, "events": ev, "timeouts_per_100_games": timeouts_all}
    (REPO / "docs" / "benchmark_5plus.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (REPO / "docs" / "BENCHMARK_5PLUS_TABLES.md").write_text(render(summary, sizes, missing), encoding="utf-8")
    print(f"{rows} clean rows; sizes {sizes}; not played {missing}")


def render(s, sizes, missing) -> str:
    head = " | ".join(str(x) for x in sizes)
    sep = "|---|---|" + "---:|" * len(sizes)
    out = [f"# Per-size benchmark tables, 5+ teams (generated)", "",
           f"Generated by `tools/benchmark_5plus.py` from AI_Test `ctf-benchmark-v3` at commit `{s['source_commit'][:12]}`. "
           f"Sizes {', '.join(map(str, missing))} were not played in either track. `+`/`-`: 95% interval excludes 0; "
           f"`++`/`--`: |z| > 3. Narrative: [BENCHMARK_5PLUS.md](BENCHMARK_5PLUS.md).", "",
           "## Key comparisons: pairwise score, percentage points (variant minus control)", "",
           f"| Track | Comparison | {head} |", sep]
    for r in s["key"]:
        out.append(f"| {r['track']} | {r['label']} | " + " | ".join(pp(r["pair"][x]) for x in sizes) + " |")
    for metric, title, scale, digits in (("score", "captures per game (score)", 1, 2), ("timeouts", "timeouts per 100 games", 100, 1),
                                          ("carrier_deaths", "carrier deaths per game", 1, 2)):
        out += ["", f"## Key comparisons: {title} (variant minus control)", "", f"| Track | Comparison | {head} |", sep]
        for r in s["key"]:
            out.append(f"| {r['track']} | {r['label']} | " + " | ".join(pp(r[metric][x], scale, digits) for x in sizes) + " |")
    out += ["", "## Single factors ranked per size (pairwise pp; rank in brackets)", "", f"| Track | Factor | {head} |", sep]
    ranks = {}
    for x in sizes:
        order = sorted(s["ranked"], key=lambda r: -r["pair"][x]["delta"])
        for i, r in enumerate(order, 1):
            ranks[(r["name"], x)] = i
    for r in s["ranked"]:
        out.append(f"| {r['track']} | {r['name']} | " + " | ".join(f"{pp(r['pair'][x])} ({ranks[(r['name'], x)]})" for x in sizes) + " |")
    out += ["", "## Timeouts per 100 games, every candidate", "", f"| Candidate | {head} |", "|---|" + "---:|" * len(sizes)]
    for b, row in sorted(s["timeouts_per_100_games"].items(), key=lambda kv: -sum(kv[1].values())):
        out.append(f"| {b} | " + " | ".join(f"{row[x]:.1f}" for x in sizes) + " |")
    out += ["", "## In-game events per game (absolute)", "",
            "conversion = captures / pickups; carrier survival = 1 - carrier deaths / pickups; K/D = carrier kills / own carrier deaths.", ""]
    for b, per in s["events"].items():
        out += [f"### {b}", "", "| Size | pair win % | pickups | captures | conversion | carrier deaths | survival | carrier kills | K/D | "
                "kills p10/p50/p90 | deaths p10/p50/p90 | timeouts/100 | games with a timeout |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
        for x in sizes:
            e = per[x]
            out.append(f"| {x} | {100 * e['pair_win']:.1f} | {e['pickups']:.2f} | {e['captures']:.2f} | {e['conversion']:.3f} | "
                       f"{e['carrier_deaths']:.2f} | {e['carrier_survival']:.3f} | {e['carrier_kills']:.2f} | {e['kill_death']:.2f} | "
                       f"{'/'.join(f'{v:g}' for v in e['carrier_kills_p10_p50_p90'])} | {'/'.join(f'{v:g}' for v in e['carrier_deaths_p10_p50_p90'])} | "
                       f"{e['timeouts_per_100_games']:.1f} | {100 * e['games_with_timeout']:.1f}% |")
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
