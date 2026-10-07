"""Paired review of the AI_Test heuristic benchmarks (ctf-benchmark-v3): which module helps, hurts, or does nothing.

    python tools/benchmark_review.py <AI_Test>/ctf-benchmark-v3 [--out docs/BENCHMARK_REVIEW.md]

Both tracks replay identical fixtures (map, flag seed, seat, opponents) for
every candidate, so a variant and its control can be compared fixture by
fixture. For each comparison and team size:

  delta    mean over shared fixtures of (variant - control) pairwise score
           (`pair_win`: share of opponents beaten, ties 0.5; what the BT rating rewards)
  SE       cluster-robust over maps (the seat rotations of one map are correlated)
  verdict  + if the 95% interval is above 0, - if below, 0 otherwise;
           ++ / -- when |z| > 3 (survives the ~200 tests in this table)

"All sizes" weights every size equally, as the official rating does. Each
track contains an identical-code pair (A0 vs B0, C17 vs C18) that measures
run-to-run noise; their intervals must contain 0.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

SIZES = (2, 3, 4, 5, 6, 8, 10, 15)

# (track, label, variant, control, what the difference isolates)
COMPARISONS = [
    ("C", "A/A noise check", "C17_R3_EXPECTIMAX_anytime", "C18_R3_EXPECTIMAX_sampled", "identical code (header comment only)"),
    ("T1", "A/A noise check", "A0_Control_Hybrid_AA", "B0_Control_Hybrid", "identical code"),
    # Foundation and weights
    ("C", "Foundation (checked lookahead + conditional fire), default weights", "C02_Control_NG_foundation", "C01_Control_Hybrid", "CHECKED + FLAGSHOOT, default instead of Hybrid's weights"),
    ("C", "Tuned weights on the foundation", "C22_R3_WEIGHTS_tuned", "C02_Control_NG_foundation", "MOD_WEIGHTS_TUNED"),
    ("C", "Foundation with tuned weights vs Hybrid", "C22_R3_WEIGHTS_tuned", "C01_Control_Hybrid", "CHECKED + FLAGSHOOT (+ tuned weights)"),
    # Single modules on the foundation (all: hybrid macro, default weights)
    ("C", "CFL: proximity-tiered prediction", "C12_R3_CFL_conditional", "C02_Control_NG_foundation", "MOD_CFL"),
    ("C", "EXPECTIMAX (anytime sampled 2-ply)", "C17_R3_EXPECTIMAX_anytime", "C02_Control_NG_foundation", "MOD_EXPECTIMAX (copy 1)"),
    ("C", "EXPECTIMAX (second identical copy)", "C18_R3_EXPECTIMAX_sampled", "C02_Control_NG_foundation", "MOD_EXPECTIMAX (copy 2)"),
    ("C", "CDP: carrier delivery DP", "C11_R3_CDP_horizon", "C02_Control_NG_foundation", "MOD_CDP"),
    ("C", "TERMINAL: endgame decay", "C13_R3_CL_terminal", "C02_Control_NG_foundation", "MOD_TERMINAL"),
    ("C", "DROP: scored voluntary drop", "C14_R3_DROP_scored", "C02_Control_NG_foundation", "MOD_DROP"),
    ("C", "ESCORT", "C16_R3_ESCORT_mech", "C02_Control_NG_foundation", "MOD_ESCORT"),
    ("C", "HEALTHY pickup", "C21_R3_HEALTHY_pickup", "C02_Control_NG_foundation", "MOD_HEALTHY"),
    ("C", "ESCORT + HEALTHY", "C15_R3_ESCORT_healthy", "C02_Control_NG_foundation", "MOD_ESCORT + MOD_HEALTHY"),
    ("C", "HABITS: online habit mixture", "C20_R3_HABITS_adaptive", "C02_Control_NG_foundation", "MOD_HABITS"),
    ("C", "FLAGS: joint flag selection", "C19_R3_FLAGS_joint", "C02_Control_NG_foundation", "MOD_FLAGS"),
    # Omega and its twins
    ("C", "Omega fusion vs Hybrid", "C07_Player5_Omega", "C01_Control_Hybrid", "CHECKED+FLAGSHOOT+DROP+TERMINAL+EXPECTIMAX(3-15)+tuned weights"),
    ("C", "Tuned weights inside Omega", "C07_Player5_Omega", "C09_Player5_Omega_defaultWeights", "MOD_WEIGHTS_TUNED"),
    ("C", "Hybrid macro layer inside Omega", "C10_Player5_Omega_hybridMacro", "C07_Player5_Omega", "MOD_HYBRID_MACRO"),
    # Search genes on Omega
    ("C", "Gene SH: sequential halving", "C06_G_SH_halving", "C07_Player5_Omega", "MOD_SH"),
    ("C", "Gene ASYM: asymmetric plans", "C03_G_ASYM_plans", "C07_Player5_Omega", "MOD_ASYM"),
    ("C", "Gene PLAYOUT: 4-ply", "C04_G_PLAYOUT_4ply", "C07_Player5_Omega", "MOD_PLAYOUT"),
    ("C", "Gene RISK: entropic risk", "C05_G_RISK_entropic", "C07_Player5_Omega", "MOD_RISK"),
    ("C", "All four genes (LIT)", "C08_Player5_Omega_LIT", "C07_Player5_Omega", "SH+ASYM+PLAYOUT+RISK"),
    # Track 1
    ("T1", "Omega fusion vs Hybrid", "Control_Omega", "B0_Control_Hybrid", "same code as C07 vs C01"),
    ("T1", "Round-4 search speed-up", "Control_Omega", "B1_Player5_Omega_r3", "cell-indexed exact rollouts (more samples)"),
    ("T1", "Player2 v005 vs Hybrid", "B2_Player2_v005", "B0_Control_Hybrid", "Player2's 15 ms expectimax bot"),
    ("T1", "Omega vs Player2 v005", "Control_Omega", "B2_Player2_v005", ""),
    ("T1", "Astra joint sampler", "Astra_Joint", "Control_Omega", "decorrelated scenario draws"),
    ("T1", "Astra habits (on joint)", "Astra_Habits", "Astra_Joint", "MOD_HABITS (Astra's version)"),
    ("T1", "Astra carrier DP (on joint)", "Astra_Carrier", "Astra_Joint", "MOD_CDP + endgame boundary"),
    ("T1", "Astra full (on joint)", "Astra_Full", "Astra_Joint", "MOD_HABITS + MOD_CDP"),
    # Composites (one variant against a stronger reference)
    ("T1", "Astra full vs Omega", "Astra_Full", "Control_Omega", "joint sampler + HABITS + CDP"),
    ("T1", "Astra full vs Player2 v005", "Astra_Full", "B2_Player2_v005", ""),
    ("C", "Gene RISK vs Hybrid", "C05_G_RISK_entropic", "C01_Control_Hybrid", "Omega + RISK"),
    ("C", "Omega LIT vs Hybrid", "C08_Player5_Omega_LIT", "C01_Control_Hybrid", "Omega + all four genes"),
    ("C", "CFL branch vs Omega", "C12_R3_CFL_conditional", "C07_Player5_Omega", "foundation + CFL against the Omega fusion"),
]


def load(path: Path) -> dict[str, dict[str, dict]]:
    """candidate -> fixture_id -> game record."""
    opener = gzip.open if path.suffix == ".gz" else open
    out = defaultdict(dict)
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            out[g["candidate"]][g["fixture_id"]] = g
    return out


def paired(games, variant, control, size=None, key="pair_win"):
    """(delta, se, n, discordant share) over fixtures both played, clustered by map."""
    a, b = games[variant], games[control]
    clusters = defaultdict(list)
    for fid, g in a.items():
        h = b.get(fid)
        if h is None or (size is not None and g["size"] != size):
            continue
        clusters[fid.rsplit("-", 1)[0]].append(float(g[key]) - float(h[key]))
    d = [x for c in clusters.values() for x in c]
    n = len(d)
    if n < 2:
        return None
    mean = sum(d) / n
    # Cluster-robust variance of the mean: sum over clusters of (sum of residuals)^2 / n^2.
    var = sum(sum(x - mean for x in c) ** 2 for c in clusters.values()) / n ** 2
    k = len(clusters)
    var *= k / max(k - 1, 1)
    return mean, math.sqrt(var), n, sum(x != 0 for x in d) / n


def verdict(mean, se):
    if se == 0:
        return "0" if mean == 0 else ("+" if mean > 0 else "-")
    z = mean / se
    return "++" if z > 3 else "+" if z > 1.96 else "--" if z < -3 else "-" if z < -1.96 else "0"


def combine(cells):
    """Equal weight per size: mean of size deltas, SE from independent sizes."""
    cells = [c for c in cells if c]
    if not cells:
        return None
    m = sum(c[0] for c in cells) / len(cells)
    se = math.sqrt(sum(c[1] ** 2 for c in cells)) / len(cells)
    return m, se, sum(c[2] for c in cells), sum(c[3] * c[2] for c in cells) / sum(c[2] for c in cells)


def review(bench: Path):
    tracks = {"C": load(bench / "track2_2k_c_series" / "results" / "games_c_2000.jsonl.gz"),
              "T1": load(bench / "track1_10k_suite" / "results" / "games.jsonl.gz")}
    rows = []
    for track, label, variant, control, note in COMPARISONS:
        g = tracks[track]
        if variant not in g or control not in g:
            raise SystemExit(f"missing {variant} or {control} in track {track}")
        per = {s: paired(g, variant, control, s) for s in SIZES}
        multi = combine([per[s] for s in SIZES if s != 2])
        rows.append({"track": track, "label": label, "variant": variant, "control": control, "note": note,
                     "all": combine(list(per.values())), "multi": multi, "sizes": per,
                     "score_all": combine([paired(g, variant, control, s, "cand_score") for s in SIZES])})
    return rows


def fmt(cell, scale=100.0):
    if not cell:
        return "-"
    m, se = cell[0] * scale, cell[1] * scale
    return f"{m:+.1f}{verdict(cell[0], cell[1]).replace('0', '')}"


def markdown(rows) -> str:
    out = ["# Benchmark review: which heuristic modules help", "",
           "Generated by `tools/benchmark_review.py` from AI_Test `ctf-benchmark-v3` per-game records "
           "(track 1: 90,207 games, 9 candidates x ~10,000 fixtures; C-series: 44,000 games, 22 candidates x 2,000).",
           "", "Cells: paired difference in **pairwise score, percentage points** (variant minus control, same fixtures), "
           "`+`/`-` = 95% interval excludes 0, `++`/`--` = |z| > 3. 'All' weights the eight sizes equally like the rating; "
           "'3-15' excludes duels. Standard errors are clustered by map.", "",
           "| Track | Comparison | All | 3-15 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 15 | Score/game (all) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sizes = " | ".join(fmt(r["sizes"][s]) for s in SIZES)
        score = r["score_all"]
        out.append(f"| {r['track']} | {r['label']} | **{fmt(r['all'])}** | {fmt(r['multi'])} | {sizes} | "
                   f"{score[0]:+.2f}{verdict(score[0], score[1]).replace('0', '')} |")
    out += ["", "Variant / control per row:", ""]
    for r in rows:
        a = r["all"]
        out.append(f"- {r['label']}: `{r['variant']}` vs `{r['control']}` ({r['note']}); all sizes "
                   f"{100 * a[0]:+.2f} pp, 95% CI [{100 * (a[0] - 1.96 * a[1]):+.2f}, {100 * (a[0] + 1.96 * a[1]):+.2f}], "
                   f"{a[2]} fixtures, outcome differs on {100 * a[3]:.0f}% of them")
    return "\n".join(out) + "\n"


def duel_losses(bench: Path) -> list[str]:
    """Which duel fixtures each candidate loses: the same set for every variant means a deterministic blind spot."""
    lines = []
    for track, path, ref in (("T1", bench / "track1_10k_suite" / "results" / "games.jsonl.gz", "B0_Control_Hybrid"),
                             ("C", bench / "track2_2k_c_series" / "results" / "games_c_2000.jsonl.gz", "C01_Control_Hybrid")):
        g = load(path)
        lost = {c: {f for f, x in fx.items() if x["size"] == 2 and x["pair_win"] < 1} for c, fx in g.items()}
        base = lost[ref]
        parts = [f"{c}: {len(v)} ({len(v & base)} shared)" for c, v in sorted(lost.items())]
        lines.append(f"- {track}: duels lost against Baseline, with the overlap with {ref}'s {len(base)}: " + "; ".join(parts))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", help="path to AI_Test/ctf-benchmark-v3")
    ap.add_argument("--out", help="write the Markdown table here")
    ap.add_argument("--json", help="write the raw numbers here")
    a = ap.parse_args()
    rows = review(Path(a.bench))
    text = markdown(rows) + "\n## Duel losses\n\n" + "\n".join(duel_losses(Path(a.bench))) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=list), encoding="utf-8")


if __name__ == "__main__":
    main()
