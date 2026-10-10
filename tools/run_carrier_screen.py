"""Script to execute paired match screens comparing carrier bot variants.

Runs identical scheduled fixtures across multiple candidate variants:
  - Vanilla control (MOD_FUSION_P=0)
  - P-S (Tier S integer neural P)
  - P-M (Tier M integer neural P)
  - P-L (Tier L integer neural P)
  ...
Computes:
  - Per-fixture paired utility differences: Delta u_i = u_i(cand) - u_i(ctrl)
  - Mean paired difference and standard error: SE = std(Delta u) / sqrt(N)
  - 95% Confidence Interval: Delta u +/- 1.96 * SE
  - Paired win / tie / loss rates
  - Turn latency statistics (mean ms, p99 ms, faults)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Ensure repository root is on sys.path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fusion import runner
from fusion.paths import sha256, git_commit
from roster.strength import schedule, play_one


def parse_args():
    p = argparse.ArgumentParser(description="Paired carrier screen")
    p.add_argument("--teams", type=int, nargs="+", default=[2, 5, 8, 15], help="Team counts to test")
    p.add_argument("--maps-per-count", type=int, default=25, help="Number of maps per team count")
    p.add_argument("--seed", type=int, default=20261020, help="Random seed for schedule fixtures")
    p.add_argument("--workers", type=int, default=8, help="Number of parallel match workers")
    p.add_argument("--root", type=str, default="c:/Users/caspe/OneDrive/Code/AI_Test", help="AI_Test root directory")
    p.add_argument("--out", type=str, default="reports/carrier_paired_screen.json", help="Path to output JSON report")
    p.add_argument("--candidates", type=str, nargs="+", default=None,
                   help="Candidate specifications in name=path format, e.g. ctrl=build/test_carrier/ctrl.exe")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    roster_build = REPO / "build" / "roster"

    # Default candidates if none specified
    if not args.candidates:
        build_dir = REPO / "build" / "test_carrier"
        candidates = {
            "control": list(build_dir.glob("C08_Player5_Omega_LIT_control*.exe"))[0],
            "p_s": list(build_dir.glob("C08_Player5_Omega_LIT_p_s*.exe"))[0],
            "p_m": list(build_dir.glob("C08_Player5_Omega_LIT_p_m*.exe"))[0],
        }
        l_bins = list(build_dir.glob("C08_Player5_Omega_LIT_p_l*.exe"))
        if l_bins:
            candidates["p_l"] = l_bins[0]
    else:
        candidates = {}
        for item in args.candidates:
            name, path = item.split("=", 1)
            candidates[name] = Path(path).resolve()

    print(f"Candidates to test ({len(candidates)}):")
    for name, path in candidates.items():
        print(f"  {name:10s} -> {path} (exists: {path.exists()})")
        assert path.exists(), f"Binary {path} does not exist!"

    ref_names = ["Astra", "Sol", "Fable", "Sonnet", "Opus", "DeepSeek", "Qwen", "MiMo", "Kimi", "GLM", "Player5", "Baseline"]
    sched = schedule(args.maps_per_count, args.teams, ref_names, args.seed, root)
    fixtures = sched["fixtures"]
    total_fixtures = len(fixtures)
    total_matches = total_fixtures * len(candidates)
    print(f"\nGenerated schedule: {len(args.teams)} counts {args.teams}, {args.maps_per_count} maps/count.")
    print(f"Total fixtures: {total_fixtures}, Total candidate matches: {total_matches}")

    # Build reference bot commands
    pool = {e["name"]: e for e in json.loads((REPO / "profiles" / "pool.json").read_text())["bots"]}
    ref_commands = {}
    info = {}
    for r in sched["reference"]:
        bpath = root / pool[r["name"]]["path"]
        ref_commands[r["name"]] = runner.command_for(bpath, roster_build)
        info[r["name"]] = {"family": pool[r["name"]]["family"], "sha256": r["source_sha256"]}
    info["@candidate"] = {"family": "candidate", "sha256": "test"}
    info["__engine__"] = sha256(root / "ctf-arena" / "Player7" / "workspace" / "engine.py")
    info["__commit__"] = git_commit(root)

    limits = (2.0, 0.05)  # 2.0s first turn, 50ms later

    # Create jobs for all (candidate, fixture) pairs
    jobs = []
    for cand_name, cand_exe in candidates.items():
        cand_dict = {"id": cand_name, "key": cand_name, "source_sha256": cand_name, "schedule": "carrier_screen"}
        cmds = dict(ref_commands)
        cmds["@candidate"] = [str(cand_exe)]
        for fx in fixtures:
            job_tuple = (fx, cand_dict, cmds, info, root, limits, None)
            jobs.append((cand_name, fx["fixture_id"], fx["teams"], job_tuple))

    print(f"Running {len(jobs)} matches across {args.workers} workers...")
    t0 = time.time()
    results_by_cand_fx = defaultdict(dict)
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {pool_exec.submit(play_one, job[3]): (job[0], job[1], job[2]) for job in jobs}
        for fut in as_completed(futures):
            c_name, fx_id, teams = futures[fut]
            try:
                res = fut.result()
                results_by_cand_fx[c_name][fx_id] = res
                completed += 1
                if completed % max(1, total_matches // 10) == 0 or completed == total_matches:
                    elap = time.time() - t0
                    print(f"Progress: {completed}/{total_matches} matches ({completed/total_matches*100:.1f}%) in {elap:.1f}s ({completed/elap:.2f} matches/s)", flush=True)
            except Exception as e:
                print(f"Error on {c_name} {fx_id}: {e}", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nAll {completed} matches finished in {elapsed_total:.1f}s.")

    # Statistical Analysis
    analysis = {
        "metadata": {
            "teams": args.teams,
            "maps_per_count": args.maps_per_count,
            "seed": args.seed,
            "candidates": list(candidates.keys()),
            "elapsed_seconds": elapsed_total,
        },
        "per_count": {},
        "aggregate": {},
    }

    # Reference control candidate
    ctrl_key = "control" if "control" in candidates else list(candidates.keys())[0]

    for n in args.teams:
        n_fxs = [f["fixture_id"] for f in fixtures if f["teams"] == n]
        analysis["per_count"][n] = {}
        for c_name in candidates.keys():
            utils = [results_by_cand_fx[c_name][fid]["utility"] for fid in n_fxs if fid in results_by_cand_fx[c_name]]
            p99s = [results_by_cand_fx[c_name][fid]["p99_ms"] for fid in n_fxs if fid in results_by_cand_fx[c_name]]
            faults = sum(results_by_cand_fx[c_name][fid]["faults"] for fid in n_fxs if fid in results_by_cand_fx[c_name])
            
            u_mean = float(np.mean(utils)) if utils else 0.0
            u_se = float(np.std(utils, ddof=1) / math.sqrt(len(utils))) if len(utils) > 1 else 0.0

            cand_stat = {
                "n_maps": len(utils),
                "mean_utility": u_mean,
                "se_utility": u_se,
                "p99_ms_mean": float(np.mean(p99s)) if p99s else 0.0,
                "faults": faults,
            }

            # Paired differences against control
            if c_name != ctrl_key:
                diffs = []
                wins, ties, losses = 0, 0, 0
                for fid in n_fxs:
                    if fid in results_by_cand_fx[c_name] and fid in results_by_cand_fx[ctrl_key]:
                        u_cand = results_by_cand_fx[c_name][fid]["utility"]
                        u_ctrl = results_by_cand_fx[ctrl_key][fid]["utility"]
                        d = u_cand - u_ctrl
                        diffs.append(d)
                        if d > 1e-6:
                            wins += 1
                        elif d < -1e-6:
                            losses += 1
                        else:
                            ties += 1
                if diffs:
                    d_mean = float(np.mean(diffs))
                    d_se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
                    cand_stat["paired"] = {
                        "diff_mean": d_mean,
                        "diff_se": d_se,
                        "ci95_lower": d_mean - 1.96 * d_se,
                        "ci95_upper": d_mean + 1.96 * d_se,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                    }

            analysis["per_count"][n][c_name] = cand_stat

    # Overall Aggregate (equal weight across tested team counts)
    for c_name in candidates.keys():
        all_u_means = [analysis["per_count"][n][c_name]["mean_utility"] for n in args.teams]
        mean_u_agg = float(np.mean(all_u_means))
        agg_stat = {"equal_count_mean_utility": mean_u_agg}

        if c_name != ctrl_key:
            all_diffs_means = [analysis["per_count"][n][c_name]["paired"]["diff_mean"] for n in args.teams if "paired" in analysis["per_count"][n][c_name]]
            all_diffs_raw = []
            for n in args.teams:
                n_fxs = [f["fixture_id"] for f in fixtures if f["teams"] == n]
                for fid in n_fxs:
                    if fid in results_by_cand_fx[c_name] and fid in results_by_cand_fx[ctrl_key]:
                        all_diffs_raw.append(results_by_cand_fx[c_name][fid]["utility"] - results_by_cand_fx[ctrl_key][fid]["utility"])
            d_raw_mean = float(np.mean(all_diffs_raw)) if all_diffs_raw else 0.0
            d_raw_se = float(np.std(all_diffs_raw, ddof=1) / math.sqrt(len(all_diffs_raw))) if len(all_diffs_raw) > 1 else 0.0
            agg_stat["paired"] = {
                "equal_count_diff_mean": float(np.mean(all_diffs_means)) if all_diffs_means else 0.0,
                "raw_diff_mean": d_raw_mean,
                "raw_diff_se": d_raw_se,
                "ci95_lower": d_raw_mean - 1.96 * d_raw_se,
                "ci95_upper": d_raw_mean + 1.96 * d_raw_se,
            }
        analysis["aggregate"][c_name] = agg_stat

    # Save JSON report
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out_path}")

    # Save per-match JSONL
    matches_jsonl = out_path.with_suffix(".jsonl")
    with matches_jsonl.open("w", encoding="utf-8") as f_jsonl:
        for fx in fixtures:
            fid = fx["fixture_id"]
            for c_name in candidates.keys():
                if fid in results_by_cand_fx[c_name]:
                    r = results_by_cand_fx[c_name][fid]
                    row = {
                        "fixture_id": fid,
                        "teams": fx["teams"],
                        "map_seed": fx["map_seed"],
                        "flag_seed": fx["flag_seed"],
                        "seat": fx["seat"],
                        "opponents": fx["opponents"],
                        "candidate": c_name,
                        "score": r.get("score"),
                        "utility": r.get("utility"),
                        "p99_ms": r.get("p99_ms"),
                        "mean_ms": r.get("mean_ms"),
                        "turns": r.get("turns"),
                        "faults": r.get("faults"),
                    }
                    f_jsonl.write(json.dumps(row) + "\n")
    print(f"Per-match results saved to {matches_jsonl}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"PAIRED MATCH SCREEN RESULTS (Reference Control = {ctrl_key})")
    print("=" * 90)
    for n in args.teams:
        print(f"\n--- {n} TEAMS ({args.maps_per_count} maps) ---")
        ctrl_u = analysis["per_count"][n][ctrl_key]["mean_utility"]
        print(f"{'Candidate':12s} | {'Utility':10s} | {'Delta vs Ctrl':15s} | {'95% CI':22s} | {'W / T / L':12s} | {'p99 ms':8s}")
        print("-" * 90)
        for c_name in candidates.keys():
            st = analysis["per_count"][n][c_name]
            u_str = f"{st['mean_utility']:.4f}"
            p99_str = f"{st['p99_ms_mean']:.1f}ms"
            if c_name == ctrl_key:
                print(f"{c_name:12s} | {u_str:10s} | {'(baseline)':15s} | {'-':22s} | {'-':12s} | {p99_str:8s}")
            else:
                pair = st["paired"]
                d_str = f"{pair['diff_mean']:+.4f} (SE {pair['diff_se']:.4f})"
                ci_str = f"[{pair['ci95_lower']:+.4f}, {pair['ci95_upper']:+.4f}]"
                wtl_str = f"{pair['wins']} / {pair['ties']} / {pair['losses']}"
                print(f"{c_name:12s} | {u_str:10s} | {d_str:15s} | {ci_str:22s} | {wtl_str:12s} | {p99_str:8s}")

    print("\n" + "=" * 90)
    print("EQUAL-COUNT AGGREGATE SUMMARY")
    print("=" * 90)
    print(f"{'Candidate':12s} | {'Equal-Count Mean Utility':26s} | {'Mean Delta vs Ctrl':20s} | {'Aggregate 95% CI':24s}")
    print("-" * 90)
    for c_name in candidates.keys():
        agg = analysis["aggregate"][c_name]
        u_str = f"{agg['equal_count_mean_utility']:.4f}"
        if c_name == ctrl_key:
            print(f"{c_name:12s} | {u_str:26s} | {'(baseline)':20s} | {'-':24s}")
        else:
            p = agg["paired"]
            d_str = f"{p['equal_count_diff_mean']:+.4f}"
            ci_str = f"[{p['ci95_lower']:+.4f}, {p['ci95_upper']:+.4f}]"
            print(f"{c_name:12s} | {u_str:26s} | {d_str:20s} | {ci_str:24s}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
