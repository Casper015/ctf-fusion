"""Paired ablation experiment script: evaluates the exact impact of Value Network (V)
by comparing:
  1. control    : Vanilla baseline (no P, no V)
  2. pm_no_v    : Policy M only, Value V disabled ("卸掉V")
  3. pm_with_v  : Policy M + Value M enabled

Computes paired deltas:
  - Delta P alone    : pm_no_v - control
  - Delta V marginal : pm_with_v - pm_no_v   <-- Direct measurement of V's impact
  - Delta PV combined: pm_with_v - control
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fusion import runner
from fusion.paths import sha256, git_commit
from roster.strength import schedule, play_one


def parse_args():
    p = argparse.ArgumentParser(description="Paired V Ablation Screen")
    p.add_argument("--carrier", type=str, choices=["C08", "v12"], default="C08", help="Carrier bot name")
    p.add_argument("--teams", type=int, nargs="+", default=[2, 5, 8, 15], help="Team counts to test")
    p.add_argument("--maps-per-count", type=int, default=20, help="Number of maps per team count")
    p.add_argument("--seed", type=int, default=20261020, help="Random seed for schedule fixtures")
    p.add_argument("--workers", type=int, default=8, help="Number of parallel match workers")
    p.add_argument("--root", type=str, default="c:/Users/caspe/OneDrive/Code/AI_Test", help="AI_Test root directory")
    p.add_argument("--out-prefix", type=str, default=None, help="Output file prefix under reports/")
    p.add_argument("--from-jsonl", type=str, default=None, help="Process existing JSONL instead of re-running matches")
    return p.parse_args()


def pair_stats(diffs):
    if not diffs:
        return {"diff_mean": 0.0, "diff_se": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0, "wins": 0, "ties": 0, "losses": 0}
    d_mean = float(np.mean(diffs))
    d_se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    wins = sum(1 for d in diffs if d > 1e-6)
    ties = sum(1 for d in diffs if abs(d) <= 1e-6)
    losses = sum(1 for d in diffs if d < -1e-6)
    return {
        "diff_mean": d_mean,
        "diff_se": d_se,
        "ci95_lower": d_mean - 1.96 * d_se,
        "ci95_upper": d_mean + 1.96 * d_se,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    roster_build = REPO / "build" / "roster"
    build_dir = REPO / "build" / "test_carrier"

    if args.carrier == "C08":
        candidates = {
            "control": list(build_dir.glob("C08_Player5_Omega_LIT_control*.exe"))[0],
            "pm_no_v": list(build_dir.glob("C08_Player5_Omega_LIT_pm_no_v*.exe"))[0],
            "pm_with_v": list(build_dir.glob("C08_Player5_Omega_LIT_pm_with_v*.exe"))[0],
        }
        default_prefix = "reports/v_ablation_c08"
    else:
        candidates = {
            "control": list(build_dir.glob("Player5_v12_control*.exe"))[0],
            "pm_no_v": list(build_dir.glob("Player5_v12_pm_no_v*.exe"))[0],
            "pm_with_v": list(build_dir.glob("Player5_v12_pm_with_v*.exe"))[0],
        }
        default_prefix = "reports/v_ablation_v12"

    out_prefix = args.out_prefix or default_prefix
    out_json = REPO / f"{out_prefix}.json"
    out_jsonl = REPO / f"{out_prefix}.jsonl"

    print(f"Carrier: {args.carrier}")
    print("Candidates:")
    for k, v in candidates.items():
        print(f"  {k:12s} -> {v.name}")
        assert v.exists(), f"Binary {v} does not exist!"

    ref_names = ["Astra", "Sol", "Fable", "Sonnet", "Opus", "DeepSeek", "Qwen", "MiMo", "Kimi", "GLM", "Player5", "Baseline"]
    sched = schedule(args.maps_per_count, args.teams, ref_names, args.seed, root)
    fixtures = sched["fixtures"]
    total_fixtures = len(fixtures)
    total_matches = total_fixtures * len(candidates)
    print(f"\nGenerated schedule: {len(args.teams)} counts {args.teams}, {args.maps_per_count} maps/count.")
    print(f"Total fixtures: {total_fixtures}, Total candidate matches: {total_matches}")

    results_by_cand_fx = defaultdict(dict)
    elapsed_total = 0.0

    if args.from_jsonl:
        in_path = Path(args.from_jsonl).resolve()
        print(f"Loading match results from {in_path}...")
        for line in in_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            results_by_cand_fx[row["candidate"]][row["fixture_id"]] = row
        print(f"Loaded {sum(len(v) for v in results_by_cand_fx.values())} match records.")
    else:
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

        jobs = []
        for cand_name, cand_exe in candidates.items():
            cand_dict = {"id": cand_name, "key": cand_name, "source_sha256": cand_name, "schedule": "v_ablation"}
            cmds = dict(ref_commands)
            cmds["@candidate"] = [str(cand_exe)]
            for fx in fixtures:
                job_tuple = (fx, cand_dict, cmds, info, root, limits, None)
                jobs.append((cand_name, fx["fixture_id"], fx["teams"], job_tuple))

        print(f"\nLaunching {len(jobs)} matches across {args.workers} workers...")
        t0 = time.time()
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

        # Save per-match JSONL
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f_jsonl:
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
                            "carrier": args.carrier,
                            "candidate": c_name,
                            "score": r.get("score"),
                            "utility": r.get("utility"),
                            "p99_ms": r.get("p99_ms"),
                            "mean_ms": r.get("mean_ms"),
                            "turns": r.get("turns"),
                            "faults": r.get("faults"),
                        }
                        f_jsonl.write(json.dumps(row) + "\n")
        print(f"Per-match results saved to {out_jsonl}")

    # Statistical Analysis
    analysis = {
        "metadata": {
            "carrier": args.carrier,
            "teams": args.teams,
            "maps_per_count": args.maps_per_count,
            "seed": args.seed,
            "candidates": list(candidates.keys()),
            "elapsed_seconds": elapsed_total,
        },
        "per_count": {},
        "aggregate": {},
    }

    # For each count, calculate stats and 3 paired comparisons:
    # 1. p_alone: pm_no_v - control
    # 2. v_marginal: pm_with_v - pm_no_v
    # 3. pv_total: pm_with_v - control
    all_diffs_raw = {"p_alone": [], "v_marginal": [], "pv_total": []}

    for n in args.teams:
        n_fxs = [f["fixture_id"] for f in fixtures if f["teams"] == n]
        analysis["per_count"][n] = {"candidates": {}, "comparisons": {}}

        # Candidate stats
        for c_name in candidates.keys():
            utils = [results_by_cand_fx[c_name][fid]["utility"] for fid in n_fxs if fid in results_by_cand_fx[c_name]]
            p99s = [results_by_cand_fx[c_name][fid].get("p99_ms") or 0.0 for fid in n_fxs if fid in results_by_cand_fx[c_name]]
            means = [results_by_cand_fx[c_name][fid].get("mean_ms") or 0.0 for fid in n_fxs if fid in results_by_cand_fx[c_name]]
            faults = sum(results_by_cand_fx[c_name][fid].get("faults", 0) for fid in n_fxs if fid in results_by_cand_fx[c_name])
            
            u_mean = float(np.mean(utils)) if utils else 0.0
            u_se = float(np.std(utils, ddof=1) / math.sqrt(len(utils))) if len(utils) > 1 else 0.0

            analysis["per_count"][n]["candidates"][c_name] = {
                "n_maps": len(utils),
                "mean_utility": u_mean,
                "se_utility": u_se,
                "mean_ms": float(np.mean(means)) if means else 0.0,
                "p99_ms": float(np.mean(p99s)) if p99s else 0.0,
                "faults": faults,
            }

        # Paired differences
        d_p = []
        d_v = []
        d_pv = []
        for fid in n_fxs:
            if fid in results_by_cand_fx["pm_no_v"] and fid in results_by_cand_fx["control"]:
                diff = results_by_cand_fx["pm_no_v"][fid]["utility"] - results_by_cand_fx["control"][fid]["utility"]
                d_p.append(diff)
                all_diffs_raw["p_alone"].append(diff)

            if fid in results_by_cand_fx["pm_with_v"] and fid in results_by_cand_fx["pm_no_v"]:
                diff = results_by_cand_fx["pm_with_v"][fid]["utility"] - results_by_cand_fx["pm_no_v"][fid]["utility"]
                d_v.append(diff)
                all_diffs_raw["v_marginal"].append(diff)

            if fid in results_by_cand_fx["pm_with_v"] and fid in results_by_cand_fx["control"]:
                diff = results_by_cand_fx["pm_with_v"][fid]["utility"] - results_by_cand_fx["control"][fid]["utility"]
                d_pv.append(diff)
                all_diffs_raw["pv_total"].append(diff)

        analysis["per_count"][n]["comparisons"]["p_alone (pm_no_v - ctrl)"] = pair_stats(d_p)
        analysis["per_count"][n]["comparisons"]["v_marginal (pm_with_v - pm_no_v)"] = pair_stats(d_v)
        analysis["per_count"][n]["comparisons"]["pv_total (pm_with_v - ctrl)"] = pair_stats(d_pv)

    # Equal-count aggregate
    analysis["aggregate"]["candidates"] = {}
    for c_name in candidates.keys():
        all_u = [analysis["per_count"][n]["candidates"][c_name]["mean_utility"] for n in args.teams]
        analysis["aggregate"]["candidates"][c_name] = {
            "equal_count_mean_utility": float(np.mean(all_u)),
        }

    analysis["aggregate"]["comparisons"] = {}
    for comp_key, raw_list in all_diffs_raw.items():
        # Equal count mean
        if comp_key == "p_alone":
            lbl = "p_alone (pm_no_v - ctrl)"
        elif comp_key == "v_marginal":
            lbl = "v_marginal (pm_with_v - pm_no_v)"
        else:
            lbl = "pv_total (pm_with_v - ctrl)"

        eq_means = [analysis["per_count"][n]["comparisons"][lbl]["diff_mean"] for n in args.teams]
        st = pair_stats(raw_list)
        st["equal_count_diff_mean"] = float(np.mean(eq_means))
        analysis["aggregate"]["comparisons"][lbl] = st

    # Save JSON report
    out_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Summary JSON saved to {out_json}")

    # Print Report Table
    print("\n" + "=" * 105)
    print(f"VALUE NETWORK (V) ABLATION STUDY: {args.carrier} CARRIER")
    print("=" * 105)

    print("\n1. CANDIDATE STANDALONE UTILITIES & LATENCY:")
    print(f"{'Count':7s} | {'Candidate':12s} | {'Mean Utility':14s} | {'p99 ms':10s} | {'Mean ms':10s} | {'Faults':6s}")
    print("-" * 75)
    for n in args.teams:
        for c_name in candidates.keys():
            c_stat = analysis["per_count"][n]["candidates"][c_name]
            print(f"{n:5d}T   | {c_name:12s} | {c_stat['mean_utility']:.4f} +/- {c_stat['se_utility']:.4f} | {c_stat['p99_ms']:7.1f} ms | {c_stat['mean_ms']:7.2f} ms | {c_stat['faults']:6d}")
        print("-" * 75)

    print("\n2. PAIRED HEAD-TO-HEAD COMPARISONS:")
    print(f"{'Count':7s} | {'Comparison':32s} | {'Mean Delta':16s} | {'95% CI':22s} | {'W / T / L':12s}")
    print("-" * 100)
    for n in args.teams:
        comps = analysis["per_count"][n]["comparisons"]
        for c_name, c_res in comps.items():
            d_str = f"{c_res['diff_mean']:+.4f} (SE {c_res['diff_se']:.4f})"
            ci_str = f"[{c_res['ci95_lower']:+.4f}, {c_res['ci95_upper']:+.4f}]"
            wtl_str = f"{c_res['wins']} / {c_res['ties']} / {c_res['losses']}"
            print(f"{n:5d}T   | {c_name:32s} | {d_str:16s} | {ci_str:22s} | {wtl_str:12s}")
        print("-" * 100)

    print("\n3. AGGREGATE SUMMARY ACROSS ALL COUNTS:")
    print("-" * 100)
    for c_name, c_res in analysis["aggregate"]["comparisons"].items():
        d_eq = f"{c_res['equal_count_diff_mean']:+.4f}"
        d_raw = f"{c_res['diff_mean']:+.4f} (SE {c_res['diff_se']:.4f})"
        ci_str = f"[{c_res['ci95_lower']:+.4f}, {c_res['ci95_upper']:+.4f}]"
        wtl_str = f"{c_res['wins']} / {c_res['ties']} / {c_res['losses']}"
        print(f"  {c_name:34s} -> Eq-Mean: {d_eq} | Raw: {d_raw} | CI: {ci_str} | W/T/L: {wtl_str}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
