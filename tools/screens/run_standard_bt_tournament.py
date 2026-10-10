"""Standard 1,000-Game Tournament and Bradley-Terry (BT) Rating Benchmark.

Evaluates 16 contenders:
  - 2 pvL Enhanced Models: C08_pvL, v12_pvL
  - 2 Control Baseline Models: C08_ctrl, v12_ctrl
  - 12 Official Reference Benchmarks: Astra, Sol, Fable, Sonnet, Opus, DeepSeek,
                                      Qwen, MiMo, Kimi, GLM, Player5 (anchor), Baseline

Across 1,000 games on the 8 official benchmark team sizes (2, 3, 4, 5, 6, 8, 10, 15):
  125 games per size = 1,000 games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fusion import matchlog, runner
from fusion.paths import sha256, git_commit

# Import official bt module from AI_Test
AI_TEST_ROOT = Path("c:/Users/caspe/OneDrive/Code/AI_Test").resolve()
sys.path.insert(0, str(AI_TEST_ROOT / "ctf-stress-pack"))
import bt


def derive(*parts, bits=31) -> int:
    h = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") % (1 << bits)


def parse_args():
    p = argparse.ArgumentParser(description="Standard 1000-game BT Tournament")
    p.add_argument("--total-games", type=int, default=1000, help="Total games in tournament")
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 15], help="Team sizes")
    p.add_argument("--seed", type=int, default=20261020, help="Random seed")
    p.add_argument("--workers", type=int, default=10, help="Parallel worker processes")
    p.add_argument("--out-json", type=str, default="reports/eval/tournament_1000_bt.json", help="Summary JSON output")
    p.add_argument("--out-jsonl", type=str, default="reports/eval/tournament_1000_matches.jsonl", help="Per-game JSONL output")
    p.add_argument("--from-jsonl", type=str, default=None, help="Compute BT ratings directly from existing JSONL")
    return p.parse_args()


def play_game(job):
    """Execute one match between N bots."""
    gid, size, map_seed, flag_seed, seats, commands, info, root, limits = job
    from fusion.paths import load_engine
    engine = load_engine(Path(root))
    game = engine.Game(size, map_seed, flag_seed=flag_seed)
    bots = [runner.Bot(commands[name], name, tuple(limits)) for name in seats]
    faults_count = defaultdict(int)
    p99_ms = {}

    try:
        while not game.done:
            turn = game.turn + 1
            inputs = [game.encode(i) for i in range(size)]
            starts = [b.send(text) for b, text in zip(bots, inputs)]
            lines = [b.receive(t0, turn) for b, t0 in zip(bots, starts)]
            orders = {}
            for i, (b, line) in enumerate(zip(bots, lines)):
                if line is None:
                    orders[i] = {}
                    continue
                try:
                    orders[i] = game.decode(i, line)
                    b.ok()
                except ValueError as exc:
                    orders[i] = {}
                    b.strike(turn, f"malformed output: {exc}")
            game.step(orders)
    finally:
        for i, b in enumerate(bots):
            rep = b.report()
            p99_ms[seats[i]] = rep.get("p99_ms") or 0.0
            faults_count[seats[i]] = rep.get("faults") or 0
            b.close()

    result = game.result()
    score = list(result["score"])

    # Compute pairwise utility (share of opponents beaten, ties 0.5)
    pw = {}
    top_score = max(score)
    is_win = {}
    for i, name in enumerate(seats):
        sc_i = score[i]
        others = [s for j, s in enumerate(score) if j != i]
        pw[name] = sum(1.0 if sc_i > s else 0.5 if sc_i == s else 0.0 for s in others) / len(others)
        is_win[name] = (sc_i == top_score)

    return {
        "gid": gid,
        "size": size,
        "map_seed": map_seed,
        "flag_seed": flag_seed,
        "seats": seats,
        "score": score,
        "pair_win": pw,
        "is_win": is_win,
        "p99_ms": dict(p99_ms),
        "faults": dict(faults_count),
    }


def main():
    args = parse_args()
    root = AI_TEST_ROOT
    build_dir = REPO / "build" / "test_carrier"
    roster_build = REPO / "build" / "roster"

    # Define all 16 contenders
    contender_bins = {
        "C08_pvL": list(build_dir.glob("C08_Player5_Omega_LIT_pvl*.exe"))[0],
        "v12_pvL": list(build_dir.glob("Player5_v12_pvl*.exe"))[0],
        "C08_ctrl": list(build_dir.glob("C08_Player5_Omega_LIT_control*.exe"))[0],
        "v12_ctrl": list(build_dir.glob("Player5_v12_control*.exe"))[0],
    }

    # Reference bots from pool.json
    pool = {e["name"]: e for e in json.loads((REPO / "profiles" / "pool.json").read_text())["bots"]}
    ref_names = ["Astra", "Sol", "Fable", "Sonnet", "Opus", "DeepSeek", "Qwen", "MiMo", "Kimi", "GLM", "Player5", "Baseline"]
    
    commands = {}
    info = {}
    for name in ref_names:
        bpath = root / pool[name]["path"]
        commands[name] = runner.command_for(bpath, roster_build)
        info[name] = {"family": pool[name]["family"], "sha256": sha256(bpath)}

    for name, exe in contender_bins.items():
        assert exe.exists(), f"Binary {exe} does not exist!"
        commands[name] = [str(exe)]
        info[name] = {"family": "candidate", "sha256": sha256(exe)}

    all_contenders = ["C08_pvL", "v12_pvL", "C08_ctrl", "v12_ctrl"] + ref_names
    print(f"Total contenders ({len(all_contenders)}):")
    for name in all_contenders:
        print(f"  {name:12s} -> {commands[name][0]}")

    info["__engine__"] = sha256(root / "ctf-arena" / "Player7" / "workspace" / "engine.py")
    info["__commit__"] = git_commit(root)
    limits = (2.0, 0.05)

    games_per_size = args.total_games // len(args.sizes)
    print(f"\nTournament schedule: {len(args.sizes)} sizes {args.sizes}, {games_per_size} games/size = {len(args.sizes)*games_per_size} games.")

    out_jsonl = REPO / args.out_jsonl
    out_json = REPO / args.out_json
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    game_records = []

    if args.from_jsonl:
        print(f"Loading tournament results from {args.from_jsonl}...")
        for line in Path(args.from_jsonl).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                game_records.append(json.loads(line))
        print(f"Loaded {len(game_records)} games.")
    else:
        # Generate scheduled games
        jobs = []
        for n in args.sizes:
            for m in range(games_per_size):
                gid = f"std1k-n{n:02d}-m{m:04d}"
                rng = random.Random(derive(args.seed, n, m, "seats"))
                seats = rng.sample(all_contenders, n)
                map_seed = derive(args.seed, n, m, "map")
                flag_seed = derive(args.seed, n, m, "flag", bits=63)
                job_tuple = (gid, n, map_seed, flag_seed, seats, commands, info, root, limits)
                jobs.append(job_tuple)

        print(f"Launching {len(jobs)} matches across {args.workers} workers...")
        t0 = time.time()
        completed = 0

        with out_jsonl.open("w", encoding="utf-8") as f_out:
            with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
                futures = {pool_exec.submit(play_game, j): j[0] for j in jobs}
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        game_records.append(res)
                        f_out.write(json.dumps(res) + "\n")
                        f_out.flush()
                        completed += 1
                        if completed % max(1, len(jobs) // 10) == 0 or completed == len(jobs):
                            dt = time.time() - t0
                            print(f"Progress: {completed}/{len(jobs)} games ({completed/len(jobs)*100:.1f}%) in {dt:.1f}s ({completed/dt:.2f} games/s)", flush=True)
                    except Exception as e:
                        print(f"Error on match {futures[fut]}: {e}", flush=True)

        elapsed = time.time() - t0
        print(f"\nAll {len(game_records)} games completed in {elapsed:.1f}s.")

    # Compute Bradley-Terry Ratings using official bt.py
    print("\nFitting Bradley-Terry ratings (anchor = Player5)...")
    bt_rows = [{
        "size": g["size"],
        "seats": g["seats"],
        "score": g["score"],
        "map_seed": g["map_seed"]
    } for g in game_records]

    ci_ratings, per_size_bt = bt.rate(bt_rows, all_contenders, anchor="Player5", boot=200, seed=args.seed)

    # Compute additional statistics: pair_win, first_place win rate, p99 ms, faults
    stats_by_bot = defaultdict(lambda: {
        "games": 0,
        "pair_wins": [],
        "first_places": 0,
        "p99_ms_list": [],
        "faults": 0,
        "per_size_pw": defaultdict(list),
        "per_size_games": defaultdict(int),
    })

    for g in game_records:
        n = g["size"]
        for seat_name in g["seats"]:
            st = stats_by_bot[seat_name]
            st["games"] += 1
            st["pair_wins"].append(g["pair_win"][seat_name])
            if g["is_win"].get(seat_name, False):
                st["first_places"] += 1
            if seat_name in g["p99_ms"]:
                st["p99_ms_list"].append(g["p99_ms"][seat_name])
            if seat_name in g["faults"]:
                st["faults"] += g["faults"][seat_name]
            st["per_size_pw"][n].append(g["pair_win"][seat_name])
            st["per_size_games"][n] += 1

    summary_bots = {}
    for name in all_contenders:
        st = stats_by_bot[name]
        pt, ci_low, ci_high = ci_ratings[name]
        mean_pw = float(np.mean(st["pair_wins"])) if st["pair_wins"] else 0.0
        se_pw = float(np.std(st["pair_wins"], ddof=1) / math.sqrt(len(st["pair_wins"]))) if len(st["pair_wins"]) > 1 else 0.0
        win_rate = st["first_places"] / st["games"] if st["games"] > 0 else 0.0
        p99_mean = float(np.mean(st["p99_ms_list"])) if st["p99_ms_list"] else 0.0

        per_sz_pw_means = {str(k): float(np.mean(st["per_size_pw"][k])) if st["per_size_pw"][k] else 0.0 for k in args.sizes}
        per_sz_bt_scores = {str(k): per_size_bt.get(k, {}).get(name, 0.0) for k in args.sizes}

        summary_bots[name] = {
            "bt_rating": pt,
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "games_played": st["games"],
            "mean_pair_win": mean_pw,
            "se_pair_win": se_pw,
            "win_rate": win_rate,
            "p99_ms": p99_mean,
            "faults": st["faults"],
            "per_size_bt": per_sz_bt_scores,
            "per_size_pw": per_sz_pw_means,
        }

    # Sort bots by BT rating descending
    ranked_bots = sorted(all_contenders, key=lambda n: summary_bots[n]["bt_rating"], reverse=True)

    analysis = {
        "metadata": {
            "total_games": len(game_records),
            "sizes": args.sizes,
            "seed": args.seed,
            "anchor": "Player5",
            "contenders": all_contenders,
        },
        "ranking": ranked_bots,
        "bots": summary_bots,
        "pvL_improvements": {
            "C08": {
                "pvL_bt": summary_bots["C08_pvL"]["bt_rating"],
                "ctrl_bt": summary_bots["C08_ctrl"]["bt_rating"],
                "delta_bt": summary_bots["C08_pvL"]["bt_rating"] - summary_bots["C08_ctrl"]["bt_rating"],
                "delta_pw": summary_bots["C08_pvL"]["mean_pair_win"] - summary_bots["C08_ctrl"]["mean_pair_win"],
            },
            "v12": {
                "pvL_bt": summary_bots["v12_pvL"]["bt_rating"],
                "ctrl_bt": summary_bots["v12_ctrl"]["bt_rating"],
                "delta_bt": summary_bots["v12_pvL"]["bt_rating"] - summary_bots["v12_ctrl"]["bt_rating"],
                "delta_pw": summary_bots["v12_pvL"]["mean_pair_win"] - summary_bots["v12_ctrl"]["mean_pair_win"],
            }
        }
    }

    out_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_json}")
    print(f"Per-match logs saved to {out_jsonl}")

    # Print Summary Tables
    print("\n" + "=" * 105)
    print("OFFICIAL STANDARD 1,000-GAME TOURNAMENT: BRADLEY-TERRY (BT) LEADERBOARD")
    print("=" * 105)
    print(f"{'Rank':4s} | {'Bot Name':12s} | {'BT Rating':10s} | {'95% CI':22s} | {'Pair Win %':12s} | {'Win Rate %':12s} | {'p99 ms':8s}")
    print("-" * 105)
    for rank, name in enumerate(ranked_bots, start=1):
        b = summary_bots[name]
        bt_str = f"{b['bt_rating']:+7.1f}"
        ci_str = f"[{b['ci95_lower']:+6.1f}, {b['ci95_upper']:+6.1f}]"
        pw_str = f"{b['mean_pair_win']*100:5.2f}%"
        win_str = f"{b['win_rate']*100:5.2f}%"
        p99_str = f"{b['p99_ms']:5.1f}ms"
        tag = ""
        if name in ("C08_pvL", "v12_pvL"):
            tag = "  <-- [pvL]"
        elif name in ("C08_ctrl", "v12_ctrl"):
            tag = "  <-- (ctrl)"
        elif name == "Player5":
            tag = "  [Anchor = 0]"
        print(f"#{rank:2d}  | {name:12s} | {bt_str:10s} | {ci_str:22s} | {pw_str:12s} | {win_str:12s} | {p99_str:8s}{tag}")
    print("=" * 105)

    print("\n" + "=" * 80)
    print("pvL ENHANCEMENT DELTA ANALYSIS (vs Control & Official Benchmarks)")
    print("=" * 80)
    c08_imp = analysis["pvL_improvements"]["C08"]
    v12_imp = analysis["pvL_improvements"]["v12"]
    print(f"1. Carrier 1 (C08_Omega_LIT):")
    print(f"   Control BT: {c08_imp['ctrl_bt']:+6.1f}  -->  pvL BT: {c08_imp['pvL_bt']:+6.1f}  |  Delta BT: {c08_imp['delta_bt']:+6.1f} pts (Pair Win: {c08_imp['delta_pw']*100:+.2f}%)")
    print(f"2. Carrier 2 (Player5_v12):")
    print(f"   Control BT: {v12_imp['ctrl_bt']:+6.1f}  -->  pvL BT: {v12_imp['pvL_bt']:+6.1f}  |  Delta BT: {v12_imp['delta_bt']:+6.1f} pts (Pair Win: {v12_imp['delta_pw']*100:+.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
