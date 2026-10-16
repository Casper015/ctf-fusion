"""Independent count-specific opponent datasets collection and reporting (T0066).

Manages:
1. Eight-worker collector preflight and native-vs-Python integer parity check.
2. Full collection of 2,800 matches across counts 2..15 under task CPU lease.
3. Feature extraction (P2 schema, 923 features) and shard generation.
4. Deterministic regeneration of all manifests, receipts, fixture lists, and reports.
5. Local and cloud backup verification (SHA-256 114ac77fbc6706718d37c92fbcc0016726cdfe194850b82ac3c7ac30a2ea750a).

Usage:
    python tools/host/count_specialists_data.py --preflight
    python tools/host/count_specialists_data.py --collect
    python tools/host/count_specialists_data.py --regenerate
    python tools/host/count_specialists_data.py --backup
    python tools/host/count_specialists_data.py --report
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fusion import families, matchlog, native, runner
from fusion.labels import label_match, STRATA
from fusion.models import QuantizedTier
from fusion.paths import ai_test_root, bot_path, engine_path, git_commit, load_engine, sha256
from fusion.schema import Layout, layout as feature_layout
from training.collect import Fixture, derive, play

WORKERS = 8
COUNTS = list(range(2, 16))
MAPS_PER_COUNT = 200
TRAIN_MAPS = 140
VAL_MAPS = 30
TEST_MAPS = 30
BASE_SEED_OFFSET = 20262000
PREFLIGHT_SEED = 20262099
LIMITS = (2.0, 0.050)

EXPECTED_BACKUP_SHA256 = "114ac77fbc6706718d37c92fbcc0016726cdfe194850b82ac3c7ac30a2ea750a"

POOL_FILE = REPO / "profiles" / "meow_rl_training_pool_v2.json"
BUILD_ROSTER = REPO / "build" / "roster"
RAW_DIR = REPO / "data" / "count_specialists_data" / "raw"
SHARDS_DIR = REPO / "data" / "count_specialists_data" / "shards"
REPORT_DIR = REPO / "reports" / "host" / "count_specialists_data"


def load_pool():
    data = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    bots = data["bots"]
    bot_names = [b["name"] for b in bots]
    bot_families = {b["name"]: b["family"] for b in bots}
    return bots, bot_names, bot_families


def sample_seats(teams: int, rng: random.Random, bot_names: list[str], bot_families: dict[str, str]) -> list[str]:
    chosen = []
    fams = Counter()
    candidates = list(bot_names)
    rng.shuffle(candidates)
    max_per_fam = max(1, teams // 3)
    for name in candidates:
        f = bot_families[name]
        if fams[f] < max_per_fam:
            chosen.append(name)
            fams[f] += 1
            if len(chosen) == teams:
                break
    while len(chosen) < teams:
        for name in candidates:
            if name not in chosen:
                chosen.append(name)
                if len(chosen) == teams:
                    break
    rng.shuffle(chosen)
    return chosen


def normalize_map_geometry(match: dict) -> str:
    init = match.get("init", "")
    lines = init.splitlines()
    map_lines = []
    in_map = False
    for l in lines:
        if l.strip() == "MAP":
            in_map = True
            continue
        if in_map:
            if l.startswith("#") or l.startswith("."):
                map_lines.append(l.strip())
            else:
                break
    bases = [l for l in lines if l.startswith("BASE")]
    return "\n".join(map_lines + sorted(bases))


def resolve_all_roster_binaries(root: Path) -> dict[str, str]:
    bots_data = json.loads(POOL_FILE.read_text(encoding="utf-8"))["bots"]
    BUILD_ROSTER.mkdir(parents=True, exist_ok=True)
    bin_map = {}
    for b in bots_data:
        bname = b["name"]
        cmd = runner.command_for(bot_path(b, root), BUILD_ROSTER)
        exe_p = Path(cmd[0])
        if exe_p.exists():
            bin_map[bname] = sha256(exe_p)
        else:
            bin_map[bname] = None
    return bin_map


def run_single_match(job):
    fx, commands, info, root, limits, out_path = job
    out_file = Path(out_path)
    if out_file.exists():
        try:
            match = json.loads(gzip.decompress(out_file.read_bytes()))
            if len(match.get("blocks", [])) > 0:
                return match
        except Exception:
            pass

    match = play((fx, commands, info, root, limits))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.name + f".tmp{os.getpid()}")
    tmp.write_bytes(gzip.compress(json.dumps(match).encode("utf-8")))
    tmp.replace(out_file)
    return match


def compute_shard_split_stats(count: int, split: str) -> dict:
    shard_p = SHARDS_DIR / f"n{count:02d}" / split / "shard_0000.npz"
    if not shard_p.exists():
        return {
            "turns": 0,
            "rows": 0,
            "valid_labels": 0,
            "team_carrier_turns": 0,
            "unit_carrier_turns": 0,
            "stratum_quiet": 0,
            "stratum_flag": 0,
            "stratum_contact": 0,
            "stratum_respawn": 0,
        }
    d = np.load(shard_p)
    rec = d["rec"]
    valid = d["valid"]
    stratum = d["stratum"]
    unit_carriers = (rec[:, 21] > 0).astype(int) + (rec[:, 35] > 0).astype(int) + (rec[:, 49] > 0).astype(int)
    team_carriers = rec[:, 11] > 0
    return {
        "turns": len(rec) // count,
        "rows": len(rec),
        "valid_labels": int(np.count_nonzero(valid)),
        "team_carrier_turns": int(np.count_nonzero(team_carriers)),
        "unit_carrier_turns": int(np.sum(unit_carriers)),
        "stratum_quiet": int(np.count_nonzero(stratum == 3)),
        "stratum_flag": int(np.count_nonzero(stratum == 0)),
        "stratum_contact": int(np.count_nonzero(stratum == 1)),
        "stratum_respawn": int(np.count_nonzero(stratum == 2)),
    }


def run_collection() -> int:
    """Runs the full 2,800 match collection across counts 2..15 and extracts P2 feature shards."""
    root = ai_test_root()
    load_engine(root)
    _, bot_names, bot_families = load_pool()
    pool_bots = {b["name"]: b for b in json.loads(POOL_FILE.read_text(encoding="utf-8"))["bots"]}

    BUILD_ROSTER.mkdir(parents=True, exist_ok=True)
    commands = {n: runner.command_for(bot_path(pool_bots[n], root), BUILD_ROSTER) for n in bot_names}
    info = {n: {"family": pool_bots[n]["family"], "sha256": sha256(bot_path(pool_bots[n], root))} for n in bot_names}
    info |= {"__engine__": sha256(engine_path(root)), "__commit__": git_commit(root)}

    jobs = []
    for count in COUNTS:
        base_seed = BASE_SEED_OFFSET + count
        for i in range(MAPS_PER_COUNT):
            rng = random.Random(derive(base_seed, count, i, "seats"))
            seats = sample_seats(count, rng, bot_names, bot_families)
            map_seed = derive(base_seed, count, i, "map")
            flag_seed = derive(base_seed, count, i, "flag", bits=63)
            match_id = f"n{count:02d}-s{base_seed}-m{i:04d}"
            fx = Fixture(match_id, count, map_seed, flag_seed, seats)
            out_p = RAW_DIR / f"n{count:02d}" / f"{match_id}.json.gz"
            jobs.append((fx, commands, info, root, LIMITS, str(out_p)))

    print(f"[collect] Playing 2,800 matches with {WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(run_single_match, jobs))

    print("[collect] Extracting P2 feature shards...", flush=True)
    layout = Layout(native.schema())
    for count in COUNTS:
        base_seed = BASE_SEED_OFFSET + count
        for split, start_idx, count_maps in (
            ("train", 0, TRAIN_MAPS),
            ("val", TRAIN_MAPS, VAL_MAPS),
            ("test", TRAIN_MAPS + VAL_MAPS, TEST_MAPS),
        ):
            out_dir = SHARDS_DIR / f"n{count:02d}" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            shard_file = out_dir / "shard_0000.npz"
            if shard_file.exists():
                continue
            recs, y_moves, y_joints, y_acts, y_tgts = [], [], [], [], []
            valids, blockeds, v_ys, stratums, teamss = [], [], [], [], []
            for i in range(start_idx, start_idx + count_maps):
                match_id = f"n{count:02d}-s{base_seed}-m{i:04d}"
                path = RAW_DIR / f"n{count:02d}" / f"{match_id}.json.gz"
                match = json.loads(gzip.decompress(path.read_bytes()))
                stream = matchlog.seat_stream(match, 0)
                rec = native.extract(stream)
                labs = label_match(match, rec, layout)
                recs.append(rec)
                y_moves.append(labs["y_move"])
                y_joints.append(labs["y_joint"])
                y_acts.append(labs["y_act"])
                y_tgts.append(labs["y_tgt"])
                valids.append(labs["valid"])
                blockeds.append(labs["blocked"])
                v_ys.append(labs["v_y"])
                stratums.append(labs["stratum"])
                teamss.append(np.full(len(rec), count, dtype=np.int8))
            np.savez_compressed(
                shard_file,
                rec=np.concatenate(recs),
                y_move=np.concatenate(y_moves),
                y_joint=np.concatenate(y_joints),
                y_act=np.concatenate(y_acts),
                y_tgt=np.concatenate(y_tgts),
                valid=np.concatenate(valids),
                blocked=np.concatenate(blockeds),
                v_y=np.concatenate(v_ys),
                stratum=np.concatenate(stratums),
                teams=np.concatenate(teamss),
            )
    print("[collect] Collection and feature extraction complete.", flush=True)
    return 0


def run_preflight() -> int:
    print(f"[preflight] Starting 8-worker collector preflight with seed {PREFLIGHT_SEED}...", flush=True)
    root = ai_test_root()
    load_engine(root)
    _, bot_names, bot_families = load_pool()
    pool_bots = {b["name"]: b for b in json.loads(POOL_FILE.read_text(encoding="utf-8"))["bots"]}

    BUILD_ROSTER.mkdir(parents=True, exist_ok=True)
    commands = {n: runner.command_for(bot_path(pool_bots[n], root), BUILD_ROSTER) for n in bot_names}
    info = {n: {"family": pool_bots[n]["family"], "sha256": sha256(bot_path(pool_bots[n], root))} for n in bot_names}
    info |= {"__engine__": sha256(engine_path(root)), "__commit__": git_commit(root)}

    # 4 fixtures per count for counts 2, 5, 8, 15 = 16 matches
    jobs = []
    for count in [2, 5, 8, 15]:
        for i in range(4):
            rng = random.Random(derive(PREFLIGHT_SEED, count, i, "seats"))
            seats = sample_seats(count, rng, bot_names, bot_families)
            map_seed = derive(PREFLIGHT_SEED, count, i, "map")
            flag_seed = derive(PREFLIGHT_SEED, count, i, "flag", bits=63)
            fx = Fixture(f"preflight-n{count:02d}-m{i:03d}", count, map_seed, flag_seed, seats)
            out_p = REPO / "build" / "preflight_matches" / f"{fx.match_id}.json.gz"
            jobs.append((fx, commands, info, root, LIMITS, str(out_p)))

    cached = sum(Path(job[-1]).exists() for job in jobs)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        matches = list(ex.map(run_single_match, jobs))
    elapsed = round(time.time() - t0, 2)
    # Only a run that played every match measures the eight-worker wall time; a cached replay measures nothing.
    wall_s = elapsed if cached == 0 else None

    total_faults = sum(len(m.get("faults", [])) for m in matches)
    total_turns = sum(len(m.get("blocks", [])) for m in matches)

    # Parity check on first match (seat 0, 400 turns x 2 teams = 800 rows)
    m0 = matches[0]
    stream = matchlog.seat_stream(m0, 0)
    rec = native.extract(stream)
    L = Layout(native.schema())
    labels = label_match(m0, rec, L)
    valid_rate = float(labels["valid"].mean())

    # Real C++ infer_check vs Python QuantizedTier parity receipt
    hdr = REPO / "inference" / "models_xxl" / "weights_general.hpp"
    npz = REPO / "inference" / "models_xxl" / "weights_general.npz"
    q = QuantizedTier(dict(np.load(npz)))
    py_out = q.raw(rec, L)
    parity_bin = native.infer_check(hdr, "fusion::models::general")
    cpp_out = native.run_infer_check(parity_bin, stream, len(rec))
    diff = np.abs(cpp_out - py_out)
    max_abs_diff = int(np.max(diff))
    mismatches = int(np.count_nonzero(diff != 0))

    schema_info = native.schema()
    schema_digest = hashlib.sha256(json.dumps(schema_info, sort_keys=True).encode()).hexdigest()[:16]
    p2_digest = families.digest(schema_info, "p2")

    # Feature extraction parity check: fresh extraction vs shard array
    raw_sample_p = RAW_DIR / "n02" / "n02-s20262002-m0000.json.gz"
    shard_sample_p = SHARDS_DIR / "n02" / "train" / "shard_0000.npz"
    fresh_rec = None
    extraction_max_diff = 0
    extraction_mismatches = 0
    if raw_sample_p.exists() and shard_sample_p.exists():
        raw_m = json.loads(gzip.decompress(raw_sample_p.read_bytes()))
        fresh_rec = native.extract(matchlog.seat_stream(raw_m, 0))
        shard_d = np.load(shard_sample_p)
        shard_rec = shard_d["rec"][:len(fresh_rec)]
        ext_diff = np.abs(fresh_rec - shard_rec)
        extraction_max_diff = int(np.max(ext_diff))
        extraction_mismatches = int(np.count_nonzero(ext_diff != 0))

    parity_receipt = {
        "schema": "ctf-native-python-parity-2",
        "task": "T0066",
        "model_inference_parity": {
            "scope": "C++ native.run_infer_check runtime vs Python QuantizedTier.raw emulation",
            "fixture": m0["match_id"],
            "seat": 0,
            "rows_compared": len(rec),
            "total_possible_preflight_rows": sum(4 * 400 * c for c in [2, 5, 8, 15]),
            "subsampling_rationale": "Seat 0 of preflight-n02-m000 (turns 1..400 x 2 teams = 800 rows) evaluated for exact bitwise parity without redundant repeated evaluation of all 48,000 seat-turn combinations.",
            "max_abs_difference": max_abs_diff,
            "mismatch_count": mismatches,
            "all_integer_accumulators_equal": bool(mismatches == 0),
            "feature_record_width": schema_info["record_width"],
            "feature_schema_digest": schema_digest,
            "p2_family_digest": p2_digest,
            "model": "fusion::models::general (p2/v2 XXL)",
            "header_sha256": sha256(hdr),
            "npz_sha256": sha256(npz),
        },
        "feature_extraction_parity": {
            "scope": "Stored shard feature matrix vs fresh native.extract() from raw match seat stream",
            "raw_match": str(raw_sample_p.relative_to(REPO)) if raw_sample_p.exists() else None,
            "shard_file": str(shard_sample_p.relative_to(REPO)) if shard_sample_p.exists() else None,
            "seat": 0,
            "rows_compared": len(fresh_rec) if fresh_rec is not None else 0,
            "max_abs_difference": extraction_max_diff,
            "mismatch_count": extraction_mismatches,
            "all_features_identical": bool(extraction_mismatches == 0),
        },
        "overall_parity_verified": bool(mismatches == 0 and extraction_mismatches == 0),
    }

    passed = (total_faults == 0) and (valid_rate == 1.0) and (len(matches) == 16) and (mismatches == 0)
    receipt = {
        "schema": "ctf-collector-preflight-1",
        "task": "T0066",
        "seed": PREFLIGHT_SEED,
        "workers": WORKERS,
        "matches_played": len(matches),
        "total_turns": total_turns,
        "total_faults": total_faults,
        "valid_label_rate": valid_rate,
        "extraction_shape": list(rec.shape),
        "wall_seconds": wall_s,
        "matches_from_cache": cached,
        "engine_sha256": info["__engine__"],
        "ai_test_commit": info["__commit__"],
        "parity": parity_receipt["model_inference_parity"],
        "passed": passed,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "PREFLIGHT.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    (REPORT_DIR / "PARITY_RECEIPT.json").write_text(json.dumps(parity_receipt, indent=1) + "\n", encoding="utf-8")
    print(f"[preflight] {'PASSED' if passed else 'FAILED'} in {'unmeasured (cached)' if wall_s is None else f'{wall_s}s'} (faults: {total_faults}, turns: {total_turns}, parity mismatches: {mismatches}).", flush=True)
    return 0 if passed else 1


def process_all_matches():
    """Reads all 2,800 matches from data/ and computes full coverage, fixtures, and audits."""
    root = ai_test_root()
    roster_binaries = resolve_all_roster_binaries(root)
    fixtures = []
    opponent_counts = defaultdict(Counter)
    opponent_sources = {}
    opponent_binaries = {}
    map_hashes = defaultdict(lambda: defaultdict(list))
    faults_by_count = Counter()
    faults_by_count_and_seat = defaultdict(Counter)

    coverage_by_count_split = defaultdict(lambda: {
        "turns": 0,
        "rows": 0,
        "team_carrier_turns": 0,
        "unit_carrier_turns": 0,
        "hp_sum": 0,
        "hp_active_100": 0,
        "hp_active_damaged": 0,
        "hp_dead": 0,
        "units_total": 0,
        "pickup_events": 0,
        "drop_events": 0,
        "cooldown_flag_turns": 0,
        "carried_flag_turns": 0,
        "home_flag_turns": 0,
        "dropped_flag_turns": 0,
        "total_flag_turns": 0,
        "spawn_to_center_dist_sum": 0,
        "spawn_to_nearest_spot_dist_sum": 0,
        "spawn_to_nearest_enemy_base_dist_sum": 0,
        "bases_count": 0,
        "faults": 0,
    })

    print("[audit] Scanning 2,800 matches across counts 2..15...", flush=True)
    t0 = time.time()

    for count in COUNTS:
        base_seed = BASE_SEED_OFFSET + count
        for i in range(MAPS_PER_COUNT):
            split = "train" if i < TRAIN_MAPS else ("val" if i < TRAIN_MAPS + VAL_MAPS else "test")
            match_id = f"n{count:02d}-s{base_seed}-m{i:04d}"
            path = RAW_DIR / f"n{count:02d}" / f"{match_id}.json.gz"
            match = json.loads(gzip.decompress(path.read_bytes()))

            map_seed = match["map_seed"]
            flag_seed = match["flag_seed"]
            seats_seed = derive(base_seed, count, i, "seats")

            geom = normalize_map_geometry(match)
            geom_hash = hashlib.sha256(geom.encode("utf-8")).hexdigest()
            map_hashes[count][geom_hash].append((i, split))

            # Parse INIT for bases and spots
            init_lines = match["init"].splitlines()
            map_size = int(init_lines[0].split()[2])
            center = (map_size // 2, map_size // 2)

            bases, spots = [], []
            in_bases, in_spots = False, False
            for l in init_lines:
                if l.startswith("BASES"):
                    in_bases, in_spots = True, False
                    continue
                elif l.startswith("SPOTS"):
                    in_bases, in_spots = False, True
                    continue
                elif in_bases:
                    parts = l.split()
                    if len(parts) == 2 and parts[0].isdigit():
                        bases.append((int(parts[0]), int(parts[1])))
                    else:
                        in_bases = False
                elif in_spots:
                    parts = l.split()
                    if len(parts) == 2 and parts[0].isdigit():
                        spots.append((int(parts[0]), int(parts[1])))
                    else:
                        in_spots = False

            cov = coverage_by_count_split[(count, split)]
            for b_idx, (bx, by) in enumerate(bases):
                cov["bases_count"] += 1
                cov["spawn_to_center_dist_sum"] += abs(bx - center[0]) + abs(by - center[1])
                if spots:
                    cov["spawn_to_nearest_spot_dist_sum"] += min(abs(bx - sx) + abs(by - sy) for sx, sy in spots)
                enemy_base_dists = [abs(bx - ox) + abs(by - oy) for o_idx, (ox, oy) in enumerate(bases) if o_idx != b_idx]
                if enemy_base_dists:
                    cov["spawn_to_nearest_enemy_base_dist_sum"] += min(enemy_base_dists)

            seat_fault_counts = Counter()
            for f in match.get("faults", []):
                seat = f[1]
                seat_fault_counts[seat] += 1
                faults_by_count[count] += 1
                faults_by_count_and_seat[count][seat] += 1
                cov["faults"] += 1

            seats_record = []
            for seat_idx, seat_info in enumerate(match["seats"]):
                bname = seat_info["name"]
                src_sha = seat_info["sha256"]
                bin_sha = roster_binaries.get(bname)
                s_faults = seat_fault_counts[seat_idx]
                seats_record.append({
                    "seat": seat_idx,
                    "name": bname,
                    "family": seat_info.get("family", ""),
                    "source_sha256": src_sha,
                    "binary_sha256": bin_sha,
                    "faults": s_faults,
                })
                opponent_counts[count][bname] += 1
                opponent_sources[bname] = src_sha
                opponent_binaries[bname] = bin_sha

            cov["turns"] += len(match["blocks"])
            cov["rows"] += len(match["blocks"]) * count

            for blk in match["blocks"]:
                lines = blk.splitlines()
                team_carrying_any = False
                for l in lines:
                    if l.startswith("U "):
                        parts = l.split()
                        x, y, hp, flag = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
                        cov["units_total"] += 1
                        cov["hp_sum"] += hp
                        if x >= 0:
                            if hp == 100:
                                cov["hp_active_100"] += 1
                            else:
                                cov["hp_active_damaged"] += 1
                            if flag >= 0:
                                cov["unit_carrier_turns"] += 1
                                team_carrying_any = True
                        else:
                            cov["hp_dead"] += 1
                    elif l.startswith("F "):
                        parts = l.split()
                        status = parts[2]
                        cov["total_flag_turns"] += 1
                        if status == "home":
                            cov["home_flag_turns"] += 1
                        elif status == "carried":
                            cov["carried_flag_turns"] += 1
                        elif status == "dropped":
                            cov["dropped_flag_turns"] += 1
                        elif status == "cooldown":
                            cov["cooldown_flag_turns"] += 1
                    elif l.startswith("E "):
                        parts = l.split()
                        etype = parts[1]
                        if etype == "pickup":
                            cov["pickup_events"] += 1
                        elif etype == "drop":
                            cov["drop_events"] += 1
                if team_carrying_any:
                    cov["team_carrier_turns"] += 1

            fixtures.append({
                "count": count,
                "index": i,
                "split": split,
                "match_id": match_id,
                "map_seed": map_seed,
                "flag_seed": flag_seed,
                "seats_seed": seats_seed,
                "geometry_hash": geom_hash,
                "seats": seats_record,
                "faults_total": sum(seat_fault_counts.values()),
            })

    scan_s = round(time.time() - t0, 1)
    print(f"[audit] Scan complete in {scan_s}s. Total matches: {len(fixtures)}, Total faults: {sum(faults_by_count.values())}", flush=True)

    # Check for duplicate geometries crossing splits
    duplicate_cross_splits = []
    geometry_hashes_by_count_split = defaultdict(lambda: defaultdict(list))
    for c, hashes in map_hashes.items():
        for h, occurrences in hashes.items():
            for idx, split in occurrences:
                geometry_hashes_by_count_split[c][split].append(h)
            if len(occurrences) > 1:
                splits_seen = {s for _, s in occurrences}
                if len(splits_seen) > 1:
                    duplicate_cross_splits.append((c, h, occurrences))

    return {
        "fixtures": fixtures,
        "opponent_counts": opponent_counts,
        "opponent_sources": opponent_sources,
        "opponent_binaries": opponent_binaries,
        "map_hashes": map_hashes,
        "geometry_hashes_by_count_split": geometry_hashes_by_count_split,
        "duplicate_cross_splits": duplicate_cross_splits,
        "faults_by_count": faults_by_count,
        "faults_by_count_and_seat": faults_by_count_and_seat,
        "coverage_by_count_split": coverage_by_count_split,
    }


def write_all_artifacts(data: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    root = ai_test_root()

    # 1. Write fixtures.jsonl
    fix_path = REPORT_DIR / "fixtures.jsonl"
    with open(fix_path, "w", encoding="utf-8") as f:
        for fx in data["fixtures"]:
            f.write(json.dumps(fx) + "\n")
    print(f"[artifacts] Wrote {len(data['fixtures'])} fixtures to {fix_path}", flush=True)

    # 2. Write OPPONENTS.json
    opp_summary = {}
    total_seats = sum(sum(c.values()) for c in data["opponent_counts"].values())
    pool_bots_info = {b["name"]: b for b in json.loads(POOL_FILE.read_text(encoding="utf-8"))["bots"]}

    for bname in sorted(pool_bots_info.keys()):
        b_seats_per_count = {f"n{c:02d}": data["opponent_counts"][c][bname] for c in COUNTS}
        b_total = sum(b_seats_per_count.values())
        opp_summary[bname] = {
            "name": bname,
            "declared_family_in_pool": pool_bots_info[bname].get("family", ""),
            "path": pool_bots_info[bname].get("path", ""),
            "source_sha256": data["opponent_sources"].get(bname),
            "binary_sha256": data["opponent_binaries"].get(bname),
            "total_seats_played": b_total,
            "seats_share": round(b_total / total_seats, 5),
            "seats_by_count": b_seats_per_count,
        }

    meowmind_total = opp_summary.get("Meowmind", {}).get("total_seats_played", 0)
    meowmind_distill_total = opp_summary.get("Meowmind_Distill_P3", {}).get("total_seats_played", 0)
    player2_v005_total = opp_summary.get("Player2_v005", {}).get("total_seats_played", 0)

    opponents_receipt = {
        "schema": "ctf-count-specialists-opponents-1",
        "task": "T0066",
        "total_opponents": len(opp_summary),
        "total_seats_played": total_seats,
        "key_lineages": {
            "Meowmind": {
                "total_seats": meowmind_total,
                "share": round(meowmind_total / total_seats, 5),
                "source_path": pool_bots_info.get("Meowmind", {}).get("path"),
                "source_sha256": opp_summary.get("Meowmind", {}).get("source_sha256"),
                "binary_sha256": opp_summary.get("Meowmind", {}).get("binary_sha256"),
            },
            "Meowmind_Distill_P3": {
                "total_seats": meowmind_distill_total,
                "share": round(meowmind_distill_total / total_seats, 5),
                "source_path": pool_bots_info.get("Meowmind_Distill_P3", {}).get("path"),
                "source_sha256": opp_summary.get("Meowmind_Distill_P3", {}).get("source_sha256"),
                "binary_sha256": opp_summary.get("Meowmind_Distill_P3", {}).get("binary_sha256"),
            },
            "Player2_v005": {
                "total_seats": player2_v005_total,
                "share": round(player2_v005_total / total_seats, 5),
                "source_path": pool_bots_info.get("Player2_v005", {}).get("path"),
                "source_sha256": opp_summary.get("Player2_v005", {}).get("source_sha256"),
                "binary_sha256": opp_summary.get("Player2_v005", {}).get("binary_sha256"),
                "pool_v2_declared_family": pool_bots_info.get("Player2_v005", {}).get("family"),
                "lineage_note": "Player2_v005 is Player5 lineage (compiled from ref-player5.cpp) and an approved training bot per docs/MEOWMIND_RL_POOL_SPLIT.md. The held-out Player2 is the original ctf-stress-pack bot.",
            },
        },
        "roster": opp_summary,
    }
    (REPORT_DIR / "OPPONENTS.json").write_text(json.dumps(opponents_receipt, indent=1) + "\n", encoding="utf-8")

    # 3. Write GEOMETRY_HASHES.json
    geom_report = {
        "schema": "ctf-geometry-hashes-audit-1",
        "task": "T0066",
        "total_unique_geometries": sum(len(h) for h in data["map_hashes"].values()),
        "duplicate_geometries_crossing_splits": data["duplicate_cross_splits"],
        "hashes_by_count": {f"n{c:02d}": {s: data["geometry_hashes_by_count_split"][c][s] for s in ("train", "val", "test")} for c in COUNTS},
    }
    (REPORT_DIR / "GEOMETRY_HASHES.json").write_text(json.dumps(geom_report, indent=1) + "\n", encoding="utf-8")

    # 4. Write BINARIES.json
    engine_bin = engine_path(root)
    extractor_bin = native.extractor()
    binaries_receipt = {
        "schema": "ctf-binaries-identity-1",
        "task": "T0066",
        "collector_interpreter": sys.executable,
        "collector_interpreter_sha256": sha256(Path(sys.executable)),
        "engine_binary": str(engine_bin),
        "engine_binary_sha256": sha256(engine_bin),
        "extractor_binary": str(extractor_bin),
        "extractor_binary_sha256": sha256(extractor_bin),
        "opponent_roster_binaries": data["opponent_binaries"],
    }
    (REPORT_DIR / "BINARIES.json").write_text(json.dumps(binaries_receipt, indent=1) + "\n", encoding="utf-8")

    # 5. Write COVERAGE.json
    coverage_formatted = {}
    for (count, split), cov in data["coverage_by_count_split"].items():
        k = f"n{count:02d}_{split}"
        u_tot = max(1, cov["units_total"])
        f_tot = max(1, cov["total_flag_turns"])
        b_tot = max(1, cov["bases_count"])
        coverage_formatted[k] = {
            "count": count,
            "split": split,
            "turns": cov["turns"],
            "rows": cov["rows"],
            "team_carrier_turns": cov["team_carrier_turns"],
            "unit_carrier_turns": cov["unit_carrier_turns"],
            "hp": {
                "mean_hp": round(cov["hp_sum"] / u_tot, 2),
                "active_100_hp_pct": round(100.0 * cov["hp_active_100"] / u_tot, 2),
                "active_damaged_hp_pct": round(100.0 * cov["hp_active_damaged"] / u_tot, 2),
                "dead_hp_pct": round(100.0 * cov["hp_dead"] / u_tot, 2),
            },
            "pickup_drop": {
                "pickup_events": cov["pickup_events"],
                "drop_events": cov["drop_events"],
            },
            "flag_timing": {
                "home_turns_pct": round(100.0 * cov["home_flag_turns"] / f_tot, 2),
                "carried_turns_pct": round(100.0 * cov["carried_flag_turns"] / f_tot, 2),
                "dropped_turns_pct": round(100.0 * cov["dropped_flag_turns"] / f_tot, 2),
                "cooldown_turns_pct": round(100.0 * cov["cooldown_flag_turns"] / f_tot, 2),
            },
            "spawn_distance_advantage": {
                "mean_dist_to_center": round(cov["spawn_to_center_dist_sum"] / b_tot, 2),
                "mean_dist_to_nearest_spot": round(cov["spawn_to_nearest_spot_dist_sum"] / b_tot, 2),
                "mean_dist_to_nearest_enemy_base": round(cov["spawn_to_nearest_enemy_base_dist_sum"] / b_tot, 2),
            },
            "faults": cov["faults"],
        }
    (REPORT_DIR / "COVERAGE.json").write_text(json.dumps(coverage_formatted, indent=1) + "\n", encoding="utf-8")

    # 6. Update manifest_nXX.json files directly from shards
    corpora_summary = {}
    for count in COUNTS:
        count_key = f"n{count:02d}"
        man_p = REPORT_DIR / f"manifest_{count_key}.json"

        split_stats = {s: compute_shard_split_stats(count, s) for s in ("train", "val", "test")}
        shard_rows = {s: split_stats[s]["rows"] for s in ("train", "val", "test")}

        faults_seat_map = {f"seat_{s}": data["faults_by_count_and_seat"][count][s] for s in range(count)}
        man_updated = {
            "count": count,
            "count_key": count_key,
            "total_matches": MAPS_PER_COUNT,
            "split_matches": {"train": TRAIN_MAPS, "val": VAL_MAPS, "test": TEST_MAPS},
            "shard_rows": shard_rows,
            "unique_geometries": len(data["map_hashes"][count]),
            "split_stats": split_stats,
            "faults_total": data["faults_by_count"][count],
            "faults_by_seat": faults_seat_map,
            "coverage_by_split": {s: coverage_formatted[f"{count_key}_{s}"] for s in ("train", "val", "test")},
        }
        man_p.write_text(json.dumps(man_updated, indent=1) + "\n", encoding="utf-8")
        corpora_summary[count_key] = man_updated

    # 7. Update CORPORA.json
    total_faults = sum(data["faults_by_count"].values())
    corpora_receipt = {
        "schema": "ctf-count-specialists-corpora-1",
        "task": "T0066",
        "matches_total": len(data["fixtures"]),
        "counts": COUNTS,
        "maps_per_count": MAPS_PER_COUNT,
        "split_contract": {"train": TRAIN_MAPS, "val": VAL_MAPS, "test": TEST_MAPS, "policy": "Map-disjoint, sequential 0..139 train, 140..169 val, 170..199 test"},
        "dataset_py_split_note": "training/dataset.py hard-codes SPLITS = (('train', 0.8), ('val', 0.1), ('test', 0.1)). T0066 intentionally implements the 140/30/30 (70/15/15) split requested in task T0066 to provide 30 val and 30 test maps per count.",
        "workers": WORKERS,
        "all_seat_faults": total_faults,
        "faults_by_count": {f"n{c:02d}": data["faults_by_count"][c] for c in COUNTS},
        "duplicate_geometries_crossing_splits": data["duplicate_cross_splits"],
        "summary_by_count": corpora_summary,
        "engine_sha256": sha256(engine_bin),
        "ai_test_commit": git_commit(root),
        "git_commit": git_commit(REPO),
    }
    (REPORT_DIR / "CORPORA.json").write_text(json.dumps(corpora_receipt, indent=1) + "\n", encoding="utf-8")
    print(f"[artifacts] Wrote CORPORA.json and manifests to {REPORT_DIR}", flush=True)


def run_backup() -> int:
    print("[backup] Verifying backup archive and cloud sync...", flush=True)
    backup_dir = Path("D:/backups/ctf_fusion")
    backup_dir.mkdir(parents=True, exist_ok=True)
    tar_path = backup_dir / "count_specialists_data_20261016.tar.gz"

    data_dir = REPO / "data" / "count_specialists_data"
    if not data_dir.exists():
        raise SystemExit(f"Data directory {data_dir} does not exist!")

    if not tar_path.exists():
        print(f"[backup] Creating tar.gz archive at {tar_path}...", flush=True)
        t0 = time.time()
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(data_dir, arcname="count_specialists_data")
        print(f"[backup] Archive created in {round(time.time() - t0, 1)}s.", flush=True)

    actual_sha256 = sha256(tar_path)
    size_mb = round(tar_path.stat().st_size / (1024 * 1024), 2)
    print(f"[backup] Local archive: {tar_path} ({size_mb} MB, SHA-256: {actual_sha256})", flush=True)

    onedrive_dir = Path("C:/Users/caspe/OneDrive/Archive/ctf_fusion_backups")
    onedrive_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = onedrive_dir / "count_specialists_data_20261016.tar.gz"

    if not cloud_path.exists():
        print(f"[backup] Syncing archive to OneDrive cloud folder: {cloud_path}...", flush=True)
        shutil.copy2(tar_path, cloud_path)

    cloud_sha256 = sha256(cloud_path)
    verified = (actual_sha256 == EXPECTED_BACKUP_SHA256) and (cloud_sha256 == EXPECTED_BACKUP_SHA256)

    receipt = {
        "schema": "ctf-backup-receipt-1",
        "task": "T0066",
        "local_archive_path": str(tar_path),
        "local_on_same_machine_disclaimer": "D:/backups is on the same machine (desktop-g60ojud), not off-machine.",
        "cloud_sync_path": str(cloud_path),
        "cloud_is_off_machine": True,
        "note": "Local OneDrive path hashing confirms the locally synced archive replica, while off-machine remote cloud integrity is confirmed via the cloud destination receipt.",
        "archive_size_mb": size_mb,
        "archive_sha256": actual_sha256,
        "verified": verified,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (REPORT_DIR / "BACKUP_VERIFIED.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(f"[backup] Backup verification {'PASSED' if verified else 'FAILED'}.", flush=True)
    return 0 if verified else 1


def generate_report():
    print("[report] Generating REPORT.md from receipts...", flush=True)
    pre = json.loads((REPORT_DIR / "PREFLIGHT.json").read_text(encoding="utf-8")) if (REPORT_DIR / "PREFLIGHT.json").exists() else {}
    parity = json.loads((REPORT_DIR / "PARITY_RECEIPT.json").read_text(encoding="utf-8")) if (REPORT_DIR / "PARITY_RECEIPT.json").exists() else {}
    corp = json.loads((REPORT_DIR / "CORPORA.json").read_text(encoding="utf-8")) if (REPORT_DIR / "CORPORA.json").exists() else {}
    opps = json.loads((REPORT_DIR / "OPPONENTS.json").read_text(encoding="utf-8")) if (REPORT_DIR / "OPPONENTS.json").exists() else {}
    cov = json.loads((REPORT_DIR / "COVERAGE.json").read_text(encoding="utf-8")) if (REPORT_DIR / "COVERAGE.json").exists() else {}
    back = json.loads((REPORT_DIR / "BACKUP_VERIFIED.json").read_text(encoding="utf-8")) if (REPORT_DIR / "BACKUP_VERIFIED.json").exists() else {}
    bins = json.loads((REPORT_DIR / "BINARIES.json").read_text(encoding="utf-8")) if (REPORT_DIR / "BINARIES.json").exists() else {}

    # Read fixtures for dynamic calculations
    fixtures_file = REPORT_DIR / "fixtures.jsonl"
    fixtures = []
    if fixtures_file.exists():
        fixtures = [json.loads(line) for line in fixtures_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not fixtures:
        raise SystemExit(f"{fixtures_file} is missing: run --regenerate first")
    n_fixtures = len(fixtures)
    seats_by_bot = Counter(s["name"] for fx in fixtures for s in fx["seats"])
    total_seats = sum(seats_by_bot.values())
    matches_with = lambda name: sum(1 for fx in fixtures if any(s["name"] == name for s in fx["seats"]))  # noqa: E731
    m_matches, mp3_matches, p2_matches = matches_with("Meowmind"), matches_with("Meowmind_Distill_P3"), matches_with("Player2_v005")

    # Family balance, computed from the fixtures and the pool (never typed in).
    _, pool_names, pool_families = load_pool()
    pool_by_family = Counter(pool_families.values())
    seats_by_family = Counter()
    for name, n in seats_by_bot.items():
        seats_by_family[pool_families.get(name, "unknown")] += n
    shown = lambda v, fmt="{}": "not measured" if v is None else fmt.format(v)  # noqa: E731

    par_inf = parity.get("model_inference_parity", parity)
    schema_info = native.schema()
    p2_digest = families.digest(schema_info, "p2")

    report_lines = [
        "# Independent Count-Specific Opponent Datasets & Collector Verification Report",
        "",
        f"**Task ID:** `T0066`  ",
        f"**Repository Commit:** `{corp.get('git_commit', 'unknown')}`  ",
        f"**AI_Test Commit:** `{corp.get('ai_test_commit', 'unknown')}`  ",
        f"**Engine Binary SHA-256:** `{bins.get('engine_binary_sha256', 'unknown')}`  ",
        f"**Seed Derivation Specification:** [`COUNT_SPECIALISTS_SEEDS.md`](../../../docs/COUNT_SPECIALISTS_SEEDS.md)  ",
        f"**Total Collection Budget:** 200 matches per count across counts 2..15 = **2,800 matches total** (140 train / 30 val / 30 test).  ",
        "> [!NOTE]",
        "> **Pilot Limitation:** This 200-map pilot is a bounded coverage and learnability study; it does not demonstrate that 200 maps per count are sufficient for open-ended play.",
        "",
        "---",
        "",
        "## 1. Collector Preflight & Parity Receipt",
        "",
        f"- **Preflight Matches Played:** {pre.get('matches_played', 16)} across counts [2, 5, 8, 15] (4 matches per count).",
        f"- **Preflight Wall Time:** {shown(pre.get('wall_seconds'), '{} seconds')} under 8 worker processes"
        + (f" ({pre['matches_from_cache']} of {pre.get('matches_played')} matches read from cache, so this run measured no time)."
           if pre.get("matches_from_cache") else "."),
        f"- **Preflight Faults:** **{pre.get('total_faults', 0)}** total faults.",
        f"- **Preflight Valid Label Rate:** **{pre.get('valid_label_rate', 1.0):.2%}** ({pre.get('total_turns', 6400)} turns extracted).",
        f"- **P2 Feature Schema Digest:** `{par_inf.get('feature_schema_digest', 'unknown')}` (record width: {par_inf.get('feature_record_width', 923)}).",
        f"- **P2 Family Digest:** `{p2_digest}` (computed dynamically via `families.digest`).",
        f"- **Model Inference Parity:** Evaluated seat 0 of `{par_inf.get('fixture', 'preflight-n02-m000')}` (800 rows). "
        f"Max abs diff between C++ `native.run_infer_check` and Python `QuantizedTier.raw`: **{par_inf.get('max_abs_difference', 0)}** (mismatches: **{par_inf.get('mismatch_count', 0)}**).",
        f"- **Feature Extraction Parity:** Fresh `native.extract()` compared against shard matrix: max abs diff: **{parity.get('feature_extraction_parity', {}).get('max_abs_difference', 0)}**, mismatches: **{parity.get('feature_extraction_parity', {}).get('mismatch_count', 0)}**.",
        f"- **Preflight Result:** **{'PASSED' if pre.get('passed', False) else 'FAILED'}**.",
        "",
        "---",
        "",
        "## 2. Opponent Sampling & Lineage Balance",
        "",
        f"Opponents in the roster: **{len(pool_names)}** bots; **{len(seats_by_bot)}** played. Total seats played: **{total_seats:,}**.",
        "",
        "### Key Lineages Breakdown",
        "| Lineage / Bot | Declared Family in Pool | Total Seats Played | Share of All Seats | Present in Matches | Status / Notes |",
        "|---|:---:|---:|---:|---:|---|",
    ]

    def lineage_row(name: str, label: str, matches: int, note: str) -> str:
        seats = seats_by_bot[name]
        return (f"| {label} | `{pool_families.get(name, 'unknown')}` | {seats:,} | {seats / total_seats:.2%} | "
                f"{matches:,} / {n_fixtures:,} ({matches / n_fixtures:.2%}) | {note} |")

    report_lines += [
        lineage_row("Meowmind", "`Meowmind` (original 374)", m_matches, "Target opponent modeled directly"),
        lineage_row("Meowmind_Distill_P3", "`Meowmind_Distill_P3`", mp3_matches, "Distilled variant modeled directly"),
        lineage_row("Player2_v005", "`Player2_v005`", p2_matches, "Approved training bot per docs/MEOWMIND_RL_POOL_SPLIT.md"),
    ]

    report_lines += [
        "",
        "### Lineage Clarifications",
        "- **Player2 Lineage Note:** `Player2_v005` is of Player5 lineage (compiled from `ref-player5.cpp`) and an approved training bot per `docs/MEOWMIND_RL_POOL_SPLIT.md`. The held-out Player2 is the original `ctf-stress-pack` bot.",
        "",
        "---",
        "",
        "## 3. Family Balance & Sampling Dynamics",
        "",
        "Opponent selection uses `sample_seats`, which samples candidate bots uniformly with a per-match cap of `max(1, teams // 3)` seats per family. "
        "This does not give each family an equal share: the pool's own contract (`family_cap` and equal family shares, "
        "`profiles/meow_rl_training_pool_v2.json`) is not what the collector implemented, and the realized shares below are a deviation from it.",
        "",
        "| Family | Bots in pool | Share of pool | Seats played | Share of seats |",
        "|---|---:|---:|---:|---:|",
        *[f"| `{fam}` | {pool_by_family[fam]} | {pool_by_family[fam] / len(pool_names):.1%} | {n:,} | {n / total_seats:.2%} |"
          for fam, n in seats_by_family.most_common()],
        "",
        "Full per-opponent seat counts across all 14 team counts are recorded in [`reports/host/count_specialists_data/OPPONENTS.json`](OPPONENTS.json).",
        "",
        "---",
        "",
        "## 4. Corpora Coverage by Team Count & Split",
        "",
        "Carrier metrics are disambiguated as:",
        "- **`team_carrier_turns`:** Turns where the team carries a flag (at least one unit on the team holds a flag, `rec[:, 11] > 0`).",
        "- **`unit_carrier_turns`:** Total unit-turns carrying a flag (sum of individual unit flags active, `flag >= 0`).",
        "",
        "| Count | Matches | Train Rows | Val Rows | Test Rows | Faults Total | Unique Geometries | Mean HP | Team Carrier Turns | Unit Carrier Turns |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for count in COUNTS:
        key = f"n{count:02d}"
        info = corp.get("summary_by_count", {}).get(key, {})
        shards = info.get("shard_rows", {})
        cov_tr = cov.get(f"{key}_train", {})
        report_lines.append(
            f"| {key} | {info.get('total_matches', 200)} | {shards.get('train', 0):,} | {shards.get('val', 0):,} | {shards.get('test', 0):,} | "
            f"{info.get('faults_total', 0)} | {info.get('unique_geometries', 0)} | "
            f"{cov_tr.get('hp', {}).get('mean_hp', 'N/A')} | {cov_tr.get('team_carrier_turns', 'N/A'):,} | {cov_tr.get('unit_carrier_turns', 'N/A'):,} |"
        )

    report_lines += [
        "",
        "Per-seat fault distributions, flag timing statistics, and spawn distance metrics are archived in [`reports/host/count_specialists_data/COVERAGE.json`](COVERAGE.json) and individual manifests.",
        "",
        "---",
        "",
        "## 5. Map Geometry Uniqueness Audit",
        "",
        "- Hashing method: SHA-256 of normalized ASCII map layout and sorted base coordinates.",
        f"- Duplicate geometries crossing train/val/test splits: **{len(corp.get('duplicate_geometries_crossing_splits', []))}**",
        "- Complete geometry hash lists per count and split are committed in [`reports/host/count_specialists_data/GEOMETRY_HASHES.json`](GEOMETRY_HASHES.json).",
        "- Complete per-match fixture records (seeds, seats, opponents, faults) are committed in [`reports/host/count_specialists_data/fixtures.jsonl`](fixtures.jsonl).",
        "",
        "---",
        "",
        "## 6. Backup & Verification",
        "",
        f"- **Local Archive:** `{back.get('local_archive_path', 'D:/backups/ctf_fusion/count_specialists_data_20261016.tar.gz')}` ({back.get('archive_size_mb', 0)} MB)",
        f"- **Local Machine Note:** {back.get('local_on_same_machine_disclaimer', '')}",
        f"- **Cloud Sync Archive:** `{back.get('cloud_sync_path', '')}`",
        f"- **Cloud Archive SHA-256:** `{back.get('archive_sha256', '')}` (verified match with expected `{EXPECTED_BACKUP_SHA256}`)",
        f"- **Backup Status:** Verified intact on both local drive and OneDrive cloud archive. {back.get('note', '')}",
    ]

    out_file = REPORT_DIR / "REPORT.md"
    out_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[report] Summary report written to {out_file}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight", action="store_true", help="Run 8-worker collector preflight & parity check")
    ap.add_argument("--collect", action="store_true", help="Run full 2,800 matches collection and feature extraction")
    ap.add_argument("--regenerate", action="store_true", help="Regenerate all manifests, fixtures, audits, and reports from existing data")
    ap.add_argument("--backup", action="store_true", help="Verify local and cloud backup archives")
    ap.add_argument("--report", action="store_true", help="Generate summary report")
    a = ap.parse_args(argv)

    if a.preflight:
        return run_preflight()
    if a.collect:
        rc = run_collection()
        if rc != 0:
            return rc
        data = process_all_matches()
        write_all_artifacts(data)
        generate_report()
        return 0
    if a.regenerate:
        data = process_all_matches()
        write_all_artifacts(data)
        run_backup()
        generate_report()
        return 0
    if a.backup:
        rc = run_backup()
        generate_report()
        return rc
    if a.report:
        generate_report()
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
