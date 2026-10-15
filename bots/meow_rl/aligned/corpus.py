"""Freeze the shared calibration corpus and its disjoint diagnostic split (task T0061, docs/MEOWMIND_RL_ALIGNED.md).

    python bots/meow_rl/aligned/corpus.py --batches runs/meow_rl_pool2/large_v_s101/batches \\
        runs/meow_rl_pool2/small_v_s101/batches --out data/meow_rl_aligned/corpus

Selects, deterministically and before any new game, 16,000 learner decisions for calibration and 16,000 for export
diagnostics from T0059's retained training batches (batches/update_*.npz written with --keep-features, each with its
update_*.json episode records). No game is played. Every export of the experiment calibrates on calibration.npz;
agreement is measured on diagnostic.npz, whose maps never appear in the calibration split.

Strata (from the episode records and the reference's own features; docs/MEOWMIND_RL_ALIGNED.md lists the indices):
  * team count 2-15 (the episode record);
  * phase: turn 1-133, 134-266, 267-400;
  * a four-bit tactical signature: carrier (an own unit carries a flag, actor feature 1), combat (an enemy is on an
    attack line of a living own unit, actor features 57/66/75/84 of the stay move), low HP (a living own unit at
    HP <= 34, actor feature 0), respawn (an own unit waits to respawn, value features 11/23/35). Combat is a proxy (an
    attack line, not an attack); respawn means waiting, the exact new-life flag is not recoverable from features.
Split: a map group (team count, map seed; the same engine for every run, whichever run played it) is diagnostic
when sha256("<seed>|diagnostic|<group>") mod 5 == 0, so a map never reaches both splits through two runs.
Quotas: 1,143 decisions per count for 2-13 and 1,142 for 14-15 (16,000), split evenly over the phases (remainder to
the earliest). Within a (count, phase) cell the selector cycles over the non-empty signatures in numeric order and
takes each signature's next row by sha256("<seed>|<file>|<episode>|<turn>") (then file and row). A shortage is filled
from the same count's other phases, then from the other counts in order; nothing is duplicated or manufactured, and
every shortage is reported. The same rule selects the diagnostic split from the diagnostic groups.

MANIFEST.json records the parent files' sha256, the seed, the rules, requested/available/selected counts per stratum,
the shortages and every selected (file, row, episode, turn, teams, seat, map seed). Each output .npz holds feats
[N,3,5,112], mask [N,125] (bool), value_feats [N,416] and those identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

SCHEMA = "ctf-meow-rl-aligned-corpus-1"
SEED = "meow_rl_aligned_corpus_v1"
COUNTS = list(range(2, 16))
TOTAL = 16_000
PHASES = ((1, 133), (134, 266), (267, 400))
CARRIER, LOW_HP, ATTACK = 1, 0, (57, 66, 75, 84)
RESPAWN = (11, 23, 35)
BITS = ("carrier", "combat", "low_hp", "respawn")


def quotas(total: int = TOTAL) -> dict[int, int]:
    base, extra = divmod(total, len(COUNTS))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(COUNTS)}


def phase_of(turn: int) -> int:
    for i, (lo, hi) in enumerate(PHASES):
        if lo <= turn <= hi:
            return i
    raise ValueError(f"turn {turn} outside 1-400")


def signature(feats: np.ndarray, value_feats: np.ndarray) -> np.ndarray:
    """The 4-bit tactical signature of each decision: feats [N,3,5,112], value_feats [N,416] -> [N] in 0..15."""
    stay = feats[:, :, 0, :]                                 # the stay move's features of the three own units
    hp = stay[:, :, LOW_HP]
    alive = hp >= 0
    carrier = (stay[:, :, CARRIER] > 0.5).any(axis=1)
    combat = ((stay[:, :, list(ATTACK)] > 0.5).any(axis=2) & alive).any(axis=1)
    low = (alive & (hp > 0) & (hp <= np.float32(0.34))).any(axis=1)
    respawn = (value_feats[:, list(RESPAWN)] > 0).any(axis=1)
    return carrier.astype(int) | combat.astype(int) << 1 | low.astype(int) << 2 | respawn.astype(int) << 3


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_key(seed: str, file: str, episode: int, turn: int) -> str:
    return hashlib.sha256(f"{seed}|{file}|{episode}|{turn}".encode()).hexdigest()


def inventory(dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    """(parent files, one row per decision with its identities and strata), reading one batch at a time."""
    files, rows = [], []
    for d in dirs:
        run = Path(d).resolve().parent.name
        npzs = sorted(Path(d).glob("update_*.npz"))
        if not npzs:
            raise SystemExit(f"no update_*.npz in {d}")
        for npz in npzs:
            js = npz.with_suffix(".json")
            if not js.is_file():
                raise SystemExit(f"{npz}: no companion {js.name} (episode records)")
            meta = json.loads(js.read_text(encoding="utf-8"))
            episodes = {int(r["index"]): r for r in meta["episodes"]}
            with np.load(npz) as z:
                if "feats" not in z or "value_feats" not in z:
                    raise SystemExit(f"{npz}: no features (the run needs --keep-features)")
                ep, turn = z["episode"].astype(np.int64), z["turn"].astype(np.int64)
                sig = signature(z["feats"], z["value_feats"])
            name = f"{run}/{npz.name}"
            files.append({"file": name, "path": str(npz), "sha256": sha256(npz), "json_sha256": sha256(js), "rows": int(len(ep))})
            seen = set()
            for i, (e, t, s) in enumerate(zip(ep, turn, sig)):
                rec = episodes.get(int(e))
                if rec is None:
                    raise SystemExit(f"{name} row {i}: episode {e} has no record")
                if (int(e), int(t)) in seen:
                    raise SystemExit(f"{name}: duplicate decision episode {e} turn {t}")
                seen.add((int(e), int(t)))
                rows.append({"file": name, "row": i, "episode": int(e), "turn": int(t), "teams": int(rec["teams"]),
                             "seat": int(rec["seat"]), "map_seed": int(rec["map_seed"]), "run": run, "sig": int(s),
                             "phase": phase_of(int(t))})
    return files, rows


def is_diagnostic(seed: str, row: dict) -> bool:
    group = f"{row['teams']}|{row['map_seed']}"
    return int(hashlib.sha256(f"{seed}|diagnostic|{group}".encode()).hexdigest(), 16) % 5 == 0


def take(cells: dict[int, deque], n: int) -> list[dict]:
    """Up to n rows, cycling over the non-empty signatures in numeric order (each list already in selection order)."""
    out = []
    while len(out) < n:
        progressed = False
        for s in sorted(cells):
            if cells[s] and len(out) < n:
                out.append(cells[s].popleft())
                progressed = True
        if not progressed:
            break
    return out


def select(rows: list[dict], seed: str, total: int = TOTAL) -> tuple[list[dict], dict]:
    cells: dict[tuple[int, int], dict[int, deque]] = defaultdict(lambda: defaultdict(deque))
    for r in sorted(rows, key=lambda r: (order_key(seed, r["file"], r["episode"], r["turn"]), r["file"], r["row"])):
        cells[(r["teams"], r["phase"])][r["sig"]].append(r)
    available = {f"{c}/{p}": sum(len(v) for v in cells[(c, p)].values()) for c in COUNTS for p in range(len(PHASES))}
    q = quotas(total)
    chosen, report = [], {"requested": {}, "selected_first_pass": {}, "shortage_by_count": {}}
    for c in COUNTS:
        per = [q[c] // len(PHASES) + (1 if p < q[c] % len(PHASES) else 0) for p in range(len(PHASES))]
        got = 0
        for p, n in enumerate(per):
            t = take(cells[(c, p)], n)
            chosen += t
            got += len(t)
            report["requested"][f"{c}/{p}"] = n
            report["selected_first_pass"][f"{c}/{p}"] = len(t)
        short = q[c] - got
        if short:                                                     # the same count's other phases first
            for p in range(len(PHASES)):
                t = take(cells[(c, p)], short)
                chosen += t
                short -= len(t)
        report["shortage_by_count"][str(c)] = short
    left = total - len(chosen)
    report["filled_across_counts"] = 0
    for c in COUNTS:                                                  # then the other counts, in order
        for p in range(len(PHASES)):
            if left <= 0:
                break
            t = take(cells[(c, p)], left)
            chosen += t
            left -= len(t)
            report["filled_across_counts"] += len(t)
    report["missing"] = max(0, left)
    report["available"] = available
    return chosen, report


def strata(sel: list[dict]) -> dict:
    out = {"teams": defaultdict(int), "phase": defaultdict(int), **{b: 0 for b in BITS}, "signature": defaultdict(int)}
    for r in sel:
        out["teams"][str(r["teams"])] += 1
        out["phase"][str(r["phase"])] += 1
        out["signature"][str(r["sig"])] += 1
        for i, b in enumerate(BITS):
            out[b] += (r["sig"] >> i) & 1
    return {k: dict(v) if isinstance(v, defaultdict) else v for k, v in out.items()}


def materialize(sel: list[dict], files: list[dict], path: Path) -> dict:
    """Write the selected rows' arrays (one batch file at a time) and return the array digests."""
    by_file = defaultdict(list)
    for k, r in enumerate(sel):
        by_file[r["file"]].append((r["row"], k))
    paths = {f["file"]: Path(f["path"]) for f in files}
    n = len(sel)
    F = np.zeros((n, 3, 5, 112), np.float32)
    M = np.zeros((n, 125), bool)
    V = np.zeros((n, 416), np.float32)
    for name, pairs in by_file.items():
        with np.load(paths[name]) as z:
            idx = np.array([i for i, _ in pairs])
            dst = np.array([k for _, k in pairs])
            F[dst], V[dst] = z["feats"][idx], z["value_feats"][idx]
            M[dst] = np.unpackbits(z["mask"][idx], axis=1)[:, :125].astype(bool)
    ids = {k: np.array([r[k] for r in sel], np.int64) for k in ("row", "episode", "turn", "teams", "seat", "map_seed", "sig")}
    arrays = {"feats": F, "mask": M, "value_feats": V, "source_file": np.array([r["file"] for r in sel]), **ids}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items() if k != "source_file"}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", nargs="+", required=True, help="training runs' batches/ directories (T0059)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default=SEED, help="selection seed (a label; it plays no game)")
    ap.add_argument("--total", type=int, default=TOTAL)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "MANIFEST.json").exists():
        raise SystemExit(f"{out} already holds a frozen corpus: never reselect into it")
    files, rows = inventory([Path(d) for d in a.batches])
    cal_pool = [r for r in rows if not is_diagnostic(a.seed, r)]
    diag_pool = [r for r in rows if is_diagnostic(a.seed, r)]
    cal, cal_rep = select(cal_pool, a.seed, a.total)
    diag, diag_rep = select(diag_pool, a.seed, a.total)
    cal_maps = {(r["teams"], r["map_seed"]) for r in cal}
    if cal_maps & {(r["teams"], r["map_seed"]) for r in diag}:
        raise SystemExit("calibration and diagnostic splits share a map group")
    digests = {"calibration": materialize(cal, files, out / "calibration.npz"),
               "diagnostic": materialize(diag, files, out / "diagnostic.npz")}
    keep = ("file", "row", "episode", "turn", "teams", "seat", "map_seed", "sig")
    manifest = {"schema": SCHEMA, "generator": "bots/meow_rl/aligned/corpus.py", "seed": a.seed, "total": a.total,
                "rules": {"phases": PHASES, "signature_bits": BITS, "low_hp_max": 0.34, "attack_line_features": ATTACK,
                          "carrier_feature": CARRIER, "respawn_value_features": RESPAWN,
                          "diagnostic_groups": "sha256(seed|diagnostic|teams|map_seed) % 5 == 0", "quotas": quotas(a.total)},
                "parents": [{k: f[k] for k in ("file", "sha256", "json_sha256", "rows")} for f in files],
                "pool_rows": {"calibration": len(cal_pool), "diagnostic": len(diag_pool)},
                "splits": {"calibration": {"rows": len(cal), "selection": cal_rep, "strata": strata(cal),
                                           "array_sha256": digests["calibration"],
                                           "file_sha256": sha256(out / "calibration.npz"),
                                           "map_groups": len(cal_maps)},
                           "diagnostic": {"rows": len(diag), "selection": diag_rep, "strata": strata(diag),
                                          "array_sha256": digests["diagnostic"],
                                          "file_sha256": sha256(out / "diagnostic.npz"),
                                          "map_groups": len({(r["teams"], r["map_seed"]) for r in diag})}},
                "selected": {"calibration": [[r[k] for k in keep] for r in cal],
                             "diagnostic": [[r[k] for k in keep] for r in diag], "columns": list(keep)}}
    tmp = out / "MANIFEST.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=None, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(out / "MANIFEST.json")
    return manifest


if __name__ == "__main__":
    m = main()
    print(json.dumps({s: {k: v for k, v in m["splits"][s].items() if k in ("rows", "strata", "map_groups")}
                      for s in m["splits"]}, indent=1))
