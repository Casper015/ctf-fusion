"""Turn raw match files into training shards (docs/DATA_FORMAT.md).

    python -m training.dataset --raw data/raw --out data/shards --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 8

For every match: rebuild a seat stream, run the C++ feature extractor (the
same encoder the bot runs), label every (turn, team) row from the reply lines,
then keep a stratified subsample:

  * every --stride-th turn (offset per match) to thin out correlated turns;
  * per-stratum keep rates (--keep), with importance weight stride / rate so
    weighted metrics still describe the natural turn distribution.

Rows are split by match: train / val / test from a hash of the match id, so
no match ever feeds two splits. --holdout-family moves every row whose acting
team belongs to a family into an extra `ood` split (generalization to an
unseen opponent family); `ood` rows never enter training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusion import matchlog, native
from fusion.labels import STRATA, label_match
from fusion.paths import REPO, load_profile
from fusion.schema import Layout

SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))
DEFAULT_KEEP = {"flag": 1.0, "contact": 1.0, "respawn": 0.6, "quiet": 0.35}
FIELDS = ("rec", "y_move", "y_joint", "y_act", "y_tgt", "blocked", "valid", "v_y", "stratum", "weight",
          "match", "teams", "family", "turn", "team")


def unit_hash(*parts) -> float:
    h = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") / 2.0 ** 64


def split_of(match_id: str) -> str:
    x, acc = unit_hash("split", match_id), 0.0
    for name, share in SPLITS:
        acc += share
        if x < acc:
            return name
    return SPLITS[-1][0]


def process(job):
    """One match -> selected rows (dict of arrays) plus its metadata."""
    path, stride, keep, holdout = job
    match = matchlog.load(path)
    layout = Layout(native.schema())
    rec = native.extract(matchlog.seat_stream(match, 0))
    lab = label_match(match, rec, layout)
    n = match["teams"]
    turn = rec[:, 0].astype(np.int64)
    team = rec[:, 1].astype(np.int64)
    rng = np.random.default_rng(int(unit_hash("rows", match["match_id"]) * 2 ** 62))
    offset = int(unit_hash("stride", match["match_id"]) * stride)
    rate = np.array([keep[s] for s in STRATA])[lab["stratum"]]
    take = ((turn - 1) % stride == offset) & (rng.random(len(rec)) < rate)
    families = [s["family"] for s in match["seats"]]
    rows = {k: v[take] for k, v in lab.items()} | {
        "rec": rec[take], "weight": (stride / rate[take]).astype(np.float32), "turn": turn[take].astype(np.int16),
        "team": team[take].astype(np.int8), "teams": np.full(int(take.sum()), n, np.int8)}
    rows["family_name"] = [families[t] for t in team[take]]
    meta = {"match_id": match["match_id"], "file": str(path), "teams": n, "split": split_of(match["match_id"]),
            "families": families, "seats": [s["name"] for s in match["seats"]], "faults": len(match["faults"])}
    ood = np.array([f in holdout for f in rows["family_name"]], bool)
    return meta, rows, ood


def write_shards(out_dir: Path, buffers, families, matches, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for (tier, split), parts in sorted(buffers.items()):
        d = out_dir / tier / split
        d.mkdir(parents=True, exist_ok=True)
        merged = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
        n = len(merged["rec"])
        counts[f"{tier}/{split}"] = n
        for i, start in enumerate(range(0, max(n, 1), args.shard_rows)):
            np.savez_compressed(d / f"shard_{i:04d}.npz", **{k: v[start:start + args.shard_rows] for k, v in merged.items()})
    index = {"schema": native.schema(), "families": families, "matches": matches, "counts": counts,
             "options": {"stride": args.stride, "keep": args.keep_map, "holdout_family": args.holdout_family}}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return counts


def load(shard_dir, tier: str, split: str, fields=FIELDS) -> dict[str, np.ndarray]:
    """Concatenate one tier's split. Adds `qh` when training.heuristic_baseline has run."""
    d = Path(shard_dir) / tier / split
    parts = [dict(np.load(p)) for p in sorted(d.glob("shard_*.npz"))]
    if not parts:
        raise SystemExit(f"no shards in {d}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in fields}
    qh = sorted(d.glob("qh_*.npy"))
    if qh and len(qh) == len(parts):
        out["qh"] = np.concatenate([np.load(p) for p in qh])
    return out


def torch_dataset(arrays: dict):
    """Optional PyTorch view over loaded arrays (requires torch; numpy training does not)."""
    import torch  # noqa: PLC0415

    class _Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(arrays["rec"])

        def __getitem__(self, i):
            return {k: torch.as_tensor(v[i]) for k, v in arrays.items()}

    return _Rows()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "data" / "raw"))
    ap.add_argument("--out", default=str(REPO / "data" / "shards"))
    ap.add_argument("--teams", type=int, nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--keep", default="", help="per-stratum keep rates, e.g. quiet=0.35,respawn=0.6")
    ap.add_argument("--holdout-family", nargs="*", default=[])
    ap.add_argument("--shard-rows", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    args.keep_map = DEFAULT_KEEP | {k: float(v) for k, v in (kv.split("=") for kv in args.keep.split(",") if kv)}
    tiers = {t: load_profile(t)["tier"] for t in args.teams}
    files = list(matchlog.iter_matches(args.raw, args.teams))
    print(f"[dataset] {len(files)} matches", flush=True)
    native.extractor()  # compile before forking
    jobs = [(f, args.stride, args.keep_map, set(args.holdout_family)) for f in files]
    buffers, families, matches = defaultdict(list), [], []
    pool = mp.Pool(args.workers) if args.workers > 1 else None
    results = pool.imap(process, jobs, chunksize=4) if pool else map(process, jobs)
    for i, (meta, rows, ood) in enumerate(results):
        midx = len(matches)
        matches.append(meta)
        for name in rows["family_name"]:
            if name not in families:
                families.append(name)
        fam = np.array([families.index(f) for f in rows.pop("family_name")], np.int16)
        rows |= {"family": fam, "match": np.full(len(fam), midx, np.int32)}
        tier = tiers[meta["teams"]]
        for split, sel in ((meta["split"], ~ood), ("ood", ood)):
            if sel.any():
                buffers[(tier, split)].append({k: v[sel] for k, v in rows.items()})
        if (i + 1) % 200 == 0:
            print(f"[dataset] {i + 1}/{len(files)}", flush=True)
    if pool:
        pool.close()
    counts = write_shards(Path(args.out), buffers, families, matches, args)
    for k, v in counts.items():
        print(f"[dataset] {k}: {v} rows")


if __name__ == "__main__":
    main()
