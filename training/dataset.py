"""Turn raw match files into training shards (training/specs/DATA_FORMAT.md).

    python -m training.dataset --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 8
    python -m training.dataset --teams ... --append      # phase 2: add new matches, keep every existing shard

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

Shards are stored per team count (data/shards/nXX/<split>/), so the general
model and every dedicated model read the same files without duplication.
--append processes only matches the index does not list yet and writes new
shard files next to the old ones; sampling options must match the index.
index.json carries the dataset manifest: matches per team count, seat and
family distribution, AI_Test and repository commits, collection seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusion import matchlog, native
from fusion.labels import STRATA, label_match
from fusion.paths import REPO, git_commit
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
            "families": families, "seats": [s["name"] for s in match["seats"]], "faults": len(match["faults"]),
            "ai_test_commit": match.get("ai_test_commit", "unknown")}
    ood = np.array([f in holdout for f in rows["family_name"]], bool)
    return meta, rows, ood


def count_key(teams: int) -> str:
    return f"n{teams:02d}"


def manifest(index: dict) -> dict:
    from collections import Counter  # noqa: PLC0415
    matches = index["matches"]
    seeds = Counter(m["match_id"].split("-")[1] for m in matches)
    return {
        "matches": len(matches),
        "matches_by_count": dict(sorted(Counter(count_key(m["teams"]) for m in matches).items())),
        "matches_by_split": dict(Counter(m["split"] for m in matches)),
        "seats_by_bot": dict(Counter(b for m in matches for b in m["seats"]).most_common()),
        "seats_by_family": dict(Counter(f for m in matches for f in m["families"]).most_common()),
        "ai_test_commits": sorted({m.get("ai_test_commit", "unknown") for m in matches}),
        "collection_seeds": dict(seeds),
        "repo_commit": git_commit(REPO),
    }


def data_digest(index: dict, counts) -> str:
    """Fingerprint of the matches behind some team counts (recorded in every run's config)."""
    ids = sorted(m["match_id"] for m in index["matches"] if m["teams"] in set(counts))
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def write_shards(out_dir: Path, buffers, index, args):
    for (key, split), parts in sorted(buffers.items()):
        d = out_dir / key / split
        d.mkdir(parents=True, exist_ok=True)
        merged = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
        n = len(merged["rec"])
        first = len(list(d.glob("shard_*.npz")))
        for i, start in enumerate(range(0, n, args.shard_rows)):
            np.savez_compressed(d / f"shard_{first + i:04d}.npz", **{k: v[start:start + args.shard_rows] for k, v in merged.items()})
        index["counts"][f"{key}/{split}"] = index["counts"].get(f"{key}/{split}", 0) + n
    index["manifest"] = manifest(index)
    index["history"].append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "matches": len(index["matches"]),
                             "added": sum(len(p["rec"]) for parts in buffers.values() for p in parts)})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


def load_index(shard_dir) -> dict:
    return json.loads((Path(shard_dir) / "index.json").read_text(encoding="utf-8"))


def load(shard_dir, counts, split: str, fields=FIELDS) -> dict[str, np.ndarray]:
    """Concatenate one split over team counts. Adds `qh` when training.heuristic_baseline has run on every shard."""
    if isinstance(counts, int):
        counts = [counts]
    parts, qh = [], []
    for t in counts:
        d = Path(shard_dir) / count_key(t) / split
        shards = sorted(d.glob("shard_*.npz"))
        for shard in shards:
            parts.append({k: v for k, v in np.load(shard).items() if k in fields})
            q = d / shard.name.replace("shard_", "qh_").replace(".npz", ".npy")
            qh.append(np.load(q) if q.exists() else None)
    if not parts:
        raise SystemExit(f"no {split} shards for team counts {list(counts)} in {shard_dir}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in fields}
    if all(q is not None for q in qh):
        out["qh"] = np.concatenate(qh)
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
    ap.add_argument("--append", action="store_true", help="add matches missing from an existing index")
    args = ap.parse_args(argv)
    args.keep_map = DEFAULT_KEEP | {k: float(v) for k, v in (kv.split("=") for kv in args.keep.split(",") if kv)}
    out_dir = Path(args.out)
    options = {"stride": args.stride, "keep": args.keep_map, "holdout_family": sorted(args.holdout_family)}
    index = {"schema": native.schema(), "options": options, "families": [], "matches": [], "counts": {}, "history": []}
    if (out_dir / "index.json").exists():
        if not args.append:
            raise SystemExit(f"{out_dir} already holds a dataset; pass --append to extend it or choose another --out")
        index = load_index(out_dir)
        if index["options"] != options:
            raise SystemExit(f"--append needs the same sampling options as the index: {index['options']}")
        if index["schema"] != native.schema():
            raise SystemExit("the feature schema changed since this dataset was built; rebuild it into a new --out")
    known = {m["match_id"] for m in index["matches"]}
    files = [f for f in matchlog.iter_matches(args.raw, args.teams) if f.name.split(".")[0] not in known]
    print(f"[dataset] {len(files)} new matches ({len(known)} already indexed)", flush=True)
    native.extractor()  # compile before forking
    jobs = [(f, args.stride, args.keep_map, set(args.holdout_family)) for f in files]
    buffers = defaultdict(list)
    families, matches = index["families"], index["matches"]
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
        for split, sel in ((meta["split"], ~ood), ("ood", ood)):
            if sel.any():
                buffers[(count_key(meta["teams"]), split)].append({k: v[sel] for k, v in rows.items()})
        if (i + 1) % 200 == 0:
            print(f"[dataset] {i + 1}/{len(files)}", flush=True)
    if pool:
        pool.close()
    write_shards(out_dir, buffers, index, args)
    for k, v in sorted(index["counts"].items()):
        print(f"[dataset] {k}: {v} rows")


if __name__ == "__main__":
    main()
