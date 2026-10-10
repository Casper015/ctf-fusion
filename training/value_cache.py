"""Compact V-only training caches: the value columns of one feature family, its targets and a few row tags.

    python -m training.value_cache --shards data/shards_v --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \\
        --family v3 --out data/vcache/v12xxl_v3

A full extractor record is 923 int16 columns; a v3 value row is 108 (v2: 54). Wide value networks are trained
for many epochs over the same rows, so the cache keeps only what training and evaluation read, as plain .npy
files that load memory-mapped: the torch backend streams bounded windows of them to the GPU
(--gpu-resident-gb, --gpu-window-rows) instead of holding whole records in host memory.

Per split (<out>/<split>/): v (N, D) int16 raw value inputs of the family; v_y (N, 3) float32 targets
(outcome, capture within 20, own death within 10); weight float32; teams int8; turn int16; match int32 (index
into the shards' match list); map int64 (map seed: the clustering unit of every interval); tags uint8.

Tags (evaluation subsets, computed from the full record whatever the family, so every arm is scored on the
same rows): 1 carrier (a flag is carried by us or by a rival), 2 drop (an available dropped flag), 4
interception (an enemy carrier one of our shooters can reach this turn; v3 columns, else the v2 shots-now
count), 8 close score (a rival within one capture: |margin| <= 1 or a tie).

meta.json records the feature family (name, digest, layout), the source shards' data manifest and options,
and the row counts; train_value and evaluate_value refuse a cache whose family the extractor no longer
reproduces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion import families
from fusion.models import value_inputs
from fusion.schema import layout as feature_layout
from training.common import data_manifest
from training.dataset import check_schema, count_key, load, load_index

FIELDS = ("rec", "v_y", "weight", "teams", "turn", "match")
TAGS = {"carrier": 1, "drop": 2, "interception": 4, "close": 8}


def row_tags(rec: np.ndarray, layout) -> np.ndarray:
    g = layout.get(rec, "g")
    gi = {n: i for i, n in enumerate(layout.names("global"))}
    v = layout.get(rec, "v")
    vi = {n: i for i, n in enumerate(layout.value_names("v2"))}
    carrier = (g[:, gi["flags_carried_own"]] > 0) | (g[:, gi["flags_carried_enemy"]] > 0)
    drop = g[:, gi["flags_dropped"]] > 0
    if layout.has("vx"):
        x = layout.get(rec, "vx")
        xi = {n: i for i, n in enumerate(layout.names("value_ext"))}
        intercept = (x[:, xi["ec0_shots_reach"]] > 0) | (x[:, xi["ec1_shots_reach"]] > 0)
    else:
        intercept = v[:, vi["carrier_shots"]] > 0
    close = (np.abs(g[:, gi["margin"]]) <= 1) | (v[:, vi["rivals_tied"]] > 0)
    return (carrier * TAGS["carrier"] | drop * TAGS["drop"] | intercept * TAGS["interception"]
            | close * TAGS["close"]).astype(np.uint8)


def build(shards: str, counts: list[int], family: str, out: Path, splits=("train", "val", "test")) -> dict:
    L = feature_layout()
    check_schema(shards, (family,))  # refuse a wrong family here; below, a missing split of a count is skipped
    index = load_index(shards)
    maps = np.array([m["map_seed"] for m in index["matches"]], np.int64)
    meta = {"features": families.record(L.schema, family), "shards": str(shards), "data": data_manifest(shards, counts),
            "options": index["options"], "rows": {}, "tags": TAGS}
    for split in splits:
        parts = []
        for t in counts:
            try:
                d = load(shards, [t], split, FIELDS, families=(family,))
            except SystemExit:
                continue  # no shards of this count in this split
            parts.append({"v": value_inputs(d["rec"], L, normalize=False, family=family).astype(np.int16),
                          "v_y": d["v_y"], "weight": d["weight"], "teams": d["teams"], "turn": d["turn"],
                          "match": d["match"], "map": maps[d["match"]], "tags": row_tags(d["rec"], L)})
        if not parts:
            continue
        (out / split).mkdir(parents=True, exist_ok=True)
        for k in parts[0]:
            np.save(out / split / f"{k}.npy", np.concatenate([p[k] for p in parts]))
        meta["rows"][split] = int(sum(len(p["v"]) for p in parts))
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def load_cache(cache: str | Path, split: str, family: str, mmap: bool = True) -> dict[str, np.ndarray]:
    """One split of a cache (memory-mapped), after checking its family against the extractor."""
    cache = Path(cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    feats = meta["features"]
    if feats["family"] != family:
        raise SystemExit(f"{cache} holds value family {feats['family']}, not {family}")
    families.check(feature_layout().schema, feats, str(cache))
    d = cache / split
    if not d.is_dir():
        raise SystemExit(f"{cache} has no {split} split")
    return {p.stem: np.load(p, mmap_mode="r" if mmap else None) for p in sorted(d.glob("*.npy"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--teams", type=int, nargs="+", required=True)
    ap.add_argument("--family", choices=families.VALUE, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta = build(a.shards, a.teams, a.family, Path(a.out), a.splits)
    print(f"[value_cache] {a.out}: family {a.family} rows {meta['rows']} (count keys {[count_key(t) for t in a.teams]})")


if __name__ == "__main__":
    main()
