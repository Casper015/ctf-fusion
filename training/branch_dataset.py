"""Turn branch continuations (training.branch_collect) into ranking data for the value network.

    python -m training.branch_dataset --branches data/branches/v12_ponly_xxl --out data/rank/v12_ponly_xxl

For every root, each candidate's successor state (the TURN block after the branch settlement) is encoded by
the extractor, the same code the bot runs (INIT + root block + successor block, so the successor's history
features see the root); its label is the mean final pairwise utility of its continuations.

Pairs: every two candidates of a root. A pair is resolved when the outcome difference exceeds the tie margin
(--tie-steps pairwise steps of 0.5 / (teams - 1), and with several repeats also --tie-z standard errors);
unresolved pairs are kept as uncertain (label -1, weight 0) and counted, never forced into a hard label.
Each root's resolved pairs share weight 1, so every root counts once.

Splits come from the map, exactly as training.dataset assigns them (split_of("map-<seed>")): a root, its
branches and the match they came from always share one split.

Output, per split: <out>/<split>/rank.npz with
  cand_rec  (C, width) int16   successor records of every candidate (layout of the extractor)
  cand_y    (C,) float32       mean continuation outcome;  cand_n (C,) repeats
  cand_root (C,) int32         root index;  cand_on_policy (C,) bool
  pair_a, pair_b (P,) int32    candidate indices;  pair_y (P,) int8 (1: a better, 0: b better, -1: uncertain)
  pair_w    (P,) float32       0 for uncertain pairs
  root_teams, root_turn, root_map (R,)
and <out>/index.json with the schema, counts and options.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusion import matchlog, native
from training.dataset import split_of


def successor_record(row: dict, branch: dict) -> np.ndarray | None:
    """The extractor record of our seat at turn t + 1 of this branch (None when the match ended at turn t)."""
    if branch["next_block"] is None:
        return None
    stream = matchlog.init_for_seat(row["init"], row["seat"]) + branch["root_block"] + branch["next_block"]
    rec = native.extract(stream)
    teams = row["teams"]
    return rec[teams + row["seat"]]  # rows are turn-major, team-minor: the second turn, our team


def build(files: list[Path], tie_steps: float, tie_z: float) -> dict[str, dict]:
    out = defaultdict(lambda: defaultdict(list))
    stats = defaultdict(lambda: defaultdict(int))
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            row = json.load(fh)
        if "dropped" in row:
            stats["all"]["dropped_roots"] += 1
            continue
        split = split_of(f"map-{row['map_seed']}")
        d, st = out[split], stats[split]
        by_cand = defaultdict(list)
        for b in row["branches"]:
            by_cand[b["candidate"]].append(b)
        recs, ys, ns, se, cands = [], [], [], [], []
        for c in sorted(by_cand):
            bs = by_cand[c]
            if len({b["next_block"] for b in bs}) != 1:
                raise ValueError(f"{f}: repeats of candidate {c} reached different successor states")
            rec = successor_record(row, bs[0])
            if rec is None:
                continue
            o = np.array([b["outcome"] for b in bs])
            recs.append(rec)
            cands.append(c)
            ys.append(o.mean())
            ns.append(len(o))
            se.append(o.std(ddof=1) / np.sqrt(len(o)) if len(o) > 1 else 0.0)
        if len(recs) < 2:
            st["roots_without_pairs"] += 1
            continue
        root = len(d["root_teams"])
        base = len(d["cand_y"])
        d["root_teams"].append(row["teams"])
        d["root_turn"].append(row["turn"])
        d["root_map"].append(row["map_seed"])
        for i, (rec, y, n) in enumerate(zip(recs, ys, ns)):
            d["cand_rec"].append(rec)
            d["cand_y"].append(y)
            d["cand_n"].append(n)
            d["cand_root"].append(root)
            d["cand_on_policy"].append(bool(by_cand[cands[i]][0]["on_policy"]))
        margin = tie_steps * 0.5 / max(row["teams"] - 1, 1)
        pairs = []
        for i, j in itertools.combinations(range(len(recs)), 2):
            diff = ys[i] - ys[j]
            noise = tie_z * np.hypot(se[i], se[j])
            label = -1 if abs(diff) <= max(margin, noise) else int(diff > 0)
            pairs.append((base + i, base + j, label))
        resolved = sum(p[2] >= 0 for p in pairs)
        st["roots"] += 1
        st["pairs"] += len(pairs)
        st["resolved_pairs"] += resolved
        for a, b, label in pairs:
            d["pair_a"].append(a)
            d["pair_b"].append(b)
            d["pair_y"].append(label)
            d["pair_w"].append(1.0 / resolved if label >= 0 else 0.0)
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tie-steps", type=float, default=1.0, help="tie margin in pairwise steps of 0.5 / (teams - 1)")
    ap.add_argument("--tie-z", type=float, default=2.0, help="with repeats: also require |difference| > z standard errors")
    a = ap.parse_args(argv)
    files = sorted(Path(a.branches).rglob("*.json.gz"))
    data, stats = build(files, a.tie_steps, a.tie_z)
    out = Path(a.out)
    dtypes = {"cand_rec": np.int16, "cand_y": np.float32, "cand_n": np.int16, "cand_root": np.int32, "cand_on_policy": bool,
              "pair_a": np.int32, "pair_b": np.int32, "pair_y": np.int8, "pair_w": np.float32,
              "root_teams": np.int8, "root_turn": np.int16, "root_map": np.int64}
    for split, d in data.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(d[k], dtype=dtypes[k]) for k in dtypes}
        np.savez_compressed(out / split / "rank.npz", **arrays)
    index = {"schema": native.schema(), "source": str(a.branches), "files": len(files),
             "options": {"tie_steps": a.tie_steps, "tie_z": a.tie_z}, "stats": {k: dict(v) for k, v in stats.items()}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    for k, v in sorted(index["stats"].items()):
        print(f"[branch_dataset] {k}: {v}")


if __name__ == "__main__":
    main()
