"""Behavior distances, A/A noise and behavior clusters from fingerprints.

    python -m roster.distance --suite data/roster/probe_suite.json --fingerprints data/roster/fingerprints \\
        --out data/roster/behavior_distance.json

For two runs on the same probe, over turns where both replied and units that
were on the board, three disagreement rates:

  move    requested moves differ (canonical, a wall bump is a stay);
  action  wait / pickup / drop / attack differ;
  target  attack targets differ, among units both runs used to attack.

They are computed separately within every event tag (roster.common.TAGS) and
combined with pre-registered weights (move .5, action .3, target .2; a
component without enough opportunities is left out, never counted as
agreement). A probe's distance is the mean over its tags with at least
--min-tag-turns turns, so the rare tactical situations weigh as much as the
quiet majority. Probes are averaged within a team count, team counts equally.

Repeated runs of one bot give its A/A noise (the same code answering the same
history differently: clocks, seeds). Two bots are behaviorally equivalent at a
team count when their distance is within the larger of their noises plus
--margin; clusters are the connected groups, per team count and over all
counts. A shared duel path does not make bots equivalent at 3-15 teams.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusion.paths import REPO
from fusion.protocol import ATTACK
from roster.common import SCHEMA_DISTANCE, TAGS, load_json, save_json, tag_masks
from roster.fingerprint import load_all

WEIGHTS = {"move": 0.5, "action": 0.3, "target": 0.2}
MIN_UNITS_PER_TURN_TAG = 1


def probe_components(runs: list[dict], tags: np.ndarray, active: np.ndarray, min_target_pairs: int):
    """Per tag, per pair of runs: (rate, opportunities) for move, action and target. Shapes (E, E, G)."""
    valid = np.array([r["valid"] for r in runs], bool)          # (E, T)
    move = np.array([r["move"] for r in runs], np.int16)        # (E, T, 3)
    act = np.array([r["act"] for r in runs], np.int16)
    tgt = np.array([r["target"] for r in runs], np.int32)
    unit = valid[:, :, None] & active[None]                      # (E, T, 3)
    tagf = tags.T.astype(np.float32)                             # (T, G)
    E, T = valid.shape
    out = {}
    both = unit[:, None] & unit[None]                            # (E, E, T, 3)
    pairs = both.sum(-1).reshape(E * E, T).astype(np.float32) @ tagf
    for name, arr in (("move", move), ("action", act)):
        dis = ((arr[:, None] != arr[None]) & both).sum(-1).reshape(E * E, T).astype(np.float32) @ tagf
        out[name] = (np.divide(dis, pairs, out=np.full_like(dis, np.nan), where=pairs > 0).reshape(E, E, -1),
                     pairs.reshape(E, E, -1))
    attack = unit & (act == ATTACK)
    tboth = attack[:, None] & attack[None]
    tpairs = tboth.sum(-1).reshape(E * E, T).astype(np.float32) @ tagf
    tdis = ((tgt[:, None] != tgt[None]) & tboth).sum(-1).reshape(E * E, T).astype(np.float32) @ tagf
    rate = np.divide(tdis, tpairs, out=np.full_like(tdis, np.nan), where=tpairs >= min_target_pairs)
    out["target"] = (rate.reshape(E, E, -1), tpairs.reshape(E, E, -1))
    return out


def nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    """Mean over finite entries along `axis`; NaN where there are none (no warnings)."""
    ok = np.isfinite(x)
    n = ok.sum(axis)
    return np.divide(np.where(ok, x, 0.0).sum(axis), n, out=np.full(n.shape, np.nan), where=n > 0)


def combine(comp: dict, tag_turns: np.ndarray, min_tag_turns: int) -> tuple[np.ndarray, np.ndarray]:
    """(probe distance (E, E), per-tag distance (E, E, G)) from the components."""
    num = np.zeros(comp["move"][0].shape, np.float64)
    den = np.zeros_like(num)
    for name, w in WEIGHTS.items():
        rate = comp[name][0]
        ok = np.isfinite(rate)
        num += np.where(ok, w * rate, 0.0)
        den += np.where(ok, w, 0.0)
    per_tag = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    per_tag[..., tag_turns < min_tag_turns] = np.nan
    return nanmean(per_tag, -1), per_tag


def union_clusters(ids: list[str], dist: np.ndarray, noise: dict[str, float], margin: float) -> list[list[str]]:
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = dist[i, j]
            limit = max(noise.get(ids[i], 0.0) or 0.0, noise.get(ids[j], 0.0) or 0.0) + margin
            if np.isfinite(d) and d <= limit:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, b in enumerate(ids):
        groups[find(i)].append(b)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def nan_to_none(x):
    if isinstance(x, np.ndarray):
        return [nan_to_none(v) for v in x]
    if isinstance(x, (list, tuple)):
        return [nan_to_none(v) for v in x]
    return None if x is None or not np.isfinite(x) else round(float(x), 5)


def compute(suite: dict, runs_by_bot: dict[str, list[dict]], min_tag_turns=5, min_target_pairs=3, margin=0.02) -> dict:
    ids = sorted(b for b, runs in runs_by_bot.items() if runs)
    index = {b: i for i, b in enumerate(ids)}
    B, G = len(ids), len(TAGS)
    by_count = defaultdict(lambda: {"sum": np.zeros((B, B)), "n": np.zeros((B, B)), "tag_sum": np.zeros((B, B, G)),
                                    "tag_n": np.zeros((B, B, G)), "noise_sum": np.zeros(B), "noise_n": np.zeros(B),
                                    "comp": {k: [np.zeros((B, B)), np.zeros((B, B))] for k in WEIGHTS}})
    coverage = {}
    for probe in suite["probes"]:
        entities = [(b, r) for b in ids for r in runs_by_bot[b] if r["probe_id"] == probe["probe_id"]]
        if len(entities) < 2:
            continue
        tags = tag_masks(probe["tags"])
        active = (np.array(probe["active"])[:, None] >> np.arange(3)) & 1
        runs = [r for _, r in entities]
        comp = probe_components(runs, tags, active.astype(bool), min_target_pairs)
        tag_turns = tags.sum(1)
        coverage[probe["probe_id"]] = {t: int(n) for t, n in zip(TAGS, tag_turns)}
        dist, per_tag = combine(comp, tag_turns, min_tag_turns)
        acc = by_count[probe["teams"]]
        o = np.array([index[b] for b, _ in entities])
        same = o[:, None] == o[None]
        finite = np.isfinite(dist)
        e, f = np.nonzero(np.triu(same, 1) & finite)       # two runs of one bot: its A/A noise on this probe
        np.add.at(acc["noise_sum"], o[e], dist[e, f])
        np.add.at(acc["noise_n"], o[e], 1)
        e, f = np.nonzero(~same & finite)
        np.add.at(acc["sum"], (o[e], o[f]), dist[e, f])
        np.add.at(acc["n"], (o[e], o[f]), 1)
        e, f = np.nonzero(~same)
        pt = per_tag[e, f]
        ok = np.isfinite(pt)
        np.add.at(acc["tag_sum"], (o[e], o[f]), np.where(ok, pt, 0.0))
        np.add.at(acc["tag_n"], (o[e], o[f]), ok)
        for name in WEIGHTS:
            rate = np.where(tag_turns >= min_tag_turns, comp[name][0][e, f], np.nan)
            m = nanmean(rate, -1)
            good = np.isfinite(m)
            np.add.at(acc["comp"][name][0], (o[e][good], o[f][good]), m[good])
            np.add.at(acc["comp"][name][1], (o[e][good], o[f][good]), 1)
    per_count, noise_count, clusters_count = {}, {}, {}
    for n, acc in sorted(by_count.items()):
        d = np.divide(acc["sum"], acc["n"], out=np.full((B, B), np.nan), where=acc["n"] > 0)
        np.fill_diagonal(d, 0.0)
        per_count[n] = d
        noise_count[n] = {b: (acc["noise_sum"][i] / acc["noise_n"][i] if acc["noise_n"][i] else None) for i, b in enumerate(ids)}
        clusters_count[n] = union_clusters(ids, d, noise_count[n], margin)
    stack = np.stack(list(per_count.values())) if per_count else np.full((1, B, B), np.nan)
    overall = nanmean(stack, 0)
    noise = {b: (float(np.mean([v for v in (noise_count[n][b] for n in per_count) if v is not None]))
                 if any(noise_count[n][b] is not None for n in per_count) else None) for b in ids}
    clusters = union_clusters(ids, overall, noise, margin)
    cluster_of = {b: f"behavior:{g[0]}" for g in clusters for b in g}
    tag_sum = sum(acc["tag_sum"] for acc in by_count.values()) if by_count else np.zeros((B, B, G))
    tag_n = sum(acc["tag_n"] for acc in by_count.values()) if by_count else np.zeros((B, B, G))
    tag_mean = np.divide(tag_sum, tag_n, out=np.full((B, B, G), np.nan), where=tag_n > 0)
    per_bot_tag = nanmean(tag_mean, 1)                            # (B, G): mean distance to the others, per tag
    population = nanmean(per_bot_tag, 0)
    distinctive = {}
    for i, b in enumerate(ids):
        rel = per_bot_tag[i] / np.where(population > 0, population, np.nan)
        order = [g for g in np.argsort(-np.nan_to_num(rel, nan=-1.0)) if np.isfinite(rel[g])][:3]
        distinctive[b] = [{"tag": TAGS[g], "distance_to_others": round(float(per_bot_tag[i, g]), 4),
                           "vs_population": round(float(rel[g]), 3)} for g in order]
    nearest = {}
    for i, b in enumerate(ids):
        row = np.where(np.arange(B) == i, np.nan, overall[i])
        j = int(np.nanargmin(row)) if np.isfinite(row).any() else None
        nearest[b] = None if j is None else {"id": ids[j], "distance": round(float(row[j]), 5)}
    comps = {}
    for name in WEIGHTS:
        s = sum(acc["comp"][name][0] for acc in by_count.values()) if by_count else np.zeros((B, B))
        c = sum(acc["comp"][name][1] for acc in by_count.values()) if by_count else np.zeros((B, B))
        comps[name] = nan_to_none(np.divide(s, c, out=np.full((B, B), np.nan), where=c > 0))
    return {"schema": SCHEMA_DISTANCE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {"weights": WEIGHTS, "min_tag_turns": min_tag_turns, "min_target_pairs": min_target_pairs,
                       "margin": margin, "tags": list(TAGS)},
            "ids": ids, "overall": nan_to_none(overall), "per_count": {str(n): nan_to_none(d) for n, d in per_count.items()},
            "components": comps, "noise": noise,
            "noise_per_count": {str(n): {b: (None if v is None else round(v, 5)) for b, v in d.items()} for n, d in noise_count.items()},
            "clusters": clusters, "cluster_of": cluster_of,
            "clusters_per_count": {str(n): c for n, c in clusters_count.items()},
            "nearest": nearest, "distinctive_tags": distinctive,
            "tag_distance": {b: {t: nan_to_none(float(per_bot_tag[i, g])) for g, t in enumerate(TAGS)} for i, b in enumerate(ids)},
            "coverage_turns": coverage}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--fingerprints", default=str(REPO / "data" / "roster" / "fingerprints"))
    ap.add_argument("--bots", nargs="*", help="bot ids (default: every bot with runs)")
    ap.add_argument("--min-tag-turns", type=int, default=5)
    ap.add_argument("--min-target-pairs", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.02, help="added to the A/A noise for equivalence")
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "behavior_distance.json"))
    a = ap.parse_args(argv)
    suite = load_json(a.suite)
    runs_by_bot = load_all(Path(a.fingerprints))
    if a.bots:
        runs_by_bot = {b: runs_by_bot.get(b, []) for b in a.bots}
    result = compute(suite, runs_by_bot, a.min_tag_turns, a.min_target_pairs, a.margin)
    save_json(a.out, result, indent=None)
    print(f"[distance] {len(result['ids'])} bots, {len(result['clusters'])} behavior clusters over all counts -> {a.out}")


if __name__ == "__main__":
    main()
