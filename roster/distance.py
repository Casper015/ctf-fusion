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
history differently: clocks, seeds). Noise is not evidence of equivalence. A
pair is scored by its excess distance, d(A,B) - (noise(A) + noise(B)) / 2, per
probe: for categorical choices this is half the squared difference between the
two conditional distributions, zero only when they match, so a 50/50 random
policy is not equivalent to either deterministic policy it mixes. A fourth
component, joint, is the rate at which the whole team's requests differ; its
excess separates a coordinated team from independent units with the same
per-unit marginals. A bot with one run per probe has unknown noise, taken as 0
(its distances can only look larger, never equivalent by noise).

Two bots are equivalent at a team count when the one-sided upper bound of the
excess over that count's probes (mean + --z standard errors, at least
--min-probes probes) is within --margin and the joint excess bound within
--joint-margin. Groups are complete-linkage: every pair inside a group is
equivalent, so no chain AB, BC merges A with C, and each group reports its
diameter. Across counts a pair is equivalent only when it is equivalent at
every count (the worst count decides; an unmeasured count is unknown), so a
duel specialist that matches a reference at 3-15 teams stays its own group.
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
STOCHASTIC_NOISE = 0.1   # A/A noise above this marks a bot as stochastic in the report


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
    # joint: on turns where both runs moved at least one unit, does the team's request differ anywhere?
    differ = ((move[:, None] != move[None]) | (act[:, None] != act[None]) | ((tgt[:, None] != tgt[None]) & tboth)) & both
    turns = both.any(-1).reshape(E * E, T).astype(np.float32) @ tagf
    jdis = differ.any(-1).reshape(E * E, T).astype(np.float32) @ tagf
    out["joint"] = (np.divide(jdis, turns, out=np.full_like(jdis, np.nan), where=turns > 0).reshape(E, E, -1),
                    turns.reshape(E, E, -1))
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


def probe_pairs(o: np.ndarray, d: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray]:
    """Per probe: (mean distance between bots (B, B), A/A noise (B,)) over their run pairs; NaN where unmeasured."""
    same = o[:, None] == o[None]
    ok = np.isfinite(d)
    s, n = np.zeros((B, B)), np.zeros((B, B))
    e, f = np.nonzero(~same & ok)
    np.add.at(s, (o[e], o[f]), d[e, f])
    np.add.at(n, (o[e], o[f]), 1)
    ns, nn = np.zeros(B), np.zeros(B)
    e, f = np.nonzero(np.triu(same, 1) & ok)
    np.add.at(ns, o[e], d[e, f])
    np.add.at(nn, o[e], 1)
    return (np.divide(s, n, out=np.full((B, B), np.nan), where=n > 0),
            np.divide(ns, nn, out=np.full(B, np.nan), where=nn > 0))


def excess(pair: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """d(A,B) - (noise(A) + noise(B)) / 2, unknown noise taken as 0."""
    z = np.nan_to_num(noise, nan=0.0)
    return pair - (z[:, None] + z[None]) / 2


def mean_upper(s: np.ndarray, ss: np.ndarray, n: np.ndarray, z: float, min_probes: int):
    """(mean, standard error, mean + z * se) of per-probe values; the bound is NaN below min_probes probes."""
    mean = np.divide(s, n, out=np.full(s.shape, np.nan), where=n > 0)
    var = np.divide(ss - n * np.nan_to_num(mean) ** 2, n - 1, out=np.zeros(s.shape), where=n > 1)
    se = np.sqrt(np.maximum(var, 0.0) / np.maximum(n, 1))
    upper = np.where(n >= max(1, min_probes), mean + z * se, np.nan)
    return mean, se, upper


def complete_clusters(ids: list[str], score: np.ndarray, limit: float = 1.0) -> list[list[str]]:
    """Complete-linkage groups: repeatedly merge the two groups whose worst pair is smallest, while it is <= limit.
    Every pair inside a group then scores <= limit; an unmeasured (NaN) pair never joins."""
    s = np.maximum(score, score.T)                                # NaN propagates: unknown stays unknown
    s = np.where(np.isfinite(s), s, np.inf)
    groups = [[i] for i in range(len(ids))]
    link = s.copy()
    np.fill_diagonal(link, np.inf)
    while len(groups) > 1:
        a, b = np.unravel_index(np.argmin(link), link.shape)
        if not link[a, b] <= limit:
            break
        a, b = min(a, b), max(a, b)
        groups[a] += groups[b]
        merged = np.maximum(link[a], link[b])
        link[a], link[:, a] = merged, merged
        link[a, a] = np.inf
        link = np.delete(np.delete(link, b, 0), b, 1)
        groups.pop(b)
    return sorted((sorted(ids[i] for i in g) for g in groups), key=lambda g: g[0])


def diameter(group: list[str], index: dict[str, int], m: np.ndarray) -> float | None:
    if len(group) < 2:
        return 0.0
    k = [index[b] for b in group]
    sub = m[np.ix_(k, k)]
    return None if not np.isfinite(sub).all() else round(float(sub.max()), 5)


def nan_to_none(x):
    if isinstance(x, np.ndarray):
        return [nan_to_none(v) for v in x]
    if isinstance(x, (list, tuple)):
        return [nan_to_none(v) for v in x]
    return None if x is None or not np.isfinite(x) else round(float(x), 5)


def compute(suite: dict, runs_by_bot: dict[str, list[dict]], min_tag_turns=5, min_target_pairs=3, margin=0.02,
            joint_margin=0.05, z=1.645, min_probes=2) -> dict:
    ids = sorted(b for b, runs in runs_by_bot.items() if runs)
    index = {b: i for i, b in enumerate(ids)}
    B, G = len(ids), len(TAGS)

    def stats():
        return {"s": np.zeros((B, B)), "ss": np.zeros((B, B)), "n": np.zeros((B, B))}

    by_count = defaultdict(lambda: {"sum": np.zeros((B, B)), "n": np.zeros((B, B)), "tag_sum": np.zeros((B, B, G)),
                                    "tag_n": np.zeros((B, B, G)), "noise_sum": np.zeros(B), "noise_n": np.zeros(B),
                                    "jnoise_sum": np.zeros(B), "jnoise_n": np.zeros(B), "probes": 0,
                                    "excess": stats(), "joint": stats(), "joint_excess": stats(),
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
        jdist = nanmean(np.where(tag_turns >= min_tag_turns, comp["joint"][0], np.nan), -1)
        acc = by_count[probe["teams"]]
        acc["probes"] += 1
        o = np.array([index[b] for b, _ in entities])
        same = o[:, None] == o[None]
        finite = np.isfinite(dist)
        e, f = np.nonzero(np.triu(same, 1) & finite)       # two runs of one bot: its A/A noise on this probe
        np.add.at(acc["noise_sum"], o[e], dist[e, f])
        np.add.at(acc["noise_n"], o[e], 1)
        e, f = np.nonzero(~same & finite)
        np.add.at(acc["sum"], (o[e], o[f]), dist[e, f])
        np.add.at(acc["n"], (o[e], o[f]), 1)
        # per-probe pair means and noise: the excess distances and their spread over probes (maps)
        pair, noise = probe_pairs(o, dist, B)
        jpair, jnoise = probe_pairs(o, jdist, B)
        ok = np.isfinite(jnoise)
        acc["jnoise_sum"][ok] += jnoise[ok]
        acc["jnoise_n"][ok] += 1
        for key, v in (("excess", excess(pair, noise)), ("joint", jpair), ("joint_excess", excess(jpair, jnoise))):
            ok = np.isfinite(v)
            acc[key]["s"][ok] += v[ok]
            acc[key]["ss"][ok] += v[ok] ** 2
            acc[key]["n"][ok] += 1
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
    per_count, noise_count, jnoise_count, clusters_count = {}, {}, {}, {}
    ex_count, ex_se_count, jex_count, joint_count, score_count, diam_count = {}, {}, {}, {}, {}, {}
    for n, acc in sorted(by_count.items()):
        d = np.divide(acc["sum"], acc["n"], out=np.full((B, B), np.nan), where=acc["n"] > 0)
        np.fill_diagonal(d, 0.0)
        per_count[n] = d
        noise_count[n] = {b: (acc["noise_sum"][i] / acc["noise_n"][i] if acc["noise_n"][i] else None) for i, b in enumerate(ids)}
        jnoise_count[n] = {b: (acc["jnoise_sum"][i] / acc["jnoise_n"][i] if acc["jnoise_n"][i] else None) for i, b in enumerate(ids)}
        ex, se, up = mean_upper(**acc["excess"], z=z, min_probes=min_probes)
        jex, _, jup = mean_upper(**acc["joint_excess"], z=z, min_probes=min_probes)
        joint, _, _ = mean_upper(**acc["joint"], z=z, min_probes=min_probes)
        score = np.maximum(up / margin, jup / joint_margin)        # <= 1: equivalent at this count
        for m in (ex, se, jex, joint, score):
            np.fill_diagonal(m, 0.0)
        ex_count[n], ex_se_count[n], jex_count[n], joint_count[n], score_count[n] = ex, se, jex, joint, score
        clusters_count[n] = complete_clusters(ids, score)
        diam_count[n] = [diameter(g, index, score) for g in clusters_count[n]]
    stack = np.stack(list(per_count.values())) if per_count else np.full((1, B, B), np.nan)
    overall = nanmean(stack, 0)
    # equivalent over all counts only if equivalent at every count both were probed at (np.max keeps NaN: unknown)
    worst = np.max(np.stack(list(score_count.values())), 0) if score_count else np.full((B, B), np.nan)
    noise = {b: (float(np.mean([v for v in (noise_count[n][b] for n in per_count) if v is not None]))
                 if any(noise_count[n][b] is not None for n in per_count) else None) for b in ids}
    clusters = complete_clusters(ids, worst)
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
    # counts at which a bot is equivalent to no other bot, and its nearest bot per count (by excess)
    distinct_counts, nearest_per_count = {b: [] for b in ids}, {b: {} for b in ids}
    for n, cl in clusters_count.items():
        for g in cl:
            if len(g) == 1:
                distinct_counts[g[0]].append(n)
        for i, b in enumerate(ids):
            row = np.where(np.arange(B) == i, np.nan, ex_count[n][i])
            if np.isfinite(row).any():
                j = int(np.nanargmin(row))
                nearest_per_count[b][str(n)] = {"id": ids[j], "excess": round(float(row[j]), 5)}
    comps = {}
    for name in WEIGHTS:
        s = sum(acc["comp"][name][0] for acc in by_count.values()) if by_count else np.zeros((B, B))
        c = sum(acc["comp"][name][1] for acc in by_count.values()) if by_count else np.zeros((B, B))
        comps[name] = nan_to_none(np.divide(s, c, out=np.full((B, B), np.nan), where=c > 0))
    return {"schema": SCHEMA_DISTANCE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {"weights": WEIGHTS, "min_tag_turns": min_tag_turns, "min_target_pairs": min_target_pairs,
                       "margin": margin, "joint_margin": joint_margin, "z": z, "min_probes": min_probes,
                       "stochastic_noise": STOCHASTIC_NOISE, "tags": list(TAGS)},
            "ids": ids, "overall": nan_to_none(overall), "per_count": {str(n): nan_to_none(d) for n, d in per_count.items()},
            "probes_per_count": {str(n): acc["probes"] for n, acc in sorted(by_count.items())},
            "excess_per_count": {str(n): nan_to_none(m) for n, m in ex_count.items()},
            "excess_se_per_count": {str(n): nan_to_none(m) for n, m in ex_se_count.items()},
            "joint_per_count": {str(n): nan_to_none(m) for n, m in joint_count.items()},
            "joint_excess_per_count": {str(n): nan_to_none(m) for n, m in jex_count.items()},
            "equivalence_score_per_count": {str(n): nan_to_none(m) for n, m in score_count.items()},
            "equivalence_score": nan_to_none(worst),
            "components": comps, "noise": noise,
            "noise_per_count": {str(n): {b: (None if v is None else round(v, 5)) for b, v in d.items()} for n, d in noise_count.items()},
            "joint_noise_per_count": {str(n): {b: (None if v is None else round(v, 5)) for b, v in d.items()}
                                      for n, d in jnoise_count.items()},
            "stochastic": sorted(b for b, v in noise.items() if v is not None and v > STOCHASTIC_NOISE),
            "clusters": clusters, "cluster_of": cluster_of,
            "cluster_diameter": [diameter(g, index, worst) for g in clusters],
            "clusters_per_count": {str(n): c for n, c in clusters_count.items()},
            "cluster_diameter_per_count": {str(n): d for n, d in diam_count.items()},
            "distinct_counts": distinct_counts, "nearest_per_count": nearest_per_count,
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
    ap.add_argument("--margin", type=float, default=0.02, help="largest equivalent excess distance (upper bound)")
    ap.add_argument("--joint-margin", type=float, default=0.05, help="largest equivalent joint-team excess (upper bound)")
    ap.add_argument("--z", type=float, default=1.645, help="standard errors over probes added to an excess")
    ap.add_argument("--min-probes", type=int, default=2, help="probes a count needs before a pair can be equivalent")
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "behavior_distance.json"))
    a = ap.parse_args(argv)
    suite = load_json(a.suite)
    runs_by_bot = load_all(Path(a.fingerprints))
    if a.bots:
        runs_by_bot = {b: runs_by_bot.get(b, []) for b in a.bots}
    result = compute(suite, runs_by_bot, a.min_tag_turns, a.min_target_pairs, a.margin, a.joint_margin, a.z, a.min_probes)
    save_json(a.out, result, indent=None)
    print(f"[distance] {len(result['ids'])} bots, {len(result['clusters'])} behavior groups over all counts, "
          f"{len(result['stochastic'])} stochastic -> {a.out}")


if __name__ == "__main__":
    main()
