"""Behavior distances, A/A noise, evidence states and behavior groups from fingerprints (schema 2).

    python -m roster.distance --suite data/roster/probe_suite.json --fingerprints data/roster/fingerprints \\
        --inventory data/roster/bot_inventory.json --out data/roster/behavior_distance.json

Fingerprint runs are first validated (roster.fingerprint.load_valid): only
runs whose identity matches this suite and the inventory's source hashes are
evidence; the rest are reported as stale, never mixed in.

Instrument (reports/reviews/FINGERPRINT_EFFECTIVENESS_REVIEW.md, P0). On each probe,
a pair of bots is compared on a support fixed by the state, not by what
either bot did: the turns where every run of both bots replied validly (their
common valid intersection), and the units on the board. On that support,
three categorical channels per unit, with fixed weights:

  move           the canonical requested move                     0.5
  action         wait / pickup / drop / attack                    0.3
  action_target  (action, target id if attack else none)          0.2

plus a team-level joint channel (the ordered tuple of all active units'
move and action_target). For every channel the disagreement between two
bots' runs, minus half of each bot's own repeat disagreement on the same
support, estimates half the squared difference of their conditional
distributions, a kernel distance that is zero only when they match; a fixed
non-negative weighted sum keeps that meaning. Estimates stay signed (finite
samples can go below zero). A bot without two valid repeats on a probe has
unknown noise there, so its excess is unknown, not raw distance.

Values are taken per event tag (roster.common.TAGS) on tags with at least
--min-tag-opps jointly valid opportunities (unit-turns; turns for joint), and
a probe's summary is the mean over its tags. Probes on the same map are one
statistical cluster: per team count, probe values are averaged per map, then
the spread over maps gives mean +- z SE. This is a heuristic bound over a few
maps, not a calibrated 95% guarantee.

Evidence states per pair and team count, with at least --min-probes maps:

  equivalent  upper bounds within --margin (weighted), --joint-margin (joint)
              and --context-margin in every adequately measured critical tag
              (every tag except quiet: a rare difference cannot be averaged
              away by quiet turns);
  different   some lower bound above its margin;
  unknown     anything else, including too few maps or unknown noise.

A bot is measured at a count when it has two or more valid repeats on at
least --min-probes maps there; groups (complete linkage: every pair inside
is equivalent) are formed among measured bots only, and the rest are listed
as unmeasured, never as distinct. Across counts a pair is grouped only when
equivalent at every count of the suite. `established_distinct_counts` lists
the counts at which a bot is demonstrably different from every other measured
bot; `coverage` gives, per bot and count, probes, maps, repeats, valid share
and the opportunities per tag. Summaries (`overall_excess`, `nearest`,
`distinctive_tags`, `tag_distance`) use the corrected excess; `overall` keeps
the raw disagreement for reference.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusion.paths import REPO
from roster.common import SCHEMA_DISTANCE, TAGS, load_json, save_json, suite_identity, tag_masks
from roster.fingerprint import load_valid

CHANNELS = ("move", "action", "action_target")
WEIGHTS = {"move": 0.5, "action": 0.3, "action_target": 0.2}
CRITICAL = tuple(t for t in TAGS if t != "quiet")
STOCHASTIC_NOISE = 0.1   # weighted A/A noise above this marks a bot as stochastic in the report
CHUNK = 16               # bots per block of the pairwise computation (memory bound)
STATE_NAMES = {1: "equivalent", -1: "different", 0: "unknown"}


def nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    """Mean over finite entries along `axis`; NaN where there are none (no warnings)."""
    ok = np.isfinite(x)
    n = ok.sum(axis)
    return np.divide(np.where(ok, x, 0.0).sum(axis), n, out=np.full(n.shape, np.nan), where=n > 0)


def bot_arrays(runs_per_bot: list[list[dict]]):
    """Padded run arrays for the bots of one probe: tokens (B, R, T, 3) per channel, run mask (B, R) and the
    common valid turns (B, T) over each bot's runs."""
    B, R = len(runs_per_bot), max(len(r) for r in runs_per_bot)
    T = len(runs_per_bot[0][0]["valid"])
    tok = {c: np.full((B, R, T, 3), -1, np.int64) for c in CHANNELS}
    mask = np.zeros((B, R), bool)
    valid = np.ones((B, T), bool)
    for b, runs in enumerate(runs_per_bot):
        for r, run in enumerate(runs):
            act = np.asarray(run["act"], np.int64)
            tok["move"][b, r] = run["move"]
            tok["action"][b, r] = act
            tok["action_target"][b, r] = act * 4096 + (np.asarray(run["target"], np.int64) + 1)
            mask[b, r] = True
            valid[b] &= np.asarray(run["valid"], bool)
    return tok, mask, valid


def own_noise(tok: dict, mask: np.ndarray, active: np.ndarray):
    """Per bot: disagreement between its own repeats per channel (B, T, 3) and joint (B, T); NaN below 2 repeats."""
    B, R = mask.shape
    out = {c: np.zeros(tok[c].shape[:1] + tok[c].shape[2:]) for c in CHANNELS}
    joint = np.zeros(out["move"].shape[:2])
    pairs = np.zeros(B)
    for r in range(R):
        for s in range(r + 1, R):
            w = (mask[:, r] & mask[:, s]).astype(float)
            pairs += w
            for c in CHANNELS:
                out[c] += w[:, None, None] * (tok[c][:, r] != tok[c][:, s])
            diff = (tok["move"][:, r] != tok["move"][:, s]) | (tok["action_target"][:, r] != tok["action_target"][:, s])
            joint += w[:, None] * (diff & active[None]).any(-1)
    den = np.where(pairs > 0, pairs, np.nan)
    return {c: out[c] / den[:, None, None] for c in CHANNELS}, joint / den[:, None]


def cross(tok: dict, mask: np.ndarray, active: np.ndarray, rows: np.ndarray):
    """Disagreement between bots `rows` and every bot, averaged over their run pairs: per channel (I, B, T, 3)
    and joint (I, B, T)."""
    B, R = mask.shape
    I = len(rows)
    out = {c: np.zeros((I, B) + tok[c].shape[2:], np.float32) for c in CHANNELS}
    joint = np.zeros((I, B, tok["move"].shape[2]), np.float32)
    for r in range(R):
        for s in range(R):
            w = (mask[rows, r][:, None] & mask[None, :, s]).astype(np.float32)
            if not w.any():
                continue
            diff = {c: tok[c][rows, r][:, None] != tok[c][None, :, s] for c in CHANNELS}
            for c in CHANNELS:
                out[c] += w[..., None, None] * diff[c]
            joint += w[..., None] * ((diff["move"] | diff["action_target"]) & active[None, None]).any(-1)
    n = (mask[rows].sum(1)[:, None] * mask.sum(1)[None]).astype(np.float32)
    return {c: out[c] / n[..., None, None] for c in CHANNELS}, joint / n[..., None]


def probe_pairs(runs_per_bot: list[list[dict]], tags: np.ndarray, active: np.ndarray, min_tag_opps: int) -> dict:
    """Every pairwise quantity of one probe among its bots (b x b, NaN on the diagonal)."""
    tok, mask, valid = bot_arrays(runs_per_bot)
    B = len(runs_per_bot)
    G = tags.shape[0]
    tagf = tags.T.astype(np.float32)                                  # (T, G)
    noise, jnoise = own_noise(tok, mask, active)
    any_active = active.any(-1)
    out = {k: np.full((B, B), np.nan) for k in ("excess", "raw", "joint_excess", "joint_raw")}
    out |= {f"excess_{c}": np.full((B, B), np.nan) for c in CHANNELS}
    out["tag_excess"] = np.full((B, B, G), np.nan)
    for start in range(0, B, CHUNK):
        rows = np.arange(start, min(B, start + CHUNK))
        d, dj = cross(tok, mask, active, rows)
        support = valid[rows][:, None, :, None] & valid[None, :, :, None] & active[None, None]
        jsupport = valid[rows][:, None] & valid[None] & any_active[None, None]
        opps = np.einsum("ibtk,tg->ibg", support.astype(np.float32), tagf)
        jopps = np.einsum("ibt,tg->ibg", jsupport.astype(np.float32), tagf)

        def per_tag(x, sup, den):
            num = np.einsum("ibtk,tg->ibg" if x.ndim == 4 else "ibt,tg->ibg", np.where(sup, x, 0.0), tagf)
            return np.where(den >= min_tag_opps, num / np.maximum(den, 1), np.nan)

        ex_total = np.zeros(d["move"].shape)
        raw_total = np.zeros(d["move"].shape)
        for c in CHANNELS:
            ex_c = d[c] - (noise[c][rows][:, None] + noise[c][None]) / 2       # NaN when either noise is unknown
            out[f"excess_{c}"][rows] = nanmean(per_tag(ex_c, support, opps), -1)
            ex_total = ex_total + WEIGHTS[c] * ex_c
            raw_total = raw_total + WEIGHTS[c] * d[c]
        tex = per_tag(ex_total, support, opps)
        out["tag_excess"][rows] = tex
        out["excess"][rows] = nanmean(tex, -1)
        out["raw"][rows] = nanmean(per_tag(raw_total, support, opps), -1)
        jex = dj - (jnoise[rows][:, None] + jnoise[None]) / 2
        out["joint_excess"][rows] = nanmean(per_tag(jex, jsupport, jopps), -1)
        out["joint_raw"][rows] = nanmean(per_tag(dj, jsupport, jopps), -1)
    idx = np.arange(B)
    for v in out.values():
        v[idx, idx] = np.nan
    own = (valid[:, :, None] & active[None]).astype(float)              # each bot's own valid opportunities
    w_noise = sum(WEIGHTS[c] * noise[c] for c in CHANNELS)
    total = own.sum((1, 2))
    out["noise"] = np.divide((np.nan_to_num(w_noise) * own).sum((1, 2)), total, out=np.full(B, np.nan),
                             where=(total > 0) & (mask.sum(1) >= 2))
    out["repeats"] = mask.sum(1)
    out["valid_share"] = valid.mean(1)
    out["own_opps"] = np.einsum("btk,tg->bg", own, tagf)
    return out


def map_stats(values: list[np.ndarray], z: float, min_maps: int):
    """values: one array per map (NaN = not measured). (mean, se, upper, lower, maps); bounds NaN below min_maps."""
    x = np.stack(values)
    n = np.isfinite(x).sum(0)
    mean = nanmean(x, 0)
    dev = np.where(np.isfinite(x), x - np.nan_to_num(mean), 0.0)
    var = np.divide((dev ** 2).sum(0), n - 1, out=np.zeros(mean.shape), where=n > 1)
    se = np.sqrt(var / np.maximum(n, 1))
    enough = n >= max(1, min_maps)
    return mean, se, np.where(enough, mean + z * se, np.nan), np.where(enough, mean - z * se, np.nan), n


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


def compute(suite: dict, runs_by_bot: dict[str, list[dict]], min_tag_opps=5, margin=0.02, joint_margin=0.05,
            context_margin=0.05, z=1.645, min_probes=2, stale: dict | None = None) -> dict:
    ids = sorted(b for b, runs in runs_by_bot.items() if runs)
    index = {b: i for i, b in enumerate(ids)}
    B, G = len(ids), len(TAGS)
    crit = np.array([t in CRITICAL for t in TAGS])
    keys = ("excess", "raw", "joint_excess", "joint_raw", *(f"excess_{c}" for c in CHANNELS))

    def new_map():
        return {"sum": {k: np.zeros((B, B)) for k in keys}, "n": {k: np.zeros((B, B)) for k in keys},
                "tag_sum": np.zeros((B, B, G)), "tag_n": np.zeros((B, B, G))}

    def new_cov():
        return {"probes": 0, "maps": set(), "repeats": [], "valid_share": [], "noise": [], "opps": np.zeros(G)}

    maps = defaultdict(lambda: defaultdict(new_map))      # count -> map -> sums (probes on one map are one cluster)
    cov = defaultdict(lambda: defaultdict(new_cov))
    coverage_turns = {}
    for probe in suite["probes"]:
        here = [b for b in ids if any(r["probe_id"] == probe["probe_id"] for r in runs_by_bot[b])]
        if not here:
            continue
        tags = tag_masks(probe["tags"])
        coverage_turns[probe["probe_id"]] = {t: int(n) for t, n in zip(TAGS, tags.sum(1))}
        active = ((np.array(probe["active"])[:, None] >> np.arange(3)) & 1).astype(bool)
        res = probe_pairs([[r for r in runs_by_bot[b] if r["probe_id"] == probe["probe_id"]] for b in here], tags, active,
                          min_tag_opps)
        o = np.array([index[b] for b in here])
        n, key = probe["teams"], probe.get("map_seed", probe["probe_id"])
        acc = maps[n][key]
        for k in keys:
            ok = np.isfinite(res[k])
            acc["sum"][k][np.ix_(o, o)] += np.where(ok, res[k], 0.0)
            acc["n"][k][np.ix_(o, o)] += ok
        ok = np.isfinite(res["tag_excess"])
        acc["tag_sum"][np.ix_(o, o)] += np.where(ok, res["tag_excess"], 0.0)
        acc["tag_n"][np.ix_(o, o)] += ok
        for j, b in enumerate(here):
            c = cov[b][n]
            c["probes"] += 1
            c["repeats"].append(int(res["repeats"][j]))
            c["valid_share"].append(float(res["valid_share"][j]))
            c["opps"] += res["own_opps"][j]
            if np.isfinite(res["noise"][j]):
                c["noise"].append(float(res["noise"][j]))
                c["maps"].add(key)
    counts = sorted(maps)
    per = {}
    for n in counts:
        mvals = list(maps[n].values())

        def mean_of(s, c):
            return np.divide(s, c, out=np.full(s.shape, np.nan), where=c > 0)

        st = {k: map_stats([mean_of(m["sum"][k], m["n"][k]) for m in mvals], z, min_probes) for k in keys}
        tag = map_stats([mean_of(m["tag_sum"], m["tag_n"]) for m in mvals], z, min_probes)
        ex, jx = st["excess"], st["joint_excess"]
        ctx_up = np.where(crit, tag[2], np.nan)                      # adequately measured critical tags only
        ctx_score = np.max(np.where(np.isfinite(ctx_up), ctx_up / context_margin, -np.inf), -1)
        score = np.maximum(np.maximum(ex[2] / margin, jx[2] / joint_margin), ctx_score)
        ctx_low = np.where(crit, tag[3], np.nan)
        different = (ex[3] > margin) | (jx[3] > joint_margin) | (np.nan_to_num(ctx_low, nan=-np.inf) > context_margin).any(-1)
        state = np.where(score <= 1.0, 1.0, np.where(different, -1.0, 0.0))
        state[~(st["raw"][4] > 0)] = np.nan                         # never compared at this count
        np.fill_diagonal(score, 0.0)
        np.fill_diagonal(state, np.nan)
        measured = [b for b in ids if n in cov[b] and len(cov[b][n]["maps"]) >= min_probes]
        per[n] = {"st": st, "tag": tag, "score": score, "state": state, "measured": measured}
    clusters_count, diam_count, unmeasured = {}, {}, {}
    distinct, unknown_pairs = {b: [] for b in ids}, {b: {} for b in ids}
    for n, p in per.items():
        k = [index[b] for b in p["measured"]]
        local = {b: i for i, b in enumerate(p["measured"])}
        sub = p["score"][np.ix_(k, k)]
        clusters_count[n] = complete_clusters(p["measured"], sub) if k else []
        diam_count[n] = [diameter(g, local, sub) for g in clusters_count[n]]
        unmeasured[n] = [b for b in ids if b not in local]
        for i, b in zip(k, p["measured"]):
            row = p["state"][i, [j for j in k if j != i]]
            if len(row) and np.all(row == -1):
                distinct[b].append(n)
            unknown_pairs[b][str(n)] = int(np.sum((row != 1) & (row != -1)))
    everywhere = [b for b in ids if all(b in per[n]["measured"] for n in counts)] if counts else []
    clusters, cdiam = [], []
    if everywhere:
        k = [index[b] for b in everywhere]
        worst = np.max(np.stack([per[n]["score"][np.ix_(k, k)] for n in counts]), 0)
        clusters = complete_clusters(everywhere, worst)
        cdiam = [diameter(g, {b: i for i, b in enumerate(everywhere)}, worst) for g in clusters]
    cluster_of = {b: f"behavior:{g[0]}" for g in clusters for b in g}
    cluster_of |= {b: f"unmeasured:{b}" for b in ids if b not in cluster_of}

    def over_counts(get, shape):
        return nanmean(np.stack([get(per[n]) for n in counts]), 0) if counts else np.full(shape, np.nan)

    overall = over_counts(lambda p: p["st"]["raw"][0], (B, B))
    overall_excess = over_counts(lambda p: p["st"]["excess"][0], (B, B))
    per_bot_tag = nanmean(over_counts(lambda p: p["tag"][0], (B, B, G)), 1)   # (B, G): mean excess to the others
    population = nanmean(per_bot_tag, 0)
    distinctive, nearest, nearest_count = {}, {}, {b: {} for b in ids}
    for i, b in enumerate(ids):
        rel = per_bot_tag[i] / np.where(population > 0, population, np.nan)
        order = [g for g in np.argsort(-np.nan_to_num(rel, nan=-np.inf)) if np.isfinite(rel[g])][:3]
        distinctive[b] = [{"tag": TAGS[g], "excess_to_others": round(float(per_bot_tag[i, g]), 4),
                           "vs_population": round(float(rel[g]), 3)} for g in order]
        row = np.where(np.arange(B) == i, np.nan, overall_excess[i])
        j = int(np.nanargmin(row)) if np.isfinite(row).any() else None
        nearest[b] = None if j is None else {"id": ids[j], "excess": round(float(row[j]), 5)}
        for n in counts:
            r = np.where(np.arange(B) == i, np.nan, per[n]["st"]["excess"][0][i])
            if np.isfinite(r).any():
                j = int(np.nanargmin(r))
                s = per[n]["state"][i, j]
                nearest_count[b][str(n)] = {"id": ids[j], "excess": round(float(r[j]), 5),
                                            "state": STATE_NAMES[int(s)] if np.isfinite(s) else "unknown"}
    coverage = {b: {str(n): {"probes": c["probes"], "maps": len(c["maps"]), "repeats_min": min(c["repeats"]),
                             "valid_share": round(float(np.mean(c["valid_share"])), 4),
                             "noise": round(float(np.mean(c["noise"])), 5) if c["noise"] else None,
                             "measured": len(c["maps"]) >= min_probes,
                             "opportunities": {t: int(v) for t, v in zip(TAGS, c["opps"])}}
                    for n, c in sorted(cov[b].items())} for b in ids}
    noise = {b: (round(float(np.mean([x for c in cov[b].values() for x in c["noise"]])), 5)
                 if any(c["noise"] for c in cov[b].values()) else None) for b in ids}

    def per_count(get):
        return {str(n): nan_to_none(get(per[n])) for n in counts}

    return {"schema": SCHEMA_DISTANCE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "suite_identity": suite_identity(suite),
            "params": {"channels": list(CHANNELS), "weights": WEIGHTS, "min_tag_opps": min_tag_opps, "margin": margin,
                       "joint_margin": joint_margin, "context_margin": context_margin, "critical_tags": list(CRITICAL),
                       "z": z, "min_probes": min_probes, "stochastic_noise": STOCHASTIC_NOISE, "tags": list(TAGS),
                       "bound": "mean + z * SE over maps (a heuristic over few maps, not a calibrated guarantee)"},
            "ids": ids, "overall": nan_to_none(overall), "overall_excess": nan_to_none(overall_excess),
            "maps_per_count": {str(n): len(maps[n]) for n in counts},
            "per_count": per_count(lambda p: p["st"]["raw"][0]),
            "excess_per_count": per_count(lambda p: p["st"]["excess"][0]),
            "excess_se_per_count": per_count(lambda p: p["st"]["excess"][1]),
            "joint_per_count": per_count(lambda p: p["st"]["joint_raw"][0]),
            "joint_excess_per_count": per_count(lambda p: p["st"]["joint_excess"][0]),
            "components": {c: per_count(lambda p, c=c: p["st"][f"excess_{c}"][0]) for c in CHANNELS},
            "equivalence_score_per_count": per_count(lambda p: p["score"]),
            "state_per_count": per_count(lambda p: p["state"]),
            "state_codes": {"1": "equivalent", "-1": "different", "0": "unknown", "null": "not compared"},
            "noise": noise, "stochastic": sorted(b for b, v in noise.items() if v is not None and v > STOCHASTIC_NOISE),
            "clusters": clusters, "cluster_of": cluster_of, "cluster_diameter": cdiam,
            "clusters_per_count": {str(n): c for n, c in clusters_count.items()},
            "cluster_diameter_per_count": {str(n): d for n, d in diam_count.items()},
            "unmeasured_per_count": {str(n): u for n, u in unmeasured.items()},
            "measured_everywhere": everywhere,
            "established_distinct_counts": distinct, "unknown_pairs_per_count": unknown_pairs,
            "nearest_per_count": nearest_count, "nearest": nearest, "distinctive_tags": distinctive,
            "tag_distance": {b: {t: nan_to_none(float(per_bot_tag[i, g])) for g, t in enumerate(TAGS)} for i, b in enumerate(ids)},
            "coverage": coverage, "coverage_turns": coverage_turns, "stale_runs": stale or {}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--fingerprints", default=str(REPO / "data" / "roster" / "fingerprints"))
    ap.add_argument("--inventory", help="check every run's source hash against the inventory")
    ap.add_argument("--bots", nargs="*", help="bot ids (default: every bot with valid runs)")
    ap.add_argument("--min-tag-opps", type=int, default=5, help="jointly valid opportunities a tag needs on a probe")
    ap.add_argument("--margin", type=float, default=0.02, help="largest equivalent weighted excess (upper bound)")
    ap.add_argument("--joint-margin", type=float, default=0.05, help="largest equivalent joint-team excess (upper bound)")
    ap.add_argument("--context-margin", type=float, default=0.05, help="largest equivalent excess in any critical tag")
    ap.add_argument("--z", type=float, default=1.645, help="standard errors over maps added to an excess")
    ap.add_argument("--min-probes", type=int, default=2, help="maps a count needs before a pair can be equivalent")
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "behavior_distance.json"))
    a = ap.parse_args(argv)
    suite = load_json(a.suite)
    runs_by_bot, stale = load_valid(Path(a.fingerprints), suite, load_json(a.inventory) if a.inventory else None)
    if a.bots:
        runs_by_bot = {b: runs_by_bot.get(b, []) for b in a.bots}
    result = compute(suite, runs_by_bot, a.min_tag_opps, a.margin, a.joint_margin, a.context_margin, a.z, a.min_probes, stale)
    save_json(a.out, result, indent=None)
    print(f"[distance] {len(result['ids'])} bots, {len(result['measured_everywhere'])} measured at every count, "
          f"{len(result['clusters'])} behavior groups among them, {len(result['stochastic'])} stochastic, "
          f"{sum(len(v) for v in stale.values())} stale runs ignored -> {a.out}")


if __name__ == "__main__":
    main()
