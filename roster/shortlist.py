"""Shortlist the most behaviorally unique opponents for P/V training data, and draft their admission.

    python -m roster.shortlist --distance roster/report/results/behavior_distance.json \\
        --inventory roster/new_bots/inventory.json --out roster/report/selection
    python -m roster.registry build --admission roster/report/selection/admission_draft.json \\
        --assessments roster/report/results/bot_assessments.json --distance roster/report/results/behavior_distance.json \\
        --inventory roster/new_bots/inventory.json --out profiles/open_training_pool.json

Only bots measured at every team count (reliable replies, two valid repeats on
enough maps) are considered. Steps, all from the distance file:

1. Families. Bots whose corrected excess (overall_excess, clipped at 0) is
   within --family (0.05) of every other member form a family (complete
   linkage: no chains). Behavior groups split on any certified difference,
   even one rare context at one count; for training data, bots this close
   are the same source of states and labels.
2. Representative. Per family: a judge bot first, then an archived bot, then
   an independent new bot, then a fork; ties by the member farthest from no
   one (smallest worst distance to the others), then by id.
3. Uniqueness. A family's distance to its nearest other family (the minimum
   over member pairs). Tiers: core (>= --core, 0.10), distinct (>= --family),
   near (below; a family split from its neighbor by a small margin).
4. Count specialists. A non-representative member that is certified
   different from its representative at a team count with excess above
   --family there is kept for those counts only.

The admission draft admits core and distinct representatives at every
count, specialists at their counts, the judge bots as the field (the
official opponents; no strength screen exists yet), and random/noise bots as
augmentation (never P labels). Near families are listed, not admitted: add
them by hand if wanted. Contribution sentences are generated from the
evidence and marked as a draft: a person reviews and edits them, sets
reserved OOD lineages and confirms before training. Similarity is behavior on
replayed probes, not strength.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fusion.paths import REPO
from roster import sampling
from roster.common import load_json, save_json
from roster.distance import complete_clusters

AUGMENTATION = ("@repo/tools/noise_bot.py", "@repo/tools/random_bot.py")


def matrix(rows) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in r] for r in rows], float)


def tier_of_source(bot: str) -> int:
    if "judge/ctf/bots" in bot:
        return 0
    if not bot.startswith("@repo/"):
        return 1
    return 2 if "/independent/" in bot else 3


def short(bot: str) -> str:
    p = Path(bot)
    return p.name if p.parent.name in ("", ".") else f"{p.parent.name}/{p.name}"


def build(dist: dict, inventory: dict, family=0.05, core=0.10) -> dict:
    ids = dist["ids"]
    ix = {b: i for i, b in enumerate(ids)}
    lineage = {b["id"]: b.get("lineage_group") for b in inventory["bots"]}
    E = matrix(dist["overall_excess"])
    np.fill_diagonal(E, 0.0)
    meas = list(dist["measured_everywhere"])
    k = [ix[b] for b in meas]
    S = np.clip(E[np.ix_(k, k)], 0.0, None)
    fams = complete_clusters(meas, S / family)
    counts = sorted(dist["excess_per_count"], key=int)
    pc = {n: matrix(dist["excess_per_count"][n]) for n in counts}
    st = {n: matrix(dist["state_per_count"][n]) for n in counts}
    rows, specialists = [], []
    for f in fams:
        fk = [ix[b] for b in f]
        worst = {b: float(np.nanmax(np.clip(E[ix[b], fk], 0.0, None))) for b in f}
        rep = min(f, key=lambda b: (tier_of_source(b), worst[b], b))
        rows.append({"representative": rep, "members": sorted(f), "family_diameter": round(max(worst.values()), 5)})
        for m in f:
            if m == rep:
                continue
            cs = [int(n) for n in counts if st[n][ix[rep], ix[m]] == -1 and pc[n][ix[rep], ix[m]] > family]
            if cs:
                specialists.append({"bot": m, "representative": rep, "counts": cs,
                                    "excess_at_counts": {str(c): round(float(pc[str(c)][ix[rep], ix[m]]), 4) for c in cs}})
    F = np.full((len(rows), len(rows)), np.inf)
    for a, ra in enumerate(rows):
        ka = [ix[b] for b in ra["members"]]
        for b in range(a + 1, len(rows)):
            kb = [ix[x] for x in rows[b]["members"]]
            F[a, b] = F[b, a] = float(np.nanmin(np.clip(E[np.ix_(ka, kb)], 0.0, None)))
    for a, r in enumerate(rows):
        j = int(np.argmin(F[a]))
        rep = r["representative"]
        r |= {"nearest_family": rows[j]["representative"], "nearest_excess": round(float(F[a, j]), 5),
              "established_distinct_counts": dist["established_distinct_counts"].get(rep, []),
              "distinctive_tags": [t["tag"] for t in dist["distinctive_tags"].get(rep, [])],
              "lineage": lineage.get(rep), "noise": dist["noise"].get(rep),
              "role": ("augmentation" if rep in AUGMENTATION else "field" if "judge/ctf/bots" in rep else "style")}
        r["tier"] = "core" if r["nearest_excess"] >= core else "distinct" if r["nearest_excess"] >= family else "near"
    rows.sort(key=lambda r: -r["nearest_excess"])
    return {"params": {"family": family, "core": core, "distance_field": "overall_excess (clipped at 0)",
                       "suite_identity": dist.get("suite_identity"), "measured_everywhere": len(meas),
                       "unmeasured": sorted(set(ids) - set(meas))},
            "families": rows, "specialists": specialists}


def contribution(r: dict) -> str:
    d = len(r["established_distinct_counts"])
    s = (f"Draft from measured evidence: nearest other behavior family {short(r['nearest_family'])} at corrected excess "
         f"{r['nearest_excess']:.3f}; certified different from every measured bot at {d}/14 team counts")
    if r["distinctive_tags"]:
        s += f"; most distinctive tags: {', '.join(r['distinctive_tags'])}"
    if len(r["members"]) > 1:
        s += f"; represents {len(r['members'])} near-identical bots"
    return s + ". Review before training."


def admission(sel: dict, revision: str) -> dict:
    bots = {}
    for r in sel["families"]:
        if r["tier"] == "near" and r["role"] != "field":
            continue
        roles = ["augmentation"] if r["role"] == "augmentation" else ["style", "field"] if r["role"] == "field" else ["style"]
        bots[r["representative"]] = {"admit": True, "roles": roles, "counts": "all", "contribution": contribution(r),
                                     "max_copies": 1}
    for s in sel["specialists"]:
        if s["bot"] in bots:
            continue
        bots[s["bot"]] = {"admit": True, "roles": ["style"], "counts": s["counts"], "max_copies": 1,
                          "contribution": (f"Draft from measured evidence: count specialist of {short(s['representative'])}, "
                                           f"certified different at team counts {s['counts']} (excess "
                                           f"{', '.join(f'{c}: {v}' for c, v in s['excess_at_counts'].items())}). Review before training.")}
    return {"_comment": "DRAFT generated by roster.shortlist from the first host run. A person reviews the contribution "
                        "sentences, roles and counts, fills reserved_ood_lineages, and only then trains on the result.",
            "revision": revision, "reserved_ood_lineages": [], "sampling": dict(sampling.DEFAULTS), "bots": bots}


def report(sel: dict, adm: dict) -> str:
    fams = sel["families"]
    tiers = {t: [r for r in fams if r["tier"] == t] for t in ("core", "distinct", "near")}
    p = sel["params"]
    out = [
        "# Unique opponents for P/V training: shortlist from the first host run",
        "",
        "<!-- Generated by `python -m roster.shortlist`. Do not edit by hand. -->",
        "",
        f"Source: `roster/report/results/behavior_distance.json` (suite `{(p['suite_identity'] or '')[:12]}`), "
        f"{p['measured_everywhere']} bots measured at every team count; {len(p['unmeasured'])} unmeasured and left out "
        f"({', '.join(short(b) for b in p['unmeasured'])}).",
        "",
        f"**Method.** Bots within {p['family']} corrected excess of every other member form a family (complete linkage). "
        f"One representative per family (judge bot, then archived, then independent new, then fork). A family's uniqueness is "
        f"its distance to the nearest other family: **core** >= {p['core']}, **distinct** >= {p['family']}, **near** below. "
        "Count specialists are family members certified different from their representative at specific team counts. "
        "This is behavior on replayed probes, not strength: no strength screen has run yet.",
        "",
        f"**Result.** {len(fams)} families: {len(tiers['core'])} core, {len(tiers['distinct'])} distinct, {len(tiers['near'])} near; "
        f"{len(sel['specialists'])} count specialists. The admission draft admits "
        f"{sum(1 for b in adm['bots'].values())} bots: core and distinct representatives at every count, the judge bots as the field, "
        "random/noise bots as augmentation, specialists at their counts.",
        "",
    ]
    for t, title in (("core", "Core: most unique"), ("distinct", "Distinct"), ("near", "Near (listed, not admitted)")):
        out += [f"## {title} ({len(tiers[t])})", "",
                "| # | Representative | Role | Nearest family (excess) | Distinct at counts | Family size | Lineage |",
                "|---:|---|---|---|---:|---:|---|"]
        for i, r in enumerate(tiers[t], 1):
            out.append(f"| {i} | `{r['representative']}` | {r['role']} | `{short(r['nearest_family'])}` ({r['nearest_excess']:.3f}) | "
                       f"{len(r['established_distinct_counts'])}/14 | {len(r['members'])} | `{r['lineage']}` |")
        out.append("")
    out += ["## Count specialists", "", "| Bot | Family representative | Admitted at team counts | Excess there |", "|---|---|---|---|"]
    out += [f"| `{s['bot']}` | `{short(s['representative'])}` | {s['counts']} | "
            f"{', '.join(f'{c}: {v}' for c, v in s['excess_at_counts'].items())} |" for s in sel["specialists"]]
    out += ["", "## Families with more than one member", ""]
    out += [f"- `{short(r['representative'])}` ({len(r['members'])}, diameter {r['family_diameter']:.3f}): "
            + ", ".join(f"`{short(m)}`" for m in r["members"] if m != r["representative"]) for r in fams if len(r["members"]) > 1]
    out += ["", "## Next", "",
            "1. A person reviews `admission_draft.json`: edit contribution sentences, roles and counts; reserve whole OOD lineages.",
            "2. `python -m roster.registry build --admission roster/report/selection/admission_draft.json --assessments "
            "roster/report/results/bot_assessments.json --distance roster/report/results/behavior_distance.json --inventory "
            "roster/new_bots/inventory.json --out profiles/open_training_pool.json`.",
            "3. Collect P/V data from that pool (`training.collect --pool profiles/open_training_pool.json`).",
            "4. Run the strength screen on the admitted bots, then rebuild the pool with confirmed field members.", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", default=str(REPO / "roster" / "report" / "results" / "behavior_distance.json"))
    ap.add_argument("--inventory", default=str(REPO / "roster" / "new_bots" / "inventory.json"))
    ap.add_argument("--family", type=float, default=0.05, help="largest corrected excess inside one family")
    ap.add_argument("--core", type=float, default=0.10, help="nearest-family excess of a core family")
    ap.add_argument("--revision", default="r1-draft")
    ap.add_argument("--out", default=str(REPO / "roster" / "report" / "selection"))
    a = ap.parse_args(argv)
    sel = build(load_json(a.distance), load_json(a.inventory), a.family, a.core)
    adm = admission(sel, a.revision)
    out = Path(a.out)
    save_json(out / "shortlist.json", sel | {"created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    save_json(out / "admission_draft.json", adm)
    (out / "SHORTLIST.md").write_text(report(sel, adm), encoding="utf-8")
    tiers = {t: sum(r["tier"] == t for r in sel["families"]) for t in ("core", "distinct", "near")}
    print(f"[shortlist] {len(sel['families'])} families {tiers}, {len(sel['specialists'])} specialists, "
          f"{len(adm['bots'])} admitted in the draft -> {out}")


if __name__ == "__main__":
    main()
