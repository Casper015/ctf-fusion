"""Ranking and regret of a search arm's choices against the incumbent's, from branch continuations.

    python bots/p_only/tools/branch_regret.py --branches data/branches/v12_portfolio --arm v12_ponly_xxl \\
        --compare-arm v12_search_k3 --out bots/p_only/report/search_portfolio/regret

Input: bots/p_only/tools/branch_portfolio.py roots of the incumbent arm, with --compare-arm's reply at every root
as a candidate (source arm:<name>). A root is valid when the reply is a legal order whose canonical form equals the
incumbent's (identical) or is a played candidate of its own (different); malformed replies, timeouts and missing
candidates or branches are excluded and counted, with the valid-root denominator. Per valid root, every
candidate's mean continuation outcome gives:
  difference     outcome(compare arm's choice) - outcome(incumbent's recorded choice); 0 where both chose the
                 same complete order (then no separate candidate exists);
  regret         best candidate's outcome minus the choice's, for the incumbent and for the compare arm;
  rank           position of each choice among the root's candidates (1 = best).
Means and 95% intervals are clustered by map: a map's roots are one sample. An interval that includes zero is
"not shown". Continuation outcomes are single (or few) games each, so regret is an estimate with sampling noise,
and a candidate set is not the space of all orders: this measures differences against the incumbent, not
optimality. Games at equal full-turn time remain the deployment test.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from branch_portfolio import classify_compare  # noqa: E402
from training.evaluate_value import clustered_mean_ci  # noqa: E402


def root_stats(row: dict, compare_arm: str) -> tuple[str, dict | None]:
    """(status, statistics) of one root. Only `identical` (the compare arm's canonical order is the incumbent's) and
    `different` (its own candidate with branch outcomes) give statistics; every other status is an exclusion and is
    counted, never a zero difference. The status is recomputed from the raw reply, the candidate orders and the
    played branches (branch_portfolio.classify_compare), so a malformed or missing reply cannot pass as identical."""
    if row.get("compare_arm") != compare_arm:
        return "none", None
    means = {}
    for b in row["branches"]:
        means.setdefault(b["candidate"], []).append(b["outcome"])
    status, _, index = classify_compare(row.get("compare_reply"), row["turn"], row["candidates"], played=set(means))
    if status not in ("identical", "different"):
        return status, None
    y = {c: float(np.mean(v)) for c, v in means.items()}
    if 0 not in y:
        return "missing_branch", None
    best = max(y.values())
    rank = lambda c: 1 + sum(v > y[c] for v in y.values())  # noqa: E731
    yc = y[index]
    return status, {"map": row["map_seed"], "teams": row["teams"], "same": status == "identical", "difference": yc - y[0],
                    "regret_incumbent": best - y[0], "regret_compare": best - yc, "rank_incumbent": rank(0),
                    "rank_compare": rank(index), "candidates": len(y)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--compare-arm", required=True)
    ap.add_argument("--out", required=True, help="output path without extension (.json and .md)")
    a = ap.parse_args(argv)
    rows, statuses = [], {}
    for f in sorted(Path(a.branches).rglob("*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            row = json.load(fh)
        if row.get("arm") != a.arm or "dropped" in row:
            continue
        status, s = root_stats(row, a.compare_arm)
        statuses[status] = statuses.get(status, 0) + 1
        if s:
            rows.append(s)
    if not rows:
        raise SystemExit(f"no valid roots (identical or different valid comparison replies); statuses {statuses}")
    maps = np.array([r["map"] for r in rows])
    one = np.ones(len(rows))
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    differ = ~np.array([r["same"] for r in rows])
    report = {"arm": a.arm, "compare_arm": a.compare_arm, "roots": len(rows), "maps": int(len(np.unique(maps))),
              "root_statuses": statuses, "excluded": {k: v for k, v in statuses.items() if k not in ("identical", "different")},
              "differs_share": float(differ.mean()),
              "difference": clustered_mean_ci(col("difference"), one, maps),
              "difference_where_differs": clustered_mean_ci(col("difference")[differ], one[differ], maps[differ]) if differ.any() else None,
              "regret_incumbent": clustered_mean_ci(col("regret_incumbent"), one, maps),
              "regret_compare": clustered_mean_ci(col("regret_compare"), one, maps),
              "regret_difference": clustered_mean_ci(col("regret_compare") - col("regret_incumbent"), one, maps),
              "rank_incumbent_mean": float(col("rank_incumbent").mean()), "rank_compare_mean": float(col("rank_compare").mean()),
              "by_teams": {str(t): clustered_mean_ci(col("difference")[col("teams") == t], one[col("teams") == t], maps[col("teams") == t])
                           for t in sorted(set(int(x) for x in col("teams")))}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    f = lambda s: "-" if not s or s["mean"] is None else f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] {s['verdict']}"  # noqa: E731
    lines = [f"# Branch comparison: {a.compare_arm} against {a.arm}", "",
             f"{report['roots']} valid roots on {report['maps']} maps (excluded: {report['excluded'] or 'none'}; a malformed, timed-out "
             "or unplayed comparison reply is excluded, never counted as identical); the compare arm chose another complete "
             f"order at {report['differs_share']:.1%} of the valid roots. Outcomes are final pairwise utilities of real continuations; 95% intervals "
             "clustered by map; not a strength result.", "",
             "| Quantity | Mean [95% interval] |", "|---|---|",
             f"| outcome(compare) - outcome(incumbent), all valid roots | {f(report['difference'])} |",
             f"| the same, where the choices differ | {f(report['difference_where_differs'])} |",
             f"| incumbent regret against the best candidate | {f(report['regret_incumbent'])} |",
             f"| compare-arm regret | {f(report['regret_compare'])} |",
             f"| regret difference (compare - incumbent) | {f(report['regret_difference'])} |",
             f"| mean rank among candidates (incumbent / compare) | {report['rank_incumbent_mean']:.2f} / {report['rank_compare_mean']:.2f} |", "",
             "| Teams | outcome difference |", "|---|---|", *[f"| {t} | {f(s)} |" for t, s in report["by_teams"].items()]]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
