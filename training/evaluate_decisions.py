"""Evaluate the exact decision rule an own-action scorer would deploy (task T0034, docs/RL_PLAN.md section 4).

    # 1. on validation: choose lambda (and the normalization) per model, freeze everything in a selection file
    python -m training.evaluate_decisions select --data data/decision/t0010_p2 --run linear=runs/decision_linear_s1 \\
        --run mlp=runs/decision_mlp_s1 --primary mlp --negligible 0.002 --out reports/models/decision/SELECTION.json
    # 2. on test, once: verifies every frozen identity before any test outcome is read
    python -m training.evaluate_decisions test --data data/decision/t0010_p2 \\
        --selection reports/models/decision/SELECTION.json --out reports/models/decision/test

The rule. At a root, among the choosable candidates (the incumbent, which is always choosable, and every other
candidate in the scorer's support), choose argmax S_norm(c) + lambda * g(c), where g(c) = Q(s, c) - Q(s, a0) is the
scorer's gain over the incumbent a0 (g(a0) = 0) and S_norm is the carrier's own score normalized per root:
  incumbent_prior  S = 1 for the incumbent and 0 otherwise (the carrier's scores were not logged): switch only
                   when lambda * g(c) > 1;
  zscore           the logged carrier scores, z-scored over the root's choosable candidates (needs every score).
Ties go to the incumbent, then to the smallest canonical key, so the rule is deterministic. lambda = 0 is the
incumbent. lambda and the normalization are chosen on validation and frozen with the model identities before
the test split is opened; test evaluates that same rule, never argmax Q.

The primary quantity. For every root whose incumbent has an outcome: Delta = mean outcome of the chosen candidate
minus that of the incumbent (0 where the rule keeps the incumbent). Unchanged and uncertain roots stay in the
denominator. Team counts get equal weight (each count's roots share 1/K), and the 95% interval is cluster-robust
over independent maps (training.evaluate_value.clustered_mean_ci). Candidates without an outcome cannot be
chosen in this offline evaluation; they are counted. Roots without an incumbent outcome are not evaluable and
are counted, never dropped silently.

Verdicts: benefit (interval above 0), harm (below 0), negligible (the whole interval inside the predeclared
+-margin), otherwise insufficient precision. An interval containing zero does not show that learning is
impossible. The primary comparison is predeclared in the selection; secondary comparisons use a Bonferroni
interval over their number.

Diagnostics, labeled as such: per count and band, changed share, coverage; headroom with independent repeat
subsets (select the best candidate on one subset, evaluate it on the other; unavailable below two repeats); and,
with --oracle-value, a V chosen on *realized* successors, which contain the opponents' actual turn-t replies and
so are unavailable at deployment: not evidence for any deployable predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

from training import decision_dataset as D

SELECTION_SCHEMA = "ctf-fusion-decision-selection-1"
NORMS = ("incumbent_prior", "zscore")
DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
BANDS = ((2, 2), (3, 5), (6, 10), (11, 15))
TIE = 1e-12


# ---------------------------------------------------------------- statistics

def count_weights(teams: np.ndarray) -> np.ndarray:
    """Equal weight per team count: a root of count k gets 1 / (K * n_k), so the weights sum to 1."""
    teams = np.asarray(teams)
    if len(teams) == 0:
        return np.zeros(0)
    keys, inv, cnt = np.unique(teams, return_inverse=True, return_counts=True)
    return 1.0 / (len(keys) * cnt[inv])


def cluster_ci(values: np.ndarray, w: np.ndarray, maps: np.ndarray, z: float = 1.96) -> dict:
    from training.evaluate_value import clustered_mean_ci  # noqa: PLC0415
    out = clustered_mean_ci(np.asarray(values, np.float64), np.asarray(w, np.float64), np.asarray(maps), z)
    if out.get("mean") is not None:
        out["roots"] = int(len(values))
        out["z"] = z
    return out


def verdict(ci: dict, margin: float) -> str:
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return "insufficient precision"
    if lo > 0:
        return "benefit" + (" (inside the negligible margin)" if hi < margin else "")
    if hi < 0:
        return "harm" + (" (inside the negligible margin)" if lo > -margin else "")
    if -margin < lo and hi < margin:
        return "negligible"
    return "insufficient precision"


def bonferroni_z(m: int, alpha: float = 0.05) -> float:
    return NormalDist().inv_cdf(1 - alpha / (2 * max(m, 1)))


# ---------------------------------------------------------------- the rule

def roots_of(d: dict):
    for r, (first, count) in enumerate(zip(d["root_first"], d["root_count"])):
        yield r, np.arange(first, first + count)


def mask_gains(d: dict, g: np.ndarray) -> np.ndarray:
    """Gains as the rule may use them: 0 at the incumbent, NaN outside the support, and NaN for every alternative
    of a root whose incumbent is itself outside the support (no comparison with it is representable)."""
    g = np.asarray(g, np.float64).copy()
    g[~d["cand_support"]] = np.nan
    bad_root = ~d["cand_support"][d["root_incumbent"]]
    g[bad_root[d["cand_root"]]] = np.nan
    g[d["root_incumbent"]] = 0.0
    return g


def choosable(d: dict) -> np.ndarray:
    """Candidates the offline rule may pick: the incumbent, and supported candidates with an outcome."""
    ok = d["cand_support"] & (d["cand_n"] > 0)
    ok[d["root_incumbent"]] = True
    return ok


def normalized_carrier_scores(d: dict, norm: str, ok: np.ndarray) -> np.ndarray:
    s = np.zeros(len(d["cand_root"]))
    if norm == "incumbent_prior":
        s[d["root_incumbent"]] = 1.0
        return s
    if norm != "zscore":
        raise SystemExit(f"unknown normalization {norm}")
    raw = d["cand_carrier_score"]
    for r, idx in roots_of(d):
        c = idx[ok[idx]]
        if not np.isfinite(raw[c]).all():
            raise SystemExit("zscore normalization needs the carrier's score of every choosable candidate (not logged here)")
        sd = raw[c].std()
        s[c] = (raw[c] - raw[c].mean()) / sd if sd > 0 else 0.0
    return s


def apply_rule(d: dict, gain: np.ndarray, lam: float, norm: str = "incumbent_prior") -> np.ndarray:
    """Global candidate index chosen at each root (deterministic ties: incumbent, then smallest canonical key)."""
    ok = choosable(d)
    gain = mask_gains(d, gain)
    g = np.where(np.isfinite(gain), gain, 0.0)
    g[d["root_incumbent"]] = 0.0
    score = normalized_carrier_scores(d, norm, ok) + lam * g
    out = np.empty(len(d["root_first"]), np.int64)
    for r, idx in roots_of(d):
        inc = int(d["root_incumbent"][r])
        c = idx[(ok[idx] & np.isfinite(gain[idx])) | (idx == inc)]
        best = score[c].max()
        top = c[score[c] >= best - TIE]
        out[r] = inc if inc in top else top[np.argsort(d["cand_key"][top], kind="stable")[0]]
    return out


def deltas(d: dict, choice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Delta per root, evaluable mask): chosen minus incumbent mean outcome; evaluable when the incumbent has one."""
    inc = d["root_incumbent"]
    ev = d["cand_n"][inc] > 0
    y = d["cand_y"]
    delta = np.where(choice == inc, 0.0, y[choice] - y[inc])
    return np.where(ev, delta, np.nan), ev


def summarize(d: dict, choice: np.ndarray, margin: float, z: float = 1.96) -> dict:
    delta, ev = deltas(d, choice)
    teams, maps = d["root_teams"][ev], d["root_map"][ev]
    w = count_weights(teams)
    ci = cluster_ci(delta[ev], w, maps, z)
    changed = choice[ev] != d["root_incumbent"][ev]
    ok = choosable(d)
    alternatives = np.array([int(ok[idx].sum()) - 1 for _, idx in roots_of(d)])
    out = {"primary": ci | {"verdict": verdict(ci, margin)}, "evaluable_roots": int(ev.sum()),
           "unevaluable_roots": int((~ev).sum()), "maps": int(len(np.unique(maps))),
           "changed_share": float(changed.mean()) if len(changed) else None, "changed_roots": int(changed.sum()),
           "changed_mean_delta": float(delta[ev][changed].mean()) if changed.any() else None,
           "roots_with_alternatives": int((alternatives[ev] > 0).sum()),
           "supported_without_outcome": int((d["cand_support"] & (d["cand_n"] == 0)).sum()),
           "per_count": {}, "per_band": {}}
    for t in np.unique(teams):
        m = teams == t
        out["per_count"][str(int(t))] = cluster_ci(delta[ev][m], np.full(m.sum(), 1.0 / m.sum()), maps[m], z)
    for lo, hi in BANDS:
        m = (teams >= lo) & (teams <= hi)
        if m.any():
            out["per_band"][f"{lo}-{hi}"] = cluster_ci(delta[ev][m], count_weights(teams[m]), maps[m], z)
    out["_delta"], out["_ev"] = delta, ev
    return out


def paired(d: dict, a: dict, b: dict, margin: float, z: float) -> dict:
    ev = a["_ev"] & b["_ev"]
    diff = a["_delta"][ev] - b["_delta"][ev]
    ci = cluster_ci(diff, count_weights(d["root_teams"][ev]), d["root_map"][ev], z)
    return ci | {"verdict": verdict(ci, margin)}


def headroom(d: dict, margin: float, z: float = 1.96) -> dict:
    """Best-candidate headroom with independent repeat subsets: select on even repeats, evaluate on odd ones."""
    ok = choosable(d)
    n = d["cand_n"]
    rows, teams, maps = [], [], []
    for r, idx in roots_of(d):
        c = idx[ok[idx]]
        if len(c) < 2 or (n[c] < 2).any():
            continue
        o = d["cand_outcome"][c]
        sel = np.nanmean(o[:, 0::2], axis=1)
        ev = np.nanmean(o[:, 1::2], axis=1)
        inc = int(np.nonzero(c == d["root_incumbent"][r])[0][0])
        best = int(np.argmax(sel))
        rows.append(ev[best] - ev[inc])
        teams.append(d["root_teams"][r])
        maps.append(d["root_map"][r])
    if len(np.unique(maps)) < 2:
        return {"status": "unavailable: fewer than two maps whose choosable candidates all have two or more repeats",
                "roots": len(rows)}
    ci = cluster_ci(np.array(rows), count_weights(np.array(teams)), np.array(maps), z)
    return {"status": "estimated with independent repeat subsets (select: even repeats, evaluate: odd repeats)",
            "roots": len(rows), "headroom": ci | {"verdict": verdict(ci, margin)}}


def oracle_value(d: dict, data: Path, split: str, run: Path, margin: float, z: float) -> dict:
    """DIAGNOSTIC ORACLE, unavailable at deployment: a V's argmax over realized successor states."""
    from fusion.models import value_columns  # noqa: PLC0415
    from fusion.schema import layout  # noqa: PLC0415
    from training.export_weights import load_value  # noqa: PLC0415
    path = Path(data) / split / "oracle.npz"
    if not path.exists():
        return {"status": "unavailable: dataset built without --successors"}
    L = layout()
    model, beta = load_value(Path(run), L)
    with np.load(path) as o:
        succ, status = o["succ_rec"], o["succ_status"]
    x = (value_columns(succ, L, model.family) / L.value_scale(model.family)).astype(np.float32)
    v = model.logits(x)[:, 0] * beta
    v = np.where(status == "ok", v, np.nan)
    gain = np.full(len(v), np.nan)
    for r, idx in roots_of(d):
        inc = d["root_incumbent"][r]
        if np.isfinite(v[inc]):
            gain[idx] = v[idx] - v[inc]
    choice = apply_rule(d, gain, 1e6)
    s = summarize(d, choice, margin, z)
    return {"status": "DIAGNOSTIC ORACLE: chooses on realized successors (actual opponent replies); unavailable at "
                      "deployment and not evidence for any deployable predictor", "run": str(run),
            "result": {k: v for k, v in s.items() if not k.startswith("_")}}


# ---------------------------------------------------------------- identities

def canonical_sha(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_identity(run: Path) -> dict:
    run = Path(run)
    return {"run": str(run), "model_sha256": D.sha256_file(run / "model.npz"),
            "config_sha256": D.sha256_file(run / "config.json")}


def dataset_identity(data: Path) -> dict:
    index = D.index_of(data)
    return {"path": str(data), "index_sha256": D.sha256_file(Path(data) / "index.json"),
            "splits": {s: e["sha256"] for s, e in index["splits"].items()}, "features": index["features"],
            "carrier": index["carrier"], "authoritative": index["authoritative"],
            "exploratory_reasons": index["exploratory_reasons"]}


def verify_selection(sel: dict, data: Path) -> None:
    body = {k: v for k, v in sel.items() if k != "selection_sha256"}
    if sel.get("schema") != SELECTION_SCHEMA or canonical_sha(body) != sel.get("selection_sha256"):
        raise SystemExit("selection file was changed after it was frozen (its own digest does not match)")
    now = dataset_identity(data)
    if now != sel["dataset"]:
        raise SystemExit("the dataset differs from the one the selection was frozen on")
    for name, m in sel["models"].items():
        cur = model_identity(Path(m["run"]))
        if cur["model_sha256"] != m["model_sha256"] or cur["config_sha256"] != m["config_sha256"]:
            raise SystemExit(f"model {name} changed after the selection was frozen")


# ---------------------------------------------------------------- commands

def model_gains(d: dict, run: Path) -> np.ndarray:
    from training.train_decision_ranker import load_scorer  # noqa: PLC0415
    scorer = load_scorer(Path(run), d["_index"])
    return scorer.gains(d)


def strip(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def select(a) -> dict:
    data = Path(a.data)
    d = D.load(data, "val")
    lambdas = sorted(set(a.lambdas))
    if 0.0 not in lambdas:
        raise SystemExit("the lambda grid must contain 0 (the incumbent)")
    runs = dict(x.split("=", 1) for x in a.run)
    if a.primary not in runs:
        raise SystemExit(f"--primary {a.primary} is not one of --run {sorted(runs)}")
    models = {}
    for name, run in sorted(runs.items()):
        g = model_gains(d, Path(run))
        grid = []
        for lam in lambdas:
            s = summarize(d, apply_rule(d, g, lam, a.norm), a.negligible)
            grid.append({"lambda": lam, "mean": s["primary"]["mean"], "changed_share": s["changed_share"]})
        best = max(grid, key=lambda r: (r["mean"] if r["mean"] is not None else -np.inf, -r["lambda"]))
        models[name] = model_identity(Path(run)) | {"lambda": best["lambda"], "norm": a.norm, "val_grid": grid}
    sel = {"schema": SELECTION_SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "dataset": dataset_identity(data), "norm": a.norm, "negligible_margin": a.negligible, "z": 1.96,
           "primary": f"{a.primary} - incumbent",
           "secondary": [f"{n} - incumbent" for n in sorted(models) if n != a.primary] +
                        [f"{a.primary} - {n}" for n in sorted(models) if n != a.primary],
           "multiple_comparisons": "Bonferroni over the secondary comparisons", "models": models}
    sel["selection_sha256"] = canonical_sha(sel)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(sel, indent=1), encoding="utf-8")
    print(f"[evaluate_decisions] selection frozen: {a.out} sha256 {sel['selection_sha256'][:16]}; "
          + ", ".join(f"{n} lambda {m['lambda']}" for n, m in models.items()))
    return sel


def evaluate(a) -> dict:
    data = Path(a.data)
    sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
    verify_selection(sel, data)     # every identity checked before any test outcome is read
    d = D.load(data, "test", test_unlocked=True)
    margin, z = sel["negligible_margin"], sel["z"]
    rules = {"incumbent": d["root_incumbent"].astype(np.int64)}
    for name, m in sel["models"].items():
        rules[name] = apply_rule(d, model_gains(d, Path(m["run"])), m["lambda"], m["norm"])
    res = {n: summarize(d, c, margin, z) for n, c in rules.items()}
    zs = bonferroni_z(len(sel["secondary"]))
    comps = {}
    for label in [sel["primary"], *sel["secondary"]]:
        x, y = (s.strip() for s in label.split(" - "))
        comps[label] = paired(d, res[x], res[y], margin, z if label == sel["primary"] else zs) | \
            {"role": "primary" if label == sel["primary"] else "secondary"}
    report = {"split": "test", "selection_sha256": sel["selection_sha256"], "dataset": sel["dataset"],
              "authoritative": sel["dataset"]["authoritative"], "comparisons": comps,
              "rules": {n: strip(s) for n, s in res.items()}, "headroom": headroom(d, margin, z)}
    if a.oracle_value:
        report["oracle"] = oracle_value(d, data, "test", Path(a.oracle_value), margin, z)
    write_report(Path(a.out), report)
    return report


def write_report(out: Path, report: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    f = lambda v: "-" if v is None else f"{v:+.4f}"  # noqa: E731
    lines = [f"# Own-action decision rule: {report['split']} split", ""]
    if not report["authoritative"]:
        lines += ["**Exploratory data:** " + "; ".join(report["dataset"]["exploratory_reasons"]) + ".", ""]
    lines += ["Delta = chosen minus incumbent mean continuation outcome, all evaluable roots, equal weight per team count, "
              "95% intervals clustered by map. No game was played to make this report; it is not a strength result.", "",
              "| Comparison | Role | Mean | Interval | Maps | Verdict |", "|---|---|---:|---|---:|---|"]
    for label, c in report["comparisons"].items():
        lines.append(f"| {label} | {c['role']} | {f(c.get('mean'))} | [{f(c.get('lo'))}, {f(c.get('hi'))}] | "
                     f"{c.get('clusters', 0)} | {c['verdict']} |")
    lines += ["", "| Rule | Evaluable / unevaluable roots | Changed share | Mean where changed |", "|---|---|---:|---:|"]
    for n, s in report["rules"].items():
        cs = s["changed_share"]
        lines.append(f"| {n} | {s['evaluable_roots']} / {s['unevaluable_roots']} | "
                     f"{'-' if cs is None else f'{cs:.3f}'} | {f(s['changed_mean_delta'])} |")
    h = report["headroom"]
    lines += ["", f"Headroom: {h['status']}" + (f"; {f(h['headroom']['mean'])} [{f(h['headroom']['lo'])}, "
                                                 f"{f(h['headroom']['hi'])}]" if "headroom" in h else "")]
    if "oracle" in report:
        lines += ["", f"Oracle: {report['oracle']['status']}"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="choose lambda per model on validation and freeze the selection")
    s.add_argument("--data", required=True)
    s.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR of an own-action scorer")
    s.add_argument("--primary", required=True, help="the predeclared primary model")
    s.add_argument("--norm", choices=NORMS, default="incumbent_prior")
    s.add_argument("--lambdas", type=float, nargs="*", default=list(DEFAULT_LAMBDAS))
    s.add_argument("--negligible", type=float, required=True, help="predeclared practically negligible |Delta|")
    s.add_argument("--out", required=True)
    t = sub.add_parser("test", help="evaluate the frozen rules on the test split (once)")
    t.add_argument("--data", required=True)
    t.add_argument("--selection", required=True)
    t.add_argument("--oracle-value", help="V run for the diagnostic realized-successor oracle (not deployable)")
    t.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        select(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
