"""Score the fixed-work trees with every critic backend and compare them (task T0061, docs/MEOWMIND_RL_ALIGNED.md).

    python bots/meow_rl/aligned/diagnose.py --trees data/meow_rl_aligned/trees --corpus data/meow_rl_aligned/corpus \\
        --out reports/host/meow_rl_aligned/diagnostics

Offline analysis, not a deployable timing result: every leaf of every frozen tree (bots/meow_rl/aligned/trees.py) is
scored by
  F   the checkpoint's FP32 critic (float32 arithmetic of inference/mlp_float.hpp, the reference here),
  R   reference16, H  wideact16 (the deployed arms' critics, bots/meow_rl/critics.py, the corpus's calibration), and
  R0  reference16_t0056 (diagnostic only: the T0056/T0057 critic export rule on the same corpus),
plus the float64 forward pass of the same weights. Each candidate's value is the scenario mean of its leaves (a
terminal leaf counts its reward, identical for every backend); the choice is the highest mean, ties to the actor's
order, exactly as search_v2.hpp. The trees are identical for every backend, so every difference is the critic's.
When the experiment's critic headers exist they must have the digests rebuilt here, and with a compiler their native
values must equal these on every leaf (scalar and SIMD). Reports, per backend against F: leaf value MAE/RMSE/max/bias,
top-action agreement, discordant candidate pairs, F's regret of the backend's choice (F's value of F's choice minus
F's value of the backend's choice), all by F's top-two margin and by team count, with 95% intervals from a bootstrap
over games (roots of one game are not independent). Writes DIAGNOSTICS.json, DIAGNOSTICS.md and ROOTS_SCORED.jsonl
(one line per root: its tree hash, candidates, every backend's candidate values and choice, and F's top-two margin),
so any disagreement can be inspected root by root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "meow_rl"))
sys.path.insert(0, str(REPO / "bots" / "meow_rl" / "aligned"))

import critics as C  # noqa: E402
import export as X  # noqa: E402
from fusion import runner  # noqa: E402

BACKENDS = {"F": "fp32", "R": "reference16", "H": "wideact16", "R0": "reference16_t0056"}
MARGINS = (0.0, 0.01, 0.03, 0.1, float("inf"))
BOOT = 2000


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(q: list[float]) -> int:
    best = 0
    for i, v in enumerate(q):
        if v > q[best]:
            best = i
    return best


def root_values(tree: dict, values: np.ndarray) -> list[float]:
    sums = [0.0] * len(tree["roots"])
    for leaf in tree["leaves"]:
        sums[leaf["root"]] += leaf["reward"] if leaf["terminal"] else float(values[leaf["row"]])
    return [s / tree["samples"] for s in sums]


def discordant(qa: list[float], qb: list[float]) -> float | None:
    n = len(qa)
    if n < 2:
        return None
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return sum(1 for i, j in pairs if np.sign(qa[i] - qa[j]) != np.sign(qb[i] - qb[j])) / len(pairs)


def boot(values: np.ndarray, groups: np.ndarray, seed: int = 20261015) -> list[float]:
    """95% percentile interval of the mean, resampling games (groups) with replacement."""
    uniq = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(BOOT):
        pick = rng.choice(uniq, len(uniq))
        rows = np.concatenate([idx[g] for g in pick])
        means.append(values[rows].mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summary(values: np.ndarray, groups: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()), "ci95_games": boot(values, groups)}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--config", default=str(X.CONFIG))
    ap.add_argument("--critic-npz", help="critic arrays (critic.layers.<i>.*) instead of the search source's checkpoint: plumbing only")
    ap.add_argument("--headers", default=str(REPO / "inference" / "meow_rl" / "aligned"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-native", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    td = Path(a.trees)
    man = json.loads((td / "TREES.json").read_text(encoding="utf-8"))
    for f, k in (("trees.jsonl", "trees_sha256"), ("leaves.f32", "leaves_sha256")):
        if sha256(td / f) != man[k]:
            raise SystemExit(f"{td / f} differs from TREES.json: the trees are frozen")
    trees = [json.loads(x) for x in (td / "trees.jsonl").read_text(encoding="utf-8").splitlines()]
    leaves = np.fromfile(td / "leaves.f32", np.float32).reshape(-1, 416)
    cal, _, corpus = X.load_corpus(Path(a.corpus))
    src = cfg["actors"][cfg["search_source"]]
    if a.critic_npz:
        z = dict(np.load(a.critic_npz))
        state = {k[len("critic."):]: v for k, v in z.items() if k.startswith("critic.")}
        source = {"kind": "npz_plumbing", "sha256": sha256(Path(a.critic_npz)), "evidence": False}
    else:
        path = REPO / src["path"]
        if sha256(path) != src["sha256"]:
            raise SystemExit(f"{src['path']} differs from its pin")
        _, state, _ = C.E.load_checkpoint(path)
        source = {"kind": "checkpoint", "path": src["path"], "sha256": src["sha256"]}
    W, b = C.E.critic_layers(state)
    C.E.check_critic(W, b, src["arm"])
    backends = {k: C.build(f, W, b, cal["value_feats"], cfg["export"]["margin"]) for k, f in BACKENDS.items()}
    vals = {k: C.value(be, leaves) for k, be in backends.items()}
    f64 = C.E.critic_forward(W, b, leaves)
    report = {"schema": "ctf-meow-rl-aligned-diagnostics-1", "generator": "bots/meow_rl/aligned/diagnose.py",
              "offline": "fixed-work analysis; no timing, no deadline, not a deployable result", "trees": man,
              "corpus": corpus, "critic_source": source, "backends": {}, "native": {}, "leaf_error_vs_F": {},
              "root": {}, "by_margin": {}, "by_count": {}}
    for k, be in backends.items():
        report["backends"][k] = {"format": be["format"], "digest": be["digest"], "describe": be["describe"],
                                 "deployed": k != "R0", "calibration": be.get("calibration")}
    hdr = Path(a.headers) / cfg["search_source"]
    cxx = None if a.no_native else runner.find_compiler()
    tmp = Path(tempfile.mkdtemp(prefix="meow-rl-diag-"))
    for k in ("F", "R", "H"):
        h = hdr / f"critic_{BACKENDS[k]}.hpp"
        if not h.is_file():
            report["native"][k] = "no header"
            continue
        if f'kDigest = "{backends[k]["digest"]}"' not in h.read_text(encoding="utf-8"):
            raise SystemExit(f"{h} is not the backend rebuilt from the corpus (digest differs)")
        if cxx:
            report["native"][k] = X.native_critic(cxx, h, backends[k], leaves, tmp)
            if not all(v["bit_equal"] for v in report["native"][k].values()):
                raise SystemExit(f"native {k} differs from its emulation on the tree leaves")
    report["leaf_error_vs_F"]["F_vs_float64"] = {"max_abs": float(np.abs(vals["F"] - f64).max()) if len(f64) else 0.0}
    for k in ("R", "H", "R0"):
        e = vals[k] - vals["F"]
        report["leaf_error_vs_F"][k] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                        "max_abs": float(np.abs(e).max()), "bias": float(e.mean()), "leaves": int(len(e)),
                                        "saturated": C.saturation(backends[k], leaves)}
    groups = np.array([t["match"] for t in trees])
    counts = np.array([t["teams"] for t in trees])
    q = {k: [root_values(t, vals[k]) for t in trees] for k in BACKENDS}
    pick = {k: np.array([choose(x) for x in q[k]]) for k in BACKENDS}
    margin = np.array([sorted(x, reverse=True)[0] - sorted(x, reverse=True)[1] if len(x) > 1 else np.inf for x in q["F"]])
    report["root"]["trees"] = len(trees)
    report["root"]["reduced"] = int(sum(len(t["roots"]) < 4 for t in trees))
    report["root"]["single_candidate"] = int(sum(len(t["roots"]) < 2 for t in trees))
    report["root"]["critic_calls_per_backend"] = int(len(leaves))
    report["root"]["terminal_leaves"] = int(sum(lf["terminal"] for t in trees for lf in t["leaves"]))
    for k in BACKENDS:
        report["root"][f"{k}_overrides_actor"] = summary((pick[k] != 0).astype(float), groups)
    for k in ("R", "H", "R0"):
        agree = (pick[k] == pick["F"]).astype(float)
        disc = [discordant(a_, b_) for a_, b_ in zip(q[k], q["F"])]
        ok = np.array([d is not None for d in disc])
        regret = np.array([qf[pf] - qf[pk] for qf, pf, pk in zip(q["F"], pick["F"], pick[k])])
        report["root"][k] = {"top_action_agreement": summary(agree, groups),
                             "discordant_pairs": summary(np.array([d for d in disc if d is not None]), groups[ok]),
                             "regret_under_F": summary(regret, groups),
                             "regret_under_F_max": float(regret.max()) if len(regret) else 0.0}
        bins = {}
        for lo, hi in zip(MARGINS, MARGINS[1:]):
            m = (margin >= lo) & (margin < hi)
            bins[f"[{lo:g},{hi:g})"] = {"roots": int(m.sum()), "agreement": float(agree[m].mean()) if m.any() else None}
        report["by_margin"][k] = bins
        report["by_count"][k] = {str(c): float(agree[counts == c].mean()) for c in sorted(set(counts.tolist()))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ROOTS_SCORED.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, t in enumerate(trees):
            f.write(json.dumps({"match": t["match"], "turn": t["turn"], "teams": t["teams"], "hash": t["hash"],
                                "roots": t["roots"], "actor": t["actor"], "values": {k: q[k][i] for k in BACKENDS},
                                "choice": {k: int(pick[k][i]) for k in BACKENDS},
                                "margin_F": None if not np.isfinite(margin[i]) else float(margin[i])}) + "\n")
    report["roots_scored_sha256"] = sha256(out / "ROOTS_SCORED.jsonl")
    (out / "DIAGNOSTICS.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (out / "DIAGNOSTICS.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    return report


def fmt_ci(s: dict, pct: bool = True) -> str:
    if not s.get("n"):
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:+.4f}")
    return f"{f(s['mean'])} [{f(s['ci95_games'][0])}, {f(s['ci95_games'][1])}]"


def markdown(r: dict) -> str:
    t = r["trees"]
    L = ["# Fixed-work search diagnostics (T0061)", "",
         "Generated by `bots/meow_rl/aligned/diagnose.py` from the frozen trees and the corpus. Do not edit. Offline analysis: "
         "no deadline, no timing; every backend scores the identical leaves of the identical trees.", "",
         f"Trees {t['trees']} ({t['reduced_trees']} reduced: fewer than 4 feasible root candidates), leaves {t['leaves']}, "
         f"terminal leaves {r['root']['terminal_leaves']}; trees `{t['trees_sha256'][:16]}`, leaves `{t['leaves_sha256'][:16]}`. "
         f"Critic source: {r['critic_source']['kind']}" + ("" if r["critic_source"]["kind"] == "checkpoint" else " (plumbing, not evidence)") + ".", "",
         "## Leaf values against F (the checkpoint's FP32 critic)", "",
         "| Backend | MAE | RMSE | max | bias | saturated activations per layer |", "|---|---:|---:|---:|---:|---|"]
    for k in ("R", "H", "R0"):
        e = r["leaf_error_vs_F"][k]
        L.append(f"| {k} ({r['backends'][k]['format']}) | {e['mae']:.5f} | {e['rmse']:.5f} | {e['max_abs']:.4f} | {e['bias']:+.5f} | {e['saturated']} |")
    L += ["", f"F against the float64 forward pass of the same weights: max {r['leaf_error_vs_F']['F_vs_float64']['max_abs']:.3g}.", "",
          "## Choices against F", "",
          "95% intervals resample games. Regret is F's value of F's choice minus F's value of the backend's choice (2U-1 units).", "",
          "| Backend | top-action agreement | discordant candidate pairs | regret under F | max regret |", "|---|---|---|---|---:|"]
    for k in ("R", "H", "R0"):
        x = r["root"][k]
        L.append(f"| {k} | {fmt_ci(x['top_action_agreement'])} | {fmt_ci(x['discordant_pairs'])} | "
                 f"{fmt_ci(x['regret_under_F'], pct=False)} | {x['regret_under_F_max']:.4f} |")
    L += ["", "Search overrides the actor's own choice (root 0): " + "; ".join(
        f"{k} {fmt_ci(r['root'][f'{k}_overrides_actor'])}" for k in ("F", "R", "H", "R0")) + ".", "",
          "## Agreement with F by F's top-two margin (descriptive)", "",
          "| Margin | " + " | ".join(k for k in ("R", "H", "R0")) + " |", "|---|" + "---|" * 3]
    for b in r["by_margin"]["R"]:
        cells = []
        for k in ("R", "H", "R0"):
            v = r["by_margin"][k][b]
            cells.append(f"{100 * v['agreement']:.1f}% ({v['roots']})" if v["agreement"] is not None else "n/a")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    L += ["", "R0 is the T0056/T0057 critic export rule on the same corpus (never deployed here). This analysis measures the "
          "critics' arithmetic on fixed trees; it does not measure whether the trees forecast the real game (opponent "
          "model, simulator fidelity), which needs real-engine continuations.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    r = main()
    print((Path(sys.argv[sys.argv.index("--out") + 1]) / "DIAGNOSTICS.md").read_text())
