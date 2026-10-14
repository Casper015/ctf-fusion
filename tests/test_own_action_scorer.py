"""Own-action scorer export and C++ float inference (task T0044).

Scorers here have fixed random weights (no training): a linear model and a small MLP of
training.train_decision_ranker.Scorer. Their exports must reproduce Python's Q(s, a) on recorded roots and requested
replies (tests/test_own_action_parity.py inputs), carry every binding, and be refused wherever a kind, identity, width,
selection or carrier contract is broken. Nothing here trains a model or says anything about scorer quality.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from fusion import matchlog, native
from fusion.paths import REPO
from fusion.protocol import parse_block
from fusion.schema import layout
from tests.test_own_action_parity import ROSTER, categories, requests
from training import decision_dataset as D
from training import evaluate_decisions as E
from training import export_decision_scorer as X
from training import train_decision_ranker as T
from training.export_weights import run_config

CARRIER = {"binary_sha256": ["c" * 64],
           "config": {"path": "build/decision/carrier.json", "sha256": "d" * 64,
                      "config": {"schema": D.CARRIER_CONFIG_SCHEMA, "status": "admitted", "variant": "v12_ponly_xxl_dlog",
                                 "opponent_policy": {"family": "p2", "models": "inference/models_xxl", "files": {}},
                                 "value": {"enabled": False, "tier": None}}}}


def make_run(root: Path, model: str, hidden=(), seed=0, carrier=None, scale=1.0) -> Path:
    L = layout()
    s = T.Scorer(model, F.root_dim(L), F.CAND_DIM, hidden, np.random.default_rng(seed))
    for w in s.net.W:
        w *= scale
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"kind": F.KIND, "model": model, "hidden": list(hidden), "features": F.feature_identity(L),
           "root_dim": F.root_dim(L), "cand_dim": F.CAND_DIM, "carrier": carrier or {"binary_sha256": ["c" * 64], "config": None},
           "dataset": {"path": "fixture"}}
    (root / "config.json").write_text(json.dumps(cfg))
    np.savez(root / "model.npz", **s.state())
    return root


def export(run: Path, out: Path, *extra) -> dict:
    args = ["--run", str(run), "--out", str(out), *extra]
    if "--selection" not in extra:
        args += ["--lambda", "0.5", "--unselected"]
    return X.main(args)


def build_tool(out: Path) -> Path:
    return native.build(REPO / "tools" / "cpp" / "own_action_scorer_check.cpp", "own_action_scorer_check",
                        deps=[out / X.HEADER], includes=[out])


def run_tool(exe: Path, *args, stream: str | None = None) -> dict:
    r = subprocess.run([str(exe), *args], input=stream.encode() if stream else None, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{r.returncode}: {r.stderr.decode()}")
    return json.loads(r.stdout)


def q64(scorer: T.Scorer, xr: np.ndarray, xc: np.ndarray) -> np.ndarray:
    """Python's forward in float64 from the float32 weights (the reference of the C++ route)."""
    h = (xc if scorer.model == "linear" else np.concatenate([xr, xc], axis=1)).astype(np.float64)
    for i, (w, b) in enumerate(zip(scorer.net.W, scorer.net.b)):
        h = h @ w.astype(np.float64).T + b.astype(np.float64)
        if i < len(scorer.net.W) - 1:
            h = np.clip(h, 0.0, 1.0)
    return h[:, 0]


class Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.L = layout()
        cls.roots = []
        for t in (4, 15):
            m = next(matchlog.load(p) for p in sorted(ROSTER.rglob("*.json.gz")) if matchlog.load(p)["teams"] == t)
            for name, turn in categories(m, 0).items():
                rec = D.root_record(m, 0, turn)
                lines = requests(rec, m["replies"][turn - 1][0], turn, 0, m["teams"])[:60]
                cls.roots.append((m, turn, rec, lines))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def python_q(self, scorer, m, turn, rec, lines):
        size = int(m["init"].split("\n", 1)[0].split()[3])
        units = parse_block(m["blocks"][turn - 1]).units
        canon = [F.canonicalize(rec, self.L, ln, turn, 0, units, size) for ln in lines]
        xc = np.stack([F.candidate_features(rec, self.L, c["move"], c["act"], c["slot"]) for c in canon])
        xr = np.repeat(F.root_features(rec, self.L)[None], len(lines), axis=0)
        return q64(scorer, xr, xc), scorer.q(xr, xc)

    def check_parity(self, model, hidden, scale=1.0):
        run = make_run(self.tmp / f"run_{model}", model, hidden, seed=3, scale=scale)
        out = self.tmp / f"out_{model}"
        export(run, out, "--allow-unbound-carrier")
        exe = build_tool(out)
        info = run_tool(exe, "--model")
        self.assertTrue(info["valid"], info["why"])
        scorer = T.load_scorer(run)
        worst64 = worst32 = 0.0
        for m, turn, rec, lines in self.roots:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
                fh.write("\n".join(lines) + "\n")
            try:
                cc = run_tool(exe, "0", str(turn), fh.name, stream=D.prefix_stream(m, 0, turn))
            finally:
                Path(fh.name).unlink()
            ref64, ref32 = self.python_q(scorer, m, turn, rec, lines)
            got = np.asarray(cc["q"])
            worst64 = max(worst64, float(np.max(np.abs(got - ref64))))
            worst32 = max(worst32, float(np.max(np.abs(got - ref32) / np.maximum(1.0, np.abs(ref64)))))
        self.assertLess(worst64, 1e-9, "C++ double accumulation vs numpy float64 of the same float32 weights")
        self.assertLess(worst32, 1e-4, "C++ vs the trainer's own float32 forward")
        return info

    def test_linear_parity(self):
        info = self.check_parity("linear", ())
        self.assertEqual((info["model"], info["layers"], info["carrier_status"], info["selection"]),
                         ("linear", 1, "unbound", "unselected"))

    def test_mlp_parity(self):
        info = self.check_parity("mlp", (64, 16), scale=3.0)       # scaled so that hidden units clip at 0 and 1
        self.assertEqual((info["model"], info["layers"]), ("mlp", 3))

    def test_selection_binding(self):
        run = make_run(self.tmp / "sel_run", "mlp", (8,), seed=4, carrier=CARRIER)
        ident = E.model_identity(run)
        sel = {"schema": E.SELECTION_SCHEMA, "norm": "zscore", "models": {"mlp": ident | {"lambda": 0.25, "norm": "zscore"}}}
        sel["selection_sha256"] = E.canonical_sha(sel)
        path = self.tmp / "SELECTION.json"
        path.write_text(json.dumps(sel))
        man = export(run, self.tmp / "sel_out", "--selection", str(path), "--model-name", "mlp")
        self.assertEqual((man["rule"]["lambda"], man["rule"]["norm"], man["rule"]["selection_sha256"]),
                         (0.25, "zscore", sel["selection_sha256"]))
        self.assertEqual((man["carrier"]["variant"], man["carrier"]["status"], man["carrier"]["p_family"], man["carrier"]["value"]),
                         ("v12_ponly_xxl_dlog", "admitted", "p2", 0))
        text = (self.tmp / "sel_out" / X.HEADER).read_text()
        self.assertIn('"v12_ponly_xxl_dlog"', text)
        self.assertIn(sel["selection_sha256"], text)
        tampered = dict(sel, norm="incumbent_override")
        path.write_text(json.dumps(tampered))
        with self.assertRaises(SystemExit):
            export(run, self.tmp / "sel_bad", "--selection", str(path), "--model-name", "mlp")
        other = make_run(self.tmp / "sel_other", "mlp", (8,), seed=5, carrier=CARRIER)
        path.write_text(json.dumps(sel))
        with self.assertRaises(SystemExit):
            export(other, self.tmp / "sel_other_out", "--selection", str(path), "--model-name", "mlp")

    def test_refusals(self):
        run = make_run(self.tmp / "ref_run", "linear", seed=6)
        with self.assertRaises(SystemExit):                       # no carrier configuration and no explicit test flag
            export(run, self.tmp / "ref_out")
        with self.assertRaises(SystemExit):                       # never into an opponent-P/V model directory
            export(run, self.tmp / "inference" / "models_x" / "s", "--allow-unbound-carrier")
        with self.assertRaises(SystemExit):                       # the P/V exporters refuse a scorer
            run_config(run, self.L, "policy")
        for kind in ("policy", None):
            bad = make_run(self.tmp / f"kind_{kind}", "linear", seed=7)
            cfg = json.loads((bad / "config.json").read_text())
            cfg["kind"] = kind
            (bad / "config.json").write_text(json.dumps(cfg))
            with self.assertRaises(SystemExit):
                export(bad, self.tmp / f"kind_out_{kind}", "--allow-unbound-carrier")
        stale = make_run(self.tmp / "stale", "linear", seed=8)
        cfg = json.loads((stale / "config.json").read_text())
        cfg["features"] = dict(cfg["features"], digest="0" * 16)
        (stale / "config.json").write_text(json.dumps(cfg))
        with self.assertRaises(SystemExit):
            export(stale, self.tmp / "stale_out", "--allow-unbound-carrier")
        wide = make_run(self.tmp / "wide", "mlp", (1100,), seed=9)
        with self.assertRaises(SystemExit):
            export(wide, self.tmp / "wide_out", "--allow-unbound-carrier")

    def test_cpp_validate_refuses(self):
        run = make_run(self.tmp / "val_run", "mlp", (8,), seed=10)
        out = self.tmp / "val_out"
        export(run, out, "--allow-unbound-carrier")
        good = (out / X.HEADER).read_text()
        cases = {"feature identity differs from this build's own_action.hpp": good.replace(F.feature_identity(self.L)["digest"], "0" * 16),
                 "not an own_action_scorer": good.replace(f'{{"{F.KIND}"', '{"policy"'),
                 "non-finite weight": good.replace("inline constexpr float W0[] = {\n    ", "inline constexpr float W0[] = {\n    __builtin_nanf(\"\"), ", 1)}
        for why, text in cases.items():
            self.assertNotEqual(text, good, why)
            bad = self.tmp / f"val_{abs(hash(why))}"
            bad.mkdir(exist_ok=True)
            (bad / X.HEADER).write_text(text)
            info = None
            try:
                info = run_tool(build_tool(bad), "--model")
            except RuntimeError as exc:                           # a NaN that changes the array length cannot compile
                self.fail(f"{why}: {exc}")
            self.assertFalse(info["valid"], why)
            self.assertEqual(info["why"], why)


if __name__ == "__main__":
    unittest.main()
