"""The aligned Meowmind RL exports and the search diagnosis (task T0061, docs/MEOWMIND_RL_ALIGNED.md).

No PyTorch, no engine, no host: the corpus, the exports and the trees are built from synthetic batches made of the
recorded fixture streams (bots/p_only/tests/fixtures), with perturbed reference actors and random critics as
plumbing. Native checks need a C++ compiler (skipped otherwise). No timing is measured or claimed.
"""
from __future__ import annotations

import gzip
import json
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from fusion import matchlog, runner
from fusion import quant_wide as FW
from training.meow_rl import bridge as B
from training.meow_rl import identity as I
from training.meow_rl import policy as P

REPO = Path(__file__).resolve().parents[1]
for p in ("bots/meow_rl", "bots/meow_rl/aligned", "bots/p_only/tools"):
    sys.path.insert(0, str(REPO / p))
import corpus as K  # noqa: E402
import critics as C  # noqa: E402
import diagnose as D  # noqa: E402
import export as X  # noqa: E402
import preflight as PF  # noqa: E402
import report as RP  # noqa: E402
import trees as TR  # noqa: E402

CXX = runner.find_compiler()
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
GAMES = {"n02": 2, "n04": 4, "n15": 15}
LARGE = [416, 1024, 1024, 512, 1]


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


def critic_arrays(rng, sizes) -> dict:
    out = {}
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        out[f"critic.layers.{i}.weight"] = (rng.standard_normal((b, a)) * np.sqrt(2 / a)).astype(np.float32)
        out[f"critic.layers.{i}.bias"] = (rng.standard_normal(b) * 0.02).astype(np.float32)
    return out


def make_batches(root: Path, lib, actor) -> list[Path]:
    """T0059-like runs: each recorded game becomes 6 episodes (every 6th turn) on distinct maps."""
    dirs = []
    for run in ("large_v_s101", "small_v_s101"):
        d = root / "runs" / run / "batches"
        d.mkdir(parents=True)
        for u, (game, teams) in enumerate(GAMES.items(), 1):
            F, M, V, T = [], [], [], []
            with B.Session(lib) as s:
                for block in B.split_stream(stream(game)):
                    o = s.observe(block)
                    F.append(o.feats), M.append(o.joint_mask), V.append(o.value_feats), T.append(o.turn)
                    s.resolve(P.greedy(P.joint_log_probs(actor, o.feats, o.joint_mask)))
            F, M, V, T = map(np.array, (F, M, V, T))
            rows, eps = {k: [] for k in ("episode", "turn", "feats", "value_feats", "mask")}, []
            for e in range(6):
                idx = np.arange(e, len(T), 6)
                for k, arr in (("turn", T), ("feats", F), ("value_feats", V)):
                    rows[k].append(arr[idx])
                rows["episode"].append(np.full(len(idx), u * 100 + e))
                rows["mask"].append(np.packbits(M[idx], axis=1))
                eps.append({"index": u * 100 + e, "teams": teams, "seat": 0, "map_seed": 1000 * u + 10 * e})   # both runs
            np.savez(d / f"update_{u:06d}.npz", **{k: np.concatenate(v) for k, v in rows.items()})
            (d / f"update_{u:06d}.json").write_text(json.dumps({"episodes": eps}))
        dirs.append(d)
    return dirs


class WideAct16Format(unittest.TestCase):
    def test_activation_rounds_half_up_and_clamps_for_every_sign(self):
        for s in (0, 1, 4, 15, 21):
            for acc in (-(1 << 40), -3, -1, 0, 1, (1 << s) // 2 - 1 if s else 1, (1 << s) // 2 if s else 2, 3 << s,
                        (32767 << s) + (1 << s) // 2, 1 << 50):
                want = max(0, min(32767, int(Fraction(acc, 1 << s) + Fraction(1, 2)) if acc >= 0 else
                                  int(np.floor(Fraction(acc, 1 << s) + Fraction(1, 2)))))
                self.assertEqual(int(FW.activate(np.array([acc]), s)[0]), want, (s, acc))

    def test_the_proved_bounds_hold_at_the_extremes(self):
        rng = np.random.default_rng(3)
        W = [rng.standard_normal((64, 416)) * 0.2, rng.standard_normal((32, 64)), rng.standard_normal((1, 32))]
        b = [rng.standard_normal(64), rng.standard_normal(32), rng.standard_normal(1)]
        q = FW.quantize(W, b, np.full(416, 4096.0), np.full(416, -16384), np.full(416, 16384))
        for L in q["layers"]:
            self.assertLessEqual(int(np.abs(L["w"]).max()), FW.W_LIMIT)
            self.assertLessEqual(L["bound"], FW.ACC_LIMIT)
        self.assertGreaterEqual(q["layers"][0]["shift"], 0)
        self.assertLess(2 * FW.W_LIMIT * FW.ACT_MAX, 1 << 31)           # one int16 pair sum fits int32
        for sign in (1, -1):                                               # inputs at the spec's bound, both signs
            x = (sign * np.sign(q["layers"][0]["w"][0]) * 16384).astype(np.int64)[None]
            FW.forward_raw(q, x)                                           # the checked emulation raises on any overflow

    def test_layer_one_weights_too_large_are_refused(self):
        W = [np.full((4, 8), 1e6), np.ones((1, 4))]
        with self.assertRaises(ValueError):
            FW.quantize(W, [np.zeros(4), np.zeros(1)], np.ones(8), np.full(8, -16384), np.full(8, 16384))

    def test_wider_activations_reduce_the_error_of_an_unsaturated_net(self):
        rng = np.random.default_rng(5)
        W = [(rng.standard_normal((48, 416)) * 0.05).astype(np.float32), (rng.standard_normal((24, 48)) * 0.2).astype(np.float32),
             (rng.standard_normal((1, 24)) * 0.2).astype(np.float32)]
        b = [np.zeros(48, np.float32), np.zeros(24, np.float32), np.zeros(1, np.float32)]
        x = np.clip(rng.standard_normal((400, 416)) * 0.5, -3, 3).astype(np.float32)
        ref = C.E.critic_forward(W, b, x)
        e = {f: np.abs(C.value(C.build(f, W, b, x), x) - ref).mean() for f in ("reference16", "wideact16")}
        self.assertLess(e["wideact16"], e["reference16"] / 4)                # calibrated on these rows: no clipping


class CriticScaleRule(unittest.TestCase):
    def test_never_active_neurons_take_the_layer_median_range(self):
        rng = np.random.default_rng(1)
        W = [(rng.standard_normal((16, 416)) * 0.05).astype(np.float32), (rng.standard_normal((1, 16))).astype(np.float32)]
        b = [np.zeros(16, np.float32), np.zeros(1, np.float32)]
        b[0][3] = -1e3                                                     # neuron 3 never fires
        x = rng.standard_normal((50, 416)).astype(np.float32)
        r = C.ranges(W, b, x)[0]
        live = np.delete(r, 3)
        self.assertEqual(r[3], np.median(live))
        self.assertEqual(C.dead_neurons(W, b, x), [1])

    def test_the_t0056_rule_is_the_deployed_export_unchanged(self):
        rng = np.random.default_rng(2)
        st = {k[len("critic."):]: v for k, v in critic_arrays(rng, [416, 64, 32, 1]).items()}
        W, b = C.E.critic_layers(st)
        x = rng.standard_normal((80, 416)).astype(np.float32)
        self.assertEqual(C.build("reference16_t0056", W, b, x)["digest"], C.E.quantize_critic(W, b, x)["digest"])


@unittest.skipUnless(CXX, "no C++ compiler")
class NativeCritics(unittest.TestCase):
    def test_every_format_is_bit_exact_on_scalar_and_simd_paths(self):
        rng = np.random.default_rng(9)
        st = {k[len("critic."):]: v for k, v in critic_arrays(rng, [416, 96, 40, 1]).items()}
        W, b = C.E.critic_layers(st)
        x = np.clip(rng.standard_normal((300, 416)), -4, 4).astype(np.float32)
        x[:3] = 4.0                                                        # features at the clamp: the largest pair sums
        x[3:6] = -4.0
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for fmt in C.FORMATS:
                be = C.build(fmt, W, b, x[:150])
                h = t / f"critic_{fmt}.hpp"
                h.write_text(C.header(be, "t", I.identity("large_v")))
                res = X.native_critic(CXX, h, be, x, t)
                for path, r in res.items():
                    self.assertTrue(r["bit_equal"], (fmt, path, r))
            self.assertLess(np.abs(C.value(C.build("fp32", W, b, None), x) - C.E.critic_forward(W, b, x)).max(), 1e-4)


@unittest.skipUnless(CXX, "no C++ compiler")
class AlignedPipeline(unittest.TestCase):
    """Corpus -> exports -> bots -> trees -> diagnostics, once, on plumbing data."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="meow-rl-aligned-"))
        cls.lib = B.load()
        cls.actor = P.check_actor(B.reference_weights(cls.lib, "policy"))
        cls.dirs = make_batches(cls.tmp, cls.lib, cls.actor)
        cls.corpus = cls.tmp / "corpus"
        cls.man = K.main(["--batches", *map(str, cls.dirs), "--out", str(cls.corpus), "--total", "280"])
        rng = np.random.default_rng(4)
        cls.npz = {}
        for name, arm in (("p1_large", LARGE), ("p2_large", LARGE), ("p2_small", [416, 64, 32, 1])):
            z = {k: (v + rng.standard_normal(v.shape).astype(np.float32) * 0.05 * np.abs(v).mean()).astype(np.float32)
                 for k, v in cls.actor.items()} | critic_arrays(rng, arm)
            np.savez(cls.tmp / f"{name}.npz", **z)
            cls.npz[name] = cls.tmp / f"{name}.npz"
        cls.args = ["--corpus", str(cls.corpus), "--actor-npz", *(f"{k}={v}" for k, v in cls.npz.items())]
        cls.inf, cls.var = cls.tmp / "inf", cls.tmp / "variants"
        cls.report = X.main([*cls.args, "--out", str(cls.inf), "--variants-dir", str(cls.var)])

    def test_corpus_is_deterministic_disjoint_and_reports_its_shortages(self):
        again = K.main(["--batches", *map(str, self.dirs), "--out", str(self.tmp / "corpus2"), "--total", "280"])
        self.assertEqual(again["splits"], self.man["splits"])
        self.assertEqual(again["selected"], self.man["selected"])
        cal, diag = (set((r[4], r[6]) for r in self.man["selected"][s]) for s in ("calibration", "diagnostic"))
        self.assertFalse(cal & diag)                                        # no map in both splits, whichever run played it
        self.assertEqual(self.man["splits"]["calibration"]["rows"], 280)
        short = self.man["splits"]["calibration"]["selection"]["shortage_by_count"]
        self.assertGreater(short["3"], 0)                                   # no 3-team data: reported, not invented
        self.assertTrue(all(len(row) == len(self.man["selected"]["columns"]) for row in self.man["selected"]["calibration"]))
        with self.assertRaises(SystemExit):
            K.main(["--batches", *map(str, self.dirs), "--out", str(self.corpus), "--total", "280"])

    def test_signature_bits_follow_the_reference_features(self):
        f = np.zeros((1, 3, 5, 112), np.float32)
        v = np.zeros((1, 416), np.float32)
        f[0, :, 0, 0] = [0.3, 1.0, -1.0]                                    # unit 0 low HP, unit 2 dead
        f[0, 1, 0, 1] = 1                                                   # unit 1 carries
        f[0, 2, 0, 57] = 1                                                  # a dead unit's attack line does not count
        self.assertEqual(int(K.signature(f, v)[0]), 1 | 4)
        f[0, 1, 0, 66] = 1
        v[0, 23] = 0.5
        self.assertEqual(int(K.signature(f, v)[0]), 15)

    def test_exports_are_reproducible_and_bit_exact_natively(self):
        r = self.report
        for key, a in r["actors"].items():
            for path, n in a["native"].items():
                self.assertTrue(n["scores_equal"] and n["draws_equal"] and n["greedy_equal"], (key, path))
            self.assertGreater(a["agreement_diagnostic"]["sampled_agreement"], 0.8)
        self.assertTrue(r["actors"]["warm"]["array_route"]["tensors_equal"])
        self.assertIn(r["actors"]["warm"]["native"]["simd1"]["kernel"], ("avx2", "sse2", "neon", "scalar"))
        self.assertEqual(r["actors"]["warm"]["native"]["simd0"]["kernel"], "scalar")
        self.assertTrue(r["actors"]["warm"]["array_route"]["log_probs_equal"])
        for fmt in C.FORMATS:
            self.assertTrue(all(v["bit_equal"] for v in r["critics"][fmt]["native"].values()), fmt)
        self.assertFalse(r["critics"]["reference16_t0056"]["deployed"])
        self.assertFalse(r["evidence"])                                     # arrays, not the pinned checkpoints
        X.main([*self.args, "--out", str(self.tmp / "inf2"), "--variants-dir", str(self.tmp / "var2"), "--no-native"])
        for f in ("p2_large/actor.hpp", "p2_large/critic_wideact16.hpp", "warm/actor.hpp"):
            self.assertEqual((self.tmp / "inf2" / f).read_bytes(), (self.inf / f).read_bytes(), f)

    def test_search_arms_share_the_actor_header_of_d(self):
        v = self.report["variants"]
        shared = {v[n]["actor_hpp_sha256"] for n in v if v[n]["actor"] == "p2_large"}
        self.assertEqual(len(shared), 1)
        self.assertEqual({v[n]["critic"] for n in v if v[n]["search"]}, set(C.FORMATS))
        for n in v:
            self.assertTrue(n.startswith("meow_rl_aligned_"))
        historical = {x["name"] for x in json.loads((REPO / "bots" / "meow_rl" / "variants.json").read_text())["variants"]}
        self.assertFalse(historical & set(v))

    def test_a_checkpoint_that_differs_from_its_pin_is_refused(self):
        cfg = json.loads(X.CONFIG.read_text())
        fake = self.tmp / "fake.pt"
        fake.write_bytes(b"not the checkpoint")
        spec = dict(cfg["actors"]["p2_large"], path=str(fake.relative_to(self.tmp)))
        cwd_repo = X.REPO
        try:
            X.REPO = self.tmp
            with self.assertRaises(SystemExit):
                X.load_actor("p2_large", spec, {}, self.lib)
        finally:
            X.REPO = cwd_repo

    def _bots(self, cfg: dict) -> dict:
        from bundle import bundle  # noqa: PLC0415
        vd = self.tmp / f"v_{cfg['search']['deadline_ms']}_{cfg['search'].get('inject', 0)}"
        vd.mkdir(exist_ok=True)
        out = {}
        for name in ("meow_rl_aligned_p2_large", "meow_rl_aligned_p2_large_r16", "meow_rl_aligned_p2_large_w16",
                     "meow_rl_aligned_p2_large_fp32"):
            (vd / f"{name}.cpp").write_text(X.variant_source(name, cfg["variants"][name], cfg, self.inf, vd), encoding="utf-8")
            text, _ = bundle(vd / f"{name}.cpp")
            src = vd / f"{name}_bundle.cpp"
            src.write_text(text, encoding="utf-8")
            exe = vd / f"{name}{runner.EXE}"
            extra = [f"-DMEOW_RL_SEARCH_TEST_INJECT_DELAY_MS={cfg['search']['inject']}"] if cfg["search"].get("inject") else []
            subprocess.run([CXX, "-std=c++20", "-O2", "-w", *extra, str(src), "-o", str(exe)], check=True, capture_output=True)
            out[name] = exe
        return out

    def test_search_v2_chooses_exactly_what_the_tree_harness_scores(self):
        cfg = json.loads(X.CONFIG.read_text(encoding="utf-8"))
        cfg["search"] |= {"deadline_ms": 100000, "first_ms": 100000}
        bots = self._bots(cfg)
        text = stream("n04")
        tree = TR.compile_tree(self.inf / "p2_large" / "actor.hpp", self.tmp / f"tree{runner.EXE}")
        (self.tmp / "n04.txt").write_text(text, encoding="utf-8")
        turns = list(range(2, 401, 9))
        r = subprocess.run([str(tree), str(self.tmp / "n04.txt"), *map(str, turns)], capture_output=True, text=True, check=True,
                           env=dict(os.environ, MEOW_RL_TREE_LEAVES=str(self.tmp / "eq.f32")))
        trees = [json.loads(x) for x in r.stdout.splitlines()]
        leaves = np.fromfile(self.tmp / "eq.f32", np.float32).reshape(-1, 416)
        st = {k[len("critic."):]: v for k, v in np.load(self.npz["p2_large"]).items() if k.startswith("critic.")}
        W, b = C.E.critic_layers(st)
        cal = np.load(self.corpus / "calibration.npz")["value_feats"]
        for name, fmt in (("meow_rl_aligned_p2_large_r16", "reference16"), ("meow_rl_aligned_p2_large_w16", "wideact16"),
                          ("meow_rl_aligned_p2_large_fp32", "fp32")):
            vals = C.value(C.build(fmt, W, b, cal), leaves)
            out = subprocess.run([str(bots[name])], input=text, capture_output=True, text=True).stdout.splitlines()
            for t in trees:
                j = t["roots"][D.choose(D.root_values(t, vals))]
                line = out[t["turn"] - 1].split()
                got = "SUDLR".index(line[1]) * 25 + "SUDLR".index(line[3]) * 5 + "SUDLR".index(line[5])
                self.assertEqual(got, j, (fmt, t["turn"]))

    def test_search_out_of_time_plays_the_actor_and_counts_the_refusal(self):
        cfg = json.loads(X.CONFIG.read_text())
        cfg["search"] |= {"deadline_ms": 30, "first_ms": 30, "inject": 40}
        bots = self._bots(cfg)
        text = "".join(B.split_stream(stream("n02"))[:6])
        direct = subprocess.run([str(bots["meow_rl_aligned_p2_large"])], input=text, capture_output=True, text=True).stdout
        r = subprocess.run([str(bots["meow_rl_aligned_p2_large_r16"])], input=text, capture_output=True, text=True)
        self.assertEqual(r.stdout, direct)
        c = PF.parse_counters(r.stderr.splitlines())
        self.assertEqual(c["search_used"], 0)
        self.assertEqual(c["search_fallback"], 6)
        self.assertEqual(c["refuse_late"] + c["refuse_trajectory"] + c["refuse_critic"], 6)

    def test_trees_select_build_and_diagnose(self):
        td = self.tmp / "trees"
        (td / "matches").mkdir(parents=True)
        for g in GAMES:
            blocks = B.split_stream(stream(g))
            init, b1 = matchlog.split_init(blocks[0])
            matchlog.save(td / "matches" / f"p-{g}.json.gz",
                          {"schema": matchlog.SCHEMA, "match_id": f"p-{g}", "teams": GAMES[g], "init": init,
                           "blocks": [b1, *blocks[1:]], "seats": [{"name": "@candidate"}] + [{"name": "o"}] * (GAMES[g] - 1)})
        ns = lambda **k: type("A", (), k)  # noqa: E731
        roots = TR.select(ns(out=str(td), roots=56, per_game=8, seed=TR.SEED))
        self.assertEqual(roots["selected"], 12)                             # 4 per available count, the rest reported
        self.assertEqual(roots["shortage_by_count"]["3"], 4)
        with self.assertRaises(SystemExit):
            TR.select(ns(out=str(td), roots=56, per_game=8, seed=TR.SEED))
        man = TR.build(ns(out=str(td), actor_header=str(self.inf / "p2_large" / "actor.hpp")))
        self.assertEqual(man["trees"], 12)
        tree = json.loads((td / "trees.jsonl").read_text().splitlines()[0])
        self.assertEqual(len(tree["mask"]), 125)
        self.assertEqual(len(tree["scenarios"]), 2)
        self.assertEqual(set(tree["scenarios"][0]), {"opp1_rng", "sim1", "opp2_rng", "cont", "sim2", "opp1_orders"})
        self.assertEqual(len(tree["leaves"]), 4 * 2)
        self.assertTrue(all(len(lf["root_actions"]) == 3 for lf in tree["leaves"]))
        r = D.main(["--trees", str(td), "--corpus", str(self.corpus), "--critic-npz", str(self.npz["p2_large"]),
                    "--headers", str(self.inf), "--out", str(self.tmp / "diag")])
        for k in ("F", "R", "H"):
            self.assertTrue(all(v["bit_equal"] for v in r["native"][k].values()), k)
        self.assertEqual(r["root"]["trees"], 12)
        self.assertIn("top_action_agreement", r["root"]["R"])
        self.assertTrue((self.tmp / "diag" / "DIAGNOSTICS.md").read_text().startswith("# Fixed-work"))
        scored = [json.loads(x) for x in (self.tmp / "diag" / "ROOTS_SCORED.jsonl").read_text().splitlines()]
        self.assertEqual(len(scored), 12)
        self.assertEqual(set(scored[0]["values"]), set(D.BACKENDS))


class Gates(unittest.TestCase):
    def row(self, **k) -> dict:
        r = {"fixture_id": "f", "teams": 2, "faults": 0, "late": 0, "dead": False, "opponent_faults": 0, "opponent_late": 0,
             "opponent_dead": 0, "max_ms": 10.0, "first_ms": 100.0, "turns_played": 400,
             "counters": {"turns": 400, "search_used": 5}}
        return r | k

    def test_preflight_verdict_fails_every_kind_of_problem(self):
        self.assertEqual(PF.verdict([self.row()], True), (True, []))
        for bad in ({"opponent_late": 1}, {"dead": True}, {"counters": None}, {"counters": {"turns": 399, "search_used": 1}},
                    {"max_ms": 48.0}, {"first_ms": 1800.0}, {"max_ms": None}):
            self.assertFalse(PF.verdict([self.row(**bad)], False)[0], bad)
        self.assertFalse(PF.verdict([self.row(counters={"turns": 400, "search_used": 0})], True)[0])
        self.assertFalse(PF.verdict([self.row(counters={"turns": 400, "refused": 400, "search_used": 0})], False)[0])
        text = "#define MEOW_RL_SEARCH_TRAJ_US 6000\n#ifndef MEOW_RL_SEARCH_TRAJ_US\n#define MEOW_RL_SEARCH_TRAJ_US 4000\n#endif\n"
        self.assertEqual(PF.first_defines(text), {"MEOW_RL_SEARCH_TRAJ_US": 6000})

    def test_counter_lines_of_both_search_versions_parse(self):
        v1 = "MEOWRL turns 7 refused 0 search_used 3 search_fallback 4 leaves 20"
        v2 = v1 + " trajectories 24 refuse_trajectory 1 refuse_expand 0 refuse_critic 2 refuse_late 1 max_setup_us 80 " \
                  "max_traj_us 900 max_expand_us 300 max_critic_us 400 max_search_us 5000 max_resolve_us 20 max_turn_us 6000 " \
                  "first_turn_us 9000 critic wideact16"
        self.assertEqual(PF.parse_counters([v1]), {"turns": 7, "refused": 0, "search_used": 3, "search_fallback": 4, "leaves": 20})
        self.assertEqual(PF.parse_counters(["x", v2])["max_critic_us"], 400)
        self.assertIsNone(PF.parse_counters(["nothing"]))

    def test_run_screen_refuses_unfrozen_or_unmeasured_screens(self):
        import run_screen as RS  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as t:
            cfg = json.loads((REPO / "profiles" / "meow_rl_aligned_screen_a.json").read_text())
            p = Path(t) / "a.json"
            p.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(SystemExit, "no seed"):
                RS.main(["--config", str(p), "--analyze-only"])
            p.write_text(json.dumps(cfg | {"seed": 1}))
            with self.assertRaisesRegex(SystemExit, "freezes 8 workers"):
                RS.main(["--config", str(p), "--workers", "6", "--analyze-only"])
            rec = Path(t) / "pre.json"
            rec.write_text(json.dumps({"passed": True, "workers": 8, "variants": {"x": {"binary_sha256": "aa"}}}))
            RS.check_preflight(rec, {"x": {"binary_sha256": "aa"}}, ["x"], 8)
            with self.assertRaises(SystemExit):
                RS.check_preflight(rec, {"x": {"binary_sha256": "bb"}}, ["x"], 8)
            with self.assertRaises(SystemExit):
                RS.check_preflight(rec, {"x": {"binary_sha256": "aa"}}, ["x"], 6)
            frozen_cfg = cfg | {"seed": 7}
            p.write_text(json.dumps(frozen_cfg))
            sched = Path(t) / "schedule.json"
            sched.write_text(json.dumps({"seed": 7, "team_counts": cfg["teams"], "maps_per_count": 28,
                                         "reference": [{"name": n} for n in cfg["reference"]]}))
            fz = Path(t) / "FREEZE.json"
            fz.write_text(json.dumps({"schedules": {"a": {"profile": str(p), "profile_sha256": RS.sha256_file(p),
                                                          "schedule_sha256": RS.sha256_file(sched)}}}))
            RS.check_freeze(fz, p, frozen_cfg, sched)
            fzd = json.loads(fz.read_text())
            fzd["schedules"]["a"]["arms"] = {"x": {"binary_sha256": "aa"}}
            fz.write_text(json.dumps(fzd))
            RS.check_freeze(fz, p, frozen_cfg, sched, {"x": {"binary_sha256": "aa"}})
            with self.assertRaisesRegex(SystemExit, "not the one frozen"):                 # rebuilt after the freeze
                RS.check_freeze(fz, p, frozen_cfg, sched, {"x": {"binary_sha256": "bb"}})
            p.write_text(json.dumps(frozen_cfg | {"margin": 0.01}))                 # the profile changed after the freeze
            with self.assertRaisesRegex(SystemExit, "changed after"):
                RS.check_freeze(fz, p, frozen_cfg, sched)
            p.write_text(json.dumps(frozen_cfg))
            sched.write_text(sched.read_text() + " ")                               # another schedule
            with self.assertRaisesRegex(SystemExit, "not the frozen schedule"):
                RS.check_freeze(fz, p, frozen_cfg, sched)
            with self.assertRaisesRegex(SystemExit, "does not exist"):
                RS.check_freeze(Path(t) / "none.json", p, frozen_cfg, sched)

    def test_schedules_are_frozen_specifications_with_their_budgets(self):
        a = json.loads((REPO / "profiles" / "meow_rl_aligned_screen_a.json").read_text())
        b = json.loads((REPO / "profiles" / "meow_rl_aligned_screen_b.json").read_text())
        for cfg in (a, b):
            self.assertIsNone(cfg["seed"])
            self.assertEqual((cfg["workers"], cfg["workers_exact"], cfg["first_ms"], cfg["turn_ms"]), (8, True, 2000, 50))
            self.assertEqual(len(cfg["teams"]) * cfg["maps_per_count"], cfg["fixtures"])
            self.assertEqual(cfg["fixtures"] * (len(cfg["arms"]) + len(cfg["aa"])), cfg["games"])
            self.assertEqual(cfg["reference"], ["Fable", "MiMo", "Kimi", "GLM", "Player4"])
        self.assertEqual(a["games"] + b["games"], 3920)
        self.assertNotEqual(a["work_dir"], b["work_dir"])
        self.assertEqual(len(b["families"]["search"]["comparisons"]), 5)
        self.assertEqual(b["multiplicity"]["family"], 5)

    def test_verdicts_use_computed_family_quantiles(self):
        nd = statistics.NormalDist()
        self.assertAlmostEqual(RP.family_z("bonferroni", 0.05, 2), nd.inv_cdf(1 - 0.05 / 4))
        self.assertAlmostEqual(RP.family_z("bonferroni", 0.05, 5), 2.5758293035489, places=9)
        self.assertAlmostEqual(RP.family_z("none", 0.05, 1), 1.959963984540054)
        cfg = json.loads((REPO / "profiles" / "meow_rl_aligned_screen_a.json").read_text())
        mk = lambda m, se: {"overall": {"mean": m, "se": se, "fixtures": 392}, "missing": 0}  # noqa: E731
        screen = {"seed": 1, "fixtures": 392, "incomplete": [], "rerun_noise": {},
                  "comparisons": {"meow_rl_aligned_p2_large - meow_rl_aligned_warm": mk(0.021, 0.01),
                                  "meow_rl_aligned_p2_large - meow_rl_aligned_p1_large": mk(0.021, 0.01),
                                  "meow_rl_aligned_p2_large - meow_rl_aligned_p2_small": mk(-0.03, 0.01)}}
        r = RP.judge(cfg, screen)
        self.assertEqual(r["families"]["primary"]["rows"][0]["verdict"], "helps")          # 2.1 > 1.96 se
        self.assertEqual(r["families"]["secondary"]["rows"][0]["verdict"], "not shown")    # 2.1 < 2.24 se
        self.assertEqual(r["families"]["secondary"]["rows"][1]["verdict"], "hurts")
        part = json.loads(json.dumps(screen))
        part["comparisons"]["meow_rl_aligned_p2_large - meow_rl_aligned_warm"]["missing"] = 390
        self.assertEqual(RP.judge(cfg, part)["families"]["primary"]["rows"][0]["verdict"], "incomplete")
        part["comparisons"]["meow_rl_aligned_p2_large - meow_rl_aligned_warm"]["overall"] = None
        self.assertEqual(RP.judge(cfg, part)["families"]["primary"]["rows"][0]["verdict"], "incomplete")
        self.assertEqual(RP.judge(cfg, screen | {"incomplete": ["f1"]})["families"]["primary"]["rows"][0]["verdict"], "incomplete")
        empty_aa = RP.judge(cfg, screen | {"rerun_noise": {"x - x~aa": {"overall": None, "fixtures": 0}}})
        self.assertIsNone(empty_aa["rerun_noise"]["x - x~aa"]["mean_pp"])


if __name__ == "__main__":
    unittest.main()
