"""Tests of the P-only package (no games; the compile and contract tests need a C++ compiler).

    python -m unittest discover -s bots/p_only/tests -v
"""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

import analyze  # noqa: E402
import bundle  # noqa: E402
import screen_runner  # noqa: E402
import size_rule  # noqa: E402
from fusion.runner import find_compiler  # noqa: E402

BOTS = REPO / "bots" / "p_only" / "bots"
HEADERS = [BOTS / "c08_ponly.hpp", BOTS / "v12_ponly.hpp", REPO / "bots" / "p_only" / "include" / "ponly.hpp"]
CXX = find_compiler()

# Identifiers of the hand-written opponent model and of the legacy mixture: none
# may remain in the P-only code (comments are stripped first).
FORBIDDEN = ["mixedMarginal", "mixedJointProb", "LegacyForecast", "drawTeam", "alpha", "MOD_FUSION_P",
             "attentionWeight", "priority(", "crossStay", "crossMove", "otherFocus", "lasttarget", "W_other_attention",
             "scriptedStep", "rivalsFirstStep", "rivalsNextStep", "plyAttention", "plyDiscount", "plies",
             "rivalAttacks", "rivalWants", "rivalDrops", "groundPickupProbability", "groundFlagAt", "closestGoal",
             "enemyDropProbability", "observeDrops", "forecastDrops", "dropCounts", "shootChance", "dropChance",
             "e.attention", "tw("]


def code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


class HandWrittenPredictionRemoved(unittest.TestCase):
    def test_no_legacy_identifiers(self):
        for h in HEADERS:
            text = code_only(h)
            for name in FORBIDDEN:
                with self.subTest(header=h.name, identifier=name):
                    self.assertNotIn(name, text)

    def test_every_opponent_prediction_reads_the_network_view(self):
        c08 = code_only(BOTS / "c08_ponly.hpp")
        for needle in ["void forecast(const Game& g){", "void rivalOrders(const Game& g,int model", "void predict(const Game& g) {"]:
            body = c08.split(needle, 1)[1][:1500]
            with self.subTest(function=needle):
                self.assertIn("fSession.cur", body)
        v12 = code_only(BOTS / "v12_ponly.hpp")
        self.assertIn("outcomes(fSession.view", v12)
        self.assertIn("rivalOrders(g,fSession.view", v12)
        self.assertIn("rivalOrders(g,branchView", v12)

    def test_derived_headers_match_their_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            for script, name in [("make_c08.py", "c08_ponly.hpp"), ("make_v12.py", "v12_ponly.hpp"),
                                 ("make_c08_control.py", "c08_control.hpp")]:
                subprocess.run([sys.executable, str(REPO / "bots" / "p_only" / "tools" / "derive" / script)], check=True,
                               env={**os.environ, "PONLY_DERIVE_DIR": tmp}, capture_output=True)
                with self.subTest(header=name):
                    self.assertEqual((Path(tmp) / name).read_text(encoding="utf-8"), (BOTS / name).read_text(encoding="utf-8"))


class Bundle(unittest.TestCase):
    def test_conditions(self):
        m = {"FUSION_TIER_M": "1", "MOD_FUSION_P": "1", "MOD_FUSION_V": "0", "GUARD": ""}
        self.assertTrue(bundle.evaluate("defined(FUSION_TIER_M)", m))
        self.assertFalse(bundle.evaluate("defined(FUSION_TIER_L)", m))
        self.assertTrue(bundle.evaluate("MOD_FUSION_P || MOD_FUSION_V", m))
        self.assertFalse(bundle.evaluate("MOD_FUSION_V && MOD_FUSION_P", m))
        self.assertTrue(bundle.evaluate("!defined(FUSION_TIER_L) && MOD_FUSION_P == 1", m))
        with self.assertRaises(bundle.Unknown):
            bundle.evaluate("defined(__AVX2__)", m)

    def test_variants_bundle_into_one_file(self):
        variants = json.loads((REPO / "bots" / "p_only" / "profiles" / "variants.json").read_text(encoding="utf-8"))["variants"]
        for v in variants:
            with self.subTest(variant=v["name"]):
                text, files = bundle.bundle(REPO / v["source"])
                active = [line for line in text.splitlines() if bundle.INCLUDE.match(line)]
                self.assertEqual(active, [], "a local include was left in the bundle")
                paths = [str(f).replace("\\", "/") for f in files]
                if v.get("models"):
                    self.assertTrue(any(f"{v['models']}/" in f for f in paths), "the bundle lacks its mixed model set")
                elif v["tier"]:
                    self.assertTrue(any(f"models_{v['tier'].lower()}/" in f or (v["tier"] == "S" and "/models/" in f)
                                        for f in paths), "the bundle lacks its tier's weights")


class SizeRule(unittest.TestCase):
    def arm(self, p99, faults=0, late=0):
        return {"games": 15, "faults": faults, "late_replies": late, "p99_ms_worst": p99, "max_ms": p99 + 3}

    def test_largest_c08_size_within_40_ms(self):
        timing = {"arms": {"c08_ponly_m": self.arm(25), "c08_ponly_l": self.arm(33), "c08_ponly_xl": self.arm(39.9),
                           "c08_ponly_xxl": self.arm(44), "v12_ponly_xl": self.arm(8), "v12_ponly_xxl": self.arm(12, late=1)}}
        d = size_rule.decide(timing)
        self.assertEqual(d["c08"], "c08_ponly_xl")
        self.assertEqual(d["v12"], ["v12_ponly_xl"])
        arms, comp = size_rule.screen_arms(d["c08"], d["v12"])
        self.assertIn("c08_control_2ply", arms)
        self.assertIn(["c08_ponly_xl", "c08_control_2ply"], comp)

    def test_no_c08_size_qualifies(self):
        d = size_rule.decide({"arms": {"c08_ponly_m": self.arm(41), "c08_ponly_l": self.arm(30, faults=1)}})
        self.assertIsNone(d["c08"])


class Runner(unittest.TestCase):
    def test_interleaved_balanced_order(self):
        fixtures = [{"fixture_id": f"f{i}"} for i in range(6)]
        arms = ["a", "b", "c"]
        order = screen_runner.job_order(fixtures, arms)
        self.assertEqual(len(order), 18)
        for i in range(6):                      # each fixture runs every arm, consecutively
            self.assertEqual(sorted(arm for fx, arm in order[3 * i:3 * i + 3]), arms)
        for pos in range(3):                    # every arm takes every position equally often
            self.assertEqual(sorted(order[3 * i + pos][1] for i in range(6)), sorted(arms * 2))

    def test_aa_identity_is_the_base_identity(self):
        manifest = {"x": {"bundle_sha256": "b", "binary_sha256": "e", "inputs": [{"path": "inference/models_xl/w.hpp", "sha256": "1"}]}}
        ids = screen_runner.identities(["x", "x~aa"], manifest, {"fixtures": [], "reference": []}, (2.0, 0.05))
        self.assertEqual(ids["x"]["digest"], ids["x~aa"]["digest"])
        manifest["x"]["inputs"][0]["sha256"] = "2"      # a changed model is another identity
        self.assertNotEqual(screen_runner.identities(["x"], manifest, {"fixtures": [], "reference": []}, (2.0, 0.05))["x"]["digest"],
                            ids["x"]["digest"])


class Analyze(unittest.TestCase):
    def test_paired_difference(self):
        fixtures = {f"f{i}": {"teams": 2 if i < 4 else 5} for i in range(8)}
        a = {f: {"utility": 1.0, "score": [1]} for f in fixtures}
        b = {f: {"utility": 0.5 if i % 2 else 1.0, "score": [1] if i % 2 == 0 else [0]} for i, f in enumerate(fixtures)}
        r = analyze.paired(a, b, fixtures, [2, 5])
        self.assertAlmostEqual(r["overall"]["mean"], 0.25)
        self.assertEqual((r["improved"], r["unchanged"], r["degraded"]), (4, 4, 0))
        self.assertEqual(r["identical_scores"], 4)
        self.assertEqual(r["bands"]["2"]["fixtures"], 4)
        self.assertIsNone(r["bands"]["6-10"])

    def test_interaction_contrast(self):
        fixtures = {f"f{i}": {"teams": 3} for i in range(4)}
        u = {"A": 0.5, "B": 0.75, "C": 0.5, "D": 1.0}          # V helps by .25 at XL and .5 at XXL
        rows = {k: {f: {"utility": v, "score": [0]} for f in fixtures} for k, v in u.items()}
        name, terms = analyze.parse_contrast("ix=D-C-B+A")
        self.assertEqual([w for w, _ in terms], [1, -1, -1, 1])
        r = analyze.contrast([(w, rows[a]) for w, a in terms], fixtures, [3])
        self.assertAlmostEqual(r["overall"]["mean"], 0.25)

    def test_missing_fixture_is_never_imputed(self):
        fixtures = {f"f{i}": {"teams": 2} for i in range(4)}
        a = {f: {"utility": 0.5, "score": [1]} for f in fixtures}
        b = {f: {"utility": 0.5, "score": [1]} for f in ("f0", "f1")}   # the comparator misses two fixtures
        for x, y in ((a, b), (b, a)):
            r = analyze.paired(x, y, fixtures, [2])
            self.assertEqual((r["fixtures"], r["scheduled"], r["missing"]), (2, 4, 2))
            self.assertAlmostEqual(r["overall"]["mean"], 0.0)            # equal arms never look better or worse
            self.assertEqual(r["overall"]["verdict"], "incomplete")
        self.assertIsNone(analyze.paired(a, {}, fixtures, [2])["overall"])

    def test_margin_and_exclusions(self):
        fixtures = {f"f{i}": {"teams": 2} for i in range(4)}
        a = {f: {"utility": 0.5 + d, "score": [0]} for f, d in zip(fixtures, (0.001, 0.0012, 0.0008, 0.001))}
        b = {f: {"utility": 0.5, "score": [0]} for f in fixtures}
        self.assertEqual(analyze.paired(a, b, fixtures, [2])["overall"]["verdict"], "helps")
        r = analyze.paired(a, b, fixtures, [2], margin=0.002)
        self.assertEqual(r["overall"]["verdict"], "not shown")          # inside the negligible margin
        self.assertEqual((r["improved"], r["unchanged"], r["degraded"]), (0, 4, 0))
        r = analyze.paired(a, b, fixtures, [2], exclude={"f3"})
        self.assertEqual((r["fixtures"], r["scheduled"], r["missing"]), (3, 3, 0))

    def test_family_wise_intervals_decide_the_verdict(self):
        self.assertAlmostEqual(analyze.family_z("bonferroni", 0.05, 6), 2.6383, places=4)
        self.assertAlmostEqual(analyze.family_z("none", 0.05, 6), 1.96, places=2)
        fixtures = {f"f{i}": {"teams": 2} for i in range(20)}
        a = {f: {"utility": 0.53 + (0.03 if i % 2 else -0.03), "score": [0]} for i, f in enumerate(fixtures)}
        b = {f: {"utility": 0.5, "score": [0]} for f in fixtures}
        plain = analyze.paired(a, b, fixtures, [2])["overall"]          # mean 0.03, se ~0.0069: 95% [0.016, 0.044]
        fam = analyze.paired(a, b, fixtures, [2], z=5.0)["overall"]     # a wide family interval: [-0.004, 0.064]
        self.assertEqual(plain["ci95"], fam["ci95"])                    # the 95% interval is always kept
        self.assertNotIn("ci", plain)
        self.assertEqual((plain["verdict"], fam["verdict"], fam["verdict_95"]), ("helps", "not shown", "helps"))
        self.assertLess(fam["ci"][0], 0)

    def test_meowrl_counters_and_their_coverage(self):
        tmp = Path(tempfile.mkdtemp(prefix="meowrl-matches-"))
        key = __import__("hashlib").sha256(b"armA").hexdigest()[:10]

        def match(lines, turns=400):
            return {"seats": [{"name": "@candidate"}, {"name": "X"}], "blocks": ["b"] * turns,
                    "bots": [{"stderr_tail": lines}, {"stderr_tail": []}]}
        for i, lines in enumerate([["MEOWRL turns 399 refused 0 search_used 300 search_fallback 99 leaves 2400",
                                    "MEOWRL turns 400 refused 1 search_used 300 search_fallback 100 leaves 2400"],
                                   ["MEOWRL model x scheme y"]]):          # the second game lost its counter line
            (tmp / f"g{i}-{key}.json.gz").write_bytes(gzip.compress(json.dumps(match(lines)).encode()))
        c = analyze.counters(tmp, {key: "armA"})["armA"]["meowrl"]
        self.assertEqual((c["games"], c["games_with_counters"]), (2, 1))
        self.assertEqual((c["turns"], c["game_turns"], c["search_used"], c["search_fallback"], c["refused"]), (400, 800, 300, 100, 1))
        self.assertEqual(c["turn_coverage"], 0.5)                       # the uncounted game's turns stay in the denominator
        for p in tmp.glob("g0-*"):
            p.unlink()
        none = analyze.counters(tmp, {key: "meow_rl_x"})["meow_rl_x"]["meowrl"]   # no counters at all: zero, not absent
        self.assertEqual((none["games_with_counters"], none["turn_coverage"]), (0, 0.0))

    def test_analysis_applies_the_family_and_reports_opponent_faults(self):
        tmp = Path(tempfile.mkdtemp(prefix="analyze-family-"))
        fx = [{"fixture_id": f"f{i}", "teams": 2 + i % 2} for i in range(12)]
        sched = {"seed": 1, "fixtures": fx, "reference": []}
        (tmp / "s.json").write_text(json.dumps(sched))
        sid = analyze.schedule_identity(sched)
        (tmp / "m.json").write_text(json.dumps({"variants": {a: {"binary_sha256": a} for a in ("A", "B", "C")}}))
        rows = []
        for f in fx:
            for arm, u in (("A", 0.6), ("B", 0.5), ("C", 0.5)):
                rows.append({"candidate": arm, "source_sha256": arm, "fixture_id": f["fixture_id"], "teams": f["teams"],
                             "schedule": sid, "utility": u + (0.01 if f["fixture_id"] in ("f1", "f2") else 0), "score": [1],
                             "faults": 0, "dead": None, "late": 0, "first_ms": 1, "p99_ms": 1, "max_ms": 1,
                             "opponent_faults": 1 if arm == "A" else 0, "opponent_dead": 0})
        (tmp / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        args = ["--results", str(tmp / "r.jsonl"), "--schedule", str(tmp / "s.json"), "--manifest", str(tmp / "m.json"),
                "--arms", "A", "B", "C", "--compare", "A:B", "--compare", "B:C", "--out", str(tmp / "out")]
        try:
            analyze.main(args + ["--multiplicity", "bonferroni"])
        except SystemExit as e:
            self.assertIn(e.code, (0, None))
        out = json.loads((tmp / "out" / "screen.json").read_text())
        self.assertEqual(out["multiplicity"]["comparisons"], 2)
        self.assertAlmostEqual(out["multiplicity"]["z"], 2.2414, places=3)
        self.assertIn("ci", out["comparisons"]["A - B"]["overall"])
        self.assertEqual(out["arms"]["A"]["opponent_faults"], 12)
        md = (tmp / "out" / "SCREEN.md").read_text()
        self.assertIn("Comparison intervals:", md)
        self.assertIn("Interval (z = 2.2414)", md)
        self.assertNotIn("Intervals: 95%,", md)
        with self.assertRaises(SystemExit) as cm:                        # a comparison of an arm not analyzed is an error
            analyze.main(args[:-2] + ["--compare", "A:Z", "--out", str(tmp / "out2")])
        self.assertNotIn(cm.exception.code, (0, None))


class MeowRLPreflight(unittest.TestCase):
    def setUp(self):
        import preflight_meow_rl  # noqa: PLC0415
        self.P = preflight_meow_rl

    def row(self, **kw):
        base = {"arm": "a", "fixture_id": "f", "teams": 15, "faults": 0, "late": 0, "dead": None, "max_ms": 10.0,
                "opponent_faults": 0, "opponent_late": 0, "opponent_dead": 0,
                "meowrl": {"turns": 400, "refused": 0, "search_used": 1, "search_fallback": 0, "leaves": 8}}
        return base | kw

    def test_missing_counters_and_opponent_faults_fail_a_level(self):
        self.assertEqual(self.P.level_verdict([self.row()]), (True, []))
        ok, why = self.P.level_verdict([self.row(meowrl=None)])
        self.assertFalse(ok)
        self.assertIn("no MEOWRL counters in 1 games", why)
        self.assertFalse(self.P.level_verdict([self.row(opponent_faults=2)])[0])
        self.assertFalse(self.P.level_verdict([self.row(max_ms=48.0)])[0])
        self.assertIsNone(self.P.parse_meowrl(["MEOWRL model x"]))
        self.assertEqual(self.P.parse_meowrl(["MEOWRL turns 3 refused 0 search_used 2 search_fallback 1 leaves 9"])["search_fallback"], 1)

    def test_the_tested_binary_must_be_the_manifests(self):
        import hashlib  # noqa: PLC0415
        tmp = Path(tempfile.mkdtemp(prefix="preflight-bind-"))
        (tmp / "b.cpp").write_text("#define MEOW_RL_SEARCH 1\n", encoding="utf-8")
        (tmp / "x.exe").write_bytes(b"binary")
        entry = {"bundle": "b.cpp", "binary": "x.exe",
                 "bundle_sha256": hashlib.sha256(b"#define MEOW_RL_SEARCH 1\n").hexdigest(),
                 "binary_sha256": hashlib.sha256(b"binary").hexdigest()}
        self.assertIn("MEOW_RL_SEARCH", self.P.checked_bundle("a", entry, tmp))
        (tmp / "b.cpp").write_text("#define MEOW_RL_SEARCH 1\n#define MEOW_RL_SEARCH_DEADLINE_MS 25\n", encoding="utf-8")
        with self.assertRaises(SystemExit):                             # a rebuilt bundle beside the old binary
            self.P.checked_bundle("a", entry, tmp)
        (tmp / "b.cpp").write_text("#define MEOW_RL_SEARCH 1\n", encoding="utf-8")
        (tmp / "x.exe").write_bytes(b"other")
        with self.assertRaises(SystemExit):
            self.P.checked_bundle("a", entry, tmp)

    def test_the_deadline_is_read_from_the_bundle(self):
        text = ("#define FUSION_MAX_WIDTH 1024\n#define MEOW_RL_SEARCH 1\n#define MEOW_RL_SEARCH_DEADLINE_MS 25\n"
                "#ifndef MEOW_RL_SEARCH_DEADLINE_MS\n#define MEOW_RL_SEARCH_DEADLINE_MS 30\n#endif\n")
        self.assertEqual(self.P.build_settings(text), {"search": True, "search_deadline_ms": 25})
        self.assertEqual(self.P.build_settings("#ifndef MEOW_RL_SEARCH\n#define MEOW_RL_SEARCH 0\n#endif\n"),
                         {"search": False, "search_deadline_ms": None})


@unittest.skipUnless(CXX, "no C++ compiler")
class Replay(unittest.TestCase):
    """Fixed-work replay (tests/replay_test.cpp): V off equals V compiled in at weight 0, and V never changes P."""

    def outputs(self, carrier: str, defines: list[str], stream: Path, tmp: str) -> tuple[list[list[str]], int]:
        exe = Path(tmp) / ("replay_" + carrier + "_" + "_".join(d.lstrip("-D").replace("=", "") for d in defines))
        subprocess.run([CXX, "-std=c++20", "-O2", f"-I{REPO / 'inference'}", f"-I{REPO / 'bots' / 'p_only' / 'include'}", f"-I{BOTS}",
                        f"-DCARRIER_{carrier}", "-DFUSION_TIER_PXL_VM", *defines, str(REPO / "bots" / "p_only" / "tests" / "replay_test.cpp"),
                        "-o", str(exe)], check=True, capture_output=True)
        lines = subprocess.run([str(exe)], input=gzip.decompress(stream.read_bytes()), capture_output=True, timeout=900,
                               check=True).stdout.decode().splitlines()
        return [line.split(" ", 2) for line in lines[:-1]], int(lines[-1].split()[1])

    def run_replay(self, carrier: str, fixed: list[str], zero: list[str]):
        stream = REPO / "bots" / "p_only" / "tests" / "fixtures" / "stream_n04.txt.gz"
        with tempfile.TemporaryDirectory() as tmp:
            off, t_off = self.outputs(carrier, fixed, stream, tmp)
            nil, t_nil = self.outputs(carrier, fixed + ["-DMOD_FUSION_V=1", *zero], stream, tmp)
            on, t_on = self.outputs(carrier, fixed + ["-DMOD_FUSION_V=1"], stream, tmp)
        if t_off or t_nil or t_on:
            self.skipTest(f"a deadline truncated the fixed-work search ({t_off}, {t_nil}, {t_on}): equivalence not testable here")
        self.assertEqual(len(off), 400)
        self.assertEqual([r[2] for r in off], [r[2] for r in nil], "V at weight 0 changed a reply")
        self.assertEqual([r[1] for r in off], [r[1] for r in nil])
        self.assertEqual([r[1] for r in off], [r[1] for r in on], "turning V on changed the network's predictions")
        changed = sum(a[2] != b[2] for a, b in zip(off, on))
        print(f"\n[replay] {carrier}: V on changed {changed} of {len(off)} replies on the 4-team game", file=sys.stderr)

    def test_c08(self):
        self.run_replay("C08", ["-DPONLY_FIXED_SAMPLES=64"], ["-DPONLY_V_REFINE=0", "-DPONLY_V_BEAM=0"])

    def test_v12(self):
        self.run_replay("V12", [], ["-DPONLY_V_ROLLOUT=0"])


@unittest.skipUnless(CXX, "no C++ compiler")
class Contract(unittest.TestCase):
    """History, distributions, orders, coherent states and the engine comparison on recorded games (contract_test.cpp)."""

    def run_contract(self, carrier: str):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / f"contract_{carrier}"
            subprocess.run([CXX, "-std=c++20", "-O2", f"-I{REPO / 'inference'}", f"-I{REPO / 'bots' / 'p_only' / 'include'}",
                            f"-I{BOTS}", "-DFUSION_TIER_M", f"-DCARRIER_{carrier}", str(REPO / "bots" / "p_only" / "tests" / "contract_test.cpp"),
                            "-o", str(exe)], check=True, capture_output=True)
            for stream in sorted((REPO / "bots" / "p_only" / "tests" / "fixtures").glob("stream_n*.txt.gz")):
                orders = Path(tmp) / stream.name.replace("stream_", "orders_").removesuffix(".gz")
                orders.write_bytes(gzip.decompress((stream.parent / stream.name.replace("stream_", "orders_")).read_bytes()))
                r = subprocess.run([str(exe), str(orders)], input=gzip.decompress(stream.read_bytes()), capture_output=True, timeout=600)
                out = r.stdout.decode()
                with self.subTest(carrier=carrier, stream=stream.name):
                    self.assertEqual(r.returncode, 0, out + r.stderr.decode()[-2000:])
                    self.assertIn(" 0 failures", out)
                    self.assertIn(" 0 inconsistent", out)
                    steps = int(re.search(r"engine steps (\d+)", out).group(1))
                    self.assertGreater(steps, 300, "too few turns compared with the engine")
                    if "n04" in stream.name:
                        branches = int(re.search(r"(\d+) branch views", out).group(1))
                        self.assertGreater(branches, 0, "the stream never exercised a simulated branch")

    def test_c08(self):
        self.run_contract("C08")

    def test_v12(self):
        self.run_contract("V12")


if __name__ == "__main__":
    unittest.main()
