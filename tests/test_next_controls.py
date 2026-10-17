"""T0075's frozen controls: the agent_v2 general-only and init-only arms, the original-Meowmind NN_SEARCH wrappers, the
arms manifest and the preflight telemetry adapter (bots/agent_v2/controls/README.md,
bots/meowmind_p/attribution/README.md).

The C++ tests need a compiler (skipped otherwise). They bundle the real variant files with bots/p_only/tools/bundle.py,
compile them and replay the recorded seat streams bots/p_only/tests/fixtures/stream_n02/n04/n15. The INT16 check
compares the integer actor the bundle computes on those streams' observations with the FP32 weights it was exported
from. Nothing here plays a game, consumes a seed, or measures or claims timing or strength.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fusion import runner

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
from bundle import bundle  # noqa: E402

from tools.screens.next_controls import manifest as M  # noqa: E402
from tools.screens.next_controls import telemetry as T  # noqa: E402
from training.meow_rl import policy as P  # noqa: E402

FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
GAMES = ("n02", "n04", "n15")
CXX = runner.find_compiler()
FLAGS = ["-std=c++20", "-O2", "-w"]
CONTROLS = REPO / "bots" / "agent_v2" / "controls"
ATTRIBUTION = REPO / "bots" / "meowmind_p" / "attribution"
PROBE = REPO / "tools" / "screens" / "next_controls" / "probe.hpp"
INIT_HEADER = REPO / "inference" / "agent_v2_controls" / "init_actor.hpp"
INIT_MANIFEST = REPO / "inference" / "agent_v2_controls" / "MANIFEST.json"
RL_MANIFEST = REPO / "inference" / "count_specialists_rl" / "MANIFEST.json"
OUT_EXP = 16                     # agent_v2_rl_csp's DynamicActor and every T0068 header
TEMPERATURE = 1.05
# The deployed T0068 exports agree with FP32 on 95.0-96.8% of greedy joints (mean |dlogp| 0.26-0.33) on their
# calibration rows; these bounds only catch a broken export or a routing to the wrong weights.
MIN_GREEDY, MAX_MEAN_LOGP = 0.90, 0.45
OBS = np.dtype([("feats", "<f4", (3, 5, 112)), ("mask", "u1", (125,)), ("scores", "<i8", (3, 5))])


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


def compile_cpp(src: Path, exe: Path) -> Path:
    out = exe.with_suffix(runner.EXE)
    r = subprocess.run([CXX, *FLAGS, *(["-static"] if runner.WINDOWS else []), str(src), "-o", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"compile failed for {src.name}:\n{r.stderr[-3000:]}")
    return out


def bundle_to(variant: Path, out: Path) -> Path:
    text, _ = bundle(variant)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


def assert_valid_orders(tc: unittest.TestCase, game: str, text: str, lines: list[str]):
    """400 complete orders: the turn, then per own unit a move in SUDLR and an action (carriers wait, targets are enemies)."""
    tc.assertEqual(len(lines), 400, game)
    states = [b for b in text.split("END\n") if "TURN" in b]
    me = int(text.split()[2])
    for turn, (line, block) in enumerate(zip(lines, states), 1):
        tok = line.split()
        tc.assertEqual((len(tok), tok[0]), (7, str(turn)), (game, line))
        units = {int(u.split()[1]): u.split() for u in block.split("\n") if u.startswith("U ")}
        for k in range(3):
            mv, act = tok[1 + 2 * k], tok[2 + 2 * k]
            tc.assertIn(mv, "SUDLR")
            if act == "X":
                continue                                                   # a drop (Meowmind only)
            if int(units[3 * me + k][6]) >= 0:
                tc.assertEqual(act, "-", (game, turn, k))
            else:
                tc.assertTrue(act == "P" or (act.isdigit() and int(act) // 3 != me), (game, turn, act))


def int16_agreement(actor: dict, obs: np.ndarray) -> dict:
    """The bundle's integer joint (acc / 2^outExp / T) against the FP32 actor on the same observations."""
    mask = obs["mask"].astype(bool)
    lp32 = P.joint_log_probs(actor, obs["feats"], mask).astype(np.float64)
    s = obs["scores"]
    acc = s[:, 0, P.JA] + s[:, 1, P.JB] + s[:, 2, P.JC]
    z = np.where(mask, np.ldexp(acc.astype(np.float64), -OUT_EXP) / TEMPERATURE, -np.inf)
    top = z.max(axis=-1, keepdims=True)
    lpq = z - top - np.log(np.exp(z - top).sum(axis=-1, keepdims=True))
    greedy = np.argmax(np.where(mask, acc, np.iinfo(np.int64).min), axis=-1)
    err = np.abs(lp32[mask] - lpq[mask])
    return {"observations": len(obs), "greedy_agreement": float(np.mean(greedy == lp32.argmax(axis=-1))),
            "mean_abs_logp_error": float(err.mean()), "max_abs_logp_error": float(err.max())}


class Manifest(unittest.TestCase):
    def test_the_manifest_pins_the_current_files(self):
        self.assertEqual(M.main(["--check"]), 0, "run python tools/screens/next_controls/manifest.py")

    def test_every_arm_is_pinned(self):
        doc = json.loads(M.OUT.read_text(encoding="utf-8"))
        self.assertEqual(set(doc["arms"]), {"agent_v2_rl_general_csp", "agent_v2_rl_init_csp", "meowmind_search_on",
                                            "meowmind_search_off"})
        for name, a in doc["arms"].items():
            self.assertRegex(a["source_sha256"], "^[0-9a-f]{64}$", name)
            if a["status"] == "ready":
                self.assertRegex(a["bundle_sha256"], "^[0-9a-f]{64}$", name)
                self.assertEqual(a["inlined"][0]["path"], a["source"], name)
            else:
                self.assertEqual((name, a["status"]), ("agent_v2_rl_init_csp", "awaiting_init_export"))
                self.assertNotIn("bundle_sha256", a)                       # nothing filled in before the export
        g = doc["arms"]["agent_v2_rl_general_csp"]["model"]
        rl = json.loads(RL_MANIFEST.read_text(encoding="utf-8"))["models"]["general"]
        self.assertEqual((g["digest"], g["header_sha256"]), (rl["digest"], rl["header_sha256"]))

    def test_the_meowmind_wrappers_differ_only_in_nn_search(self):
        doc = json.loads(M.OUT.read_text(encoding="utf-8"))["meowmind"]
        self.assertTrue(doc["original_unmodified"])
        self.assertEqual(doc["on_off_bundle_diff"], ["-#define NN_SEARCH 1", "+#define NN_SEARCH 0"])
        on = (ATTRIBUTION / "meowmind_search_on.cpp").read_text(encoding="utf-8").splitlines()
        off = (ATTRIBUTION / "meowmind_search_off.cpp").read_text(encoding="utf-8").splitlines()
        self.assertEqual([i for i, (a, b) in enumerate(zip(on, off)) if a != b], [on.index("#define NN_SEARCH 1")])
        self.assertEqual(len(on), len(off))
        # The bundle carries the original source byte for byte between its markers.
        text, files = bundle(ATTRIBUTION / "meowmind_search_on.cpp")
        text = M.canonical(text)
        self.assertEqual([M.rel(f) for f in files], ["bots/meowmind_p/attribution/meowmind_search_on.cpp", M.MEOWMIND])
        begin, end = f"// [bundle] begin {M.MEOWMIND}\n", f"\n// [bundle] end {M.MEOWMIND}"
        inner = text[text.index(begin) + len(begin):text.index(end)]
        self.assertEqual(inner, (REPO / M.MEOWMIND).read_text(encoding="utf-8"))

    def test_the_lists_are_dedicated_and_not_defaults(self):
        t68 = {v["name"] for v in json.loads((REPO / "bots" / "agent_v2" / "variants.json").read_text(encoding="utf-8"))["variants"]}
        profile = (REPO / "profiles" / "agent_v2.json").read_text(encoding="utf-8")
        for listed in M.LISTS:
            for v in json.loads((REPO / listed).read_text(encoding="utf-8"))["variants"]:
                self.assertNotIn(v["name"], t68)
                self.assertNotIn(v["name"], profile)
                self.assertEqual(T.format_for(v), v["telemetry"])
                self.assertTrue((REPO / v["source"]).is_file(), v["source"])


class Telemetry(unittest.TestCase):
    IDENT = ("AGENTV2 carrier agent_v2_v1 actions agent_v2_actions_v1 resolver assign_v1/dp_v1 scenarios 8 actor "
             "count_specialists_rl_general_only digest abc features meow_policy112_v1/meow_public416_v1 predictor P "
             "model csp_n02 search 0 greedy 0")

    def test_agent_v2_identity_and_last_counters(self):
        lines = [self.IDENT, "AGENTV2 turns 1 refused 0 inconsistent 0 search_used 0 search_fallback 0 leaves 0",
                 "AGENTV2 turns 400 refused 0 inconsistent 2 search_used 0 search_fallback 0 leaves 0"]
        p = T.parse("agent_v2", lines)
        self.assertEqual(p["identity"]["actor"], "count_specialists_rl_general_only")
        self.assertEqual((p["identity"]["model"], p["identity"]["search"]), ("csp_n02", 0))
        self.assertEqual((p["counters"]["turns"], p["counters"]["inconsistent"]), (400, 2))
        # The no-search line has no search fields: they stay absent, never zero.
        self.assertNotIn("trajectories", p["counters"])
        self.assertIn("trajectories", p["absent"])
        self.assertEqual(T.complete("agent_v2", p, 400), (True, []))
        self.assertFalse(T.complete("agent_v2", p, 401)[0])
        self.assertNotIn("trajectories", T.work_totals("agent_v2", [p, p]))

    def test_missing_agent_v2_counters_are_missing(self):
        p = T.parse("agent_v2", [self.IDENT])
        self.assertIsNone(p["counters"])
        self.assertIn("turns", p["absent"])
        self.assertFalse(T.complete("agent_v2", p, 400)[0])

    def test_meowmind_reports_no_counters_and_none_are_invented(self):
        p = T.parse("meowmind", [])
        self.assertEqual((p["identity"], p["counters"]), (None, None))
        self.assertEqual(set(p["absent"]), set(T.FIELDS))
        self.assertIsNone(T.work_totals("meowmind", [p]))
        self.assertFalse(T.complete("meowmind", p, 400)[0])               # the runner's turn count is required
        self.assertTrue(T.complete("meowmind", {**p, "runner_turns": 400}, 400)[0])
        self.assertFalse(T.complete("meowmind", {**p, "runner_turns": 399}, 400)[0])

    def test_formats(self):
        self.assertEqual(T.format_for("meowmind_search_off"), "meowmind")
        self.assertEqual(T.format_for("agent_v2_rl_general_csp"), "agent_v2")
        with self.assertRaises(ValueError):
            T.format_for({"name": "x", "telemetry": "other"})


@unittest.skipUnless(CXX, "needs a C++ compiler")
class AgentV2Controls(unittest.TestCase):
    """G and I: the bundled control, a probe over the same bundle (routing, observations), three recorded streams."""

    ARMS = {"agent_v2_rl_general_csp": "fusion::models::rl_general::kActor",
            "agent_v2_rl_init_csp": "fusion::models::rl_init::kActor"}

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="next-controls-"))
        cls.probes = {}
        todo = [n for n in cls.ARMS if n != "agent_v2_rl_init_csp" or INIT_HEADER.is_file()]

        def build(name):
            b = bundle_to(CONTROLS / f"{name}.cpp", cls.tmp / f"{name}.bundle.cpp")
            src = cls.tmp / f"{name}.probe.cpp"
            src.write_text("\n".join(["#define main agent_v2_variant_main", f'#include "{b.as_posix()}"', "#undef main",
                                      f"#define CONTROL_EXPECTED_ACTOR {cls.ARMS[name]}",
                                      f'#include "{PROBE.as_posix()}"', ""]), encoding="utf-8")
            return name, compile_cpp(src, cls.tmp / name)
        with ThreadPoolExecutor(2) as ex:
            cls.probes = dict(ex.map(build, todo))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def probe(self, name: str):
        if name not in self.probes:
            self.skipTest("inference/agent_v2_controls/init_actor.hpp not exported yet "
                          "(tools/screens/next_controls/export_init_actor.py on the host)")
        return self.probes[name]

    def actor(self, name: str) -> tuple[dict, str]:
        if name == "agent_v2_rl_general_csp":
            m = json.loads(RL_MANIFEST.read_text(encoding="utf-8"))["models"]["general"]
        else:
            m = json.loads(INIT_MANIFEST.read_text(encoding="utf-8"))["model"]
        z = dict(np.load(REPO / m["npz_path"]))
        return P.check_actor({k: z[k] for k in P.ACTOR_SHAPES}), m["digest"]

    def check_routing(self, name: str):
        r = subprocess.run([str(self.probe(name)), "routing"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout)
        rows = [line.split() for line in r.stdout.splitlines() if line.startswith("ROUTE")]
        self.assertEqual([int(t[2]) for t in rows], list(range(17)))         # counts 0..16, beyond 2..15 too
        for t in rows:
            self.assertEqual((t[4], t[6], t[8]), ("1", str(OUT_EXP), str(OUT_EXP)), t)

    def check_streams(self, name: str):
        exe = self.probe(name)
        actor, digest = self.actor(name)
        obs = []
        for game in GAMES:
            text = stream(game)
            play = subprocess.run([str(exe), "play"], input=text, capture_output=True, text=True, timeout=900)
            self.assertEqual(play.returncode, 0, play.stderr[-500:])
            assert_valid_orders(self, game, text, play.stdout.splitlines())
            parsed = T.parse("agent_v2", play.stderr.splitlines())
            self.assertEqual(T.complete("agent_v2", parsed, 400), (True, []), game)
            self.assertEqual((parsed["counters"]["refused"], parsed["counters"]["inconsistent"]), (0, 0), game)
            ident = parsed["identity"]
            self.assertEqual((ident["actor"], ident["digest"]), (name.replace("agent_v2_rl_", "count_specialists_rl_")
                                                                .replace("_csp", "_only"), digest))
            self.assertEqual((ident["predictor"], ident["model"], ident["search"], ident["greedy"]),
                             ("P", f"csp_{game}", 0, 0))
            self.assertEqual(ident["resolver"], "assign_v1/dp_v1")
            dump = subprocess.run([str(exe), "dump"], input=text.encode(), capture_output=True, timeout=900)
            self.assertEqual(dump.returncode, 0)
            self.assertIn(f"DUMP teams {int(game[1:])} answered 400 refused 0", dump.stderr.decode())
            obs.append(np.frombuffer(dump.stdout, dtype=OBS))
        report = int16_agreement(actor, np.concatenate(obs))
        self.assertEqual(report["observations"], 1200)
        self.assertGreaterEqual(report["greedy_agreement"], MIN_GREEDY, report)
        self.assertLessEqual(report["mean_abs_logp_error"], MAX_MEAN_LOGP, report)
        return report

    def test_general_only_routes_every_count_to_rl_general(self):
        self.check_routing("agent_v2_rl_general_csp")

    def test_general_only_replays_the_streams_with_its_int16_actor(self):
        self.check_streams("agent_v2_rl_general_csp")

    def test_init_only_routes_every_count_to_the_init_actor(self):
        self.check_routing("agent_v2_rl_init_csp")
        m = json.loads(INIT_MANIFEST.read_text(encoding="utf-8"))["model"]
        self.assertEqual((m["out_exp"], m["scheme"], m["in_bits"], m["temperature"]),
                         (OUT_EXP, "meow_rl_reference16_v1", 12, TEMPERATURE))

    def test_init_only_replays_the_streams_with_its_int16_actor(self):
        self.check_streams("agent_v2_rl_init_csp")


@unittest.skipUnless(CXX, "needs a C++ compiler")
class MeowmindAttribution(unittest.TestCase):
    """ON and OFF: the bundled wrappers compile, replay the streams, and ON preprocesses to the original source."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="meowmind-attribution-"))

        def build(arm):
            b = bundle_to(ATTRIBUTION / f"meowmind_search_{arm}.cpp", cls.tmp / f"{arm}.cpp")
            return arm, compile_cpp(b, cls.tmp / arm)
        with ThreadPoolExecutor(2) as ex:
            cls.exe = dict(ex.map(build, ("on", "off")))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def preprocess(self, path: Path) -> str:
        r = subprocess.run([CXX, "-std=c++20", "-E", "-P", str(path)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        return r.stdout

    def test_on_is_the_original_and_off_is_not(self):
        original = self.preprocess(REPO / M.MEOWMIND)
        self.assertEqual(self.preprocess(self.tmp / "on.cpp"), original)
        self.assertNotEqual(self.preprocess(self.tmp / "off.cpp"), original)

    def test_both_replay_the_streams_with_valid_orders_and_no_counters(self):
        for arm, exe in self.exe.items():
            for game in GAMES:
                text = stream(game)
                r = subprocess.run([str(exe)], input=text, capture_output=True, text=True, timeout=900)
                self.assertEqual(r.returncode, 0, (arm, game, r.stderr[-500:]))
                lines = r.stdout.splitlines()
                assert_valid_orders(self, game, text, lines)
                parsed = T.parse("meowmind", r.stderr.splitlines())
                self.assertIsNone(parsed["counters"])
                self.assertTrue(T.complete("meowmind", {**parsed, "runner_turns": len(lines)}, 400)[0])


if __name__ == "__main__":
    unittest.main()
