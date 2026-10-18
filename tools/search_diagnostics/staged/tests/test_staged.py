"""Tests of the staged search (task T0087): inference/agent_v2/search.hpp's StagedSearch and its bot.

    python -m unittest discover -s tools/search_diagnostics/staged/tests -t .

They live beside the tool because tests/ is reserved by other tasks. Needs a C++ compiler (skipped otherwise); the
recorded seat streams of bots/p_only/tests/fixtures drive the binaries. Fixed-work mode throughout: these are
correctness checks, never timing or strength evidence.
"""
from __future__ import annotations

import gzip
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
from fusion import runner  # noqa: E402

CXX = runner.find_compiler()
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
ACTORS = REPO / "inference" / "meow_rl" / "aligned" / "p2_large"
VARIANTS = REPO / "bots" / "agent_v2" / "staged_search" / "variants"
FLAGS = ["-std=c++20", "-O2", "-w"]
TREE = ["-DFUSION_TIER_P2GENERAL_VM=1", "-DFUSION_MAX_WIDTH=1024", "-DAGENT_V2_SEARCH_CANDIDATES=8",
        "-DAGENT_V2_SEARCH_SAMPLES=4", "-DAGENT_V2_SEARCH_TAU=0.002"]
SCHEDULES = {"rounds": [], "widen": ["-DAGENT_V2_STAGED_SCHEDULE={4,1},{4,2},{8,2},{8,3},{8,4}"]}
GAMES = ("n02", "n04", "n15")
COUNTERS = re.compile(r"(\w+) (-?\d+)")


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


def short_rules(text: str, turns: int) -> str:
    """The stream with INIT's turn count, hp and damage changed, cut after the new last turn."""
    head, rest = text.split("\n", 1)
    tok = head.split()
    tok[4], tok[5], tok[6] = str(turns), "150", "40"
    keep = []
    for block in (" ".join(tok) + "\n" + rest).split("END\n"):
        m = re.search(r"TURN (\d+)", block)
        if not block.strip() or (m and int(m.group(1)) > turns):
            break
        keep.append(block)
    return "END\n".join(keep) + "END\n"


def last_counters(err: str) -> dict[str, int]:
    return {k: int(v) for k, v in COUNTERS.findall(err.strip().splitlines()[-1])}


@unittest.skipUnless(CXX, "no C++ compiler")
class Harness(unittest.TestCase):
    """tools/search_diagnostics/staged/staged_check.cpp, both frozen schedules."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.exe = {}
        for name, extra in SCHEDULES.items():
            exe = d / f"staged_check_{name}{runner.EXE}"
            r = subprocess.run([CXX, *FLAGS, "-I", str(REPO / "inference"), "-I", str(REPO / "tools" / "cpp" / "meow_rl"), *TREE,
                                *extra, f'-DACTOR_HEADER="{(ACTORS / "actor.hpp").as_posix()}"',
                                f'-DCRITIC_HEADER="{(ACTORS / "critic_fp32.hpp").as_posix()}"',
                                str(REPO / "tools" / "search_diagnostics" / "staged" / "staged_check.cpp"), "-o", str(exe)],
                               capture_output=True, text=True)
            if r.returncode:
                raise AssertionError(r.stderr[-3000:])
            cls.exe[name] = exe
        cls.streams = {}
        for g in GAMES:
            p = d / f"stream_{g}.txt"
            p.write_text(stream(g), encoding="utf-8")
            cls.streams[g] = p
        p = d / "stream_rules.txt"
        p.write_text(short_rules(stream("n04"), 125), encoding="utf-8")
        cls.streams["rules"] = p

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_check(self, schedule: str, *args) -> str:
        r = subprocess.run([str(self.exe[schedule]), *map(str, args)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(r.stdout.startswith("ok "), r.stdout)
        return r.stdout

    def test_schedules_are_validated(self):
        self.assertIn("compiled 8:1 8:2 8:3 8:4", self.run_check("rounds", "schedule"))
        self.assertIn("compiled 4:1 4:2 8:2 8:3 8:4", self.run_check("widen", "schedule"))

    def test_the_full_schedule_is_search_and_branches_are_isolated(self):
        for s in SCHEDULES:
            for g in GAMES:
                with self.subTest(schedule=s, game=g):
                    self.run_check(s, "match", self.streams[g], 1, 2, 5, 120, 333, 399, 400)

    def test_every_interruption_keeps_the_last_completed_checkpoint(self):
        for s in SCHEDULES:
            for g in GAMES:
                with self.subTest(schedule=s, game=g):
                    self.assertIn("ok interrupt runs", self.run_check(s, "interrupt", self.streams[g], 5, 120, 333))

    def test_a_nan_never_overrides_the_actor(self):
        for s in SCHEDULES:
            with self.subTest(schedule=s):
                self.run_check(s, "nan", self.streams["n04"], 120)

    def test_nondefault_rules_and_terminal_leaves(self):
        for s in SCHEDULES:
            with self.subTest(schedule=s):
                out = self.run_check(s, "match", self.streams["rules"], 5, 123, 124, 125)
                self.assertGreater(int(re.search(r"terminal (\d+)", out).group(1)), 0, out)
                self.run_check(s, "interrupt", self.streams["rules"], 124, 125)

    def test_fixed_work_runs_repeat_exactly(self):
        out = self.run_check("rounds", "repeat", self.streams["n02"])
        self.assertEqual(out.count("ok repeat"), 4, out)


@unittest.skipUnless(CXX, "no C++ compiler")
class Bot(unittest.TestCase):
    """The frozen variants through bots/p_only/tools/bundle.py, as the host builds them."""

    @classmethod
    def setUpClass(cls):
        from bundle import bundle  # noqa: PLC0415
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bins = {}
        for v in sorted(VARIANTS.glob("*.cpp")):
            for mode, defines in (("timed", []), ("fixed", ["-DAGENT_V2_STAGED_FIXED_WORK=13"])):
                text, _ = bundle(v)
                src = Path(cls.tmp.name) / f"{v.stem}_{mode}.cpp"
                src.write_text(text, encoding="utf-8")
                exe = src.with_suffix(runner.EXE)
                subprocess.run([CXX, *FLAGS, *defines, *(["-static"] if runner.WINDOWS else []), str(src), "-o", str(exe)],
                               check=True, capture_output=True)
                cls.bins[(v.stem, mode)] = exe

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def play(self, exe: Path, text: str) -> tuple[list[str], str]:
        r = subprocess.run([str(exe)], input=text, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return r.stdout.splitlines(), r.stderr

    def test_replies_are_complete_and_counters_add_up(self):
        text = stream("n04")
        for (name, mode), exe in self.bins.items():
            with self.subTest(variant=name, mode=mode):
                lines, err = self.play(exe, text)
                self.assertEqual(len(lines), 400)
                for turn, line in enumerate(lines, 1):
                    tok = line.split()
                    self.assertEqual((len(tok), tok[0]), (7, str(turn)), line)
                self.assertIn("staged schedule", err.splitlines()[0])
                self.assertIn(f"fixed_work {13 if mode == 'fixed' else -1}", err.splitlines()[0])
                c = last_counters(err)
                self.assertEqual((c["refused"], c["inconsistent"]), (0, 0))
                self.assertEqual(c["search_used"] + c["search_fallback"], c["turns"])
                self.assertEqual(c["overrides"] + c["baseline_kept"], c["search_used"])
                self.assertLessEqual(c["gate_kept"], c["baseline_kept"])
                stops = sum(c[k] for k in ("refuse_setup", "refuse_trajectory", "refuse_expand", "refuse_critic",
                                           "refuse_late", "stopped_work_cap", "complete"))
                self.assertEqual(stops, c["turns"])
                self.assertGreaterEqual(c["eff_scenarios"], c["search_used"])          # >= 1 scenario per decision
                if mode == "fixed":
                    self.assertEqual(c["stopped_work_cap"] + c["complete"], c["turns"])  # the clock never stops it

    def test_fixed_work_bots_are_deterministic(self):
        text = stream("n02")
        for (name, mode), exe in self.bins.items():
            if mode != "fixed":
                continue
            with self.subTest(variant=name):
                a, ea = self.play(exe, text)
                b, eb = self.play(exe, text)
                self.assertEqual(a, b)
                timing = re.compile(r"max_\w+_us")
                ca = {k: v for k, v in last_counters(ea).items() if not timing.match(k)}
                cb = {k: v for k, v in last_counters(eb).items() if not timing.match(k)}
                self.assertEqual(ca, cb)


if __name__ == "__main__":
    unittest.main()
