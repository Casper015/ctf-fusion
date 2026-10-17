"""The agent_v2 search override gate (task T0076, docs/AGENT_V2_SEARCH_GATE.md).

Needs a C++ compiler (skipped otherwise). Recorded seat streams (bots/p_only/tests/fixtures/stream_n*.txt.gz) drive
the binaries. The behaviour builds use the reference16 critic only to keep compile times down: the gate and the
fallbacks do not depend on the critic's arithmetic. Every tree completes in those builds (a 10-minute deadline), so the
counters are deterministic. Nothing here measures time or strength.
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fusion import runner

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
CXX = runner.find_compiler()
ACTORS = REPO / "inference" / "meow_rl" / "aligned" / "p2_large"
GATE = REPO / "bots" / "agent_v2" / "search_gate"
DIAG = GATE / "diagnostics" / "DIAGNOSTICS.json"
FLAGS = ["-std=c++20", "-O2", "-w"]
UNBOUNDED = ["-DAGENT_V2_SEARCH_DEADLINE_MS=600000", "-DAGENT_V2_SEARCH_FIRST_MS=600000"]
WIDE = ["-DAGENT_V2_SEARCH_CANDIDATES=8", "-DAGENT_V2_SEARCH_SAMPLES=4"]


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


def compile_bundle(source: Path, exe: Path, defines=()) -> Path:
    from bundle import bundle  # noqa: PLC0415
    text, _ = bundle(source)
    src = exe.with_suffix(".cpp")
    src.write_text(text, encoding="utf-8")
    out = exe.with_suffix(runner.EXE)
    subprocess.run([CXX, *FLAGS, *defines, *(["-static"] if runner.WINDOWS else []), str(src), "-o", str(out)],
                   check=True, capture_output=True)
    return out


def counters(err: str) -> dict[str, str]:
    last = err.strip().splitlines()[-1].split()
    return dict(zip(last[1::2], last[2::2]))


def frozen() -> dict | None:
    return json.loads(DIAG.read_text(encoding="utf-8")).get("frozen") if DIAG.exists() else None


@unittest.skipUnless(CXX, "no C++ compiler")
class SearchGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="agent-v2-gate-"))
        probe = cls.tmp / "probe.cpp"                                   # the gate bot with a light critic
        probe.write_text("#define FUSION_MAX_WIDTH 1024\n#define FUSION_TIER_PXXL_VM 1\n"
                         f'#include "{(ACTORS / "actor.hpp").as_posix()}"\n'
                         f'#include "{(ACTORS / "critic_reference16.hpp").as_posix()}"\n'
                         f'#include "{(GATE / "bot.hpp").as_posix()}"\n', encoding="utf-8")
        tau = frozen()["tau"] if frozen() and "tau" in frozen() else 0.01
        cls.tau = tau
        jobs = {
            "direct": (REPO / "bots" / "agent_v2" / "variants" / "agent_v2_p2xxl_direct.cpp", []),
            "ungated": (probe, [*WIDE, *UNBOUNDED, "-DAGENT_V2_SEARCH_TAU=0"]),
            "gated": (probe, [*WIDE, *UNBOUNDED, f"-DAGENT_V2_SEARCH_TAU={tau!r}"]),
            "never": (probe, [*WIDE, *UNBOUNDED, "-DAGENT_V2_SEARCH_TAU=1e9"]),
            "capped": (probe, [*WIDE, *UNBOUNDED, "-DAGENT_V2_SEARCH_TAU=0", "-DAGENT_V2_SEARCH_TEST_WORK_CAP=4"]),
            "late": (probe, [*WIDE, "-DAGENT_V2_SEARCH_DEADLINE_MS=0", "-DAGENT_V2_SEARCH_FIRST_MS=0", "-DAGENT_V2_SEARCH_TAU=0"]),
            "narrow0": (probe, [*UNBOUNDED, "-DAGENT_V2_SEARCH_TAU=0"]),
        }
        check = REPO / "tools" / "cpp" / "agent_v2" / "check.cpp"

        def check_build(name, defines):
            exe = cls.tmp / f"{name}{runner.EXE}"
            subprocess.run([CXX, *FLAGS, "-I", str(REPO / "inference"), "-I", str(REPO / "tools" / "cpp" / "meow_rl"),
                            "-DFUSION_TIER_PXXL_VM=1", *defines, f'-DACTOR_HEADER="{(ACTORS / "actor.hpp").as_posix()}"',
                            f'-DCRITIC_HEADER="{(ACTORS / "critic_reference16.hpp").as_posix()}"', str(check), "-o", str(exe)],
                           check=True, capture_output=True)
            return exe

        with ThreadPoolExecutor(max_workers=8) as pool:
            bots = {k: pool.submit(compile_bundle, src, cls.tmp / k, d) for k, (src, d) in jobs.items()}
            checks = {"check": pool.submit(check_build, "check", []), "check_wide": pool.submit(check_build, "check_wide", WIDE)}
            cls.exe = {k: f.result() for k, f in bots.items()}
            cls.check = {k: f.result() for k, f in checks.items()}
        cls.text = stream("n02")
        cls.runs = {k: cls.play(exe) for k, exe in cls.exe.items()}

    @classmethod
    def play(cls, exe: Path) -> tuple[list[str], str]:
        r = subprocess.run([str(exe)], input=cls.text, capture_output=True, text=True, timeout=1800)
        if r.returncode:
            raise AssertionError(r.stderr[-500:])
        return r.stdout.splitlines(), r.stderr

    def test_gate_unit_cases(self):
        out = subprocess.run([str(self.check["check"]), "gate"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertRegex(out.stdout, r"^ok gate cases \d+")

    def test_identity_names_the_gate_and_the_tree(self):
        m = re.search(r"gate tau (\S+) candidates 8 samples 4\n", self.runs["gated"][1])
        self.assertIsNotNone(m)
        self.assertEqual(float(m.group(1)), self.tau)
        self.assertIn("gate tau 0 candidates 4 samples 2\n", self.runs["narrow0"][1])

    def test_every_tree_completes_in_the_unbounded_builds(self):
        for k in ("ungated", "gated", "never", "narrow0"):
            c = counters(self.runs[k][1])
            self.assertEqual((c["search_used"], c["search_fallback"], c["refused"]), ("400", "0", "0"), k)

    def test_the_ungated_tree_overrides_and_nothing_is_kept_out(self):
        c = counters(self.runs["ungated"][1])
        self.assertGreater(int(c["overrides"]), 0)                   # not a vacuous comparison
        self.assertEqual(c["gate_kept"], "0")
        self.assertNotEqual(self.runs["ungated"][0], self.runs["direct"][0])

    def test_the_gate_only_removes_overrides(self):
        u, g = counters(self.runs["ungated"][1]), counters(self.runs["gated"][1])
        self.assertEqual(int(g["overrides"]) + int(g["gate_kept"]), int(u["overrides"]))   # the same trees (a recorded stream)
        self.assertLessEqual(int(g["overrides"]), int(u["overrides"]))

    def test_margins_at_or_below_tau_keep_the_actors_complete_order(self):
        lines, err = self.runs["never"]
        self.assertEqual(lines, self.runs["direct"][0])
        c = counters(err)
        self.assertEqual(c["overrides"], "0")
        self.assertEqual(c["gate_kept"], counters(self.runs["ungated"][1])["overrides"])

    def test_incomplete_work_keeps_the_actors_complete_order(self):
        lines, err = self.runs["capped"]
        self.assertEqual(lines, self.runs["direct"][0])
        c = counters(err)
        self.assertEqual(c["overrides"], "0")
        self.assertEqual(int(c["search_used"]) + int(c["refuse_trajectory"]), 400)
        self.assertGreater(int(c["refuse_trajectory"]), 0)

    def test_refusals_keep_the_actors_complete_order(self):
        lines, err = self.runs["late"]
        self.assertEqual(lines, self.runs["direct"][0])
        c = counters(err)
        self.assertEqual((c["search_used"], c["search_fallback"], c["overrides"]), ("0", "400", "0"))

    def test_the_wide_search_never_writes_back(self):
        for g in ("n02", "n04"):
            p = self.tmp / f"stream_{g}.txt"
            p.write_text(stream(g), encoding="utf-8")
            out = subprocess.run([str(self.check["check_wide"]), "isolation", str(p)], capture_output=True, text=True).stdout
            self.assertTrue(out.startswith("ok isolation"), (g, out))
            self.assertGreater(int(re.search(r"branches (\d+)", out).group(1)), 0, g)


class FrozenVariants(unittest.TestCase):
    """The delivered variants carry exactly the breadth and tau that the diagnostics froze."""

    def test_variants_match_the_frozen_diagnostics(self):
        f = frozen()
        if f is None or "blocked" in f:
            self.skipTest("no frozen gate")
        doc = json.loads((GATE / "variants.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["include"], ["bots/agent_v2/variants.json"])
        names = {v["name"]: v for v in doc["variants"]}
        self.assertEqual(len(names), 2)
        for v in names.values():
            text = (REPO / v["source"]).read_text(encoding="utf-8")
            defs = dict(re.findall(r"^#define (AGENT_V2_SEARCH_\w+) (\S+)", text, re.M))
            self.assertEqual(int(defs["AGENT_V2_SEARCH_CANDIDATES"]), f["candidates"], v["name"])
            self.assertEqual(int(defs["AGENT_V2_SEARCH_SAMPLES"]), f["samples"], v["name"])
            self.assertIn("critic_fp32.hpp", text)
            self.assertIn("FUSION_TIER_PXXL_VM 1", text)
            self.assertNotIn("AGENT_V2_SEARCH_TEST_WORK_CAP", text)
        taus = sorted(float(dict(re.findall(r"^#define (AGENT_V2_SEARCH_TAU) (\S+)", (REPO / v["source"]).read_text(), re.M))
                                 ["AGENT_V2_SEARCH_TAU"]) for v in names.values())
        self.assertEqual(taus, sorted([0.0, f["tau"]]))

    def test_bundles_are_single_files(self):
        from bundle import bundle  # noqa: PLC0415
        for f in (GATE / "variants").glob("*.cpp"):
            text, _ = bundle(f)
            self.assertEqual([l for l in text.split("\n") if l.lstrip().startswith('#include "')], [], f.name)
            self.assertNotIn("meowmind_reference_main", text, f.name)


if __name__ == "__main__":
    unittest.main()
