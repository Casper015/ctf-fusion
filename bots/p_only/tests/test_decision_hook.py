"""Decision shortlist hook of the family carriers (task T0036, bots/p_only/include/decision_hook.hpp).

  * off is off: with PONLY_DECISION 0 (the default) the family carriers reply as the production carriers, and the
    log-only build (PONLY_DECISION 1) writes exactly the same orders on recorded games;
  * every log-only reply carries a well-formed shortlist: the reply's own order first, distinct orders, at most
    PONLY_DECISION_K entries, scores or "na", bound to the recorded TURN block by the same root digest that
    training/decision_branch_collect.py computes;
  * deployed scoring (PONLY_DECISION 2) does not compile;
  * the production carriers and their generators are untouched (the hook lives in the family carriers only).
Fixed-work replays (C08 with PONLY_FIXED_SAMPLES): no timing, not strength evidence.
"""
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.decision_branch_collect import deciding_scores, parse_log, root_digest  # noqa: E402

TESTS = REPO / "bots" / "p_only" / "tests"
CXX = shutil.which(os.environ.get("CXX", "")) or shutil.which("g++") or shutil.which("clang++")
INCLUDES = [f"-I{REPO / 'inference'}", f"-I{REPO / 'bots' / 'p_only' / 'include'}", f"-I{REPO / 'bots' / 'p_only' / 'bots'}"]
FIXED = {"C08": ["-DPONLY_FIXED_SAMPLES=64"], "V12": []}


def stream(game: str) -> str:
    return gzip.decompress((TESTS / "fixtures" / f"stream_{game}.txt.gz").read_bytes()).decode()


def blocks(text: str) -> list[str]:
    out = ["TURN " + b for b in text.split("\nTURN ")[1:]]
    return [b if b.endswith("\n") else b + "\n" for b in out]


@unittest.skipUnless(CXX, "no C++ compiler")
class DecisionHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ponly-decision-"))
        cls.built = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, name: str, defines: list[str], expect_ok: bool = True):
        if name in self.built:
            return self.built[name]
        exe = self.tmp / name
        r = subprocess.run([CXX, "-std=c++20", "-O2", *INCLUDES, *defines, str(TESTS / "replay_test.cpp"), "-o", str(exe)],
                           capture_output=True, text=True)
        if not expect_ok:
            return r
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.built[name] = exe
        return exe

    def replay(self, carrier: str, decision: int | None, game: str, family: bool = True, attempts: int = 4):
        """[(turn, reply)] of an untruncated fixed-work replay; None when a deadline cut every attempt."""
        defines = [f"-DCARRIER_{carrier}", *FIXED[carrier], *(["-DPONLY_FAMILIES"] if family else [])]
        if decision is not None:
            defines.append(f"-DPONLY_DECISION={decision}")
        exe = self.build(f"{carrier}_{family}_{decision}", defines)
        for _ in range(attempts):
            r = subprocess.run([str(exe)], input=stream(game).encode(), capture_output=True, timeout=1800)
            self.assertEqual(r.returncode, 0, r.stderr.decode()[-2000:])
            lines = r.stdout.decode().splitlines()
            if int(lines[-1].split()[1]) == 0:
                return [(int(x.split(" ", 2)[0]), x.split(" ", 2)[2]) for x in lines[:-1]]
        return None

    def test_off_and_log_only_reply_alike(self):
        for carrier in ("V12", "C08"):
            for game in ("n02", "n04"):
                prod = self.replay(carrier, None, game, family=False)
                off = self.replay(carrier, 0, game)
                log = self.replay(carrier, 1, game)
                if None in (prod, off, log):
                    self.skipTest("a deadline truncated every fixed-work attempt: not testable on this machine now")
                with self.subTest(carrier=carrier, game=game):
                    self.assertEqual(prod, off)                                  # default off: production replies
                    self.assertFalse(any("#" in r for _, r in off))
                    self.assertEqual(off, [(t, r.partition(" #")[0]) for t, r in log])  # log-only: same orders

    def test_logs_are_well_formed_and_bound_to_the_root(self):
        for carrier in ("V12", "C08"):
            for game in ("n02", "n04"):
                log = self.replay(carrier, 1, game)
                if log is None:
                    self.skipTest("a deadline truncated every fixed-work attempt")
                recorded = blocks(stream(game))
                stages = set()
                with self.subTest(carrier=carrier, game=game):
                    for turn, reply in log:
                        parsed = parse_log(reply, turn)
                        self.assertIsNotNone(parsed, reply)
                        self.assertLessEqual(parsed["k"], 8)
                        self.assertEqual(parsed["root"], root_digest(recorded[turn - 1]))
                        stages.add(parsed["stage"])
                        deciding_scores(parsed)                                   # never raises on a valid log
                if carrier == "C08" and game == "n02":
                    self.assertEqual(stages, {"duel"})
                if carrier == "V12":
                    self.assertTrue(stages <= {"beam", "refine", "cut"})

    def test_deployed_scoring_is_not_built(self):
        r = self.build("V12_scoring", ["-DCARRIER_V12", "-DPONLY_FAMILIES", "-DPONLY_DECISION=2"], expect_ok=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not implemented", r.stderr)

    def test_production_carriers_untouched(self):
        for name in ("v12_ponly.hpp", "c08_ponly.hpp"):
            self.assertNotIn("decision_hook", (REPO / "bots" / "p_only" / "bots" / name).read_text(encoding="utf-8"))
        tmp = self.tmp / "derive"
        tmp.mkdir(exist_ok=True)
        env = dict(os.environ, PONLY_DERIVE_DIR=str(tmp))
        subprocess.run([sys.executable, str(REPO / "bots/p_only/tools/derive/make_families.py")], env=env, check=True,
                       capture_output=True)
        for name in ("v12_ponly_f.hpp", "c08_ponly_f.hpp"):
            self.assertEqual((tmp / name).read_text(encoding="utf-8"),
                             (REPO / "bots" / "p_only" / "bots" / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
