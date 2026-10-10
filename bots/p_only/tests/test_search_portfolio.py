"""Contracts of the joint-action search experiment (bots/p_only/tools/derive/search_experiment.py, task T0018).

  * defaults: with every experimental setting off, the family carriers reply exactly as the production carriers
    on recorded games where no wall-clock deadline binds (2 and 4 teams; v12's fixed 24 ms refinement deadline
    makes the 15-team replay depend on the machine's speed, for the production carrier too);
  * the action portfolio: same movement with different complete actions stays in the candidate set; wait,
    pickup, drop and attack serialize to the engine's requests; coordinated thresholds without overkill;
    explicit waits; carriers never fire;
  * per-branch ply-2 predictions (C08): exact-state cache keys; the cached and the uncached search are
    identical; the diagnostic mode plays exactly as the current search;
  * the real root's predictions and history are the same whatever the experimental mode, branch order or count;
  * a deadline inside the search or the alternatives still returns a valid order.
Fixed-work replays only (no timing, not strength evidence).
"""
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fusion.protocol import ATTACK, DROP, PICKUP, WAIT, parse_reply  # noqa: E402

TESTS = REPO / "bots" / "p_only" / "tests"
FIXTURES = TESTS / "fixtures"
CXX = shutil.which(os.environ.get("CXX", "")) or shutil.which("g++") or shutil.which("clang++")
INCLUDES = [f"-I{REPO / 'inference'}", f"-I{REPO / 'bots' / 'p_only' / 'include'}", f"-I{REPO / 'bots' / 'p_only' / 'bots'}"]


def stream(game: str) -> bytes:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes())


@unittest.skipUnless(CXX, "no C++ compiler")
class SearchPortfolio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ponly-search-"))
        cls.built = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build(self, src: str, name: str, defines: list[str]) -> Path:
        if name in self.built:
            return self.built[name]
        exe = self.tmp / name
        r = subprocess.run([CXX, "-std=c++20", "-O2", *INCLUDES, "-DFUSION_TIER_XXL", "-DPONLY_STATS=1", *defines,
                            str(TESTS / src), "-o", str(exe)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.built[name] = exe
        return exe

    def replay(self, name: str, carrier: str, defines: list[str], game: str, family: bool = True):
        exe = self.build("replay_test.cpp", name, [f"-DCARRIER_{carrier}", *(["-DPONLY_FAMILIES"] if family else []), *defines])
        r = subprocess.run([str(exe)], input=stream(game), capture_output=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr.decode()[-2000:])
        lines = r.stdout.decode().splitlines()
        rows = [line.split(" ", 2) for line in lines[:-1]]
        return rows, int(lines[-1].split()[1]), r.stderr.decode()

    @staticmethod
    def counters(stderr: str, tag: str) -> dict:
        line = next(l for l in stderr.splitlines() if l.startswith(tag + " "))
        tok = line.split()[1:]
        return {tok[i]: float(tok[i + 1]) for i in range(0, len(tok) - 1, 2)}

    def assert_valid_replies(self, rows):
        for turn, _, reply in rows:
            self.assertIsNotNone(parse_reply(reply, int(turn)), f"turn {turn}: {reply!r} is not a valid order")

    # ------------------------------------------------------------------ defaults
    def test_defaults_reply_as_production(self):
        for carrier, fixed in (("V12", []), ("C08", ["-DPONLY_FIXED_SAMPLES=64"])):
            for game in ("n02", "n04"):
                prod, t1, _ = self.replay(f"prod_{carrier}", carrier, fixed, game, family=False)
                fam, t2, _ = self.replay(f"fam_{carrier}", carrier, fixed, game)
                if t1 or t2:
                    self.skipTest("a deadline truncated the fixed-work search: equivalence not testable here")
                with self.subTest(carrier=carrier, game=game):
                    self.assertEqual(prod, fam)

    # ------------------------------------------------------------------ portfolio cases
    def portfolio_cases(self, carrier: str) -> tuple[dict, list]:
        exe = self.build("portfolio_test.cpp", f"portfolio_{carrier}", [f"-DCARRIER_{carrier}", "-DPONLY_FAMILIES", "-DPONLY_PORTFOLIO_K=3"])
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines()
        cases, texts, cur = {}, [], None
        for line in out:
            tok = line.split()
            if tok[0] == "CASE":
                cur = cases.setdefault(tok[1], [])
            elif tok[0] == "ALT":
                cur.append(([int(x) for x in tok[1:4]], [int(x) for x in tok[4:7]], float(tok[7])))
            elif tok[0] == "TEXT":
                texts.append((int(tok[2]), int(tok[3]), tok[4]))
        return cases, texts

    def test_portfolio_assignments(self):
        for carrier in ("C08", "V12"):
            cases, _ = self.portfolio_cases(carrier)
            with self.subTest(carrier=carrier):
                targets = cases["targets"]
                self.assertGreaterEqual(len(targets), 2)                                        # same movement ...
                self.assertEqual(len({tuple(m) for m, _, _ in targets}), 1)
                self.assertEqual(len({tuple(t) for _, t, _ in targets}), len(targets))          # ... distinct actions
                for _, t, _ in cases["overkill"]:
                    self.assertLessEqual(t.count(3), 1, "two shooters on a one-hit target")
                self.assertTrue(any(t.count(3) == 1 for _, t, _ in cases["overkill"]))
                best = max(cases["threshold"], key=lambda a: a[2])
                self.assertEqual(best[1], [3, 3, 3], "a full-health target needs the coordinated three-shot volley")
                acts = {t[0] for _, t, _ in cases["pickup"]}
                self.assertEqual(acts, {-2, -1}, "pickup and an explicit wait")
                self.assertTrue(all(t[0] == -1 for _, t, _ in cases["carrier"]), "a carrier never fires")
                values = [v for _, _, v in targets]
                self.assertEqual(values, sorted(values, reverse=True))

    def test_actions_serialize_to_engine_requests(self):
        effect = {"P": PICKUP, "X": DROP, "-": WAIT}
        for carrier in ("C08", "V12"):
            _, texts = self.portfolio_cases(carrier)
            for target, carrying, text in texts:
                req = parse_reply(f"9 S {text} S - S -", 9)
                self.assertIsNotNone(req, text)
                act = req[0].act
                with self.subTest(carrier=carrier, target=target, carrying=carrying):
                    if target >= 0:
                        self.assertEqual((act, req[0].target), (ATTACK, target))
                    elif target == -1:
                        self.assertEqual(act, WAIT, "an explicit wait must not become a pickup")
                    elif target == -2 and not carrying:
                        self.assertEqual(act, PICKUP)
                    elif target == -3 and carrying:
                        self.assertEqual(act, DROP)
                    else:  # a code without effect for this unit: the engine treats it as a wait
                        self.assertIn(act, (WAIT, PICKUP, DROP))
                    self.assertEqual(act, effect.get(text, ATTACK))

    # ------------------------------------------------------------------ modes on a recorded game
    def test_v12_portfolio_on_a_recorded_game(self):
        base, _, _ = self.replay("fam_V12", "V12", [], "n04")
        port, _, err = self.replay("v12_k3", "V12", ["-DPONLY_PORTFOLIO_K=3"], "n04")
        self.assertEqual([r[1] for r in base], [r[1] for r in port], "the root's predictions changed")
        self.assert_valid_replies(port)
        c = self.counters(err, "PORTFOLIO")
        for key in ("movements", "assignments", "waits", "refined", "same_move", "overkill_skipped"):
            self.assertGreater(c[key], 0, key)

    def test_c08_modes_on_a_recorded_game(self):
        fixed = ["-DPONLY_FIXED_SAMPLES=16", "-DPONLY_TURN_MS=100000"]
        a, ta, ea = self.replay("c08_a16", "C08", fixed, "n04")
        d, td, ed = self.replay("c08_d16", "C08", fixed + ["-DPONLY_PORTFOLIO_K=3", "-DPONLY_PLY2_MODE=1"], "n04")
        nc, tn, _ = self.replay("c08_d16_nocache", "C08", fixed + ["-DPONLY_PORTFOLIO_K=3", "-DPONLY_PLY2_MODE=1", "-DPONLY_PLY2_CACHE=0"], "n04")
        dg, tg, eg = self.replay("c08_diag16", "C08", fixed + ["-DPONLY_PLY2_MODE=2"], "n04")
        k2, tk, _ = self.replay("c08_k2m2", "C08", fixed + ["-DPONLY_PORTFOLIO_K=2", "-DPONLY_PORTFOLIO_MOVES=2", "-DPONLY_PLY2_MODE=1"], "n04")
        self.assertFalse(ta or td or tn or tg or tk)
        # The real root's predictions do not depend on the mode, the branch order or the number of branches.
        for other in (d, nc, dg, k2):
            self.assertEqual([r[1] for r in a], [r[1] for r in other])
        self.assertEqual(d, nc, "the cached search must equal the uncached one")
        self.assertEqual(a, dg, "the diagnostic mode must play as the current search")
        self.assert_valid_replies(d)
        port = self.counters(ed, "PORTFOLIO")
        for key in ("searched", "refined", "same_move", "waits"):
            self.assertGreater(port[key], 0, key)
        p0, p1, pd = self.counters(ea, "PLY2"), self.counters(ed, "PLY2"), self.counters(eg, "PLY2")
        self.assertGreater(p1["predictions"], p0["predictions"])   # one prediction per plan instead of per sample
        self.assertGreater(p1["view_hits"], 0)
        self.assertGreater(p1["memo_hits"], 0)                     # same-movement alternatives reuse settlement
        self.assertEqual(p0["memo_hits"], 0)                       # one plan per movement: nothing to reuse
        self.assertGreater(pd["diag_pairs"], 0)
        self.assertGreater(pd["diag_move_diff"], 0, "branch successors never changed a ply-2 prediction")

    def test_cache_keys_are_exact_states(self):
        exe = self.build("ply2_cache_test.cpp", "ply2_cache", ["-DPONLY_FAMILIES"])
        out = dict(((t[0], t[1]), int(t[2])) for t in (l.split() for l in subprocess.run([str(exe)], capture_output=True, text=True,
                                                                                          check=True).stdout.splitlines()))
        self.assertEqual(out[("SAME", "identical")], 1)
        for case in ("hp", "death", "flag_timer", "carried", "attack", "score"):
            self.assertEqual(out[("SAME", case)], 0, case)
        for case in ("hp", "death", "carried", "attack", "score"):  # HP, deaths, flags, attacks reach the encoder
            self.assertEqual(out[("ENCODER", case)], 1, case)

    def test_root_history_with_every_mode(self):
        defines = ["-DPONLY_FIXED_SAMPLES=16", "-DPONLY_PORTFOLIO_K=3", "-DPONLY_PLY2_MODE=1"]
        for carrier in ("C08", "V12"):
            exe = self.build("contract_test.cpp", f"contract_{carrier}", [f"-DCARRIER_{carrier}", "-DPONLY_FAMILIES", *defines])
            for game in ("n04", "n15"):
                orders = self.tmp / f"orders_{game}.txt"
                orders.write_bytes(gzip.decompress((FIXTURES / f"orders_{game}.txt.gz").read_bytes()))
                r = subprocess.run([str(exe), str(orders)], input=stream(game), capture_output=True, timeout=1800)
                out = r.stdout.decode()
                with self.subTest(carrier=carrier, game=game):
                    self.assertEqual(r.returncode, 0, out + r.stderr.decode()[-2000:])
                    self.assertIn(" 0 failures", out)
                    self.assertIn(" 0 inconsistent", out)

    def test_deadlines_return_valid_orders(self):
        c08, _, _ = self.replay("c08_deadline", "C08", ["-DPONLY_TURN_MS=0", "-DPONLY_PORTFOLIO_K=3", "-DPONLY_PLY2_MODE=1"], "n04")
        self.assert_valid_replies(c08)
        v12, _, _ = self.replay("v12_deadline", "V12", ["-DPONLY_REFINE_MS=0", "-DPONLY_PORTFOLIO_K=3"], "n04")
        self.assert_valid_replies(v12)
        cut, _, err = self.replay("v12_cut", "V12", ["-DPONLY_PORTFOLIO_K=3", "-DPONLY_TEST_CUT_AT=5"], "n04")
        self.assert_valid_replies(cut)
        self.assertGreater(self.counters(err, "PORTFOLIO")["cut"], 0, "the cut inside the alternatives was never exercised")


if __name__ == "__main__":
    unittest.main()
