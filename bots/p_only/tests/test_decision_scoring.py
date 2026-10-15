"""Own-action scoring hook of the v12 family carrier: PONLY_DECISION 2 (task T0045, bots/p_only/include/decision_score.hpp).

Fixed-weight scorers (no trained model) on the recorded streams of bots/p_only/tests/fixtures:
  * off is off: the scoring carrier with PONLY_DECISION 0 or 1 replies exactly as the family carrier, and a lambda = 0
    export compiles the hook away (the family carrier's replies, no counters);
  * every scored decision is the offline pipeline on the carrier's own shortlist: the logged shortlist, its deciding
    scores and candidates (training/decision_branch_collect), canonical orders and support (fusion/decision_features),
    Q of the exported scorer (numpy float64 of the same weights) and training/evaluate_decisions.apply_rule;
  * the reply stands on an invalid model, the deadline, a stage without deciding scores and K = 1, with counters;
  * a scorer bound to another carrier, an unbound test export, a missing scorer or a missing data-carrier name
    refuses to build;
  * the generated carrier is re-derived, and the files T0037's frozen identities hash are untouched.
The carrier's refinement keeps its wall-clock budget, so replies are compared between builds run back to back, as
test_decision_hook.py does. No timing or strength evidence.
"""
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fusion import decision_features as F  # noqa: E402
from fusion import native  # noqa: E402
from fusion.protocol import parse_block  # noqa: E402
from fusion.schema import layout  # noqa: E402
from tests.test_own_action_scorer import CARRIER, make_run, q64  # noqa: E402
from training import evaluate_decisions as E  # noqa: E402
from training import export_decision_scorer as X  # noqa: E402
from training import train_decision_ranker as T  # noqa: E402
from training.decision_branch_collect import deciding_scores, parse_log, select_candidates  # noqa: E402

TESTS = REPO / "bots" / "p_only" / "tests"
CXX = shutil.which(os.environ.get("CXX", "")) or shutil.which("g++") or shutil.which("clang++")
INCLUDES = [f"-I{REPO / 'inference'}", f"-I{REPO / 'bots' / 'p_only' / 'include'}", f"-I{REPO / 'bots' / 'p_only' / 'bots'}"]
DATA_CARRIER = '-DPONLY_DSCORE_DATA_CARRIER="v12_ponly_xxl_dlog"'
GAMES = ("n02", "n04")
# T0037's branch collector hashes decision_hook.hpp into its frozen identity (decision_branch_collect.COLLECTOR_FILES).
DECISION_HOOK_SHA256 = "0e70f752aaf6d253d8fe382c3ec6be4436c47157bb682638727a0203b058dfdc"


def stream(game: str) -> str:
    return gzip.decompress((TESTS / "fixtures" / f"stream_{game}.txt.gz").read_bytes()).decode()


@unittest.skipUnless(CXX, "no C++ compiler")
class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ponly-dscore-"))
        cls.L = layout()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def scorer(self, name, lam=5.0, carrier=CARRIER, unbound=False, seed=1) -> Path:
        run = make_run(self.tmp / f"run_{name}", "mlp", (16,), seed=seed, carrier=None if unbound else carrier, scale=3.0)
        out = self.tmp / f"exp_{name}"
        X.main(["--run", str(run), "--out", str(out), "--lambda", str(lam), "--unselected",
                *(["--allow-unbound-carrier"] if unbound else [])])
        return out

    def build(self, name, defines, scorer: Path | None = None, expect_ok=True):
        exe = self.tmp / (name + (".exe" if os.name == "nt" else ""))   # g++ on Windows appends .exe to -o
        cmd = [CXX, "-std=c++20", "-O2", *INCLUDES, "-DCARRIER_V12", "-DPONLY_FAMILIES", *defines,
               *(["-include", str(scorer / X.HEADER)] if scorer else []), str(TESTS / "replay_test.cpp"), "-o", str(exe)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not expect_ok:
            return r
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return exe

    def replay(self, exe: Path, game: str) -> tuple[list[tuple[int, str]], str]:
        r = subprocess.run([str(exe)], input=stream(game).encode(), capture_output=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr.decode()[-2000:])
        lines = r.stdout.decode().splitlines()[:-1]
        return [(int(x.split(" ", 2)[0]), x.split(" ", 2)[2]) for x in lines], r.stderr.decode()

    @staticmethod
    def counters(err: str) -> dict:
        line = next((x for x in err.splitlines() if x.startswith("DSCORE ")), None)
        if line is None:
            return {}
        tok = line.split()[1:]
        return {tok[i]: int(tok[i + 1]) for i in range(0, len(tok), 2)}

    def test_off_and_log_only_equal_the_family_carrier(self):
        for decision in (0, 1):
            fam = self.build(f"f{decision}", [f"-DPONLY_DECISION={decision}"])
            fs = self.build(f"fs{decision}", ["-DPONLY_SCORING", f"-DPONLY_DECISION={decision}"])
            for game in GAMES:
                with self.subTest(decision=decision, game=game):
                    a, b = self.replay(fam, game)[0], self.replay(fs, game)[0]
                    if decision:
                        a, b = [(t, r.partition(" #")[0]) for t, r in a], [(t, r.partition(" #")[0]) for t, r in b]
                    self.assertEqual(a, b)

    def test_lambda_zero_compiles_the_hook_away(self):
        exp = self.scorer("lam0", lam=0.0)
        self.assertIn("#define OWN_ACTION_SCORER_LAMBDA_ZERO 1", (exp / X.HEADER).read_text())
        fam = self.build("f_off", [])
        fs = self.build("fs_lam0", ["-DPONLY_SCORING", "-DPONLY_DECISION=2", DATA_CARRIER], exp)
        for game in GAMES:
            prod, (got, err) = self.replay(fam, game)[0], self.replay(fs, game)
            self.assertEqual(prod, got)
            self.assertFalse(any("#" in r for _, r in got))
            self.assertEqual(self.counters(err), {})                    # no hook, no counters

    def test_choices_follow_the_offline_rule(self):
        exp = self.scorer("rule", lam=5.0)
        scorer = T.load_scorer(self.tmp / "run_rule")
        fs = self.build("fs_rule", ["-DPONLY_SCORING", "-DPONLY_DECISION=2", "-DPONLY_DSCORE_TRACE=1", DATA_CARRIER], exp)
        scored = switched = 0
        for game in GAMES:
            text = stream(game)
            init, rest = text.split("TURN ", 1)
            me = int(init.split()[2])
            size = int(init.split()[3])
            blocks = ["TURN " + b for b in ("TURN " + rest).split("TURN ")[1:]]
            replies, err = self.replay(fs, game)
            c = self.counters(err)
            self.assertEqual(c["roots"], len(replies))
            for turn, reply in replies:
                order, _, trace = reply.partition(" # ")
                comment, _, dsc = trace.partition(" ## dsc1 ")
                incumbent = comment.split(" | ")[1].split(";")[0]          # the log is written for the reply's own order
                log = parse_log(f"{turn} {incumbent} # {comment}", turn)
                if not dsc.startswith("stage="):
                    continue                                            # no deciding scores, K = 1: the reply stands
                ok, why = deciding_scores(log)
                self.assertIsNotNone(ok, why)
                expect = select_candidates(ok, 3)
                head, *items = dsc.split(" | ")
                chosen = int(head.split("chosen=")[1])
                got = [it.split(";") for it in items]
                self.assertEqual([g[0] for g in got], [e["order"] for e in expect])
                self.assertEqual([float(g[1]) for g in got], [e["score"] for e in expect])   # exactly the logged values
                # Python: canonical orders, support, gains and the rule on the same shortlist
                stream_t = init + "".join(blocks[:turn])
                rec = native.extract(stream_t)
                rec = rec[(rec[:, 0] == turn) & (rec[:, 1] == me)][0]
                units = parse_block(blocks[turn - 1]).units
                lines = [f"{turn} {e['order']}" for e in expect]
                canon = [F.canonicalize(rec, self.L, ln, turn, me, units, size) for ln in lines]
                flags = [x["flags"] for x in canon]
                F.mark_duplicates([x["key"] for x in canon], flags, incumbent=0)
                self.assertEqual([g[3] for g in got], [x["key"] for x in canon])
                xr = np.repeat(F.root_features(rec, self.L)[None], len(lines), axis=0)
                xc = np.stack([F.candidate_features(rec, self.L, x["move"], x["act"], x["slot"]) for x in canon])
                q = q64(scorer, xr, xc)
                gain = q - q[0]
                for i, g in enumerate(got):
                    if g[2] != "na":
                        self.assertAlmostEqual(float(g[2]), gain[i], delta=1e-9)
                d = {"root_first": np.array([0]), "root_count": np.array([len(lines)]), "root_incumbent": np.array([0]),
                     "cand_root": np.zeros(len(lines), np.int64), "cand_support": F.supported(np.array(flags)),
                     "cand_n": np.ones(len(lines), np.int64), "cand_key": np.array([x["key"] for x in canon]),
                     "cand_carrier_score": np.array([e["score"] for e in expect])}
                py = int(E.apply_rule(d, gain, 5.0, "zscore")[0])
                self.assertEqual(chosen, py, f"{game} turn {turn}")
                self.assertEqual(order, f"{turn} {expect[chosen]['order']}")
                scored += 1
                switched += chosen != 0
            self.assertEqual(c["scored"] + c["no_scores"] + c["k1"] + c["deadline"], c["roots"])
        self.assertGreater(scored, 50)
        self.assertGreater(switched, 0, "the fixed scorer never switched: the rule was not exercised")

    def test_invalid_model_and_deadline_keep_the_reply(self):
        exp = self.scorer("nan", lam=5.0)
        text = (exp / X.HEADER).read_text()
        (exp / X.HEADER).write_text(text.replace("inline constexpr float W0[] = {\n    ",
                                                 'inline constexpr float W0[] = {\n    __builtin_nanf(""), ', 1))
        fam = self.build("f_off2", [])
        bad = self.build("fs_nan", ["-DPONLY_SCORING", "-DPONLY_DECISION=2", DATA_CARRIER], exp)
        late = self.build("fs_late", ["-DPONLY_SCORING", "-DPONLY_DECISION=2", DATA_CARRIER, "-DPONLY_DSCORE_DEADLINE_MS=-1"],
                          self.scorer("late", lam=5.0))
        for game in GAMES:
            prod = self.replay(fam, game)[0]
            got, err = self.replay(bad, game)
            self.assertEqual(prod, got)
            c = self.counters(err)
            self.assertEqual((c["invalid"], c["scored"]), (c["roots"], 0))
            got, err = self.replay(late, game)
            self.assertEqual(prod, got)
            c = self.counters(err)
            self.assertEqual(c["scored"], 0)
            self.assertGreater(c["deadline"], 0)

    def test_identity_refused_at_build(self):
        other = json.loads(json.dumps(CARRIER))
        other["config"]["config"]["variant"] = "c08_ponly_xxl_dlog"
        cases = {
            "other carrier": (self.scorer("other", carrier=other), [DATA_CARRIER], "not trained on data of this carrier"),
            "unbound": (self.scorer("unbound", unbound=True), [DATA_CARRIER], "not trained on data of this carrier"),
            "no data carrier": (self.scorer("nodc"), [], "PONLY_DSCORE_DATA_CARRIER"),
            "no scorer": (None, [DATA_CARRIER], "needs an exported own-action scorer"),
        }
        for name, (exp, defines, message) in cases.items():
            r = self.build(f"bad_{name.replace(' ', '_')}", ["-DPONLY_SCORING", "-DPONLY_DECISION=2", *defines], exp,
                           expect_ok=False)
            with self.subTest(name):
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(message, r.stderr)
        ok = self.build("unbound_allowed", ["-DPONLY_SCORING", "-DPONLY_DECISION=2", DATA_CARRIER,
                                            "-DPONLY_DSCORE_ALLOW_UNBOUND"], self.tmp / "exp_unbound")
        self.assertTrue(ok.exists())

    def test_generated_and_frozen_files(self):
        out = self.tmp / "derive"
        out.mkdir(exist_ok=True)
        env = dict(os.environ, PONLY_DERIVE_DIR=str(out))
        subprocess.run([sys.executable, str(REPO / "bots/p_only/tools/derive/make_scoring.py")], env=env, check=True,
                       capture_output=True)
        self.assertEqual((out / "v12_ponly_fs.hpp").read_text(encoding="utf-8"),
                         (REPO / "bots/p_only/bots/v12_ponly_fs.hpp").read_text(encoding="utf-8"))
        hook = hashlib.sha256((REPO / "bots/p_only/include/decision_hook.hpp").read_bytes()).hexdigest()
        self.assertEqual(hook, DECISION_HOOK_SHA256, "decision_hook.hpp is frozen by T0037's collector identity")


if __name__ == "__main__":
    unittest.main()
