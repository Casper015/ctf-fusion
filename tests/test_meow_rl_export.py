"""The Meowmind RL CPU bots and their export (task T0056, bots/meow_rl, docs/MEOWMIND_RL_DEPLOYMENT.md).

Needs a C++ compiler (skipped otherwise); no PyTorch, no engine. Recorded seat streams drive the compiled bots
exactly as the engine does, and training.meow_rl's bridge replays them in Python: every reply must match. The
critics here are synthetic plumbing fixtures, never trained exports. No timing is measured or claimed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import quant as FQ
from fusion import runner
from training.meow_rl import bridge as B
from training.meow_rl import identity as I
from training.meow_rl import policy as P
from training.meow_rl import quant as Q

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bots" / "meow_rl"))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
import sampling as SM  # noqa: E402

CXX = runner.find_compiler()
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
GAMES = ("n02", "n04", "n15")


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


def load_export():
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("meow_rl_export", REPO / "bots" / "meow_rl" / "export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = load_export()


def compile_variant(variant: Path, out: Path, defines=()) -> Path:
    from bundle import bundle  # noqa: PLC0415
    text, _ = bundle(variant)
    src = out.with_suffix(".cpp")
    src.write_text(text, encoding="utf-8")
    exe = out.with_suffix(runner.EXE)
    subprocess.run([CXX, "-std=c++20", "-O2", "-w", *defines, *(["-static"] if runner.WINDOWS else []), str(src), "-o", str(exe)],
                   check=True, capture_output=True)
    return exe


def run_bot(exe: Path, text: str) -> tuple[list[str], str]:
    r = subprocess.run([str(exe)], input=text, capture_output=True, text=True, timeout=600)
    return r.stdout.splitlines(), r.stderr


@unittest.skipUnless(CXX, "no C++ compiler")
class Bots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="meow-rl-export-"))
        cls.lib = B.load()
        cls.actor = P.check_actor(B.reference_weights(cls.lib, "policy"))
        cls.inf, cls.var = cls.tmp / "inference", cls.tmp / "variants"
        cls.manifest = E.main(["--warm-start", "--name", "ws", "--out", str(cls.inf), "--variants-dir", str(cls.var)])
        cal, _, _ = E.observations(cls.lib, cls.actor, E.CALIBRATION)
        cls.q = Q.quantize(cls.actor, cal)
        cls.exe = compile_variant(cls.var / "ws.cpp", cls.tmp / "ws")
        cls.exe_greedy = compile_variant(cls.var / "ws.cpp", cls.tmp / "ws_greedy", ["-DMEOW_RL_GREEDY=1"])

    def expected(self, text: str, greedy: bool) -> list[str]:
        blocks = B.split_stream(text)
        seed = SM.seed(blocks[0])
        out = []
        with B.Session(self.lib) as s:
            for block in blocks:
                o = s.observe(block)
                out.append(s.resolve(SM.choose(self.q, o.feats, o.joint_mask, seed, o.turn, greedy))[0])
        return out

    def test_export_is_reproducible_and_carries_its_identity(self):
        m = self.manifest
        self.assertEqual(m["identity"]["actor_weights"], "reference_warm_start")
        self.assertEqual(m["identity"]["temperature"], I.TEMPERATURE)
        self.assertEqual(m["format"]["actor"], Q.SCHEME)
        self.assertGreater(m["agreement_held_out"]["actor"]["greedy_agreement"], 0.9)
        header = (self.inf / "ws" / "actor.hpp").read_text()
        self.assertIn(I.digest(m["identity"]), header)
        E.main(["--warm-start", "--name", "ws", "--out", str(self.tmp / "inf2"), "--variants-dir", str(self.tmp / "v2")])
        for f in ("actor.hpp", "MANIFEST.json"):                           # byte-reproducible: no time, no machine path
            self.assertEqual((self.tmp / "inf2" / "ws" / f).read_bytes(), (self.inf / "ws" / f).read_bytes(), f)
        self.assertIn("sampled", m["deployment"]["choice"])
        committed = json.loads((REPO / "inference" / "meow_rl" / "warm_start" / "MANIFEST.json").read_text())
        self.assertEqual(committed["actor_digest"], m["actor_digest"])           # the committed control is this export
        self.assertEqual(m["files"]["actor.hpp"], hashlib.sha256((self.inf / "ws" / "actor.hpp").read_bytes()).hexdigest())

    def test_sampled_bot_equals_the_python_pipeline(self):
        for game in GAMES:
            text = stream(game)
            got, err = run_bot(self.exe, text)
            self.assertEqual(got, self.expected(text, greedy=False), game)
            self.assertIn("MEOWRL model ws", err)
            self.assertIn("refused 0", err)

    def test_greedy_mode_equals_the_python_pipeline(self):
        for game in ("n02", "n15"):
            text = stream(game)
            self.assertEqual(run_bot(self.exe_greedy, text)[0], self.expected(text, greedy=True), game)

    def test_common_uniforms_depend_only_on_the_public_init(self):
        blocks = B.split_stream(stream("n04"))
        a = SM.seed(blocks[0])
        turn_part = blocks[0][blocks[0].index("TURN 1"):]
        moved = blocks[0].replace(turn_part, turn_part.replace("\nU 0 0 6 14 ", "\nU 0 0 7 14 ", 1))
        self.assertNotEqual(moved, blocks[0])
        self.assertEqual(a, SM.seed(moved))                                 # turn data (positions, score) never enters
        self.assertNotEqual(a, SM.seed(blocks[0].replace("INIT 4 0 ", "INIT 4 1 ", 1)))   # the seat does
        row = blocks[0].split("\n")[3]
        self.assertNotEqual(a, SM.seed(blocks[0].replace(row, row[:5] + ("#" if row[5] == "." else ".") + row[6:], 1)))
        u = [SM.uniform(a, t) for t in range(1, 401)]
        self.assertTrue(all(0 <= x < 1 for x in u))
        self.assertLess(abs(np.mean(u) - 0.5), 0.05)
        mask = np.zeros(125, bool)
        mask[[0, 7]] = True
        f = np.zeros((3, 5, 112), np.float32)
        self.assertIn(SM.draw(self.q, f, mask, 0.0), (0, 7))
        self.assertEqual(SM.draw(self.q, f, mask, np.nextafter(1.0, 0)), 7)

    def test_malformed_input_gets_the_safe_reply(self):
        blocks = B.split_stream(stream("n02"))
        bad = blocks[3].replace("\nU 3 1 ", "\nU -1 1 ", 1)                # a unit id the reference would index
        empty = blocks[6].replace("\nUNITS 6\n", "\nUNITS 0\n", 1)       # a unit count the history must not adopt
        self.assertNotEqual(empty, blocks[6])
        text = "".join(blocks[:3]) + bad + "".join(blocks[4:6]) + empty + "".join(blocks[7:10])
        got, err = run_bot(self.exe, text)
        self.assertEqual(len(got), 10)
        self.assertEqual(got[3], "4 S - S - S -")
        self.assertIn("refused turn 4", err)
        self.assertEqual(got[6], "7 S - S - S -")
        seed, expected = SM.seed(blocks[0]), []                             # the bridge replays the same recovery
        with B.Session(self.lib) as s:
            for block in B.split_stream(text):
                try:
                    o = s.observe(block)
                except B.BridgeError:
                    s.forget_history()
                    expected.append(None)
                    continue
                expected.append(s.resolve(SM.choose(self.q, o.feats, o.joint_mask, seed, o.turn))[0])
        self.assertEqual([g for g, e in zip(got, expected) if e is not None], [e for e in expected if e is not None])
        for line in got:
            B.parse_reply(line)                                             # every line is a valid reply
        cut, _ = run_bot(self.exe, "".join(blocks[:2]) + blocks[2][: len(blocks[2]) // 2])
        self.assertEqual(len(cut), 2)                                       # end of input mid-block: a clean stop


class Identities(unittest.TestCase):
    def test_a_critic_must_match_its_declared_architecture_and_be_finite(self):
        rng = np.random.default_rng(2)
        small = [416, 64, 32, 1]
        W = [rng.standard_normal((o, i)).astype(np.float32) for i, o in zip(small, small[1:])]
        b = [np.zeros(o, np.float32) for o in small[1:]]
        E.check_critic(W, b, "small_v")
        with self.assertRaises(SystemExit):
            E.check_critic(W, b, "large_v")                                 # a small critic under a large-V identity
        two = [w.copy() for w in W]
        two[-1] = rng.standard_normal((2, 32)).astype(np.float32)
        with self.assertRaises(SystemExit):
            E.check_critic(two, b[:-1] + [np.zeros(2, np.float32)], "small_v")   # two outputs: the bot reads one
        bad = [c.copy() for c in b]
        bad[-1][0] = np.nan
        with self.assertRaises(SystemExit):
            E.check_critic(W, bad, "small_v")

    def test_mismatched_identities_are_refused(self):
        tmp = Path(tempfile.mkdtemp(prefix="meow-rl-id-"))
        good = I.identity("large_v")
        E.check_identity(good)
        for key, value in [("temperature", 1.0), ("policy_features", "other"), ("reward_units", "shaped"),
                           ("resolver", "other"), ("reference_sha256", "0" * 64), ("critic", "unknown_arch")]:
            with self.assertRaises(SystemExit, msg=key):
                E.check_identity(good | {key: value})
        np.savez(tmp / "a.npz", **{k: np.zeros(s, np.float32) for k, s in P.ACTOR_SHAPES.items()})
        (tmp / "id.json").write_text(json.dumps(good | {"temperature": 2.0}))
        with self.assertRaises(SystemExit):
            E.main(["--actor-npz", str(tmp / "a.npz"), "--identity-json", str(tmp / "id.json"), "--name", "x",
                    "--out", str(tmp / "o"), "--variants-dir", str(tmp / "v")])

    def test_the_large_critic_fits_the_int16_format(self):
        """The full 416-1024-1024-512-1 architecture: proved int32 bounds and the 1024 width of a search build."""
        rng = np.random.default_rng(1)
        sizes = [416, 1024, 1024, 512, 1]
        W = [(rng.standard_normal((b, a)) * np.sqrt(2 / a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        b = [np.zeros(n, np.float32) for n in sizes[1:]]
        x = np.clip(rng.standard_normal((200, 416)).astype(np.float32), -4, 4)
        q = E.quantize_critic(W, b, x[:100])
        self.assertEqual(sum(w.size + c.size for w, c in zip(W, b)), 2_001_921)
        for layer in q["net"]["layers"]:
            self.assertLessEqual(layer["bound"], 1 << 30)
        err = np.abs(E.critic_value_int(q, x[100:]) - E.critic_forward(W, b, x[100:]))
        scale = np.abs(E.critic_forward(W, b, x[100:])).mean()
        self.assertLess(err.mean(), 0.2 * scale)


@unittest.skipUnless(CXX, "no C++ compiler")
class Search(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="meow-rl-search-"))
        lib = B.load()
        np.savez(cls.tmp / "actor.npz", **B.reference_weights(lib, "policy"))
        rng = np.random.default_rng(0)
        sizes, c = [416, 64, 32, 1], {}
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            c[f"layers.{i}.weight"] = (rng.standard_normal((b, a)) * np.sqrt(2 / a)).astype(np.float32)
            c[f"layers.{i}.bias"] = np.zeros(b, np.float32)
        np.savez(cls.tmp / "critic.npz", **c)
        (cls.tmp / "id.json").write_text(json.dumps(I.identity("small_v")))
        cls.inf, cls.var = cls.tmp / "inference", cls.tmp / "variants"
        args = ["--actor-npz", str(cls.tmp / "actor.npz"), "--identity-json", str(cls.tmp / "id.json"),
                "--critic-npz", str(cls.tmp / "critic.npz"), "--critic", "--out", str(cls.inf), "--variants-dir", str(cls.var)]
        cls.manifest = E.main([*args, "--search", "--name", "srch"])
        E.main([*args, "--name", "direct"])
        cls.critic = c
        cls.search = compile_variant(cls.var / "srch.cpp", cls.tmp / "srch")
        cls.search0 = compile_variant(cls.var / "srch.cpp", cls.tmp / "srch0",
                                      ["-DMEOW_RL_SEARCH_DEADLINE_MS=0", "-DMEOW_RL_SEARCH_FIRST_MS=0"])
        cls.direct = compile_variant(cls.var / "direct.cpp", cls.tmp / "direct")

    def test_a_search_out_of_time_plays_exactly_the_actor(self):
        for game in ("n02", "n15"):
            text = stream(game)
            got, err = run_bot(self.search0, text)
            self.assertEqual(got, run_bot(self.direct, text)[0], game)
            self.assertIn("search_used 0 search_fallback 400", err)

    def test_search_choices_are_feasible_and_counted(self):
        text = stream("n04")
        got, err = run_bot(self.search, text)
        self.assertIn("refused 0", err)
        counters = dict(zip(err.strip().splitlines()[-1].split()[1::2], err.strip().splitlines()[-1].split()[2::2]))
        self.assertEqual(int(counters["search_used"]) + int(counters["search_fallback"]), 400)
        with B.Session(B.load()) as s:
            for block, line in zip(B.split_stream(text), got):
                o = s.observe(block)
                _, moves, _ = B.parse_reply(line)
                j = B.joint_index(moves)
                self.assertTrue(o.joint_mask[j])
                self.assertEqual(s.resolve(j)[0], line)                     # the resolver completed it, nothing else

    def test_critic_int16_is_bit_exact_in_cpp(self):
        W, b = E.critic_layers(self.critic)
        _, _, values = E.observations(B.load(), P.check_actor(B.reference_weights(B.load(), "policy")), ("n15",))
        q = E.quantize_critic(W, b, values[:200])
        head = self.tmp / "critic_check.hpp"
        head.write_text(E.critic_header(q, "check", I.identity("small_v")), encoding="utf-8")
        src = self.tmp / "critic_check.cpp"
        src.write_text("#define FUSION_MAX_WIDTH 1024\n#include <cstdio>\n#include <cstring>\n#include <iostream>\n"
                       f'#include "{head.as_posix()}"\n'
                       "int main(){int n;std::cin>>n;for(int r=0;r<n;r++){int32_t x[416],acc[1];"
                       "for(int i=0;i<416;i++){unsigned u;std::cin>>std::hex>>u;float f;std::memcpy(&f,&u,4);"
                       "double v=std::floor(double(f)*4096.0+0.5);x[i]=int32_t(std::clamp(v,-16384.0,16384.0));}"
                       "std::cin>>std::dec;fusion::forwardRaw(meow_rl_critic::kCritic,x,acc);std::printf(\"%d\\n\",acc[0]);}}\n",
                       encoding="utf-8")
        exe = self.tmp / f"critic_check{runner.EXE}"
        subprocess.run([CXX, "-std=c++20", "-O2", "-w", f"-I{REPO / 'inference'}", "-include", "algorithm", "-include", "cmath",
                        str(src), "-o", str(exe)], check=True, capture_output=True)
        rows = values[200:320]
        text = f"{len(rows)}\n" + "\n".join(" ".join(format(int(v), "x") for v in r.view(np.uint32)) for r in rows) + "\n"
        out = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True).stdout.split()
        np.testing.assert_array_equal(np.array([int(v) for v in out]), FQ.forward_raw(q["net"], Q.quantize_input(rows))[:, 0])


class Specification(unittest.TestCase):
    def test_variants_and_screen_spec_agree(self):
        variants = json.loads((REPO / "bots" / "meow_rl" / "variants.json").read_text())["variants"]
        names = [v["name"] for v in variants]
        self.assertEqual(len(names), len(set(names)))
        for v in variants:
            self.assertTrue((REPO / v["source"]).is_file(), v["source"])
        spec = json.loads((REPO / "profiles" / "meow_rl_screen.json").read_text())
        known = set(names) | set(spec["candidates"])
        for block in ("primary", "secondary"):
            cfg = spec[block]
            self.assertIsNone(cfg["seed"])                                  # allocated fresh on the host, never here
            self.assertEqual(cfg["fixtures"], cfg["maps_per_count"] * len(cfg["teams"]))
            self.assertEqual(cfg["games"], cfg["fixtures"] * (len(cfg["arms"]) + len(cfg["aa"])))
            self.assertEqual(cfg["teams"], list(range(2, 16)))
            for arm in cfg["arms"] + cfg["aa"] + [x for c in cfg["comparisons"] for x in c]:
                self.assertIn(arm, known)
            for c in cfg["comparisons"]:
                self.assertTrue(set(c) <= set(cfg["arms"]), c)              # analyze.py skips a contrast of a missing arm
        self.assertEqual(spec["primary"]["games"], 1176)
        self.assertNotEqual(spec["primary"]["work_dir"], spec["secondary"]["work_dir"])
        m = spec["secondary"]["multiplicity"]
        self.assertEqual(m["family"], len(spec["secondary"]["comparisons"]))
        z = statistics.NormalDist().inv_cdf(1 - m["alpha"] / m["family"] / 2)
        self.assertAlmostEqual(z, m["z"], places=3)


if __name__ == "__main__":
    unittest.main()
