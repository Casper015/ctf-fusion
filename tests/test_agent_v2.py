"""The agent_v2 carrier (task T0065, docs/AGENT_V2.md): rules, model contracts, resolver, simulator, protocol.

Needs a C++ compiler (skipped otherwise). The engine-parity test also needs an arena checkout (AI_TEST_ROOT: the public
meow-ai-arena clone or the AI_Test tree) and is skipped without one. Recorded seat streams
(bots/p_only/tests/fixtures/stream_n*.txt.gz) drive the binaries. These are contract and correctness tests of a
redesigned carrier: nothing here compares it with the Meowmind bot's decisions, and no timing is measured or claimed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fusion import runner

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
GAMES = ("n02", "n04", "n15")
CXX = runner.find_compiler()
ACTORS = REPO / "inference" / "meow_rl" / "aligned" / "p2_large"
FLAGS = ["-std=c++20", "-O2", "-w"]


def stream(game: str) -> str:
    return gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()


def bundle_and_build(variant: Path, exe: Path, defines=()) -> Path:
    from bundle import bundle  # noqa: PLC0415
    text, _ = bundle(variant)
    src = exe.with_suffix(".cpp")
    src.write_text(text, encoding="utf-8")
    out = exe.with_suffix(runner.EXE)
    subprocess.run([CXX, *FLAGS, *defines, *(["-static"] if runner.WINDOWS else []), str(src), "-o", str(out)],
                   check=True, capture_output=True)
    return out


def build_check(exe: Path, defines=("-DFUSION_TIER_PXXL_VM=1",)) -> subprocess.CompletedProcess:
    return subprocess.run([CXX, *FLAGS, "-I", str(REPO / "inference"), "-I", str(REPO / "tools" / "cpp" / "meow_rl"), *defines,
                           f'-DACTOR_HEADER="{(ACTORS / "actor.hpp").as_posix()}"',
                           f'-DCRITIC_HEADER="{(ACTORS / "critic_reference16.hpp").as_posix()}"',
                           str(REPO / "tools" / "cpp" / "agent_v2" / "check.cpp"), "-o", str(exe)], capture_output=True, text=True)


def assert_valid_replies(tc: unittest.TestCase, game: str, text: str, lines: list[str], err: str):
    """Complete valid orders on every turn, nothing refused or inconsistent (the carrier's reply contract)."""
    tc.assertEqual(len(lines), 400, game)
    states = [b for b in text.split("END\n") if "TURN" in b]
    me = int(text.split()[2])
    for turn, (line, block) in enumerate(zip(lines, states), 1):
        tok = line.split()
        tc.assertEqual((len(tok), tok[0]), (7, str(turn)), line)
        units = {int(u.split()[1]): u.split() for u in block.split("\n") if u.startswith("U ")}
        for k in range(3):
            mv, act = tok[1 + 2 * k], tok[2 + 2 * k]
            tc.assertIn(mv, "SUDLR")
            carrying = int(units[3 * me + k][6]) >= 0
            if carrying:
                tc.assertEqual(act, "-", (game, turn, k))                     # carriers wait
            else:
                tc.assertTrue(act == "P" or (act.isdigit() and int(act) // 3 != me), (game, turn, act))
            tc.assertNotEqual(act, "X")                                        # assign_v1 never drops
    tc.assertIn("refused 0 inconsistent 0", err.strip().splitlines()[-1])
    tc.assertIn("actions agent_v2_actions_v1 resolver assign_v1/dp_v1", err)


def engine_root() -> Path | None:
    try:
        from fusion.paths import ai_test_root  # noqa: PLC0415
        return ai_test_root()
    except (SystemExit, Exception):
        return None


class Extract(unittest.TestCase):
    def test_the_frozen_feature_code_is_a_verbatim_extract(self):
        r = subprocess.run([sys.executable, str(REPO / "bots" / "agent_v2" / "extract.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)
        text = (REPO / "inference" / "agent_v2" / "meowmind_features.hpp").read_text(encoding="utf-8")
        self.assertIn("no license notice", text)
        for absent in ("struct Planner", "struct World", "bool read_turn", "workingExperts", "enemyOrders"):
            self.assertNotIn(absent, text)


@unittest.skipUnless(CXX, "no C++ compiler")
class Contracts(unittest.TestCase):
    """The native contract checks (tools/cpp/agent_v2/check.cpp) on every recorded stream."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="agent-v2-"))
        cls.check = cls.tmp / f"check{runner.EXE}"
        build_check(cls.check).check_returncode()
        cls.streams = {}
        for g in GAMES:
            p = cls.tmp / f"{g}.txt"
            p.write_text(stream(g), encoding="utf-8")
            cls.streams[g] = p

    def run_check(self, *args) -> str:
        r = subprocess.run([str(self.check), *map(str, args)], capture_output=True, text=True, timeout=1200)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(r.stdout.startswith("ok"), r.stdout)
        return r.stdout

    def test_resolver_matches_its_unpruned_brute_force_and_the_engine_rules(self):
        for g, p in self.streams.items():
            out = self.run_check("resolver", p)
            self.assertGreater(int(re.search(r"cases (\d+)", out).group(1)), 1000, g)

    def test_opponent_model_contracts_and_cache_identity(self):
        for g, p in self.streams.items():
            out = self.run_check("p", p)
            self.assertIn(f"model {g}", out)                                    # the dedicated model of that team count

    def test_the_default_bank_is_p2_xxl(self):
        out = self.run_check("bank")
        self.assertIn("ok bank models_pxxl_vm models 2:n02 ", out)
        self.assertIn(" 9:general ", out)                                       # P2 XXL has no dedicated n09
        self.assertIn(" 15:n15 distinct 11", out)

    def test_search_branches_never_write_back(self):
        for g, p in self.streams.items():
            out = self.run_check("isolation", p)
            self.assertGreater(int(re.search(r"branches (\d+)", out).group(1)), 0, g)

    def test_fixed_work_tree_replay_is_deterministic(self):
        outs = []
        for _ in range(2):
            leaves = self.tmp / f"leaves{len(outs)}.f32"
            r = subprocess.run([str(self.check), "tree", str(self.streams["n04"]), "5", "120", "333"], capture_output=True, text=True,
                               env=dict(os.environ, AGENT_V2_TREE_LEAVES=str(leaves)), check=True)
            outs.append((r.stdout, leaves.read_bytes()))
        self.assertEqual(outs[0], outs[1])
        trees = [json.loads(x) for x in outs[0][0].splitlines()]
        self.assertEqual([t["turn"] for t in trees], [5, 120, 333])
        rows = sum(1 for t in trees for lf in t["leaves"] if lf["row"] >= 0)
        self.assertEqual(rows * 416 * 4, len(outs[0][1]))
        self.assertTrue(all(len(t["leaves"]) == len(t["roots"]) * t["samples"] for t in trees))

    @unittest.skipUnless(engine_root(), "no arena checkout (AI_TEST_ROOT)")
    def test_simulator_matches_the_engine_and_refreshes_lawfully(self):
        from fusion.paths import load_engine  # noqa: PLC0415
        eng = load_engine(engine_root())
        status = {"home": 0, "carried": 1, "dropped": 2, "cooldown": 3}

        def parse(text):
            st = {"units": [], "flags": [], "attacks": []}
            for line in text.strip().split("\n"):
                w = line.split()
                if w[0] == "TURN": st["turn"] = int(w[1])
                elif w[0] == "SCORE": st["score"] = list(map(int, w[1:]))
                elif w[0] == "U": st["units"].append(tuple(map(int, w[3:8])))
                elif w[0] == "F": st["flags"].append((status[w[2]], *map(int, w[3:7])))
                elif w[0] == "E" and w[1] == "attack": st["attacks"].append((int(w[2]), int(w[3])))
            return st

        def dump(line, teams, nu, nf):
            head, _, att = line.partition(" |")
            t = list(map(int, head.split()))
            k = 1 + teams
            units = [tuple(t[k + 5 * i:k + 5 * i + 5]) for i in range(nu)]
            k += 5 * nu
            a = list(map(int, att.split()))
            return {"turn": t[0], "score": t[1:1 + teams], "units": units,
                    "flags": [tuple(t[k + 5 * i:k + 5 * i + 5]) for i in range(nf)], "attacks": sorted(zip(a[::2], a[1::2]))}

        rnd = random.Random(20261016)
        steps = lawful = 0
        for teams in range(2, 16):
            g = eng.Game(teams, 900 + teams, flag_seed=7 * teams + 1)
            spots_all = [tuple(p) for p in g.flag_spots]
            first = g.encode(0)
            init, block = first[:first.index("TURN")], first
            for _ in range(160 if teams <= 6 else 60):
                if g.done:
                    break
                st = parse(block)
                orders = []
                for uid, (x, y, hp, flag, _) in enumerate(st["units"]):
                    mv, r = rnd.randrange(5), rnd.random()
                    foes = [j for j, u in enumerate(st["units"]) if j // 3 != uid // 3 and u[0] >= 0 and abs(u[0] - x) + abs(u[1] - y) <= 3]
                    if flag >= 0 and r < .15: orders.append((mv, 2, -1))
                    elif r < .5 and foes: orders.append((mv, 3, rnd.choice(foes)))
                    elif r < .85: orders.append((mv, 1, -1))
                    else: orders.append((mv, 0, -1))
                cmds = {}
                for t in range(teams):
                    toks = [str(g.turn + 1)]
                    for m, a, tg in orders[3 * t:3 * t + 3]:
                        toks += ["SUDLR"[m], "X" if a == 2 else "-" if a == 0 else "P" if a == 1 else str(tg)]
                    cmds[t] = g.decode(t, " ".join(toks))
                g.step(cmds)
                nxt = g.encode(0)
                want = parse(nxt)
                returned = [f for f, (a, b) in enumerate(zip(st["flags"], want["flags"])) if a[0] in (2, 3) and b[0] == 0]
                spots = [(want["flags"][f][1], want["flags"][f][2]) if f in returned else (-1, -1) for f in range(len(st["flags"]))]
                base = init + block[block.index("TURN"):] + f"{rnd.randrange(1, 2 ** 62)}\n" + "".join(f"{m} {a} {t}\n" for m, a, t in orders)
                got = dump(subprocess.run([str(self.check), "step"], input=base + "spots\n" + "".join(f"{x} {y}\n" for x, y in spots),
                                          capture_output=True, text=True, check=True).stdout.strip(), teams, len(st["units"]), len(st["flags"]))
                for key in ("turn", "score", "units", "flags"):
                    self.assertEqual(got[key], want[key], (teams, st["turn"], key))
                self.assertEqual(got["attacks"], sorted(want["attacks"]), (teams, st["turn"]))
                if returned:
                    samp = dump(subprocess.run([str(self.check), "step"], input=base, capture_output=True, text=True, check=True).stdout.strip(),
                                teams, len(st["units"]), len(st["flags"]))
                    home = [(f[1], f[2]) for f in st["flags"] if f[0] == 0]
                    for f in returned:
                        p = (samp["flags"][f][1], samp["flags"][f][2])
                        self.assertIn(p, spots_all)
                        self.assertTrue(p not in home or len(set(home)) >= len(spots_all), p)
                        home.append(p)
                        lawful += 1
                steps += 1
                block = nxt
        self.assertGreater(steps, 1000)


@unittest.skipUnless(CXX, "no C++ compiler")
class Bot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="agent-v2-bot-"))
        v = REPO / "bots" / "agent_v2" / "variants"
        cls.direct = bundle_and_build(v / "agent_v2_p2xxl_direct.cpp", cls.tmp / "direct")
        cls.search = bundle_and_build(v / "agent_v2_p2xxl_search_w16.cpp", cls.tmp / "search")
        cls.search0 = bundle_and_build(v / "agent_v2_p2xxl_search_w16.cpp", cls.tmp / "search0",
                                       ["-DAGENT_V2_SEARCH_DEADLINE_MS=0", "-DAGENT_V2_SEARCH_FIRST_MS=0"])

    def play(self, exe: Path, text: str) -> tuple[list[str], str]:
        r = subprocess.run([str(exe)], input=text, capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        return r.stdout.splitlines(), r.stderr

    def test_bundles_are_single_files_without_the_reference(self):
        from bundle import bundle  # noqa: PLC0415
        for f in (REPO / "bots" / "agent_v2" / "variants").glob("*.cpp"):
            text, _ = bundle(f)
            for absent in ("meowmind_reference_main", "struct Planner", "workingExperts", "headPosterior", "struct World"):
                self.assertNotIn(absent, text, f.name)
            unresolved = [l for l in text.split("\n") if l.lstrip().startswith('#include "')]
            self.assertEqual(unresolved, [], f.name)                          # every local include is inlined

    def test_replies_are_complete_valid_orders(self):
        for g in GAMES:
            text = stream(g)
            lines, err = self.play(self.direct, text)
            assert_valid_replies(self, g, text, lines, err)
            self.assertIn(f"predictor P model {g} ", err)

    def test_a_run_is_reproducible(self):
        text = stream("n04")
        self.assertEqual(self.play(self.direct, text)[0], self.play(self.direct, text)[0])

    def test_a_malformed_block_is_refused_and_the_next_turn_answered(self):
        blocks = stream("n04").split("END\n")
        blocks[10] = blocks[10].replace(" home ", " sideways ", 1) if " home " in blocks[10] else blocks[10].replace("SCORE", "SCORE 9", 1)
        lines, err = self.play(self.direct, "END\n".join(blocks))
        self.assertEqual(lines[10].split(), ["11", "S", "-", "S", "-", "S", "-"])     # the refused turn's own number
        self.assertEqual(lines[11].split()[0], "12")
        self.assertIn("refused 1", err.strip().splitlines()[-1])

    def test_out_of_time_search_plays_the_ready_default_order(self):
        text = "END\n".join(stream("n02").split("END\n")[:40]) + "END\n"
        self.assertEqual(self.play(self.search0, text)[0], self.play(self.direct, text)[0])
        last = self.play(self.search0, text)[1].strip().splitlines()[-1]
        self.assertIn("search_used 0 search_fallback 40", last)

    def test_search_counters_add_up(self):
        lines, err = self.play(self.search, stream("n04"))
        c = dict(zip(err.strip().splitlines()[-1].split()[1::2], err.strip().splitlines()[-1].split()[2::2]))
        self.assertEqual(int(c["search_used"]) + int(c["search_fallback"]), 400)
        self.assertEqual(c["critic"], "wideact16")


@unittest.skipUnless(CXX, "no C++ compiler")
class CountSpecialistsBank(unittest.TestCase):
    """FUSION_TIER_CSP: T0067's per-count opponent P specialists (inference/count_specialists_p) in the carrier."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="agent-v2-csp-"))
        cls.check = cls.tmp / f"check{runner.EXE}"
        build_check(cls.check, ["-DFUSION_TIER_CSP=1", "-DFUSION_MAX_WIDTH=2048"]).check_returncode()
        v = REPO / "bots" / "agent_v2" / "variants"
        cls.csp = bundle_and_build(v / "agent_v2_csp_direct.cpp", cls.tmp / "csp")
        cls.direct = bundle_and_build(v / "agent_v2_p2xxl_direct.cpp", cls.tmp / "direct")

    def play(self, exe: Path, text: str) -> tuple[list[str], str]:
        r = subprocess.run([str(exe)], input=text, capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        return r.stdout.splitlines(), r.stderr

    def test_every_team_count_has_its_own_specialist(self):
        out = subprocess.run([str(self.check), "bank"], capture_output=True, text=True, check=True).stdout
        want = " ".join(f"{n}:csp_n{n:02d}" for n in range(2, 16))
        self.assertEqual(out.strip(), f"ok bank count_specialists_p models {want} distinct 14")

    def test_opponent_model_contracts_and_cache_identity(self):
        for g in GAMES:
            p = self.tmp / f"{g}.txt"
            p.write_text(stream(g), encoding="utf-8")
            out = subprocess.run([str(self.check), "p", str(p)], capture_output=True, text=True, timeout=1200).stdout
            self.assertTrue(out.startswith("ok"), out)
            self.assertIn(f"model csp_{g}", out)

    def test_the_variant_replays_without_refusals(self):
        replies = {}
        for g in GAMES:
            text = stream(g)
            lines, err = self.play(self.csp, text)
            assert_valid_replies(self, g, text, lines, err)
            self.assertIn(f"predictor P model csp_{g} ", err)                  # the bank is named, never a bare n02
            self.assertIn("actor aligned_p2_large ", err)                      # the same actor as agent_v2_p2xxl_direct
            replies[g] = lines
        self.assertEqual(self.play(self.csp, stream("n04"))[0], replies["n04"])  # reproducible

    def test_the_wider_buffers_do_not_change_the_actor_or_resolver(self):
        from bundle import bundle  # noqa: PLC0415
        text, _ = bundle(REPO / "bots" / "agent_v2" / "variants" / "agent_v2_p2xxl_direct.cpp")
        self.assertIn("#define FUSION_MAX_WIDTH 1024", text)
        src = self.tmp / "direct2048.cpp"
        src.write_text(text.replace("#define FUSION_MAX_WIDTH 1024", "#define FUSION_MAX_WIDTH 2048", 1), encoding="utf-8")
        exe = self.tmp / f"direct2048{runner.EXE}"
        subprocess.run([CXX, *FLAGS, *(["-static"] if runner.WINDOWS else []), str(src), "-o", str(exe)], check=True, capture_output=True)
        game = stream("n04")
        self.assertEqual(self.play(exe, game)[0], self.play(self.direct, game)[0])

    def test_two_banks_at_once_are_refused(self):
        r = build_check(self.tmp / f"both{runner.EXE}", ["-fsyntax-only", "-DFUSION_TIER_CSP=1", "-DFUSION_TIER_PXXL_VM=1",
                                                          "-DFUSION_MAX_WIDTH=2048"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("define exactly one FUSION_TIER_*", r.stderr)

    def test_a_narrow_build_is_refused(self):
        r = build_check(self.tmp / f"narrow{runner.EXE}", ["-fsyntax-only", "-DFUSION_TIER_CSP=1", "-DFUSION_MAX_WIDTH=1024"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exceeds kMaxWidth", r.stderr)

    def test_the_variant_is_registered_and_not_a_production_default(self):
        reg = json.loads((REPO / "bots" / "agent_v2" / "variants.json").read_text())
        v = {x["name"]: x for x in reg["variants"]}["agent_v2_csp_direct"]
        self.assertEqual(v["source"], "bots/agent_v2/variants/agent_v2_csp_direct.cpp")
        prof = json.loads((REPO / "profiles" / "agent_v2.json").read_text())
        self.assertNotIn("agent_v2_csp_direct", json.dumps(prof))


@unittest.skipUnless(CXX, "no C++ compiler")
class CountSpecialistsRL(unittest.TestCase):
    """T0068's exact-count RL policy specialists in agent_v2_rl_csp."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="agent-v2-rl-csp-"))
        v = REPO / "bots" / "agent_v2" / "variants"
        cls.variant_src = v / "agent_v2_rl_csp.cpp"
        cls.binary = bundle_and_build(cls.variant_src, cls.tmp / "rl_csp")

    def play(self, exe: Path, text: str) -> tuple[list[str], str]:
        r = subprocess.run([str(exe)], input=text, capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        return r.stdout.splitlines(), r.stderr

    def test_variant_is_registered_and_not_a_production_default(self):
        reg = json.loads((REPO / "bots" / "agent_v2" / "variants.json").read_text(encoding="utf-8"))
        v = {x["name"]: x for x in reg["variants"]}
        self.assertIn("agent_v2_rl_csp", v)
        self.assertEqual(v["agent_v2_rl_csp"]["source"], "bots/agent_v2/variants/agent_v2_rl_csp.cpp")
        prof = json.loads((REPO / "profiles" / "agent_v2.json").read_text(encoding="utf-8"))
        self.assertNotIn("agent_v2_rl_csp", json.dumps(prof))

    def test_count_specialists_rl_variant_routing_and_out_exp(self):
        probe_src = self.tmp / "probe_routing.cpp"
        variant_rel = self.variant_src.relative_to(REPO).as_posix()
        probe_code = [
            "#define main test_main_ignored",
            f'#include "{variant_rel}"',
            "#undef main",
            "#include <cassert>",
            "#include <iostream>",
            "",
            "int main() {",
            "    assert(meow_rl_model::kActor.outExp == 16);",
            "    for (int n = 2; n <= 15; ++n) {",
            "        agent_v2_bot::g_active_teams = n;",
            "        const fusion::Net& net = meow_rl_model::kActor;",
            "        assert(&net != &fusion::models::rl_general::kActor);",
            "    }",
            "    agent_v2_bot::g_active_teams = 0;",
            "    const fusion::Net& gen = meow_rl_model::kActor;",
            "    assert(&gen == &fusion::models::rl_general::kActor);",
            '    std::cout << "OK_ROUTING_REAL_ENTRYPOINT\\n";',
            "    return 0;",
            "}",
        ]
        probe_src.write_text("\n".join(probe_code), encoding="utf-8")
        exe = self.tmp / f"probe{runner.EXE}"
        subprocess.run([CXX, *FLAGS, *(["-static"] if runner.WINDOWS else []),
                        "-I", str(REPO), "-I", str(REPO / "inference"), "-I", str(REPO / "bots" / "agent_v2"),
                        str(probe_src), "-o", str(exe)],
                       check=True, capture_output=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
        self.assertIn("OK_ROUTING_REAL_ENTRYPOINT", r.stdout)

    def test_real_variant_binary_on_stream(self):
        text = stream("n02")
        lines, err = self.play(self.binary, text)
        assert_valid_replies(self, "n02", text, lines, err)
        self.assertIn("predictor P model csp_n02 ", err)
        self.assertIn("actor count_specialists_rl ", err)


class Profile(unittest.TestCase):
    def test_pins_match_the_files_and_names_are_new(self):
        prof = json.loads((REPO / "profiles" / "agent_v2.json").read_text())
        sha = lambda p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()  # noqa: E731
        self.assertEqual(sha(prof["actor"]["header"]), prof["actor"]["sha256"])
        for c in prof["critics"].values():
            self.assertEqual(sha(c["header"]), c["sha256"])
        self.assertEqual(sha("inference/models_pxxl_vm/MANIFEST.json"), prof["predictor"]["manifest_sha256"])
        others = set()
        for f in ("bots/meow_rl/variants.json", "bots/meow_rl/variants_aligned.json", "bots/p_only/profiles/variants.json"):
            p = REPO / f
            if p.is_file():
                others |= {v["name"] for v in json.loads(p.read_text())["variants"]}
        self.assertFalse(set(prof["variants"]) & others)
        self.assertTrue(all(n.startswith("agent_v2_") for n in prof["variants"]))
        self.assertEqual(prof["host_gate"]["workers"], 8)
        self.assertEqual(prof["host_gate"]["limits_ms"], [2000, 50])
        self.assertIsNone(prof["host_gate"]["seeds"]["preflight"])


if __name__ == "__main__":
    unittest.main()
