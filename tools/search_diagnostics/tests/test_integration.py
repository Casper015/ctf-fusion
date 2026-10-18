"""Integration tests of the branch diagnostics (task T0085): the native programs, the engine and real bot processes.

    AI_TEST_ROOT=<arena checkout> python -m unittest discover -s tools/search_diagnostics/tests -t .

Needs a C++ compiler (skipped otherwise); the engine tests also need an arena checkout (AI_TEST_ROOT: the public
meow-ai-arena clone or the AI_Test tree) and are skipped without one. Programs are cached under
build/search_diagnostics/test by their bundle hash. Nothing here is timing or strength evidence.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fusion import matchlog, runner
from fusion.paths import REPO
from tools.search_diagnostics import build as diag_build
from tools.search_diagnostics.schemas import ValidationError

HERE = Path(__file__).resolve().parent
OUT = REPO / "build" / "search_diagnostics" / "test"
STREAM = REPO / "bots" / "p_only" / "tests" / "fixtures" / "stream_n04.txt.gz"


def arena():
    root = os.environ.get("AI_TEST_ROOT")
    if not root:
        return None
    from fusion.paths import workspace_rel  # noqa: PLC0415
    return Path(root) if workspace_rel(Path(root)) else None


def production(package: str) -> Path:
    """The package's committed production variant, bundled and compiled like bots/p_only/tools/build.py does."""
    sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
    from bundle import bundle, sha256_text  # noqa: PLC0415
    src = REPO / diag_build.load_packages()["packages"][package]["production_variant"]
    text, _ = bundle(src)
    exe = OUT / "bin" / f"production_{package}-{sha256_text(text)[:12]}{runner.EXE}"
    if not exe.is_file():
        (OUT / "src").mkdir(parents=True, exist_ok=True)
        s = OUT / "src" / f"production_{package}.cpp"
        s.write_text(text, encoding="utf-8")
        exe.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([runner.find_compiler(), *runner.CXX_FLAGS, str(s), "-o", str(exe)], check=True,
                       capture_output=True, timeout=900)
    return exe


def replies(exe: Path, stream: str, args=()) -> list[str]:
    r = subprocess.run([str(exe), *args], input=stream, capture_output=True, text=True, timeout=600)
    return r.stdout.splitlines()


@unittest.skipUnless(runner.find_compiler(), "no C++ compiler")
class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids = diag_build.build_package("p2general", OUT)
        cls.tool = diag_build.binary_path(cls.ids["root_tool"])
        cls.bot = diag_build.binary_path(cls.ids["continuation_bot"])
        cls.prod = production("p2general")
        text = gzip.decompress(STREAM.read_bytes()).decode()
        cls.blocks = [b + "END\n" for b in text.split("END\n") if b.strip()]

    def root(self, turn: int) -> dict:
        d = Path(tempfile.mkdtemp())
        (d / "s.txt").write_text("".join(self.blocks[:turn]))
        r = subprocess.run([str(self.tool), "root", str(d / "s.txt"), str(turn)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)
        return json.loads(r.stdout)

    def test_the_root_tool_replays_the_production_bot(self):
        for turn in (2, 50, 333):
            with self.subTest(turn=turn):
                stream = "".join(self.blocks[:turn])
                prod = replies(self.prod, stream)[:turn]
                j = self.root(turn)
                self.assertEqual(j["replies"], prod[:-1])
                self.assertEqual(j["default_reply"], prod[-1])
                self.assertEqual(j["candidates"][0]["joint"], j["sampled_joint"])
                self.assertEqual(j["candidates"][0]["reply"], j["default_reply"])
                orders = [c["reply"] for c in j["candidates"]]
                self.assertEqual(len(orders), len(set(orders)))
                self.assertEqual(j["mask"].count("1") >= len(orders), True)
                self.assertTrue(all(len(c["values"]) == 4 for c in j["candidates"]))

    def test_the_root_tool_refuses_a_stream_that_ends_early(self):
        d = Path(tempfile.mkdtemp())
        (d / "s.txt").write_text("".join(self.blocks[:10]))
        r = subprocess.run([str(self.tool), "root", str(d / "s.txt"), "11"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("error", r.stdout)

    def test_the_continuation_bot_is_the_production_bot_until_its_key_turn(self):
        stream = "".join(self.blocks)
        prod = replies(self.prod, stream)
        self.assertEqual(replies(self.bot, stream, ("0", "0")), prod)
        keyed = replies(self.bot, stream, ("987654321", "100"))
        self.assertEqual(keyed[:99], prod[:99])
        self.assertGreater(sum(a != b for a, b in zip(keyed[99:], prod[99:])), 0)
        self.assertEqual(replies(self.bot, stream, ("987654321", "100")), keyed)

    def test_the_first_eight_roots_are_the_8x4_search_tree(self):
        packages = diag_build.load_packages()
        packages["tree_candidates"], packages["tree_samples"] = 8, 4
        small = diag_build.binary_path(diag_build.build_program("p2general", "root_tool", OUT / "tree8", packages))
        d = Path(tempfile.mkdtemp())
        (d / "s.txt").write_text("".join(self.blocks[:120]))
        r = subprocess.run([str(small), "root", str(d / "s.txt"), "120"], capture_output=True, text=True)
        eight, wide = json.loads(r.stdout)["candidates"], self.root(120)["candidates"]
        self.assertEqual(len(eight), 8)
        self.assertEqual(eight, wide[:8])

    def test_the_offline_gate_agrees_with_the_root_tool(self):
        from tools.search_diagnostics.analyze import gated_choice  # noqa: PLC0415
        d = Path(tempfile.mkdtemp())
        (d / "s.txt").write_text("".join(self.blocks[:60]))
        r = subprocess.run([str(self.tool), "gate", str(d / "s.txt"), "0.002"], capture_output=True, text=True)
        rows = [json.loads(x) for x in r.stdout.splitlines()]
        self.assertEqual([x["turn"] for x in rows], list(range(1, 61)))
        for turn in (20, 40, 60):
            j = self.root(turn)
            means = [sum(c["values"]) / len(c["values"]) for c in j["candidates"][:8]]
            self.assertEqual(rows[turn - 1]["pick"], gated_choice(means, 0.002))

    def test_pmarg_prints_valid_distributions(self):
        from tools.search_diagnostics.p_eval import check_distribution, run_pmarg  # noqa: PLC0415
        rows = run_pmarg(self.tool, "".join(self.blocks[:30]))
        self.assertEqual(len(rows), 30)
        units = [u for r in rows for u in r["units"]]
        self.assertTrue(units)
        for u in units:
            check_distribution(u["move"], "move")
            for m in range(5):
                if u["legal"][m] == "1":
                    check_distribution(u["act"][m], "act")


@unittest.skipUnless(runner.find_compiler() and arena(), "needs a C++ compiler and an arena checkout (AI_TEST_ROOT)")
class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fusion.paths import bot_path, engine_path, git_commit, load_engine, load_pool, sha256  # noqa: PLC0415
        from training.collect import Fixture, play  # noqa: PLC0415
        from tools.search_diagnostics.extract import extract_root, select_roots  # noqa: PLC0415
        cls.root_dir = arena()
        cls.engine = load_engine(cls.root_dir)
        cls.ids = diag_build.build_package("p2general", OUT)
        prod = production("p2general")
        pool = {e["name"]: e for e in load_pool()["bots"]}
        opp = bot_path(pool["Baseline"], cls.root_dir)
        commands = {"@candidate": [str(prod)], "Baseline": runner.command_for(opp, REPO / "build" / "roster")}
        info = {"@candidate": {"family": "agent_v2", "sha256": sha256(prod)},
                "Baseline": {"family": "baseline", "sha256": sha256(opp)},
                "__engine__": sha256(engine_path(cls.root_dir)), "__commit__": git_commit(cls.root_dir)}
        cls.tmp = Path(tempfile.mkdtemp())
        fx = Fixture("t0085-n02-m00000", 2, 31337, 4242, ["@candidate", "Baseline"])
        cls.match = play((fx, commands, info, cls.root_dir, (2.0, 0.05)))
        if cls.match["faults"]:
            raise unittest.SkipTest(f"the source game had faults on this machine: {cls.match['faults'][:2]}")
        cls.match_path = cls.tmp / "n02" / f"{fx.match_id}.json.gz"
        matchlog.save(cls.match_path, cls.match)
        tool = diag_build.binary_path(cls.ids["root_tool"])
        plan = select_roots(cls.match, 0, [("unconditioned", 2)], "test")
        cls.roots_dir = cls.tmp / "roots"
        cls.roots = []
        for row in plan:
            r, leaves = extract_root(cls.match_path, cls.match, 0, row, cls.ids["root_tool"], tool, "test")
            d = cls.roots_dir / "n02"
            d.mkdir(parents=True, exist_ok=True)
            (d / r["leaf_features"]["file"]).write_bytes(leaves)
            from tools.search_diagnostics.schemas import write_gz  # noqa: PLC0415
            write_gz(d / f"{r['root_id']}.json.gz", r)
            cls.roots.append(r)
        cls.root = cls.roots[0]
        from tools.search_diagnostics.branch import commands_for  # noqa: PLC0415
        cls.commands = commands_for(cls.match, 0, diag_build.binary_path(cls.ids["continuation_bot"]), cls.root_dir,
                                    REPO / "build" / "roster")

    def play(self, match=None, commands=None, candidate=0, keys=(11, 22)):
        from tools.search_diagnostics.branch import play_branch  # noqa: PLC0415
        return play_branch(self.engine, match or self.match, self.root, candidate, keys[0], keys[1],
                           commands or self.commands)

    def test_a_branch_is_faithful_reproducible_and_keyed(self):
        a, b = self.play(), self.play()
        self.assertEqual(a["final_score"], b["final_score"])
        self.assertEqual((a["final_turn"], a["terminal"], a["faults"]["prefix"], a["faults"]["root"]), (400, True, [], []))
        others = {tuple(self.play(keys=(k, k + 1))["final_score"]) for k in (101, 202, 303, 404)}
        cands = {tuple(self.play(candidate=c)["final_score"]) for c in range(1, 5)}
        self.assertGreater(len(others | cands | {tuple(a["final_score"])}), 1)

    def test_a_diverging_process_refuses_the_branch(self):
        from tools.search_diagnostics.branch import BranchRefused  # noqa: PLC0415
        chaos = copy.deepcopy(self.commands)
        chaos[1] = [sys.executable, str(HERE / "chaos_bot.py")]
        with self.assertRaisesRegex(BranchRefused, "diverged from the record"):
            self.play(commands=chaos)

    def test_a_changed_record_refuses_the_branch(self):
        from tools.search_diagnostics.branch import BranchRefused  # noqa: PLC0415
        m = copy.deepcopy(self.match)
        m["blocks"][3] = m["blocks"][3].replace("SCORE 0 0", "SCORE 1 0")
        with self.assertRaisesRegex(BranchRefused, "input differs"):
            self.play(match=m)
        m = copy.deepcopy(self.match)
        m["init"] = m["init"].replace("MAP", "MAP ", 1)
        with self.assertRaisesRegex(BranchRefused, "turn-1 input differs"):
            self.play(match=m)
        m = copy.deepcopy(self.match)
        m["replies"][4][1] = None
        with self.assertRaisesRegex(BranchRefused, "recorded fault"):
            self.play(match=m)
        m = copy.deepcopy(self.match)
        m["faults"] = [[3, 1, "timeout (>50 ms)"]]
        with self.assertRaisesRegex(BranchRefused, "recorded a fault"):
            self.play(match=m)

    def test_a_malformed_recorded_root_reply_is_never_a_complete_branch(self):
        from tools.search_diagnostics.branch import BranchRefused, tokens  # noqa: PLC0415
        t = self.root["turn"]
        m = copy.deepcopy(self.match)
        m["replies"][t - 1][1] = "999 " + " ".join(tokens(m["replies"][t - 1][1])[1:])
        with self.assertRaisesRegex(BranchRefused, "diverged"):
            self.play(match=m)

    def test_another_engine_is_refused(self):
        from tools.search_diagnostics.branch import BranchRefused, commands_for  # noqa: PLC0415
        m = copy.deepcopy(self.match)
        m["engine_sha256"] = "0" * 64
        with self.assertRaisesRegex(BranchRefused, "engine differs"):
            commands_for(m, 0, diag_build.binary_path(self.ids["continuation_bot"]), self.root_dir, REPO / "build" / "roster")

    def test_collection_binds_resumes_reuses_and_never_replays(self):
        from tools.search_diagnostics.analyze import analyze  # noqa: PLC0415
        from tools.search_diagnostics.collect import collect, row_path  # noqa: PLC0415
        from tools.search_diagnostics.schemas import read_gz, write_gz  # noqa: PLC0415
        base = Path(tempfile.mkdtemp())
        ledger = base / "TEST_SEEDS.md"
        ledger.write_text("| 31415926 | test | Reserved |\n| 27182818 | other test | Reserved |\n")
        args = dict(roots_dir=self.roots_dir, matches_dir=self.tmp, build=self.ids, ai_test=self.root_dir, repeats=1,
                    seed=31415926, ledger=ledger, workers=2, dev=True)
        first = collect(out=base / "primary", cand_set="primary", **args)
        self.assertEqual((first["played"], first["episodes_played"]), (2, 16))
        again = collect(out=base / "primary", cand_set="primary", **args)
        self.assertEqual((again["kept"], again["played"]), (2, 0))
        ext = collect(out=base / "extended", cand_set="extended", reuse_dir=base / "primary", **args)
        extra = sum(len(r["candidate_sets"]["extended"]) - len(r["candidate_sets"]["primary"]) for r in self.roots)
        self.assertEqual(ext["episodes_played"], extra)
        with self.assertRaisesRegex(ValidationError, "already consumed"):
            collect(out=base / "copy", cand_set="primary", **args)
        out = analyze(self.roots_dir, base / "extended")
        self.assertEqual(out["analysed"] + sum(out["excluded"].values()), 2)
        p = row_path(base / "primary", self.root)
        row = read_gz(p)
        row["branches"][0]["utility"] = 1.0 - row["branches"][0]["utility"]
        write_gz(p, row)
        with self.assertRaises(ValidationError):
            collect(out=base / "primary", cand_set="primary", **args)

    def test_extraction_writes_the_leaf_features_it_records(self):
        from tools.search_diagnostics.schemas import verify_leaves  # noqa: PLC0415
        for r in self.roots:
            p = self.roots_dir / "n02" / r["leaf_features"]["file"]
            verify_leaves(r, p)
            self.assertEqual(p.stat().st_size, r["leaf_features"]["rows"] * 416 * 4)
            self.assertGreater(r["leaf_features"]["rows"], 0)

    def test_extraction_refuses_a_seat_the_package_did_not_play(self):
        from tools.search_diagnostics.extract import RootRefused, extract_root  # noqa: PLC0415
        m = copy.deepcopy(self.match)
        line = m["replies"][9][0]
        m["replies"][9][0] = line.split()[0] + " S - S - S -" if line.split()[1:] != ["S", "-"] * 3 else line.split()[0] + " U - U - U -"
        tool = diag_build.binary_path(self.ids["root_tool"])
        row = {"turn": self.root["turn"], "stratum": "unconditioned", "eligible": 1, "weight": 1.0}
        with self.assertRaisesRegex(RootRefused, "does not reproduce"):
            extract_root(self.match_path, m, 0, row, self.ids["root_tool"], tool, "test")

    def test_the_override_stratum_reads_the_offline_gate(self):
        from tools.search_diagnostics.extract import run_gate, select_roots  # noqa: PLC0415
        tool = diag_build.binary_path(self.ids["root_tool"])
        overrides = run_gate(tool, self.match, 0, 0.002)
        rows = select_roots(self.match, 0, [("override", 2)], "test", overrides)
        self.assertTrue(all(r["turn"] in overrides for r in rows))
        with self.assertRaises(ValidationError):
            select_roots(self.match, 0, [("override", 1)], "test")

    def test_collection_refuses_a_changed_source_file(self):
        from tools.search_diagnostics.collect import _job  # noqa: PLC0415
        changed = self.tmp / "changed.json.gz"
        m = copy.deepcopy(self.match)
        m["created"] = "later"
        matchlog.save(changed, m)
        with self.assertRaisesRegex(ValidationError, "changed since"):
            _job((self.root, changed, "primary", [0], 1, 5, self.ids, str(self.root_dir), "id", None))

    def test_parity_passes_with_default_coverage_and_gates_on_it(self):
        from tools.search_diagnostics.parity import COVERAGE, audit, build_check  # noqa: PLC0415
        exe = build_check(OUT / "check")
        rep = audit(self.engine, exe, [2, 5, 8, 15], 400, 4242)
        self.assertEqual((rep["status"], rep["differences"], rep["short_of_minimum"]), ("passed", 0, {}))
        self.assertTrue(all(rep["coverage"][k] >= 1 for k in COVERAGE))
        strict = audit(self.engine, exe, [2], 1, 7)
        self.assertEqual(strict["status"], "insufficient_coverage")


if __name__ == "__main__":
    unittest.main()
