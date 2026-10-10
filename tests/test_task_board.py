"""tools/repo/task.py: the committed board is valid, and claims, handoffs, lanes and locks hold under concurrency.

The concurrency tests run two clones of a throwaway bare repository, standing in for two agents on two machines."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "repo"))
import task  # noqa: E402

GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def run(*args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def task_text(tid, title, writes, status="open", owner="", needs=(), depends=()):
    meta = {"id": tid, "title": title, "status": status, "owner": owner, "needs": list(needs), "writes": list(writes),
            "depends": list(depends), "land": "direct", "branch": "", "updated": "2026-10-10"}
    return task.render(meta, "\n## Goal\n\ng\n\n## Done when\n\nd\n\n## Handoff\n\nNot started.\n\n## Log\n\n- created\n")


class TestCommittedBoard(unittest.TestCase):
    def test_board_is_valid(self):
        errors, _ = task.check(task.read_worktree(REPO))
        self.assertEqual(errors, [])

    def test_parse_render_round_trip(self):
        text = task_text("T0001", "A: title, with commas", ["bots/p_only/report/"], needs=["host"])
        meta, body = task.parse(text)
        self.assertEqual(meta["title"], "A: title, with commas")
        self.assertEqual(task.render(meta, body), text)

    def test_lanes(self):
        self.assertTrue(task.covers("bots/p_only/", "bots/p_only/report/x.md"))
        self.assertTrue(task.covers("bots/p_only", "bots/p_only/report/x.md"))
        self.assertFalse(task.covers("bots/p_only/report", "bots/p_only/reports.md"))
        self.assertTrue(task.overlaps(["bots/"], ["bots/p_only/report/"]))
        self.assertFalse(task.overlaps(["bots/hybrid/"], ["bots/p_only/"]))
        self.assertEqual(task.outside_lanes(["bots/p_only/a.py", "fusion/x.py"], ["bots/p_only/"]), ["fusion/x.py"])


class TestConcurrentAgents(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ)
        os.environ.update(GIT_ENV)
        self.tmp = Path(tempfile.mkdtemp())
        origin = self.tmp / "origin.git"
        run("init", "-q", "--bare", "-b", "main", str(origin), cwd=self.tmp)
        seed = self.tmp / "seed"
        run("clone", "-q", str(origin), str(seed), cwd=self.tmp)
        (seed / "tasks").mkdir()
        (seed / "tasks" / "T0001-screen.md").write_text(task_text("T0001", "screen", ["bots/p_only/report/"], needs=["host"]), encoding="utf-8")
        (seed / "tasks" / "T0002-tune.md").write_text(task_text("T0002", "tune", ["bots/p_only/"]), encoding="utf-8")
        (seed / "tasks" / "T0003-docs.md").write_text(task_text("T0003", "docs", ["docs/"]), encoding="utf-8")
        run("add", "-A", cwd=seed)
        run("commit", "-q", "-m", "seed", cwd=seed)
        run("push", "-q", "origin", "HEAD:main", cwd=seed)
        self.a, self.b = self.tmp / "a", self.tmp / "b"
        for c in (self.a, self.b):
            run("clone", "-q", str(origin), str(c), cwd=self.tmp)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def board(self, clone):
        return task.Board(clone)

    def status(self, clone, tid):
        b = self.board(clone)
        return b.tasks(b.fetch())[tid][1]

    def test_only_one_agent_wins_a_claim(self):
        task.op_claim(self.board(self.a), "T0001", "codex@host", {"host"})
        with self.assertRaisesRegex(task.BoardError, "held by codex@host"):
            task.op_claim(self.board(self.b), "T0001", "claude@mac", {"host"})
        self.assertEqual(self.status(self.b, "T0001")["owner"], "codex@host")

    def test_needs_and_lanes_are_enforced(self):
        with self.assertRaisesRegex(task.BoardError, "needs host"):
            task.op_claim(self.board(self.b), "T0001", "claude@mac", set())
        task.op_claim(self.board(self.a), "T0001", "codex@host", {"host"})
        with self.assertRaisesRegex(task.BoardError, "active task T0001"):
            task.op_claim(self.board(self.b), "T0002", "claude@mac", set())    # bots/p_only/ contains T0001's lane
        task.op_claim(self.board(self.b), "T0003", "claude@mac", set())        # docs/ does not overlap

    def test_lost_race_is_recomputed_not_overwritten(self):
        """B pushes between A's read and A's push: A's push is rejected, A recomputes on B's main, both changes survive."""
        board_a, raced = self.board(self.a), []

        def edit(tasks):
            if not raced:
                raced.append(1)
                task.op_claim(self.board(self.b), "T0003", "claude@mac", set())
            name, meta, body = tasks["T0001"]
            return {name: task.render(meta, task.append_log(body, "- a note from codex"))}, None

        board_a.cas(edit, "tasks: note")
        self.assertEqual(len(raced), 1)
        tasks = board_a.tasks(board_a.fetch())
        self.assertIn("a note from codex", tasks["T0001"][2])
        self.assertEqual(tasks["T0003"][1]["owner"], "claude@mac")

    def test_handoff_releases_and_new_numbers_after_the_last(self):
        task.op_claim(self.board(self.a), "T0003", "codex@host", set())
        with self.assertRaisesRegex(task.BoardError, "only the owner"):
            task.op_update(self.board(self.b), "T0003", "claude@mac", status="done", verb="done")
        task.op_update(self.board(self.a), "T0003", "codex@host", status="open", handoff="half done; next: X", verb="handoff")
        meta = self.status(self.b, "T0003")
        self.assertEqual((meta["status"], meta["owner"]), ("open", ""))
        tid = task.op_new(self.board(self.b), "claude@mac", "New thing", [], ["docs/x.md"], ["T0003"], "direct", "", "")
        self.assertEqual(tid, "T0004")
        errors, _ = task.check(self.board(self.a).tasks(self.board(self.a).fetch()))
        self.assertEqual(errors, [])

    def test_land_refuses_files_outside_the_lane(self):
        task.op_claim(self.board(self.a), "T0003", "codex@host", set())
        (self.a / "docs").mkdir()
        (self.a / "docs" / "ok.md").write_text("ok\n", encoding="utf-8")
        (self.a / "fusion.py").write_text("x = 1\n", encoding="utf-8")
        run("add", "-A", cwd=self.a)
        run("commit", "-q", "-m", "work", cwd=self.a)
        with self.assertRaisesRegex(task.BoardError, "outside: fusion.py"):
            task.op_land(self.board(self.a), "T0003", "codex@host", test=None)
        run("rm", "-q", "fusion.py", cwd=self.a)
        run("commit", "-q", "-m", "keep to the lane", cwd=self.a)
        head = task.op_land(self.board(self.a), "T0003", "codex@host", test=None)
        board_b = self.board(self.b)
        tasks = board_b.tasks(board_b.fetch())
        self.assertIn(f"landed: {head[:10]}", tasks["T0003"][2])
        self.assertEqual(task.git("show", "refs/remotes/origin/main:docs/ok.md", cwd=self.b), "ok")


class TestMachineLock(unittest.TestCase):
    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("CTF_LOCK_DIR")
            os.environ["CTF_LOCK_DIR"] = tmp
            try:
                with task.machine_lock("cpu", {"agent": "codex@host"}):
                    with self.assertRaisesRegex(task.BoardError, "codex@host"):
                        with task.machine_lock("cpu", {"agent": "claude@host"}):
                            pass
                with task.machine_lock("cpu", {"agent": "claude@host"}):      # released after the first run
                    pass
            finally:
                os.environ.pop("CTF_LOCK_DIR") if old is None else os.environ.__setitem__("CTF_LOCK_DIR", old)


if __name__ == "__main__":
    unittest.main()
