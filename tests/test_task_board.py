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

    def test_lock_options_before_or_after_the_name(self):
        self.assertEqual(task.split_lock_args(["--task", "T0019", "--", "python", "x.py"]), ("T0019", 0.0, ["python", "x.py"]))
        self.assertEqual(task.split_lock_args(["--", "python", "-m", "a", "--", "b"]), ("", 0.0, ["python", "-m", "a", "--", "b"]))
        self.assertEqual(task.split_lock_args(["--wait", "60", "--task", "T1", "--", "c"]), ("T1", 60.0, ["c"]))
        self.assertEqual(task.split_lock_args(["python", "x.py"]), ("", 0.0, ["python", "x.py"]))
        with self.assertRaisesRegex(task.BoardError, "unknown options"):
            task.split_lock_args(["--bogus", "--", "c"])

    def test_dot_paths_keep_their_dot(self):
        self.assertEqual(task.norm(".github/"), ".github/")
        self.assertEqual(task.norm("./docs\\x.md"), "docs/x.md")
        self.assertTrue(task.covers(".github/", ".github/workflows/ci.yml"))
        self.assertFalse(task.covers("github/", ".github/workflows/ci.yml"))

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

    def test_amend_widens_lanes_only_where_free(self):
        task.op_claim(self.board(self.a), "T0003", "codex@host", set())          # docs/
        task.op_claim(self.board(self.b), "T0002", "claude@mac", set())          # bots/p_only/
        self.assertEqual(task.op_amend(self.board(self.a), "T0003", "codex@host", [".github/"]), [".github/", "docs/"])
        with self.assertRaisesRegex(task.BoardError, "active task T0002"):
            task.op_amend(self.board(self.a), "T0003", "codex@host", ["bots/p_only/report/"])
        with self.assertRaisesRegex(task.BoardError, "only the owner"):
            task.op_amend(self.board(self.b), "T0003", "claude@mac", ["docs/y.md"])
        self.assertIn("amended writes", self.board(self.a).tasks(self.board(self.a).fetch())["T0003"][2])

    def test_every_board_change_rewrites_the_summary(self):
        board = self.board(self.a)
        task.op_claim(board, "T0003", "codex@host", set())
        rev = board.fetch()
        summary = task.git("show", f"{rev}:tasks/SUMMARY.md", cwd=self.a)
        self.assertEqual(summary.strip(), task.render_summary(board.tasks(rev)).strip())
        self.assertIn("1 active", summary)
        errors, warnings = task.check(board.tasks(rev), summary)
        self.assertEqual((errors, warnings), ([], []))
        _, stale = task.check(board.tasks(rev), "# old summary")
        self.assertTrue(any("SUMMARY.md is behind" in w for w in stale))

    def test_merge_folds_duplicates_and_repoints_dependents(self):
        t4 = task.op_new(self.board(self.b), "claude@mac", "Depends on the duplicate", [], ["docs/y.md"], ["T0002"], "direct", "", "")
        self.assertEqual(task.op_merge(self.board(self.a), "T0001", ["T0002"], "codex@host", "same screen"), "T0001")
        tasks = self.board(self.b).tasks(self.board(self.b).fetch())
        _, keep, kbody = tasks["T0001"]
        _, dup, dbody = tasks["T0002"]
        self.assertEqual(dup["status"], "dropped")
        self.assertIn("Merged into T0001", dbody)
        self.assertEqual(keep["writes"], ["bots/p_only/", "bots/p_only/report/"])           # lanes carried over
        self.assertIn("Merged from T0002", kbody)
        self.assertIn("merged T0002 into this task (same screen)", kbody)
        self.assertEqual(tasks[t4][1]["depends"], ["T0001"])                                # dependents re-pointed
        self.assertIn("1 of them merged", task.git("show", "refs/remotes/origin/main:tasks/SUMMARY.md", cwd=self.b))
        self.assertEqual(task.check(tasks)[0], [])

    def test_merge_refuses_held_duplicates_and_losing_open_work(self):
        task.op_claim(self.board(self.a), "T0003", "codex@host", set())
        with self.assertRaisesRegex(task.BoardError, "held by codex@host"):
            task.op_merge(self.board(self.b), "T0001", ["T0003"], "claude@mac")
        task.op_update(self.board(self.a), "T0003", "codex@host", status="done", verb="done")
        with self.assertRaisesRegex(task.BoardError, "would disappear"):
            task.op_merge(self.board(self.b), "T0003", ["T0002"], "claude@mac")             # open work into a done task
        task.op_update(self.board(self.b), "T0002", "claude@mac", status="dropped", verb="dropped")
        task.op_merge(self.board(self.b), "T0003", ["T0002"], "claude@mac")                 # closed into closed
        _, keep, _ = self.board(self.a).tasks(self.board(self.a).fetch())["T0003"]
        self.assertEqual(keep["writes"], ["docs/"])                                       # its record is unchanged
        with self.assertRaisesRegex(task.BoardError, "at least one duplicate"):
            task.op_merge(self.board(self.b), "T0001", ["T0001"], "claude@mac")

    def test_amend_definition_while_nobody_holds_it(self):
        lanes = task.op_amend(self.board(self.b), "T0002", "claude@mac", title="Tune, narrowed", depends=["T0003"],
                              goal="Only the tuning part; the screen moved to T0001.")
        self.assertEqual(lanes, ["bots/p_only/"])
        _, meta, body = self.board(self.a).tasks(self.board(self.a).fetch())["T0002"]
        self.assertEqual((meta["title"], meta["depends"]), ("Tune, narrowed", ["T0003"]))
        self.assertIn("Only the tuning part", body)
        self.assertIn("amended depends -> T0003; title -> Tune, narrowed; goal", body)
        with self.assertRaisesRegex(task.BoardError, "cannot depend on itself"):
            task.op_amend(self.board(self.b), "T0002", "claude@mac", depends=["T0002"])

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


    def files(self, clone) -> list[str]:
        b = self.board(clone)
        return sorted(task.git("ls-tree", "-r", "--name-only", b.fetch(), "tasks", cwd=clone).split("\n"))

    def test_finished_tasks_move_to_the_archive_and_back(self):
        task.op_update(self.board(self.a), "T0003", "codex@host", status="done", handoff="result: docs/x.md", verb="done")
        files = self.files(self.b)
        self.assertIn("tasks/archive/T0003-docs.md", files)
        self.assertNotIn("tasks/T0003-docs.md", files)
        b = self.board(self.b)
        name, meta, _ = b.tasks(b.fetch())["T0003"]
        self.assertEqual((name, meta["status"]), ("archive/T0003-docs.md", "done"))
        self.assertIn("(archive/T0003-docs.md)", task.git("show", f"{b.ref}:tasks/SUMMARY.md", cwd=self.b))
        task.op_update(self.board(self.b), "T0002", "claude@mac", status="dropped", verb="dropped")
        self.assertIn("tasks/archive/T0002-tune.md", self.files(self.a))
        task.op_update(self.board(self.a), "T0003", "codex@host", status="open", handoff="reopened: X is missing", verb="handoff")
        files = self.files(self.a)
        self.assertIn("tasks/T0003-docs.md", files)                      # reopened: back among the live tasks
        self.assertNotIn("tasks/archive/T0003-docs.md", files)
        self.assertEqual(task.op_new(self.board(self.b), "claude@mac", "next", [], ["x/"], [], "pr", "", ""), "T0004")
        task.op_claim(self.board(self.b), "T0003", "claude@mac", set())  # claims and lanes work across both folders

    def test_merged_duplicates_are_archived(self):
        task.op_merge(self.board(self.a), "T0001", ["T0002"], "codex@host", "same screen")
        files = self.files(self.b)
        self.assertIn("tasks/archive/T0002-tune.md", files)
        self.assertIn("tasks/T0001-screen.md", files)

    def test_archive_command_and_readme_board(self):
        seed = self.a
        (seed / "tasks" / "T0004-old.md").write_text(task_text("T0004", "old", ["old/"], status="done", owner="x@y"), encoding="utf-8")
        (seed / "tasks" / "README.md").write_text(f"# The task board\n\nintro\n\n{task.BOARD_START}\n{task.BOARD_END}\n\nmore\n",
                                                encoding="utf-8")
        run("add", "-A", cwd=seed)
        run("commit", "-q", "-m", "a finished task in the old place, and a README", cwd=seed)
        run("push", "-q", "origin", "HEAD:main", cwd=seed)
        errors, warnings = task.check(task.read_worktree(seed))
        self.assertEqual(errors, [])
        self.assertTrue(any("T0004 not where the status puts them" in w for w in warnings))
        self.assertEqual(task.op_archive(self.board(self.b), "claude@mac"), ["T0004"])
        self.assertIn("tasks/archive/T0004-old.md", self.files(self.b))
        self.assertEqual(task.op_archive(self.board(self.b), "claude@mac"), [])     # nothing left to move
        b = self.board(self.b)
        readme = task.git("show", f"{b.fetch()}:tasks/README.md", cwd=self.b)
        self.assertTrue(readme.startswith("# The task board\n\nintro"))
        self.assertTrue(readme.rstrip().endswith("more"))
        self.assertIn("[T0004](archive/T0004-old.md)", readme)             # recently finished
        task.op_claim(self.board(self.a), "T0003", "codex@host", set())
        readme = task.git("show", f"{b.fetch()}:tasks/README.md", cwd=self.b)
        self.assertIn("| [T0003](T0003-docs.md) | active | codex@host | docs |", readme)   # every board change rewrites it
        run("pull", "-q", "origin", "main", cwd=self.b)
        tasks = task.read_worktree(self.b)
        self.assertEqual(task.check(tasks, readme=(self.b / "tasks" / "README.md").read_text(encoding="utf-8"))[1], [])


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
