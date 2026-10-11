"""fusion.paths finds the engine and the judge bots in either checkout layout, and says so when it substitutes a bot."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from fusion import paths

JUDGE = "ctf-arena/Player7/judge/ctf/bots/opus-5.5.cpp"


def touch(root: Path, rel: str, text: str = "") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


class TestArenaLayout(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        paths.SUBSTITUTED.clear()

    def tearDown(self):
        self._tmp.cleanup()
        paths.SUBSTITUTED.clear()

    def test_public_layout_is_accepted(self):
        touch(self.root, "ctf/workspace/engine.py")
        self.assertEqual(paths.ai_test_root(self.root), self.root.resolve())
        self.assertEqual(paths.workspace(self.root), self.root / "ctf" / "workspace")
        self.assertEqual(paths.engine_path(self.root), self.root / "ctf" / "workspace" / "engine.py")
        self.assertFalse(paths.has_legacy_planner(self.root))

    def test_ai_test_layout_is_accepted(self):
        touch(self.root, "ctf-arena/Player7/workspace/engine.py")
        touch(self.root, "ctf-arena/Player7/workspace/experiments/round3/nextgen/tests/forecast_dump.cpp")
        self.assertEqual(paths.workspace(self.root), self.root / "ctf-arena" / "Player7" / "workspace")
        self.assertTrue(paths.has_legacy_planner(self.root))

    def test_neither_layout_names_both(self):
        with self.assertRaises(SystemExit) as e:
            paths.ai_test_root(self.root)
        msg = str(e.exception)
        self.assertIn("ctf/workspace/engine.py", msg)
        self.assertIn("ctf-arena/Player7/workspace/engine.py", msg)
        self.assertIn("chen-006/meow-ai-arena", msg)

    def test_judge_bot_is_taken_from_the_public_repo_and_reported(self):
        touch(self.root, "ctf/workspace/engine.py")
        public = touch(self.root, "ctf/bots/opus-5.5.cpp", "// the original, with comments\nint main(){}\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = paths.bot_path({"path": JUDGE}, self.root)
            again = paths.bot_path({"path": JUDGE}, self.root)
        self.assertEqual((got, again), (public, public))
        self.assertEqual(paths.SUBSTITUTED, {JUDGE: str(public)})
        self.assertEqual(err.getvalue().count("[paths] judge bot"), 1)           # said once, not per call

    def test_the_ai_test_copy_wins_and_nothing_is_substituted(self):
        touch(self.root, "ctf/workspace/engine.py")
        touch(self.root, "ctf/bots/opus-5.5.cpp", "public")
        original = touch(self.root, JUDGE, "archived")
        self.assertEqual(paths.bot_path({"path": JUDGE}, self.root), original)
        self.assertEqual(paths.SUBSTITUTED, {})

    def test_baseline_maps_to_the_identical_public_file_without_a_report(self):
        touch(self.root, "ctf/workspace/engine.py")
        public = touch(self.root, "ctf/workspace/baseline.cpp", "int main(){}")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = paths.bot_path({"path": "ctf-stress-pack/bots/baseline.cpp"}, self.root)
        self.assertEqual(got, public)
        self.assertEqual((paths.SUBSTITUTED, err.getvalue()), ({}, ""))

    def test_other_bots_are_never_substituted(self):
        touch(self.root, "ctf/workspace/engine.py")
        touch(self.root, "ctf/bots/opus-5.5.cpp", "x")
        entry = {"path": "ctf-stress-pack/bots/Player5.cpp"}
        self.assertEqual(paths.bot_path(entry, self.root), self.root / "ctf-stress-pack/bots/Player5.cpp")
        self.assertEqual(paths.bot_path({"path": "@repo/tools/random_bot.py"}, self.root), paths.REPO / "tools" / "random_bot.py")


if __name__ == "__main__":
    unittest.main()
