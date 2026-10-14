"""The repository's order is a tested contract (CONTRIBUTING.md): where files live, what is never tracked, and that
markdown links resolve. When a test here fails after you add a folder, update ROOT_DIRS (and the README layout)."""
from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]

ROOT_DIRS = {".github", "bots", "docs", "fusion", "inference", "profiles", "reports", "roster", "tasks", "tests", "tools", "training"}
ROOT_FILES = {".gitattributes", ".gitignore", "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", "README.md", "compile_flags.txt", "requirements-dev.txt", "requirements.txt"}
TOOLS_DIRS = {"cpp", "host", "pipeline", "reporting", "repo", "screens"}
TOOLS_FILES = {"random_bot.py", "noise_bot.py", "README.md"}   # the bots are named by path in committed pools: do not move
REPORT_DIRS = {"models", "host", "eval", "reviews"}
TIERS = {"s", "m", "l", "xl", "xxl"}
MODEL_STUDIES = {"v_pilot", "p_features", "v_capacity_5xl", "p_ood"}   # controlled model comparisons (docs/FEATURE_FAMILIES.md, docs/V_CAPACITY_5XL.md)
LOCAL_ONLY = ("build/", "data/", "runs/")
VENDORED = ("roster/bots/src/", "roster/new_bots/")


def moved_task(path: Path) -> bool:
    """A link to a task file is valid while the task exists in tasks/ or tasks/archive/: tools/repo/task.py moves the
    file between them with the task's status, and the task id is what the link means."""
    if not re.fullmatch(r"T\d{4}-[a-z0-9-]+\.md", path.name):
        return False
    folder = path.parent.resolve()
    tasks = folder.parent if folder.name == "archive" else folder
    return tasks == (REPO / "tasks").resolve() and ((tasks / path.name).exists() or (tasks / "archive" / path.name).exists())


def tracked() -> list[str]:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        raise unittest.SkipTest("not a git checkout")
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n") if f and (REPO / f).exists()]


class TestRepoLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = tracked()

    def test_root_is_tidy(self):
        top = {f.split("/")[0] for f in self.files}
        dirs = {f.split("/")[0] for f in self.files if "/" in f}
        files = {f for f in self.files if "/" not in f}
        self.assertEqual(sorted(dirs - ROOT_DIRS), [], "new top-level folder: add it to ROOT_DIRS and the README layout")
        self.assertEqual(sorted(files - ROOT_FILES), [], "tracked file in the repository root: move it into a folder")
        self.assertEqual(sorted(top & set(LOCAL_ONLY)), [])

    def test_tools_are_grouped(self):
        for f in self.files:
            parts = f.split("/")
            if parts[0] != "tools":
                continue
            ok = (len(parts) == 2 and parts[1] in TOOLS_FILES) or (len(parts) >= 3 and parts[1] in TOOLS_DIRS)
            self.assertTrue(ok, f"{f}: tools/ holds only {sorted(TOOLS_DIRS)} subfolders and {sorted(TOOLS_FILES)}")

    def test_reports_are_grouped(self):
        for f in self.files:
            parts = f.split("/")
            if parts[0] != "reports":
                continue
            if parts == ["reports", "README.md"]:
                continue
            self.assertGreaterEqual(len(parts), 3, f"{f}: reports/ holds only {sorted(REPORT_DIRS)} subfolders")
            self.assertIn(parts[1], REPORT_DIRS, f)
            if parts[1] == "models":
                self.assertIn(parts[2], TIERS | MODEL_STUDIES,
                              f"{f}: reports/models/<tier or study>/ with one of {sorted(TIERS | MODEL_STUDIES)}")
            self.assertRegex(parts[-1], r"\.(md|json|jsonl)$", f"{f}: reports/ tracks .md, .json and .jsonl only")

    def test_local_only_never_tracked(self):
        for f in self.files:
            self.assertFalse(f.startswith(LOCAL_ONLY) or re.match(r"runs_[^/]+/", f), f)
            self.assertNotIn("__pycache__", f)
            self.assertFalse(f.endswith((".pyc", ".DS_Store")), f)

    def test_models_dirs_are_complete(self):
        """Every exported tier directory carries its registry, manifest and one header per model."""
        for tier in ("models", "models_m", "models_l", "models_xl", "models_xxl"):
            names = {Path(f).name for f in self.files if f.startswith(f"inference/{tier}/")}
            self.assertIn("registry.hpp", names, tier)
            self.assertIn("MANIFEST.json", names, tier)
            self.assertIn("weights_general.hpp", names, tier)

    def test_markdown_links_resolve(self):
        link = re.compile(r"\]\(([^)\s]+)\)")
        broken = []
        for f in self.files:
            if not f.endswith(".md") or f.startswith(VENDORED):
                continue
            fence = False
            for n, line in enumerate((REPO / f).read_text(encoding="utf-8", errors="replace").split("\n"), 1):
                if line.lstrip().startswith("```"):
                    fence = not fence
                if fence:
                    continue
                for m in link.finditer(line):
                    target = m.group(1)
                    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "<")):
                        continue
                    target = unquote(target.split("#")[0])
                    if target and not ((REPO / f).parent / target).exists() and not moved_task((REPO / f).parent / target):
                        broken.append(f"{f}:{n}: {m.group(1)}")
        self.assertEqual(broken, [], "broken markdown links")


if __name__ == "__main__":
    unittest.main()
