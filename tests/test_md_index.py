"""tools/repo/md_index.py: the Markdown index lists every tracked file once and maps it to its DIGEST.md section.

The committed docs/MARKDOWN_INDEX.md is deliberately not required to be fresh: every task that adds a Markdown file
would then have to write the index too, and two such tasks could never be active at once. CI reports a stale index
as a warning instead (`md_index.py --check`)."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "repo"))
import md_index  # noqa: E402
import task  # noqa: E402

DIGEST = """# Digest

## First topic

**Sources:** [a.md](../a.md), [docs/](./), [elsewhere](https://example.com/x.md)

A link that is not on a Sources line: [c](../b/c.md).

## Second: topic, with punctuation

**Sources:** [b/](../b/), [b/c.md](../b/c.md), [b/d/](../b/d/)
"""


class TestMdIndex(unittest.TestCase):
    def test_sources_map_files_to_sections(self):
        sources = md_index.digest_sources(DIGEST)
        self.assertEqual(sources, [("a.md", "First topic"), ("docs/", "First topic"),
                                   ("b/", "Second: topic, with punctuation"),
                                   ("b/c.md", "Second: topic, with punctuation"),
                                   ("b/d/", "Second: topic, with punctuation")])
        section = lambda path: md_index.section_for(path, sources)                     # noqa: E731
        self.assertEqual(section("a.md"), "First topic")
        self.assertEqual(section("docs/x/y.md"), "First topic")                       # a folder link covers its tree
        self.assertEqual(section("b/c.md"), "Second: topic, with punctuation")        # an exact link wins
        self.assertEqual(section("b/d/e.md"), "Second: topic, with punctuation")
        self.assertIsNone(section("c.md"))
        self.assertIsNone(section("bb/x.md"))                                          # 'b/' is not a prefix of 'bb/'

    def test_longest_folder_and_exact_links_win(self):
        sources = [("b/", "Outer"), ("b/d/", "Inner"), ("b/d/e.md", "Exact")]
        self.assertEqual(md_index.section_for("b/d/f.md", sources), "Inner")
        self.assertEqual(md_index.section_for("b/d/e.md", sources), "Exact")
        self.assertEqual(md_index.section_for("b/g.md", sources), "Outer")

    def test_titles(self):
        meta = {"id": "T0001", "title": 'Fix "quoted" titles: and colons', "status": "open", "owner": "", "needs": [],
                "writes": ["a/"], "depends": [], "land": "direct", "branch": "", "updated": "2026-10-11"}
        text = task.render(meta, "\n## Goal\n\n# Not the title\n")
        self.assertEqual(md_index.title_of(text, "tasks/T0001-fix.md"), 'Fix "quoted" titles: and colons')
        self.assertEqual(md_index.title_of("Intro line\n\n# Heading\n", "a.md"), "Heading")
        self.assertEqual(md_index.title_of("@AGENTS.md\n\nMore text.\n", "CLAUDE.md"), "Imports AGENTS.md")
        self.assertEqual(md_index.title_of("\nJust text\n", "a.md"), "Just text")
        self.assertEqual(md_index.title_of("", "x/empty.md"), "empty.md")

    def test_anchors_follow_github(self):
        self.assertEqual(md_index.anchor("Models: capacity, features, offline gates"),
                         "models-capacity-features-offline-gates")
        self.assertEqual(md_index.anchor("AI_Test heuristic benchmarks and duels"),
                         "ai_test-heuristic-benchmarks-and-duels")
        self.assertEqual(md_index.anchor("bots/p_only/report"), "botsp_onlyreport")

    def test_render_lists_each_file_once(self):
        files = ["README.md", "a.md", "b/c.md", "b/d/e.md", "docs/x.md", "zz/new.md"]
        text = md_index.render(files, {f: f"Title of {f}" for f in files}, DIGEST)
        for f in files:
            self.assertEqual(text.count(f"](../{f})"), 1, f)
        self.assertIn("All 6 tracked Markdown files", text)
        self.assertIn("2 file(s) are not summarised", text)                            # README.md and zz/new.md
        headings = re.findall(r"^## (.+)$", text, re.M)
        self.assertEqual(headings[0], "Repository root")
        self.assertIn("zz", headings)                                                   # an unknown folder still gets a group
        for target in re.findall(r"\]\(#([^)]+)\)", text):
            self.assertIn(target, [md_index.anchor(h) for h in headings])
        self.assertIn("[Second: topic, with punctuation](DIGEST.md#second-topic-with-punctuation)", text)

    def test_every_tracked_markdown_file_is_listed(self):
        if not (REPO / ".git").exists():
            self.skipTest("not a git checkout")
        text, files, _ = md_index.build()
        tracked = md_index.tracked()
        self.assertTrue(set(tracked) <= set(files))
        self.assertIn(md_index.INDEX, files)
        for f in files:
            self.assertEqual(text.count(f"](../{f})"), 1, f)
        digest = (REPO / md_index.DIGEST).read_text(encoding="utf-8")
        sections = {md_index.anchor(h) for h in re.findall(r"^## (.+)$", digest, re.M)}
        for target in set(re.findall(r"\]\(DIGEST\.md#([^)]+)\)", text)):
            self.assertIn(target, sections, "an index row points at a DIGEST.md section that does not exist")


if __name__ == "__main__":
    unittest.main()
