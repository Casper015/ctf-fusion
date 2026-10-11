"""reports/eval/tournament_50k_report.md is exactly what its generator renders from the committed sources.

Every bot and team count of the summary must appear in the report with the summary's own numbers, and every replay
highlight must agree with its recorded scores (task T0028)."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "reporting"))
import tournament_50k_report as gen  # noqa: E402


class TestTournament50kReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(gen.SUMMARY.read_text(encoding="utf-8"))
        cls.highlights = json.loads(gen.HIGHLIGHTS.read_text(encoding="utf-8"))
        cls.report = gen.REPORT.read_text(encoding="utf-8")

    def test_committed_report_is_the_rendered_one(self):
        self.assertEqual(self.report, gen.render(self.summary, self.highlights),
                         "regenerate: python tools/reporting/tournament_50k_report.py")

    def test_every_bot_and_count_matches_the_summary(self):
        per_size = self.summary["per_size_summary"]
        sizes = sorted(per_size, key=int)
        for title, key, fmt in (("Win rate by team count", "win_rate", gen.pct),
                                ("Average score by team count", "avg_score", lambda x: f"{x:.2f}"),
                                ("Games by team count", "games", lambda x: f"{x:,}")):
            section = self.report.split(f"## {title}\n")[1].split("\n## ")[0]
            rows = {m.group(1): [c.strip() for c in m.group(2).split("|")]
                    for m in re.finditer(r"^\| `([^`]+)` \|(.*)\|$", section, re.M)}
            self.assertEqual(set(rows), set(self.summary["ratings_plackett_luce"]), title)
            for bot, cells in rows.items():
                self.assertEqual(cells, [fmt(per_size[s][bot][key]) for s in sizes], f"{title}: {bot}")

    def test_highlights_agree_with_their_scores(self):
        self.assertEqual([h["id"] for h in self.highlights], gen.HIGHLIGHT_IDS)
        for h in self.highlights:
            self.assertRegex(h["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(h["names"]), h["teams"])
            self.assertEqual(len(h["scores"]), h["teams"])
            top = max(h["scores"])
            self.assertEqual(h["winners"], [n for n, s in zip(h["names"], h["scores"]) if s == top])
            self.assertEqual([b["name"] for b in h["bots"]], h["names"])

    def test_late_reply_counts_leave_out_the_first_reply(self):
        bot = {"timeouts_50ms": 7, "first_ms": 223.96, "max_ms": 65.77}
        self.assertEqual(gen.later_late(bot), 6)
        self.assertEqual(gen.later_late({"timeouts_50ms": 0, "first_ms": 11.9, "max_ms": 12.9}), 0)
        self.assertTrue(gen.first_counted(self.highlights))

    def test_differences_and_verdicts(self):
        per_size = {"2": {"a": {"games": 400, "win_rate": 0.6}, "b": {"games": 400, "win_rate": 0.5}}}
        d, se = gen.diff(per_size, "2", "a", "b")
        self.assertAlmostEqual(d, 0.1)
        self.assertAlmostEqual(se, (0.6 * 0.4 / 400 + 0.25 / 400) ** 0.5)
        self.assertEqual(gen.verdict(d, se), "above 0")
        self.assertEqual(gen.verdict(-d, se), "below 0")
        self.assertEqual(gen.verdict(0.01, se), "includes 0")


if __name__ == "__main__":
    unittest.main()
