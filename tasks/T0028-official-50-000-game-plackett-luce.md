---
id: T0028
title: "Official 50,000-game Plackett-Luce tournament evaluation report"
status: active
owner: claude-2@desktop-g60ojud
needs: [host]
writes: [docs/DIGEST.md, docs/MARKDOWN_INDEX.md, reports/eval/]
depends: []
land: pr
branch: claude-2/t0028-official-50-000-game-plackett-luce
updated: 2026-10-11
---

## Goal

Incorporate official 50,000-game (10 match sizes, 5000 games each, 16 workers, 21 bots) tournament summary and Plackett-Luce (Bradley-Terry) evaluation report into reports/eval/, comparing new pure policy models (XL/XXL), carriers, controls, and official benchmarks, and link curated match highlights.

## Done when

reports/eval/tournament_50k_summary.json and reports/eval/tournament_50k_report.md committed; docs/MARKDOWN_INDEX.md updated; core tests pass; PR submitted to repository.

## Handoff

Not started.

## Log

- 2026-10-11 06:09Z claude-2@desktop-g60ojud: created
- 2026-10-11 06:09Z claude-2@desktop-g60ojud: claimed
- 2026-10-11 06:10Z claude-2@desktop-g60ojud: amended writes -> docs/DIGEST.md, docs/MARKDOWN_INDEX.md, reports/eval/
- 2026-10-11 06:15Z claude-2@desktop-g60ojud: note: PR submitted: https://github.com/Casper015/ctf-fusion/pull/20
