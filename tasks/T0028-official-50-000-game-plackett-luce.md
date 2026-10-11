---
id: T0028
title: "50,000-game tournament report: Plackett-Luce ratings reported by the run"
status: open
owner: ""
needs: [host]
writes: [docs/DIGEST.md, docs/MARKDOWN_INDEX.md, reports/eval/, tests/test_tournament_50k_report.py, tools/reporting/tournament_50k_report.py]
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

*2026-10-11 06:30Z claude-ci@caspers-macbook:*

PR #20 (branch claude-2/t0028-official-50-000-game-plackett-luce, head 9c3b798) now fixes the review findings that the committed data can settle.

Done (claude-ci@caspers-macbook, at Casper's request):
- `tools/reporting/tournament_50k_report.py` regenerates reports/eval/tournament_50k_report.md from tournament_50k_summary.json and tournament_50k_highlights.json. Regenerate with `python tools/reporting/tournament_50k_report.py`; `--check` reports staleness.
  - tests/test_tournament_50k_report.py checks every bot × count against the summary.
  - This fixed the ten wrong v12_ponly_pxl cells.
- The three highlights are read from the published replay JSONs and bound by SHA-256. Rebuild them with `--replays DIR`.
- The ratings are labelled as the run's own Plackett-Luce estimates: not the official BT scorer, not reproducible, no intervals.
- Win-rate comparisons carry approximate binomial intervals. The XL/XXL mechanisms are stated as hypotheses.
- Time limits: the replays show late replies recorded but never dropped (`dropped_official` 0). Their `timeouts_50ms` field also counts the first reply.
- DIGEST paragraph and seed row corrected; markdown index regenerated.

Still open, host only (review items 3 and 4, and part of 5). These must come from the run directory on desktop-g60ojud:
1. Commit the ratings fitter or its pinned source, with the exact command: likelihood, tie handling, scale, anchor, count weights.
2. Commit the ordered per-game results (compact JSONL plus a digest). The generator can then refit the ratings with intervals clustered by game.
3. Commit a run manifest: engine/runner/binary/model hashes, seeds, schedule digest, independent and duplicate game counts.
4. Commit a per-game fault/dead/late/timing audit with field coverage. Keep faulted games in the denominators.
5. Map the replay highlights to tournament game ids.

Extend the generator to read these files, rerun it, and push to the same branch. No new 50,000-game run is requested.

## Log

- 2026-10-11 06:09Z claude-2@desktop-g60ojud: created
- 2026-10-11 06:09Z claude-2@desktop-g60ojud: claimed
- 2026-10-11 06:10Z claude-2@desktop-g60ojud: amended writes -> docs/DIGEST.md, docs/MARKDOWN_INDEX.md, reports/eval/
- 2026-10-11 06:15Z claude-2@desktop-g60ojud: note: PR submitted: https://github.com/Casper015/ctf-fusion/pull/20
- 2026-10-11 06:15Z claude-2@desktop-g60ojud: handoff -> review
- 2026-10-11 06:22Z codex-review@caspers-macbook: review: Casper requested PR #20 review and T0028 update; recorded six required correction/acceptance checks. Review: https://github.com/Casper015/ctf-fusion/pull/20#pullrequestreview-5482159308
- 2026-10-11 06:26Z claude-ci@caspers-macbook: note: Casper asked claude-ci@caspers-macbook (in chat) to fix the PR #20 review findings; taking the task over with claim --force. The fixes that need the host (PL fitter, run manifest, raw-log timing audit) will be handed back.
- 2026-10-11 06:26Z claude-ci@caspers-macbook: claimed (forced)
- 2026-10-11 06:26Z claude-ci@caspers-macbook: amended writes -> docs/DIGEST.md, docs/MARKDOWN_INDEX.md, reports/eval/, tests/test_tournament_50k_report.py, tools/reporting/tournament_50k_report.py; title -> 50,000-game tournament report: Plackett-Luce ratings reported by the run
- 2026-10-11 06:30Z claude-ci@caspers-macbook: handoff -> open
