---
id: T0028
title: "Official 50,000-game Plackett-Luce tournament evaluation report"
status: review
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

*2026-10-11 06:22Z codex-review@caspers-macbook:*

PR #20 review: corrections required before merge or marking T0028 done. Casper explicitly requested this review and this task-board update. The existing owner, status and write lanes remain unchanged.

Review: https://github.com/Casper015/ctf-fusion/pull/20#pullrequestreview-5482159308
Reviewed head: 5d8c283475e0b5564a1a4f605be490bd81792006.

Required acceptance checks:
1. Regenerate every report table from the machine-readable source using a named, committed generator or an immutable referenced implementation. Correct all ten v12_ponly_pxl breakdown cells; 2T must reflect JSON win_rate 0.7566 and avg_score 30.78. Add a report/source consistency check covering every bot and count; do not edit results to match prose.
2. Reconcile all three curated highlights with their actual replay JSON. The 4T replay records Player5/baseline/C08 XL/v12 control scoring 14/12/9/10; the duel opponent is Player5; the 8T replay has C08 XXL tied at 10. Generate participants, scores and winner labels from hash-bound replay artifacts and link their tournament game IDs.
3. Identify the exact PL scorer/command, likelihood, tie handling, rating scale, anchor, count weights and result/schedule digests. Keep PL and official pairwise BT scores distinct. If official BT is claimed, demonstrate conformance to the official scorer and weights or publish its separately fitted result. Rename unsupported official-score wording consistently in the report, task title and DIGEST.
4. Bind the run to engine, runner, model/header, candidate and opponent binary hashes, seeds, independent map/game counts, duplicate/stale checks, actual limits, and retained-row coverage. Audit faults, late/dead/timeout cases, first-reply and subsequent p50/p99/max timings with missing-field coverage; retain faulted games in denominators. Explain the 16-worker load and 2 s / 50 ms enforcement. The linked 4T replay records C08 XL max_ms 65.77 / p99_ms 53.78; distinguish a loose replay from official-limit evidence rather than assuming compliance.
5. Supply uncertainty for carrier rating deltas and XL-versus-XXL differences clustered by independent map/game groups under declared count weights, or clearly label the comparisons as descriptive point estimates. Remove statistically certain ranking/gain claims without supporting intervals; global XL/XXL gaps of 1.2 and 0.3 PL points do not establish superiority by themselves.
6. Treat higher-capacity tactical accuracy and extra rollout throughput as hypotheses until measured. Correct the dense-board generalization: the displayed C08 XXL win rates are higher at 8T and 10T. Explain that P predicts opponent behavior inside the heuristic/search carrier. Regenerate DIGEST and the markdown index after correcting the report, and rerun the relevant consistency/layout/board checks.

Use the existing results and logs first. No new 50,000-game batch, model training, production setting changes or data rewriting is authorized by this review. Missing source artifacts should be reported explicitly, not replaced by inferred statistics.

Retained owner handoff:
*2026-10-11 06:15Z claude-2@desktop-g60ojud:*

PR #20 opened with 50,000-game Plackett-Luce tournament evaluation report, summary.json, and documentation index update: https://github.com/Casper015/ctf-fusion/pull/20

## Log

- 2026-10-11 06:09Z claude-2@desktop-g60ojud: created
- 2026-10-11 06:09Z claude-2@desktop-g60ojud: claimed
- 2026-10-11 06:10Z claude-2@desktop-g60ojud: amended writes -> docs/DIGEST.md, docs/MARKDOWN_INDEX.md, reports/eval/
- 2026-10-11 06:15Z claude-2@desktop-g60ojud: note: PR submitted: https://github.com/Casper015/ctf-fusion/pull/20
- 2026-10-11 06:15Z claude-2@desktop-g60ojud: handoff -> review
- 2026-10-11 06:22Z codex-review@caspers-macbook: review: Casper requested PR #20 review and T0028 update; recorded six required correction/acceptance checks. Review: https://github.com/Casper015/ctf-fusion/pull/20#pullrequestreview-5482159308
- 2026-10-11 06:26Z claude-ci@caspers-macbook: note: Casper asked claude-ci@caspers-macbook (in chat) to fix the PR #20 review findings; taking the task over with claim --force. The fixes that need the host (PL fitter, run manifest, raw-log timing audit) will be handed back.
