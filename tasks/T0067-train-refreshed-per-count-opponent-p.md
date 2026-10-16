---
id: T0067
title: "Train refreshed per-count opponent P specialists"
status: open
owner: ""
needs: [host, cuda, ai-test]
writes: [reports/models/count_specialists_p/, reports/host/count_specialists_p/, inference/count_specialists_p/, profiles/count_specialists_p.json]
depends: [T0066]
land: pr
branch: ""
updated: 2026-10-16
---

## Goal

Train one opponent P specialist for each exact count 2-15 on its own nXX training corpus from the upstream data task. Freeze one existing P family and capacity (initial reference P2 XXL) and its per-count warm-start artifact before losses/test outcomes. Reuse existing joint movement, conditional action and target heads, native feature schema and export pipeline. Prefer existing matching nXX weights; if absent, use a documented compatible general initialization. No other count's new examples enter that specialist's supervised fit. Previous cross-count pretraining is allowed and explicitly recorded.

Capacity stays identical across counts in this first pilot. Freeze seeds 101/102/103 as optimizer seeds, never as game-seed substitutes. Use a fixed maximum of eight epochs per seed with a common validation-checkpoint rule; choose the seed-101 validation-selected checkpoint as the primary deployment artifact, with seeds 102/103 providing replication diagnostics, not a best-seed contest. Record any existing trainer scheduler/batch adaptations before training. Fit calibration only on permitted train/validation data, never test; use aligned frozen calibration data for paired old/new artifacts within each count. Compare against the current deployed nXX P baseline, not an artificially weak shared baseline. Retain the same family/pool and evaluate movement NLL, joint/top-k, probability calibration, conditional actions/targets, flag/combat strata and quantized/native parity. Use map-cluster intervals and prespecified multiplicity across countwise claims; one held-out test read after selection. Training mode and model routing must remain identities, not path aliases.

Run CUDA training under GPU lease. All timing or deployment collection requires the CPU lease, exactly eight game workers, 2s initialization/50ms full later replies, and GPU training stopped. Report total model initialization cost and inference calls for 15-team games. If native export or time limits fail, retain failures and do not silently fall back to the original Meowmind predictor. Offline prediction gains are not strength admission. Prepare new P bank for T0065 carrier and matched deployment comparisons, with production defaults unchanged.

## Done when

Fourteen pinned, count-specific P specialists and three-seed training diagnostics delivered, or explicit shortages/failures. Same-capacity baselines, fresh per-count datasets, fixed selection/test procedure, native export parity, quantization diagnostics and host feasibility receipts verified. New registry selects exact global team count from INIT and never runs all 14 specialists per decision. Any shared-code changes require lane amendment or a separate task. Publish a PR; no guaranteed improvement or automatic promotion.

## Handoff

*2026-10-16 00:12Z codex-count-plan@caspers-macbook:*

User-authorized future host work. Read Goal and Done when in full, inspect current upstream tasks, then claim with the required capabilities. These tasks are newly registered; no collection/training was launched by the planning agent. Keep ongoing T0064 host work and T0065 carrier implementation isolated. Specialized data and policies use exact counts 2-15, not band labels.

## Log

- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: created
- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: handoff prepared
- 2026-10-16 08:12Z claude-review@caspers-macbook: note: Casper's decision (2026-10-16): Player2 is held out. Train only on T0066 corpora with the 207 Player2_v005 matches excluded (see T0066 log and PR 61); splits are then short of 140/30/30 per count.
- 2026-10-16 08:15Z claude-review@caspers-macbook: note: CORRECTION: ignore my note above. Train on all 2,800 T0066 matches; Player2_v005 is Player5 lineage and a training bot (docs/MEOWMIND_RL_POOL_SPLIT.md), not the held-out Player2. https://github.com/Casper015/ctf-fusion/pull/61#issuecomment-6211153284
