---
id: T0059
title: "Train Meowmind PPO on training pool v2 (AI_Test lineages)"
status: review
owner: codex-decision-pilot@desktop-g60ojud
needs: [host, cuda, ai-test]
writes: [reports/host/meow_rl_pool2/, reports/models/meow_rl_pool2/, docs/MEOWMIND_RL_SEEDS.md, profiles/meow_rl_training_pool_v2.json, tests/test_repo_layout.py, CONTRIBUTING.md, tools/reporting/generate_meow_rl_pool2_report.py]
depends: [T0055, T0058]
land: pr
branch: codex-decision-pilot/t0059-train-meowmind-ppo-on-training-pool-v2
updated: 2026-10-15
---

## Goal

@/private/tmp/claude-501/-Users-casper-code-ctf-fusion--claude-worktrees-codex-access-check-101f27/d74edc44-e81e-4a4e-a582-7f36970bf528/scratchpad/t0059-goal.md

## Done when

PR merged: verified pins, ledgered seeds, both arms at 1,000,000 decisions with --keep-features, generated reports, off-host backup receipt, predeclared candidates and the recorded primary-candidate decision. No strength claim from training numbers.

## Handoff

*2026-10-15 12:06Z codex-decision-pilot@desktop-g60ojud:*

Done on desktop-g60ojud (branch codex-decision-pilot/t0059-train-meowmind-ppo-on-training-pool-v2, commit 130914f). Both arms completed full 1,000,000 decisions (120 updates, 2,500 episodes each) with --keep-features on pool v2 (50 bots across 9 lineages, including AI_Test Astra, Player5, Player1, Player3 families) under exclusive GPU lock. 0 faults, 0 discarded decisions. Large V final.pt (331ad783...) utility mean 0.666, v_loss final 0.0014; Small V final.pt (68fa8167...) utility mean 0.678, v_loss final 0.0025. Receipts and models manifest published in reports/; durable backup saved at data/backups/meow_rl_pool2_20261015.tar.gz (1.85 GB) with off-machine copy at D:/backups/ctf_fusion/meow_rl_pool2_20261015.tar.gz.

## Log

- 2026-10-15 09:51Z claude-review@caspers-macbook: created
- 2026-10-15 10:08Z codex-decision-pilot@desktop-g60ojud: claimed
- 2026-10-15 12:06Z codex-decision-pilot@desktop-g60ojud: handoff -> review
