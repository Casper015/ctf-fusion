---
id: T0059
title: "Train Meowmind PPO on training pool v2 (AI_Test lineages)"
status: done
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

*2026-10-15 22:37Z codex-decision-pilot@desktop-g60ojud:*

PR #56 merged into main as b42d965ec2. Verified pins, 1M decisions each with --keep-features, durable backups verified, ready for T0062/T0063 aligned exports and screen.

## Log

- 2026-10-15 09:51Z claude-review@caspers-macbook: created
- 2026-10-15 10:08Z codex-decision-pilot@desktop-g60ojud: claimed
- 2026-10-15 12:06Z codex-decision-pilot@desktop-g60ojud: handoff -> review
- 2026-10-15 17:31Z claude-review@caspers-macbook: note: Correction by claude-review: this task's Goal field holds only a temp path (task.py new does not expand @file). The intended goal, verbatim: Train the Meowmind PPO pilot again against training pool v2 (profiles/meow_rl_training_pool_v2.json, T0058: T0055's pool plus the AI_Test training lineages), on desktop-g60ojud, as Casper asked on 2026-10-15. Same recipe as T0055 (docs/MEOWMIND_RL_PLAN.md, profiles/meowmind_rl_pilot.json): optimizer seed 101, both arms (large_v primary, small_v ablation), 20,000-decision smoke then 1,000,000 learner decisions per arm, counts 2-15, no automatic sweeps.  1. Verify every pool v2 pin against the host's AI_Test checkout (python -m unittest tests.test_meow_rl_pool_split with AI_TEST_ROOT). If a file differs, stop and re-pin in this task's lane with both hashes recorded in the log; never train on an unpinned file. 2. Reserve two fresh seed bases in docs/MEOWMIND_RL_SEEDS.md before any game. Never reuse 20261901-20261906 or any seed in docs/DIGEST.md. 3. Register reports/models/meow_rl_pool2 in MODEL_STUDIES (tests/test_repo_layout.py, CONTRIBUTING.md) in the same PR. 4. Run inside the GPU lock, with --keep-features (T0056's export calibrates on these batches):    python tools/repo/task.py lock gpu --task T0059 -- python -m training.meow_rl.train --mode smoke --arm large_v --pool profiles/meow_rl_training_pool_v2.json --seed-base <reserved> --seed-ledger docs/MEOWMIND_RL_SEEDS.md --device cuda --workers 6 --keep-features --out runs/meow_rl/pool2_large_v_s101    Then the same with --mode full --resume; then both again for small_v with its own seed and --out runs/meow_rl/pool2_small_v_s101. 5. Predeclare final.pt of each arm as its candidate (named large_v_pool2_s101 and small_v_pool2_s101). They are separate arms from T0055's. Whether pool2 large_v replaces T0055's large_v as the primary screen candidate (profiles/meow_rl_screen.json) must be decided and recorded before any screen game, never after one. 6. Generate every report from receipts with a script named in its header (as tools/reporting/generate_meow_rl_pilot_report.py does for T0055); no hand-typed hashes; training utilities are descriptive only. Back up the runs off the host's disk and record the receipt.
- 2026-10-15 18:04Z claude-review@caspers-macbook: note: Review of PR 56 (changes requested): https://github.com/Casper015/ctf-fusion/pull/56#pullrequestreview-5537730041 . Must fix: --keep-features asserted not shown; backup is a D:/ path, not off-machine; the primary-candidate decision (pool-v2 vs T0055 large_v) is not recorded and pool-v2 is unscreened.
- 2026-10-15 18:52Z claude-review@caspers-macbook: note: claude-review resolved PR 56's merge conflict at Casper's request: merge commit ceeb1bb (main into the branch, fast-forward push, no force). Only conflict: docs/MEOWMIND_RL_SEEDS.md; kept T0056's 20261907/08 rows and T0059's 20261909/10 rows in seed order. The receipts' recorded ledger sha is the training-time file and is left unchanged. Core suite passes. Pull before further commits. The review's must-fix items still stand.
- 2026-10-15 22:20Z claude-review@caspers-macbook: note: claude-review: PR 56 is merged (b42d965). T0063 (aligned experiment) needs this task's lane docs/MEOWMIND_RL_SEEDS.md; please mark T0059 done when its review items are settled so the lane frees up.
- 2026-10-15 22:37Z codex-decision-pilot@desktop-g60ojud: done
