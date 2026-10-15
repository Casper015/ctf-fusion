---
id: T0054
title: "Implement Meowmind on-policy PPO training"
status: review
owner: claude-review@caspers-macbook
needs: []
writes: [".github/workflows/ci.yml", docs/MEOWMIND_RL_IMPLEMENTATION.md, tests/test_meow_rl.py, tests/test_meow_rl_bridge.py, tools/cpp/meow_rl/, training/meow_rl/]
depends: [T0053]
land: pr
branch: claude-review/t0054-implement-meowmind-on-policy-ppo
updated: 2026-10-15
---

## Goal

Implement docs/MEOWMIND_RL_PLAN.md and profiles/meowmind_rl_pilot.json. Build the missing causal native feature/action bridge, live engine adapter, masked joint actor, separate 2M and 28.8K critics, actual on-policy PPO/GAE trainer and resumable rollout store. Preserve Meowmind attribution and original source. No match batches on development machines; no model-strength claim.

## Done when

Feature/actor/resolver parity; causal observations; actual sampled-action log probabilities; legal masks; terminal/truncation GAE; finite updates; checkpoint resume; source/config identity tests pass. Provide a tested host CLI with smoke and full-budget modes, CPU opponent rollout plus CUDA learning, and no search/proposal probability mismatch. PR merged before host training.

## Handoff

*2026-10-15 07:05Z claude-review@caspers-macbook:*

**Done** (PR https://github.com/Casper015/ctf-fusion/pull/52, branch claude-review/t0054-implement-meowmind-on-policy-ppo, head d994756):

Code:
- native bridge tools/cpp/meow_rl/bridge.cpp: the unmodified reference as a C-ABI library;
- training/meow_rl/: actor, critics, GAE, PPO, env, pool, store, train CLI;
- reference16 INT16 actor export (quant.py, actor_int.hpp);
- tests/test_meow_rl{,_bridge}.py and docs/MEOWMIND_RL_IMPLEMENTATION.md;
- CI: the torch job also runs tests.test_meow_rl.

Verified on caspers-macbook, no PyTorch:
- reference actor parity within 2e-5;
- resolver/reply identical to the original bot built without search, on 1,200 recorded turns;
- masks, causality, GAE, storage and recovery;
- bit-exact integer export (97.4% greedy agreement with FP32 for the warm start);
- one live public-engine episode and spawned-worker games (plumbing only);
- core suite 363 OK.

Codex gpt-6.1-sol reviewed three rounds; every finding is fixed.

**Next:**
1. CI green on PR 52 (its torch job is the first PyTorch run), then merge.
2. T0055 on desktop-g60ojud:
   - freeze profiles/meow_rl_training_pool.json (schema ctf-meow-rl-pool-1, at least 7 families at cap 0.15, pinned sha256);
   - reserve seed bases in docs/MEOWMIND_RL_SEEDS.md;
   - run the smoke inside lock gpu: python -m training.meow_rl.train --mode smoke --arm large_v --pool profiles/meow_rl_training_pool.json --seed-base <reserved> --seed-ledger docs/MEOWMIND_RL_SEEDS.md --device cuda --workers <n> --out runs/meow_rl/large_v_s101;
   - read smoke_report.json, then rerun with --mode full --resume.

**Not checked:**
- PyTorch execution (CI only);
- CUDA;
- the Windows DLL build/load (-static -shared via w64devkit);
- the full CLI with the real ProcessCollector;
- host throughput and opponent fault rates under the chosen worker count.

The INT16 export of the trained actor may need QAT in T0056 if its agreement is low.

## Log

- 2026-10-15 05:50Z codex-bigv@caspers-macbook: created
- 2026-10-15 06:04Z claude-review@caspers-macbook: claimed
- 2026-10-15 06:19Z claude-review@caspers-macbook: amended writes -> .github/workflows/ci.yml, docs/MEOWMIND_RL_IMPLEMENTATION.md, tests/test_meow_rl.py, tests/test_meow_rl_bridge.py, tools/cpp/meow_rl/, training/meow_rl/
- 2026-10-15 06:20Z claude-review@caspers-macbook: note: Widened lanes to .github/workflows/ci.yml: the CPU-torch CI job will also run tests.test_meow_rl, because the PPO/trainer code has no torch on the Mac (Casper declined a local torch install).
- 2026-10-15 07:04Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/52 opened (trainer, bridge, reference16 INT16 export, tests, doc). Reviewed by Codex gpt-6.1-sol in 3 rounds, all findings fixed. PyTorch parts first run in CI's torch job; host smoke is T0055.
- 2026-10-15 07:05Z claude-review@caspers-macbook: handoff -> review
