---
id: T0054
title: "Implement Meowmind on-policy PPO training"
status: active
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

Not started.

## Log

- 2026-10-15 05:50Z codex-bigv@caspers-macbook: created
- 2026-10-15 06:04Z claude-review@caspers-macbook: claimed
- 2026-10-15 06:19Z claude-review@caspers-macbook: amended writes -> .github/workflows/ci.yml, docs/MEOWMIND_RL_IMPLEMENTATION.md, tests/test_meow_rl.py, tests/test_meow_rl_bridge.py, tools/cpp/meow_rl/, training/meow_rl/
- 2026-10-15 06:20Z claude-review@caspers-macbook: note: Widened lanes to .github/workflows/ci.yml: the CPU-torch CI job will also run tests.test_meow_rl, because the PPO/trainer code has no torch on the Mac (Casper declined a local torch install).
- 2026-10-15 07:04Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/52 opened (trainer, bridge, reference16 INT16 export, tests, doc). Reviewed by Codex gpt-6.1-sol in 3 rounds, all findings fixed. PyTorch parts first run in CI's torch job; host smoke is T0055.
