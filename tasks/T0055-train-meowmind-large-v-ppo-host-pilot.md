---
id: T0055
title: "Train Meowmind large V PPO host pilot"
status: active
owner: codex-decision-pilot@desktop-g60ojud
needs: [host, cuda, ai-test]
writes: [reports/host/meow_rl_pilot/, reports/models/meow_rl_pilot/, profiles/meow_rl_training_pool.json, docs/MEOWMIND_RL_SEEDS.md]
depends: [T0054]
land: pr
branch: codex-decision-pilot/t0055-train-meowmind-large-v-ppo-host-pilot
updated: 2026-10-15
---

## Goal

Execute the user-authorized bounded training pilot in docs/MEOWMIND_RL_PLAN.md on desktop-g60ojud, RTX 4080 Laptop 12GB. Freeze pool identities and fresh seeds against the global ledger; run 20K correctness smoke then 1M total decisions for large V, followed by 1M for the small-V control. Optimizer seed 101, uniform counts 2-15, no automatic sweeps. Do not use old branch labels as PPO data.

## Done when

Acquire GPU and CPU leases for live rollouts/training; publish actual launch command, PID and immutable config hashes. Verify smoke, resume, opponent lifecycle, all-seat faults, no data leakage and finite gradients. Preserve complete episode/checkpoint receipts outside git data/build/runs, with durable backup. Report realized pool/count coverage, resource cost and learning diagnostics. Freeze final checkpoints; training loss is not admission. Handoff export identities and replication requirements.

## Handoff

Not started.

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
- 2026-10-15 08:20Z codex-decision-pilot@desktop-g60ojud: claimed (forced)
