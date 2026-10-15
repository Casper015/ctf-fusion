---
id: T0055
title: "Train Meowmind large V PPO host pilot"
status: review
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

*2026-10-15 09:18Z codex-decision-pilot@desktop-g60ojud:*

Done on desktop-g60ojud (branch codex-decision-pilot/t0055-train-meowmind-large-v-ppo-host-pilot, commit f3efb3b). Both arms completed full 1,000,000 decisions (120 updates, 2,500 episodes each) under exclusive GPU lock. Large V final.pt (f8b98a34...) utility 0.738, v_loss 0.0015; Small V final.pt (328494a9...) utility 0.642, v_loss 0.0018. 0 faults, 0 discarded decisions, max logp gap 2.67e-5. Seven families with newest bots verified under 15% cap. Receipts and models manifest published in reports/; durable backup saved at data/backups/meow_rl_pilot_20261015.tar.gz (312MB).

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
- 2026-10-15 08:20Z codex-decision-pilot@desktop-g60ojud: claimed (forced)
- 2026-10-15 09:03Z claude-review@caspers-macbook: note: For the T0055 run: PR 53 (T0056) will change training/meow_rl's code digest (bridge.cpp now wraps tools/cpp/meow_rl/session.hpp; behavior identical by the bridge parity tests). A run started before PR 53 merges keeps resuming only on that same code: do not pull main into the run's checkout mid-run. Its final checkpoint exports unchanged with bots/meow_rl/export.py. Please run train with --keep-features so export can calibrate on the run's own observations.
- 2026-10-15 09:18Z codex-decision-pilot@desktop-g60ojud: handoff -> review
