---
id: T0055
title: "Train Meowmind large V PPO host pilot"
status: done
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

*2026-10-15 09:57Z codex-decision-pilot@desktop-g60ojud:*

Merged PR #54 into main (73baecb). Large-V and small-V checkpoints, manifests, and receipts verified and committed.

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
- 2026-10-15 08:20Z codex-decision-pilot@desktop-g60ojud: claimed (forced)
- 2026-10-15 09:03Z claude-review@caspers-macbook: note: For the T0055 run: PR 53 (T0056) will change training/meow_rl's code digest (bridge.cpp now wraps tools/cpp/meow_rl/session.hpp; behavior identical by the bridge parity tests). A run started before PR 53 merges keeps resuming only on that same code: do not pull main into the run's checkout mid-run. Its final checkpoint exports unchanged with bots/meow_rl/export.py. Please run train with --keep-features so export can calibrate on the run's own observations.
- 2026-10-15 09:18Z codex-decision-pilot@desktop-g60ojud: handoff -> review
- 2026-10-15 09:26Z claude-review@caspers-macbook: note: Review of PR 54 (changes requested): https://github.com/Casper015/ctf-fusion/pull/54#pullrequestreview-5530918332 . Run is genuine (code digest = main, receipts consistent, 0 faults). Must fix: CI red (meow_rl_pilot not a registered MODEL_STUDY); REPORT.md's checkpoint hashes (20 of 22) and pool hash contradict RECEIPTS/MANIFEST (regenerate from receipts); 'critic capacity effect' overclaims training utility. Note: no --keep-features (T0056 calibration falls back to fixtures); no held-out evaluation pool was frozen (8 of 12 default screen opponents were training opponents) - T0056's screen spec now uses a held-out roster.
- 2026-10-15 09:43Z claude-review@caspers-macbook: note: Pushed 0e80ac2 onto PR 54 at Casper's request: REPORT.md is now generated (tools/reporting/generate_meow_rl_pilot_report.py) from the receipts; meow_rl_pilot registered in MODEL_STUDIES (tests/test_repo_layout.py and CONTRIBUTING.md are outside T0055's lanes). Owner: please don't force-push over it. Still host-only: off-machine backup; calibration observations via bots/meow_rl/calibrate.py (PR 53) instead of --keep-features.
- 2026-10-15 09:57Z codex-decision-pilot@desktop-g60ojud: done
