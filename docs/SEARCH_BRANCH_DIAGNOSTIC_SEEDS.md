# Search Branch Diagnostics Seed Ledger (T0086)

This ledger reserves immutable seed bases for task T0086 (`Host collect bounded branch diagnostics and select the next intervention`).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Purpose | Schedule / Scope | Status |
|---|---|---|---|
| 20262160 | Source games schedule across team counts 2..15 (8 maps/count, 112 matches total) | `profiles/search_branch_diagnostics.json` source screen | Reserved |
| 20262161 | Bounded operational smoke test scenario seed (<= 16 branches) | Host feasibility verification and cost estimation | Reserved |
| 20262162 | Bounded branch collection scenario seed (primary 10,752 episodes + extended 1,792 episodes) | Faithful-replay-v1 branch rollouts on 8 workers | Reserved |
| 20262164 | Simulator parity verification seed | Controlled step parity vs official engine (counts 2, 5, 8, 15) | Reserved |

## Screen and Diagnostic Specification

- **Task:** T0086
- **Configuration:** `profiles/search_branch_diagnostics.json`
- **Candidate Arm:** `agent_v2_p3exp_p2_direct` (general P2 package, direct play)
- **Source Opponent Pool:** Approved training/development pool with lineage provenance (`profiles/meow_rl_training_pool_v2.json`: Sol, Astra, Opus, Sonnet, DeepSeek, Qwen, Player5, Baseline)
- **Source Games:** 8 maps per count across counts 2..15 = 112 formal source matches
- **Source Map Splits per Count:** 4 train/dev (maps 0..3), 2 validation (maps 4..5), 2 holdout (maps 6..7)
- **Causal Roots:** 3 roots per map selected outcome-blind (`unconditioned:1,contact:1,override:1`), margin 5, tau 0.002 = 336 roots total
- **Candidate Sets:**
  - Primary: 8 nested candidates (including baseline at index 0) x 4 independent repeats = 10,752 branch episodes
  - Extended: 24 nested candidates on predeclared subset (2 maps per count, 1 unconditioned root per map = 28 roots x 16 extra candidates x 4 repeats = 1,792 branch episodes)
  - Cap: Maximum 12,544 branch episodes + 112 source games
- **Environment & Limits:**
  - Exactly 8 workers under `task.py lock cpu --task T0086` on desktop-g60ojud
  - First reply limit: 2,000 ms; steady reply limit: 50 ms
  - Replay protocol: `faithful-replay-v1`
