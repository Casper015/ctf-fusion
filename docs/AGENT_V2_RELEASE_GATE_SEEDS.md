# Agent V2 Release Gate Seed Ledger

This ledger reserves immutable seed bases for task T0083 (`Host: release gate, general PPO policy G versus current default D`).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Block | Fixtures | Games | Purpose | Status |
|---|---|---|---|---|---|
| 20262140 | preflight | 64 (16/count x 4) | 192 | Eight-worker host preflight gate across arms G, D, G~aa (counts 2, 5, 8, 15) | Reserved |
| 20262141 | screen | 1,344 (96/count x 14) | 4,032 | Paired release gate screen: G (`agent_v2_rl_general_csp`), D (`agent_v2_p2xxl_direct`) + G~aa | Reserved |

## Screen Specification

- **Task:** T0083
- **Arms:**
  - $G$: `agent_v2_rl_general_csp` (general PPO policy, CSP P bank, actor-direct)
  - $D$: `agent_v2_p2xxl_direct` (current default, P2 XXL, actor-direct)
  - $G\sim\text{aa}$: `agent_v2_rl_general_csp~aa` (re-run A/A noise control, identical binary)
- **Primary Contrast:** $G - D$ (tests whether general PPO policy improves upon current default D)
- **Family Multiplicity:** Single comparison (multiplicity = 1, standard 95% interval)
- **Promotion Rule:** G becomes default only if $G - D$ "helps" (95% CI above 0); "not shown" or "hurts" keeps D.
- **Budget:** 96 fixtures per count across counts 2..15 = 1,344 paired fixtures $\times$ 3 arms = 4,032 formal games
- **Environment & Limits:**
  - Exactly 8 workers under `task.py lock cpu --task T0083`
  - Single-tenant host conditions (no GPU training or competing CPU processes)
  - First reply limit: 2,000 ms; steady reply limit: 50 ms
  - Opponents: held-out reference roster (`profiles/meow_rl_screen.json`: Fable, MiMo, Kimi, GLM, Player4)
