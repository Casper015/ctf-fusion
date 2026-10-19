# Strongest Bot Ranking Screen Seed Ledger

This ledger reserves immutable seed bases for task T0094 (`Host: rank the strongest bots (ctf_duel_challenger, Meowmind, Meowmind+P3, agent_v2) in one paired screen`).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Block | Fixtures | Games | Purpose | Status |
|---|---|---|---|---|---|
| 20262170 | preflight | 64 (16/count x 4) | 320 | Eight-worker host preflight gate across 5 contenders (counts 2, 5, 8, 15) | Reserved |
| 20262171 | screen | 784 (56/count x 14) | 4,704 | Paired strongest-bot screen across 6 arms (5 contenders + ctf_duel_challenger~aa) | Reserved |

## Screen Specification

- **Task:** T0094
- **Arms:**
  - `ctf_duel_challenger` (external challenger uploaded 2026-10-19, MIT license)
  - `meowmind_search_on` (original Meowmind with NN_SEARCH=1)
  - `meowmind_p3_xxl` (Meowmind with our P3 XXL in opponent mixture)
  - `meowmind_p3_distill_xxl` (Meowmind with our distilled P3 XXL in opponent mixture)
  - `agent_v2_p2xxl_direct` (current production default, P2 XXL, actor-direct)
  - `ctf_duel_challenger~aa` (re-run A/A noise control, identical binary)
- **Primary Comparisons:** One 4-contrast Bonferroni family ($\alpha = 0.05$):
  1. `ctf_duel_challenger - meowmind_search_on`
  2. `ctf_duel_challenger - agent_v2_p2xxl_direct`
  3. `meowmind_p3_xxl - meowmind_search_on`
  4. `meowmind_p3_distill_xxl - meowmind_search_on`
- **A/A Contrast:** `ctf_duel_challenger~aa - ctf_duel_challenger` (verifies baseline execution noise)
- **Budget:** 56 fixtures per count across counts 2..15 = 784 paired fixtures $\times$ 6 arms = 4,704 formal games
- **Environment & Limits:**
  - Exactly 8 workers under `task.py lock cpu --task T0094`
  - Single-tenant host conditions (no GPU training or competing CPU processes)
  - First reply limit: 2,000 ms; steady reply limit: 50 ms
  - Opponents: held-out reference roster (`profiles/meow_rl_screen.json`: Fable, MiMo, Kimi, GLM, Player4)
