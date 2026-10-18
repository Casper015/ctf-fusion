# Agent V2 P3 Opponent Prediction in Search Seed Ledger

This ledger reserves immutable seed bases for task T0081 (`Host compare P3 and P2 in matched agent_v2 search`).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Block | Fixtures | Games | Purpose | Status |
|---|---|---|---|---|---|
| 20262150 | preflight | 64 (16/count x 4) | 256 | Eight-worker host preflight gate across arms P2-dir, P3-dir, P2-gw, P3-gw (counts 2, 5, 8, 15) | Reserved |
| 20262151 | screen | 784 (56/count x 14) | 3,920 | Paired screen: P2-dir, P3-dir, P2-gw, P3-gw + P3-gw~aa across counts 2..15 | Reserved |

## Screen Specification

- **Task:** T0081
- **Variants Configuration:** `bots/agent_v2/p3_search/variants.json`
- **Arms:**
  - `agent_v2_p3exp_p2_direct`: general P2 XXL opponent predictor, direct play
  - `agent_v2_p3exp_p3_direct`: general P3 XXL opponent predictor, direct play
  - `agent_v2_p3exp_p2_gw8x4_fp32`: general P2 XXL opponent predictor, gated 8x4 search ($\tau = 0.002$, FP32 critic)
  - `agent_v2_p3exp_p3_gw8x4_fp32`: general P3 XXL opponent predictor, gated 8x4 search ($\tau = 0.002$, FP32 critic)
  - `agent_v2_p3exp_p3_gw8x4_fp32~aa`: re-run noise control (identical binary to `agent_v2_p3exp_p3_gw8x4_fp32`)
- **Primary Contrasts (Bonferroni Family of 3, $\alpha = 0.05$):**
  1. `agent_v2_p3exp_p3_gw8x4_fp32` - `agent_v2_p3exp_p2_gw8x4_fp32` (P3 search vs P2 search)
  2. `agent_v2_p3exp_p3_direct` - `agent_v2_p3exp_p2_direct` (P3 direct vs P2 direct)
  3. `agent_v2_p3exp_p3_gw8x4_fp32` - `agent_v2_p3exp_p3_direct` (P3 search vs P3 direct)
- **Budget:** 56 maps per count across counts 2–15 = 784 paired fixtures $\times$ 5 arms = 3,920 formal games
- **Environment & Limits:**
  - Exactly 8 workers under `task.py lock cpu --task T0081`
  - Single-tenant host conditions on desktop-g60ojud (no GPU training or competing CPU processes)
  - First reply limit: 2,000 ms; steady reply limit: 50 ms
  - Opponents: held-out reference roster (`profiles/meow_rl_screen.json`: Fable, MiMo, Kimi, GLM, Player4)
