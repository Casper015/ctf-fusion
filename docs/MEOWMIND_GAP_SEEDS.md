# Meowmind Gap Attribution Screen Seed Ledger

This ledger reserves immutable seed bases for task T0078 (`Host: original Meowmind search-on/off gap attribution`).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Block | Fixtures | Games | Purpose | Status |
|---|---|---|---|---|---|
| 20262110 | preflight | 64 (16/count x 4) | 192 | Eight-worker host preflight gate across arms ON, OFF, V (counts 2, 5, 8, 15) | Reserved |
| 20262111 | screen | 784 (56/count x 14) | 3,136 | Paired screen: ON (`meowmind_search_on`), OFF (`meowmind_search_off`), V (`agent_v2_p2xxl_direct`) + ON~aa | Reserved |

## Screen Specification

- **Task:** T0078
- **Arms:**
  - $ON$: `meowmind_search_on` (original Meowmind with NN_SEARCH=1, from T0075)
  - $OFF$: `meowmind_search_off` (identical source with NN_SEARCH=0, from T0075)
  - $V$: `agent_v2_p2xxl_direct` (deployed actor-direct default)
  - $ON\sim\text{aa}$: `meowmind_search_on~aa` (re-run A/A noise control, identical binary)
- **Primary Contrasts:** $ON - OFF$ (causal effect of Meowmind's search) and $ON - V$ (current gap between Meowmind and agent_v2) share one 2-contrast Bonferroni family ($\alpha = 0.05, z \approx 2.2414$).
- **Descriptive Contrasts:** $OFF - V$, per-count rows (2..15), 3–15 equal-weight band, and 2-team band.
- **Budget:** 56 fixtures per count across counts 2..15 = 784 paired fixtures $\times$ 4 arms = 3,136 formal games
- **Environment & Limits:**
  - Exactly 8 workers under `task.py lock cpu --task T0078`
  - Single-tenant host conditions (no GPU training or competing CPU processes)
  - First reply limit: 2,000 ms; steady reply limit: 50 ms
  - Opponents: held-out reference roster (`profiles/meow_rl_screen.json`: Fable, MiMo, Kimi, GLM, Player4)
