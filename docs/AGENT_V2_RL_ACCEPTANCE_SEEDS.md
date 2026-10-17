# Agent V2 RL Acceptance Screen Seed Ledger

This ledger reserves immutable seed bases for task T0077 (`Host: exact-count RL specialists versus general control and common initialization`).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Block | Fixtures | Games | Purpose | Status |
|---|---|---|---|---|---|
| 20262100 | preflight | 64 (16/count x 4) | 192 | Eight-worker host preflight gate across arms S, G, I (counts 2, 5, 8, 15) | Reserved |
| 20262101 | screen | 1,344 (96/count x 14) | 5,376 | Paired acceptance screen: S (`agent_v2_rl_csp`), G (`agent_v2_rl_general_csp`), I (`agent_v2_rl_init_csp`) + S~aa | Reserved |

## Screen Specification

- **Task:** T0077
- **Arms:**
  - $S$: `agent_v2_rl_csp` (14 exact-count PPO specialists from T0068)
  - $G$: `agent_v2_rl_general_csp` (compute-matched general control from T0068, 222–279 episodes/count)
  - $I$: `agent_v2_rl_init_csp` (common initialization baseline from T0075)
  - $S\sim\text{aa}$: `agent_v2_rl_csp~aa` (re-run noise control, identical binary)
- **Primary Contrast:** $S - G$ (tests whether per-count specialization helps)
- **Supporting Contrast:** $G - I$ (tests whether continuing PPO on `agent_v2` carrier helps)
- **Family Multiplicity:** 2-contrast Bonferroni family ($\alpha = 0.05$, $z \approx 2.2414$)
- **Descriptive Contrasts:** $S - I$, per-count rows (2..15), 3–15 equal-weight band, and 2-team band
- **Budget:** 96 fixtures per count across counts 2–15 = 1,344 paired fixtures $\times$ 4 arms = 5,376 formal games
- **Environment & Limits:**
  - Exactly 8 workers under `task.py lock cpu --task T0077`
  - Single-tenant host conditions (no GPU training or competing CPU processes)
  - First reply limit: 2,000 ms; steady reply limit: 50 ms
  - Opponents: held-out reference roster (`profiles/meow_rl_screen.json`: Fable, MiMo, Kimi, GLM, Player4)

## Schedule Derivation Contract

Per fixture $i$ under seed base $S$:
- Team count $N \in \{2..15\}$
- Seat uniform over $0..N-1$
- Map seed: derived deterministically via SHA-256 derivations
- Opponents: drawn from held-out reference roster (`profiles/meow_rl_screen.json`)
