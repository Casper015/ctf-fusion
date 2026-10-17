# Agent V2 Search Gate Screen Seed Ledger

This ledger reserves immutable seed bases for task T0079 (`Host: gated wider agent_v2 search versus direct and ungated wider search`, [AGENT_V2_SEARCH_GATE.md](AGENT_V2_SEARCH_GATE.md)).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Block | Fixtures | Games | Purpose | Status |
|---|---|---|---|---|---|
| 20262130 | preflight | 64 (16/count x 4) | 192 | Eight-worker host preflight gate across arms D, GW, UW (counts 2, 5, 8, 15) | Reserved |
| 20262131 | screen | 1,344 (96/count x 14) | 5,376 | Paired search gate screen: D (`agent_v2_p2xxl_direct`), GW (`agent_v2_p2xxl_gw8x4_fp32`), UW (`agent_v2_p2xxl_uw8x4_fp32`) + GW~aa | Reserved |

## Screen Specification

- **Task:** T0079
- **Arms:**
  - $D$: `agent_v2_p2xxl_direct` (actor-direct baseline, no search)
  - $GW$: `agent_v2_p2xxl_gw8x4_fp32` (gated-wide 8 candidates $\times$ 4 scenarios, $\tau = 0.002$, FP32 critic)
  - $UW$: `agent_v2_p2xxl_uw8x4_fp32` (ungated-wide 8 candidates $\times$ 4 scenarios, $\tau = 0.0$, FP32 critic)
  - $GW\sim\text{aa}$: `agent_v2_p2xxl_gw8x4_fp32~aa` (re-run noise control, identical binary)
- **Primary Contrast:** $GW - D$ (tests whether gated search improves upon direct play)
- **Supporting Contrast:** $GW - UW$ (tests whether the $\tau = 0.002$ override gate helps vs ungated search)
- **Family Multiplicity:** 2-contrast Bonferroni family ($\alpha = 0.05$, $z \approx 2.2414$)
- **Descriptive Contrasts:** $UW - D$, per-count rows (2..15), 3–15 equal-weight band, and 2-team band
- **Budget:** 96 fixtures per count across counts 2–15 = 1,344 paired fixtures $\times$ 4 arms = 5,376 formal games
- **Environment & Limits:**
  - Exactly 8 workers under `task.py lock cpu --task T0079`
  - Single-tenant host conditions (no GPU training or competing CPU processes)
  - First reply limit: 2,000 ms; steady reply limit: 50 ms
  - Opponents: held-out reference roster (`profiles/meow_rl_screen.json`: Fable, MiMo, Kimi, GLM, Player4)

## Schedule Derivation Contract

Per fixture $i$ under seed base $S$:
- Team count $N \in \{2..15\}$
- Seat uniform over $0..N-1$
- Map seed: derived deterministically via SHA-256 derivations
- Opponents: drawn from held-out reference roster (`profiles/meow_rl_screen.json`)
