# Count Specialists RL Training Seed Ledger

This ledger reserves immutable seed bases for task T0068 (`Train exact-count PPO policy and value specialists`).
Seeds are verified by `training.meow_rl.train.check_ledger` before any rollout collection or gradient step begins.
No reserved seed in this file may be reused for different configurations, evaluations, or test screens.

## Active Reservations

| Seed Base | Task | Arm / Count | Budget (Decisions) | Purpose | Status |
|---|---|---|---|---|---|
| 20262001 | T0068 | `general_control` (all counts 2-15) | 1,400,000 total (incl. 20,000 smoke) | Compute-matched general RL control arm | Reserved |
| 20262002 | T0068 | `spec_n02` (count 2) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (2 teams) | Reserved |
| 20262003 | T0068 | `spec_n03` (count 3) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (3 teams) | Reserved |
| 20262004 | T0068 | `spec_n04` (count 4) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (4 teams) | Reserved |
| 20262005 | T0068 | `spec_n05` (count 5) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (5 teams) | Reserved |
| 20262006 | T0068 | `spec_n06` (count 6) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (6 teams) | Reserved |
| 20262007 | T0068 | `spec_n07` (count 7) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (7 teams) | Reserved |
| 20262008 | T0068 | `spec_n08` (count 8) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (8 teams) | Reserved |
| 20262009 | T0068 | `spec_n09` (count 9) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (9 teams) | Reserved |
| 20262010 | T0068 | `spec_n10` (count 10) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (10 teams) | Reserved |
| 20262011 | T0068 | `spec_n11` (count 11) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (11 teams) | Reserved |
| 20262012 | T0068 | `spec_n12` (count 12) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (12 teams) | Reserved |
| 20262013 | T0068 | `spec_n13` (count 13) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (13 teams) | Reserved |
| 20262014 | T0068 | `spec_n14` (count 14) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (14 teams) | Reserved |
| 20262015 | T0068 | `spec_n15` (count 15) | 100,000 total (incl. 20,000 smoke) | Count-specialist PPO pi/V (15 teams) | Reserved |
| 20262016 | T0068 | `calibration` | 560 episodes (40/count) | Aligned calibration fixtures per count | Reserved |
| 20262017 | T0068 | `host_preflight` | 64 games (16/count) | Exact-eight-worker host feasibility preflight (`agent_v2_rl_csp`) | Reserved |

## Seed Derivation Contract

Per episode $i$ of run with seed base $S$, all per-episode randomness is derived deterministically via SHA-256 derivations of `(S, purpose, i)`:
- Team count: specified by count constraint or uniform over 2..15 for general control
- Learner seat: uniform over $0..N-1$
- Map seed: `derive(S, "map", i)`
- Flag seed: `derive(S, "flag", i, bits=63)`
- Actor sampling seed: `derive(S, "act", i, bits=63)`
- Opponent families & members: drawn from `profiles/meow_rl_training_pool_v2.json`
