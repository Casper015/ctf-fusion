# Meowmind RL Training Seed Ledger

This ledger reserves immutable seed bases for the Meowmind on-policy reinforcement learning pipeline ([MEOWMIND_RL_PLAN.md](MEOWMIND_RL_PLAN.md), tasks T0054-T0057).
Seeds are verified by `training.meow_rl.train.check_ledger` before any rollout collection or gradient step begins.
No reserved seed in this file may be reused for different configurations, evaluations, or test screens.

## Active Reservations

| Seed Base | Task | Arm | Budget (Decisions) | Purpose | Status |
|---|---|---|---|---|---|
| 20261901 | T0055 | `large_v` (`meow_public416_mlp_1024_1024_512_v1`) | 20,000 smoke + 1,000,000 full | Primary Large-V PPO Host Pilot | Reserved |
| 20261902 | T0055 | `small_v` (64-32-1 baseline) | 20,000 smoke + 1,000,000 full | Small-V Capacity Ablation Control | Reserved |
| 20261903 | T0055 | `large_v` | 1,000,000 | Replication Seed 102 (Large V) | Pre-reserved |
| 20261904 | T0055 | `large_v` | 1,000,000 | Replication Seed 103 (Large V) | Pre-reserved |
| 20261905 | T0055 | `small_v` | 1,000,000 | Replication Seed 102 (Small V) | Pre-reserved |
| 20261906 | T0055 | `small_v` | 1,000,000 | Replication Seed 103 (Small V) | Pre-reserved |
| 20261909 | T0059 | `large_v` (`meow_public416_mlp_1024_1024_512_v1`) | 20,000 smoke + 1,000,000 full | Pool v2 Primary Large-V PPO Run | Reserved |
| 20261910 | T0059 | `small_v` (64-32-1 baseline) | 20,000 smoke + 1,000,000 full | Pool v2 Small-V Capacity Ablation Control | Reserved |

## Seed Derivation Contract

Per episode $i$ of run with seed base $S$, all per-episode randomness is derived deterministically via SHA-256 derivations of `(S, purpose, i)`:
- Team count: uniform over 2..15
- Learner seat: uniform over $0..N-1$
- Map seed: `derive(S, "map", i)`
- Flag seed: `derive(S, "flag", i, bits=63)`
- Actor sampling seed: `derive(S, "act", i, bits=63)`
- Opponent families & members: drawn from `profiles/meow_rl_training_pool.json`

Schedule ordering is purely a function of $(S, i)$ and never depends on worker timing, process completion order, or host thread interleaving.
