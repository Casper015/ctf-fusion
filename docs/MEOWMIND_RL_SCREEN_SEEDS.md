# Meowmind RL Screen Seed Ledger

This ledger reserves immutable seed bases for the Meowmind RL deployment screens (task T0057, [MEOWMIND_RL_DEPLOYMENT.md](MEOWMIND_RL_DEPLOYMENT.md)).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Active Reservations

| Seed Base | Block | Fixtures | Games | Purpose | Status |
|---|---|---|---|---|---|
| 20261950 | preflight | 16 | 32 | Host timing preflight (ladder 1..6 workers, 2..15 teams) | Reserved |
| 20261951 | primary | 392 (28/count) | 1,176 | Primary screen: RL vs Warm Start + A/A | Reserved |
| 20261952 | secondary | 392 (28/count) | 3,136 | Secondary screen: 6 prespecified Bonferroni contrasts | Reserved |

## Schedule Derivation Contract

Per fixture $i$ under seed base $S$:
- Team count $N \in \{2..15\}$
- Seat uniform over $0..N-1$
- Map seed: derived deterministically via SHA-256 derivations
- Opponents: drawn from held-out reference roster (`profiles/meow_rl_screen.json`)
