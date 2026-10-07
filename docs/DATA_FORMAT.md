# Data format

## Raw matches: `data/raw/nXX/<match_id>.json.gz` (schema `ctf-fusion-match-1`)

One gzip JSON object per match, written by `training/collect.py`. It stores what crossed the protocol and nothing derived.

| Field | Meaning |
|---|---|
| `match_id` | `n{teams:02d}-s{seed}-{index:06d}`. Fixtures are a pure function of seed, team count and index. |
| `teams`, `map_seed`, `flag_seed` | Rebuild the engine game: `engine.Game(teams, map_seed, flag_seed=flag_seed)`. |
| `seats` | Per seat: `name`, `family` and the bot source's `sha256`. |
| `engine_sha256`, `limits_ms` | Engine file hash, and the (turn 1, later turns) time limits used. |
| `init` | Seat 0's INIT block. Seats differ only in the third token (`me`); see `matchlog.init_for_seat`. |
| `blocks[t-1]` | The TURN block every seat received before turn `t` settled. It is identical for all seats; `collect.py` asserts this every turn. |
| `final_block` | The state after the last settlement (TURN 401). Bots never receive it; labels use its events and score. |
| `replies[t-1][seat]` | The reply line exactly as sent, or `null` for a timeout or a dead process. |
| `faults` | `[turn, seat, reason]` for every strike, including malformed lines. |
| `result` | Final `score`, `ranks` and per-team engine `stats`. |
| `bots` | Per seat: timing (`avg_ms`, `max_ms`, `first_ms`), fault count and stderr tail. |

`matchlog.seat_stream(match, me)` rebuilds the exact stdin of seat `me`. The legacy forecast replay and the parity driver both consume it.

## Shards: `data/shards/<tier>/<split>/shard_XXXX.npz`

Written by `training/dataset.py`. Rows are `(turn, team)` pairs from the kept turns.

| Array | Shape, type | Meaning |
|---|---|---|
| `rec` | (R, 640) int16 | Extractor record (layout in [FEATURES.md](FEATURES.md)) |
| `y_move` | (R, 3) int8 | Requested move per unit, canonical (a move into a wall becomes S) |
| `y_joint` | (R,) int16 | `m0*25 + m1*5 + m2`, or -1 when the reply was missing or malformed |
| `y_act` | (R, 3) int8 | 0 wait, 1 pickup, 2 drop, 3 attack; requests the engine would ignore become 0; -1 when the unit is off the board |
| `y_tgt` | (R, 3) int8 | Candidate slot of the attack target; -1 if no attack, or a hittable target beyond the 8 slots |
| `blocked` | (R, 3) bool | A non-stay legal request that ended where it started (collision) |
| `valid` | (R,) bool | The reply parsed; policy rows need it, value rows do not |
| `v_y` | (R, 3) float32 | Pairwise final outcome, capture within 20 turns, own death within 10 turns |
| `stratum` | (R,) int8 | 0 flag, 1 contact, 2 respawn, 3 quiet (`labels.strata`) |
| `weight` | (R,) float32 | stride / keep rate: importance weight back to the natural turn distribution |
| `match`, `teams`, `family`, `turn`, `team` | (R,) | Index into `index.json` matches and families; team count; acting team's bot family |
| `qh_XXXX.npy` | (R, 3, 5) float32 | Optional, from `heuristic_baseline.py`: legacy forecast move marginals (NaN = no entry) |

`data/shards/index.json` holds the extractor schema, the match table (id, file, split, families, seats), the family names, the options and the row counts.

## Leakage rules

1. **Public inputs only.** Features are computed only from blocks up to the current one: the public stream a bot sees. Other teams' replies are never features; previous moves are inferred from public positions.
2. **No future randomness.** Future flag respawn spots (`flag_seed`) never reach features; only blocks do.
3. **Split by match.** Train, val and test are split by a hash of the match id, so adjacent turns never straddle splits. `--holdout-family` adds an `ood` split for an opponent family never seen in training.
4. **Requested, not resolved.** Labels are requested actions. Resolved positions only feed the `blocked` diagnostic.
