# Data format

## Opponent pool (`profiles/pool.json`)

| Group | Bots | Source in AI_Test |
|---|---|---|
| Judge field (Doubao excluded by request) | Astra, Sol, Fable, Sonnet, Opus, DeepSeek, Qwen, MiMo, Kimi, GLM | `ctf-arena/Player7/judge/ctf/bots/` |
| Earlier players | Player5 (the original, track 1's +347.3 reference), Player4 | `ctf-stress-pack/bots/` |
| Our lineage | Control_Hybrid, Player5_Omega, Player5_Omega_LIT, R3_CFL_conditional, Player2_v005, Astra_Full | `round3/export/`, `ctf-benchmark-v3/track1_10k_suite/bots/` |
| Reference | Baseline | `ctf-stress-pack/bots/baseline.cpp` |
| Noise | Noise: uniformly random moves (`tools/noise_bot.py`), weight 0.25 | this repository |

**Seat sampling.** Seats are drawn by weight without replacement, so each seat goes to a different bot when possible. At weight 0.25, Noise sits in about 4% of duels, 3% of 3-team, 12% of 8-team and 29% of 15-team matches (measured on 4,000 scheduled fixtures per count).

**Noise labels.** Noise has `"policy_labels": false`. Its own requests are excluded from opponent-policy training and evaluation (`training.common.policy_rows`; `--keep-unlabeled` overrides). Its rows still train the value net, and its random movement shapes everyone else's states.

**Required AI_Test version.** Collection needs an AI_Test checkout at `main` c840ab5 or later, which has the exports and benchmark bots.

## Raw matches: `data/raw/nXX/<match_id>.json.gz` (schema `ctf-fusion-match-1`)

One gzip JSON object per match, written by `training/collect.py`. It stores what crossed the protocol and nothing derived.

| Field | Meaning |
|---|---|
| `match_id` | `n{teams:02d}-s{seed}-{index:06d}`. Fixtures are a pure function of seed, team count and index. |
| `teams`, `map_seed`, `flag_seed` | Rebuild the engine game: `engine.Game(teams, map_seed, flag_seed=flag_seed)`. |
| `seats` | Per seat: `name`, `family` and the bot source's `sha256`. |
| `engine_sha256`, `limits_ms` | Engine file hash, and the (turn 1, later turns) time limits used. |
| `ai_test_commit`, `platform` | AI_Test checkout commit and the collecting host's platform. |
| `init` | Seat 0's INIT block. Seats differ only in the third token (`me`); see `matchlog.init_for_seat`. |
| `blocks[t-1]` | The TURN block every seat received before turn `t` settled. It is identical for all seats; `collect.py` asserts this every turn. |
| `final_block` | The state after the last settlement (TURN 401). Bots never receive it; labels use its events and score. |
| `replies[t-1][seat]` | The reply line exactly as sent, or `null` for a timeout or a dead process. |
| `faults` | `[turn, seat, reason]` for every strike, including malformed lines. |
| `result` | Final `score`, `ranks` and per-team engine `stats`. |
| `bots` | Per seat: timing (`first_ms`, `avg_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `max_ms`, `late`), fault count, stderr tail. |

`matchlog.seat_stream(match, me)` rebuilds the exact stdin of seat `me`. The legacy forecast replay and the parity driver both consume it.

## Shards: `data/shards/nXX/<split>/shard_XXXX.npz`

Written by `training/dataset.py`, one directory per team count. The general model and every dedicated model read the same files ([MODEL_MATRIX.md](MODEL_MATRIX.md)). Rows are `(turn, team)` pairs from the kept turns.

`--append` adds only matches the index does not list yet, as new shard files. It refuses changed sampling options or a changed feature schema, so phase-1 rows stay valid when phase 2 arrives.

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

`data/shards/index.json` holds:
- the extractor schema and sampling options;
- the match table (id, file, split, seats, families, AI_Test commit);
- the family names;
- row counts per `nXX/split`;
- the append history;
- the **dataset manifest**: matches per team count and split, seat distribution by bot and by family, AI_Test commits, collection seeds and the repository commit.

Every training run records the manifest of its team counts, with a digest of their match ids, in its `config.json`.

## Leakage rules

1. **Public inputs only.** Features are computed only from blocks up to the current one: the public stream a bot sees. Other teams' replies are never features; previous moves are inferred from public positions.
2. **No future randomness.** Future flag respawn spots (`flag_seed`) never reach features; only blocks do.
3. **Split by match.** Train, val and test are split by a hash of the match id, so adjacent turns never straddle splits. `--holdout-family` adds an `ood` split for an opponent family never seen in training.
4. **Requested, not resolved.** Labels are requested actions. Resolved positions only feed the `blocked` diagnostic.
