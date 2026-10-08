# Phase A: evaluating every candidate opponent and selecting a diverse 15

The `roster/` package implements Phase A of the handoff [`handoffs/diverse15_distance_tiered_policy`](../handoffs/diverse15_distance_tiered_policy/NEXT_AGENT_PROMPT.md). It inventories every candidate bot, freezes a probe suite, fingerprints behavior, measures distances and noise, screens strength, and turns the evidence into the selector's input and a hash-locked collection pool.

The tools are built and tested. Fingerprints, strength screens and the final selection must be produced on the host. No roster has been selected yet.

## What runs where

| Step | Tool | Where | Evidence it gives |
|---|---|---|---|
| 1. Inventory | `roster.inventory` | anywhere (compiles every candidate) | identities, builds, aliases, code lineage, exclusions |
| 2. Probe suite | `roster.probes` | anywhere, from recorded matches | frozen states with event tags |
| 3. Fingerprints | `roster.fingerprint` | **host** (hours of bot CPU) | requested moves, actions, targets on identical histories |
| 4. Distances | `roster.distance` | anywhere | behavior distance, A/A noise, clusters per team count |
| 5. Strength | `roster.strength schedule / run / aggregate` | **host** (matches) | utility per team count with map-level intervals; reply times under official limits |
| 6. Assessments | `roster.assess` | anywhere | one row per candidate (`all_bot_assessments.csv`) |
| 7. Review | `roster.catalog --review-template` then a person | anywhere | eligibility, verified skills, contribution sentences |
| 8. Catalog and selection | `roster.catalog`, then the handoff's `select_roster.py` | anywhere | `selected15.json` |
| 9. Pool | `roster.pool` | anywhere | `profiles/pool_selected15.json`, source hashes enforced by `training.collect` |

## Host command sequence

```bash
export AI_TEST_ROOT=C:/path/to/AI_Test
python -m roster.inventory --out data/roster/bot_inventory.json
python -m training.collect --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --matches 6 --out data/raw_probes  # diverse histories, default pool
python -m roster.probes --raw data/raw_probes --per-count 3 --out data/roster/probe_suite.json
python -m roster.fingerprint --suite data/roster/probe_suite.json --inventory data/roster/bot_inventory.json --repeat 2 --workers 6
python -m roster.distance --suite data/roster/probe_suite.json --out data/roster/behavior_distance.json
python -m roster.strength schedule --maps-per-count 50 --out data/roster/screen_schedule.json
python -m roster.strength run --schedule data/roster/screen_schedule.json --inventory data/roster/bot_inventory.json --workers 8 --keep-matches data/raw_screen
python -m roster.strength aggregate --results data/roster/screen_results.jsonl --out data/roster/strength.json
python -m roster.assess --inventory data/roster/bot_inventory.json --distance data/roster/behavior_distance.json --strength data/roster/strength.json
python -m roster.catalog --assessments data/roster/bot_assessments.json --review-template data/roster/roster_review.json
#   ... a person fills in roster_review.json for the finalists, then confirmation runs for them:
python -m roster.strength schedule --maps-per-count 715 --seed 7 --out data/roster/confirm_schedule.json
python -m roster.strength run --schedule data/roster/confirm_schedule.json --inventory data/roster/bot_inventory.json --only <finalists> --results data/roster/confirm_results.jsonl
python -m roster.strength aggregate --results data/roster/confirm_results.jsonl --out data/roster/strength_confirm.json
python -m roster.catalog --assessments data/roster/bot_assessments.json --distance data/roster/behavior_distance.json --strength data/roster/strength_confirm.json --review data/roster/roster_review.json
python handoffs/diverse15_distance_tiered_policy/tools/select_roster.py --input data/roster/roster_evidence.json --out data/roster/selected15.json
python -m roster.pool --selected data/roster/selected15.json --inventory data/roster/bot_inventory.json
```

**Approximate cost, to be checked on the host:**
- **Fingerprints:** 122 eligible bots × 42 probes × 2 repeats × 400 turns, at 5–30 ms per reply, is about 10–25 CPU-hours, so 2–4 hours on 6 workers. `--turns 200` halves it.
- **Screen:** 50 maps per count is 85,400 games for 122 bots, about 5 hours at the S1 run's 4.6 games/s.
- **Confirmation:** 715 maps per count is 10,010 games per finalist, so about 25 finalists take roughly 15 hours.

## Methods

**Inventory.** It scans the handoff's directories plus `New_AI_Test` and every pool entry; `--scan-dir` adds more. Each candidate gets:
- a source hash, and the hash of its local `#include` closure;
- a code hash over its token stream without comments or layout;
- a build with the collection flags, or the compiler's error.

**Aliases** come from the byte hash and the code hash, never from binary hashes: toolchains stamp every build, so one file compiled twice already gives two different binaries on macOS, and Windows PE files carry a timestamp. One representative per code identity is evaluated. Representatives prefer pool entries, then the judge bots, exports, benchmark suites and stress-pack bots, then the rest.

**Lineage** is the Jaccard similarity of 8-token shingles, grouped into connected components at 0.7 (the default); groups at 0.5 and 0.9 are recorded too. At 0.5, forks chain official bots into our own family: a Player2 experiment shares 56% of Sol's code. **Exclusions** always carry a reason: no `main`, test or tool driver, build failure, unresolved include, or alias.

**First local inventory** (AI_Test working tree at `player2-ctf-bot`, 2026-10-08):

| Measure | Count |
|---|---:|
| Candidates | 191 |
| Distinct sources | 132 |
| Distinct programs (comments ignored) | 123 |
| Eligible | 122 |
| Lineage groups at 0.7 | 29 |

All builds succeeded. Code aliases include the stress-pack copies of the judge bots, and C18 is the same code as C17 apart from comments. The largest 0.7 groups are our own Control_Hybrid family (64) and the Player2 `cand-002` series (12).

**Probes.** A probe is one seat of one recorded match, replayed from turn 1, so a bot always sees a complete, valid history. Every bot answers the same probes. The history was produced by other bots' orders, which is the price of identical states; on-policy behavior comes from the screen's live matches, which `--keep-matches` stores.

Selection is deterministic: per team count, matches with the most distinct seat families and distinct maps, and in each the seat with the most tagged turns. Each turn carries pre-registered tags (`roster/common.py`):
- the four strata: quiet, contact, flag, respawn;
- pickup option, own carrier, escort, interception, focus fire, overkill risk, collision risk, return risk, endgame.

**Fingerprints.** Replies are canonicalized exactly like the training labels. Limits are relaxed (10 s first turn, 1 s later): a fingerprint asks what a bot wants to do. A replay of a deterministic bot reproduces its recorded replies exactly; the tests check this.

**Distance.** Within each tag, the rates of disagreeing moves, actions and attack targets are combined with weights 0.5, 0.3 and 0.2. A component without opportunities is left out, never counted as agreement. Tags weigh equally, so quiet turns cannot dominate. Probes are averaged per team count, and team counts equally.

Repeats give each bot's A/A noise. Two bots are equivalent at a team count when their distance is within the larger noise plus 0.02. Clusters are connected components, so close chains merge; review close pairs by hand.

**Strength.** Every candidate plays the same frozen fixtures: per team count and map, the candidate takes one seat and the frozen reference roster (ten judge bots, Player5, Baseline) fills the others. One map is one independent group. Utility is pairwise (beat 1, tie 0.5); the lower bound is mean − 1.96 SE over maps; team counts weigh equally. Reply times and faults are recorded under the official limits. The evidence level becomes "confirmation" at 10,000 maps with at least 500 per count.

**Review and selection.** The catalog takes every number from the measured files. Eligibility, verified skills and the one-sentence contribution come only from the human review file, which starts closed (`eligible: false`) with the measured hints. Bots with incomplete distances are dropped, worst coverage first, so one gap cannot remove the rest. The handoff's selector then enforces distinct clusters, at most three per lineage, a minimum pair distance, and the confirmation sample sizes.

## Not built yet

- **Legal alternate-state probes.** These are branches of a recorded state with different orders, continued through the engine. The replay probes and live screen matches cover the main need first.
- **Validation of the selection itself.** This compares the diverse 15 against the 15 strongest and a broader pool, at an equal training budget, on held-out lineages. It needs trained models.
