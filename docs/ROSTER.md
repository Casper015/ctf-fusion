# Phase A: opponent evidence, open training admission and frozen evaluation

The `roster/` package provides inventory, probes, behavior distances, strength evidence, assessments, the open training registry with its hierarchical sampler, a frozen evaluation roster writer and a fixed-panel pool writer. The updated handoff now requires an **open training pool with no 15-bot cap**, plus a separate frozen evaluation roster: [admission plan](../handoffs/diverse15_distance_tiered_policy/BOT_SELECTION_PLAN.md), [grouping/sampling review](../handoffs/diverse15_distance_tiered_policy/BOT_GROUP_AND_SAMPLING_REVIEW.md).

The tools are built and tested on synthetic and replayed data. Fingerprints, strength confirmation and the reviewed admission file remain host work. The fixed-size panel commands are a reference-panel workflow, not a limit on training membership. No final pool or panel is claimed here.

## What runs where

| Step | Tool | Where | Evidence it gives |
|---|---|---|---|
| 1. Inventory | `roster.inventory` | anywhere (compiles every candidate) | identities, builds, aliases, code lineage, exclusions |
| 1b. Source archive | `roster.archive save / check` | anywhere | every inventoried source stored in `roster/bots/` by sha256; later checks need no AI_Test working tree |
| 2. Probe suite | `roster.probes` | anywhere, from recorded matches | frozen states with event tags |
| 3. Fingerprints | `roster.fingerprint` | **host** (hours of bot CPU) | requested moves, actions, targets on identical histories |
| 4. Distances | `roster.distance` | anywhere | behavior distance, A/A noise, noise-corrected excess with probe-level uncertainty, joint-team distance, complete-linkage groups per team count |
| 5. Strength | `roster.strength schedule / run / aggregate` | **host** (matches) | utility per team count with map-level intervals; reply times under official limits |
| 6. Assessments | `roster.assess` | anywhere | one row per candidate (`all_bot_assessments.csv`) |
| 7. Review | `roster.catalog --review-template` then a person | anywhere | eligibility, verified skills, contribution sentences |
| 8. Admission | `roster.registry template`, then a person | anywhere | `admission.json`: who is admitted, why, roles, counts, reserved OOD lineages, sampling weights |
| 9. Open training pool | `roster.registry build` | anywhere | `profiles/open_training_pool.json`: members, per-count groups, mixture, simulated inclusion; `training.collect --pool` samples from it |
| 10. Frozen evaluation roster | `roster.registry freeze-eval` | anywhere | `profiles/frozen_eval_roster.json`: fixed members, source hashes, fixture policy, training/OOD overlap |
| Optional panel | `roster.catalog`, the handoff's `select_roster.py`, `roster.pool` | anywhere | A fixed-size reference panel (e.g. the members to freeze for evaluation); it never gates training admission. |

## Host command sequence

Run the evidence stages for all candidates, then admit training opponents and freeze an evaluation roster as separate artifacts. Reserve strict OOD lineages in the admission file before any collection.

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
python -m roster.registry template --assessments data/roster/bot_assessments.json --distance data/roster/behavior_distance.json --out data/roster/admission.json
#   ... a person admits bots (contribution sentence, roles, counts) and reserves OOD lineages, then:
python -m roster.registry build --admission data/roster/admission.json --assessments data/roster/bot_assessments.json --distance data/roster/behavior_distance.json --inventory data/roster/bot_inventory.json --out profiles/open_training_pool.json
python -m training.collect --pool profiles/open_training_pool.json --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 8
#   Confirmation for the members whose strength claims matter (field members, panel candidates):
python -m roster.catalog --assessments data/roster/bot_assessments.json --review-template data/roster/roster_review.json
python -m roster.strength schedule --maps-per-count 715 --seed 7 --out data/roster/confirm_schedule.json
python -m roster.strength run --schedule data/roster/confirm_schedule.json --inventory data/roster/bot_inventory.json --only <finalists> --results data/roster/confirm_results.jsonl
python -m roster.strength aggregate --results data/roster/confirm_results.jsonl --out data/roster/strength_confirm.json
python -m roster.catalog --assessments data/roster/bot_assessments.json --distance data/roster/behavior_distance.json --strength data/roster/strength_confirm.json --review data/roster/roster_review.json
python handoffs/diverse15_distance_tiered_policy/tools/select_roster.py --input data/roster/roster_evidence.json --out data/roster/selected15.json
python -m roster.registry freeze-eval --inventory data/roster/bot_inventory.json --selected data/roster/selected15.json --registry profiles/open_training_pool.json --name core15 --out profiles/frozen_eval_roster.json
```

After confirmation, re-run `roster.assess` and `roster.registry build` (a new `revision`) so members show `confirmed` evidence. Freeze the pool file and its `identity` for each training run.

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

**Source archive.** `python -m roster.archive save` copies every inventoried source into `roster/bots/src/<sha256>.<ext>`, once per content. That covers aliases and excluded candidates, plus local includes. `roster/bots/manifest.json` maps each candidate to its file, its AI_Test path and commit, lineage, alias and exclusion; bots already in this repository are recorded, not copied. `.gitattributes` marks the files `-text`, so no checkout converts line endings; 16 of the archived sources use CRLF. `python -m roster.archive check --pool profiles/open_training_pool.json --ai-test AI_TEST` verifies three things:

- every stored file still hashes to its name;
- every member of the given pools or rosters is archived;
- which AI_Test files changed since archiving.

Re-run `save` after each new inventory; entries accumulate. First archive: 191 candidates, 130 stored files, 8.5 MiB, from AI_Test `c9c115b`.

**Probes.** A probe is one seat of one recorded match, replayed from turn 1, so a bot always sees a complete, valid history. Every bot answers the same probes. The history was produced by other bots' orders, which is the price of identical states; on-policy behavior comes from the screen's live matches, which `--keep-matches` stores.

Selection is deterministic: per team count, matches with the most distinct seat families and distinct maps, and in each the seat with the most tagged turns. Each turn carries pre-registered tags (`roster/common.py`):
- the four strata: quiet, contact, flag, respawn;
- pickup option, own carrier, escort, interception, focus fire, overkill risk, collision risk, return risk, endgame.

**Fingerprints.** Replies are canonicalized exactly like the training labels. Limits are relaxed (10 s first turn, 1 s later): a fingerprint asks what a bot wants to do. A replay of a deterministic bot reproduces its recorded replies exactly; the tests check this.

**Distance.** Within each tag, the rates of disagreeing moves, actions and attack targets are combined with weights 0.5, 0.3 and 0.2. A component without opportunities is left out, never counted as agreement. Tags weigh equally, so quiet turns cannot dominate. Probes are averaged per team count, and team counts equally (`overall`, a summary only).

**Equivalence** follows the review's three safeguards:

- **Noise is not equivalence.** Repeats give each bot's A/A noise. A pair is scored by its excess distance per probe, d(A,B) − (noise(A) + noise(B)) / 2. For categorical choices this is half the squared difference of the two conditional distributions, zero only when they match. A 50/50 mixture of two deterministic policies is therefore equivalent to neither; under the old rule (the larger noise plus 0.02) all three chained together. Bots with one run per probe have unknown noise, taken as 0, so they can only look more distinct. Bots with noise above 0.1 are listed as `stochastic`.
- **Joint team requests.** A fourth measure, `joint`, is the rate at which the whole team's requests differ. Its excess separates a coordinated team from independent units with identical per-unit marginals.
- **Uncertainty, per count, no chains.** A pair is equivalent at a team count when the upper bound of its excess over that count's probes (mean + 1.645 SE, at least 2 probes) is within 0.02, and the joint excess bound within 0.05. Groups are complete-linkage, so every pair inside a group is equivalent and each group reports its diameter. Over all counts a pair is equivalent only if it is equivalent at every count. A duel specialist that matches a reference at 3–15 teams therefore stays its own group; the equal-count average would have put it at 0.014 against a 0.02 margin. `distinct_counts` lists the counts at which a bot matches nobody.

**Strength.** Every candidate plays the same frozen fixtures: per team count and map, the candidate takes one seat and the frozen reference roster (ten judge bots, Player5, Baseline) fills the others. One map is one independent group. Utility is pairwise (beat 1, tie 0.5); the lower bound is mean − 1.96 SE over maps; team counts weigh equally. Reply times and faults are recorded under the official limits. The evidence level becomes "confirmation" at 10,000 maps with at least 500 per count.

**Review and selection.** The catalog takes every number from the measured files. Eligibility, verified skills and the one-sentence contribution come only from the human review file, which starts closed (`eligible: false`) with the measured hints. Bots with incomplete distances are dropped, worst coverage first, so one gap cannot remove the rest. The handoff's selector then enforces distinct clusters, at most three per lineage, a minimum pair distance, and the confirmation sample sizes.

Those constraints describe the optional panel selector. Open training admission instead keeps reliable, useful specialists with explicit count/context evidence and weighted redundancy control. A source-lineage group is not a demonstrated style.

**Open training pool (`roster.registry`).** There is no size cap. A person admits a bot with a contribution sentence, roles and team counts (`template` lists every evaluable representative, nothing admitted, with measured hints). The measured gates then apply:

- the bot is an inventory representative; an alias shares its representative's entry and mass;
- it is outside the reserved OOD lineages;
- its source hash is the one it was measured with;
- valid replies are at least 99%, with no dead probe runs, and strength-game faults are at most 0.01 per game when screened;
- it has behavior evidence at each admitted count; counts without it are dropped and reported.

Strength is evidence, not a gate. Members are `confirmed` with the 10,000-map confirmation, otherwise `provisional`. Roles:

- `style`: behavior coverage; every admitted policy.
- `field`: the intended strong competition.
- `hard`: a verified hard case, at its `hard_counts`.
- `augmentation`: random or noise policies. They sit outside the mixture, and their family is prefixed `augmentation:` so training never uses their requests as P labels.

Behavior groups per count are recomputed among the admitted members only.

**Sampling (`roster.sampling`).** At each team count N:

ρ_N = λ_field·ρ_field + λ_style·ρ_style + λ_hard·ρ_hard

The default weights are 0.6, 0.3 and 0.1, an initial hypothesis to tune and freeze. Each component gives every behavior group at N equal mass, split among its versions (1 / (K_N·|C_N(b)|)). A lineage cap (default 0.5 of a component) then applies. An empty component is dropped and the others renormalized. Seats are filled one by one: component, then a bot that still has copies left (`max_copies`, default 1). Optional regimes are an augmentation seat (`augmentation`) and a controlled same-policy concentration (`focus`, `focus_fraction`).

The pool file records each count's groups, effective weights, ρ and the inclusion probabilities the sampler actually produces (simulated). A count whose games always have the same composition is a warning; 15 policies at 15 seats is one example. `training.collect --pool profiles/open_training_pool.json` draws seats this way (the profile's anchors are not used). Each fixture and match record carries the regime, each seat's component and the pool's `identity`, a hash of membership and sampling.

**Frozen evaluation roster (`roster.registry freeze-eval`).** It lists fixed members with source hashes, a fixture policy and its own identity. Each member is marked as in training or not, and as in a reserved OOD lineage or not. A selector panel can be frozen this way.

## Not built yet

- **Context-level sampling and count weighting.** The sampler balances behavior groups per team count. Per event-tag profiles are reported (`distinctive_tags`, `contexts` in the admission file) but do not yet drive state-level stratification or loss weights. Equal matches per count give N acting-team rows per turn, so training must choose its count weighting explicitly.
- **Response profiles.** Per-count results against the frozen reference roster exist (`per_count` utility). Matchup response diversity, meaning how the same opponents fare against each candidate, is not yet a sampling component.
- **Continuing-policy declaration for V.** The pool records who was sampled and how, but the V label's opponent distribution is not yet declared in the training manifest.
- **Compiler-aware aliasing.** Token-stream code aliases ignore preprocessor line structure; treat them as provisional until behavior confirms them.
- **Legal alternate-state probes.** These are branches of a recorded state with different orders, continued through the engine. The replay probes and live screen matches cover the main need first.
- **Validation of the selection itself.** This compares the diverse 15 against the 15 strongest and a broader pool, at an equal training budget, on held-out lineages. It needs trained models.
