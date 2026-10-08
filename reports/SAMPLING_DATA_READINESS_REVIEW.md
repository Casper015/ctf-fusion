# Sampling and training-data readiness review

Reviewed source: `915ddc7e7c0fd85ac754295f1084de5367489be0` in Casper015/ctf-fusion. Review date: 2026-10-08. This is a source audit plus small synthetic sampling tests; it is not a training run, match benchmark, latency benchmark, or certification of a collected dataset.

## Verdict

The new sampler is a substantial improvement over sampling source files uniformly or fixing the training pool at 15 bots. It implements the intended open pool, behavior groups per team count, a field/style/hard mixture, lineage redundancy control, seat randomization, optional concentration and augmentation regimes, and actual inclusion diagnostics. It can generate a useful first training dataset. It does **not** yet demonstrate that a particular model size, especially a model beyond XXL, has enough independent and strategically relevant data.

No `data/raw`, `data/shards/index.json`, generated `profiles/open_training_pool.json`, or trained-run manifests were present in the reviewed checkout. The host capacity reports measure inference work, not training coverage or predictive strength. All sample-volume estimates below are planning calculations, not observations.

## What is implemented well

- Behavior grouping is per count and complete-linkage: an unmeasured pair cannot cause a merge, and A-B/B-C agreement cannot chain dissimilar A-C policies together. Requested joint team actions help distinguish coordination even when individual marginals match. A/A noise is treated separately from equivalence. [Distance implementation](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/distance.py#L127-L161)
- Each sampling component balances groups before versions, then caps lineage mass. Empty components are renormalized. Augmentation does not enter P labels; optional focus games deliberately test same-policy concentration. Actual inclusion, composition entropy and forced-repeat counts are computed with the same drawing code as collection. [Sampler](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/sampling.py#L89-L189)
- Registry membership has no size cap and requires a contribution statement. Strong bots, weak specialists and augmentation have distinct roles. Raw games carry registry identity and seat components. [Registry](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/registry.py#L103-L205), [raw collection metadata](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/collect.py#L164-L174)
- Labels retain requested collision-blocked movement instead of reverse-engineering it from resolved positions. Invalid replies do not become P labels. Default OOD mode reserves every team's rows in a fixture containing a held-out family. The feature encoder is shared between training and inference. [Labels](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/fusion/labels.py#L42-L107), [split and subsampling](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/dataset.py#L77-L106)

## Findings to fix before a large collection run

### P2: Unknown count coverage can be admitted as measured behavior

`distance.compute` uses the global bot ID set in every count's clustering. A bot with no runs at count N still gets a singleton cluster there. `registry.build` then accepts membership in that singleton as evidence at N. This confuses "unknown, do not merge" with "measured, admit". It can allocate style mass to an unmeasured specialist and fail to warn about missing coverage. [Global IDs and count clusters](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/distance.py#L180-L256), [admission check](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/registry.py#L141-L147)

Synthetic reproduction: A was replayed at 3 and 5 teams; B only at 3 teams. The distance output still listed B as a singleton at 5. The registry admitted B for `[3,5]`, with an empty `counts_without_behavior_evidence` warning. Require explicit per-bot/per-count successful probes and distinct-map coverage, not cluster membership. Add a regression test for an unmeasured singleton; keep that singleton out of the "distinctive measured specialist" list as well.

### P2: Existing evidence can be reused under the wrong source identity

Fingerprint resume skips existing filenames, while loading groups records by bot ID only. Assessments stamp the current inventory source hash but do not compare it with each fingerprint run's recorded source hash. Updating a bot at the same ID and rebuilding the inventory can therefore attach stale behavior evidence to its current source, allowing the registry's source-hash comparison to pass. Strength aggregation likewise drops the original source identity while grouping by candidate and fixture. [Fingerprint resume and loading](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/fingerprint.py#L133-L169), [assessment identity](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/assess.py#L47-L71), [strength aggregation](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/roster/strength.py#L150-L187)

Fail closed on incompatible source/include-closure identity, probe suite identity and relevant limits. Use those identities in evidence-directory naming or verify every reused file. Preserve them in the aggregate, not only the original records.

### P2: Collection resume and shard manifests do not freeze the complete experiment

Collection fixture filenames use count, seed and index, not registry identity. Resume skips an existing file without checking its seats, registry revision, bot hashes, engine hash or limits. Reusing an output directory after a pool update can silently retain old games. The shard builder then drops raw `sampling.registry`, regime, seat components, source hashes, flag seed, engine hash and limits from its indexed match metadata. Its data digest hashes only match IDs. Consequently the training config cannot prove which continuation distribution or program identities produced the data. [Resume](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/collect.py#L210-L234), [indexed metadata and digest](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/dataset.py#L96-L132), [training manifest](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/common.py#L86-L95)

Create an immutable collection-run manifest, refuse incompatible resume, and preserve raw file content hashes plus the sampling/engine/bot identities in the dataset index. Deliberate mixtures of registry revisions can be supported, but must be declared and reported. Raw paths remain available to recover this information; it is not presently part of the training manifest.

### P2: General-model training does not implement a declared count weighting

Equal matches per count do not mean equal training influence: a game at N teams produces N acting-team rows per turn. The `stride / keep_rate` weights restore the natural row distribution **inside the collected fixtures**, but neither P nor V applies equal-count or target-competition count weights. `--max-rows` also samples uniformly over the pooled rows. [Row weights](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/dataset.py#L87-L94), [P preparation and loss](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/train_policy.py#L28-L35), [V preparation and loss](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/train_value.py#L57-L79)

For otherwise equal complete games at all counts 2..15, natural team-row weighting gives count 15 about 7.5 times count 2's mass. Before filtering, its expected share is 15/119 = 12.61%, versus 2/119 = 1.68%. This is not inherently wrong if it matches the intended deployment distribution, but it is wrong to describe the objective as equal-count. Add a declared count prior `q(N)`; use balanced batches or a count factor proportional to `q(N) / observed_mass(N)`, and report both natural and equal-count validation metrics.

## Sampling caveats that are not implementation bugs

### Component lambdas are not the final seat distribution

Without replacement, a small component can be exhausted. Its remaining seats come from other components. Group and lineage caps apply to the initial component distributions, not a per-game composition guarantee. This is correctly visible in `inclusion()`, but the diagnostics need to be an actual admission/tuning gate.

A synthetic 3,000-draw example with 25 distinct policies, only two field members, 15 seats and one copy per policy had effective field/style lambda 0.6667/0.3333. Its actual component seat shares were **0.1294 field, 0.8706 style**, and it used both field bots in every game. More maps cannot make this ensemble a 60% field composition. Enlarge the field subset, explicitly allow controlled repeats, or choose regimes that match the intended mixed-opponent distribution. Preserve source diversity within those regimes.

### Rare events still need coverage checks

`contexts` and distinctive tags are metadata, not collection quotas or loss weights. Shard strata are mutually exclusive flag/contact/respawn/quiet labels; flag and contact are kept at rate 1, but still undergo the global stride. Thus an isolated critical turn can be discarded. Declaring a larger pool or more ordinary games does not prove coverage of drop-then-fire, enemy drop, score denial, synchronized focus, multi-party interference, or late-game reversals. [Declared limitations](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/docs/ROSTER.md#L131-L138)

Report multi-tag event counts and **distinct maps containing each event**, by count, family/group and split. Retain critical windows without stride, with correct inclusion weights; add targeted legal engine-generated scenarios or branch continuations when ordinary games do not cover them. Keep a natural-distribution evaluation set to avoid mistaking oversampled rare events for overall gains.

### Strict OOD must be checked at map level too

Fixture holdout is much stronger than actor-row holdout and fixes the ordinary same-game leakage problem. If a map seed is reused across fixtures and only some fixtures contain an OOD family, current fixture-mode logic can still place the same map into OOD and a regular split. For a strict unseen-map claim, reserve the union of OOD map IDs across all fixtures, and audit split intersections. Reserve OOD lineages before collection as already documented.

### V is a continuation-policy value, not a bot-independent oracle

Final outcomes label behavior generated by the sampled bots, including fault-affected games and augmentation regimes. They estimate value under that continuation distribution, conditional on the compressed features. A changed bot policy or search-generated state distribution needs fresh continuations, policy-compatible targets or an explicitly justified correction. Increasing hidden width cannot repair missing state information, unobserved branches or a mismatched continuing-policy target. The capture/death auxiliaries are not flag-specific delivery labels. [V targets](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/fusion/labels.py#L92-L104)

## Planned sample volume

Profiles specify 1,500 matches per count for phase 1 and 7,500 total per count for phase 2. The collector reads `matches`; reaching phase 2 requires an explicit collection command/start range. Default shard sampling has stride 2 and keep rates flag/contact 1.0, respawn 0.6, quiet 0.35. [Profile](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/profiles/n05.json), [defaults](https://github.com/Casper015/ctf-fusion/blob/915ddc7e7c0fd85ac754295f1084de5367489be0/training/dataset.py#L53-L56)

For M complete 400-turn games at N teams, expected retained team rows are:

`R_N = M × 400 × N × (p_flag + p_contact + 0.6 p_respawn + 0.35 p_quiet) / 2`.

The stratum shares sum to one. The broad endpoints below correspond to all quiet versus all fully kept turns; real volume depends on observed play. P then loses invalid and augmentation-family rows. These are **team rows**, not independent maps and not three independent unit decisions.

| Count | Phase-1 retained rows | Approx. train rows, before OOD/P filtering | Independent maps train/val/test, before OOD |
|---:|---:|---:|---:|
| 2 | 210,000–600,000 | 168,000–480,000 | 1,200 / 150 / 150 |
| 5 | 525,000–1,500,000 | 420,000–1,200,000 | 1,200 / 150 / 150 |
| 8 | 840,000–2,400,000 | 672,000–1,920,000 | 1,200 / 150 / 150 |
| 15 | 1,575,000–4,500,000 | 1,260,000–3,600,000 | 1,200 / 150 / 150 |
| All 2..15 | 12,495,000–35,700,000 | 9,996,000–28,560,000 | 16,800 / 2,100 / 2,100 |

Map split sizes are expectations under the hash splitter, not fixed quotas. Phase 2 at 7,500 games/count multiplies these volumes and map counts by five. Repeated maps, incomplete games, OOD fixture removal, reply validity and labels change usable counts. One game's V final result is repeated over its turns: 400 repeated labels are not 400 independent outcome observations.

These volumes are sufficient to begin a measured learning-curve experiment. They are not a universal proof of sufficiency for XXL or larger models. An effective sample count cannot be inferred from row count divided by parameter count.

## Required pilot and scaling decision

1. Freeze the pool, actual seat distribution, collection seeds, engine, sources and split policy. Fix the evidence-coverage and resume/provenance findings first.
2. Collect a pilot for representative counts 2, 5, 8 and 15; publish a manifest with valid P rows, source/group masses, faults, unique maps and multi-tag event/map coverage. Freeze untouched regular and whole-lineage OOD evaluations.
3. Train multiple capacities on the **same map subsets** and training objective. Plot learning curves at increasing fractions of independent training maps; repeat training seeds. Compare integer deployment P NLL/calibration, conditional action/target performance, joint coordination, V outcome error, and event-specific failures with map-clustered intervals.
4. If a larger model improves train loss but worsens OOD/validation, gather better/diverse independent data or adjust regularization. If both train and validation remain poor, inspect representation and targets before assuming additional capacity will help. If curves still improve with maps, continue collecting; do not stop at an arbitrary row quota.
5. Test decision regret and equal-time paired bot outcomes separately. Use a full-turn budget and compare added inference time against the search work it displaces. Faster inference makes a capacity candidate feasible, not necessarily useful. Distill a larger offline teacher into a smaller CPU student only if measured prediction and decision improvements survive distillation.

## Tests actually run for this review

`python3 -m unittest -v tests.test_roster.TestDistance tests.test_roster.TestSampling tests.test_roster.TestRegistry`: **15 tests passed**. These are pure synthetic tests; no games, inference timing or training were run. Two additional inline synthetic checks reproduced (1) the unmeasured-count singleton admission and (2) field-seat depletion described above. Their existence shows why passing the current unit tests alone does not establish collection-data readiness.

No core source files were changed by this review.
