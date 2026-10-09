# Fingerprint effectiveness review and v2 implementation plan

Reviewed source: `Casper015/ctf-fusion` at `1876a8ef747236e0684722af448aa7b17fa6d9fb` (2026-10-08). This audit checks the supplied description against source and bounded synthetic experiments, followed by an implementable modification plan. It does not measure actual roster separation, train models, or run bot matches. Everything in the implementation-plan sections is proposed, not already implemented.

## Assessment of the supplied report

The central distinction is correct: common-history replay measures **conditional requested behavior on selected histories**, while live matches measure resulting performance. The existing tooling is a useful first stage; describing it as already “very precise” at separating tactical styles is unsupported without held-out validation.

- There are exactly 13 tags, move/action/target disagreement weights are 0.5/0.3/0.2, and whole-team disagreement, repeated-run A/A measurements, per-count comparisons and complete-link clustering are implemented. Existing tests cover marginal-versus-joint behavior and prevent chaining. [Tags](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/common.py#L18-L23), [distance implementation](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/distance.py#L59-L161), [tests](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/tests/test_roster.py#L89-L192).
- Tags describe **opportunities or state proxies**, not executed tactics. `pickup_option` means that some unit has a legal pickup at some candidate move; `focus_fire` means a candidate has supporting shooters; `overkill_risk` adds HP <= 34; `collision_risk` means some legal destination is currently occupied. `return_risk` merely checks for any dropped flag counted by the global feature, not that our team faces a dangerous return. There is no explicit escape-direction, effective-shot, survival or delivery-rate fingerprint. [Tag definitions](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/probes.py#L51-L79), [feature semantics](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/training/specs/FEATURES.md).
- Actions are canonicalized to requested labels using the shared feasibility features: some ineffective requests collapse to wait/stay. This is desirable for behavioral comparison, but is not a substitute for engine settlement with all players' actual moves. Raw ignored-request rate should remain a separate quality metric, distinct from malformed replies. Replay retains raw output lines. [Canonicalization](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/fusion/labels.py#L65-L89), [fingerprint records](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/fingerprint.py#L77-L125).
- The supplied report misses an existing source of outcome metrics: collection already stores `result.stats`; `strength.play_one` discards those statistics from its compact record unless full matches are retained. Extending that aggregation is easier than reconstructing everything from events. Parsed events also currently omit death details and capture flag identity retained in the raw block. [Collection](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/training/collect.py#L162-L173), [strength](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/strength.py#L77-L91), [event parser](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/fusion/protocol.py#L35-L46).

## Demonstrated limitations

### Missing evidence can become apparent distinctiveness

`compute` builds every count's clusters over every globally observed bot ID, including bots with no runs at that count. An unknown pair cannot merge, so an unmeasured bot becomes a singleton and is added to `distinct_counts`. Registry admission currently interprets cluster membership as count-specific measurement. Unknown is therefore conflated with distinctive, and can also pass into the collection pool. [Cluster construction and distinct counts](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/distance.py#L240-L291), [admission](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/registry.py#L141-L147).

Tiny reproduction: use `TestDistance().suite()` and identical two-repeat stay policies; A has all four probes, B only the two 3-team probes. Actual output: `B.distinct_counts == [5]`, `clusters_per_count['5'] == [['A'], ['B']]`, and their 5-team equivalence score is `None`. B has no 5-team evidence.

### The universal squared-distribution interpretation does not hold

For a **fixed categorical outcome and fixed support**, disagreement correction satisfies

`P(X != Y) - 0.5 P(X != X') - 0.5 P(Y != Y') = 0.5 ||p - q||_2^2`.

The current final weighted distance does not always satisfy this identity. Target comparisons are conditioned on both runs attacking; insufficient target support removes that component and renormalizes weights separately for each run pair. Consequently A/B and A/A can use different outcome populations and different weights. [Conditional target support](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/distance.py#L75-L86), [weight renormalization and subtraction](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/distance.py#L97-L130).

Exact synthetic example, one active unit and unchanged movement: A always waits; B waits with probability 3/4 or attacks one of eight targets with probability 1/32 each. For B's two repeat sequences, enumerate all 32 x 32 outcome pairs once: `x = repeat(arange(32), 32)`, `y = tile(arange(32), 32)`, with outcomes 24..31 denoting attack targets 0..7. Repeat the same probe construction on two distinct synthetic probe IDs. The implemented rates are:

| Quantity | Exact result |
|---|---:|
| A/B combined disagreement | 0.09375 |
| B/B A/A disagreement | 0.28750 |
| Final adjusted combined distance | -0.05000 |
| Adjusted whole-team disagreement | 0.03515625 |
| Default equivalence score | 0.703125 |

The default settings group A and B. Grouping is not by itself proof of a false positive: their valid joint squared distance also lies below the chosen 0.05 joint tolerance. The demonstrated problem is the universal mathematical claim and negative population adjusted distance caused by pair-dependent conditioning. Merely clipping the final value to zero does not restore the identity or ranking.

### Reported interesting tags still reward raw noise

`distinctive_tags`, `tag_distance` and the overall nearest-neighbor summary use raw disagreement, not corrected conditional differences. A/A correction improves equivalence clustering but does not guarantee that the “interesting model” ranking excludes stochastic variation. Whether this substantially affects the real roster requires empirical validation. [Ranking](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/distance.py#L265-L291).

### Coverage and uncertainty are weaker than the wording suggests

Default selection is three distinct maps per count and a single selected seat per match. Family count is prioritized before choosing the seat with most extra-tag turns; this is not quota-based coverage of every critical tactic. Tags with insufficient raw turns disappear from the average. Valid pair-specific opportunities can be much smaller than the raw tag count, because the threshold is checked against suite turns rather than jointly valid unit opportunities. [Selection](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/probes.py#L105-L137), [support and thresholding](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/distance.py#L59-L108).

`mean + 1.645 * SE` over two or three maps is a heuristic, not a generally calibrated 95% guarantee. For three identical observed probe values of zero, `mean_upper` returns mean=SE=upper=0; unseen-map and unobserved-event uncertainty remains. Whole turns within a game are correlated, and multiple overlapping tags are not independent observations. [Bound calculation](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/distance.py#L133-L139).

## Other verified design limits

- Every bot sees a complete shared observation history, but its own outputs do not drive later states. Bots that maintain private plans, enemy-habit estimates or action-dependent memory may enter histories inconsistent with those private plans. This is an applicability limitation, not evidence that all such bots are mismeasured. Pair the probe instrument with live outcome/episode evidence. [Replay contract](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/probes.py#L5-L11), [replay loop](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/fingerprint.py#L99-L120).
- Probe files are checked against frozen SHA-256, but stored fingerprint runs are resumed by filename existence, loaded by bot ID, and later combined by probe ID. There is no enforced source/dependency/suite identity validation at those joins. `assess` copies current inventory source identity rather than verifying each run's recorded identity. Changed code or suites can therefore reuse stale evidence. [Resume/load](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/fingerprint.py#L129-L169), [assessment](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/roster/assess.py#L47-L65).

## Priorities for the implementation plan

1. Make identity validation and measured bot/count/context coverage mandatory. Keep unknown, equivalent and distinct as separate evidence states.
2. Compute fixed-weight channel discrepancies on state-defined support. Encode action together with target-or-none for valid categorical correction; retain raw target IDs and semantic target roles as separately documented channels.
3. Gate equivalence on adequate exposure and acceptable bounds in each predeclared critical context, rather than averaging a rare distinctive behavior away. Use map-cluster resampling and repeated runs, with a held-out confirmation suite after adaptive discovery.
4. Add interpretable action-derived tendencies and opportunity-normalized live outcomes, with explicit denominators, time horizons and censoring. A contact turn alone does not prove successful escape; a dropped flag alone does not prove a tactical handoff.
5. Rank reproducible contribution using corrected semantic channels and held-out outcomes. Keep stochastic augmentation separate from reliable policy-label sources, without treating every genuinely stochastic strategy as useless.

## Bounded checks performed

`python3 -m unittest -v tests.test_roster.TestDistance`: all 8 existing tests passed. The two synthetic examples above and the zero-spread-bound example were executed directly against the reviewed source. The parent independently reproduced the negative-distance and missing-count examples and verified that the detailed death/capture fields are omitted by the current event parser. They are not additions to the repository test suite. No production inference source was modified.

## Modification plan: preserve three kinds of evidence

Keep three linked profiles for each `(source identity, team count)`, rather than replacing the current fingerprint with one large tactical score:

1. **Conditional policy profile:** how requests differ on identical public histories, including joint coordination and semantic tendencies. This supports P-data coverage and duplicate-policy detection.
2. **Live trajectory profile:** realized attacks, flag episodes, survival and outcomes when the bot actually controls its units. This supports finding useful specialists and V-data coverage.
3. **Reliability and strength profile:** official-deadline faults and pairwise utility against declared opponents. High survival, many kills or unusual requests are not themselves strength.

Store direction/target ID comparisons as well as semantic comparisons. Two routes can express the same macro preference while remaining different predictions that P needs to learn. Conversely equal overall attack rates can hide attacking carriers versus attacking harmless nearby units. Never declare two bots interchangeable solely because their final scores or marginal rates match.

### P0: repair identities, support and the distance estimator first

**Evidence identities.** Version the suite, fingerprint, distance and assessment schemas. Hash the suite's actual inputs, seat, prefixes, tags, engine/rules and feature specification. Key each run by source/include-closure identity, build identity, suite/probe identity, repeat, limits and host load configuration. Validate these fields on resume and every aggregation join. Split incompatible evidence instead of overwriting it under a familiar bot ID. Strength records also need source/schedule/engine identities; a current inventory hash cannot certify old records.

**Measured coverage.** Export `coverage[bot][count][context]` containing valid repeats, jointly comparable opportunities, unique maps, unique scenario roots and faults. Cluster only count-measured bots. A singleton with insufficient pair comparisons is **unknown**, not a distinctive specialist. Admit a count only on explicit valid coverage; do not use singleton membership as proof. A context with zero opportunities is missing evidence, never a zero rate or an agreement.

**Fixed comparison instrument.** For a state `s` and active unit, use these tokens:

```text
move_token          = canonical requested move
action_token        = wait / pickup / drop / attack
action_target_token = (action_token, target_id if attack else NONE)
joint_token         = ordered tuple of all active units' request tokens
```

Choose masks and channel weights from the frozen state/context before observing either bot's action. In particular, an action-target token exists for waits too: target comparison must not disappear because one side declined to attack. Use one predeclared weight vector for A/B and A/A on the same support. Conditional target accuracy among attacking units remains a useful **diagnostic**, but does not enter this corrected distance with pair-specific renormalization.

Malformed/missing replies are reliability evidence, not novel tactics. Quarantine failed probes or report a common valid intersection shared by all required A/B/A/A comparisons as explicitly validity-conditioned evidence; do not silently claim equivalence on the full support. Exposure thresholds must count this actual support. Emit raw ignored-request diagnostics separately, preserving both requested intent and feasible canonical behavior.

For each fixed categorical channel on the same state population, with independent repeats:

```text
e_c(A,B) = E[1(X_A != X_B)]
           - 0.5 E[1(X_A != X_A')]
           - 0.5 E[1(X_B != X_B')]
         = 0.5 * sum_z (p_A(z | s) - p_B(z | s))^2, averaged over s.
```

A fixed nonnegative weighted sum of these channels has the same kernel interpretation. Finite estimates can still be negative from sampling error; retain the signed estimate for diagnostics and confidence procedures, and apply nonnegative clipping only for display/ranking if declared. If support/repeats are missing, return unknown; do not silently replace unknown A/A noise with zero for an equivalence certificate. Keep semantic channels separate unless they too have fixed state-defined transforms and masks.

**Context gates and uncertainty.** Separate equivalent, demonstrably different and unknown using preregistered practical margins and confidence procedures. Require upper bounds below margins for equivalence in every adequately measured critical context; require lower bounds above a margin before calling a difference established. A wide interval is neither proof of distinctiveness nor proof of equivalence. Maintain complete linkage on certified pairwise equivalence. Average overlapping tags only as a summary; do not let quiet/flag averages hide a rare drop-and-fire difference. Use map/root-cluster resampling and independently rerun stochastic policies to include repeat noise. Two or three maps cannot support a calibrated broad certificate, and a degenerate bootstrap over three identical maps does not fix that limitation. Calibrate false merges and splits on known controls and untouched maps; account for testing many pairs/contexts through a declared simultaneous or false-discovery procedure.

### P1: make the conditional fingerprint interpretable

Add `roster/behavior.py` for offline semantic features. Reuse exact map geometry and rules; do not change deployed P/V feature contracts as a side effect. Every metric carries a numerator, denominator, context, count, mode and evidence identity.

| Tendency | Opportunity denominator | Observable numerator / quantity |
|---|---|---|
| Pickup preference | Active non-carrier decisions with at least one feasible pickup move | Pickup request and movement toward that opportunity; report actual contested success separately |
| Voluntary drop preference | Decisions beginning with a live carrier | Requested X; separate successful drop events from death-caused drops |
| Return-home preference | Live carrier decisions with reachable home base | Change in shortest-path distance to any own-base cell for the requested move |
| Risk-reducing movement | Threatened, non-immune decisions with at least one feasible lower-risk move | Request reduces a fixed public-state threat field; call this escape-like movement, not inferred intent |
| Attack preference | Active, non-carrier, non-immune decisions with some feasible target/move | Attack request; keep hit rate as a live outcome |
| Target preference | The fixed shared target/opportunity context | Carrier/low-HP/high-threat/near-base target roles, target HP and allied support, alongside exact target IDs |
| Coordinated allocation | Squad decisions with multiple potential shooters | Intended target allocation, excessive requested shots, spreading and carrier protection; settled efficiency is separate |
| Positional control | Active unit decisions within declared reachable regions | Time near home/front/center flag spots, base approaches and contested corridors, plus movement direction |

Use map-relative distances rather than only literal U/D/L/R to summarize roles across maps. For a first risk field, count enemy non-carrier, non-immune shooters that could reach and attack a cell in one legal move, accounting for walls and range. This is a conservative reachability proxy, not a calibrated death probability; actual enemy simultaneous movement and collisions still matter. Store requested and resolved movement separately in live games.

Add context tags that can actually discriminate the requested specialties: threatened own carrier; lethal enemy shot opportunity; drop-and-fire opportunity; threatened carrier near home; contested pickup; dropped flag close to automatic return with a feasible pickup; base respawn blockage; score behind/tied/ahead with remaining time; multi-team crossfire. Current tags remain as backward-compatible diagnostics. All thresholds and rules must be named/versioned; read damage and timers from INIT instead of assuming 34 forever. These flags are neutral: avoid metrics that assume each team owns a flag to defend.

### P1: extract live outcomes already present in the logs

Add `roster/trajectory.py`. First pass through retained matches, exporting the five existing engine statistics to compact strength records without replaying games. Existing `result.stats` values are rounded by the engine; absent keys mean zero, not missing measurement. Records without retained logs or statistics cannot be repaired by inventing zeros. Then add a detailed event parser that preserves death position, dropped flag, all killer IDs and capture flag identity. Keep `fusion.protocol.parse_block`'s current consumer contract compatible, or expose a separate detailed API. Settlement t's events appear in the next TURN block; include `final_block` so turn-400 events are counted exactly once.

The engine credits a kill equally among that settlement's successful attackers, including different teams; it does not credit earlier damage contributors. Match this convention for `kills` and `carrier_kills`, and name other assist conventions separately. Attack events denote **hits**, not every attack request. Movement precedes attack; attack precedes pickup, voluntary drop and capture. A carrier cannot drop and shoot in the same settlement; drop-then-fire starts on later turns. [Engine semantics, AI_Test at c9c115b](https://github.com/Casper015/AI_Test/blob/c9c115b47251f5a124051e0b3ba9a5ee42efdfed/ctf-arena/Player7/workspace/engine.py#L303-L360), [raw recording and event alignment](https://github.com/Casper015/ctf-fusion/blob/1876a8ef747236e0684722af448aa7b17fa6d9fb/training/collect.py#L141-L173).

| Live metric | Required definition |
|---|---|
| Kills, carrier kills, deaths, pickups, captures | Engine totals per game and per count; keep fractional kill credit |
| Exposure / uptime | Alive unit-turns, shoot-capable unit-turns, carrier unit-turns, threatened unit-turns; report immune respawn turns explicitly |
| Death hazard | Deaths per 100 vulnerable alive unit-turns; also condition on threat, HP, carrying and match phase |
| Hit and damage efficiency | Successful hit events / attack requests; useful damage / nominal hit damage; report ignored shots separately |
| Overkill waste | For each victim with HP h entering attack settlement and k hits: max(0, damage*k-h); split shared useful/wasted damage among teams by their hit shares, declaring this accounting convention |
| Carrier episode | Key by flag ID, unit ID and pickup turn; end at capture, voluntary drop, death or right censoring at match end |
| Delivery and loss | Episode capture rate, forced-drop/death rate, carrying duration and travel time; show censored counts, not automatic failures for unfinished episodes |
| Voluntary-drop chain | After a drop, next acquisition by own team/enemy/return; subsequent attack within 1-3 turns; later own capture or loss. Report unit episodes and team possession chains separately |
| Escape outcome | Survival at fixed horizons such as 5 and 10 turns from a declared threatened decision, with HP/carrier/context strata and eligible follow-up length |
| Carrier denial | Credited enemy-carrier kills plus subsequent flag control and enemy delivery within a fixed horizon; non-delivery alone is not proof our action caused denial |
| Map control and coordination | Resolved region occupancy, requested moves blocked by collisions, concentration of successful shots and coverage around the team's carrier |

Rates use the eligible opportunity count, not all 400 turns. Aggregate either opportunity-weighted or map-balanced rates explicitly; keep both where useful. Estimate uncertainty over independent map/fixture groups, not individual turns. A bot hiding at home can have excellent survival and poor score; a frequent dropper can either hand off effectively or lose flags. Preserve utility and reliability alongside all these metrics rather than rewarding a universal weighted sum of kills/survival/drop rates.

Persist a fixed-order feature vector only with its schema and evidence masks. A profile entry should carry `metric`, `mode`, `count`, `context`, `numerator`, `denominator`, `unique_maps`, `estimate`, `interval`, `censored` and `identity`; unavailable dimensions carry a mask, not a fabricated zero. This makes later opponent-style retrieval/cache use possible without losing the distinction between an unobserved habit and a habit observed never to occur. The vector is an offline roster artifact until a separate bot integration adopts it.

### P2: improve probes and selectively add branches

Build discovery and confirmation suites separately. Discovery can start at 6-12 independent maps/count and 2 repeats for cost estimation; increase repeats for clock-sensitive/stochastic bots and maps for uncertain pairs. These are pilot budgets, not universal proof thresholds. Confirmation must use untouched maps and predeclared critical-context coverage; decide its size from interval precision and control-test errors. A minimum such as 20 independent roots containing an event is a coverage floor to investigate, not a general accuracy guarantee.

Replace family-count-first selection with a quota/coverage optimizer: retain a natural-distribution panel, then greedily add maps/seats covering missing tactical opportunities and underrepresented map layouts, phases and seat geometries. Cap the contribution of one map and strongly overlapping events. If several seats share a map, count them as one statistical cluster. Never rank them as independent maps. Preserve late-game segments through full legal prefixes; truncating every stream at turn 200 would remove endgame evidence.

Keep adaptive high-disagreement probes in discovery only. Confirm resulting style claims on the frozen second suite. Add legal engine-generated challenge states only with declared construction rules and reachability validation; tag their artificial distribution separately from ordinary games.

For rare drop/handoff/denial behavior, run short engine continuations and a small set of full live matches. Reuse each continuation's engine state and initial flag RNG state consistently across candidates within a declared comparison. The same seed does not guarantee identical later flag locations once actions change return times or available spots. A process's private memory is not restored by cloning just the engine state. On Windows/black-box bots, either replay the complete common prefix and label the result a common-prefix branch, or obtain an authentic on-policy warm-up and accept that candidates reach different states. Do not claim arbitrary binary hidden-state checkpointing exists. Public histories can be aligned, but action-dependent private plans can still be inconsistent. Compare several opponent continuations and map roots; one branch outcome is not causal tactical mastery.

### P2: rank useful novelty, then verify training contribution

Export **reproducible distinctive tendencies** and **effective specialties** separately. For example, high escape-like request rate is a tendency; better 10-turn threatened-carrier survival at comparable exposure is an outcome specialty. Use corrected, adequately covered per-context differences and held-out uncertainty for novelty rankings. A population-relative raw-distance ratio with a tiny denominator is not a contribution certificate.

Retain the open training pool. A reliable specialist can contribute useful states even if its overall utility is mediocre; random augmentation is separately labeled. Stochastic strategic policies should not be discarded merely because their A/A noise is high. Require enough repeats to distinguish their distribution from noise, and preserve lineage redundancy caps. Equivalence between two stochastic policies means similar conditional distributions, not that either is deterministic.

Define profile queries such as:

```text
count in {8, 10, 15}
AND threatened_carrier opportunity/map coverage sufficient
AND held-out survival improvement has a positive practical lower bound
AND official-deadline reliability acceptable
RETURN utility, escape tendency, delivery rate, CIs, evidence identities
```

Finally test pool contribution with an equal-training-budget add/remove experiment on untouched maps and reserved lineages: P integer NLL/calibration plus conditional target and joint coordination errors; V error under its declared continuation distribution; decision regret and equal-time paired bot outcomes. Diversity is a means to improve generalization, not an end score. This is the test that can establish whether an "interesting" bot improves the trained system.

## File-level execution sequence and acceptance checks

| Stage | Files / deliverables | Required acceptance evidence |
|---|---|---|
| 0. Evidence contract | `common.py`, `fingerprint.py`, `strength.py`, `assess.py`, `registry.py`; versioned identity/coverage fields | Changed source/include/suite/engine/limits refuse incompatible reuse; unmeasured count stays unknown and cannot be admitted |
| 1. Correct distances | `distance.py` and regression tests | Fixed-support categorical/kernel identity on exact distributions; action-target conditional counterexample; joint-only coordination, sparse valid support, missing repeats and noise controls |
| 2. Live backfill | New `trajectory.py`, detailed event API, `strength.play_one/aggregate`, assessment JSON/CSV | Match engine totals including split kills; count final-turn events; distinguish hit/request, forced/voluntary drop; censored episode and absent-data tests |
| 3. Semantic probes | New `behavior.py`, `probes.py`, versioned context quota report | Same opportunity definitions for every bot; useful critical-context map coverage; geometry/seat transforms preserve the meanings they are intended to preserve |
| 4. Confirmation | Adaptive discovery + untouched confirmation manifests, `catalog.py`, `registry.py` | Known duplicates/controlled variants/random mixtures/stateful controls; stable held-out separation with declared unknowns and empirical false-merge/split rates |
| 5. Training utility | Frozen pool variants and fixed-data/training-budget comparison | Prediction and decision improvement, not only a larger number of clusters |

Run the existing roster tests before and after each implemented stage, then the stage-specific regression tests. Include controls that differ only in target choice, voluntary drop, carrier-route risk, shot over-allocation, and private action-dependent memory. For symmetry tests, distinguish an intended invariant semantic summary from an intentionally ID-sensitive raw policy; do not force all real bots to be symmetric. Faulting/ignored-action policies must not earn novelty just from invalid requests.

Recommended first delivery: stages 0-2 plus a small pilot coverage report. They repair the evidence foundation and expose the user's requested drop/survival/attack/kill metrics using existing logs. Confirmed usefulness and more elaborate branch collection can follow. All this roster analysis is offline; it adds no per-turn work to the deployed 50ms bot unless a separate integration explicitly adopts a profile.
