# T0004 near/far policy-size evidence

NLL scan completed 2026-10-12 00:11:55 UTC. The 260 maps are absent from every partition in the original dataset index. All three sizes use the same canonical opponent labels, P2 features and frozen folded temperatures. No training, model export or new game took place. Decision-swap diagnostics and the independent receipt/interval/order audit are complete; these are offline results, not strength evidence.

832,000 valid opponent team-turns; no malformed replies or excluded opponent families. Primary:789,836 rows with both reference/opponent teams active and reachable. Inactive-opponent22,129 and inactive-reference20,686 overlap and must not be summed. No active unreachable reference rows.

Negative size-minus-S joint NLL is better. Overall summaries weight all13 team counts equally. Confidence intervals use independent map clusters within each count (normal95% interval, finite-cluster correction). Grouped summaries are exploratory.

| Reference BFS distance | Rows | L-S joint NLL (95% CI) | XXL-S joint NLL (95% CI) |
|---|---:|---|---|
| All primary | 789,836 | -0.079302 [-0.082442, -0.076162] | -0.135705 [-0.140806, -0.130604] |
| 0–3 | 77,928 | -0.073864 [-0.078531, -0.069198] | -0.124528 [-0.131109, -0.117948] |
| 4–6 | 126,625 | -0.077148 [-0.081008, -0.073288] | -0.133379 [-0.139201, -0.127558] |
| 7+ | 585,283 | -0.082551 [-0.086883, -0.078220] | -0.141110 [-0.147957, -0.134263] |

The paired-map near-minus-far difference in size-minus-S NLL is positive for both sizes:

L-S: +0.007360 [0.002392, 0.012327], 260 maps with both near and far rows, 13 equally weighted counts.
XXL-S: +0.014291 [0.007630, 0.020952], 260 maps with both near and far rows, 13 equally weighted counts.

This shows greater joint-NLL improvement in the far context on these records, rather than improvement concentrated near the reference team. It does not support a near-large/far-small production routing rule. It does not establish that model sizes are equivalent, nor that prediction improvement raises game strength. Keep the currently admitted single model size pending separate strength confirmation.

NLL_SUMMARY.json includes per-count estimates, flag/carrier distance groups, geometric contact groups and conditional action/target/top1/complete-order metrics. Target loss uses only observed target slots; missing/out-of-cap target labels never become zero loss. Complete-order NLL is missing when an attack target is outside the modeled slots. Geometric contact ignores occupancy, immunity and future damage.

Native validation: all served counts3–15 for S/L/XXL, exact320 raw policy-head accumulator values per probe, stable NLL agreement within1e-12, and independent Python BFS/line-of-fire geometry checks. FROZEN_INPUTS.json, FROZEN_NATIVE.json, NATIVE_VALIDATION.json and SCORE_RECEIPTS.json bind inputs, source, native binaries, schema and per-match scores.

## Conditional heads

The following loss differences use observed unit labels only. Team counts are equally weighted when their group has labels; exact per-count denominators and missing-label groups are in NLL_SUMMARY.json. Complete-order loss is unavailable when a requested attack target is outside the modeled candidate slots.

| Metric | L-S (95% CI) | XXL-S (95% CI) |
|---|---|---|
| act | -0.003989 [-0.004293, -0.003685] | -0.005623 [-0.006067, -0.005180] |
| target | -0.000149 [-0.000374, 0.000076] | -0.000482 [-0.000720, -0.000244] |
| complete | -0.088886 [-0.092141, -0.085630] | -0.149338 [-0.154518, -0.144158] |

## Chosen-order swaps

26 deterministic held-out maps (two lexicographically first retained IDs for each count3-15), all400 recorded observations per map:10,400 complete decision states per mode. These replays never advance the arena engine or change the recorded future. The root remains the original admitted C08 PXXL_VM model, V off. All modes use128 fixed samples per search call and deadlines removed. Only second-turn policy size changes; history, random keys, depth, scoring and the original wanted-team mask stay fixed.

Original and disabled-routing controls match all10,400 complete replies and root digests, with identical search sample counts. Both actual S/L DLL modules match direct native policy raw outputs for all13 counts. Separate native modules preserve unchanged exported registries and a common compiler/2048-buffer ABI. A fresh prediction View pins the serving model for every team and every policy head; conditional caches belong to that View. Each mode has a separate process, preventing cross-size reuse.

The intervention baseline uses S for all second-turn opponent predictions. The admitted original root is unchanged. Changed-order categories overlap; action and target changes are conditional and must not be summed with movement changes.

| Second-turn intervention vs all-S | Complete reply changes | Movement | Action | Target | Actual L-routed team prediction calls |
|---|---:|---:|---:|---:|---:|
| L at reference BFS7+ | 14 /10,400 | 13 | 6 | 5 | 382723 |
| L at BFS0-3 | 385 /10,400 | 385 | 78 | 113 | 1003800 |
| L for all teams | 504 /10,400 | 503 | 101 | 140 | 2713536 |

Every mode has zero root-digest mismatches, zero truncation and zero invalid complete replies. Per-count changes and branch near/far/middle/inactive-reference call denominators are in INDEPENDENT_AUDIT.json and DECISION_SUMMARY.json.10400 is the complete recorded-decision denominator, including states with no eligible branch prediction. Branch eligible calls are observed, but turn-level branch eligibility was not traced. Distances are evaluated on each predicted branch state, so configurations can reach different conditional eligibility counts as their replies diverge.

Near substitutions change many more decisions than far substitutions on this diagnostic; this is a sensitivity result. Larger models improve prediction NLL in both contexts, with greater paired improvement far away. Neither fact demonstrates a50ms strength benefit or supports silently dropping model capacity far away.

## Decision and limits

Do not build a production near-large/far-small arm from this evidence. Retain the existing single admitted size; no weights, calibration, exporter, production defaults or core paths changed. Any later routing proposal needs its own identity/exporter PR and fresh-seed equal-time strength confirmation after finalist freeze.

This uses one existing retained match seed and260 independent held-out maps from the current opponent pool; it is not an unseen-opponent-lineage study. NLL measures requested opponent orders under canonical labels, not post-collision movement. Potential two-turn contact is geometry only, ignoring occupancy, immunity and future damage. Loss intervals are normal map-cluster intervals with20 maps per count and exploratory conditioning, not a multiple-testing guarantee. Elapsed replay time with deadlines removed is not latency or a capacity comparison.

## Evidence and delivery

FROZEN_INPUTS/FROZEN_NATIVE/FROZEN_SWAPS bind source, model matrix, native binaries, generated diagnostic copies and selected records. SCORE_RECEIPTS preserve every scan output; native and actual-module validation receipts precede full computation. INDEPENDENT_AUDIT independently verifies raw/binary/source hashes, primary and conditional label denominators, paired joint means and cluster intervals, complete protocol-order changes and per-count counters. No completed game, continuation or scoring stage was replayed. Initial width and Windows macro compiler failures are preserved in ignored build logs; they occurred before measurement in their affected stages.

Local delivery checks have passed; CHECKS.md records their actual totals. Independent PR delivery and exact-head CI tracking are the final steps.
