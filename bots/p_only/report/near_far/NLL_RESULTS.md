# T0004 native held-out NLL results

NLL scan completed 2026-10-12 00:11:55 UTC. The 260 maps are absent from every partition in the original dataset index. All three sizes use the same canonical opponent labels, P2 features and frozen folded temperatures. No training, model export or new game took place. Decision-swap diagnostics and the independent audit have completed; these are offline prediction-loss results, not strength evidence.

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
