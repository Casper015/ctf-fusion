# Distance-tiered opponent policy and memory-aware CPU inference

This is an implementation plan, not a measured performance result. Keep the current joint-v2 adapter as the default and evaluate the scheduler as part of its artifact. Model capacity and neural mixture alpha are separate parameters.

## Architecture and artifact contracts

Suggested initial hidden widths:

| Capacity | Mover | Squad residual | Conditional action | Target pointer |
|---|---|---|---|---|
| Small | 8 | 32,16 | 8 | 4 |
| Base | 16 | 64,32 | 16 | 8 |
| Large | 32 | 128,64 | 32 | 16 |

These are starting points, not proven optimal sizes. Keep the shared feature-v2 dimensions, 125 movement outputs, four conditional actions, eight target candidates, legal masks and clipped integer activations unchanged. Parameterize every hidden head rather than only `--hidden-squad`. The current default `FUSION_MAX_WIDTH` is 256; newer capacity tooling supports explicitly larger compile-time buffers with export assertions. Respect the configured capacity, integer accumulator bounds and stack footprint. Capacity configurations must have distinct training/export identities.

`tools/capacity_bench.py` now measures S/M/L/XL/XXL and custom tiers on the host. Its S P matches this plan's base, its M P matches large, and this plan's small is below S. Avoid confusing these labels. Hold V widths/call counts fixed when measuring P capacity; the built-in benchmark tiers also change V. Benchmark larger options before selecting a ceiling, but do not infer trained strength or full-turn compliance from synthetic weights and native-only timing.

Initially train small/base/large general P, calibrate them independently, and test representative n02/n05/n15 dedicated tiers. Expand to all dedicated counts only after the useful tiers survive confirmation. The full matrix has 33 P networks plus 11 V networks: 44 instances in 11 dispatch bundles if using three P capacities. Do not silently call that the original 22-artifact matrix. The V network can remain unchanged during the P experiment.

Optional later distillation: train the large teacher on requested joint/action labels, and use a weighted combination of supervised loss and teacher KL for small models. Distill joint distributions before legal masking/temperature contracts diverge. The student still needs its own integer gate; a stronger teacher does not certify the student. Independent networks are the simplest first implementation. Shared/nested residual tiers require a separate parity and strength experiment.

## Spawn-distance prior

At the first public turn, save the initial spawn cells of all squads. If the protocol does not expose usable initial cells, use the authoritative base/spawn anchor from INIT. Never infer hidden random flag locations. Define enemy-team distance as the minimum wall-aware BFS distance from its initial squad cells to ours; document the exact convention. Unreachable distances sort last.

Sort opponent teams by this distance. As an initial policy, assign the nearest quarter large, the next quarter base, and the remainder small; the only opponent in a duel is large. Break ties deterministically by team id. Also log normalized absolute distance so the rank rule can later be compared with an absolute-distance rule. Birth proximity is a prior about likely interaction, not a proof that a distant opponent is irrelevant.

Static tier order must be monotonic in spawn distance before dynamic overrides. A test should enumerate ties, unreachable regions, two teams and 15 teams and verify this property.

## Dynamic overrides and budget admission

Promote a team when its current units can contact ours within the search horizon, it carries a relevant flag, threatens an imminent delivery/interception, or shares a contested target/flag. Use wall-aware travel distances and exact attack geometry. A team born far away can now be our nearest threat.

Use deterministic priority: immediate scoring/death interaction; then current contact; then spawn proximity. Add modest hysteresis to stop capacity from alternating on nearly identical turns, but allow urgent threats to promote immediately. Calibrate and evaluate the resulting scheduler, not only each tier on a favorable subset.

This is a two-level rule: spawn distance establishes the default capacity; tactical decision risk and the remaining global budget decide overrides. Uncertain actions are not automatically important actions. High entropy may be irreducible randomness, and a confidently wrong model can have low entropy. Prefer a cheap geometric sensitivity proxy or an offline-trained estimate of how much a tier upgrade reduces decision regret. Freeze and evaluate any learned router separately.

Do not require all opponents to run small before they can run large. With independent networks, an upgrade costs small plus large, not merely the width difference. Direct large prediction is sensible for an urgent near opponent; a small-first cascade is an optional experiment for low-priority opponents. Shared/slimmable weights do not automatically make upgrades free. See `ADAPTIVE_COMPUTE_RESEARCH.md` for measured examples and their limits.

For a measured target-host upper-cost estimate \(c_{t,k}\), admit team t at tier k only if the prediction budget remains sufficient. Begin with a provisional total P budget of 5ms and a whole-compute target around 35ms, with output initiated by roughly 40ms. These are margins to verify on the host, not permission to violate 50ms. Include feature construction, legal masks, conditional heads and dispatch in P cost. Estimate all-opponent worst cases, not a single network call.

Capacity fallback order: requested tier -> a lower tier with a valid gate -> unchanged legacy predictor. Do not use an unapproved small model to recover from an approved large model's deadline. Keep a baseline legal action available before expensive work. Interrupt only at bounded search/inference batches and measure their largest noninterruptible cost. Output a complete flushed reply before the official cutoff.

Prediction should run once per opponent team at the root unless a separately measured design predicts branch states too. In branch mode, changed global/unit/target/legal features invalidate the root cache. Freezing predictions deeper into search is an explicit approximation and must be ablated.

## What extra memory should buy

1. **Static geometry.** Store packed passability, next cells, home/spawn distances, attack reachability/LOS and optionally all-pairs BFS distances. At a 49x49 board, a cell-indexed uint16 distance table is 2401²×2 = 11,529,602 bytes, about 11.0 MiB; unreachable values use a sentinel. Compute it within the startup deadline or retain an on-demand bounded fallback. Mobile/WASM may use only relevant-source BFS tables rather than all pairs.
2. **One shared turn representation.** Build occupancy, respawn/immunity, available flags, nearby threats and target candidates once. Share stable primitives across opponent encoders and requested tier alternatives. Do not reconstruct the entire board for each head or sample.
3. **Exact per-feature caches.** For static/categorical first-layer contributions, cache only measured profitable pieces. For repeated features, reuse first-layer results or compute deltas, preserving the exact integer summation/overflow contract. Dense features changing every turn may have few useful deltas; benchmark before extending the V accumulator pattern to P.
4. **Exact inference memoization.** Key entries by full encoded input, feature version, model/weight identity, head and conditioning move/target context. A hash lookup must confirm the full key, not assume no collisions. Keep a fixed bounded table and preallocated arenas. Add counters for hit rate, avoided MACs, lookup cost and evictions. Do not cache by team id or turn number alone.
5. **Sampler setup.** Cache cumulative distributions for each immutable root team prediction. Preserve one mixture-component draw per team under joint-v2. Conditional action/target results may be reused only for identical encoded inputs and sampled move. Consider alias tables only after comparing their setup cost against 125-entry inverse CDF draws.
6. **Packed model and scratch layout.** Load/validate static weights during startup, align contiguous integer matrices, warm the actually used paths, and preallocate feature/output buffers for all 15 teams. Check SIMD implementation against the scalar integer reference. A large multiplication lookup table can increase memory bandwidth and be slower than vectorized arithmetic; treat it as an experiment, not the default.

Separate native-large-memory and browser-compact profiles. Unlimited arena RAM does not mean unlimited cache locality or mobile device memory. In the browser, compile the same CPU kernel to WASM and run it in a Worker; share immutable map/weight data where the environment supports it. Match the total device budget when multiple games run concurrently.

## Runtime integration seam

Extend model dispatch to return a per-count tier set. Encode an opponent once, choose the admitted tier, run integer heads, and build a `TeamPrediction` carrying tier id, full artifact id, adapter id, calibrated alpha and joint/conditional probabilities. Search must sample through the existing adapter contract. Analytical marginals and sampled joint actions must refer to this same prediction.

Record tier exposure by count, distance bucket and tactical event. Fit alpha/temperature on independent tune data for each tier and evaluate the frozen scheduler on confirmation data; if a tier is only enabled in a distance/event bucket, its gate must cover that exposure. A schedule-only change changes the deployed prediction artifact and requires a new report. Do not increase alpha merely because an opponent is nearer.

## Acceptance and ablations

- Full versus cached logits must match for all heads, capacities, movement conditions and replayed branch inputs.
- Deliberately collide cache hashes and change score/respawn/carry/target/legal features to verify invalidation.
- Large-near/small-far prior, dynamic overrides, tier downgrade, no-valid-tier fallback and alpha-zero legacy equivalence need focused tests.
- Replay simulator coverage, feature-v2 parity, bounded widths and int32 bounds must stay green.
- Compare fixed-small, fixed-large, distance-only, distance+dynamic, and cached distance+dynamic with equal search work; then compare useful search work under the real complete-turn deadline.
- Host reports must include startup, p50/p95/p99/p99.9/max complete-turn time, late/invalid/crashed turns, tier-call counts, cache hit rate and search nodes/scenarios completed. MAC counts alone do not certify 50ms.
- Ship only tiers/counts with an integer/artifact-valid offline gate and independently confirmed play. An unmeasured deadline or inconclusive benefit remains explicitly unverified.
