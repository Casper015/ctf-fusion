# T0019: search changes do not show an equal-time utility improvement

All seven formal stages completed successfully on the Windows host at 2026-10-11 21:36:29 UTC. The 2,240-game paired screen does not show an improvement for C08 B, C, D or v12 B over their incumbent A. Every overall and band interval, both A/A comparisons and the C08 interaction includes zero. Keep the production defaults unchanged. No positive overall candidate is selected for confirmation from this study; future tuning candidates still require an unseen T0006 confirmation seed.

The measurement source is `05f4aca0acd2f4d31b101b9b71b4afa95c888330`, with committed XXL p2 policies, V off, fixed depth and evaluation coefficients. The fresh tuning seed is **20261230**; the task's originally proposed 20261220 was already consumed by T0016 and was not reused. The screen used six workers, official 2,000/50 ms limits, 20 paired fixtures per count 2–15, six variants and two A/A copies. Neither models nor production defaults changed.

## Equal-time deployment screen

Utility is the share of opposing teams beaten, with ties worth half. These differences are utility differences, not win-rate percentage points. Each fixture pairs map and flag seeds, seat and opponents across all eight arms. Team counts receive equal weight; intervals use within-count map-group variation. All completed games, including the two games with candidate timeouts, remain in the denominator.

| Contrast | Mean utility difference | 95% interval | Conclusion |
|---|---:|---|---|
| C08 B − A: action portfolio | −0.01719 | [−0.04177, +0.00739] | not shown |
| C08 C − A: per-branch ply-2 predictions | +0.00410 | [−0.02055, +0.02874] | not shown |
| C08 D − A: both changes | −0.01033 | [−0.03562, +0.01496] | not shown |
| v12 B − A: action portfolio | −0.00657 | [−0.03661, +0.02347] | not shown |
| C08 interaction D − C − B + A | +0.00277 | [−0.02841, +0.03394] | not shown |

See `screen.json` and `SCREEN.md` for every band, pairwise comparison, A/A interval, latency summary and search counter. The independent audit recomputed the overall, band and per-count comparisons directly from all raw score records and checked them against the generated JSON. “Not shown” does not establish equivalence or rule out a small effect.

Four candidate faults occurred, all >50 ms timeouts: C08 B at n05/m00015 turns255/257 and C08 A at n05/m00016 turns63/64. Candidate dead games and late replies were zero. All seats together had13 timeout events: four candidate and nine opponent events. The C08 A/A n05/m00016 game contained seven opponent timeouts at turns140/142 across several seats. In the two candidate-fault games Astra and Opus respectively also timed out at nearby turns. `INDEPENDENT_AUDIT.json` includes every seat's fault events. Desktop game activity was observed during the run; a scheduling contribution is possible but was not established. No user application, resource limit or paging setting was changed, and no faulted game was excluded or replayed. Reported p99/max values summarize successful replies and cannot certify all replies met the deadline.

## Fixed-work diagnostics

All seven configurations processed 60 recorded games and 24,000 turns each, with zero root-digest mismatch and zero truncation. Against their respective A, changed replies were 8.32% for C08 B, 7.75% for C08 C, 14.61% for C08 D and 4.40% for v12 B. The diagnostic-only C08 replay changed no emitted A reply; its draw differences were 10.89% and first/final top-plan changes were 10.2%/6.0%.

C08 A/B performed about 2.90 million ply-2 predictions each, versus 28.03 million for C and 30.51 million for D. At equal time the generated samples per search were A275.5/B271.1/C118.1/D117.0: extra per-branch prediction work displaces search samples. Fixed-work decision changes and cache hits establish that the switches are exercised; they do not establish stronger play. Replay wall times are not reply latency or equal-cost capacity measurements. `diag_fixed_work.json` retains all prediction, state-build, view-hit, memo-hit and alternatives counters. Settings such as mode/cache/fixed in diagnostic totals must not be interpreted as work counts.

The raw screen counter audit found one PONLY line per game. C08 SEARCH/PLY2 lines occur in260 games per arm because its two-team path uses a separate planner; the20 two-team games are included in PONLY per-turn denominators. Numeric counters were summed separately from settings; primary per-turn, per-search and timing aggregates independently match all eight arms. Missing timeout replies remain missing.

## Continuation regret

Each arm study selected200 roots by the frozen hash ordering, with one root per match, at most eight distinct complete orders and two repeats per candidate. Only legal-candidate shortage caused drops. Root coverage includes those dropped roots; generated regret denominators contain only valid retained roots. There were no invalid comparison-reply exclusions.

| Quantity | v12 B against A | C08 D against A |
|---|---:|---:|
| Selected / dropped / valid roots | 200 / 5 / 195 | 200 / 3 / 197 |
| Identical / different complete orders | 192 / 3 | 150 / 47 |
| Continuations | 3,002 | 3,014 |
| Resolved / unresolved candidate pairs | 1,850 / 3,250 | 1,671 / 3,421 |
| Outcome difference, all valid roots | +0.0009324 | +0.0022233 |
| 95% map-cluster interval | [−0.0008951, +0.0027599] | [−0.0047953, +0.0092418] |
| Difference where orders differ | +0.0606061 | +0.0093189 |
| Conditional 95% interval | [−0.0581818, +0.1793939] | [−0.0202485, +0.0388862] |
| Mean incumbent / comparison rank | 2.6974 / 2.6821 | 2.7107 / 2.5685 |

Both overall and conditional continuation differences are **not shown**. Only three map clusters support the v12 conditional estimate. C08's n13 exploratory subgroup has a positive interval, but the overall C08 interval and its deployment screen do not; this isolated subgroup is not a confirmation claim and does not justify a default change.

Prefix and live continuation faults were zero in both studies. Replay reply mismatches were7,525 across228 v12 continuations (maximum68) and21,945 across2,473 C08 continuations (maximum99). These count bot replies differing during a replayed prefix, whose recorded orders are forced; they are not reproduced TURN-block mismatches. The audit verified recorded root blocks, candidate/repeat coverage, complete-order comparison identity, outcome utility, source counts, pair resolution and regret intervals from every retained root.

Regret compares each chosen order with the best sampled candidate's mean continuation outcome. It is affected by finite candidate coverage, continuation noise, forced orders and bot memory after a counterfactual choice. It is neither optimality nor deployment strength. Large unresolved-pair counts reinforce that limitation. Full quantities and per-count intervals are in `regret_v12.json`, `regret_c08.json` and the branch summaries.

## Provenance and delivery

`FROZEN_INPUTS.json`, `FROZEN_SCREEN.json`, `MANIFEST_METADATA_RECEIPT.json` and `PIPELINE_RESULT.json` bind source, engine, roster, models,60 diagnostic inputs, six actual binaries, bundles, runner identities and schedule. Only build-manifest updated/compile_seconds metadata is ignored. Bundles use the build contract's logical-LF hash; `INDEPENDENT_AUDIT.json` also retains raw transport hashes. Frozen source and completed stages were never modified or restarted.

The first post-run audit attempt used an incorrect local compiler path and failed before scanning raw games; it was corrected to the original compiler. A supplemental audit initially expected SEARCH lines in two-team games and a settings dictionary for every arm; source and raw counter inventory resolved both assumptions. These were audit setup failures, not measurement failures. Original receipts and measurements remain intact. The earlier raw-CRLF-versus-logical-LF monitor assertion and PID-reuse CPU-counter issue are retained in the reproduction/status records; neither triggered game reruns.

`audit_raw.py` and `audit_aggregates.py` produce the independent receipt. Invoke from this frozen checkout, after reconstructing the orchestration helper in `REPRODUCE.md`, with the same source environment and exclusive CPU lease. `CHECKS.md` records local validation. All committed changes stay inside the T0019 report lane and are delivered through an independent PR; the shared main checkout and task board are not changed.
