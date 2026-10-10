# Current Integration Status and V Ablation Review

Client review date: 2026-10-09. Source snapshot: [0a0badc](https://github.com/Casper015/ctf-fusion/commit/0a0badc).

This report reviews committed source and JSON/JSONL records. No new games, training, native execution, or timing tests were run. Original host-report dates and measurements are retained as source data; this report's date follows the user's America/Los_Angeles client context.

## Current conclusion

The project now has trained S/M/L artifacts, an optional CUDA trainer, P carrier pilots, and an experimental V ablation. These are useful advances. **Neither the best carrier nor an independent positive V contribution is confirmed.** The main implementation/evidence blockers are still history isolation, matching the deployed P distribution to its evaluation, valid V branch states, and reproducible result identity/timing.

## Status of earlier findings

| Finding | Status at 0a0badc | Next evidence |
|---|---|---|
| Ladder did not forward CUDA arguments | Addressed in 29a6412 | Record actual run backend/device, not just the requested flags |
| No measured GPU training benefit | A limited host benchmark now exists | Full streaming dataset and time-to-quality comparison |
| No L export or live P test | Addressed by e11c45b's artifacts and pilot | Native artifact checks and independent all-count play |
| C08 branch update contaminates P history | Still open in source | Root-history/branch-isolation regression and corrected runs |
| Consumers differ from the complete approved P adapter | Still open | Carrier-specific deployment NLL, alpha, and tactical-event tests |
| V remained untested in play | A small additive-hybrid ablation now exists | Independent marginal V confirmation after consumer repair |
| Candidate hashes and full timing are missing | Still open in committed pilot records | Content-bound logs, first/max/late/raw durations, fault reasons |
| Universal C08-best claim | Not established | Direct paired carrier confirmation under equal CPU time |

The history finding remains visible in [C08 source](https://github.com/Casper015/ctf-fusion/blob/0a0badc/candidate/C08_Player5_Omega_LIT_Fusion.hpp): `planTurn(next,false)` updates the persistent Session, while Runtime save/restore excludes it. The V addition does not repair this path.

## What the new V pilot actually measures

Each carrier has 240 records: control, P-M with V off, and P-M with V on, over 80 fixtures. Counts are 2, 5, 8, and 15, with 20 maps each. Across both carriers there are 480 games but **80 shared fixture identities**. Every fixture matches the first 20 maps/count of the earlier 100-fixture P pilot, including map seed, flag seed, seat, and opponents. This is reused exploratory evidence, not fresh confirmation.

| Carrier | Control utility | P-M only | P-M plus V | Marginal V gain | Published approximate 95% interval |
|---|---:|---:|---:|---:|---:|
| C08 | 0.75357 | 0.75446 | 0.75558 | +0.00112 | [-0.05082, +0.05305] |
| v12 | 0.68772 | 0.72143 | 0.75804 | +0.03661 | [-0.01407, +0.08728] |

Both marginal intervals contain zero. Combined v12 P+V versus control gains +0.07031 with an unadjusted interval [+0.00999, +0.13064]. That tests the combined configuration; it does not prove the additional V term helps beyond P. C08's combined gain is +0.00201 with an interval [-0.06461, +0.06863]. [C08 results](https://github.com/Casper015/ctf-fusion/blob/0a0badc/reports/v_ablation_c08.json), [v12 results](https://github.com/Casper015/ctf-fusion/blob/0a0badc/reports/v_ablation_v12.json)

The positive C08 15-team interval highlighted in the write-up belongs to **P+V minus control**, not V minus P. V alone at that count has a point gain +0.06071 and interval [-0.02119, +0.14262]. No per-count marginal V interval excludes zero. Screening several carriers/counts/comparisons further limits the interpretation of borderline unadjusted significance.

The W/T/L tallies again describe positive, zero, or negative paired utility changes, not actual wins/draws/losses. Rename them accordingly. A zero observed difference or a degenerate empirical interval does not establish zero population effect.

## V consumer semantics need validation

The hooks add scaled V terms to existing heuristic scores: C08 uses coefficients 50, 35, and 25 in different paths, while v12 uses 40 at a rollout endpoint. The experiment therefore evaluates a specific additive hybrid. It does not establish the best probability-to-score conversion or a replacement value evaluator. The coefficients should be frozen on tune data and compared with simple score/time baselines under the same continuation policy.

The trained outcome head predicts a fractional pairwise-score utility under the recorded continuation mixture. It is not a multiclass classifier or an optimal-action value. Raw NLL and Brier magnitudes cannot be compared to rank P versus V. Small observed-state error changes can matter at decision boundaries, but the claimed causal explanation requires branch/regret evidence; this pilot has not supplied it.

The two experiments also answer different questions: the earlier small V Brier reduction compares **M versus S capacity**, while this ablation compares **M V enabled versus disabled**. The ablation cannot show that the small S-to-M error reduction caused its observed neural-V-versus-heuristic result.

Two C08 details matter:

- The duel V term sits inside a refinement condition requiring `g.teams >= 4` or active Expectimax. The frozen C08 enables Expectimax only for multi-team counts, so this V path is inactive at N=2. Identical duel results do not test whether V would help an active duel consumer. Instrument actual V call counts and path coverage.
- If that duel path is activated later, it currently creates a purported successor by moving coordinates from `enemyMoves[0]` while retaining the original turn, HP, flags, and scores. That is not a fully settled successor or a mean over realized scenarios. Repair the branch representation before testing it.

The consumers call `fusion::value()` directly rather than the exact-terminal-aware `nodeValue()`. Use exact pairwise terminal utility at game end, and ensure respawns, flags, attacks, and turn indexing match the shared encoder's state contract. [C08 V hook](https://github.com/Casper015/ctf-fusion/blob/0a0badc/candidate/C08_Player5_Omega_LIT_Fusion.hpp), [v12 V hook](https://github.com/Casper015/ctf-fusion/blob/0a0badc/candidate/Player5_v12_Fusion.hpp)

## Timing and provenance limitations

Zero candidate faults are recorded in all 480 new rows. That is a useful observation. It does not prove the write-up's zero-timeout or less-than-0.2-ms-cost claims: the committed JSONL omits late replies, first/max reply times, raw turn timings, and fault reasons. All 480 `mean_ms` and `turns` entries are null; the summary substitutes zero for missing means. Missing timing must remain unavailable rather than zero.

Reported p99 summaries average game-level p99s. The maximum recorded game p99 is 30.589 ms for C08 and 5.958 ms for v12; neither is the maximum turn latency. To isolate V overhead, record V encode/update/tail time and call counts, plus complete-turn costs and displaced search work. The current C08 and v12 changes should also be profiled under identical concurrency and warm/cold conditions.

Candidate identifiers still stand in for source hashes, executable selection still uses the first matching glob, and saved rows lack candidate/include/binary/weights/engine identities. The script can suppress failed jobs and analyze partial sets; require complete paired fixtures or a predeclared symmetric infrastructure-exclusion rule. [Ablation runner](https://github.com/Casper015/ctf-fusion/blob/0a0badc/tools/screens/run_v_ablation_screen.py)

The new write-up's claim that the question is settled conclusively is therefore too strong. The appropriate status is: **experimental hybrid has promising v12 point results; independent V benefit, carrier superiority, and complete runtime reliability remain unconfirmed.** [Original V write-up](https://github.com/Casper015/ctf-fusion/blob/0a0badc/reports/v_ablation_report.md)

## Next decision

Repair and verify the integration first. On fresh tuning fixtures, keep P fixed and compare V off, a simple score/time value baseline, and the trained V hybrid with declared coefficients. Measure actual V activation and branch regret. Freeze finalists before confirmation over all fourteen counts, with at least 10,000 independent map groups per final integrated version and repeated trained seeds. More matches on these reused 80 maps cannot create an untouched confirmation set.

No code or models were changed by this report.
