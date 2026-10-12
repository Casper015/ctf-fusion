# T0013 bounded screen preparation

User steering: T0013 uses approximately 2,000 games; T0032 gets an additional separate budget. Wait for T0032 to finish and freeze its artifacts before coordinating the host screens. This is preparation only: no compilation, games, cost measurement, new schedule, training, export or seed consumption.

## Frozen budget proposal

Keep the existing screen_p_features.json: 14 counts (2–15), 20 map fixtures per count, 280 fixtures. Six arms (C08 and v12, each original XXL reference, retrained P2 XXL and P3 XXL) plus one identical v12 P3 A/A copy give 1,960 games. Every arm shares fixture map, opponents, candidate seat and official 2,000/50 ms limits. V is off. Six host workers remain the planned concurrency under a single exclusive CPU lease.

P2/P3 exported sets are byte-matched against the completed T0011 checkout in PREPARATION.json. The four prior carrier builds are historical correctness evidence; do not treat their binaries or timing as current host measurements. The source, runner, model, transitive include and actual binaries must be frozen again at launch after integration. Shared main checkout is unchanged.

## Decision rule and stopping

Use one fixed endpoint at 1,960 games, with no automatic expansion to 10,000+ games or extra seeds. Primary comparisons are P3 minus P2 separately for C08 and v12, using paired utility, equal count weighting and map-clustered 95% intervals. Positive intervals excluding zero support a tuning-screen improvement for that carrier; negative intervals excluding zero show degradation; intervals containing zero are not shown. Report per-count and bands 2, 3–5, 6–10, 11–15 and the carrier interaction as secondary context. A single favorable band does not substitute for the overall carrier comparison.

The A/A copy measures scheduling noise. Report its interval and any drift beside the P3/P2 effect; investigate substantial drift before interpreting a small advantage. Retain every timeout/dead/late fault and original outcome in its denominator; no favorable reruns or dropping failed seats. Report successful reply p99/max separately from missing timeout replies, plus C08 samples/calls and other available search counters. If inconclusive at the cap, report not shown and stop; confirmation belongs to a separately frozen T0006 decision. This plan does not stop at a favorable intermediate ordinary 95% interval.

## Cost and coordinated T0032 follow-up

Run the existing value_cost.py on actual P2/P3 sets with calls=3 and copies=1,6,15 under the CPU lease. Keep shared policy view cost, per-team encoding and joint inference distinct; full 50 ms reply/search effects come from games. Cost probes use retained recorded fixtures and are not extra games.

T0032 remains another agent's offline training/validation task. Wait for its completion receipts and immutable checkpoint/config hashes. Its 8XL teacher is offline-only. Choose deployable XXL student/control/QAT candidates from validation evidence before any host game and bind the exact arithmetic, temperatures and parity. Its current task does not provide inference exports or carrier builds; these are a separate integration prerequisite, not something already delivered. Freeze its additional game budget, paired controls and a fresh seed separately; do not mix its arms into T0013's 1,960-game denominator or reuse T0013's seed for a new comparison.

## Launch prerequisites

Refresh the remote owner/lane/dependency and seed ledger. T0011 is done and PR21 merged. Seed 20261210 is assigned to this screen, but preparation does not establish non-consumption across all prior runs; check actual schedules and original train/val/test map intersections before launching. Reuse valid completed fixtures only by verified receipts and never repeat a successful measurement.

When T0032 is frozen, use the independent T0013 worktree, confirm scope and exact frozen source, then build/dry-run and verify 1,960 unique jobs under the CPU lease, without starting games. Preserve runner/source/helper/model/binary/plan receipts and the passive 15-second observer. No concurrent formal GPU job; user desktop GPU applications are untouched. Run the bounded screen only once all prerequisites are met. Do not pull or rebuild during the run.
