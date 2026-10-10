# Choosing a Heuristic Carrier for P: C08 Versus Player5 v12

Date: 2026-10-09. Evidence snapshot: [e11c45b](https://github.com/Casper015/ctf-fusion/commit/e11c45b).

**Bundle status:** This preserves the e11c45b P-carrier audit. The subsequent 0a0badc V ablation does not repair the documented history or adapter issues. Read the [bundle index](review_bundle_20261009/README.md) and [current status/V review](review_bundle_20261009/CURRENT_STATUS_AND_V_ABLATION_REVIEW.md) for the newer evidence and proposed execution order.

Scope: read-only source and result analysis. Existing JSON/JSONL records were inspected; no bot code, configuration, or model was changed. No new match, native execution, or latency benchmark was run.

## Recommendation

**C08 is a useful reference carrier, but the available evidence does not establish it as the best carrier.** Its M integration has the highest aggregate utility point estimate in the new small screen. Player5 v12 gains more from the current P hook and has substantially more CPU headroom. Retain both for confirmation rather than treating the original README choice as a measured optimum.

Before a larger C08 confirmation run, repair and test its neural history isolation. Before calling the v12 experiment a test of the approved joint-v2 policy, implement and evaluate the distribution its consumer actually uses. These are findings and proposed next steps, not changes performed by this review.

There are two separate objectives:

$$
J_c=U(c+P),\qquad \Delta_c=U(c+P)-U(c).
$$

J selects the strongest complete candidate under the CPU deadline. Delta measures how much the carrier benefited from P. A larger Delta does not imply a larger J.

## The latest integrated screen

The new [C08 JSONL](https://github.com/Casper015/ctf-fusion/blob/e11c45b/reports/carrier_c08_screen_with_l.jsonl) and [v12 JSONL](https://github.com/Casper015/ctf-fusion/blob/e11c45b/reports/carrier_v12_screen_with_l.jsonl) each contain 400 match records: 100 fixtures across 2, 5, 8, and 15 teams, with 25 maps per count and four variants per map. The two carriers share the same fixture keys, maps, flag seeds, seats, and opponents. There are 100 shared experimental fixtures, not 800 independent maps. Utility measures the final score comparison against each opponent, including ties; it is not first-place win rate or a Bradley-Terry rating.

| Carrier | Control utility | P-M utility | M gain | P-L utility | L gain |
|---|---:|---:|---:|---:|---:|
| C08 Omega_LIT | 0.7725 | 0.7809 | +0.0084 | 0.7432 | -0.0293 |
| Player5 v12 | 0.6996 | 0.7477 | +0.0480 | 0.7479 | +0.0482 |

The committed approximate 95% intervals for M gain are [-0.0454, +0.0622] for C08 and [-0.0127, +0.1088] for v12. Both include zero. v12 L gain has a borderline unadjusted interval [+0.0012, +0.0953], which should remain exploratory after screening several sizes, counts, and carriers. [C08 summary](https://github.com/Casper015/ctf-fusion/blob/e11c45b/reports/carrier_c08_screen_with_l.json), [v12 summary](https://github.com/Casper015/ctf-fusion/blob/e11c45b/reports/carrier_v12_screen_with_l.json)

Direct paired arithmetic on those same fixtures gives:

| Comparison | Equal-count utility difference | Approximate count-stratified 95% interval |
|---|---:|---:|
| C08 P-M minus v12 P-M | +0.0332 | [-0.0223, +0.0888] |
| C08 P-M minus v12 P-L | +0.0330 | [-0.0191, +0.0852] |
| v12 P-L minus v12 P-M | +0.0002 | [-0.0535, +0.0539] |

These are normal approximations from per-count paired sample variances, not adjusted final confirmation intervals. They do not establish a superior carrier or L over M.

| Teams | C08 control | C08 P-M | v12 control | v12 P-M | v12 P-L |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.8600 | 0.8600 | 0.6200 | 0.7800 | 0.7200 |
| 5 | 0.8000 | 0.7350 | 0.7400 | 0.7150 | 0.8200 |
| 8 | 0.7657 | 0.7857 | 0.7143 | 0.7629 | 0.7743 |
| 15 | 0.6643 | 0.7429 | 0.7243 | 0.7329 | 0.6771 |

C08 M's aggregate point advantage is not just a uniformly better P response: it includes a stronger control, especially in duels, while its five-team point result regresses. Do not tune a count router on these inspected maps and then reuse them as final confirmation.

At 15 teams, the mean of game-level p99 times is 32.50 ms for C08 M and 7.85 ms for v12 M. v12 L is 7.71 ms. These summaries indicate substantial v12 headroom, but are not pooled turn p99 or maximum latency. The C08 records contain nine candidate faults in total: S three, M two, L four. v12 records contain zero faults. The cause of C08 faults cannot be diagnosed from the committed timing fields alone.

## Carrier source differences that matter

### C08: richer sampling consumer, but a history-isolation defect

[C08's fusion header](https://github.com/Casper015/ctf-fusion/blob/e11c45b/candidate/C08_Player5_Omega_LIT_Fusion.hpp) retains its search and heuristic evaluation. Its new hook provides joint opponent movement draws in the multi-team sampler and replaces movement marginals in analytic calculations. The duel predictor is also hooked separately. Conditional attack probabilities enter part of the multi-team forecast. This is closer to using the learned squad correlations than replacing only unit marginals.

However, the multi-team implementation mutates the persistent Session inside simulated branches:

1. `planTurn()` calls `fSession.update(g)` unconditionally, including its `refineAllowed=false` branch path.
2. `refineChecked()` calls `planTurn(next, false)` for counterfactual successors.
3. `saveRuntime()` / `restoreRuntime()` preserve planner fields but omit `fSession`.
4. Session update copies its current state into `prevState`. [makeView()](https://github.com/Casper015/ctf-fusion/blob/e11c45b/inference/state.hpp) reads previous unit positions without verifying adjacent turn numbers.

Consequently, another candidate branch or the next real observation can receive a simulated branch as its previous observation. It can alter movement-history features and cached policy outputs. The defect follows directly from the call and restore paths; its match-strength impact has not been isolated experimentally. A root-only immutable observation history and separate branch-local policy contexts are needed. Test that branch enumeration and branch count leave the root Session unchanged and that the next real state's encoded features match a clean replay.

The C08 hook is also a partial P consumer: it does not sample the complete conditional pickup/drop/attack-target policy. On ground flags, the subsequent heuristic assignment overwrites the neural `shootChance`. Do not describe it as complete flagshoot/drop/target integration.

### v12: simpler, faster consumer, but movement marginals only

[v12's fusion header](https://github.com/Casper015/ctf-fusion/blob/e11c45b/candidate/Player5_v12_Fusion.hpp) calls the Session at the real root and replaces each enemy's movement marginal with `mixedMarginal()`. Its existing tactical analysis and two scripted continuation models remain. It does not call `drawTeam`, `drawAct`, or attack-target prediction; the root rollout uses a modal first-step forecast in one script. It therefore does not use the full learned joint policy as a joint opponent sampler.

The current joint-v2 gate evaluates a mixture of joint distributions. A consumer that uses products of mixed marginals implements a different distribution. Even where linear expectations of individual unit outcomes are valid, joint tactical events and nonlinear rollout scores require a consumer-specific evaluation. Gate and calibrate the actual marginal consumer or implement the approved joint sampling path. The reused alpha must also be checked against each carrier's original heuristic forecast, particularly C08's separate duel predictor.

Thus the new screen compares complete experimental systems with different P consumers. It is useful deployment evidence, but does not isolate intrinsic carrier architecture or the full P model's benefit.

## Corrections to the new integration write-up

The [integration write-up](https://github.com/Casper015/ctf-fusion/blob/e11c45b/reports/CARRIER_INTEGRATION_REPORT.md) contains encouraging point results but overstates several conclusions:

- Its executive statement that both v12 M and L significantly beat control is inconsistent with M's aggregate interval, which crosses zero. No published paired M/L-versus-S test supports the accompanying claim.
- Its W/T/L columns classify whether candidate utility increased relative to control. They are not actual match wins, draws, and losses. v12 L's duel scores give 17 wins, two draws, and six losses, despite the four-positive/21-unchanged/zero-negative paired-delta tally.
- With only four nonzero positive L duel differences, a simple exact two-sided sign test yields p=0.125. This illustrates why the small-sample normal-interval claim is fragile; it does not prove no improvement.
- The [smoke scripts](https://github.com/Casper015/ctf-fusion/blob/e11c45b/tools/screens/test_carrier_smoke.py) compare final scores on one two-team and one five-team fixture. They do not prove bit-identical commands across all seeds. Final-score equality is weaker than a stream null test.
- The JSONL files omit late replies, first/max reply timing, and candidate/model/build hashes. A zero-timeout claim and full provenance cannot be verified from them. Candidate identifiers passed as `source_sha256` in the screen script are names, not content hashes.
- Complete-path zero-allocation is not established by the MLP kernel's allocation behavior. The hooks create vector-backed State objects, and C08's new sampler constructs a vector per draw. Instrument the integrated path before making that claim.
- The reproduction command uses `--carrier` and `--maps`, which [run_carrier_screen.py](https://github.com/Casper015/ctf-fusion/blob/e11c45b/tools/screens/run_carrier_screen.py) does not accept. The implemented arguments are `--candidates` and `--maps-per-count`.

These issues do not erase the recorded utilities; they limit what can be concluded and reproduced.

## How to choose the next foundation

Retain C08 P-M as the highest-scoring current point candidate and v12 P-M as the faster candidate with a larger observed P increment. v12 P-L is practically tied with M in this sample; larger size alone is not a selection argument. Neither is a confirmed winner.

Before expensive confirmation:

1. Resolve C08 branch-history isolation and verify command-level null behavior under a controlled deterministic workload.
2. Define equal P scope in both carriers: first a movement-only comparison, then joint movement and conditional action/target variants where supported. Evaluate and fit each actual adapter/legacy baseline on tune data.
3. Record content hashes of source, dependencies, binaries, weights, adapter, engine, and schedule. Preserve raw first/post-startup durations, late replies, fault reasons, and match logs. Add A/A repetitions to measure deadline-sensitive search variability.
4. Compare C08 and v12 under the same representative CPU load and complete 50-ms budget, using fresh paired fixtures. Keep the existing faults in outcome metrics.
5. Use all counts 2–15 for the frozen final bot, including stronger duel opponents and unseen lineages. After screening, confirm finalists with the previously specified 10,000 independent map groups per final version and repeated trained seeds.

The older light screen favored v12's raw utility point estimate (0.7703 versus C08 0.7109), while the new integrated four-count screen favors C08 M's absolute point estimate. The different maps, count weights, opponents, and sample sizes explain why those numbers cannot be pooled into one ranking. [Older selection evidence](https://github.com/Casper015/ctf-fusion/blob/e11c45b/roster/report/training_selection_20261009/README.md)

Choose the final carrier by confirmed J under the runtime budget, while using Delta and the component tests to diagnose improvements. The present evidence justifies two finalists and integration repairs, not a universal C08-best claim.
