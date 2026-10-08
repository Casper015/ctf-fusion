# Model and integrated-bot evaluation contract

This is an English version of the previous evaluation template, synchronized with ctf-fusion `93fa7e604ca9a05e48b8e04d787dcb387401fba8` and its preceding contract fixes. The standalone calculator summarizes evidence; it does not replace the repository's artifact-bound export gate or official rating implementation.

## Data acquisition and identity

The existing `profiles/pool.json` contains official opponents, historical players, related experimental versions, Baseline and Noise. Two-team collection uses a selected subset; other counts use the default pool. Every profile starts at 1,500 matches, with a cumulative phase-two target of 7,500. With probability 0.5, scheduling first includes one of Omega_LIT/Player2_v005; this is not a 50% seat share. The updated plan replaces the fixed-15 training cap with an open, hash-locked training registry and a separate frozen evaluation roster; those are implementation tasks, not already produced data.

Sample count and matchup regime, then measured behavior profiles and versions, using lineage to control redundancy and holdouts. A source family is not automatically one playing style. Record per-count/context weights and actual inclusion probabilities; equal matches do not imply equal acting-team row mass across counts. P learns opponent behavior, including useful weaker styles; it is not our action policy. V estimates outcomes under the recorded continuing-policy/opponent distribution. Mixed behavior-policy labels do not automatically estimate the value of states under our future search policy. Label exploratory/noise continuations separately.

Separate train, tune, offline-confirm, integration-confirm, strict OOD and stress data. Tune chooses architecture, alpha, temperature, thresholds and tier scheduler. Freeze them before confirmation. Group repeated maps, seats, flag-seed repetitions and related fixtures consistently. Whole held-out lineages and their close code descendants must not contribute training fixtures. Preserve the source's newly implemented strict holdout and feature-v2 contracts.

An open training pool evolves between versioned dataset revisions; it is frozen within a run. A frozen evaluation roster is independent of training admission and may contain reserved unseen opponents. Keep sources, fixtures, weights and repeated-policy rules identical across a comparison. Test whether open-pool or behavior-weighted collection helps at equal training budgets, rather than giving a larger pool more samples and attributing the gain only to diversity. Existing fixed-size selection tools can create an optional reference panel, not enforce the training membership count.

Record source/dependency/build hashes, model and adapter identities, seeds, raw requested actions, public state, resolved events, label exclusions, sampling probabilities and fault status. Current enemy requests and future random flag returns are offline labels, never runtime inputs.

## Four distinct capability measures

Do not add log-loss, milliseconds, rating and accuracy into an arbitrary composite score. Keep prediction, value, decisions, real play and runtime status separate.

### P: probability quality of the actual deployment

For observed requested action \(a_i\), sampling-restoration weight \(w_i\), and actual integer-deployment distribution \(q_i\):

$$
L_P=-\frac{\sum_i w_i\log q_i(a_i)}{\sum_i w_i},\qquad \Delta_P=L_H-L_P.
$$

Positive gain favors the model. Compare with the current legacy forecast, calibrated inertia and uniform legal actions. Beating inertia does not replace beating the legacy forecast. Joint-v2 must score the joint mixture; marginal-v1 must score the product of mixed marginals. Distance-tier dispatch and fallback rules are part of the deployment distribution, with their own immutable identity.

Report per-count and per-lineage movement NLL, conditional action NLL, attack-target NLL and target-candidate coverage. Split quiet/contact/flag/respawn/endgame cases and threat/health/distance buckets. Top-1 accuracy and calibration bins are diagnostics, not substitutes for probability scoring. Label or support omissions must be visible.

Use the same simulator to estimate tactical-event probabilities: collision, lethal focus fire, flagshoot, manual drop, interception and possession change. Score actual event outcomes with binary Brier and reliability bins. A lower average movement NLL may still hide worse lethal-event predictions.

### V: the intended outcome under the declared continuing policy

For team i among N teams, final score \(S_i\) defines

$$
y_i=\frac1{N-1}\sum_{j\ne i}\left[\mathbf1(S_i>S_j)+\tfrac12\mathbf1(S_i=S_j)\right].
$$

This is a pairwise outcome score, not the probability of sole first place. Use

$$
L_V=\frac{\sum_i w_i(V_i-y_i)^2}{\sum_i w_i},\qquad \Delta_V=L_{V,H}-L_V.
$$

For fractional outcomes this is MSE for a conditional mean; for a binary event head it is the binary Brier score. Fit score/time and calibrated legacy baselines only on tune. Test exact terminal values and score ties. Ranking teams in one observed state does not establish ranking counterfactual actions for one team.

A carrier-specific delivery model requires distinct per-flag possession episodes: specified start, success, death/drop/takeover endpoints, and treatment of horizon/game-end censoring. An auxiliary probability of any team capture within 20 turns is not a specific carrier's delivery probability. Retain carrier DP until an appropriate consumer and labels exist.

### Decisions: does search choose better actions?

Freeze legal action candidates and a reference continuation evaluator using independent opponent/chance scenarios. Let \(Q^*(s,a)\) denote its high-budget estimate, not a claim of game-theoretic optimality:

$$
R(s)=\max_{a\in A_s}Q^*(s,a)-Q^*(s,\hat a),\qquad \Delta_R=\mathbb E[R_H-R_{new}].
$$

Use separate scenario batches for choosing the reference best action and measuring its advantage. Report reference uncertainty, near ties, large-regret mistakes and changed-decision rate. Evaluate branch states, not just recorded policy trajectories. Keep reference policy, action set and reward semantics fixed. When averaging branch values, use the mean of V on realized states, not V on averaged feature vectors.

### Real play: paired improvement of the complete bot

For each frozen fixture, replace the same seat with baseline/candidate in two runs, preserving map, flag seed and opponent binaries. Compute each run's actual pairwise outcome. Average scheduled seat/repeat differences within each map group first, then across independent groups:

$$
d_g=u_g^{new}-u_g^{base},\qquad
\Delta_N=\frac1{G_N}\sum_g d_g,\qquad
\Delta=\frac1{14}\sum_{N=2}^{15}\Delta_N.
$$

Tie and lower-rank outcomes contribute. This transparent paired utility is **not** the official Bradley–Terry rating. Run the official frozen scorer separately, with the same opponents/count weights, and report its uncertainty. Never linearly convert a utility gain into BT.

Each integrated version needs at least 10,000 independent confirmation map groups, initially at least 500 for each count. About 714–715 groups per count gives 10,000 across all 14. Counts served by general still need evaluation. Paired baseline runs, alternate seats and 400 turns cannot multiply the independent sample count. Three independently trained seeds need separate reports; more matches for one checkpoint do not replace training-seed replication.

Keep candidate timeout/crash/invalid-output consequences in the official main results. Infrastructure exclusions must be predetermined and symmetric. A/A checks expose runner/load instability. New final confirmation data are required after tuning multiple candidates.

## Uncertainty and practical thresholds

Bootstrap independent map/history groups, preserving all dependent turns, teams and paired results. Stratify by team count and frozen roster strata, then weight counts equally. Resample weighted loss numerators and denominators together. Never bootstrap individual unit rows as if independent.

The included calculator uses fixed-sample 95% cluster percentile-bootstrap intervals. These are approximate and depend on representative independent clusters. A degenerate empirical interval is explicitly flagged and is not proof of no population uncertainty. Its intervals condition on one checkpoint and frozen roster; training-seed uncertainty requires another level of analysis.

Pre-register one main confirmation comparison. Per-count switches chosen from the final set need multiplicity control; preferably choose switches on tune and confirm the whole frozen bot once. Do not repeatedly check ordinary 95% intervals and stop at the first favorable result. Use a fixed sample size or implement a valid sequential procedure.

Possible product thresholds are +0.005 overall practical gain and a -0.01 per-count noninferiority margin. These are choices, not mathematical constants. Evidence must actually rule out worse-than-margin loss; nonsignificance is not noninferiority. At paired-difference SD 0.3 and 10,000 independent groups, the ordinary large-sample 95% half-width is about 0.00588, so a +0.005 observed gain may remain inconclusive. Use actual cluster intervals for decisions.

## Runtime is an independent acceptance condition

Measure complete startup/first output against 2,000ms and complete later input/output against 50ms. Report per-count p50/p95/p99/p99.9/max, late-turn rate, games with any late turn, faults, cache/tier use and useful search work. Parsing, all heads, geometry, search and flushing are included. Kernels and MAC estimates cannot certify the deadline.

Keep raw durations or mergeable histograms. Averaging game-level p99s does not yield pooled p99. For G independent games with no game-level timeout, the one-sided 95% Bernoulli upper bound is \(1-0.05^{1/G}\), under an identical-distribution assumption; zero observed failures is not a guarantee of zero future failures. Repeat on the intended CPU and a weaker CPU/browser profile when relevant.

## Reusable scorecard

| Count/suite | Independent groups | P delta NLL [CI] | V delta MSE [CI] | Delta regret [CI] | Paired play gain [CI] | p99/max | Faults | Evidence status |
|---|---:|---|---|---|---|---|---|---|
| Every count 2–15 | | | | | | | | |
| Equal-count aggregate | | | | | | | | |
| Strict unseen lineage | | | | | | | | |
| Carrier/contact/respawn stress | | | | | | | | |

If P improves but play does not, write “prediction improved; strength unconfirmed.” Missing data stay missing, never zero. Separate artifact/data, prediction, decision, strength and runtime verdicts. Inconclusive tiers remain disabled.

## Calculator input and use

`scorecard_plan.json` freezes counts, stratum weights, comparisons and bootstrap parameters. Suggested main evaluation weights are elite 70%/mixed 30%, separate from 60/30/10 training quotas; prefer the official frozen representative opponent schedule when available. OOD and tactical stress have separate strata.

`scorecard.py` reads one JSON object per fixture. Each metric is `[weighted_loss_sum, weight_sum]`, using exactly the same fixed evaluation rows and weights for baseline/candidate. For P use actual deployment NLL; for V use squared error; for regret use the frozen reference's regret. `baseline_scores`/`candidate_scores` are full score arrays and `ego` is the substituted seat. Score arrays may be omitted for offline-only records, which cannot count as confirmed play.

`fixture_id` is unique. `cluster_id` groups a map's scheduled seats/repetitions and must belong to one suite/count/stratum in this calculator. Timing optionally supplies both sides' `first_ms`, raw post-startup `turn_ms`, and fault count. Missing strata/counts/paired metrics remain missing. Every populated stratum needs at least two independent clusters for the calculator's complete estimate. The sample-complete flag certifies only configured main-play sample coverage, not model quality or deployment approval.

```sh
cd evaluation
python3 -m unittest -v test_scorecard
python3 scorecard.py --plan scorecard_plan.json --input paired_metrics.jsonl --out report.json
```

`example_metrics.jsonl` and `example_report.json` are **synthetic arithmetic fixtures**, never league/model results. The calculator records input/plan/code hashes but does not verify training artifact identity, collect matches, infer models, fit official BT, authorize export or certify the host. Those remain pipeline responsibilities.

## Sources and mathematical checks

Probability scoring follows [Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437). Interval-based multi-task evaluation is informed by [Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264) and [the authors' rliable implementation](https://github.com/google-research/rliable); map clusters and equal-count utility are this project's adaptation, not reported CTF experiments in those papers.

Tests cross-check explicit score-pair enumeration and utility conservation; hand-calculated NLL/MSE/count weights; repeated-map dependence and equal-map play weighting; A/A zero gain; missing metrics/counts/strata; duplicate and invalid records; exact deadline boundaries; and pooled raw-time quantiles. Synthetic correctness tests cannot establish strength.
