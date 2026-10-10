# Bot groups and training-opponent sampling review

The original source audit reviewed ctf-fusion `af11de8` and the local S1/S2 source, manifests and existing equivalence logs. This requirement revision was synchronized with `aa3fcc1`. It is a source/method review, not a completed implementation. The full host fingerprints, strength confirmation and trained-model pool ablation have not been run by this reviewer. Synthetic checks below demonstrate algorithm behavior, not measured behavior of the 122 candidates.

## Conclusion

The reported 29 groups are **source-lineage groups**, not 29 demonstrated playing styles. Shared code is useful for finding near relatives and preventing training/test leakage. Actual opponent sampling should primarily follow measured, context-dependent behavior and matchup response, with lineage used to control redundant versions. Uniform file sampling, uniform source-family sampling and uniform global-style sampling each have limitations.

The Phase A infrastructure is a useful improvement: it separates inventory from strength, preserves requested actions, measures probe disagreement, records A/A variation and keeps review eligibility closed. Its own workflow acknowledges that full host evidence and final selection are still pending. Passing tool tests does not establish style equivalence.

## Updated requirement: open training membership, frozen evaluation

**Remove the fixed 15-bot training cap.** An opponent can enter the training reservoir when it offers a stable, learnable behavior, useful state/event coverage or a distinctive matchup response, with reliable protocol data and traceable identity. There is no mandatory global training-pool size. A weaker specialist can contribute; a new source file or unused mechanism name alone does not establish contribution.

| Artifact | Membership policy | Purpose |
|---|---|---|
| `open_training_pool` | Extensible between versioned data revisions; frozen within a run | Collect relevant behavioral and state coverage. |
| `frozen_eval_roster` | Fixed for the full comparison, with source/build hashes and fixture/weight rules | Compare model versions reproducibly. |
| Reserved OOD lineages/maps | Outside all training and tuning data | Test generalization to unseen families and environments. |

A core panel of 15 is a convenient historical budget, not a training admission limit. The current fixed-size selector can still construct such a panel, but its global distance, cluster uniqueness and family caps must not exclude otherwise useful training specialists. The current source does not yet implement the open-registry/hierarchical-sampling path; the updated report and execution prompt specify that work.

Record each member's contribution, count/context coverage, evidence level, source/dependency/build identity and sampling mass. Provisional exploratory collection is distinct from a confirmed strength claim. Preserve the existing 10,000-independent-map confirmation requirement for final bot assessments and integrated-version strength evidence; admitting limited exploratory data does not certify strength.

Admission is not an instruction to sample every file equally. Closely related or behaviorally redundant versions share a group's mass; a distinctive duel or rare-event specialist receives weight in the regime it actually contributes to. Strong competition coverage, verified behavior coverage and hard cases are separate components of the sampling mixture. Persistent random/noisy policies remain labeled augmentation rather than automatically qualifying as useful styles.

As new candidates arrive, admit, reweight or retire versions using tuning evidence, then freeze the dataset revision for confirmation. Keep whole OOD families reserved; adding every available family into training would remove the basis for an unseen-family claim. A larger pool should be tested against smaller pools at equal sample/training budgets, using P prediction, tactical calibration, branch regret and paired play.

More training opponent identities do not imply one deployed P network per identity. The shared models retain their team-count/capacity dispatch. If diversity increases ambiguity, evaluate public-history descriptors or measured online adaptation; bot names/source ids are metadata, not privileged runtime inputs. V labels retain their declared continuing-policy and opponent-distribution semantics.

## Concrete evidence that lineage is not style

Using the repository's exact 8-token-shingle Jaccard function on current local exports:

| Pair with S1_A_Conservative | Code similarity | Conditional mechanism difference |
|---|---:|---|
| S1_B_DuelPlus | 99.96% | Enables the extra duel refinement at 2 teams. |
| S1_C_Adaptive6p_Fast15 | 99.76% | Different macro/search mode at 6 teams; shorter budget from 10 teams. |
| S1_D_CrowdHabit | 84.02% | Habit mixture, carrier DP and decorrelated scenarios at 8+ teams. |
| S2_3_RiskStrong | 99.95% | Changes risk sensitivity at 5+ teams. |
| S2_4_ReturnAware | 99.29% | Models returning-flag turns that the reference skips. |

All of these pairs exceed the 0.7 lineage threshold. The existing equivalence log records S1_B differing from A on 106/1,600 duel probe turns; it also records S2_3 differing on 20–35/800 turns at several crowded sizes. These are prior replay results, not newly run games. The source switches and masks independently establish that the active mechanism changes depend on team count.

These differences also do **not** automatically prove five entirely new macro styles. A timer-only change, a rare tactical refinement and a persistent carrier-risk preference should receive different labels. Describe the actual behavioral tendency and its context rather than infer style from a filename or module switch.

Source lineage is explicitly implemented as thresholded Jaccard connected components: [inventory.py](https://github.com/Casper015/ctf-fusion/blob/af11de8/roster/inventory.py#L178-L225). It does not measure opponent actions or outcomes.

## Behavior clustering needs stronger safeguards

### Averaging counts can hide a specialist

The code computes per-count distances but forms the global `cluster_of` from the equal-count averaged matrix. `roster.catalog` uses this global label for selection. If two policies differ by 0.20 at 2 teams and are identical at the other 13 counts, their global distance is about 0.0143, below the default 0.02 margin even with no A/A noise. A strategically useful duel specialist can therefore be collapsed into the reference.

Use per-count/context behavior profiles for sampling and retain an explicit exception for a verified specialist. Whole-suite distance can remain a summary, but cannot prove equivalence in every rated size. Global selection should cover the collection of count/event profiles rather than require only one flat cluster per bot.

Sources: [global averaging and clustering](https://github.com/Casper015/ctf-fusion/blob/af11de8/roster/distance.py#L159-L202), [catalog uses global cluster](https://github.com/Casper015/ctf-fusion/blob/af11de8/roster/catalog.py#L50-L59).

### Connected components do not bound cluster diameter

A synthetic distance matrix with AB=0.02, BC=0.02 and AC=0.04 makes the current `union_clusters` merge all three at margin 0.02. Every adjacent edge passes, while the endpoints fail. Prefer complete-linkage or a stated maximum-diameter rule for equivalence-like groups, and report maximum intra-group distances and their uncertainty. Clustering can organize the catalog without claiming every member is interchangeable.

### High A/A noise is not evidence of equivalence

The current pair threshold is `max(noise_A, noise_B) + margin`. In a synthetic example, deterministic left/right policies have distance 1, while a balanced random policy has distance 0.5 to each and A/A noise 0.5. The current graph merges all three through the noisy policy. This demonstrates a failure mode; it does not assert that a particular inventoried bot produced those measurements.

Collect enough repeated runs on identical states to estimate conditional distributions and separate within-policy randomness from between-policy differences. Measure joint-team move/action distributions, not only per-unit disagreement. Two policies can have identical individual marginals while one always coordinates and the other acts independently. Use map-level uncertainty and mark unobserved tactical regimes as unknown rather than interchangeable.

Source: [noise threshold and union](https://github.com/Casper015/ctf-fusion/blob/af11de8/roster/distance.py#L90-L108).

### Fixed replay is one kind of evidence

Shared histories enable controlled comparisons, but they were produced by other policies. A persistent bot's requested actions can disagree with the next recorded state, affecting internal role/route/habit state. Combine fixed-history probes with live trajectories and valid counterfactual continuation probes. Preserve raw requests alongside canonical effective actions so no-op normalization does not hide anticipatory return/pickup behavior. Rare-event coverage must be reported explicitly.

## Recommended sampling design

Maintain three separate descriptions:

1. **Lineage:** shared source/strategy ancestry, for redundancy control and strict holdouts.
2. **Behavior profile:** requested joint action and tactical tendencies, conditioned on team count, state and history.
3. **Response profile:** performance against the same frozen strong opponents/rosters, per team count. Similar-looking policies may have different exploitable weaknesses or specialist strengths.

If “type” means aggressive/carrier-focused/defensive, it is already a behavior label. If it means heuristic/search/DP/habit architecture or source ancestry, it is only a structural descriptor. Use multi-label mechanisms and profiles rather than force every bot into one universal style.

For each count N, let C_N(b) be a verified behavior group and K_N the number of groups. An exploratory style-balanced bot distribution is

$$
\rho_{style,N}(b)=\frac{1}{K_N\,|C_N(b)|}.
$$

This gives each behavior group equal mass and divides that mass among its versions. It is appropriate only after groups are credible; it is not automatically the target competition distribution.

A proposed collection mixture is

$$
\rho_N=\lambda_f\rho_{field,N}+\lambda_s\rho_{style,N}+\lambda_h\rho_{hard,N},\qquad
\lambda_f+\lambda_s+\lambda_h=1.
$$

The first term represents the intended strong competition field, the second supplies behavioral coverage, and the third supplies verified difficult matchups or tactical cases. Starting weights such as 0.6/0.3/0.1 are hypotheses to tune and freeze, not proven optima. A high prediction loss caused by irreducible random actions is not sufficient to prioritize a “hard” opponent.

Sample team counts with explicit balance, then matchup regime, behavior profiles and versions. Add state/event stratification for pickup, delivery, interception, focus fire, manual drop and respawn. Track inclusion probabilities and avoid letting a large lineage dominate through its number of files. A lineage cap such as “at most three” is a provisional regularizer; verified distinct specialists from one lineage may justify relaxing it.

For general-model training, equal matches per count do not automatically give equal training mass: N-team matches yield N acting-team rows per turn. Explicitly choose the intended count weighting in minibatches/losses. V labels also require a declared continuing policy and opponent distribution; P style balancing alone does not define V's decision value.

## The current collector does not yet sample groups

`roster.pool` writes every selected bot with `weight: 1` and stores family/cluster labels as metadata. `training.collect.schedule` chooses names directly by these file weights, without first choosing a style or lineage. A diverse pool is helpful, but this is not hierarchical style sampling.

Sources: [pool entries](https://github.com/Casper015/ctf-fusion/blob/af11de8/roster/pool.py#L33-L49), [name-level sampling](https://github.com/Casper015/ctf-fusion/blob/af11de8/training/collect.py#L62-L84).

With exactly 15 candidates and 15 seats, sampling without replacement includes all candidates in every match. Weights cannot change the composition. The pool writer also adds Noise by default, giving 16 entries if the selected 15 exclude it; a small relative weight does not mean a negligible game-inclusion probability when filling 15 seats.

To change composition deliberately, retain the wider open reservoir or allow controlled repeated policy instances in designated training regimes. Log the resulting joint roster distribution. Use separate regimes for representative mixed games and selected same-style concentration tests; retain a representative frozen roster for confirmation. A 15-member comparison panel may coexist with a larger training reservoir, with independent per-count version weights and explicit repetition rules.

## Validation and additional report limits

Compare file-uniform, lineage-balanced and count/context behavior-balanced collection under equal data and training budgets. Confirm on new maps and held-out lineages, reporting P NLL, tactical-event calibration, branch regret and real paired play. Add response-profile coverage to the selection audit. The literature explicitly distinguishes behavioral diversity from response diversity; applying both here is a proposed CTF adaptation: [NeurIPS 2021 paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/07bba581a2dd8d098a3be0f683560643-Abstract.html). Population mixtures and empirical matchup analysis also appear in [PSRO](https://arxiv.org/abs/1711.00832).

The token-stream alias detector should remain provisional: removing layout also removes preprocessor line/function-macro boundaries, and macros such as `__LINE__` can depend on formatting. A synthetic pair with different preprocessor line structure already produces the same normalized identity. Preserve compiler-aware preprocessing/build context and verify observable equivalence before propagating another program's evidence.

The attachment's XL 4.73ms figure was from the earlier scalar capacity workload with 24 V calls and a safety multiplier. Commit `aa3fcc1` includes newer bit-exact SIMD reports, so use the latest host/backend result rather than that older ceiling. Native capacity measurements still do not establish the full bot's 50ms deadline or that fixed large capacity is stronger than routing. Fixed-capacity models should be serious comparison candidates; routing should earn its complexity through measured quality and useful search work.

No full pool or panel was selected or evaluated by this review. The five pairwise code similarities, connected-component examples and token normalization example were computed locally; existing replay results were read from the project's own logs. This update changes the requirements and documentation, not training/runtime behavior.
