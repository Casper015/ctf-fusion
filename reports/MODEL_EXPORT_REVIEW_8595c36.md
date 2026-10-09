# Model-export review: training evidence and native correctness

Reviewed commit: `8595c369edfee1033f71188abca69acd34baa26c`, with training changes in parent `fda8153b27d374478d656c8072f0897290742d4b`. Scope: committed reports, export manifest, profiles and source contracts. No matches, training, latency tests or raw-data reconstruction were run. Float checkpoints, optimizer logs, raw matches, shard arrays and the dataset index were not present in the reviewed snapshot.

## Conclusion

This is a real step from an untrained pipeline to a reviewable exported model matrix with substantial **reported held-out opponent-movement gains**. The evidence is internally consistent and far stronger than the earlier 400-game collection pilot. It supports trying the P movement model in controlled integration tests. It does not establish match-strength improvement, out-of-family generalization, rare-action mastery, or a safe V-based search replacement.

No contradiction was found between the reported P gates and their manifest artifact identities. Absence of raw training data prevents independent recomputation; it is a verification limitation, not evidence the reported training was fabricated. Root review separately covers NPZ/header and native parity.

## Exported-artifact checks performed by the root reviewer

All eleven NPZ bundles have distinct file/content identities. Conservative accumulator bounds were independently recomputed for all 55 component networks, match the manifest, and remain within 2^30 for the declared feature/activation ranges. Their equal 76,618-byte sizes reflect equal default network shapes and the exporter's uncompressed `np.savez` format, not identical weights. Every NPZ file hash and semantic integer-weight digest agrees with its manifest. Every gate report matches the manifest's artifact identity, gate rows and alpha values. Header hashes match the exporter-recorded Windows CRLF bytes; Git's LF normalization alone explains the checkout-byte differences.

The actual committed headers were compiled, including the complete dispatch registry. On bounded synthetic streams at 2 and 15 teams, six turns each, all eleven models passed:

- **360,162 raw int32 comparisons**, covering mover, squad, action, target and value outputs, with zero differences against the NPZ integer emulator;
- full/incremental value agreement and zero allocating turns on the instrumented runtime-check path;
- registry dispatch over all 14 official counts.

These are correctness checks of real exported arrays. They are not training validation, complete-bot match results, or a development-host latency benchmark. The stream sample does not certify every future path is allocation-free. [Machine-readable check record](MODEL_NATIVE_REVIEW_8595c36.json), [parity driver](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/tools/infer_check.cpp), [integer emulator](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/fusion/models.py#L216-L238).

## Actual-model timing and integration remain separate

The updated environment report explicitly benchmarks **`weights_placeholder.hpp`**, not one of these eleven trained artifacts. That placeholder has the same default shapes, so it is a useful cost proxy; its 0.179ms policy p99 is not a complete integrated-bot deadline certificate. Measure an actual model and then the full reply under representative concurrency, including startup, feature encoding, action/target forecasts, search and output. Keep the 2s/50ms hard limits. [Host report](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/reports/env_check_DESKTOP-G60OJUD.md), [benchmark model selection](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/tools/env_check.py#L149-L155).

On the designated host, the existing checker accepts an actual export:

```sh
python tools/env_check.py --ai-test AI_TEST_ROOT --model inference/models/weights_n15.hpp --teams 15 --out reports/trained_n15_host
```

Replace `AI_TEST_ROOT` with the checkout path. This command launches host matches and timing; it was not run by this review. Repeat representative counts and add loaded complete-bot checks. The small models are plausible CPU candidates; no larger-capacity result is added by this commit.

Each exported header correctly declares `kValueApproved=false`; the manifest also says `value_approved=false`. However the generic registry still exposes a non-null value pointer. Integration must explicitly preserve the approval boundary; consider carrying `valueApproved` in `ModelSet` or disabling value use until a separate experimental flag is deliberately enabled. A pointer's presence is not an approval. [Header boundary](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/inference/models/weights_general.hpp#L1728-L1732), [registry](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/inference/models/registry.hpp).

Use the declared **joint-v2** adapter when testing the published alpha. If the planned mathematical layer consumes only movement marginals or otherwise changes the sampler, evaluate that actual adapter and refit its alpha; the existing joint-v2 certificate does not transfer automatically.

## Data volume is no longer the old pilot

The general model's embedded training configuration reports **1,500 matches at every count 2..15: 21,000 matches**, 30,049,121 kept team rows overall, comprising:

| Split | V rows | Valid, policy-labelled P rows |
|---|---:|---:|
| Train | 24,193,790 | 24,153,825 |
| Validation | 2,986,476 | 2,981,919 |
| Test | 2,868,855 | 2,864,979 |

Across the 14 served counts, reports contain **2,038 map clusters**, with **120–163 maps/count**. These are reported map counts, not millions of independent samples: rows within one trajectory and the repeated V outcome labels are correlated. The earlier pilot was only 100 matches at counts 2,5,8,15 (400 total), and is a separate plumbing/coverage artifact. [Training manifest](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/inference/models/MANIFEST.json), [earlier pilot](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/roster/report/training_selection_20261009/pilot_summary.json)

The training registry identity `85f96f37...` matches the committed recommended 58-program pool. Manifest regimes report 20,586 mixed and 414 augmented fixtures, consistent with the intended roughly 2% augmentation-fixture prior. The parent commit now preserves sampling metadata in the dataset index. General P/V training explicitly used `count_weighting=inv_teams`; dedicated models used `none`. This addresses the previous automatic N-times row weighting for general training when equal full games are collected per count. The regime counts embedded in dedicated model configs are global dataset counts, not dedicated-count counts. [Recommended pool](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/profiles/open_training_pool_recommended.json), [parent changes](https://github.com/Casper015/ctf-fusion/commit/fda8153b27d374478d656c8072f0897290742d4b)

All exported models use the small/default shapes: mover 16, squad 64/32, action 16, target 8; V 128/16. These results do not evaluate larger capacities or a capacity learning curve.

## P evidence: encouraging and appropriately separated from playing strength

Every served count reports a positive integer-model mixture gain over the legacy heuristic. Full ranges are:

- Joint-move log-loss gain: **0.9023–1.4377 nats**, with map-clustered standard errors.
- Validation-fitted mixture alpha: **0.9364–0.9711**.
- At 2 teams: heuristic 3.0934 versus integer mixture 1.9834, gain 1.1099 ± 0.0416 SE, 151 maps.
- At 15 teams: heuristic 2.9825 versus integer mixture 2.0802, gain 0.9023 ± 0.0096 SE, 128 maps.

Gains appear in quiet/contact/flag strata as well as respawn strata; they are not explained solely by predicting off-board stay actions. Reported integer quantization losses are close to float losses. Around one nat of gain means substantially higher geometric-average likelihood for the observed joint movement, not one nat of rating gain or a guaranteed better action. [2-team report](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/reports/n02.md), [15-team report](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/reports/n15.md), [general served counts](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/reports/general.md)

The gate's source contract is sound for the reported default run: inertia and alpha fit on validation, evaluation on test, score the **integer joint-v2 adapter**, require complete q_H from one recorded legacy build, and cluster gain uncertainty by map. Every report's checkpoint hashes and integer-weight digest agree with the exported manifest identity. The common baseline source digest is `fd62e8cbe942f541`; reports say no missing active-unit forecasts. [Gate implementation](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/evaluate_offline.py#L170-L247), [baseline replay](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/heuristic_baseline.py#L54-L99)

Different fit/test match digests and map-hash split source are positive provenance evidence. Without the actual index or match-ID lists, their map intersections and raw content cannot be independently checked. Once test results are repeatedly used to select capacities or redesign the model, reserve a further untouched confirmation set.

## Important limits before deployment claims

### 1. No unseen-opponent evaluation is published

Every report has `holdout_family=[]`. The collection pool correctly reserves AI_Arena_Player02 and Player4_v2 source lineages, but those lineages are outside collection; no separate OOD report includes their games. The present test is fresh-map/in-distribution evaluation over the trained pool, not demonstrated unseen-family generalization. Add legal fresh-map games with the reserved families and report per-family/per-count movement NLL, calibration and joint errors. [Dataset options in report](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/reports/n02.json), [map splitter](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/dataset.py#L73-L106)

### 2. Published action/target accuracy does not certify the exported whole-action predictor

`heads()` evaluates **float** action/target heads conditioned on the **observed requested move**. It does not run the integer action/target heads or test their performance conditioned on moves sampled by the P/heuristic mixture. Target accuracy also excludes unavailable labels and requests outside the candidate slots. The reported 90–95% values therefore cannot be read as deployed full-team-action accuracy. The movement gate approves movement likelihood only. [Head diagnostics](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/evaluate_offline.py#L135-L155), [gate scope](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/evaluate_offline.py#L184-L265)

Add exported-integer head metrics, legal masking checks, joint `(move, action, target)` likelihood, target-coverage counts and simple baselines such as majority-wait/nearest-target. Report pickup/drop/attack separately, including rare return/drop/fire opportunities. Overall action accuracy can be dominated by WAIT.

### 3. V's gains need a stronger baseline and a continuation contract

At 15 teams, V Brier is 0.0293 versus constant 0.5's 0.0932. However, its early/middle/late Brier values are 0.0577/0.0236/0.0067. At 2 teams they are 0.1176/0.0518/0.0203. The strong late accuracy is compatible with much of the final ordering already being visible in scores; it does not establish accurate long-horizon branch evaluation.

Compare against **current pairwise score outcome**, a validation-fitted score-margin/remaining-time baseline, and the existing search evaluation. Report early phases, close score margins and actual alternative-branch decisions, with map-clustered intervals. The millions of ranking pairs are not millions of independent outcomes. Current V Brier/BCE/calibration and ranking summaries have no uncertainty interval, and pair ranking counts are unweighted despite the rest of the diagnostics using sampling weights. [Value diagnostics](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/evaluate_offline.py#L268-L289), [ranking metric](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/fusion/metrics.py#L60-L73)

V labels estimate final pairwise utility under the collected mixture of bot continuation policies: the fraction of other teams beaten, with half credit for ties. This is not the probability of finishing first. They are not optimal-play values, not our search policy's value automatically, and not a specific carrier's delivery-before-loss probability. Capture20/death10 auxiliaries do not supply that delivery target. The export correctly keeps `value_approved=false`; preserve that boundary until value/decision ablations show benefit. [Targets](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/fusion/labels.py#L92-L104), [export approval boundary](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/export_weights.py#L175-L177)

### 4. Rare-event volume is not reported for the full dataset

The earlier 400-game pilot reported 131 voluntary drops, with hit follow-ups explicitly counting any friendly hit. The full training/evaluation manifests do not publish retained drop labels, distinct maps carrying drop/fire episodes, per-class action counts or censored carrier delivery episodes. The default stride 2 still discards about half of eligible critical turns even when flag/contact keep rate is 1. These omissions do not invalidate the overall P movement results; they limit claims about specialized tactical prediction. [Pilot coverage](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/roster/report/training_selection_20261009/pilot_summary.json), [subsampling](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/dataset.py#L87-L94)

### 5. Content provenance remains weaker than experiment metadata

Recording registry IDs is an improvement, but the data digest and evaluation split digests still hash match IDs rather than raw/shard content. The shard index retains sampling metadata but not complete bot/engine/source hashes or raw content checksums. Collection resume still skips existing files solely by filename. These are reuse risks, not established contamination in this corpus. Ship a compact dataset index, immutable collection-run manifest, raw/shard content checksums and training logs, and reject incompatible resume before the next expanded dataset. [Data digest](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/dataset.py#L96-L137), [evaluation identity](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/evaluate_offline.py#L382-L387), [collection resume](https://github.com/Casper015/ctf-fusion/blob/8595c369edfee1033f71188abca69acd34baa26c/training/collect.py#L210-L218)

## Next decision

Keep these exports as the first serious **P-movement candidate**. Verify integer action/target behavior and the exact deployed joint sampler, add reserved-family reports, then run paired equal-time bot tests with movement P enabled and V held fixed. Test V separately against score/search baselines and declared continuations. Only after this should larger capacities be compared with the same map subsets and training seeds. These reports demonstrate useful predictive learning on the recorded distribution; they do not yet demonstrate stronger competitive decisions.
