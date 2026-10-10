# Tier M Results and the L/XL/XXL Evaluation Plan

Date: 2026-10-09. Reviewed commit: [7c3f26d](https://github.com/Casper015/ctf-fusion/commit/7c3f26d6775d79aff29cc96ff810627dee7d2534).

**Bundle status:** This remains a snapshot-specific S/M review. Later commits add L exports, integrated P screens, and experimental V ablations. Read the [review bundle index](review_bundle_20261009/README.md) and [current status at 0a0badc](review_bundle_20261009/CURRENT_STATUS_AND_V_ABLATION_REVIEW.md) before interpreting older missing-feature findings as current blockers. The separate carrier report and next-agent prompt are linked there.

**Publication update:** Before uploading this report, main advanced to [29a6412](https://github.com/Casper015/ctf-fusion/commit/29a6412). That subsequent commit fixes the ladder's missing backend/device forwarding and defaults it to Torch/CUDA. It also adds a [measured RTX 4080 Laptop benchmark](https://github.com/Casper015/ctf-fusion/blob/29a6412/reports/training_bench_DESKTOP-G60OJUD.md): L, n02, 200,000 rows, one epoch, batch 1,024. Reported epoch throughput improves 3.8 times for P and 1.9 times for V; complete trainer wall-time improves approximately 1.27 times and 1.31 times respectively. Epoch time includes validation, while wall time also includes loading and post-training work. This small resident-data experiment does not establish full-dataset streaming performance or XL/XXL convergence. The missing-CUDA/missing-benchmark findings below describe the reviewed 7c3f26d snapshot and are superseded by that update; the M-versus-S quality findings and evaluation requirements remain applicable.

This is a report-only review. The commit was fetched and read without changing the checked-out source. Existing JSON results were compared and artifact checksums were recomputed. No training, native execution, match collection, or timing benchmark was run for this review.

## Assessment

**Tier M improves opponent movement prediction over S on the committed test set at every count from 2 through 15. It does not yet establish stronger integrated play.** The improvement is consistent enough to justify testing the larger tiers. P benefits more visibly than V, so P and V should be selected independently.

The commit exports eleven M bundles: ten dedicated counts plus general, which serves 9, 11, 13, and 14 teams. It also includes the preceding optional CUDA backend and large-capacity support. There are no committed L/XL/XXL trained results in this snapshot, and the M reports contain no integrated-bot match-strength result.

## Measured S-to-M differences

S and M use the same training dataset digest `e2472a8610a9776c`, and the same evaluation/fit fixture identities across all eleven bundles. Both use seed 1 and the same batch/epoch-limit settings. This is a useful architecture comparison, but only one training seed is represented. The following numbers come from [S reports](https://github.com/Casper015/ctf-fusion/tree/7c3f26d6775d79aff29cc96ff810627dee7d2534/reports) and [M reports](https://github.com/Casper015/ctf-fusion/tree/7c3f26d6775d79aff29cc96ff810627dee7d2534/reports_m).

| Teams | S integer P NLL | M integer P NLL | Reduction | Joint top-1 increase |
|---:|---:|---:|---:|---:|
| 2 | 1.9854 | 1.8784 | 0.1070 | 2.11 percentage points |
| 3 | 1.9841 | 1.8821 | 0.1020 | 1.85 pp |
| 4 | 2.0370 | 1.9508 | 0.0862 | 1.49 pp |
| 5 | 2.0982 | 2.0430 | 0.0552 | 0.88 pp |
| 6 | 2.1467 | 2.0886 | 0.0581 | 0.85 pp |
| 7 | 2.1341 | 2.0804 | 0.0537 | 0.74 pp |
| 8 | 2.1497 | 2.0993 | 0.0504 | 0.62 pp |
| 9 | 2.1383 | 2.0925 | 0.0458 | 0.66 pp |
| 10 | 2.1428 | 2.0970 | 0.0458 | 0.79 pp |
| 11 | 2.1020 | 2.0525 | 0.0495 | 0.83 pp |
| 12 | 2.0944 | 2.0472 | 0.0472 | 0.72 pp |
| 13 | 2.1220 | 2.0758 | 0.0463 | 0.76 pp |
| 14 | 2.1203 | 2.0758 | 0.0445 | 0.76 pp |
| 15 | 2.0825 | 2.0414 | 0.0411 | 0.69 pp |

The equal-count mean integer-NLL reduction is **0.05949 nats**. The mean of the fourteen relative reductions is **2.86%**; this is a loss reduction, not a win-rate improvement. The deployed neural/heuristic mixture also improves at all counts, with an equal-count mean reduction of **0.05765 nats**. Top-1 and ECE in the current evaluator describe the integer neural predictor, while mixture NLL scores the deployment adapter.

The M mixture alphas range from 0.9481 to 0.9835. Alpha is a fitted blending weight, not prediction accuracy or a probability of winning. Integer-versus-float P NLL penalties are approximately 0.00055–0.00370 nats, much smaller than the observed S-to-M gain. Action-head NLL improves in every model report, but these action/target metrics use the float model conditioned on the observed requested move; they do not validate the complete sampled integer action pipeline.

There are two important limits:

- ECE worsens at **10 of 14 counts**, despite lower NLL. Calibration therefore needs separate attention as capacity grows.
- V's float Brier reductions are only **0.000082–0.000275** across the eleven reports. For n15, Brier changes from 0.029258 to 0.029175. Early-game Brier changes from 0.057668 to 0.057478. This is much smaller than the movement-prediction gain and does not establish better action ranking.

Existing gain standard errors compare each model against the heuristic baseline. They are **not** confidence intervals for the paired M-minus-S difference. Recompute per-map paired loss differences to establish that comparison statistically.

## Cost and artifact status

| Tier | Joint P MACs per opponent team | V MACs per node | Combined native buffer width |
|---|---:|---:|---:|
| S | 30,096 | 8,976 | 256 |
| M | 64,288 | 22,048 | 256 |
| L | 120,576 | 60,480 | 512 |
| XL | 273,920 | 120,896 | 1,024 |
| XXL | 678,912 | 372,864 | 2,048 |

M exports 70,196 parameters versus S's 31,068. Its joint P arithmetic is **2.14 times** S and V arithmetic is **2.46 times** S. Joint P MACs exclude the additional conditional action/target work; MAC ratios are not measured wall-time ratios. These figures are derived from the manifests and [capacity report](https://github.com/Casper015/ctf-fusion/blob/7c3f26d6775d79aff29cc96ff810627dee7d2534/reports/capacity_DESKTOP-G60OJUD.md).

All eleven M NPZ hashes match their manifest entries. Declared accumulator bounds remain below 2^30, with a maximum of 59,183,385. These are artifact/metadata checks, not a new native-parity test. All committed HPP byte hashes differ from their advertised Windows hashes, but reconstructing CRLF line endings reproduces every advertised hash exactly. This identifies a line-ending portability issue, not evidence that weights changed; future cross-platform identity checks should declare their byte-normalization rule.

Every M manifest still has `value_approved: false`. The exported [registry](https://github.com/Casper015/ctf-fusion/blob/7c3f26d6775d79aff29cc96ff810627dee7d2534/inference/models_m/registry.hpp) nevertheless exposes non-null V pointers. Consumers must not treat pointer availability as approval to enable V in competition search.

## Check the ongoing training backend

The new [Torch backend](https://github.com/Casper015/ctf-fusion/blob/7c3f26d6775d79aff29cc96ff810627dee7d2534/fusion/torch_backend.py) implements the important compatibility details: strict clipped-ReLU gradients, legal-support smoothing, matrix-only coupled Adam, globally shuffled row order, compact P windows, and a slim 54-feature V table. Validation runs on the device; calibration and export return to NumPy. The previous CPU-only report describes the older source snapshot and should not be read as the status of this new backend.

However, [train_tier_ladder.py](https://github.com/Casper015/ctf-fusion/blob/7c3f26d6775d79aff29cc96ff810627dee7d2534/tools/train_tier_ladder.py) does **not** pass `--backend torch --device cuda`, and [the trainer default](https://github.com/Casper015/ctf-fusion/blob/7c3f26d6775d79aff29cc96ff810627dee7d2534/training/common.py) is NumPy. The ladder has no CLI forwarding option for those arguments. If the ongoing L/XL/XXL jobs were started through this script unchanged, they are CPU jobs. Direct trainer commands can select CUDA. Check the actual command, run configuration, and the `[torch]` environment line before claiming GPU use. The committed M metadata records training against the older 8595c36 source and contains no CUDA backend record.

The ladder's skip logic also checks file existence rather than full data/model/config identity. Before trusting a resumed result, verify the report/checkpoint hashes, shape, seed, dataset digest, and export identity. A previous report left in the same directory must not approve new weights.

The CUDA resident budget is applied to each split separately, not to total process VRAM. Account for train plus validation, current windows, permutations, activations, and temporary allocations together, especially for XXL. The reviewed 7c3f26d snapshot has no committed training-throughput benchmark; the subsequent measurement is summarized in the publication update above.

## How to test L, XL, and XXL

Use the existing [evaluation contract](https://github.com/Casper015/ctf-fusion/blob/7c3f26d6775d79aff29cc96ff810627dee7d2534/handoffs/diverse15_distance_tiered_policy/evaluation/EVALUATION_PLAN.md). It already separates probability quality, V quality, branch regret, real play, and runtime. The following sequence makes the capacity comparison actionable.

### 1. Freeze the comparison

Keep the dataset and map splits fixed across S/M/L/XL/XXL. Fit temperatures, mixture alpha, tier scheduling, and any count-specific choices on validation/tune only. Because the current test set has already been inspected for S/M and will inform larger-tier selection, use a new untouched confirmation set for the final chosen system. Reserve complete unseen opponent lineages; the current manifest has an empty `holdout_family` list.

Finish the initial seed-1 ladder as a screen, then repeat finalists with at least three independent training seeds. Compare both equal-data quality and time to comparable held-out quality. Matching epochs alone does not establish convergence or equal training compute.

If the larger tiers move to CUDA, add S/M CUDA-FP32 controls with the same batch and optimizer-step protocol before attributing all changes to capacity. Existing NumPy S/M remain legitimate deployment candidates, but a backend/precision change is another experimental factor.

### 2. Validate the actual integer artifacts

For every trained bundle, check float-to-integer degradation, native integer parity, incremental/full V equality, registry dispatch, buffer width, allocation behavior, and overflow bounds. Exercise real held-out records and tactical stress cases. The existing XXL synthetic test validates wide-buffer plumbing; it does not validate every newly trained artifact.

Useful existing correctness commands, for the updated checkout when testing is authorized:

```sh
python -m unittest -v tests.test_capacity_profiles
python tools/capacity_bench.py --verify --tiers L XL XXL --teams 2 15 --verify-turns 4 --value-calls 2
```

### 3. Compare prediction and value separately

Score the actual integer deployment mixture, not only float-model loss. Use paired per-map deltas and map-cluster confidence intervals against **M**, the current neural reference. Include each count 2–15, each opponent lineage, and flag/contact/respawn/quiet cases. Report lethal-hit, interception, collision, flagshoot, manual-drop, and possession-change probability errors where labels support them. Check target-candidate coverage and minority actions separately from common wait actions.

For V, compare against a tuned score/time baseline and the current heuristic evaluator, not only constant 0.5. Separate early/middle/late game. Measure counterfactual branch ranking for the same team under the intended continuing policy; team ranking on observed states is insufficient.

For example, after an L pair is trained and its paths are verified:

```sh
python -m training.evaluate_offline --model n15 --policy runs_l/policy_n15 --value runs_l/value_n15 --out reports_l/n15
```

The current evaluator does not automatically compute paired M-minus-L confidence intervals or all proposed tactical-event diagnostics. Those are evaluation work still required.

### 4. Measure better decisions under two budgets

First hold action candidates, search depth/nodes, and opponent/chance scenarios constant to isolate the model's decision effect. Then hold the **complete CPU turn-time budget** constant to test the deployable tradeoff. A larger V may improve each node while allowing fewer nodes; a larger P may sharpen rollout predictions while reducing rollout count.

Use a frozen higher-budget continuation evaluator for branch regret, with separate scenario samples for selecting the reference best action and estimating its advantage. Record changed-decision rate, large-regret mistakes, near ties, and uncertainty.

Run ablations: unchanged heuristic bot; M P plus unchanged heuristic V; candidate P plus unchanged heuristic V; fixed P plus candidate V; and the combined candidate. Test mixed capacities such as XL P with S/M V. V configurations remain experimental until independently approved.

### 5. Confirm complete-bot strength and latency

Use paired fixtures with the same maps, chance seeds, opponent binaries, and substituted seat. Keep faults and timeouts in the official outcome. Screen with smaller suites, then require **at least 10,000 independent confirmation map groups per final integrated version**, initially at least 500 per count. Seat rotations and baseline replays belong to the same cluster and do not multiply the independent sample count. About 714–715 groups per count covers all fourteen counts. Report paired utility and the official Bradley–Terry rating separately, with uncertainty.

Collect complete first-reply and subsequent-reply durations, including parsing, encoding, all heads, search, and flushing. Report pooled p99, p99.9, max, faults, and late-game frequency under representative concurrent CPU load. Validate the hard 2-second/50-ms limits on the intended CPU.

The historical [16-process capacity test](https://github.com/Casper015/ctf-fusion/blob/7c3f26d6775d79aff29cc96ff810627dee7d2534/reports/capacity_extreme_scaling_16seats.md) estimates XL at 1.38 ms and XXL at 2.74 ms for inference with a 1.5 safety factor. It uses synthetic weights, one 15-team stream, and only 24 V calls. These figures establish promising cost headroom, not a complete-bot deadline certificate. Replay more maps and the real trained/search workload.

## Selection rule

Keep the smallest P/V combination that improves confirmed decisions and paired play within the CPU budget. A lower NLL can justify further experiments; it cannot approve stronger play by itself. If XXL improves offline P but loses equal-time matches, use it as a teacher or improve its consumer rather than choosing it by size. If larger V shows only the current tiny Brier gains, retain a cheaper evaluator while investigating features, continuation-policy targets, and branch-specific training.

No code was changed by this review.
