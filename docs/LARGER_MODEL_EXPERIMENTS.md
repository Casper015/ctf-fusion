# Larger models: capacity presets up to XXL

Larger networks are plausible now that the host kernels are fast and training can run on the GPU (`--backend torch --device cuda`, [ENVIRONMENT.md](../training/specs/ENVIRONMENT.md#training-backend)). A larger network does not by itself improve predictions or play.

This change adds shared shape presets and carries their buffer requirements through training, evaluation, export and the native correctness tools. The default S architecture and the 2–15 dispatch matrix are unchanged; selecting a preset enables no new model.

## Shared presets

`fusion/capacity.py` is the single definition used by the trainers (`--capacity`), `tools/host/capacity_bench.py` and `tools/host/bench_training.py`. P and V can choose capacities independently, and the hidden-layer flags override individual heads.

| Preset | Mover hidden | Squad hidden | Action hidden | Target hidden | V hidden | Buffer width |
|---|---|---|---|---|---|---:|
| S (default) | 16 | 64,32 | 16 | 8 | 128,16 | 256 |
| M | 32 | 128,64 | 32 | 16 | 256,32 | 256 |
| L | 64 | 256,64 | 64 | 16 | 512,64 | 512 |
| XL | 128 | 512,128 | 128 | 32 | 1024,64 | 1024 |
| XXL | 256 | 1024,256 | 256 | 32 | 2048,128 | 2048 |

These are experiment starting points, not optimal shapes or latency approvals.
- **Ceilings checked early:** an explicit ceiling that is too small fails before any data loads.
- **Custom shapes:** widths without a preset still need an explicit `--max-width`.
- **Compatibility:** existing calls and defaults are unchanged.

## Why the presets stop at XXL

The host capacity reports (`f36e9f3`) also measured two wider shapes, 3XL and 4XL: XXL's widths doubled and quadrupled. The figures below are from one 15-team real-engine stream with synthetic weights and 24 V calls per turn. Each is the estimated inference time, 1.5 × (worst P p99 + worst V-turn p99), against the configured 5 ms per-turn budget. It is not a complete-bot turn time.

| Shape | One process | 16 processes | Within 5 ms? |
|---|---:|---:|---|
| XXL | 1.68 ms | 2.74 ms | yes, under both loads |
| 3XL | 2.93 ms | 23.04 ms | only alone |
| 4XL | 9.73 ms | 191.50 ms | no |

Sources: [isolated](../reports/host/capacity_extreme_scaling_1seat.json) and [loaded](../reports/host/capacity_extreme_scaling_16seats.json) host measurements. Shapes wider than XXL could only serve as offline teachers distilled into an XXL-or-smaller student. They can still be trained as custom widths (`--hidden-* ... --max-width N`) but are not presets.

## Train, evaluate and export

Start with the general model on a frozen dataset, then fine-tune the dedicated counts from the same-capacity general checkpoint. Warm-start shape checks reject mismatched sizes.

```sh
python -m training.train_policy --model general --backend torch --device cuda --capacity XXL --count-weighting inv_teams --out runs/policy_general_xxl
```

```sh
python -m training.train_value --model general --backend torch --device cuda --capacity XXL --count-weighting inv_teams --out runs/value_general_xxl
```

```sh
python -m training.evaluate_offline --model general --policy runs/policy_general_xxl --value runs/value_general_xxl --out reports/models/xxl/general
```

```sh
python -m training.export_weights --model general --policy runs/policy_general_xxl --value runs/value_general_xxl --report reports/models/xxl/general.json --out inference/models_xxl
```

**Testing P and V separately.** P and V can be compared independently, for example an XXL policy with an S value net: `--capacity XXL` on one trainer and the default on the other. The model id stays `general` or `nXX`, so export each configuration you want to keep into its own directory.

**Buffer width at evaluation and export.** Evaluation and export infer a rounded buffer width from the actual layer shapes (`--max-width 0`); explicit ceilings are still enforced. Generated headers record `FUSION_MAX_WIDTH_REQUIRED` and `kRequiredMaxWidth`, and manifests record `required_buffer_width`. The native inference, runtime and registry checks build with the declared width automatically, including older headers that only have the `Widest layer` comment.

**Building the bot.** The generated registry defines `FUSION_MAX_WIDTH` (2048 for XXL) before its model includes. A bot that includes the inference headers before the registry must define `FUSION_MAX_WIDTH` globally first. The native static assertions stay authoritative. Stack scratch, thread-stack limits and the 2-second startup (weights and geometry preparation) still need target-platform checks at the deployed size.

## Correctness first, host timing separately

```sh
python -m unittest -v tests.test_capacity_profiles
```

```sh
python tools/host/capacity_bench.py --verify --tiers XL XXL --teams 2 15 --verify-turns 4 --value-calls 2
```

**What these check.** Verification derives the buffer width from shape metadata without building full synthetic exports. It checks that the native full and incremental value passes agree on synthetic streams; that is correctness, not quality or latency. A separate test exports an XXL-width artifact and checks:
- integer emulator against native output;
- automatic wide compilation;
- registry dispatch;
- the allocation counter.

On the target CPU, replay real match streams under representative concurrent load:

```sh
python tools/host/capacity_bench.py --raw data/raw --tiers XL XXL --teams 2 5 8 15 --seats 16 --kernels both --value-calls 24 --tag xxl
```

Also sweep actual V call counts and complete-bot search workloads. The 50 ms complete reply and 2 s first reply are the hard limits: measure p99, max and faults, not nominal MACs.

## Deciding on capacity

**Use the same data for every size.** Compare capacities on the same independent-map subsets and objective. Plot learning curves at increasing map fractions, with repeated seeds.

**Track every quality and cost measure:**
- integer P NLL and calibration;
- action and target coverage;
- joint coordination;
- V error and branch regret;
- equal-time paired play on untouched maps and lineages.

Lower training loss alone is not enough: representation, legal support, public-history ambiguity and V's continuing-policy target can matter more than size. Keep the smallest model on the confirmed quality/latency frontier. The data-readiness findings are in [SAMPLING_DATA_READINESS_REVIEW.md](../reports/reviews/SAMPLING_DATA_READINESS_REVIEW.md); see its status note for what has since been addressed.
