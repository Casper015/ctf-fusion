# Training and checking capacities beyond XXL

The faster host kernels make larger networks plausible candidates. They do not establish that extra capacity improves predictions or play. This change adds shared shape presets and propagates their buffer requirements through training, evaluation, export and native correctness tools. Existing default S architecture and the 2–15 dispatch matrix stay unchanged. No new model is enabled by selecting a preset.

## Shared presets

`fusion/capacity.py` is the source of shape definitions used by trainers and `tools/capacity_bench.py`. P and V can choose capacities independently.

| Preset | Mover hidden | Squad hidden | Action hidden | Target hidden | V hidden | Paired buffer capacity |
|---|---|---|---|---|---|---:|
| XXL | 256 | 1024,256 | 256 | 32 | 2048,128 | 2048 |
| XXXL | 512 | 2048,512 | 512 | 64 | 4096,256 | 4096 |
| XXXXL | 1024 | 4096,1024 | 1024 | 64 | 8192,512 | 8192 |

These are experiment starting points, not optimal shapes or latency approvals. Hidden-layer flags can override individual heads. Explicit undersized ceilings fail before data loading. Custom widths without a preset retain the explicit `--max-width` requirement; existing calls and defaults remain compatible.

## Latest target-host evidence

The target-host reports added at `f36e9f3` already test the same shapes, named **3XL** and **4XL** there; they correspond to **XXXL** and **XXXXL** here. They replay one 15-team real-engine stream with synthetic network weights, 24 V calls/turn, and either one or 16 concurrent benchmark processes. The reported budget estimate is `1.5 * (worst-seat P p99 + worst-seat V-turn p99)`, not an observed complete-bot turn time or a playing-strength result.

| Paired capacity | One process, estimated inference budget | 16 processes, estimated inference budget | Interpretation for the configured 5ms budget |
|---|---:|---:|---|
| XXL | 1.68ms | 2.74ms | Fits this replay under both loads |
| XXXL / 3XL | 2.93ms | 23.04ms | Fits only the isolated run |
| XXXXL / 4XL | 9.73ms | 191.50ms | Fails both loads |

Sources: [isolated raw JSON](../reports/capacity_extreme_scaling_1seat.json) and [loaded raw JSON](../reports/capacity_extreme_scaling_16seats.json). These are host measurements supplied by the repository, not new timings from this change. The 5ms inference allocation is a planning choice; the 50ms complete reply is the hard rule. XXXL exceeding 5ms alone does not prove that every possible XXXL bot exceeds 50ms, but it cannot keep the current allocation without cutting other work. Different counts, model mixes, concurrency, features and actual search nodes need their own measurements. Cache/bandwidth contention is a plausible explanation for the observed cliff; this replay has no hardware-counter evidence establishing that cause.

Treat larger paired capacities as offline teachers or bounded experimental candidates. Next test larger P for a small set of influential nearby opponents with the other P calls held small, and independently test a larger V at a few root/finalist nodes with a small V elsewhere. A trained routing/mixing contract and complete-turn checks are still required; no such scheduler is implemented by these presets. More RAM can hold all artifacts and workspaces, but does not remove cache or bandwidth limits. Compare these choices with a large teacher distilled into an XXL-or-smaller student at equal full-turn time.

Do not interpret affordable V evaluations as complete search rollouts: transition simulation, feature extraction, opponent sampling and search bookkeeping still cost time. The consolidated report's tiny RSS/stack and zero-allocation statements also do not certify all larger artifacts or a whole integrated bot. At width 8192 the three arrays in `forwardFrom` alone reserve 64KiB before caller scratch; `ValueAccumulator` has a 32KiB preactivation array. Measure complete stacks, startup, weight residency and instrumented runtime at the deployed sizes. Zero observed allocations proves only the instrumented path was allocation-free, not the absence of every possible leak.

## Train, evaluate and export

Start with a general model and a frozen dataset, then fine-tune dedicated counts from the same-capacity general checkpoint. Warm-start shape checks deliberately reject incompatible sizes.

```sh
python -m training.train_policy --model general --capacity XXXL --out runs/policy_general_xxxl
python -m training.train_value --model general --capacity XXL --out runs/value_general_xxl
python -m training.evaluate_offline --model general --policy runs/policy_general_xxxl --value runs/value_general_xxl --out reports/general_pxxxl_vxxl
python -m training.export_weights --model general --policy runs/policy_general_xxxl --value runs/value_general_xxl --report reports/general_pxxxl_vxxl.json
```

This isolates a larger P with V held at XXL. Separately test P at XXL with V at XXXL before increasing both. Use distinct run/export directories for configurations you want to retain: the model id is still `general` or `nXX`, so exporting into the same directory replaces that id's previous bundle.

The existing trainer is NumPy; these commands add architecture support, not a CUDA backend. Large full-corpus experiments should use the designated host and the planned parity-tested GPU trainer when available. Training speed, inference speed and time to equivalent validation quality are separate measurements.

Evaluation/export infer a rounded buffer ceiling from actual layer shapes by default (`--max-width 0`). Explicit ceilings are still enforced. Generated headers record `FUSION_MAX_WIDTH_REQUIRED` and `kRequiredMaxWidth`; manifests record the buffer capacity. Native inference/runtime/registry checks automatically build with the declared capacity, including compatibility with older generated width comments.

A generated registry selects the largest exported capacity before its model includes. If a bot has already included the inference headers, define `FUSION_MAX_WIDTH` globally before those headers or pass the compiler flag shown by the export. Native static assertions remain authoritative; metadata cannot bypass actual dimension checks. Large stack scratch and thread-stack limits still need target-platform checks. The 2-second startup includes model/geometry preparation; bigger weights do not relax it.

## Correctness first, host timing separately

```sh
python3 -m unittest -v tests.test_capacity_profiles
python3 tools/capacity_bench.py --verify --tiers XXXL XXXXL --teams 2 15 --verify-turns 4 --value-calls 2
```

Verification uses shape metadata without allocating or formatting full synthetic exports simply to discover the required buffer. It checks native full/incremental value agreement on synthetic streams; this is correctness evidence, not game quality or host latency. A separate >XXL exported-artifact test checks integer emulator/native output equality, automatic wide compilation, registry dispatch and the instrumented kernel's allocation counter.

On the target CPU, replay diverse real match streams with representative concurrent load:

```sh
python tools/capacity_bench.py --raw data/raw --tiers XXL XXXL XXXXL --teams 2 5 8 15 --seats 16 --kernels both --value-calls 24 --tag beyond_xxl
```

Also sweep actual V call counts and complete-bot search workloads. The inference budget is configurable; increasing it must displace measured search work or consume verified whole-turn slack. The 50ms complete reply and 2s first reply remain the hard limits. Measure p99/max and faults, not just nominal MACs. Do not extrapolate the earlier 2.83ms XXL result to XXXL/XXXXL without a new host run.

## Is the sampler sufficient?

See [SAMPLING_DATA_READINESS_REVIEW.md](../reports/SAMPLING_DATA_READINESS_REVIEW.md). The current open hierarchy is suitable for a pilot and can generate millions of usable rows. No real collected dataset or trained-run manifest was present in the reviewed checkout. Phase 1's 1,500 maps/count nominally yields approximately 1,200/150/150 train/validation/test maps before OOD and invalid-row filtering. Repeated turns are not independent map evidence.

Resolve per-count evidence admission, stale/resumed evidence identity and shard provenance before expensive collection. Inspect the **actual** seat/component distribution after depletion and lineage constraints. Declare count weights: equal games otherwise give larger counts more team-row training mass. Report rare-event coverage by distinct maps, not only rows; stride can discard isolated tactical turns.

For a scaling decision, use the same independent-map subsets and objective across capacities. Plot learning curves at increasing map fractions and repeat seeds. Track integer P NLL/calibration, conditional action/target coverage, joint coordination, V error, branch regret and equal-time paired play on untouched maps/lineages. Larger training loss gains alone are insufficient. Representation, legal support, public-history ambiguity and V's continuing-policy target can dominate capacity.

Retain the smallest model on the confirmed quality/latency frontier; larger offline teachers may be distilled into smaller CPU students. No parameter-to-row ratio can certify sufficiency for this game, and extra architecture capacity is not itself a strength gain.
