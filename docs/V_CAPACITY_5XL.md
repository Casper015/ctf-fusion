# T0016: explicit 5XL value capacity experiment

This is a separate experiment from T0012's four-arm feature pilot. Production
remains the committed `inference/models_pxxl_v3xl` reference. The experiment
uses v3's unchanged 108 inputs, p2 XXL arrays, the same C08 family carrier,
search depth, candidate generation, heuristic and initial V coefficients.
No P3, AMP, TF32 or architecture change is part of this comparison.

| Name | Training dimensions | Training parameters | Outcome-only MACs |
|---|---|---:|---:|
| XL | 108 / 1024 / 64 / 3 | 177,411 | 176,192 |
| 4XL | 108 / 8192 / 512 / 3 | 5,089,283 | 5,079,552 |
| 5XL | 108 / 16384 / 1024 / 3 | 18,567,171 | 18,547,712 |

These counts come from the actual `ValueNet` matrices, not preset names.
`5XL` is an experimental shape passed with `--hidden` and `--max-width`.
The dense tail alone has 16,777,216 MACs; a first-layer cache cannot remove it.

## Source and interpretation

The first bounded run reuses `data/vcache/designated_v3`, source dataset digest
`ccf6f46945b83ad3`: 390,216 train, 52,728 validation and 32,703 test rows.
It is a **mixed-policy exploratory experiment** from `screen_ood_20261102`.
Designated selects the candidate seat; the candidate binary identities define
a mixture of C08 and V12 controls and P variants, including A/A duplicates.
Every capacity uses these identical rows, targets, source weights, map splits
and mixture definition. Fresh XL is retrained because the bounded procedure
and validation batch differ from the external reference's original run.

`training.value_capacity_audit` records raw-file and cache digests, split map
membership, carrier and opponent identities, collection seeds, faults and
branch prefix mismatches. `cache_identity` refuses maps crossing splits.
Test membership and hashes are audited; test labels are not scored by this
pilot. The existing test maps and reserved opponent families have already
appeared in prior work, so they cannot substitute for fresh confirmation.

Existing branch data uses single-unit movement deviations, mostly with one
continuation each. Such labels are noisy estimates. The optional
`branch_collect --candidate-mode complete` adds bounded coordinated movements,
alternate/focused attack allocations, pickup/wait alternatives and carrier
drop plans. It does not alter any deployed carrier's candidate generation.
Use a new output directory and real repeated continuations for that step.

## Integer and storage contracts

Width 16384 follows the existing explicit ceiling, generated static assertions
and family checks. Quantization selects per-layer exponents using the actual
weights and feature bounds and proves each accumulator stays inside ±2^30.
The wider shape does not require weakening a check or changing integer
semantics. Report trained-model bounds and errors separately from shape tests.

`tests.test_value_5xl` instantiates all 18,567,171 parameters, quantizes the
outcome head, independently recomputes bounds, and compares Python integer
inference with native full and incremental passes, both with and without the
transposed first layer. Its deterministic synthetic fixture requires maximum
logit quantization error below 0.1; this is not a trained-model quality gate.
The native executable starts and loads embedded weights on Windows's default
stack. Full deployed-carrier startup and deadlines need separate measurements.

The generic MLP scratch is 8 bytes per width: 131,072 bytes at 16384. A wide
build dispatches small P networks to a separate 2048-width scratch frame
(16,384 bytes), retaining identical kernels and MAC counts. Scratch buffers
are not zeroed; v3 accumulators initialize their active rows on refresh, behind
the valid flag. Value's outer buffers/accumulator add further stack storage,
so native startup and carrier tests remain mandatory.

`export_mixed --value-run` writes one shared V for all registries, exact copies
of frozen P arrays and one shared NPZ. Large initializer arrays are split into
deterministic fragments of at most 500,000 literals, each below 50 MB. Every
fragment's hash, size, array offset and element count is bound into each
entry's `shared_value.chunks`. Regeneration removes only stale generated
fragments. Native cache identities include all local transitive dependencies;
editing a shared V or fragment invalidates the binary. Bundling follows these
includes normally; the large single-file judge bundle stays under `build/`.

## Reproducible bounded pilot

Run from the T0016 worktree using a verified CUDA Python. Set truthful
`CTF_AGENT=codex@desktop-g60ojud`, `CTF_CAN=host,cuda,ai-test`, and `AI_TEST_ROOT`.
Acquire the repository's default machine-wide locks. Do not pull, rebuild or
clean while collecting evidence. Commands below use the existing pipeline.
Put `--task` before the lock name: the current task parser captures everything
after that positional name as the child command. For timing isolation, nest
the CPU lock around the GPU lock while running measured sweeps/training.

```powershell
python -m training.value_capacity_audit --cache data/vcache/designated_v3 --index data/dataset_designated_index.json --matches <source-matches> --branches data/branches/c08_v3xl --out reports/models/v_capacity_5xl/source_audit.json
python tools/repo/task.py lock --task T0016 gpu -- python -m training.value_sweep --cache data/vcache/designated_v3 --value-family v3 --hidden 16384,1024 --batches 1024 2048 4096 --warmup 5 --steps 30 --gpu-resident-gb 0 --gpu-window-rows 65536 --device cuda --out reports/models/v_capacity_5xl/sweep_5xl_DESKTOP-G60OJUD.json
```

Repeat the measured sweep for `1024,64` and `8192,512` with separate output
names. The sweep starts each batch with a fresh model and optimizer, releases
previous device objects, and measures total allocated/reserved VRAM through
training and a full validation pass. A 2048 × 16384 FP32 activation is
134,217,728 bytes (128 MiB); autograd, optimizer, windows and workspaces are
additional. Select a batch from measured memory, throughput and subsequent
validation-quality curves; use the same batch and validation batch in all arms.

For each shape and its own output directory, fresh initialization, seed 1:

```powershell
python tools/repo/task.py lock --task T0016 gpu -- python -m training.train_value --model general --cache data/vcache/designated_v3 --value-family v3 --hidden 16384,1024 --max-width 16384 --backend torch --device cuda --epochs 30 --patience 4 --plateau-decay 1 --lr-schedule cosine --lr 1e-3 --min-lr 1e-4 --batch <B> --val-batch <B> --gpu-resident-gb 0 --gpu-window-rows 65536 --seed 1 --checkpoint --out runs/v_capacity_5xl/5XL_s1
python tools/repo/task.py lock --task T0016 gpu -- python -m training.distill_value --teacher runs/v_capacity_5xl/5XL_s1 --teacher-batch 2048 --cache data/vcache/designated_v3 --soft-weight 0.5 --hidden 1024,64 --max-width 16384 --backend torch --device cuda --epochs 30 --patience 4 --plateau-decay 1 --lr-schedule cosine --lr 1e-3 --min-lr 1e-4 --batch <B> --val-batch <B> --gpu-resident-gb 0 --gpu-window-rows 65536 --seed 1 --checkpoint --out runs/v_capacity_5xl/Student_s1
python -m training.evaluate_value --cache data/vcache/designated_v3 --split val --batch 2048 --run Reference=runs/reference_v3xl --run XL=runs/v_capacity_5xl/XL_s1 --run 4XL=runs/v_capacity_5xl/4XL_s1 --run 5XL=runs/v_capacity_5xl/5XL_s1 --run Student=runs/v_capacity_5xl/Student_s1 --rank data/rank/c08_v3xl --compare 4XL:XL --compare 5XL:XL --compare 5XL:4XL --compare 5XL:Reference --compare Student:XL --compare Student:5XL --out reports/models/v_capacity_5xl/validation
python -m training.export_mixed --policy-tier xxl --value-run runs/v_capacity_5xl/5XL_s1 --max-width 16384 --out build/v_capacity_5xl/sets/5XL
python tools/host/value_cost.py --set build/v_capacity_5xl/sets/5XL --verify
python tools/repo/task.py lock --task T0016 cpu -- python tools/host/value_cost.py --set build/v_capacity_5xl/sets/5XL --calls 1 --copies 1 6 15 --out reports/models/v_capacity_5xl/value_cost_DESKTOP-G60OJUD
python -m training.prepare_value_capacity --out build/v_capacity_5xl
python -m training.check_value_capacity --set build/v_capacity_5xl/sets/5XL --run runs/v_capacity_5xl/5XL_s1 --cache data/vcache/designated_v3 --out reports/models/v_capacity_5xl/integer_5XL.json
python bots/p_only/tools/build.py --stats --variants build/v_capacity_5xl/variants.json
python tools/repo/task.py lock --task T0016 cpu -- python bots/p_only/tools/run_screen.py --config build/v_capacity_5xl/screen_time.json
python tools/repo/task.py lock --task T0016 cpu -- python bots/p_only/tools/run_screen.py --config build/v_capacity_5xl/screen_work.json
python tools/repo/task.py lock --task T0016 cpu -- python -m training.branch_collect --matches data/v_capacity_5xl/branch_roots --carrier cap_xl_time --out data/branches/v_capacity_5xl_complete_isolated --roots-per-match 1 --candidates 6 --candidate-mode complete --repeats 2 --workers 4
python -m training.branch_dataset --branches data/branches/v_capacity_5xl_complete_isolated --out data/rank/v_capacity_5xl_complete_isolated
python -m training.summarize_value_capacity --build build/v_capacity_5xl --out reports/models/v_capacity_5xl
```

The existing screen analyzer runs into ignored `build/` reports. With only one
map per count, its within-count intervals cannot estimate variance. The summary
supplement binds the schedule, binary identities and result digest, refuses
stale or duplicate rows, and computes paired utility differences over the four
independent maps with a small-sample t multiplier. It records game-level full
reply percentiles, `VALUE`, `VBUDGET` and `SEARCH` counters, and observed deadline
status. None of these small screens establishes game-strength superiority.

For `branch_roots`, copy only the four base `cap_xl_time` match logs, preserving
their `n02/n04/n08/n15` directories. The filename's arm key is the first ten
hex characters of SHA-256(`cap_xl_time`). A/A copies are not additional maps.
The retained complete-order collection runs with no concurrent checks; an
earlier plumbing run is not retained as host evidence. Its continuation policy
is the exact matched C08/p2-XXL/XL binary, eight calls and 20 ms, with the same
opponent sources and exploratory seed. This is distinct from the mixed-policy
training target. The summary audits outcomes, repetitions, faults, prefix
mismatches, available order coverage and map splits without scoring test models.

The student is experimental even if its teacher has lower validation loss.
Distillation retains v3 and combines real outcomes with teacher targets.
Ranking supervision is off in the capacity arms and first student, preserving
the primary controlled comparison. Teacher targets bind cache and checkpoint
digests; validation/calibration always use real outcomes.

Learning curves, best checkpoint, optimizer steps, training/validation time,
fit time, calibration time and pipeline time are recorded separately. Equal
epochs are not equal compute or evidence of equal convergence.

## Decision and deployment gates

The evaluation reports log loss, Brier, calibration, phase/count metrics,
pair accuracy with a map-clustered interval, random/majority baselines, selected
candidate regret and per-root paired regret differences against the incumbent.
Cross-model comparisons also subtract per-root regrets before computing the
map-clustered interval. An absolute interval for regret cannot establish an
advantage. Small map counts, unresolved pairs, noisy single continuations and
prefix mismatches limit the exploratory ranking data.

Route A uses the existing `PONLY_V_MAX_CALLS` and `PONLY_V_DEADLINE_MS` checks.
The deadline is checked **before** a network call, not an interruptible tail:
reserve the measured worst-call cost. A cut decision falls back consistently
to its no-V choice; partially scored candidates must never be compared against
unscored candidates. These bounds can consume time without changing a decision
if they cut a path, so record fallback counts and displaced search samples.

Route B is the distilled student, with the same carrier and frozen P. Strictly
matched V-off/XL/4XL/direct 5XL/student builds and identical A/A copies need
both fixed-work diagnostics and equal-full-turn-time screens. Measure 2 s
startup / 50 ms subsequent replies, p50/p99/max, faults, V time/calls, cuts,
search samples and truncations. `value_cost` now feeds multi-copy processes
concurrently; earlier sequential-feed multi-copy numbers need remeasurement.

Seed `20261220` was reserved and consumed by the tiny exploratory screen in T0016.
T0019 was subsequently created with the same seed labelled fresh; the collision
is recorded in its task log. It must choose and reserve another unseen seed
before host execution. Previously
used `20261031`, `20261102`, `20261107`, `20261108`, and other tasks' reserved
`20261201`, `20261210`, `20261211` are not fresh confirmation seeds. Freeze
settings on validation/exploration first, collect one specified C08 continuation
policy if C08 is selected, repeat finalists with three independent training
seeds, and run the repository's 10,000 independent map-group confirmation with
at least 500 maps per count 2–15. Leave its seed unchosen here.

Evidence is separated into implementation, training, integer-export verification,
CPU deadlines, decision improvement and game-strength improvement. A confidence
interval crossing zero means **not shown**. No successful training run, lower
training loss or single screen ranking authorizes a production replacement.
