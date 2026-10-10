# ctf-fusion

Opponent policy and board value networks for the multi-team Capture The Flag bots of
[AI_Test](https://github.com/Casper015/AI_Test) (`ctf-arena/Player7`). The repository covers:
- data collection;
- training;
- calibration against the legacy heuristic;
- integer export to standalone C++ headers;
- an engine-exact forward simulator.

**First time here?** [docs/FIRST_RUN.md](docs/FIRST_RUN.md) takes a fresh checkout through setup, checks, a 2-minute pilot and the first full roster run of the 130 archived bots, step by step.

**Changing something?** [CONTRIBUTING.md](CONTRIBUTING.md) says where a file goes, what is committed, how to name branches and commits, and what must pass before a push.

**Design and decisions:**
- [docs/DESIGN.md](docs/DESIGN.md): the design brief.
- [docs/REVIEW.md](docs/REVIEW.md): decisions and experiment order.
- [training/specs/](training/specs/README.md): the contracts — features, action indexing, data format, the 22-model matrix, simulator audit, environment.
- [docs/ROSTER.md](docs/ROSTER.md): Phase A, evaluating every candidate opponent, admitting an open training pool sampled by behavior group, and freezing evaluation rosters (`roster/`).
- [reports/reviews/FINGERPRINT_EFFECTIVENESS_REVIEW.md](reports/reviews/FINGERPRINT_EFFECTIVENESS_REVIEW.md): source-backed fingerprint audit and proposed v2 plan for noise/support corrections, flag/survival/combat profiles and useful-novelty validation. Stages 0–2 (evidence identity and coverage, fixed-support distances with evidence states, live outcomes) are implemented; see [docs/ROSTER.md](docs/ROSTER.md).

**Status:**
- **Done:**
  - the pipeline, simulator and runtime contracts are built and tested on this machine (macOS), with seeded test bots and two real judge-bot matches;
  - the Phase A roster tooling ([docs/ROSTER.md](docs/ROSTER.md)): inventory of 191 candidates (122 eligible after aliases and exclusions), probe suite, behavior fingerprints, noise-corrected distances with complete-linkage groups per team count, strength screen, the open training registry with its hierarchical sampler, frozen evaluation rosters and an optional fixed panel. Fingerprints, screens and admission still run on the host;
  - the deployment contracts from the source review of 2026-10-08 ([REVIEW.md §7](docs/REVIEW.md#7-source-review-2026-10-08-and-what-changed)): binary tool output, one declared adapter scored by the gate, a fail-closed gate bound to the exact artifact, feature spec v2, strict holdouts, layer-width limits.
- **Not yet done:**
  - no model trained on real data;
  - nothing integrated into a bot;
  - no Windows run;
  - no host latency baseline. Every latency figure in this repository is an estimate until `tools/host/env_check.py` runs on the host.

The baseline comes from `tools/host/env_check.py` on the target host, never from a development machine.

## How it fits together

```text
AI_Test engine + bots ──collect.py (fusion/runner.py, Windows-safe)──▶ data/raw/nXX/*.json.gz   (protocol text + replies)
                                          │ dataset.py: C++ extractor (inference/features.hpp) + labels, --append for phase 2
                                          ▼
                                   data/shards/nXX/<split>/*.npz  (+ index.json manifest)
             heuristic_baseline.py ──▶ qh_*.npy  (legacy forecast q_H, replayed from AI_Test nextgen)
                                          │ train_policy.py / train_value.py  --model general, then --model nXX --init general
                                          ▼
                                   runs/<policy|value>_<model>/  (model.npz + config.json with data digest, seed, commits)
                                          │ evaluate_offline.py ──▶ reports/models/s/<model>.md/.json  (alpha + use gate per team count)
                                          │ export_weights.py
                                          ▼
              inference/models/weights_<model>.hpp, MANIFEST.json, registry.hpp (forTeams dispatch)
```

**The model matrix** ([MODEL_MATRIX.md](training/specs/MODEL_MATRIX.md)):
- dedicated P and V models for 2, 3, 4, 5, 6, 7, 8, 10, 12 and 15 teams;
- a general P and V, pretrained on every count, which warm-starts the dedicated models and serves 9, 11, 13 and 14 — 22 artifacts in all.

`profiles/nXX.json` and `profiles/general.json` define the matrix, so there is no per-model code.

**Safety properties, each enforced by a test:**
- **One feature implementation.** Features exist once, in C++ (`inference/features.hpp`), shared by the extractor and the bot.
- **Exact integer inference.** The network accumulators (and so the logits) are integer and bit-identical to the Python emulation; the incremental value accumulator equals the full pass. Probabilities come from `exp` in double precision, so they can differ in the last bits across platforms.
- **SIMD multiply-add kernels, same bits.** Layers run on 16-bit multiply-add kernels (`inference/simd.hpp`): AVX2 chosen at runtime on x86-64, SSE2 otherwise (every x86-64 CPU, no compiler flags), NEON on ARM, scalar elsewhere or with `-DFUSION_SIMD=0`. Integer sums are exact in any order, so every path gives the scalar loop's bits; `tools/cpp/kernel_check.cpp` checks each path the CPU runs (tests, `env_check`, `capacity_bench`).
- **Byte-exact tool output.** Raw-record tools write binary stdout on Windows too, and every record's turn and team columns are checked on read.
- **No heap use per turn.** Nothing allocates on the per-turn path after turn 1.
- **Engine-exact simulator.** `inference/sim.hpp` reproduces every recorded engine turn.
- **Safe fallback, fail closed.** The gate scores the integer model the exporter ships, through the declared adapter (`inference/adapter.hpp`), against the legacy forecast. It needs the gain to clear two map-clustered standard errors on enough maps. A team count that fails, or lacks a complete legacy forecast, exports α = 0, which is exactly the legacy bot. The report names the artifact it approves (checkpoint hashes, weight digest, schema, adapter), and the exporter refuses any other.
- **Bounded buffers.** Explicit capacity ceilings and generated native assertions enforce layer dimensions. The default is 256; named presets supply larger ceilings, evaluation/export infer actual shapes, and native artifact tools compile with the recorded capacity. A bot including inference headers before its registry must set `FUSION_MAX_WIDTH` globally first.

**Capacity experiments.** Training accepts `--capacity S|M|L|XL|XXL` ([docs/LARGER_MODEL_EXPERIMENTS.md](docs/LARGER_MODEL_EXPERIMENTS.md)); P and V can choose different presets and individual hidden flags still override them. XXL is the largest preset: wider shapes failed the inference budget under load on the host. Custom wide architectures without a preset retain explicit `--max-width` validation. The default model stays S. Wider models are experimental until trained quality and complete-turn timing are confirmed.

## Layout

Everything tracked belongs to exactly one of the folders below. **Where does my file go, and what may I commit?** See [CONTRIBUTING.md](CONTRIBUTING.md); it is enforced by `tests/test_repo_layout.py`.

```text
README.md, CONTRIBUTING.md        this map and the rules for using the repository
requirements.txt, compile_flags.txt, .gitignore, .gitattributes

── code ────────────────────────────────────────────────────────────────────────────────
fusion/           Python core: paths, runner (Windows-safe bots), native (C++ tool builds), schema, matchlog,
                  protocol, labels, nn (numpy MLP + Adam), models, quant (integer format + exact emulation),
                  mixture, metrics, capacity, torch_backend
training/         pipeline commands: collect, dataset, heuristic_baseline, train_policy, train_value,
                  evaluate_offline, export_weights, common; specs/ = the contracts (FEATURES generated, ACTIONS,
                  DATA_FORMAT, MODEL_MATRIX, SIMULATOR_AUDIT, ENVIRONMENT)
roster/           Phase A, the opponent roster: inventory, archive, probes, fingerprint, distance, strength,
                  trajectory, assess, registry, sampling, catalog, pool (docs/ROSTER.md)
  bots/           every inventoried bot source by sha256, manifest, inventory snapshot (vendored, never edit)
  new_bots/       136 more CTF bots found in AI_Test (8 independent, 128 forks), with a full-roster inventory
  report/         Phase A results (fingerprints, distances, strength screen, admission, training selection)
inference/        C++ headers that ship in a bot: state, features, simd, mlp, policy, value, adapter, sim;
                  models/ (tier S) and models_m, models_l, models_xl, models_xxl = generated weights + registry.hpp
tests/            test_math, test_contracts, test_roster, test_pipeline, test_capacity_profiles,
                  test_generate_roster_report, test_repo_layout

── configuration ───────────────────────────────────────────────────────────────────────
profiles/         n02.json ... n15.json, general.json (the model matrix), pool.json + open_training_pool*.json
                  (opponents), screen_light.json

── bots we build ───────────────────────────────────────────────────────────────────────
bots/hybrid/      C08 and Player5 v12 carriers with the mixture integration (<carrier>_Fusion.hpp + one thin
                  wrapper .cpp per policy size, control and pvl variants)
p_only/           the P-only bots: self-contained sub-project with its own include/, bots/, tools/, tests/,
                  profiles/ and report/ (p_only/README.md, p_only/TASK.md)

── tools ───────────────────────────────────────────────────────────────────────────────
tools/cpp/        C++ contract checks and benchmark drivers, compiled by fusion/native.py (infer_check, sim_check,
                  kernel_check, adapter_check, registry_check, runtime_check, size_bench, bench_inference, ...)
tools/host/       run on the target machine: env_check, capacity_bench, bench_training
tools/pipeline/   steps that produce committed inputs: train_tier_ladder, recommend_training_pool,
                  check_pilot_evidence, build_carrier_variants, gen_feature_spec, fetch_benchmark
tools/screens/    match-playing experiments: run_light_screen, run_carrier_screen, run_standard_bt_tournament,
                  run_v_ablation_screen and their smoke tests
tools/reporting/  report generators and analyses: generate_*_report, benchmark_*, duel_review, export_results, plots
tools/random_bot.py, tools/noise_bot.py   test opponents; committed pools name them by path, so they stay put

── results and documents ───────────────────────────────────────────────────────────────
reports/models/   offline gate report per model: s/ m/ l/ xl/ xxl/ (n02.md/.json ... general.md/.json, SUMMARY.md)
reports/host/     host measurements: env_check_*, capacity_*, training_bench_*, multithreading, SIMD, consolidated
reports/eval/     match results: Bradley-Terry tournament, carrier screens, V ablation
reports/reviews/  written reviews and audits (one file per review, commit hash in the name when it pins a state)
docs/             DESIGN, REVIEW, ROSTER, SOURCES, FIRST_RUN, LARGER_MODEL_EXPERIMENTS, PATH_MAP
  benchmark/      analysis of the AI_Test heuristic benchmark + games_5plus.csv.gz (cleaned per-game rows)
  handoffs/       briefs, plans and prompts written for the next agent or the host

── local only, never committed ─────────────────────────────────────────────────────────
build/  data/  runs/  runs_*/     compiled binaries, raw match logs and shards, training checkpoints
```

Moved a path you remember? [docs/PATH_MAP.md](docs/PATH_MAP.md) lists every old location.

## Requirements

**Software:**
- Python 3.10+ and numpy. PyTorch is optional: `--backend torch --device cuda` trains on the GPU with the same objective (parity-tested); see [ENVIRONMENT.md](training/specs/ENVIRONMENT.md#training-backend).
- A C++20 compiler: g++ 11+, clang 14+, or w64devkit on Windows. `CXX` overrides the choice.
- A checkout of AI_Test `main`, which provides the engine, every opponent bot, and the legacy planner for q_H.

**Placement.** Keep this repository outside OneDrive.

```bash
export AI_TEST_ROOT=~/code/AI_Test
```

## Tests (about 25 s, light load)

```bash
python -m unittest discover -s tests -t .
```

`tests/test_math.py` and `tests/test_contracts.py` need only a C++ compiler. `tests/test_pipeline.py` needs `AI_TEST_ROOT`; it plays about 20 short matches between instant test bots and checks:
- features against the engine;
- the simulator against every turn;
- labels;
- appends;
- the model matrix: general, a warm-started dedicated model, the gate report, export, registry, parity, the runtime contract, and that an export refuses a report made for another artifact or adapter;
- strict (fixture) and actor-row holdouts;
- the legacy baseline;
- learnability.

## On the Windows host

```bash
python tools/host/env_check.py --ai-test C:/path/to/AI_Test --teams 2 6 15
```

It builds every pool bot, plays headless matches under the real limits, checks the protocol, records each bot's first-turn and p50/p95/p99/max latency, replays the matches through the simulator, and times the inference. The output is `reports/host/env_check_<host>.md`.

### How big can the models be?

```bash
python tools/host/capacity_bench.py --ai-test C:/path/to/AI_Test --tiers S M L XL
```

It first checks the SIMD kernels against the scalar loop on that CPU, then replays a real-sized match for every team count 2–15, runs as many copies at once as the match has seats, and times the bot's per-turn inference (policy for every opponent team, plus value-network search nodes) for a ladder of network sizes. Each size gets a verdict against the turn's inference budget (5 ms, with a 1.5× safety factor for a slower judge machine), the number of value nodes that still fit, and the header source size it adds. The output is `reports/host/capacity_<host>.md`; its ceiling is the largest size to train. By default every size is timed twice, with the SIMD kernels and with the old scalar loop (`--kernels both`), and the report's last table gives the speedup. `tools/reporting/generate_multithreading_report.py` compares a loaded run with a `--seats 1` run, computing every statement from the two reports.

| Tier | P MACs per opponent team | V MACs per node | header per model | all 11 models | `FUSION_MAX_WIDTH` |
|---|---:|---:|---:|---:|---:|
| S (default today) | 30,096 | 8,976 | 209 KiB | 2.2 MiB | 256 |
| M | 64,288 | 22,048 | 470 KiB | 5.0 MiB | 256 |
| L | 120,576 | 60,480 | 993 KiB | 10.7 MiB | 512 |
| XL | 273,920 | 120,896 | 2.1 MiB | 23.5 MiB | 1024 |
| XXL | 678,912 | 372,864 | 5.9 MiB | 64.4 MiB | 2048 |

These sizes need no timing. The timing columns come only from the host. `--verify` checks correctness on any machine, without timing. A tier above width 256 is trained and exported with `--max-width` and needs the bot built with `-DFUSION_MAX_WIDTH`; the exported header refuses to compile otherwise.

## Full run

```bash
# E0: data (phase 1: 1,500 matches per count; profiles/ define pools). Resumable, deterministic fixtures.
python -m training.collect --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 8
python -m training.dataset --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 8
python -m training.heuristic_baseline --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --split val test

# E1: general model, then dedicated models warm-started from it. Start with a few representative ones
#     (n02, n05, n15); train the whole matrix only once they beat the legacy forecast on held-out data,
#     strict holdout included.
python -m training.train_policy --model general --out runs/policy_general
python -m training.train_value  --model general --out runs/value_general
for m in n02 n03 n04 n05 n06 n07 n08 n10 n12 n15; do
  python -m training.train_policy --model $m --init runs/policy_general --out runs/policy_$m
  python -m training.train_value  --model $m --init runs/value_general  --out runs/value_$m
done
for m in general n02 n03 n04 n05 n06 n07 n08 n10 n12 n15; do
  # evaluate_offline quantizes the runs itself and gates the integer model; export must use the same --adapter
  python -m training.evaluate_offline --model $m --policy runs/policy_$m --value runs/value_$m --out reports/models/s/$m
  python -m training.export_weights   --model $m --policy runs/policy_$m --value runs/value_$m --report reports/models/s/$m.json
done

# Phase 2 for a count that needs more data (keeps every phase-1 shard)
python -m training.collect --teams 8 --start 1500 --matches 6000 --workers 8
python -m training.dataset --teams 8 --append
```

**Options:**
- `--workers`: every match runs one process per team, all searching at once. Keep `workers × teams` at or below the physical core count, or collection causes the timeouts it records.
- `--turn-ms` relaxes the limit on an oversubscribed machine. Faults are recorded either way, and missing replies never become labels.
- `--holdout-family <family>` in `dataset.py` keeps one opponent family out of training, for an out-of-distribution report. `--holdout-mode fixture` (default) reserves whole matches with that family; `actor-row` only that family's own rows, a weaker test. Splits group by map (`--split-by map`).
- `evaluate_offline.py`: `--adapter` (`joint-v2` default, or `marginal-v1`), `--min-gain`, `--min-maps` (20), `--z` (2).

**Decision rule.** A count uses q_θ only where its report gate says `use`. Then compare `Player5_Omega_LIT` with and without it in paired matches, per size ([REVIEW.md](docs/REVIEW.md) §4–5).

## Rules carried over from AI_Test

- **Budget.** The search keeps its 30 ms budget, inside the 35 ms compute target and the 40 ms output cutoff of `turn_budget_ms`. Inference must fit inside it.
- **Where to test.** No latency tests or large match batches on development machines; the host and cluster measure strength and timing.
- **Language.** Everything in this repository is English.
