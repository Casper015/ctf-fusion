# ctf-fusion

Opponent policy and board value networks for the multi-team Capture The Flag bots of
[AI_Test](https://github.com/Casper015/AI_Test) (`ctf-arena/Player7`). The repository covers:
- data collection;
- training;
- calibration against the legacy heuristic;
- integer export to standalone C++ headers.

[docs/DESIGN.md](docs/DESIGN.md) is the design brief. [docs/REVIEW.md](docs/REVIEW.md) explains what was built, what changed from the brief and why, and the experiment order.

**Status:**
- **Done and tested:** the pipeline runs end to end. It has been tested on seeded random bots and on two matches between real judge bots.
- **Not done yet:**
  - no model has been trained on real data;
  - nothing is integrated into a bot;
  - no latency has been measured.

Strength and timing are measured on the external cluster, never on a development machine.

## How it fits together

```text
AI_Test engine + bots ──collect.py──▶ data/raw/nXX/*.json.gz   (protocol text + reply lines, lossless)
                                          │ dataset.py: C++ extractor (inference/features.hpp) + labels
                                          ▼
                                   data/shards/<tier>/<split>/*.npz
             heuristic_baseline.py ──▶ qh_*.npy  (legacy forecast q_H, replayed from AI_Test nextgen)
                                          │ train_policy.py / train_value.py   (numpy, CPU)
                                          ▼
                                   runs/<name>/model.npz
                                          │ evaluate_offline.py ──▶ reports/<tier>.md + alpha per team count
                                          │ export_weights.py
                                          ▼
                     inference/models/<tier>.hpp   (int16 constexpr arrays, std-only C++20)
```

**One feature implementation.** Features exist once, in C++ (`inference/features.hpp`). The training extractor and the bot run the same code, so they cannot disagree.

**Exact integer inference.** Inference is integer end to end, and `tests/test_pipeline.py` proves the C++ output bit-identical to the Python emulation.

**Tiers.** Models are trained per map size, with team count as an input:

| Tier | Map | Team counts |
|---|---|---|
| `m23` | 23 × 23 | 2 |
| `m31` | 31 × 31 | 3–5 |
| `m41` | 41 × 41 | 6–10 |
| `m49` | 49 × 49 | 11–15 |

The mixture weight α is fitted per team count.

## Layout

```text
docs/        DESIGN (brief), REVIEW (decisions + plan), FEATURES (spec v1), DATA_FORMAT, SOURCES
profiles/    nXX.json per team count (tier, collection plan), pool.json (opponent bots and families)
fusion/      library: paths, native (C++ tool builds), schema, matchlog, protocol, labels,
             nn (numpy MLP + Adam), models, quant (integer format + exact emulation), mixture, metrics
training/    collect, dataset, heuristic_baseline, train_policy, train_value, export_weights, evaluate_offline
inference/   state.hpp, features.hpp, mlp.hpp, policy.hpp, value.hpp; models/<tier>.hpp (generated)
tools/       extract_features.cpp, infer_check.cpp (parity driver), stream.hpp (protocol parser), random_bot.py
tests/       test_math (gradients, quantization, convex fits), test_pipeline (engine cross-checks, parity)
```

## Requirements

- Python 3.10+ and numpy. PyTorch is optional; the models are small enough for numpy on CPU.
- A C++20 compiler: g++ 11+ or clang 14+; `CXX` overrides the choice.
- A checkout of AI_Test `main`, which provides the engine, the match runner, every opponent bot and the legacy planner for q_H. Set its path once:

```bash
export AI_TEST_ROOT=~/code/AI_Test
```

## Smoke test (about a minute, light load)

Seeded random bots answer in microseconds, so this checks the whole chain without any timing pressure:

```bash
python -m unittest discover -s tests -t .
```

The same steps by hand:

```bash
python -m training.collect --teams 2 3 4 --matches 10 --bots Random --out data/smoke/raw
python -m training.dataset --raw data/smoke/raw --out data/smoke/shards --teams 2 3 4 --stride 1
python -m training.heuristic_baseline --shards data/smoke/shards --tier m31 --split val test
python -m training.train_policy --shards data/smoke/shards --tier m31 --out runs/smoke_policy --epochs 2
python -m training.train_value --shards data/smoke/shards --tier m31 --out runs/smoke_value --epochs 2
python -m training.evaluate_offline --shards data/smoke/shards --tier m31 --policy runs/smoke_policy --value runs/smoke_value --out reports/smoke_m31
python -m training.export_weights --tier m31 --policy runs/smoke_policy --value runs/smoke_value --alpha reports/smoke_m31.json --out data/smoke/models
```

Random bots are unpredictable, so their policy log-loss sits at the uniform value; the smoke test checks plumbing, not learning.

## Full run (cluster)

```bash
# E0: data. 1,500 matches per team count from profiles/; resumable; fixtures are deterministic.
python -m training.collect --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 32
python -m training.dataset --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 32
for t in m23 m31 m41 m49; do python -m training.heuristic_baseline --tier $t --split val test; done

# E1: offline models, per tier
for t in m31 m41 m49; do
  python -m training.train_policy --tier $t --out runs/policy_$t
  python -m training.train_value  --tier $t --out runs/value_$t
  python -m training.evaluate_offline --tier $t --policy runs/policy_$t --value runs/value_$t --out reports/$t
  python -m training.export_weights --tier $t --policy runs/policy_$t --value runs/value_$t --alpha reports/$t.json
  python -m training.evaluate_offline --tier $t --policy runs/policy_$t --value runs/value_$t \
      --quantized inference/models/$t.npz --out reports/$t
done
```

**Other options:**
- `--workers`: every match runs one process per team, all searching at once. Keep `workers × teams` at or below the physical core count, or collection itself causes the timeouts it records.
- `--turn-ms` relaxes the per-turn limit during collection if the machine is oversubscribed. Faults are recorded either way, and missing replies never become labels.
- `--holdout-family <family>` in `dataset.py` keeps one opponent family out of training, for an out-of-distribution report.

**The decision rule:** integrate q_θ for a team count only where `reports/<tier>.md` shows it beating the legacy forecast. Then compare `Player5_Omega_LIT` with and without it in paired matches, per size ([REVIEW.md](docs/REVIEW.md) §4–5).

## Rules carried over from AI_Test

- **Budget.** The search keeps its 30 ms per-turn budget; inference must fit inside it, not extend it.
- **Where to test.** No latency tests or large match batches on development machines; the cluster measures strength and timing.
- **Language.** Everything in this repository is English.
