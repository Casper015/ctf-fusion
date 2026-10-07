# Model matrix, artifacts and lifecycle

## 22 artifacts

| Model id | Serves team counts | Trained on | Warm start | Artifacts |
|---|---|---|---|---|
| `n02` `n03` `n04` `n05` `n06` `n07` `n08` `n10` `n12` `n15` | its own count | its own count's shards | `general` | P + V each (20) |
| `general` | 9, 11, 13, 14, and every count whose dedicated model is not exported | all counts 2–15 | none | P + V (2) |

`profiles/nXX.json` (one per count, 2–15) and `profiles/general.json` define the matrix. Code reads it through `fusion.paths.models()`, `model_counts()`, `served_counts()` and `init_model()`, so there is no per-model code. The fields are:

| Field | Meaning |
|---|---|
| `team_count`, `map_size` | the count and its board (23 / 31 / 41 / 49) |
| `model`, `weights` | artifact id and its header, `weights_<model>.hpp` |
| `init_from` | warm-start source (`general` for dedicated models) |
| `search` | **proposal only**: depth, extensions, root candidates, enemy scenarios, return-DP horizon from docs/DESIGN.md. Counts without a design row copy the nearest one (`copied_from_team_count`). The C++ search does not read these yet (`search::configFor`). |
| `turn_budget_ms` | `search` 30 (AI_Test rule), `compute_target` 35, `output_cutoff` 40, `engine_limit` 50, `first_turn_limit` 2000 |
| `collect` | phase-1 and phase-2 match counts, opponent pool, anchors |

## Pipeline per model

```text
python -m training.train_policy --model general --out runs/policy_general
python -m training.train_value  --model general --out runs/value_general
python -m training.train_policy --model n08 --init runs/policy_general --out runs/policy_n08
python -m training.train_value  --model n08 --init runs/value_general  --out runs/value_n08
python -m training.evaluate_offline --model n08 --policy runs/policy_n08 --value runs/value_n08 --out reports/n08
python -m training.export_weights   --model n08 --policy runs/policy_n08 --value runs/value_n08 --report reports/n08.json
```

`export_weights` writes into `inference/models/`:
- `weights_<model>.hpp` and its `.npz` twin;
- `MANIFEST.json`;
- `registry.hpp`, regenerated from every exported model. `fusion::models::forTeams(t)` returns the team count's dedicated model, else `general`, else `nullptr` (legacy behavior).

`tools/registry_check.cpp` prints the dispatch.

## Size and latency ceilings, and how each is checked

| Ceiling | Where it is checked |
|---|---|
| P ≤ 5 ms per turn (all opponent teams, every head at every legal move) | `tools/bench_inference.cpp` via `tools/env_check.py` on the host. About 30k multiply-accumulates per opponent team. |
| V single-digit µs per node | Same benchmark: encode + incremental accumulator + 128 → 16 → 1 tail |
| No heap allocation during turn evaluation | `tools/runtime_check.cpp` counts `operator new` calls after turn 1; the tests require 0 |
| Incremental V equals full V | `tools/runtime_check.cpp`, bit-exact on every team and turn |
| Integer C++ equals the Python emulation | `tools/infer_check.cpp` against `QuantizedTier.raw`, bit-exact |
| int32 never overflows | `fusion/quant.py` proves the accumulator bound for every input inside the feature spec |
| Tens to a few hundred KB per pair | About 30k int16 parameters per pair (~200 KiB of header text); `MANIFEST.json` records exact counts |
| Turn 1 ≤ 2 s | Model data is `constexpr`; `Board::init` runs a few BFS passes (milliseconds). The host check measures every bot's first turn. |

## Lifecycle

**Phases.** Phase 1 collects `collect.matches` (1,500) per count. Phase 2 (`collect.phase2_matches`, 7,500) continues the same deterministic fixture sequence:

```text
python -m training.collect --teams 8 --start 1500 --matches 6000
python -m training.dataset --teams 8 --append
```

`--append` processes only matches missing from `index.json` and writes new shard files beside the old ones. It refuses changed sampling options or a changed feature schema, so phase-1 samples stay valid.

**Manifests:**
- `data/shards/index.json` → `manifest`: matches per count and split, seat distribution by bot and family, AI_Test commits, collection seeds, repository commit, history of appends.
- Each run's `config.json`: model id, seed, warm-start source, the data manifest of its counts (with a digest of its match ids), repository commit, every argument.
- `inference/models/MANIFEST.json`: header and npz hashes, the full source run configs, alpha and gate per count, sizes, accumulator bounds.

**Training control:**
- cosine learning-rate schedule (`--lr`, `--min-lr`);
- on a plateau of `--patience` epochs the rate halves `--plateau-decay` times and training resumes from the best state, then stops;
- `--init` warm-starts any run from another, with shape checks;
- a temperature is fitted on validation after training.

**Safe fallback:**
1. `evaluate_offline` fits α per team count on the fit split.
2. It sets `use: true` only where the mixture beats the baseline on the evaluation split by `--min-gain` nats. The baseline is the legacy forecast, else inertia.
3. `export_weights` writes `kAlpha[t] = α` only for those counts, and 0 elsewhere. α = 0 is the legacy bot exactly.

So a model that does not beat the heuristic cannot change play. Retrain, or collect phase-2 data, and the gate re-decides.

**Retraining triggers.** Validation log-loss does not beat the baseline, or the gate is off for most served counts. Responses, in order:
1. more data (phase 2);
2. warm start from the latest `general`;
3. a lower learning rate or longer schedule;
4. keep the legacy forecast. Gate off, nothing to do.
