# Using this repository

The rules for working in `ctf-fusion`: where a file goes, what is committed, how branches and commits are named, and what must pass before a push. The map of folders is in the [README](README.md#layout). `tests/test_repo_layout.py` enforces the structural rules below.

## The short version

1. **Never commit to `main`.** Branch, commit, open a pull request.
2. **One folder per kind of thing** (table below). Don't add files to the repository root, and don't create a new top-level folder without updating the test and the README in the same commit.
3. **Source and small results are committed. Binaries, raw logs and checkpoints are not.**
4. **Generated files are regenerated, never hand-edited.**
5. **Run the tests before you push:** `python -m unittest discover -s tests -t .`
6. **Move files with `git mv`, then fix every reference** and re-run the tests.

## Where things go

| You have | It goes in | Notes |
|---|---|---|
| Library code used by several commands (parsing, models, quantization, paths) | `fusion/` | Importable package. No scripts here. |
| A pipeline command (`python -m training.xxx`) | `training/` | Contracts and specs: `training/specs/`. |
| Opponent-roster logic (inventory, fingerprints, strength, pool) | `roster/` | Results of a roster run go in `roster/report/`. |
| C++ that ships inside a bot (headers) | `inference/` | The include root. See [Frozen paths](#frozen-paths). |
| A C++ check or benchmark driver that is not shipped | `tools/cpp/` | Compiled through `fusion/native.py`. |
| A script you run on the target host (timing, capacity) | `tools/host/` | Writes to `reports/host/`. |
| A script that produces a committed input (pools, specs, trained tiers) | `tools/pipeline/` | |
| A script that plays matches for an experiment | `tools/screens/` | Writes to `reports/eval/`. |
| A script that turns results into a report or plot | `tools/reporting/` | |
| A model matrix, search budget or opponent pool definition | `profiles/` | JSON only. |
| A bot built on a carrier with the mixture integration | `bots/hybrid/` | |
| Work on the P-only bots | `p_only/` | Self-contained: its own `tools/`, `tests/`, `report/`. |
| A unit or contract test | `tests/` | One `test_<area>.py` per area. |
| Offline gate report of one model | `reports/models/<tier>/` | Written by `training.evaluate_offline`. |
| A host measurement | `reports/host/` | Name ends in the host: `capacity_<HOST>.json`. |
| A match-result summary | `reports/eval/` | |
| A written review or audit | `reports/reviews/` | One file per review; put the commit hash it pins in the name. |
| A design note, plan, or decision | `docs/` | Briefs for the next agent: `docs/handoffs/<topic>/`. |
| Analysis of the AI_Test heuristic benchmark | `docs/benchmark/` | |

If nothing fits, the file probably doesn't belong in the repository. Ask before inventing a folder.

## Committed, and not committed

**Committed:** source, tests, profiles, generated model artifacts under `inference/models*/` (with their `MANIFEST.json`), and results as `.md`, `.json` or `.jsonl` under `reports/`.

**Never committed** (all are in `.gitignore`, and rebuilt or re-collected on demand):

| Path | Holds |
|---|---|
| `build/` | compiled tool and bot binaries, named by source hash |
| `data/` | raw match records, dataset shards, benchmark downloads, screen results |
| `runs/`, `runs_*/` | training checkpoints |
| `__pycache__/`, `.DS_Store` | tooling noise |

Other rules:
- **Size.** Keep a tracked file under 50 MB. GitHub rejects 100 MB, and a file that large stays in the history forever. Compress (`.gz`) or summarize instead.
- **No secrets, no host paths.** Don't commit tokens, a user's home directory or an `AI_TEST_ROOT`. Paths in code are relative to the repository or come from `--ai-test` / `AI_TEST_ROOT`.
- **Line endings.** LF everywhere (`.gitattributes`). Binary suffixes (`.npz`, `.npy`, `.gz`) are marked binary there.
- **Keep the working copy outside OneDrive.** Sync clients corrupt build caches and partially written shards.

## Generated files: don't edit by hand

| File(s) | Generator |
|---|---|
| `inference/models*/weights_*.hpp`, `.npz`, `registry.hpp`, `MANIFEST.json` | `python -m training.export_weights` |
| `inference/models_p<tier>_v<tier>/` (mixed P/V sets: exact copies of committed arrays) | `python -m training.export_mixed --policy-tier <t> --value-tier <t>` (`tests/test_mixed_export.py` re-derives and compares) |
| `training/specs/FEATURES.md` | `python tools/pipeline/gen_feature_spec.py` (a test fails when it drifts) |
| `bots/hybrid/*_Fusion.hpp` variants | `tools/pipeline/build_carrier_variants.py` |
| `p_only/bots/c08_ponly.hpp`, `v12_ponly.hpp`, `c08_control.hpp` | `p_only/tools/derive/make_*.py` (a test re-derives and compares) |
| `reports/**`, `docs/benchmark/*TABLES.md`, `roster/report/**` | the script named in the file's header |
| `roster/bots/src/**` | vendored sources, named by sha256: never edit |

To change a generated file, change its generator, regenerate, and commit both together.

## Frozen paths

These locations are named by committed hashes, manifests or the host's build scripts. Renaming one silently breaks provenance, so don't, unless you regenerate everything that cites it in the same change:

- `inference/` as the include root (`-Iinference`), and `inference/models`, `models_m`, `models_l`, `models_xl`, `models_xxl` (bots include `models_m/registry.hpp`).
- `tools/random_bot.py` and `tools/noise_bot.py` (pools and roster results name them as `@repo/tools/...`).
- `profiles/pool.json`, `profiles/open_training_pool*.json`, `profiles/n*.json`, `profiles/general.json`.
- Everything under `roster/bots/` and `roster/new_bots/`.

Result files record the paths of the day they were made. [docs/PATH_MAP.md](docs/PATH_MAP.md) maps the old layout to the current one.

## Branches, commits, pull requests

**Branch names** are `<area>/<short-topic>`, lowercase, hyphenated: `docs/model-export-review`, `models/tier-xl-export`, `chore/reorganize-repo`, `fix/gate-adapter-mismatch`. Start from an up-to-date `main`. Keep a branch to one topic.

**Commit messages** start with the area, then an imperative summary, 72 characters or fewer:

```text
train: complete Tier XL and XXL training on CUDA
p_only: consistent second-turn states, 2-ply scripted control
docs: review trained model exports and native artifact correctness
```

Areas in use: `docs`, `train`, `training`, `models`, `eval`, `bench`, `roster`, `p_only`, `handoff`. Use `chore` for moves and cleanups. Put the why in the body when it isn't obvious from the diff. One logical change per commit; a pure move (`git mv` plus reference fixes) never also changes behavior.

**Pull requests** target `main`, say what changed and how it was checked, and link the report or review that justifies a model or bot change. Merge with a merge commit, as the history already does. Don't force-push a branch someone else has pulled, and never force-push `main`.

**Trained models and bots.** A change to `inference/models*/` ships with the offline gate report it was exported from (`reports/models/<tier>/`) in the same pull request. A bot variant that goes to the host ships with its screen result under `reports/eval/` or `p_only/report/`.

## Before you push

```bash
export AI_TEST_ROOT=~/code/AI_Test                 # without it the end-to-end pipeline tests are skipped
python -m unittest discover -s tests -t .          # ~10 s without AI_Test, ~25 s with it
python -m unittest discover -s p_only/tests        # ~20 s, when you touched p_only/ or bots/hybrid/
```

- A moved or renamed file: `git grep -n "<old path>"` returns nothing outside `docs/PATH_MAP.md`, frozen results and vendored sources.
- A new top-level folder, `tools/` subfolder or `reports/` subfolder: update `tests/test_repo_layout.py` and the README layout in the same commit.
- Timing and strength numbers come from the host, never from a development machine. Don't commit a latency figure measured here.

## Rules carried over from AI_Test

- **Budget.** The search keeps its 30 ms budget inside the 35 ms compute target and the 40 ms output cutoff. Inference must fit inside it.
- **Where to test.** No latency tests or large match batches on development machines; the host and cluster measure strength and timing.
- **Language.** Everything in this repository is English.
