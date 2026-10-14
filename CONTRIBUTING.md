# Using this repository

The rules for working in `ctf-fusion`: where a file goes, what is committed, how branches and commits are named, and what must pass before a push. How several people and agents share the work (the task board, worktrees, lanes, locks) is in [AGENTS.md](AGENTS.md). The map of folders is in the [README](README.md#where-things-are). `tests/test_repo_layout.py` and `tests/test_task_board.py` enforce the structural rules.

## The short version

1. **Every piece of work is a task on the [board](tasks/README.md).** Agents claim it, work in their own worktree and branch, and land with `task.py land` or a pull request ([AGENTS.md](AGENTS.md)). Humans may commit a small, self-contained change straight to `main` when no active task writes those paths: check `python tools/repo/task.py list` first.
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
| Work on the P-only bots | `bots/p_only/` | Self-contained: its own `tools/`, `tests/`, `report/`. |
| A unit or contract test | `tests/` | One `test_<area>.py` per area. |
| Offline gate report of one model | `reports/models/<tier>/` | Written by `training.evaluate_offline`. |
| A controlled comparison of models (a study) | `reports/models/<study>/` | Register the folder in `MODEL_STUDIES` (`tests/test_repo_layout.py`): `v_pilot`, `p_features` ([FEATURE_FAMILIES.md](docs/FEATURE_FAMILIES.md)), `v_capacity_5xl` ([V_CAPACITY_5XL.md](docs/V_CAPACITY_5XL.md)). |
| A host measurement | `reports/host/` | Name ends in the host: `capacity_<HOST>.json`. |
| A match-result summary | `reports/eval/` | |
| A written review or audit | `reports/reviews/` | One file per review; put the commit hash it pins in the name. |
| A design note, plan, or decision | `docs/` | |
| Work to do, who is doing it, what the next person needs to know | `tasks/` | Through `tools/repo/task.py`. No separate "prompt for the next agent" files: write the task's Handoff instead. |
| Repository housekeeping (the task board, bot bundling) | `tools/repo/` | |
| A CI or release workflow | `.github/workflows/` | Pin actions by SHA; see [Automatic checks](#automatic-checks-ci). |
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
| `dist/` | bot bundles written by `tools/repo/release_bots.py` |
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
| `inference/models_p<tier>_v<tier>/` and other mixed P/V sets (exact copies of committed arrays) | `python -m training.export_mixed --policy-tier <t> --value-tier <t>`, or from training runs: `--policy-run` (one model, `general` serving every count) and `--value-run` (quantized once, one V header shared by every model of the set). `tests/test_mixed_export.py` re-derives and compares |
| `training/specs/FEATURES.md` | `python tools/pipeline/gen_feature_spec.py` (a test fails when it drifts) |
| `bots/hybrid/*_Fusion.hpp` variants | `tools/pipeline/build_carrier_variants.py` |
| `bots/p_only/bots/c08_ponly.hpp`, `v12_ponly.hpp`, `c08_control.hpp` | `bots/p_only/tools/derive/make_*.py` (a test re-derives and compares) |
| `bots/p_only/include/ponly_p3.hpp`, `bots/p_only/bots/v12_ponly_f.hpp`, `c08_ponly_f.hpp` (the feature-family carriers, [FEATURE_FAMILIES.md](docs/FEATURE_FAMILIES.md)) | `python bots/p_only/tools/derive/make_families.py` (`bots/p_only/tests/test_families.py` re-derives and compares) |
| `reports/**`, `docs/benchmark/*TABLES.md`, `roster/report/**` | the script named in the file's header |
| `roster/bots/src/**` | vendored sources, named by sha256: never edit |
| `bots/p_only/include/ponly_precision.hpp`, `ponly_p3_precision.hpp` (precision-aware views, task T0030; [P_MIXED_PRECISION.md](docs/P_MIXED_PRECISION.md)) | `python bots/p_only/tools/derive/make_families.py` (`bots/p_only/tests/test_precision_carriers.py` re-derives and compares) |
| `bots/p_only/bots/v12_ponly_fs.hpp` (the v12 own-action scoring carrier, task T0045; [OWN_ACTION_RUNTIME.md](docs/OWN_ACTION_RUNTIME.md)) | `python bots/p_only/tools/derive/make_scoring.py` after `make_families.py` (`bots/p_only/tests/test_decision_scoring.py` re-derives and compares) |
| `inference/own_action_scorers/<name>/` (exported own-action scorers, never P or V) | `python -m training.export_decision_scorer` |
| `build/p_precision/**` (experimental low-bit P exports, never committed) | `python -m training.export_precision export ...` |
| `tasks/T*.md` | `tools/repo/task.py`. Humans may edit by hand; run `task.py check` before pushing. |
| `tasks/SUMMARY.md` | `tools/repo/task.py`, with every board change (`task.py summary --write` after a hand edit of a task) |
| `docs/MARKDOWN_INDEX.md` | `python tools/repo/md_index.py --write` after adding, moving, removing or retitling a Markdown file. CI warns when it is stale. The summaries in [docs/DIGEST.md](docs/DIGEST.md) are written by hand; give a new topic a section with a `**Sources:**` line. |

To change a generated file, change its generator, regenerate, and commit both together.

## Frozen paths

These locations are named by committed hashes, manifests or the host's build scripts. Renaming one silently breaks provenance, so don't, unless you regenerate everything that cites it in the same change:

- `inference/` as the include root (`-Iinference`), and `inference/models`, `models_m`, `models_l`, `models_xl`, `models_xxl` (bots include `models_m/registry.hpp`).
- `tools/random_bot.py` and `tools/noise_bot.py` (pools and roster results name them as `@repo/tools/...`).
- `profiles/pool.json`, `profiles/open_training_pool*.json`, `profiles/n*.json`, `profiles/general.json`.
- Everything under `roster/bots/` and `roster/new_bots/`.
- The local folders `build/p_only/` and `data/p_only/`, which keep their names although the code moved to `bots/p_only/` (the host's runs live there). `bots/p_only/tools/bundle.py` prints inlined paths relative to `bots/` so that bundles stay byte-identical: don't change that.

Result files record the paths of the day they were made. [docs/PATH_MAP.md](docs/PATH_MAP.md) maps the old layout to the current one.

## Branches, commits, pull requests

**Branch names.** A task's branch is made by `task.py claim`: `<tool>/<task id>-<slug>`, for example `codex/t0002-p-only-stage-1-capacity-and-sampling`. Work outside the board uses `<area>/<short-topic>`, lowercase and hyphenated: `docs/model-export-review`, `fix/gate-adapter-mismatch`. Start from an up-to-date `main`. Keep a branch to one topic.

**Commit messages** start with the area, then an imperative summary, 72 characters or fewer:

```text
train: complete Tier XL and XXL training on CUDA
p_only: consistent second-turn states, 2-ply scripted control
docs: review trained model exports and native artifact correctness
```

Areas in use: `docs`, `train`, `training`, `models`, `eval`, `bench`, `roster`, `p_only`, `screen`. Use `chore` for moves and cleanups. `tasks:` commits come from `task.py` only. Agents end every commit message with an `Agent: <tool>@<machine>` line. Put the why in the body when it isn't obvious from the diff. One logical change per commit; a pure move (`git mv` plus reference fixes) never also changes behavior.

**Pull requests** are required when the task says `land: pr`, and for shared core code (`fusion/`, `training/`, `inference/*.hpp`, `tests/`, `tools/cpp/`), models, and bots that go to the host. A PR targets `main`, says what changed and how it was checked, links its task, and links the report or review that justifies a model or bot change. Merge with a merge commit, as the history already does. Don't force-push a branch someone else has pulled, and never force-push `main`.

**Trained models and bots.** A change to `inference/models*/` ships with the offline gate report it was exported from (`reports/models/<tier>/`) in the same pull request. A bot variant that goes to the host ships with its screen result under `reports/eval/` or `bots/p_only/report/`.

## Automatic checks (CI)

GitHub Actions runs these on every pull request and on every push to `main` (workflows in `.github/workflows/`). They are the commands under [Before you push](#before-you-push), so a red check reproduces locally.

| Check | Runs | Reproduce locally |
|---|---|---|
| **Layout, board and links** (`ci.yml`) | folders, `tasks/*.md`, markdown links, and a warning when `docs/MARKDOWN_INDEX.md` is stale; seconds | `python -m unittest tests.test_repo_layout tests.test_task_board`, `python tools/repo/task.py check` and `python tools/repo/md_index.py --check` |
| **Tests** (`ci.yml`) | the core suite on Linux (Python 3.10 and 3.13), macOS arm64 (the NEON kernels) and Windows (informational) | `python -m unittest discover -s tests -t .` |
| **P-only bots** (`ci.yml`) | contract, replay and derive tests; about 10 min | `python -m unittest discover -s bots/p_only/tests` |
| **Torch parity** (`ci.yml`) | NumPy and PyTorch trainers agree (CPU) | `pip install torch`, then `python -m unittest tests.test_torch_backend` |
| **End-to-end pipeline** (`ci.yml`) | the tests that need the engine, against `chen-006/meow-ai-arena` pinned to a commit (no secret); informational until its first green run on `main`. The `P-only` job uses the same checkout. | `git clone https://github.com/chen-006/meow-ai-arena`, `export AI_TEST_ROOT=$PWD/meow-ai-arena`, then the core suite |
| **Bundle and compile** (`bots.yml`) | every P-only variant is bundled into its single file and compiled with `-fsyntax-only`; runs when `bots/`, `inference/` or the packager change | `python tools/repo/release_bots.py --out dist --check` |
| **Task board** (`board.yml`) | `tasks/*.md` after each board commit | `python tools/repo/task.py check` |

A clean checkout needs only `pip install -r requirements-dev.txt` (NumPy, plus matplotlib for the report generators), and a GNU `g++` or `clang++` with libstdc++: the tests' fixture bots include `<bits/stdc++.h>`, which Apple's compiler lacks. On macOS: `brew install gcc`; `fusion/runner.py` then finds `g++-N`, or set `CXX=g++-N`.

**Releasing bots.** The `Bots` workflow is also the delivery path to the host. In GitHub: Actions, Bots, Run workflow, tick *publish* (from `main` only), and optionally list variant names. It attaches every bundle as `<variant>.cpp.gz`, with `MANIFEST.json` (commit and SHA-256 of each bundle) and `SHA256SUMS`, to a release named `bots-<date>-<commit>`. Without *publish*, the same files are kept for 14 days as the run's artifact. A release bundle is the deployed form: it has no `PONLY_STATS` define, so its hash differs from a screen manifest's `bundle_sha256` by exactly that line.

**The engine pin.** CI checks out `chen-006/meow-ai-arena` at the commit named in `ci.yml` (`ref:`), so a change in that repository cannot turn this one red or run new code in CI by itself. To update: change both `ref:` lines in a commit of their own, run the pipeline job, and read the judge-bot differences listed in [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md#where-the-engine-and-the-bots-come-from).

**Workflow rules.**
- Third-party actions are pinned to a commit SHA with the version in a comment. Dependabot proposes the updates monthly.
- Workflows get a read-only token. Only the release job has `contents: write`.
- `pull_request`, never `pull_request_target`: code from a fork never runs with secrets.
- Add a job only if its command also works on a laptop and is written down in the table above.

## Before you push

```bash
export AI_TEST_ROOT=~/code/meow-ai-arena           # a clone of github.com/chen-006/meow-ai-arena; without it the end-to-end tests skip
python -m unittest discover -s tests -t .          # ~20 s without the arena, ~2 min with it (3 tests need the AI_Test working tree and stay skipped)
python -m unittest discover -s bots/p_only/tests   # ~80 s, when you touched bots/p_only/ or bots/hybrid/
python tools/repo/task.py check                   # when you touched tasks/ by hand
python tools/repo/release_bots.py --out dist --check   # when you touched bots/ or inference/ (a few minutes)
```

- A moved or renamed file: `git grep -n "<old path>"` returns nothing outside `docs/PATH_MAP.md`, frozen results and vendored sources.
- A new top-level folder, `tools/` subfolder or `reports/` subfolder: update `tests/test_repo_layout.py` and the README layout in the same commit.
- Timing and strength numbers come from the host, never from a development machine. Don't commit a latency figure measured here.

## Rules carried over from AI_Test

- **Budget.** The search keeps its 30 ms budget inside the 35 ms compute target and the 40 ms output cutoff. Inference must fit inside it.
- **Where to test.** No latency tests or large match batches on development machines; the host and cluster measure strength and timing.
- **Language.** Everything in this repository is English.
