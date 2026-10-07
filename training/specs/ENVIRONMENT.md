# Environment: training host and inference target

| Role | Platform | What runs there |
|---|---|---|
| Data collection | Windows (or Linux) host with many cores | `training/collect.py`: bot processes under the real 50 ms limit |
| Dataset, training, evaluation, export | Windows with a CUDA GPU (CPU works too) | `training/dataset.py`, `train_*.py`, `evaluate_offline.py`, `export_weights.py` |
| Inference | Referee CPU only | Pure C++20 bot with embedded integer weights: no PyTorch, CUDA, Python, files or network |

## Windows readiness

- **Bots:** `fusion/runner.py` mirrors the original arena core with reader threads, so pipe timeouts work. Compilation uses `g++` / `clang++` from `PATH`, `CXX`, or w64devkit (`tools/` near the repository, or `C:/w64devkit`). Flags are `-std=c++20 -O2 -static`, and binaries carry `.exe`.
- **Paths:** every path is a `pathlib.Path`; manifests store POSIX-style relative paths. Atomic writes use `os.replace`.
- **Encodings:** every text file is read and written as UTF-8 (Windows code pages differ). Tool pipes are bytes, so no `\n` → `\r\n` translation; the stream parser also tolerates `\r`.
- **Processes:** `multiprocessing` uses top-level worker functions, so the Windows spawn start method works.
- **Git:** keep the repository outside OneDrive. OneDrive interrupts git's memory-mapped I/O and has truncated files on this project before.

## Training backend

The trainers use numpy on the CPU. The models are 10–30k parameters, so an epoch over millions of rows takes minutes, and the same code path produces the weights the exporter quantizes and the tests check bit for bit. `training.dataset.torch_dataset()` exposes shards as a PyTorch dataset.

A PyTorch/CUDA trainer is the planned next step once the data exists (E1). It must reproduce `fusion/models.py`'s architecture and heads, and export through the same `quant.quantize` path. Until then no CUDA is required, and `env_check` only reports whether PyTorch and CUDA are present.

## Host check and latency baseline (Task E)

Run this on the Windows host, with nothing else heavy running:

```bash
python tools/env_check.py --ai-test C:/path/to/AI_Test --teams 2 6 15
```

It writes `reports/env_check_<host>.md` and `.json`, with a verdict per line:

| Check | Pass condition |
|---|---|
| `all_bots_built` | Every pool bot with weight > 0 (or `--bots`) compiles |
| `every_bot_played` | Every built bot was seated in at least one match (seats continue across the 2-, 6- and 15-team matches) |
| `protocol_clean` | Every reply parses under the engine's rules (correct turn number, 3 × move + act) |
| `first_turn_within_limit` | Every bot's first reply < 2.0 s |
| `no_timeouts` | No reply later than 50 ms |
| `p99_under_40ms` | p99 of turns 2–400 < 40 ms (the brief's output cutoff) |
| `simulator_exact` | `inference/sim.hpp` replays each match exactly |
| `policy_p99_under_5ms` | Opponent policy for every other team, p99 per turn < 5 ms (placeholder weights of the exported shape, or `--model`) |

The report also lists the CPU, compiler, Python, numpy and PyTorch/CUDA versions.

**Status:** not yet run on the Windows host. The development machine is macOS, and AI_Test's rule is no latency tests there. The tool was exercised only with the instant test bot, to check its plumbing.
