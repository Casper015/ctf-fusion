# Environment: training host and inference target

| Role | Platform | What runs there |
|---|---|---|
| Data collection | Windows (or Linux) host with many cores | `training/collect.py`: bot processes under the real 50 ms limit |
| Dataset, training, evaluation, export | Windows with a CUDA GPU (CPU works too) | `training/dataset.py`, `train_*.py`, `evaluate_offline.py`, `export_weights.py` |
| Inference | Referee CPU only | Pure C++20 bot with embedded integer weights: no PyTorch, CUDA, Python, files or network |

## Windows readiness

- **Bots:** `fusion/runner.py` mirrors the original arena core with reader threads, so pipe timeouts work. Compilation uses `g++` / `clang++` from `PATH`, `CXX`, or w64devkit (`tools/` near the repository, or `C:/w64devkit`). Flags are `-std=c++20 -O2 -static`, and binaries carry `.exe`.
- **Paths:** every path is a `pathlib.Path`; manifests store POSIX-style relative paths. Atomic writes use `os.replace`.
- **Encodings:** every text file is read and written as UTF-8 (Windows code pages differ). The stream parser tolerates `\r`.
- **Binary tool output:** Python reads tool pipes as bytes, but that alone does not stop the Windows C runtime from writing stdout in text mode, which turns each `0x0A` byte into `0x0D 0x0A` (a turn-10 record already contains one). The raw-record tools (`extract_features`, `infer_check`) switch stdout to `_O_BINARY` first (`tools/stream.hpp` `binaryStdout`). `fusion.native.extract` checks every record's turn and team columns, so a misaligned stream fails loudly. `tests/test_contracts.py` feeds a stream whose records contain `0x0A`, `0x0D` and `0x1A`, and requires the exact byte count; `tools/env_check.py` repeats the alignment and integer-parity checks on the host.
- **Processes:** `multiprocessing` uses top-level worker functions, so the Windows spawn start method works.
- **Git:** keep the repository outside OneDrive. OneDrive interrupts git's memory-mapped I/O and has truncated files on this project before.

## Training backend

Two backends train the same models with the same objective:

- **NumPy** (default, the reference): CPU; the code path every test and export was built on.
- **PyTorch** (`--backend torch --device cuda`, [`fusion/torch_backend.py`](../../fusion/torch_backend.py)): CUDA on the host GPU, or CPU.

The PyTorch backend reproduces the NumPy training operation by operation:
- the initial weights (copied from the NumPy model built with the run's seed);
- the clipped ReLU, including its gradient at the 0 and 1 endpoints;
- the masked, label-smoothed cross-entropy and the value loss, including ignored labels and empty masks;
- the weighted-sum normalization and the head weights;
- Adam with L2 decay on matrices only;
- the minibatch order: rows resident on the GPU and rows streamed in windows give identical minibatches, and team counts stay mixed in every minibatch.

Per-epoch validation runs on the GPU. Calibration, checkpoints and export run on the NumPy model, so artifacts are produced exactly as before.

`tests/test_torch_backend.py` checks the parity with NumPy:
- gradients and one optimizer step;
- two-epoch training of policy and value;
- validation loss;
- streamed against resident rows (bit-identical);
- the endpoint gradient.

`tests/test_pipeline.py` trains, evaluates and exports with the torch backend and keeps the C++ integer parity. Two output biases (mover, target) have an exactly zero true gradient, because they shift every choice of a unit equally. Both backends move them only by rounding noise, which cannot change any prediction (tested). The tests skip without PyTorch. On the host, `FUSION_TEST_DEVICE=cuda python -m unittest tests.test_torch_backend` runs them on the GPU.

**Setup on the host (RTX 4080 Laptop, 12 GB):** install a CUDA build of PyTorch for the host's Python from the [official selector](https://pytorch.org/get-started/locally/). It is optional; NumPy remains the default. Then:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
FUSION_TEST_DEVICE=cuda python -m unittest tests.test_torch_backend
```

**Measure before relying on it:**

```bash
python tools/bench_training.py --model n15 --tier S XXL --backends numpy torch-cuda --max-rows 500000
```

It writes `reports/training_bench_<host>.md`: rows per second, epoch and wall time, validation loss at equal epochs, and peak VRAM, per backend and tier. Small networks (tier S) can be limited by loading and kernel launches; the larger tiers are where the GPU has arithmetic to do.

**Memory.**
- **Fitting the GPU:** a split's rows stay on the GPU when they fit `--gpu-resident-gb` (default 6). Otherwise they stream in windows of `--gpu-window-rows` (default 1,000,000 rows, 1.2 GiB of records), with the next window gathered on the CPU in the background.
- **Policy:** the value net needs only its 54 int16 columns, so its training split usually stays resident. The general policy's training split (about 29 GiB of records) streams, while its validation split usually stays resident.
- **CPU RAM:** the loader still holds the whole split in RAM, as the NumPy path does.

**Precision.** Float32; TF32 is off unless `--tf32`, which changes rounding and has to be compared before use. The run's `config.json` records the backend, PyTorch and CUDA versions, the GPU, epoch timings, rows per second and peak VRAM.

**The largest tier (XXL, from `tools/capacity_bench.py`; it fits the per-turn budget at every team count on the host):**

```bash
python -m training.train_policy --model general --backend torch --device cuda --hidden-mover 256 --hidden-squad 1024,256 --hidden-act 256 --hidden-target 32 --max-width 2048 --count-weighting inv_teams --out runs/policy_general_xxl
```

```bash
python -m training.train_value --model general --backend torch --device cuda --hidden 2048,128 --max-width 2048 --count-weighting inv_teams --out runs/value_general_xxl
```

Dedicated models warm-start from it with the same widths (`--init runs/policy_general_xxl`). Export needs `--max-width 2048`, and the bot must be built with `-DFUSION_MAX_WIDTH=2048`. Whether XXL predicts better than the exported models is decided by `evaluate_offline` on the same test maps, not by training speed.

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

## Capacity: how big the networks can be

```bash
python tools/capacity_bench.py --ai-test C:/path/to/AI_Test                   # every team count, tiers S M L XL
python tools/capacity_bench.py --raw data/raw --teams 2 6 10 15 --tiers S M L XL XXL
```

`tools/size_bench.cpp` builds each size tier at runtime with random int16 weights (integer inference costs the same whatever the values) and runs the bot's per-turn inference on every turn of a recorded match:
- the policy for every opponent team with every head (an upper bound);
- `--value-calls` value-network search nodes (default 24), alternating between consecutive states so every accumulator update has real deltas.

`capacity_bench.py` runs one copy per seat at once, so the CPU is as busy as during a real match, and does this for every team count on its real map. It takes the worst p99 over all turns and seats, multiplies by `--safety` (1.5), and compares the result with the inference budget (`compute_target - search` = 5 ms, or `--budget-ms`). Output: `reports/capacity_<host>.md` and `.json`, with a verdict per tier and team count, the value nodes that still fit, header source sizes, and the recommended ceiling.

`--verify` runs no timing: the SIMD kernels must equal the scalar loop bit for bit (`tools/kernel_check.cpp`), and every tier's incremental value must equal the full pass with buffers as wide as the tier needs. The tests run it; it is safe on any machine.

**Kernels.** Every timing run starts with the same kernel parity check and stops on any mismatch. `--kernels both` (default) times every tier with the SIMD kernels and with the old scalar loop; verdicts use the SIMD kernels, which the bot runs. The kernels need no compiler flags: SSE2 is part of every x86-64 CPU, AVX2 is chosen at runtime after a CPU check, and NEON is standard on ARM64.

**Loaded against isolated.** Run once with the default seats and once with `--seats 1 --tag 1seat` (it writes `reports/capacity_1seat_<host>.json`), then `python tools/generate_multithreading_report.py`. Every statement in that report is computed from the two files.

**Status:** built and verified (all tiers up to width 2048); timing not yet run on the host.
