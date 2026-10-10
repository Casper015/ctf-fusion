# Optional CUDA training with CPU-only deployment

## Scope and verified baseline

Implement an optional PyTorch CUDA training backend after the data and evaluation contracts are ready. Keep the existing NumPy backend as a reference and fallback. CUDA is an offline training optimization: competition inference must remain embedded integer C++ on the CPU, with no PyTorch, CUDA, runtime files, or network dependency.

Source inspection at ctf-fusion `7bbeeb0d7eb1f7a974997e0aae9a095bc970e0ac` verified that `training/train_policy.py` and `training/train_value.py` currently train NumPy float32 networks through `fusion/models.py` and the custom Adam in `fusion/nn.py`. `training/dataset.py::torch_dataset` exposes rows but is not a GPU trainer. Default P hidden widths are mover 16, squad 64/32, action 16, and target 8; default V widths are 128/16. The current exporter already checks dimensions and binds approval to checkpoint hashes, integer-weight digest, feature schema, and adapter identity. Preserve these protections.

No CUDA training, hardware speed measurement, or package installation was performed for this handoff. Throughput and strength benefits remain hypotheses.

## 1. Reproduce semantics before optimizing

Add a lazily imported backend with explicit `numpy`, `torch-cpu`, and `torch-cuda` choices. Requesting CUDA on an unavailable device must fail clearly; an automatic mode may fall back only if it records the effective backend. Log hardware, driver, library versions, seeds, precision, batching, and dataset identity.

Start with float32, identical initialization arrays and fixed minibatch indices. Reproduce:

- C++-derived feature order, normalization scales, requested-action labels, legal masks, and invalid-label handling;
- the mover sum plus 125-way squad residual, conditional action head, and attack-target pointer head;
- weighted losses, including the existing action/target sums over three units divided by team-row weight, and smoothing over legal joint support;
- V's fractional outcome targets and existing auxiliary-head weights;
- clipped ReLU and its exact boundary derivative convention: NumPy uses a strict `0 < z < 1` mask, so test the Torch implementation explicitly at both endpoints;
- Adam hyperparameters and coupled weight decay on matrices only, with biases excluded. Do not silently substitute AdamW. [PyTorch Adam reference](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html)

Compare forward outputs, each loss, each gradient, and several optimizer steps on fixed small fixtures. Use both absolute and relative tolerances and report maximum errors; do not demand bit equality between GPU and CPU floating-point matrix multiplication. Separately retain exact integer-accumulator parity between the exported Python emulator and C++.

Save NumPy-compatible `model.npz` tensors with the existing names and shapes. Round-trip them through the current model loaders and exporter. Backend-specific optimizer/RNG checkpoints may be separate; they must not alter deployment semantics.

## 2. Spend memory on reused data

The shared C++ extractor remains the feature authority. Extract immutable records once per dataset version, retain provenance and split assignments, and reuse them across epochs, seeds, and capacity tiers. Avoid replaying JSON or recomputing map geometry inside minibatches.

Measure three layouts: contiguous compact records cached in host RAM, reusable normalized tensors cached in host RAM, and a GPU-resident working set. Estimate memory from measured bytes per row times rows, plus minibatch activations and optimizer state. Expanded mover inputs repeat global/unit features; retaining every expanded input may be much larger than retaining compact records. Cache whichever representation minimizes measured total time within an explicit RAM/VRAM budget.

For data exceeding VRAM, transfer large contiguous chunks through a bounded reusable pinned-memory staging pool. Shuffle rows within the training split and periodically rotate chunks so ordering does not become a curriculum artifact. Validation/test records remain separate. Optionally prefetch the next chunk using a copy stream with explicit completion events; never overwrite a host buffer while its transfer is pending.

NVIDIA recommends minimizing host/device traffic and batching small transfers. Unlimited host RAM does not imply unlimited VRAM or pinned-memory capacity. [CUDA memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#data-transfer-between-host-and-device)

Benchmark pinned/nonblocking transfer rather than enabling it by assumption. Calling `pin_memory()` on each freshly created batch can add more overhead than it saves. Start with zero DataLoader workers for cached arrays, then measure worker/prefetch alternatives; Windows workers require spawn-safe entry points. The existing `torch_dataset()` returns a function-local class: move that class and its state to a picklable module-level definition before using Windows multiprocessing workers. [PyTorch transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

## 3. Benchmark the small-network workload

These MLPs can be dominated by Python dispatch, many tiny kernels, and transfers. Sweep representative batch sizes such as 1,024, 4,096, and 16,384, subject to memory and validation behavior. Flatten independent mover/target evaluations into batched operations. Accumulate training statistics on-device and transfer them periodically instead of calling `.item()` for every head and batch. Do not enlarge deployment networks merely to occupy the GPU.

Report separate timings for collection, feature extraction, dataset loading/normalization, host-to-device transfer, forward/backward/optimizer, validation/calibration, and export. Also report cold setup, warm examples/second, epoch wall time, peak memory, and time to a fixed validation target. Warm the device before steady-state measurement; use CUDA events for device work and synchronized wall-clock boundaries for complete phases. Asynchronous launch time alone is not training time. [CUDA timing documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#timing)

Use the same frozen rows and initial batch size for the NumPy, Torch CPU, and Torch CUDA baseline comparison. Then report tuned configurations separately. Repeat measurements and include median and range. Optimizations such as `torch.compile` or CUDA graphs are later experiments: count compilation/capture time and require stable-shape parity tests. [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

If only fraction `f` of the original end-to-end workflow is training, a training speedup `s` gives at most `1 / ((1-f) + f/s)` overall speedup. For example, `f=0.10` and `s=10` yield about 1.10x. CPU bot collection may remain the dominant expense; report it separately from CUDA throughput.

## 4. Mixed precision is a measured option

After float32 parity passes, compare float32 with supported BF16 or FP16 autocast. FP16 training ordinarily uses `torch.amp.GradScaler("cuda")`; keep sensitive masked normalization/loss accumulation in float32 where necessary, and save scaler state when resuming. Record overflow/skipped updates and validate finite gradients. Explicitly control and record TF32 settings during parity checks.

Tiny workloads may see little AMP benefit because they do not saturate the GPU. Promote AMP only if it reduces time to equivalent validation quality, without degrading the quantized deployment scorecard across opponent families and distance tiers. Faster samples per second with worse convergence is insufficient. [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

## 5. Train shared capacity tiers and export through one contract

Use the architecture metadata defined in `DISTANCE_TIERED_POLICY_PLAN.md` for small, base, and large P. One metadata schema must drive both training backends, checkpoint loaders, quantization, and C++ dispatch. Preserve 125-way joint movement and compatible conditional heads across tiers; do not create an independent predictor for every enemy identity.

Train a large teacher first only if it improves held-out predictions. Optionally distill its masked joint/action/target distributions into smaller students using supervised labels plus temperature-scaled KL loss. Generate teacher targets from training states only; choose distillation weights on calibration/validation data and keep the final test untouched. Cache teacher distributions with teacher/schema hashes. Distillation is an experiment, not a prerequisite or an assumed improvement. [PyTorch distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)

Calibrate each actual integer tier, then evaluate the same routing policy and adapter used by the bot. Approval must bind the tier bundle, temperatures, mixture weights, routing configuration, and feature schema. Preserve existing checkpoint/integer hashes, fail-closed policy gates, and the separate unapproved V switch. GPU training does not authorize V to change play.

## 6. Platforms and acceptance

Use the official installation selector for a supported NVIDIA Windows/Linux CUDA build and verify `torch.cuda.is_available()` on the intended host. Do not copy an assumed CUDA version into permanent setup instructions. [PyTorch installation](https://pytorch.org/get-started/locally/)

On Apple silicon, GPU acceleration uses the separate Metal/MPS backend; it is not this CUDA path. Retain NumPy/Torch CPU support on the development Mac. An MPS backend would need its own parity and performance evidence. [Apple PyTorch Metal documentation](https://developer.apple.com/metal/pytorch/)

Deliver a backend implementation, semantic parity tests, cache manifests, resumable checkpoints, a hardware-specific throughput report, and CPU export parity results. Then use the evaluation tools in this folder to assess predictive quality, decision quality, and frozen bot strength. Run full-turn CPU timing on the designated host under first-turn 2 s and subsequent-turn 50 ms limits, including parsing, feature construction, routing, inference, search, and output. CUDA speedup and additional RAM do not guarantee that deadline; the runtime scheduler and safe fallback must satisfy it independently.
