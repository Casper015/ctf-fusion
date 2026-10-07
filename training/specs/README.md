# Training specifications

| Spec | Contents |
|---|---|
| [FEATURES.md](FEATURES.md) | Feature spec v1: every P and V input with its range and scale, model input layouts, the extractor record. Generated from the C++ encoder. |
| [ACTIONS.md](ACTIONS.md) | Move order, the 125-index joint formula, action and target heads, label canonicalization, value targets |
| [DATA_FORMAT.md](DATA_FORMAT.md) | Raw match files, per-count shards, index and manifest, leakage rules, phase-2 appends |
| [MODEL_MATRIX.md](MODEL_MATRIX.md) | The 22 artifacts, profile fields, warm start, export and registry, size and latency ceilings and their checks, retraining lifecycle and safe fallback |
| [SIMULATOR_AUDIT.md](SIMULATOR_AUDIT.md) | Task A: `stepExact` and rollout audit, the decoupled simulator, requested-vs-resolved telemetry, Windows match launching |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Training and inference platforms, Windows readiness, the host check and latency baseline (Task E) |
