# Source baseline and scope of this package

The initial source review inspected `34a83b0969098b46eeee4fd825ee639de7528b28`. It identified Windows binary stdout, an offline/runtime adapter mismatch, incomplete or stale deployment approval, missing score-tie features, insufficient family/map isolation, and unchecked native widths. Its findings were source review evidence, not new match results.

The current repository was fetched at `7bbeeb0d7eb1f7a974997e0aae9a095bc970e0ac`. The subsequent commit and README document fixes for those contracts, including `inference/adapter.hpp`, exact integer-artifact gating, feature spec v2, strict fixture/map holdouts and width checks. This package preserves them; it does not present the earlier defects as unresolved current findings. The full updated repository test suite was not rerun when preparing this handoff. The package's standalone tools were tested separately.

Before publication, the checkout was fast-forwarded to `93fa7e604ca9a05e48b8e04d787dcb387401fba8`. This adds `tools/capacity_bench.py`, a native size workload, synthetic model data and configurable `FUSION_MAX_WIDTH` with checks. The execution prompt now reuses this profiler and distinguishes its tier names from this proposal. No capacity timing was run while preparing this package. New default profiles remain to be selected from actual host evidence.

The repository still explicitly reports no real-data trained model, no integrated bot, no Windows execution and no target-host latency baseline. Distance-tier dispatch, CUDA training, the evaluated diverse-15 roster and their gains are future implementation tasks. This handoff must not be cited as evidence that any of them already won a league or met 50ms.

Sources: [review baseline](https://github.com/Casper015/ctf-fusion/tree/34a83b0969098b46eeee4fd825ee639de7528b28), [current repair commit](https://github.com/Casper015/ctf-fusion/commit/7bbeeb0d7eb1f7a974997e0aae9a095bc970e0ac), [current status](https://github.com/Casper015/ctf-fusion/blob/7bbeeb0d7eb1f7a974997e0aae9a095bc970e0ac/README.md).
