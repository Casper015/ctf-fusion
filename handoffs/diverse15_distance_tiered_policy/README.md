# Open opponent training pool, adaptive policy compute, and CPU deployment

This folder is an English execution package for the agent maintaining ctf-fusion.
Start with [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md). The scope is opponent evaluation and selection,
distance-aware policy capacity, memory-aware CPU inference, CUDA-assisted offline
training, and frozen evaluation. It does not contain trained weights or new league
results. Existing source contracts remain authoritative.

The initial package baseline was ctf-fusion `93fa7e604ca9a05e48b8e04d787dcb387401fba8`,
including the contract fixes at `7bbeeb0` and the capacity benchmark. This report
revision was synchronized with `aa3fcc1`, including roster tooling and newer SIMD reports.
The evaluation calculator is a standalone diagnostic tool, not an export gate.

**Updated requirement:** training opponents have no fixed 15-bot limit. Admit reliable, meaningfully distinct behaviors and weight them by count/context. A fixed panel of 15 remains an optional comparison budget, not the training registry. The historical folder name is retained to preserve existing links. This changes the implementation plan; the current collector has not yet implemented open admission and hierarchical sampling.

## Contents

| File | Purpose |
|---|---|
| `NEXT_AGENT_PROMPT.md` | Ordered implementation request and acceptance criteria. |
| [BOT_SELECTION_PLAN.md](BOT_SELECTION_PLAN.md) | Admit useful opponents to an open training pool and keep evaluation panels separate. |
| [BOT_GROUP_AND_SAMPLING_REVIEW.md](BOT_GROUP_AND_SAMPLING_REVIEW.md) | Evidence about lineage versus style, clustering limits, and the updated sampling policy. |
| `bot_inventory_unmeasured.json` | Initial local scan: 129 source candidates, 89 distinct source hashes; not strength results or a complete-discovery claim. |
| `DISTANCE_TIERED_POLICY_PLAN.md` | Spawn-distance prior, dynamic risk/budget override, exact caches and CPU deadlines. |
| `ADAPTIVE_COMPUTE_RESEARCH.md` | Primary-paper evidence for adaptive networks and constrained computation. |
| `CUDA_TRAINING_PLAN.md` | Optional GPU training with NumPy parity and CPU integer export. |
| [FIXED_POINT_PROBABILITY_PLAN.md](FIXED_POINT_PROBABILITY_PLAN.md) | Integer softmax/sigmoid, value semantics, sampling precision and acceptance checks. |
| `SOURCE_BASELINE.md` | Current source baseline and the status of earlier review fixes. |
| `evaluation/` | English evaluation contract, calculator, plan, tests and synthetic example. |
| `tools/` | Source inventory and evidence-based roster selection utilities with tests. |

## Run the package tools

The utilities need Python 3.10+ and NumPy; they reuse the repository's NumPy dependency. No CUDA is required to use this folder.

```sh
python3 -m unittest discover -s tools -p 'test_*.py' -v
cd evaluation
python3 -m unittest -v test_scorecard
python3 scorecard.py --plan scorecard_plan.json --input example_metrics.jsonl --out example_report.json
```

Inventory source paths are repository-relative. Refresh the scan on the executor host, including new export directories and branches. The snapshot may contain uncommitted local candidates and is not a frozen remote registry. The existing fixed-size selector needs a completed measured-evidence catalog and is now scoped to optional reference panels. Open training-pool admission is a separate implementation task; no final training or evaluation roster is fabricated here.

The example report is synthetic and intentionally fails the confirmation sample requirement. Seventeen focused tool tests passed during preparation. These tests establish arithmetic/input behavior, not playing strength, trained-model accuracy, Windows correctness or target-host latency. Publication of this folder does not enable any new model.
