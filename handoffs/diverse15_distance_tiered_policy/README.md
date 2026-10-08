# Diverse-15 opponents, distance-tiered policy, and CPU deployment

This folder is an English execution package for the agent maintaining ctf-fusion.
Start with [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md). The scope is opponent evaluation and selection,
distance-aware policy capacity, memory-aware CPU inference, CUDA-assisted offline
training, and frozen evaluation. It does not contain trained weights or new league
results. Existing source contracts remain authoritative.

The publication baseline is ctf-fusion `93fa7e604ca9a05e48b8e04d787dcb387401fba8`,
including the contract fixes at `7bbeeb0` and the new capacity benchmark.
The evaluation calculator is a standalone diagnostic tool, not an export gate.

## Contents

| File | Purpose |
|---|---|
| `NEXT_AGENT_PROMPT.md` | Ordered implementation request and acceptance criteria. |
| `BOT_SELECTION_PLAN.md` | Assess every bot and choose 15 evidenced, distinct opponents. |
| `bot_inventory_unmeasured.json` | Initial local scan: 129 source candidates, 89 distinct source hashes; not strength results or a complete-discovery claim. |
| `DISTANCE_TIERED_POLICY_PLAN.md` | Spawn-distance prior, dynamic risk/budget override, exact caches and CPU deadlines. |
| `ADAPTIVE_COMPUTE_RESEARCH.md` | Primary-paper evidence for adaptive networks and constrained computation. |
| `CUDA_TRAINING_PLAN.md` | Optional GPU training with NumPy parity and CPU integer export. |
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

Inventory source paths are repository-relative. Refresh the scan on the executor host, including new export directories and branches. The snapshot may contain uncommitted local candidates and is not a frozen remote registry. The roster selector needs a completed measured-evidence catalog; no final 15-bot roster is fabricated here.

The example report is synthetic and intentionally fails the confirmation sample requirement. Seventeen focused tool tests passed during preparation. These tests establish arithmetic/input behavior, not playing strength, trained-model accuracy, Windows correctness or target-host latency. Publication of this folder does not enable any new model.
