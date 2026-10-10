# reports/

Results, as `.md`, `.json` and `.jsonl` only. Raw match logs, shards and checkpoints live in the untracked `data/` and `runs/`.

| Folder | What | Written by |
|---|---|---|
| `models/<tier>/` | offline gate report per model (`n02` ... `n15`, `general`; `.md` + `.json`) and the tier's `SUMMARY.md`. Tiers: `s` (default), `m`, `l`, `xl`, `xxl`. | `training.evaluate_offline`, `tools/pipeline/train_tier_ladder.py` |
| `host/` | what the target host measured: `env_check_<HOST>`, `capacity_<HOST>`, `training_bench_<HOST>`, and analyses computed from them | `tools/host/`, `tools/reporting/` |
| `eval/` | match results: the 1,000-game Bradley-Terry tournament, carrier screens, V ablation | `tools/screens/` |
| `reviews/` | written reviews and audits; the commit hash a review pins is part of its name | by hand |

A report names the artifact it approves (checkpoint hashes, weight digest, schema). The exporter refuses a report made for another artifact, so don't edit a report to "fix" a number: re-run the evaluation.
