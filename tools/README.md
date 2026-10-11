# tools/

Scripts and C++ drivers that are not part of the pipeline's importable code. Run them from the repository root.

| Folder | What | Writes to |
|---|---|---|
| `cpp/` | contract checks and benchmark drivers (`infer_check`, `sim_check`, `kernel_check`, `adapter_check`, `registry_check`, `runtime_check`, `size_bench`, `bench_inference`, `mem_check`, `extract_features`). Built and run by `fusion/native.py`; includes resolve through `../../inference/`. | `build/` |
| `host/` | `env_check.py`, `capacity_bench.py`, `bench_training.py`. Run on the target host; a development machine's timings are not evidence. | `reports/host/` |
| `pipeline/` | steps that produce committed inputs: `train_tier_ladder.py`, `recommend_training_pool.py`, `check_pilot_evidence.py`, `build_carrier_variants.py`, `gen_feature_spec.py`, `fetch_benchmark.py` | `inference/models_*/`, `profiles/`, `bots/hybrid/`, `training/specs/` |
| `screens/` | experiments that play matches: `run_light_screen.py`, `run_carrier_screen.py`, `run_standard_bt_tournament.py`, `run_v_ablation_screen.py`, plus smoke tests | `reports/eval/`, `roster/report/` |
| `reporting/` | turn results into reports and plots: `generate_*_report.py`, `benchmark_*`, `duel_review.py`, `export_results.py`, `plot_*.py` | `reports/`, `docs/benchmark/` |
| `repo/` | `task.py`: the shared task board, agent worktrees, lane checks and machine locks ([AGENTS.md](../AGENTS.md)). `release_bots.py`: packages the P-only bots as single files with hashes (what the `Bots` workflow publishes) | `tasks/` (on `main`, atomically); `dist/` |

`random_bot.py` and `noise_bot.py` sit directly in `tools/` on purpose: the committed pools and roster results name them as `@repo/tools/random_bot.py`. Don't move them.

Adding a script: pick the folder by what it does, compute the repository root as `Path(__file__).resolve().parents[2]`, and register nothing else. Rules: [CONTRIBUTING.md](../CONTRIBUTING.md).
