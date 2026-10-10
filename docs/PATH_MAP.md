# Path map: where things were before the 2026-10-10 reorganization

The repository was reorganized in one change (`chore/reorganize-repo`): files were moved with `git mv` and every reference in code, tests and documents was rewritten to match. No behavior changed.
Result files and vendored sources keep the paths of the day they were made; use this table to resolve them. `git log --follow <new path>` follows a file across the move.

## Directories

| Before | Now |
|---|---|
| `candidate/` | `bots/hybrid/` |
| `handoffs/` | `docs/handoffs/` |
| `benchmarks/games_5plus.csv.gz` | `docs/benchmark/games_5plus.csv.gz` |
| `docs/BENCHMARK_*.md`, `docs/DUEL_*.md`, `docs/benchmark_5plus.json`, `docs/duel_review.json` | `docs/benchmark/` |
| `reports/n02.md` ... `reports/general.json` (tier S gate reports) | `reports/models/s/` |
| `reports_m/`, `reports_l/`, `reports_xl/`, `reports_xxl/` (gate reports and `SUMMARY.md`) | `reports/models/m/`, `l/`, `xl/`, `xxl/` |
| `reports/env_check_*`, `capacity_*`, `training_bench_*`, `MULTITHREADING_ANALYSIS.md`, `SIMD_OPTIMIZATION_REPORT.md`, `BENCHMARK_CONSOLIDATED_REPORT.md` | `reports/host/` |
| `reports/tournament_*`, `carrier_*`, `v_ablation_*` | `reports/eval/` |
| `reports/*_REVIEW*.md`, `CARRIER_INTEGRATION_REPORT.md`, `MODEL_NATIVE_REVIEW_8595c36.json`; `reports_m/C08_VS_V12_*`, `TIER_M_REVIEW_*`, `review_bundle_20261009/` | `reports/reviews/` |

Unchanged: `fusion/`, `training/`, `roster/`, `inference/` (including `models*/`), `profiles/`, `tests/`, `p_only/`, `tools/random_bot.py`, `tools/noise_bot.py`.

## `tools/`

| Before | Now |
|---|---|
| `tools/adapter_check.cpp` | `tools/cpp/adapter_check.cpp` |
| `tools/bench_inference.cpp` | `tools/cpp/bench_inference.cpp` |
| `tools/extract_features.cpp` | `tools/cpp/extract_features.cpp` |
| `tools/infer_check.cpp` | `tools/cpp/infer_check.cpp` |
| `tools/kernel_check.cpp` | `tools/cpp/kernel_check.cpp` |
| `tools/mem_check.cpp` | `tools/cpp/mem_check.cpp` |
| `tools/registry_check.cpp` | `tools/cpp/registry_check.cpp` |
| `tools/runtime_check.cpp` | `tools/cpp/runtime_check.cpp` |
| `tools/sim_check.cpp` | `tools/cpp/sim_check.cpp` |
| `tools/size_bench.cpp` | `tools/cpp/size_bench.cpp` |
| `tools/stream.hpp` | `tools/cpp/stream.hpp` |
| `tools/bench_training.py` | `tools/host/bench_training.py` |
| `tools/capacity_bench.py` | `tools/host/capacity_bench.py` |
| `tools/env_check.py` | `tools/host/env_check.py` |
| `tools/build_carrier_variants.py` | `tools/pipeline/build_carrier_variants.py` |
| `tools/check_pilot_evidence.py` | `tools/pipeline/check_pilot_evidence.py` |
| `tools/fetch_benchmark.py` | `tools/pipeline/fetch_benchmark.py` |
| `tools/gen_feature_spec.py` | `tools/pipeline/gen_feature_spec.py` |
| `tools/recommend_training_pool.py` | `tools/pipeline/recommend_training_pool.py` |
| `tools/train_tier_ladder.py` | `tools/pipeline/train_tier_ladder.py` |
| `tools/analyze_multithreading.py` | `tools/reporting/analyze_multithreading.py` |
| `tools/benchmark_5plus.py` | `tools/reporting/benchmark_5plus.py` |
| `tools/benchmark_review.py` | `tools/reporting/benchmark_review.py` |
| `tools/compare_s_m.py` | `tools/reporting/compare_s_m.py` |
| `tools/duel_review.py` | `tools/reporting/duel_review.py` |
| `tools/export_results.py` | `tools/reporting/export_results.py` |
| `tools/generate_bt_tournament_report.py` | `tools/reporting/generate_bt_tournament_report.py` |
| `tools/generate_consolidated_report.py` | `tools/reporting/generate_consolidated_report.py` |
| `tools/generate_multithreading_report.py` | `tools/reporting/generate_multithreading_report.py` |
| `tools/generate_roster_report.py` | `tools/reporting/generate_roster_report.py` |
| `tools/generate_simd_report.py` | `tools/reporting/generate_simd_report.py` |
| `tools/generate_v_ablation_report.py` | `tools/reporting/generate_v_ablation_report.py` |
| `tools/plot_matrix.py` | `tools/reporting/plot_matrix.py` |
| `tools/plot_style_space.py` | `tools/reporting/plot_style_space.py` |
| `tools/run_carrier_screen.py` | `tools/screens/run_carrier_screen.py` |
| `tools/run_light_screen.py` | `tools/screens/run_light_screen.py` |
| `tools/run_standard_bt_tournament.py` | `tools/screens/run_standard_bt_tournament.py` |
| `tools/run_v_ablation_screen.py` | `tools/screens/run_v_ablation_screen.py` |
| `tools/smoke_v_ablation.py` | `tools/screens/smoke_v_ablation.py` |
| `tools/test_carrier2_smoke.py` | `tools/screens/test_carrier2_smoke.py` |
| `tools/test_carrier_smoke.py` | `tools/screens/test_carrier_smoke.py` |

## Not moved on purpose

`inference/models`, `models_m`, `models_l`, `models_xl`, `models_xxl` stay where they are: bots include them as `models_m/registry.hpp`. `tools/random_bot.py` and `tools/noise_bot.py` stay: the committed pools and roster results name them by path.
