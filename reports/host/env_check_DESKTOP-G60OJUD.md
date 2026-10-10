# Environment check: DESKTOP-G60OJUD

2026-10-09T09:27:21

## Host

- host: DESKTOP-G60OJUD
- cpu: AMD Ryzen 9 7945HX with Radeon Graphics
- os: Windows-10-10.0.19045-SP0
- machine: AMD64
- processor: AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD
- logical_cpus: 32
- python: 3.14.3
- numpy: 2.5.3
- compiler: C:\path\w64devkit\bin\g++.EXE
- compiler_version: g++.EXE (GCC) 15.2.0
- cxx_flags: ['-std=c++20', '-O2', '-static']
- torch: not installed

## Verdict

- PASS: all_bots_built
- PASS: every_bot_played
- PASS: protocol_clean
- PASS: first_turn_within_limit
- PASS: no_timeouts
- PASS: p99_under_40ms
- PASS: simulator_exact
- PASS: records_aligned
- PASS: kernel_parity
- PASS: integer_parity
- PASS: runtime_contract
- PASS: policy_p99_under_5ms

## Bot latency (turns 2-400, ms)

| teams | bot | first | p50 | p95 | p99 | max | faults |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | Astra | 1.2 | 0.857 | 5.301 | 6.175 | 6.829 | 0 |
| 2 | Sol | 5.3 | 0.062 | 0.159 | 0.206 | 0.225 | 0 |
| 6 | Fable | 19.1 | 0.263 | 0.361 | 0.401 | 0.408 | 0 |
| 6 | Sonnet | 12.4 | 0.057 | 0.129 | 0.148 | 0.159 | 0 |
| 6 | Opus | 12.7 | 0.145 | 0.178 | 0.205 | 0.347 | 0 |
| 6 | DeepSeek | 0.5 | 0.338 | 0.433 | 0.472 | 0.595 | 0 |
| 6 | Qwen | 0.9 | 0.232 | 0.371 | 0.456 | 0.625 | 0 |
| 6 | MiMo | 7.6 | 0.208 | 0.396 | 0.493 | 0.736 | 0 |
| 15 | Kimi | 1.8 | 0.566 | 0.783 | 1.076 | 1.145 | 0 |
| 15 | GLM | 1.3 | 0.373 | 0.594 | 0.651 | 0.85 | 0 |
| 15 | Control_Hybrid | 34.2 | 0.42 | 2.586 | 3.785 | 4.836 | 0 |
| 15 | Player5_Omega | 33.8 | 30.117 | 30.288 | 30.343 | 30.417 | 0 |
| 15 | Player5_Omega_LIT | 33.4 | 19.655 | 23.306 | 24.308 | 25.496 | 0 |
| 15 | R3_CFL_conditional | 32.6 | 0.488 | 2.99 | 3.599 | 4.357 | 0 |
| 15 | Player2_v005 | 31.8 | 15.141 | 15.302 | 15.42 | 15.501 | 0 |
| 15 | Astra_Full | 30.9 | 24.524 | 30.198 | 30.257 | 30.276 | 0 |
| 15 | Player5 | 34.0 | 0.348 | 0.795 | 0.962 | 1.139 | 0 |
| 15 | Player4 | 53.0 | 0.118 | 0.169 | 0.184 | 0.402 | 0 |
| 15 | Baseline | 1.9 | 0.355 | 0.552 | 1.021 | 1.167 | 0 |
| 15 | Noise | 0.3 | 0.187 | 0.303 | 0.473 | 0.538 | 0 |
| 15 | Astra | 26.2 | 2.324 | 4.647 | 6.447 | 7.186 | 0 |
| 15 | Sol | 36.0 | 3.912 | 6.42 | 7.971 | 9.469 | 0 |
| 15 | Fable | 41.3 | 0.552 | 0.805 | 1.033 | 1.096 | 0 |

## Inference

Model `weights_placeholder.hpp` on the 15-team match: policy per turn p50 0.130 / p99 0.179 ms (ceiling 5 ms); value per node encode 0.72 + incremental 0.29 us (full pass 0.80 us).

## Contracts

- kernel_parity: True
- kernel_check: kernel_check active avx2 paths scalar sse2 avx2 checks 3139 mismatches 0
- records_aligned: True
- integer_parity: True
- runtime_check: turns 400 allocating_turns 0 accumulator_checks 6000 accumulator_mismatches 0 (0)
- runtime_contract: True

## Simulator

- 2 teams: turns 400 compared 400 skipped 0 mismatches 0 events 261
- 6 teams: turns 400 compared 400 skipped 0 mismatches 0 events 394
- 15 teams: turns 400 compared 400 skipped 0 mismatches 0 events 1300
