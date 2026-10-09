# Environment check: DESKTOP-G60OJUD

2026-10-08T18:59:36

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
| 2 | Astra | 1.0 | 0.809 | 5.199 | 5.821 | 6.436 | 0 |
| 2 | Sol | 2.1 | 0.06 | 0.16 | 0.196 | 0.227 | 0 |
| 6 | Fable | 17.7 | 0.257 | 0.359 | 0.383 | 0.427 | 0 |
| 6 | Sonnet | 11.3 | 0.058 | 0.13 | 0.16 | 0.173 | 0 |
| 6 | Opus | 12.6 | 0.141 | 0.181 | 0.243 | 0.393 | 0 |
| 6 | DeepSeek | 0.5 | 0.33 | 0.416 | 0.489 | 0.594 | 0 |
| 6 | Qwen | 0.6 | 0.228 | 0.362 | 0.444 | 0.529 | 0 |
| 6 | MiMo | 1.9 | 0.21 | 0.383 | 0.508 | 0.601 | 0 |
| 15 | Kimi | 1.8 | 0.55 | 0.775 | 0.927 | 1.095 | 0 |
| 15 | GLM | 1.1 | 0.378 | 0.591 | 0.672 | 0.702 | 0 |
| 15 | Control_Hybrid | 36.8 | 0.387 | 2.469 | 3.644 | 4.157 | 0 |
| 15 | Player5_Omega | 31.9 | 30.109 | 30.267 | 30.361 | 30.434 | 0 |
| 15 | Player5_Omega_LIT | 32.1 | 19.492 | 23.679 | 25.444 | 26.884 | 0 |
| 15 | R3_CFL_conditional | 36.0 | 0.499 | 2.951 | 3.894 | 4.217 | 0 |
| 15 | Player2_v005 | 31.1 | 15.139 | 15.302 | 15.464 | 15.549 | 0 |
| 15 | Astra_Full | 30.7 | 23.123 | 30.153 | 30.241 | 30.361 | 0 |
| 15 | Player5 | 33.3 | 0.335 | 0.771 | 0.979 | 1.217 | 0 |
| 15 | Player4 | 54.6 | 0.112 | 0.16 | 0.183 | 0.239 | 0 |
| 15 | Baseline | 1.4 | 0.321 | 0.509 | 0.999 | 1.151 | 0 |
| 15 | Noise | 0.3 | 0.177 | 0.271 | 0.359 | 0.416 | 0 |
| 15 | Astra | 25.4 | 2.098 | 3.778 | 4.568 | 9.627 | 0 |
| 15 | Sol | 33.9 | 3.186 | 5.567 | 6.388 | 6.767 | 0 |
| 15 | Fable | 40.2 | 0.473 | 0.765 | 1.037 | 1.07 | 0 |

## Inference

Model `weights_placeholder.hpp` on the 15-team match: policy per turn p50 0.126 / p99 0.173 ms (ceiling 5 ms); value per node encode 0.69 + incremental 0.30 us (full pass 0.77 us).

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
- 15 teams: turns 400 compared 400 skipped 0 mismatches 0 events 1333
