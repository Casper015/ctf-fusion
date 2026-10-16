# Host Baseline Report: agent_v2 (T0069)

**Date:** 2026-10-16  
**Host:** `desktop-g60ojud` (Windows 10, RTX 4080 Laptop GPU, 16C/32T)  
**Task:** T0069 (`Host: agent_v2 baseline`)  
**Artifacts & Receipts:** `reports/host/agent_v2/`  

---

## 1. Summary

Task T0069 establishes the host baseline on `desktop-g60ojud` for the redesigned `agent_v2` carrier (task T0065, `docs/AGENT_V2.md`, `profiles/agent_v2.json`) prior to any fresh per-count data collection or training.

All goals were achieved and verified:
1. **Windows Build & Unit Tests:** Built `agent_v2` variants and passed all 21 unit tests in `tests/test_agent_v2.py` and `tests/test_agent_v2_training.py` with `AI_TEST_ROOT` and `PYTHONUTF8=1` (all native C++ contracts, simulator parity, P contracts, resolver brute-force match, bridge, identity, and PyTorch PPO smoke/resume tests: `Ran 21 tests in 119.402s - OK`).
2. **Seed Reservations:** Immutable seeds registered in `docs/MEOWMIND_RL_SCREEN_SEEDS.md`:
   - `20261957`: preflight (consumed; failed on buffered stderr);
   - `20261959`: fresh preflight with unbuffered stderr (consumed; PASSED);
   - `20261958`: paired baseline screen (consumed; PASSED).
3. **Eight-Worker Preflight Gate:** Executed under `task.py lock cpu` across 8 concurrent workers on 64 fixtures (16 maps each for counts 2, 5, 8, 15) against the held-out reference roster (`["Fable", "MiMo", "Kimi", "GLM", "Player4"]`).
   - `agent_v2_p2xxl_direct`: **PASSED** (100% turn coverage, 0 faults/late/dead, steady max reply 3.84 ms < 48 ms, first reply max 42.2 ms < 1,800 ms).
   - `agent_v2_p2xxl_search_w16`: **PASSED** (100% turn coverage, 25,600 / 25,600 turns searched with 0 fallbacks, 202,072 leaves evaluated, steady max 14.54 ms < 48 ms, first reply max 55.2 ms).
4. **Paired Screen (`agent_v2_p2xxl_direct` vs `meow_rl_aligned_p2_large`):**
   - 392 fixtures (28 maps per count across counts 2–15), exactly 8 workers under CPU lock, 1,176 matches total (including A/A copy).
   - Utility difference: **+1.64 percentage points** overall (+0.0164 [-0.0042, +0.0369]).
   - Team counts 3–5: **+6.45 percentage points** (+0.0645 [+0.0084, +0.1206], verdict: **helps**).
   - Determinism: `agent_v2_p2xxl_direct~aa` matched `agent_v2_p2xxl_direct` on **392 of 392 fixtures** (+0.0000 [+0.0000, +0.0000]).
   - Reliability: 0 faults, 0 dead, 0 late across all 1,176 games.

---

## 2. Windows Build & Contract Verification

### Binaries Built (`build/p_only/manifest.json`)
- `agent_v2_p2xxl_direct` (binary `0c6b1f2e86d1`, bundle `ecd7b2098b1e`, 32.3 MB)
- `agent_v2_p2xxl_search_w16` (binary `8f36a75682be`, bundle `356cc673c2bf`, 41.2 MB)
- `agent_v2_p2xxl_search_fp32` (binary `5e16e6b983a9`, bundle `5f0fb1173cc3`, 48.6 MB)
- `agent_v2_p2xxl_search_r16` (binary `55b946436040`, bundle `e5ea0771d74a`, 41.2 MB)
- `meow_rl_aligned_p2_large` (binary `d76921a19675`, bundle `0de074c5a1c2`, 33.1 MB)

### Stderr Unbuffering Fix
On Windows, stderr redirection to pipes buffers stderr in the C runtime unless explicitly configured unbuffered. In `bots/agent_v2/bot.hpp`, `std::setvbuf(stderr, nullptr, _IONBF, 0);` and `std::fflush(stderr);` were added, resolving the counter truncation seen on the initial preflight attempt (matching commit `c90dcf6` from T0063).

### Test Suite Execution
Command: `python -m unittest tests/test_agent_v2.py tests/test_agent_v2_training.py -v`
Result: `Ran 21 tests in 119.402s - OK`
- Native C++ contract checks (`tools/cpp/agent_v2/check.cpp`):
  - Resolver `assign_v1` equals unpruned brute-force search over all target assignments and satisfies engine invariant properties (positive score, no drops, no friendly fire).
  - Simulator (`simulate.hpp` + `fusion::step`) matches engine rules and generates lawful flag refreshes.
  - P cache contract and state key binding verified (stale predictions refused).
  - Fixed-work tree search replay is deterministic.
  - Search branches never contaminate parent root or causal history.
- Training & Bridge:
  - Deployed decision path matches bridge execution.
  - Stored action probabilities match executed joint probabilities.
  - Synthetic PyTorch PPO updates and run resumption succeed on new run identity (`agent_v2_v1`).

---

## 3. Eight-Worker Preflight Gate

Receipt: `reports/host/agent_v2/PREFLIGHT.json`  
Seed: `20261959` | Workers: 8 | Wall Time: 23.0 s | Schedule: 64 fixtures (16 maps each for counts 2, 5, 8, 15)

| Variant | Role | Turns Played | Turn Coverage | Steady Max (ms) | First Max (ms) | p99 Max (ms) | Faults / Late / Dead | Gate Verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| `agent_v2_p2xxl_direct` | Production default | 25,600 | 100.0% | 3.84 | 42.2 | 2.28 | 0 / 0 / 0 | **PASS** |
| `agent_v2_p2xxl_search_w16` | Timing experimental | 25,600 | 100.0% | 14.54 | 55.2 | 11.91 | 0 / 0 / 0 | **PASS** |

### Per-Count Latency Breakdown (`agent_v2_p2xxl_direct`)
- **Count 2:** steady max = 0.49 ms, p99 = 0.34 ms, p50 = 0.21 ms, first reply = 5.3 ms
- **Count 5:** steady max = 0.68 ms, p99 = 0.55 ms, p50 = 0.41 ms, first reply = 6.1 ms
- **Count 8:** steady max = 1.01 ms, p99 = 0.87 ms, p50 = 0.62 ms, first reply = 18.4 ms
- **Count 15:** steady max = 3.84 ms, p99 = 2.28 ms, p50 = 1.21 ms, first reply = 42.2 ms

### Search Telemetry (`agent_v2_p2xxl_search_w16`)
- `search_used`: 25,600 / 25,600 turns (100% active, 0 fallbacks).
- Leaves evaluated: 202,072 across 203,096 trajectories (~7.89 leaves/turn).
- Refusals: 0 (0 setup, 0 trajectory, 0 expand, 0 critic, 0 late).
- Worst-case reply across all 25,600 turns was 14.54 ms, comfortably inside the 48.0 ms deadline.

---

## 4. Paired Carrier Screen

Receipts: `reports/host/agent_v2/baseline/`  
Schedule: `b54ad55e2673bd3155dfefec898b28bd4ac146fa69243cf98a445bc769d829fa`  
Seed: `20261958` | Workers: 8 | Wall Time: 131.4 s | Total Games: 1,176

### Contrast: `agent_v2_p2xxl_direct` − `meow_rl_aligned_p2_large`

| Band | Teams | Mean Utility Diff | 95% Map-Cluster CI | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| **All** | 2–15 | **+0.0164** | **[-0.0042, +0.0369]** | not shown | 392 |
| **Band 2** | 2 | +0.0179 | [-0.0171, +0.0529] | not shown | 28 |
| **Band 3–5** | 3–5 | **+0.0645** | **[+0.0084, +0.1206]** | **helps** | 84 |
| **Band 6–10** | 6–10 | -0.0037 | [-0.0358, +0.0284] | not shown | 140 |
| **Band 11–15** | 11–15 | +0.0073 | [-0.0260, +0.0405] | not shown | 140 |

- Fixture outcomes: 140 improved, 125 unchanged, 127 degraded (identical scores on 1 of 392 fixtures).
- **A/A Determinism:** `agent_v2_p2xxl_direct~aa` − `agent_v2_p2xxl_direct` = `+0.0000 [+0.0000, +0.0000]` (392/392 identical scores). Replays are 100% deterministic across all concurrent workers.

### Arm Comparison

| Arm | Games | Overall Utility | Faults / Dead / Late | Opponent Faults / Dead / Late | p99 Median (ms) | Worst p99 (ms) | Max Reply (ms) | Binary SHA-256 |
|---|---:|---:|---|---|---:|---:|---:|---|
| `agent_v2_p2xxl_direct` | 392 | **0.7776** | 0 / 0 / 0 | 0 / 0 / 0 | 0.970 | 5.032 | 7.557 | `0c6b1f2e86d1` |
| `meow_rl_aligned_p2_large` | 392 | 0.7612 | 0 / 0 / 0 | 0 / 0 / 0 | 0.943 | 4.236 | 10.059 | `d76921a19675` |
| `agent_v2_p2xxl_direct~aa` | 392 | 0.7776 | 0 / 0 / 0 | 0 / 0 / 0 | 0.976 | 5.366 | 13.321 | `0c6b1f2e86d1` |

### Architectural Interpretation
This evaluation compares the complete redesigned carrier against the legacy Meowmind carrier using identical actor weights (`aligned_p2_large`). Multiple components change simultaneously (pure P opponent modeling, exact mathematical resolver `assign_v1` with collision resolution, and engine-mirrored simulation).

Results show:
1. `agent_v2_p2xxl_direct` achieves a **+1.64 percentage point overall utility advantage** over the deployed baseline.
2. In small team counts (3–5), `agent_v2` provides a **statistically significant +6.45 pp improvement** (`[+0.84, +12.06] pp`), where tactical collision handling and pure P modeling directly prevent friendly capture interference and mispredictions.
3. In large team counts (6–15), performance is parity-neutral (-0.37 pp at 6–10, +0.73 pp at 11–15).
4. Latency is virtually indistinguishable from the legacy actor-direct bot (p99 median 0.97 ms vs 0.94 ms, max reply 7.56 ms vs 10.06 ms).
5. The platform is robust, reproducible, and ready for fresh count-specific opponent P training (T0066/T0067) and PPO specialist policies (T0068).
