# First Own-Action Host Pilot Report: Export, Deadline Gate, and Paired Screen

**Task:** T0051  
**Agent:** `codex-decision-pilot@desktop-g60ojud`  
**Host Machine:** `desktop-g60ojud` (Windows)  
**Date:** 2026-10-15  
**Frozen Schedule:** `data/p_only/own_action_first_trial/schedule.json` (Seed: 20261801, 392 fixtures, 14 team counts)  
**Deliverables Directories:** `bots/p_only/report/own_action_first_trial/`, `reports/host/own_action_first_trial/`  

---

## 1. Executive Summary & Outcome

This report documents the completion of task **T0051**: the first end-to-end deployment feasibility, host concurrency gating, and matched-replacement paired-game screen evaluating learned own-action scoring models trained during T0050.

- **Pre-declared Candidate Exports:**
  - Exported `first_trial_mlp_s101` ($\lambda = 100.0$, `zscore` norm, primary candidate) and `first_trial_linear_s101` ($\lambda = 50.0$, `zscore` norm, secondary comparator) using the frozen selection from T0050.
  - Bit-exact float64 C++/Python parity verified on recorded roots ($< 2 \times 10^{-16}$).
  - Windows toolchain scoring hook tests verified and passing (20/20 unit tests).
- **Concurrency & Deadline Preflight:**
  - Conducted concurrency ladder (1, 2, 4, 6 workers) under official limits (2,000 ms init / 50 ms turn) and $n=15$ stress load.
  - At 6 workers, scoring deadline drops occurred (2 drops).
  - At 4 workers, peak latency was 11.29 ms, with 0 deadline drops, 0 late replies, and 0 faults.
  - **Admitted host concurrency:** **4 workers** (frozen prior to measurement).
- **Matched Replacement Paired Pilot:**
  - Evaluated 4 arms across 392 independent fixtures (1,568 total games) under exclusive CPU lock:
    1. `v12_ponly_xxl` (unchanged baseline control)
    2. `v12_ponly_xxl~aa` (byte-identical A/A replay control)
    3. `v12_ponly_xxl_first_trial_linear_s101` (secondary candidate)
    4. `v12_ponly_xxl_first_trial_mlp_s101` (primary candidate)
  - 100% completion: 1,568 games finished with **0 faults, 0 dead games, and 0 late replies**.
  - Maximum turn reply across all games: **12.05 ms** (well within 50 ms limit).
- **Empirical Findings:**
  - **A/A Diagnostic:** Mean difference $\Delta = +0.0000$ $[+0.0000, +0.0000]$, **392 of 392 identical final scores** (100% deterministic reproducibility).
  - **Primary Endpoint (`mlp_s101 - v12_ponly_xxl`):** Mean $\Delta = -0.1828$, 95% Map-Clustered CI $[-0.2114, -0.1543]$. Verdict: **`hurts`** (56 improved / 95 unchanged / 241 degraded).
  - **Secondary Endpoint (`linear_s101 - v12_ponly_xxl`):** Mean $\Delta = -0.1555$, 95% Map-Clustered CI $[-0.1848, -0.1261]$. Verdict: **`hurts`** (69 improved / 103 unchanged / 220 degraded).
- **Formal Outcome:** **Explicit NO-ADMISSION**.
  - Both learned scoring heads statistically significantly degrade playing strength relative to the unguided $v12$ baseline across all player counts (2 through 15).
  - Baseline `v12_ponly_xxl` remains completely unchanged as the production incumbent. Neither candidate is admitted.

---

## 2. Model Export and Parity Verification

Candidates from T0050 (`reports/models/own_action_first_trial/SELECTION.json`) were exported into dedicated C++ header directories via `training.export_decision_scorer`:

| Scorer Directory | Selected Model | Normalization | Parameter $\lambda$ | Source-Bound Carrier | Header SHA-256 |
|---|---|---|---:|---|---|
| `inference/own_action_scorers/first_trial_linear_s101/` | `linear_s101` | `zscore` | 50.0 | `v12_ponly_xxl_dlog` | `669484b712b8...` |
| `inference/own_action_scorers/first_trial_mlp_s101/` | `mlp_s101` | `zscore` | 100.0 | `v12_ponly_xxl_dlog` | `7db989e4b1f4...` |

### Parity and Equivalence Checks
1. **Float64 Parity:** Evaluation of exported C++ scorer code against Python NumPy implementation on the test dataset roots confirmed maximum difference $< 2 \times 10^{-16}$.
2. **Scoring Hook Unit Tests:** All 20 tests in `test_own_action_parity`, `test_own_action_scorer`, `test_decision_hook`, and `test_decision_scoring` pass with MinGW g++ on Windows.
3. **$\lambda = 0$ Equivalence:** Build with $\lambda = 0.0$ confirmed identical candidate rank choices and zero altered moves against control.

---

## 3. Host Concurrency and Deadline Preflight

The official preflight ladder evaluated 1, 2, 4, and 6 workers across team counts 2, 5, 10, and 15 under 2,000 ms initialization and 50 ms subsequent reply deadlines:

| Workers | Games | Faults | Late Replies | Dead Procs | $n=15$ Max Latency | $n=15$ p99 Latency | DSCORE Deadline Drops | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 0 | 0 | 0 | 10.18 ms | 8.60 ms | 0 | Pass |
| 2 | 16 | 0 | 0 | 0 | 9.67 ms | 9.01 ms | 0 | Pass |
| **4** | **16** | **0** | **0** | **0** | **11.29 ms** | **9.14 ms** | **0** | **Admitted (Safe)** |
| 6 | 16 | 0 | 0 | 0 | 10.97 ms | 9.65 ms | 2 | Rejected (Drop > 0) |

- At 6 workers, 2 deadline drops occurred where scoring fell back to carrier heuristic rank due to approaching the internal 40 ms scoring budget.
- Concurrency was strictly frozen at **4 workers** prior to the screen.

---

## 4. Frozen Evaluation Schedule

- **Seed:** `20261801` (newly reserved in `docs/DIGEST.md` before games).
- **Structure:** 392 fixtures (28 independent maps for each count $n \in \{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}$).
- **Seat Assignment & Interleaving:** Same designated seat replaced within each fixture, identical reference opponents, interleaved arm order to distribute host drift evenly.
- **Reference Opponents:** Astra, Sol, Fable, Sonnet, Opus, DeepSeek, Qwen, MiMo, Kimi, GLM, Player5, Baseline (hashes verified from pool).

---

## 5. Host Timing and Reliability Receipts

Screen executed under CPU lock (`task.py lock cpu --task T0051 -- python bots/p_only/tools/screen_runner.py ...`).

| Arm | Games | Faults | Dead | Late | First Max (ms) | p99 Median (ms) | p99 Worst (ms) | Max Reply (ms) | Binary SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v12_ponly_xxl` | 392 | 0 | 0 | 0 | 47.2 | 4.86 | 9.68 | 11.32 | `d8095b80b5bd` |
| `v12_ponly_xxl_first_trial_linear_s101` | 392 | 0 | 0 | 0 | 46.9 | 5.04 | 9.91 | 11.52 | `08df6938190c` |
| `v12_ponly_xxl_first_trial_mlp_s101` | 392 | 0 | 0 | 0 | 45.5 | 4.91 | 9.92 | 12.05 | `4409462c5012` |
| `v12_ponly_xxl~aa` | 392 | 0 | 0 | 0 | 49.2 | 4.95 | 9.63 | 10.86 | `d8095b80b5bd` |

All four arms executed cleanly without a single timing violation or process crash.

---

## 6. Paired Statistical Differences

Intervals: Map-clustered 95% confidence intervals, team counts weighted equally.

### Detailed Contrast Results

| Comparison | Band | Mean $\Delta$ | 95% Interval | Verdict | Fixtures | Improved / Unchanged / Degraded |
|---|---|---:|---|---|---:|---|
| **`mlp_s101 - v12_ponly_xxl` (Primary)** | **all** | **-0.1828** | **[-0.2114, -0.1543]** | **`hurts`** | **392** | **56 / 95 / 241** |
| | 2 | -0.3036 | [-0.4657, -0.1415] | `hurts` | 28 | |
| | 3–5 | -0.1860 | [-0.2606, -0.1114] | `hurts` | 84 | |
| | 6–10 | -0.1589 | [-0.2008, -0.1171] | `hurts` | 140 | |
| | 11–15 | -0.1807 | [-0.2203, -0.1410] | `hurts` | 140 | |
| **`linear_s101 - v12_ponly_xxl` (Secondary)** | **all** | **-0.1555** | **[-0.1848, -0.1261]** | **`hurts`** | **392** | **69 / 103 / 220** |
| | 2 | -0.2321 | [-0.3874, -0.0769] | `hurts` | 28 | |
| | 3–5 | -0.1493 | [-0.2300, -0.0686] | `hurts` | 84 | |
| | 6–10 | -0.1209 | [-0.1649, -0.0770] | `hurts` | 140 | |
| | 11–15 | -0.1783 | [-0.2173, -0.1393] | `hurts` | 140 | |
| **`v12_ponly_xxl~aa - v12_ponly_xxl` (A/A Diagnostic)** | **all** | **+0.0000** | **[+0.0000, +0.0000]** | **`not shown`** | **392** | **0 / 392 / 0** (392/392 identical scores) |

---

## 7. Analysis and Discussion

1. **Why Did Offline Selection Fail Online?**
   - In T0050, both linear and MLP models achieved apparent positive validation gains on 37 validation maps at $\lambda \ge 20.0$. However, on the 31 test maps in T0050, the mean delta was already negative ($-0.015$ and $-0.020$).
   - In full online tournament gameplay, altering the incumbent bot's candidate ranking based on a short offline rollout gain surrogate introduces distribution shift: subsequent moves follow the altered policy, diverging from the incumbent distribution from which continuation trajectories were sampled.
   - The degradation is pronounced across all team sizes ($-0.30$ in 2-team duels, $-0.18$ in large melees).
2. **Re-run Noise and Reproducibility:**
   - The A/A diagnostic replaying the exact binary achieved 392 out of 392 identical match scores, demonstrating zero re-run noise under deterministic referee conditions.
   - The measured performance drop is genuine policy degradation, not evaluation variance.
3. **No Cherry-Picking:**
   - No post-hoc sweep or alternative lambda values were tried online. The pre-declared primary and secondary models were tested once and reported honestly.

---

## 8. Admission Decision and Handoff

- **Decision:** **REJECTED_NO_ADMISSION**.
- **Action:** Production bot defaults, carrier logic, and baseline configurations remain completely untouched. Neither `first_trial_mlp_s101` nor `first_trial_linear_s101` will be promoted to main or substituted for `v12_ponly_xxl`.
- **Artifacts:** All raw logs, results, schedules, binaries, and preflight records are durable and committed in accordance with repository specifications.
