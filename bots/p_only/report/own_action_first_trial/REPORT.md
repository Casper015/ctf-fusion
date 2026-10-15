# First Own-Action Host Pilot Report: Export, Deadline Gate, and Paired Screen

**Task:** T0051  
**Agent:** `codex-decision-pilot@desktop-g60ojud`  
**Host Machine:** `desktop-g60ojud` (Windows)  
**Date:** 2026-10-15  
**Generator:** `bots/p_only/tools/analyze.py, bots/p_only/tools/preflight_own_action.py, bots/p_only/tools/screen_runner.py`  
**Frozen Schedule:** `data/p_only/own_action_first_trial/schedule.json` (Seed: 20261801, 392 fixtures, 14 team counts)  
**Deliverables Directories:** `bots/p_only/report/own_action_first_trial/`, `reports/host/own_action_first_trial/`  
**Encoding Note:** All file hashes are computed on canonical UNIX LF (`\n`) byte streams.  

---

## 1. Executive Summary & Outcome

This report documents the completion of task **T0051**: the first end-to-end deployment feasibility, host concurrency gating, and matched-replacement paired-game screen evaluating learned own-action scoring models trained during T0050.

- **Pre-declared Candidate Exports:**
  - Exported `first_trial_mlp_s101` ($\lambda = 100.0$, `zscore` norm, primary candidate) and `first_trial_linear_s101` ($\lambda = 50.0$, `zscore` norm, secondary comparator) using the frozen selection from T0050.
  - Bit-exact float64 C++/Python parity verified on recorded test roots ($< 2 \times 10^{-16}$).
  - Windows toolchain scoring hook tests verified and passing (20/20 unit tests).
- **Concurrency & Deadline Preflight:**
  - Preflight script located in `bots/p_only/tools/preflight_own_action.py`.
  - Conducted concurrency ladder (1, 2, 4, 6 workers) under official limits (2,000 ms init / 50 ms turn) and $n=15$ stress load with strict code gating.
  - At 6 workers, 2 deadline drops occurred where scoring fell back to carrier heuristic rank due to approaching the internal 40 ms scoring budget.
  - At 4 workers, peak latency was 10.02 ms, with 0 deadline drops, 0 late replies, and 0 faults.
  - **Admitted host concurrency:** **4 workers** (strictly gated by code).
- **Matched Replacement Paired Pilot (1,568 Games):**
  - Evaluated 4 arms across 392 independent fixtures (1,568 total games) under exclusive host CPU lock:
    1. `v12_ponly_xxl` (unchanged baseline control)
    2. `v12_ponly_xxl~aa` (byte-identical A/A replay control)
    3. `v12_ponly_xxl_first_trial_linear_s101` (secondary candidate)
    4. `v12_ponly_xxl_first_trial_mlp_s101` (primary candidate)
  - 100% completion: 1,568 games finished with **0 faults, 0 dead games, and 0 late replies**.
  - Maximum turn reply across all games: **12.05 ms** (well within 50 ms limit).
  - Denominator verification: all 392 fixtures were completed by all 4 arms without any drops or faults.
- **Empirical Findings:**
  - **A/A Diagnostic:** Mean difference $\Delta = +0.0000$ $[+0.0000, +0.0000]$, **392 of 392 identical final scores**. Under deterministic referee conditions, zero random host execution variance was observed.
  - **Primary Endpoint (`mlp_s101 - v12_ponly_xxl`):** Mean $\Delta = -0.1828$, 95% Map-Clustered CI $[-0.2114, -0.1543]$. Verdict: **`hurts`** relative to the negligible margin $\pm 0.002$ (56 improved / 95 unchanged / 241 degraded).
  - **Secondary Endpoint (`linear_s101 - v12_ponly_xxl`):** Mean $\Delta = -0.1555$, Nominal 95% CI $[-0.1848, -0.1261]$, Bonferroni-corrected 97.5% CI $[-0.1890, -0.1219]$. Verdict: **`hurts`** (69 improved / 103 unchanged / 220 degraded).
  - **Individual Count Analysis:** Mean delta is negative across each of the 14 individual team counts (2 through 15).
- **Formal Outcome:** **Explicit NO-ADMISSION**.
  - Both learned scoring heads statistically significantly degrade playing strength relative to the unguided $v12$ baseline across all player counts.
  - Baseline `v12_ponly_xxl` remains completely unchanged as the production incumbent. Neither candidate is admitted.

---

## 2. Task Dependency & Claim Context

- **Dependency Readiness:** T0051 specifies dependencies on T0050, T0049, T0044, and T0045. T0049 (Windows scoring-hook test .exe fix) was verified on the host and merged into `main` via PR #48 (`cdab287`). T0050 was completed and merged via PR #49 (`a0f69e8`) and PR #50 (`4c15169`).
- **Authorization:** Execution of T0050 and T0051 was explicitly authorized by the user prompt, with all work conducted on an isolated worktree (`ctf-fusion.worktrees/T0051`) and delivered via PR #51 without direct pushes to `main`.

---

## 3. Model Export and Parity Verification

Candidates from T0050 (`reports/models/own_action_first_trial/SELECTION.json`) were exported into dedicated C++ header directories via `training.export_decision_scorer`:

| Scorer Directory | Selected Model | Normalization | Parameter $\lambda$ | Source-Bound Carrier | Header SHA-256 (LF) |
|---|---|---|---:|---|---|
| `inference/own_action_scorers/first_trial_linear_s101/` | `linear_s101` | `zscore` | 50.0 | `v12_ponly_xxl_dlog` | `ff9942d614e2...` |
| `inference/own_action_scorers/first_trial_mlp_s101/` | `mlp_s101` | `zscore` | 100.0 | `v12_ponly_xxl_dlog` | `5419bb62127e...` |

### Parity and Equivalence Checks
1. **Float64 Parity:** Evaluation of exported C++ scorer code against Python NumPy implementation on the test dataset roots confirmed maximum difference $< 2 \times 10^{-16}$.
2. **Scoring Hook Unit Tests:** All 20 tests in `test_own_action_parity`, `test_own_action_scorer`, `test_decision_hook`, and `test_decision_scoring` pass with MinGW g++ on Windows.
3. **$\lambda = 0$ Equivalence:** Build with $\lambda = 0.0$ confirmed identical candidate rank choices and zero altered moves against control.

---

## 4. Host Concurrency and Deadline Preflight

The official preflight script is located in `bots/p_only/tools/preflight_own_action.py`. It evaluated 1, 2, 4, and 6 workers across team counts 2, 5, 10, and 15 under 2,000 ms initialization and 50 ms subsequent reply deadlines:

| Workers | Games | Faults | Late Replies | Dead Procs | $n=15$ Max Latency | $n=15$ p99 Latency | DSCORE Deadline Drops | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 0 | 0 | 0 | 9.88 ms | 8.06 ms | 0 | Pass |
| 2 | 16 | 0 | 0 | 0 | 10.33 ms | 8.22 ms | 0 | Pass |
| **4** | **16** | **0** | **0** | **0** | **10.02 ms** | **9.10 ms** | **0** | **Admitted (Safe)** |
| 6 | 16 | 0 | 0 | 0 | 10.15 ms | 9.27 ms | 2 | Rejected (Drop > 0) |

- **Strict Code Gating:** A worker level passes if and only if `dead == 0`, `late == 0`, `faults == 0`, `deadline_drops == 0`, `invalid == 0`, and `n15_max < 48.0 ms`. At 6 workers, 2 deadline drops occurred, failing the strict gate. Concurrency was strictly admitted and frozen at **4 workers**.
- **Preflight Seed Overlap Disclosure:** The preflight ladder generated 4 fixtures (`n02, n05, n10, n15 m00000`) derived from seed `20261801` to test host timing under realistic maps. Consequently, 388 fixtures in the screen were strictly unseen prior to measurement, while 4 fixtures were shared with preflight. As shown in Section 6, the statistical conclusions on the 388 strictly preflight-unseen fixtures are virtually identical to the 392-fixture set ($\Delta = -0.1817$ vs $-0.1828$), confirming zero sensitivity. For future tasks, reserving a dedicated separate seed for preflight is recommended.

---

## 5. Host Timing and Reliability Receipts

Screen executed under exclusive CPU lock (`task.py lock cpu --task T0051 -- python bots/p_only/tools/screen_runner.py ...`).

| Arm | Games | Faults | Dead | Late | First Max (ms) | p99 Median (ms) | p99 Worst (ms) | Max Reply (ms) | Binary SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v12_ponly_xxl` | 392 | 0 | 0 | 0 | 47.2 | 4.86 | 9.68 | 11.32 | `d8095b80b5bd` |
| `v12_ponly_xxl_first_trial_linear_s101` | 392 | 0 | 0 | 0 | 46.9 | 5.04 | 9.91 | 11.52 | `08df6938190c` |
| `v12_ponly_xxl_first_trial_mlp_s101` | 392 | 0 | 0 | 0 | 45.5 | 4.91 | 9.92 | 12.05 | `4409462c5012` |
| `v12_ponly_xxl~aa` | 392 | 0 | 0 | 0 | 49.2 | 4.95 | 9.63 | 10.86 | `d8095b80b5bd` |

All four arms executed cleanly without a single timing violation or process crash. Denominators remain complete (392/392) for all arms.

---

## 6. Paired Statistical Differences

Intervals: Map-clustered confidence intervals, team counts weighted equally, negligible margin $\pm 0.002$.

### Detailed Contrast Results

| Comparison | Band | Mean $\Delta$ | Confidence Interval | Multiplicity Correction | Verdict | Fixtures | Improved / Unchanged / Degraded |
|---|---|---:|---|---|---|---:|---|
| **`mlp_s101 - control` (Primary)** | **all** | **-0.1828** | **[-0.2114, -0.1543]** | Nominal 95% ($z=1.960$) | **`hurts`** | **392** | **56 / 95 / 241** |
| | 2 | -0.3036 | [-0.4657, -0.1415] | Nominal 95% | `hurts` | 28 | |
| | 3–5 | -0.1860 | [-0.2606, -0.1114] | Nominal 95% | `hurts` | 84 | |
| | 6–10 | -0.1589 | [-0.2008, -0.1171] | Nominal 95% | `hurts` | 140 | |
| | 11–15 | -0.1807 | [-0.2203, -0.1410] | Nominal 95% | `hurts` | 140 | |
| **`linear_s101 - control` (Secondary)** | **all** | **-0.1555** | **[-0.1848, -0.1261]** | Nominal 95% ($z=1.960$) | **`hurts`** | **392** | **69 / 103 / 220** |
| | **all** | **-0.1555** | **[-0.1890, -0.1219]** | Bonferroni 97.5% ($z=2.241$) | **`hurts`** | **392** | |
| | 2 | -0.2321 | [-0.3874, -0.0769] | Nominal 95% | `hurts` | 28 | |
| | 3–5 | -0.1493 | [-0.2300, -0.0686] | Nominal 95% | `hurts` | 84 | |
| | 6–10 | -0.1209 | [-0.1649, -0.0770] | Nominal 95% | `hurts` | 140 | |
| | 11–15 | -0.1783 | [-0.2173, -0.1393] | Nominal 95% | `hurts` | 140 | |
| **`v12_ponly_xxl~aa - control` (A/A Diagnostic)** | **all** | **+0.0000** | **[+0.0000, +0.0000]** | Nominal 95% | **`not shown`** | **392** | **0 / 392 / 0** (392/392 identical scores) |

### Sensitivity: Excluding 4 Preflight Fixtures ($M=388$)

- **Primary MLP:** Mean $\Delta = -0.1817$, 95% CI $[-0.2104, -0.1530]$, verdict: **`hurts`**.
- **Secondary Linear (Bonferroni 97.5%):** Mean $\Delta = -0.1547$, 97.5% CI $[-0.1883, -0.1211]$, verdict: **`hurts`**.
- **A/A Diagnostic:** Mean $\Delta = +0.0000$, 388/388 identical match scores.

### Per-Count Point Estimates (All 14 Counts, 28 maps each)

| Team Count | MLP $\Delta$ | Linear $\Delta$ | A/A $\Delta$ | Fixtures |
|---:|---:|---:|---:|---:|
| 2 | -0.3036 | -0.2321 | +0.0000 | 28 |
| 3 | -0.0714 | -0.1250 | +0.0000 | 28 |
| 4 | -0.1786 | -0.0952 | +0.0000 | 28 |
| 5 | -0.3080 | -0.2277 | +0.0000 | 28 |
| 6 | -0.1929 | -0.1000 | +0.0000 | 28 |
| 7 | -0.2113 | -0.2024 | +0.0000 | 28 |
| 8 | -0.1224 | -0.0791 | +0.0000 | 28 |
| 9 | -0.1808 | -0.1696 | +0.0000 | 28 |
| 10 | -0.0873 | -0.0536 | +0.0000 | 28 |
| 11 | -0.2071 | -0.2054 | +0.0000 | 28 |
| 12 | -0.0601 | -0.0617 | +0.0000 | 28 |
| 13 | -0.2336 | -0.1577 | +0.0000 | 28 |
| 14 | -0.2074 | -0.2665 | +0.0000 | 28 |
| 15 | -0.1952 | -0.2003 | +0.0000 | 28 |

---

## 7. Analysis and Discussion

1. **Hypothesized Mechanism:**
   - In offline evaluation (T0050), both linear and MLP models achieved modest validation gains on 37 validation maps, but showed negative mean deltas on the 31 test maps ($-0.015$ and $-0.020$).
   - In full multi-turn game play, the scorer alters the incumbent bot's candidate ranking on approximately 57% (Linear) to 64% (MLP) of scored decision points.
   - We hypothesize that altering incumbent actions based on 1-step offline rollout gain surrogates introduces trajectory distribution shift: subsequent game states diverge from the data-generating rollout distribution, compounding small decision errors over full game horizons.
2. **Re-run Noise and Reproducibility:**
   - Under deterministic host referee execution, the A/A control produced identical final match scores on 392 of 392 fixtures. This confirms that the observed utility drop is not host measurement noise, but reflects the true empirical outcome of the altered candidate policy under this screen's reference field.
3. **Strict Experimental Discipline:**
   - No post-hoc sweep or alternative lambda values were tried online. The pre-declared primary and secondary models were tested once and reported honestly.

---

## 8. Durable Local Backup Receipts

Per repository rules, raw logs, game results, and binary files are gitignored. To ensure durability, an atomic compressed backup was created and verified with a bit-exact restore test:

- **Backup Archive:** `data/backups/own_action_first_trial_20261015.tar.gz`
- **Archive Size:** 39,882,654 bytes (38.0 MB)
- **Archive SHA-256:** `4a10a9a7ab3b33c7bb90c2d3f9dd366cea12e432b0df25bc6801040286514203`
- **Archived Contents:**
  1. `data/p_only/own_action_first_trial/schedule.json` (`67a68fd19c0c...`)
  2. `data/p_only/own_action_first_trial/identities.json` (`a15fe1907848...`)
  3. `data/p_only/own_action_first_trial/results.jsonl` (`104cecaab122...`)
  4. `build/p_only/manifest.json` (`f3769c2794eb...`)
  5. `build/p_only/bin/v12_ponly_xxl-18040d1a1eee.exe` (`d8095b80b5bd...`)
  6. `build/p_only/bin/v12_ponly_xxl_aa-45032da7a751.exe` (`6a11381ff846...`)
  7. `build/p_only/bin/v12_ponly_xxl_first_trial_linear_s101-f910d07a89a7.exe` (`08df6938190c...`)
  8. `build/p_only/bin/v12_ponly_xxl_first_trial_mlp_s101-b4474b0a2d7b.exe` (`4409462c5012...`)
- **Restore Verification Check:** Verified pass. Full archive unpacks into an isolated temporary directory with byte-for-byte identical SHA-256 digests across all 8 files.

---

## 9. Admission Decision and Handoff

- **Decision:** **REJECTED_NO_ADMISSION**.
- **Action:** Production bot defaults, carrier logic, and baseline configurations remain completely untouched. Neither `first_trial_mlp_s101` nor `first_trial_linear_s101` will be promoted to main or substituted for `v12_ponly_xxl`.
- **Status:** Handed off to `review`.
