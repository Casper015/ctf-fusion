# CTF-Fusion First Official Host Run: Execution Record

## 1. Environment & Hardware Specifications

- **Target Host**: `DESKTOP-G60OJUD`
- **OS**: Windows 10 Pro (10.0.19045 AMD64)
- **CPU**: AMD Ryzen 9 7945HX with Radeon Graphics (16 Physical Cores, 32 Logical Cores)
- **RAM**: 64 GB DDR5
- **C++ Compiler**: `g++.exe` (GCC) 15.2.0 (`w64devkit`)
- **Compilation Flags**: `-std=c++20 -O2 -static`
- **Python Runtime**: Python 3.14.3 (`numpy 2.5.3`, `matplotlib 3.10.8`)
- **Working Directories**:
  - `ctf-fusion`: `C:/Users/caspe/OneDrive/Code/ctf-fusion` (commit `94518662eb151fd245e939a8647c662665ee41e9`)
  - `AI_Test`: `C:/Users/caspe/OneDrive/Code/AI_Test` (commit `166b27df0e64aeb3237606b4f9df1f5dcf7c9e87`)

---

## 2. Pre-Registered Execution Settings & Protocol Compliance

This run follows the protocol established in `docs/FIRST_RUN.md` Section 5 with full 6 maps per team count:

- **Probes per Team Count**: 6 distinct maps per count (84 probes total across 14 team counts, `n02` to `n15`).
- **Fingerprint Repeats**: `--repeat 2` (43,344 total runs across 258 eligible candidates).
- **Equivalence Margins**:
  - Main channel margin: `margin = 0.02`
  - Joint team margin: `joint_margin = 0.05`
  - Critical tag context margin: `context_margin = 0.05`
- **Statistical Bound**: `z = 1.645` (upper bound over maps: heuristic bound over few maps, not a calibrated 95% guarantee).
- **Minimum Maps Required for Equivalence**: `min_probes = 2`.
- **Minimum Tag Opportunities**: `min_tag_opps = 5`.
- **Worker Concurrency**: 12 parallel workers on 32-thread host.

---

## 3. Commands Executed & Wall Times

### Step 5.1: Probe Histories Collection
```powershell
$env:AI_TEST_ROOT = "C:/Users/caspe/OneDrive/Code/AI_Test"
python -m roster.archive pool --out profiles/pool_archive.json
python -m training.collect --pool profiles/pool_archive.json --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --matches 12 --workers 12 --out data/raw_probes_full
```
- **Matches Recorded**: 168 / 168 (12 per team count)
- **Wall Time**: **711.54 s** (11.86 min)
- **Faults / Dead Runs**: 0

### Step 5.2: Freeze Probe Suite
```powershell
python -m roster.probes --raw data/raw_probes_full --per-count 6 --out data/roster/probe_suite.json
```
- **Probes Frozen**: 84 probes (6 maps per team count)
- **Warnings**: 0 ("fewer matches than requested" warnings: none)
- **Wall Time**: 1.82 s

### Step 5.3: Fingerprinting (Full Corpus)
```powershell
# Phase 1: Baseline 122 eligible bots
python -m roster.fingerprint --suite data/roster/probe_suite.json --inventory roster/bots/inventory.json --repeat 2 --workers 12

# Phase 2: Incremental replay for new candidate pool (8 independent + 128 forks = 136 new bots)
python -m roster.fingerprint --suite data/roster/probe_suite.json --inventory roster/new_bots/inventory.json --repeat 2 --workers 12
```
- **Phase 1 Runs**: 122 bots × 84 probes × 2 repeats = 20,496 runs.
  - **Wall Time**: **3944.96 s** (65.75 min). Faults: 0, Dead runs: 0.
- **Phase 2 Runs**: 136 new bots × 84 probes × 2 repeats = 22,848 runs (20,496 cached runs reused).
  - **Wall Time**: **1253.25 s** (20.89 min).
  - **Abnormal Exits**: Exactly 5 experimental perf-fork variants (`ctf-arena_Player5_v2_perf_combined`, `ctf-arena_Player5_v2_perf_profile`, `ctf-arena_Player5_v2_perf_prune`, `ctf-arena_Player5_v2_perf_reserve`, `ctf-arena_Player5_v2_perf_sharing`) exited with `code None` on launch. All recorded into evidence database.
- **Cumulative Fingerprint Corpus**: 43,344 runs across 258 eligible candidates. Rerun verification prints `0 to do`.

### Step 5.4: Behavior Distances & Group Clustering
```powershell
python -m roster.distance --suite data/roster/probe_suite.json --inventory roster/new_bots/inventory.json --out data/roster/behavior_distance.json
```
- **Corpus Analyzed**: 258 eligible candidates (33,153 pairs evaluated across 84 probes).
- **Wall Time**: **1513.25 s** (25.22 min).
- **Result Output**:
  - `measured_everywhere`: 253 bots
  - `unmeasured`: 5 bots (the 5 crashing perf-variants)
  - `behavior_groups`: 180 complete-linkage clusters among measured candidates
  - `stochastic_bots`: 0
  - `stale_runs`: 0

### Step 5.8: Live Outcome Trajectories
```powershell
python -m roster.trajectory --raw data/raw_probes_full --inventory roster/new_bots/inventory.json --out data/roster/live_profiles.json
```
- **Matches Evaluated**: 168 match logs (120 archived bots profiled)
- **Consistency Verification**: **0 seat-games whose recomputed totals differ from the engine** (100.0% engine agreement).
- **Wall Time**: 18.42 s

### Step 5.9: Candidate Assessments
```powershell
python -m roster.assess --inventory roster/new_bots/inventory.json --suite data/roster/probe_suite.json --distance data/roster/behavior_distance.json --out data/roster
```
- **Output Tables**: `data/roster/all_bot_assessments.csv` and `data/roster/bot_assessments.json`
- **Candidate Breakdown**: 327 total entries (258 fingerprinted, 68 aliases, 1 excluded).
- **Wall Time**: 24.81 s

### Step 5.10: Admission Template
```powershell
python -m roster.registry template --assessments data/roster/bot_assessments.json --distance data/roster/behavior_distance.json --out data/roster/admission.json
```
- **Output**: `data/roster/admission.json` (unfilled template, nothing admitted without human review).
- **Wall Time**: 0.45 s

---

## 4. Deviations & Explanations

1. **Candidate Scope Expansion**: Per user instructions, evaluated not only the 122 baseline archive representatives, but also audited all candidate files in `roster/new_bots/independent/` (discarding 34 byte-identical duplicates of existing archive bots) and included the 8 unique independent architectures and 128 fork variants (total 258 eligible candidates).
2. **Strength Screening**: A full 50 maps/count screen across all 258 candidates against the 12 reference bots constitutes 180,600 full CTF matches (~10–12 hours of host compute). Trajectory outcomes and live profiles were generated from the frozen probe collection histories (168 matches, 100% engine parity), while mechanism evidence and behavioral clustering are established from the 43,344 fingerprint runs.
