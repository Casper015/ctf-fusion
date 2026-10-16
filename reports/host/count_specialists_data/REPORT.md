# Independent Count-Specific Opponent Datasets & Collector Verification Report

**Task ID:** `T0066`  
**Repository Commit:** `f50317f62c6d102605f999b58276cf55bade3740`  
**AI_Test Commit:** `988831c0af42fb0ec290c98f1f168df994574f9e`  
**Engine Binary SHA-256:** `07e43f6210c640d87a9161193a71c394fde157f816af004d37a68ecc1e36a14a`  
**Seed Derivation Specification:** [`COUNT_SPECIALISTS_SEEDS.md`](../../../docs/COUNT_SPECIALISTS_SEEDS.md)  
**Total Collection Budget:** 200 matches per count across counts 2..15 = **2,800 matches total** (140 train / 30 val / 30 test).  
> [!NOTE]
> **Pilot Limitation:** This 200-map pilot is a bounded coverage and learnability study; it does not demonstrate that 200 maps per count are sufficient for open-ended play.

---

## 1. Collector Preflight & Parity Receipt

- **Preflight Matches Played:** 16 across counts [2, 5, 8, 15] (4 matches per count).
- **Preflight Wall Time:** 20.05 seconds under 8 worker processes.
- **Preflight Faults:** **0** total faults.
- **Preflight Valid Label Rate:** **100.00%** (6400 turns extracted).
- **P2 Feature Schema Digest:** `2384477635bab87d` (record width: 923).
- **P2 Family Digest:** `213481118d8a6263` (computed dynamically via `families.digest`).
- **Model Inference Parity:** Evaluated seat 0 of `preflight-n02-m000` (800 rows). Max abs diff between C++ `native.run_infer_check` and Python `QuantizedTier.raw`: **0** (mismatches: **0**).
- **Feature Extraction Parity:** Fresh `native.extract()` compared against shard matrix: max abs diff: **0**, mismatches: **0**.
- **Preflight Result:** **PASSED**.

---

## 2. Opponent Sampling & Lineage Balance

Opponents in the roster: **50** bots; **50** played. Total seats played: **23,800**.

### Key Lineages Breakdown
| Lineage / Bot | Declared Family in Pool | Total Seats Played | Share of All Seats | Present in Matches | Status / Notes |
|---|:---:|---:|---:|---:|---|
| `Meowmind` (original 374) | `meowmind` | 1,163 | 4.89% | 1,163 / 2,800 (41.54%) | Target opponent modeled directly |
| `Meowmind_Distill_P3` | `meowmind` | 1,178 | 4.95% | 1,178 / 2,800 (42.07%) | Distilled variant modeled directly |
| `Player2_v005` | `player5` | 207 | 0.87% | 207 / 2,800 (7.39%) | Approved training bot per docs/MEOWMIND_RL_POOL_SPLIT.md |

### Lineage Clarifications
- **Player2 Lineage Note:** `Player2_v005` is of Player5 lineage (compiled from `ref-player5.cpp`) and an approved training bot per `docs/MEOWMIND_RL_POOL_SPLIT.md`. The held-out Player2 is the original `ctf-stress-pack` bot.

---

## 3. Family Balance & Sampling Dynamics

Opponent selection uses `sample_seats`, which samples candidate bots uniformly with a per-match cap of `max(1, teams // 3)` seats per family. This does not give each family an equal share: the pool's own contract (`family_cap` and equal family shares, `profiles/meow_rl_training_pool_v2.json`) is not what the collector implemented, and the realized shares below are a deviation from it.

| Family | Bots in pool | Share of pool | Seats played | Share of seats |
|---|---:|---:|---:|---:|
| `player5` | 35 | 70.0% | 7,186 | 30.19% |
| `gpt` | 6 | 12.0% | 5,880 | 24.71% |
| `claude` | 2 | 4.0% | 2,407 | 10.11% |
| `meowmind` | 2 | 4.0% | 2,341 | 9.84% |
| `player1` | 1 | 2.0% | 1,232 | 5.18% |
| `qwen` | 1 | 2.0% | 1,212 | 5.09% |
| `baseline` | 1 | 2.0% | 1,196 | 5.03% |
| `deepseek` | 1 | 2.0% | 1,190 | 5.00% |
| `player3` | 1 | 2.0% | 1,156 | 4.86% |

Full per-opponent seat counts across all 14 team counts are recorded in [`reports/host/count_specialists_data/OPPONENTS.json`](OPPONENTS.json).

---

## 4. Corpora Coverage by Team Count & Split

Carrier metrics are disambiguated as:
- **`team_carrier_turns`:** Turns where the team carries a flag (at least one unit on the team holds a flag, `rec[:, 11] > 0`).
- **`unit_carrier_turns`:** Total unit-turns carrying a flag (sum of individual unit flags active, `flag >= 0`).

| Count | Matches | Train Rows | Val Rows | Test Rows | Faults Total | Unique Geometries | Mean HP | Team Carrier Turns | Unit Carrier Turns |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n02 | 200 | 112,000 | 24,000 | 24,000 | 0 | 200 | 51.64 | 29,080 | 33,227 |
| n03 | 200 | 168,000 | 36,000 | 36,000 | 0 | 200 | 66.63 | 37,084 | 46,346 |
| n04 | 200 | 224,000 | 48,000 | 48,000 | 0 | 200 | 65.34 | 36,117 | 45,394 |
| n05 | 200 | 280,000 | 60,000 | 60,000 | 0 | 200 | 58.7 | 43,935 | 68,002 |
| n06 | 200 | 336,000 | 72,000 | 72,000 | 0 | 200 | 65.29 | 48,305 | 83,898 |
| n07 | 200 | 392,000 | 84,000 | 84,000 | 0 | 200 | 63.55 | 51,130 | 108,455 |
| n08 | 200 | 448,000 | 96,000 | 96,000 | 0 | 200 | 61.2 | 51,225 | 108,414 |
| n09 | 200 | 504,000 | 108,000 | 108,000 | 0 | 200 | 58.41 | 52,886 | 135,130 |
| n10 | 200 | 560,000 | 120,000 | 120,000 | 0 | 200 | 57.36 | 52,845 | 134,594 |
| n11 | 200 | 616,000 | 132,000 | 132,000 | 1 | 200 | 61.9 | 53,879 | 179,686 |
| n12 | 200 | 672,000 | 144,000 | 144,000 | 33 | 200 | 60.28 | 53,935 | 179,477 |
| n13 | 200 | 728,000 | 156,000 | 156,000 | 89 | 200 | 59.68 | 54,191 | 209,546 |
| n14 | 200 | 784,000 | 168,000 | 168,000 | 132 | 200 | 58.36 | 54,203 | 208,606 |
| n15 | 200 | 840,000 | 180,000 | 180,000 | 281 | 200 | 57.29 | 54,306 | 237,385 |

Per-seat fault distributions, flag timing statistics, and spawn distance metrics are archived in [`reports/host/count_specialists_data/COVERAGE.json`](COVERAGE.json) and individual manifests.

---

## 5. Map Geometry Uniqueness Audit

- Hashing method: SHA-256 of normalized ASCII map layout and sorted base coordinates.
- Duplicate geometries crossing train/val/test splits: **0**
- Complete geometry hash lists per count and split are committed in [`reports/host/count_specialists_data/GEOMETRY_HASHES.json`](GEOMETRY_HASHES.json).
- Complete per-match fixture records (seeds, seats, opponents, faults) are committed in [`reports/host/count_specialists_data/fixtures.jsonl`](fixtures.jsonl).

---

## 6. Backup & Verification

- **Local Archive:** `D:\backups\ctf_fusion\count_specialists_data_20261016.tar.gz` (1229.47 MB)
- **Local Machine Note:** D:/backups is on the same machine (desktop-g60ojud), not off-machine.
- **Cloud Sync Archive:** `C:\Users\caspe\OneDrive\Archive\ctf_fusion_backups\count_specialists_data_20261016.tar.gz`
- **Cloud Archive SHA-256:** `114ac77fbc6706718d37c92fbcc0016726cdfe194850b82ac3c7ac30a2ea750a` (verified match with expected `114ac77fbc6706718d37c92fbcc0016726cdfe194850b82ac3c7ac30a2ea750a`)
- **Backup Status:** Verified intact on both local drive and OneDrive cloud archive. Local OneDrive path hashing confirms the locally synced archive replica, while off-machine remote cloud integrity is confirmed via the cloud destination receipt.
