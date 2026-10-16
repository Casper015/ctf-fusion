# Independent Count Specialists Dataset Report (T0066)

**Date:** 2026-10-15  
**Host:** `desktop-g60ojud` (Windows 10, RTX 4080 Laptop GPU, 16C/32T)  
**Task:** T0066 (`Collect independent count-specific opponent datasets`)  
**Matches Total:** 2800 matches (200 maps per count for team counts 2..15)  
**Engine SHA-256:** `07e43f6210c640d87a9161193a71c394fde157f816af004d37a68ecc1e36a14a`  
**Workers:** 8 concurrent workers under task CPU lease  

---

## 1. Summary

Task T0066 collects 14 distinct, independent, count-specific opponent corpora (`n02` through `n15`) on desktop-g60ojud:
- 200 fresh maps per count (14 counts = 2,800 matches total).
- Map-disjoint splits: 140 train / 30 validation / 30 test per count.
- Globally disjoint seed derivations registered in [`docs/COUNT_SPECIALISTS_SEEDS.md`](file:///C:/Users/caspe/OneDrive/Code/ctf-fusion/docs/COUNT_SPECIALISTS_SEEDS.md).
- Opponents drawn strictly from approved training roster (`profiles/meow_rl_training_pool_v2.json`), keeping held-out testing models (`Fable`, `MiMo`, `Kimi`, `GLM`, `Player2`, `Player4`, `Player6`) unpolluted.
- All-seat fault total: **536** across all 2,800 matches.
- Normalized map geometry audit: **0 duplicate geometries cross train/val/test splits**.

---

## 2. Eight-Worker Preflight Gate

- Status: **PASSED**
- Matches played: 16 across counts [2, 5, 8, 15] (seed 20262099)
- Total turns: 6400 | Faults: 0
- C++/Python feature extraction parity: 100% valid labels (shape [800, 923])

---

## 3. Corpora & Shard Coverage by Team Count

| Count | Total Matches | Train Maps | Val Maps | Test Maps | Train Rows | Val Rows | Test Rows | Unique Geometries |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n02 | 200 | 140 | 30 | 30 | 112,000 | 24,000 | 24,000 | 200 |
| n03 | 200 | 140 | 30 | 30 | 168,000 | 36,000 | 36,000 | 200 |
| n04 | 200 | 140 | 30 | 30 | 224,000 | 48,000 | 48,000 | 200 |
| n05 | 200 | 140 | 30 | 30 | 280,000 | 60,000 | 60,000 | 200 |
| n06 | 200 | 140 | 30 | 30 | 336,000 | 72,000 | 72,000 | 200 |
| n07 | 200 | 140 | 30 | 30 | 392,000 | 84,000 | 84,000 | 200 |
| n08 | 200 | 140 | 30 | 30 | 448,000 | 96,000 | 96,000 | 200 |
| n09 | 200 | 140 | 30 | 30 | 504,000 | 108,000 | 108,000 | 200 |
| n10 | 200 | 140 | 30 | 30 | 560,000 | 120,000 | 120,000 | 200 |
| n11 | 200 | 140 | 30 | 30 | 616,000 | 132,000 | 132,000 | 200 |
| n12 | 200 | 140 | 30 | 30 | 672,000 | 144,000 | 144,000 | 200 |
| n13 | 200 | 140 | 30 | 30 | 728,000 | 156,000 | 156,000 | 200 |
| n14 | 200 | 140 | 30 | 30 | 784,000 | 168,000 | 168,000 | 200 |
| n15 | 200 | 140 | 30 | 30 | 840,000 | 180,000 | 180,000 | 200 |

---

## 4. Backup & Verification

- **Local Archive:** `D:\backups\ctf_fusion\count_specialists_data_20261016.tar.gz` (1229.47 MB)
- **Archive SHA-256:** `114ac77fbc6706718d37c92fbcc0016726cdfe194850b82ac3c7ac30a2ea750a`
- **Verification:** Integrity verified on disk and synchronized to cloud backup (`C:\Users\caspe\OneDrive\Archive\ctf_fusion_backups\count_specialists_data_20261016.tar.gz`).
