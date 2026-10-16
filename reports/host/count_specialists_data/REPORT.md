# Independent Count Specialists Dataset Report (T0066)

**Date:** 2026-10-16  
**Host:** `desktop-g60ojud` (Windows 10, RTX 4080 Laptop GPU, 16C/32T)  
**Task:** T0066 (`Collect independent count-specific opponent datasets`)  
**Matches Total:** 2800 matches (200 maps per count for team counts 2..15)  
**Engine Binary SHA-256:** `07e43f6210c640d87a9161193a71c394fde157f816af004d37a68ecc1e36a14a`  
**Extractor Binary SHA-256:** `4073bea143fc5b771ce8e96f9acb00cf2972a5a2c6b7c6318955638d12db6e1d`  
**Python Collector SHA-256:** `cce21c0e8710e304273e98ac4b2b0f5aceb639acbcd2343cbaa5c4e81619c45b`  
**Workers:** 8 concurrent workers under task CPU lease  

> [!IMPORTANT]
> **Pilot Insufficiency Statement:** This 200-map per count dataset (2,800 matches total) is a learnability and coverage pilot, not evidence that the data is sufficient for asymptotic policy convergence or complete tactical generalization. A future PPO task must collect its own on-policy trajectories.

---

## 1. Summary & Exact Execution Commands

Task T0066 collects 14 distinct, independent, count-specific opponent corpora (`n02` through `n15`) on desktop-g60ojud:
- 200 fresh maps per count (14 counts = 2,800 matches total).
- Map-disjoint splits: 140 train / 30 validation / 30 test per count (70% / 15% / 15%).
  - *Split implementation note:* `training/dataset.py` defines `SPLITS = (('train', 0.8), ('val', 0.1), ('test', 0.1))`. T0066 implements the 140/30/30 map-disjoint split requested in the task contract (map index $0..139$ train, $140..169$ val, $170..199$ test) so all seats, turns, and prefixes of a map stay strictly together.
- Globally disjoint seed derivations registered in [`docs/COUNT_SPECIALISTS_SEEDS.md`](../../../docs/COUNT_SPECIALISTS_SEEDS.md).
- All-seat fault total: **536** across all 2,800 matches (all faults filtered out of valid labels).
- Normalized map geometry audit: **0 duplicate geometries cross train/val/test splits**.

### Exact Reproduction Commands
```powershell
# 1. Run 8-worker collector preflight and bit-exact native/Python parity check:
python tools/repo/task.py lock cpu --task T0066 -- python tools/host/count_specialists_data.py --preflight

# 2. Run full 2,800 matches collection and P2 feature extraction (under task CPU lease):
python tools/repo/task.py lock cpu --task T0066 -- python tools/host/count_specialists_data.py --collect

# 3. Regenerate all manifests, fixture list, audit records and reports from existing data/:
python tools/host/count_specialists_data.py --regenerate

# 4. Verify local and cloud backup archives:
python tools/host/count_specialists_data.py --backup
```

---

## 2. Eight-Worker Preflight Gate & Parity Receipt

- Status: **PASSED**
- Matches played: 16 across counts [2, 5, 8, 15] (seed 20262099)
- Total turns: 6400 | Faults: 0
- **Native-vs-Python Integer Parity Receipt:**
  - Model evaluated: `fusion::models::general (p2/v2 XXL)`
  - Rows compared: **800** (seat 0 of fixture `preflight-n02-m000`, 400 turns $\times 2$ teams)
  - Subsampling explanation: Seat 0 of preflight-n02-m000 (turns 1..400 x 2 teams = 800 rows) evaluated for exact bitwise parity without redundant repeated evaluation of all 48,000 seat-turn combinations.
  - Max absolute difference: **0**
  - Mismatches count: **0** (all integer accumulators equal: **True**)
  - Feature record width: **923**
  - Schema digest: `2384477635bab87d`
  - P2 family digest: `213481118d8a6263`

---

## 3. Opponent Sampling & Key Lineage Audit

Total opponents in roster: **50** bots. Total seats played: **23,800** seats.

### Key Lineages Breakdown
| Lineage / Bot | Declared Family in Pool | Total Seats Played | Share of All Seats | Present in Matches | Status / Notes |
|---|:---:|---:|---:|---:|---|
| `Meowmind` (original 374) | `meowmind` | 1,163 | 4.89% | 1,154 / 2,800 (41.2%) | Target opponent modeled directly |
| `Meowmind_Distill_P3` | `meowmind` | 1,178 | 4.95% | 1,166 / 2,800 (41.6%) | Distilled variant modeled directly |
| `Player2_v005` | `player5` | 207 | 0.87% | 207 / 2,800 (7.4%) | Sampled due to pool v2 inclusion (see note below) |

> [!WARNING]
> **Player2 Contradiction Disclosure:** Player2_v005 is included in profiles/meow_rl_training_pool_v2.json under family 'player5' despite the pool file text declaring the Player2 lineage held out. The collector sampled strictly according to this approved pool without editing the pool file.

Full per-opponent seat counts across all 14 team counts are recorded in [`reports/host/count_specialists_data/OPPONENTS.json`](OPPONENTS.json).

---

## 4. Corpora Coverage by Team Count & Split

| Count | Matches | Train Rows | Val Rows | Test Rows | Faults Total | Unique Geometries | Mean HP | Pickup Events | Drop Events | Spawn Center Dist |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n02 | 200 | 112,000 | 24,000 | 24,000 | 0 | 200 | 51.64 | 8068 | 0 | 22.0 |
| n03 | 200 | 168,000 | 36,000 | 36,000 | 0 | 200 | 66.63 | 7891 | 9 | 29.33 |
| n04 | 200 | 224,000 | 48,000 | 48,000 | 0 | 200 | 65.34 | 8219 | 10 | 30.0 |
| n05 | 200 | 280,000 | 60,000 | 60,000 | 0 | 200 | 58.7 | 13153 | 16 | 28.4 |
| n06 | 200 | 336,000 | 72,000 | 72,000 | 0 | 200 | 65.29 | 10576 | 9 | 40.0 |
| n07 | 200 | 392,000 | 84,000 | 84,000 | 0 | 200 | 63.55 | 14587 | 12 | 39.86 |
| n08 | 200 | 448,000 | 96,000 | 96,000 | 0 | 200 | 61.2 | 15699 | 19 | 40.0 |
| n09 | 200 | 504,000 | 108,000 | 108,000 | 0 | 200 | 58.41 | 20417 | 21 | 39.0 |
| n10 | 200 | 560,000 | 120,000 | 120,000 | 0 | 200 | 57.36 | 21768 | 37 | 40.0 |
| n11 | 200 | 616,000 | 132,000 | 132,000 | 1 | 200 | 61.9 | 21729 | 15 | 47.73 |
| n12 | 200 | 672,000 | 144,000 | 144,000 | 33 | 200 | 60.28 | 22929 | 25 | 48.0 |
| n13 | 200 | 728,000 | 156,000 | 156,000 | 89 | 200 | 59.68 | 26626 | 29 | 47.77 |
| n14 | 200 | 784,000 | 168,000 | 168,000 | 132 | 200 | 58.36 | 27995 | 31 | 48.0 |
| n15 | 200 | 840,000 | 180,000 | 180,000 | 281 | 200 | 57.29 | 32090 | 26 | 48.2 |

Per-seat fault distributions and detailed flag return/cooldown timing statistics are archived in [`reports/host/count_specialists_data/COVERAGE.json`](COVERAGE.json) and individual manifests.

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
- **Cloud Archive SHA-256:** `114ac77fbc6706718d37c92fbcc0016726cdfe194850b82ac3c7ac30a2ea750a` (verified match with local)
- **Backup Status:** Verified intact on both local drive and OneDrive cloud archive.
