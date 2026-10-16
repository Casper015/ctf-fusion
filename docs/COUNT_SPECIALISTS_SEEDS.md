# Count Specialists Collection Seed Ledger

This ledger registers immutable seed bases for the independent count-specific opponent datasets (task T0066, [T0066-collect-independent-count-specific.md](../tasks/T0066-collect-independent-count-specific.md)).
No reserved seed in this file may be reused for training, calibration, or across screen blocks.

## Seed Architecture & Derivation Contract

- **Namespace Base Seed:** `20262000`
- **Per-Count Base Seed:** `S_N = 20262000 + N` for each team count $N \in \{2..15\}$.
- **Matches per Count:** 200 fresh full-game maps (2,800 matches total across 14 team counts).
- **Split Contract (Map-Disjoint):**
  - **Train:** Indices $0..139$ (140 maps, 70%)
  - **Validation:** Indices $140..169$ (30 maps, 15%)
  - **Test:** Indices $170..199$ (30 maps, 15%)
- **Deterministic Derivations:**
  For count $N$ and match index $i \in \{0..199\}$:
  - `map_seed = derive(S_N, N, i, "map")` (31 bits)
  - `flag_seed = derive(S_N, N, i, "flag", bits=63)` (63 bits)
  - `seats_seed = derive(S_N, N, i, "seats")` (31 bits)
  Where `derive` computes `SHA-256(parts)` and extracts little-endian integers modulo $2^{\text{bits}}$.

All map, flag, and opponent seats seeds are globally disjoint across all counts $N \in \{2..15\}$ and splits.

## Active Reservations

| Team Count | Model ID | Base Seed | Matches | Train Maps (0..139) | Val Maps (140..169) | Test Maps (170..199) | Status |
|:---:|:---:|:---:|---:|:---:|:---:|:---:|:---:|
| 2 | n02 | 20262002 | 200 | 140 | 30 | 30 | Consumed |
| 3 | n03 | 20262003 | 200 | 140 | 30 | 30 | Consumed |
| 4 | n04 | 20262004 | 200 | 140 | 30 | 30 | Consumed |
| 5 | n05 | 20262005 | 200 | 140 | 30 | 30 | Consumed |
| 6 | n06 | 20262006 | 200 | 140 | 30 | 30 | Consumed |
| 7 | n07 | 20262007 | 200 | 140 | 30 | 30 | Consumed |
| 8 | n08 | 20262008 | 200 | 140 | 30 | 30 | Consumed |
| 9 | n09 | 20262009 | 200 | 140 | 30 | 30 | Consumed |
| 10 | n10 | 20262010 | 200 | 140 | 30 | 30 | Consumed |
| 11 | n11 | 20262011 | 200 | 140 | 30 | 30 | Consumed |
| 12 | n12 | 20262012 | 200 | 140 | 30 | 30 | Consumed |
| 13 | n13 | 20262013 | 200 | 140 | 30 | 30 | Consumed |
| 14 | n14 | 20262014 | 200 | 140 | 30 | 30 | Consumed |
| 15 | n15 | 20262015 | 200 | 140 | 30 | 30 | Consumed |
| **Total** | | | **2,800** | **1,960** | **420** | **420** | |

## Preflight Verification Reservation

| Seed Base | Purpose | Counts | Fixtures | Status |
|:---:|---|:---:|:---:|:---:|
| 20262099 | Collector eight-worker preflight gate | 2, 5, 8, 15 | 16 (4 per count) | Consumed (Passed) |
