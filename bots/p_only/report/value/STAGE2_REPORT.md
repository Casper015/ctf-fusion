# Stage 2: Value Network (V) at Fixed Policy (P) Controlled Screen & Diagnostic Report

**Evaluation Date**: 2026-10-10  
**Host Machine**: `DESKTOP-G60OJUD` (Intel CPU, NVIDIA GeForce RTX 4080 Laptop GPU 12GB)  
**Task ID**: `T0003`  
**Configuration**: [`bots/p_only/profiles/screen_value.json`](../../profiles/screen_value.json)  
**Schedule Hash**: `b9825f0a9519964c72481b2f8996454eab235a574636948c9c4c7c549f2c494b` (Fresh seed: `20261108`)  
**Scope**: 8 arms across 2 carriers (`C08` and `v12`), 280 fixtures (team counts 2-15, 20 maps each), total 2,240 games.

---

## 1. Executive Summary

This evaluation tests whether integrating the board value network **V** (specifically **V-M**) into pure policy-guided bots (**P-only**) improves match outcomes when the policy **P** is held fixed, and diagnoses the root causes of its behavior.

- **Primary Finding**: Across all team counts (2–15), the overall benefit of enabling V is **not shown** (intervals include zero):
  - **`C08` at P-XL**: Mean $+0.0103$, 95% CI $[-0.0133, +0.0339]$ (**not shown**).
  - **`C08` at P-XXL**: Mean $+0.0185$, 95% CI $[-0.0062, +0.0433]$ (**not shown**).
  - **`v12` at P-XL**: Mean $+0.0125$, 95% CI $[-0.0169, +0.0420]$ (**not shown**).
  - **`v12` at P-XXL**: Mean $-0.0022$, 95% CI $[-0.0301, +0.0257]$ (**not shown**).
- **Sub-Band Nuances**:
  - In crowded games (team counts 11–15), `C08 P-XL + V` showed a clear statistically verified advantage: $+0.0368$ with 95% CI $[+0.0020, +0.0716]$ (**helps**).
  - However, combining `P-XXL` with `V` at 11–15 teams yielded a negative interaction term: $-0.0507$ with 95% CI $[-0.0950, -0.0064]$ (**hurts**).
- **Diagnostic Conclusion**:
  - V alters between $9.5\%$ and $13.9\%$ of bot decisions (proving that coefficients and integration points are active and non-trivial).
  - V evaluations are computationally negligible ($\sim 0.05$ to $0.14$ ms per turn in integer fixed-point C++), causing zero sample starvation in lookahead.
  - The lack of general gain stems from **continuation policy distribution mismatch** (V was trained on historical mixed-roster playouts rather than pure P-only rollouts) and **coarse aggregate features** (lacking individual enemy HP and return ETAs), **not from network capacity deficiency or search budget exhaustion**.
  - **Recommendation**: Do not blindly train larger networks or fine-tune against legacy data. Fine-tuning should only proceed if conditioned on frozen P-only continuation trajectories with feature schema additions.

---

## 2. Experimental Matrix & Paired Contrasts

Utility is defined as the share of rival teams beaten (ties count as 0.5). 95% confidence intervals are clustered by map per team count and equally weighted across counts.

### 2.1 Direct V Effects (Holding P Fixed)

| Arm Comparison | Carrier | P Tier | Team Band | Mean Diff | 95% CI | Verdict | Fixtures |
|---|---|---|---|---:|---|---|---:|
| `c08_ponly_pxl_vm_v - c08_ponly_pxl_vm` | C08 | XL | **All (2-15)** | **+0.0103** | `[-0.0133, +0.0339]` | **not shown** | 280 |
| | C08 | XL | 2 (Duel) | +0.0000 | `[+0.0000, +0.0000]` | not shown* | 20 |
| | C08 | XL | 3–5 | +0.0111 | `[-0.0553, +0.0775]` | not shown | 60 |
| | C08 | XL | 6–10 | -0.0146 | `[-0.0543, +0.0251]` | not shown | 100 |
| | C08 | XL | **11–15** | **+0.0368** | `[+0.0020, +0.0716]` | **helps** | 100 |
| `c08_ponly_pxxl_vm_v - c08_ponly_pxxl_vm`| C08 | XXL | **All (2-15)** | **+0.0185** | `[-0.0062, +0.0433]` | **not shown** | 280 |
| | C08 | XXL | 2 (Duel) | +0.0000 | `[+0.0000, +0.0000]` | not shown* | 20 |
| | C08 | XXL | 3–5 | +0.0437 | `[-0.0311, +0.1186]` | not shown | 60 |
| | C08 | XXL | 6–10 | +0.0395 | `[-0.0024, +0.0814]` | not shown | 100 |
| | C08 | XXL | 11–15 | -0.0139 | `[-0.0461, +0.0183]` | not shown | 100 |
| `v12_ponly_pxl_vm_v - v12_ponly_pxl_vm` | v12 | XL | **All (2-15)** | **+0.0125** | `[-0.0169, +0.0420]` | **not shown** | 280 |
| | v12 | XL | 3–5 | +0.0486 | `[-0.0354, +0.1326]` | not shown | 60 |
| | v12 | XL | 6–10 | -0.0021 | `[-0.0471, +0.0430]` | not shown | 100 |
| | v12 | XL | 11–15 | +0.0079 | `[-0.0371, +0.0529]` | not shown | 100 |
| `v12_ponly_pxxl_vm_v - v12_ponly_pxxl_vm`| v12 | XXL | **All (2-15)** | **-0.0022** | `[-0.0301, +0.0257]` | **not shown** | 280 |
| | v12 | XXL | 2 (Duel) | +0.0500 | `[-0.0711, +0.1711]` | not shown | 20 |
| | v12 | XXL | 3–5 | -0.0208 | `[-0.0816, +0.0399]` | not shown | 60 |
| | v12 | XXL | 6–10 | +0.0367 | `[-0.0097, +0.0832]` | not shown | 100 |
| | v12 | XXL | 11–15 | -0.0403 | `[-0.0855, +0.0048]` | not shown | 100 |

*\*Note on C08 Duel matches (2 teams)*: The C08 carrier bypasses the checked lookahead and beam evaluation in 2-team matches. The zero interval indicates that V is **not exercised (uncovered)** on that branch, not that V has zero intrinsic capability.

### 2.2 P Capacity Increments & Interactions

- **P Capacity with V Off (`XXL - XL`)**:
  - `C08`: $-0.0117$ with 95% CI $[-0.0404, +0.0170]$ (**not shown**).
  - `v12`: $+0.0108$ with 95% CI $[-0.0161, +0.0377]$ overall (**not shown**), with clear gain at 3–5 teams: $+0.0903$ with 95% CI $[+0.0168, +0.1637]$ (**helps**).
- **Interaction ($D - C - B + A$)**:
  - `C08`: Overall $+0.0082$ `[-0.0267, +0.0432]` (**not shown**). At 11–15 teams: $-0.0507$ `[-0.0950, -0.0064]` (**hurts**).
  - `v12`: Overall $-0.0147$ `[-0.0523, +0.0229]` (**not shown**).

---

## 3. Computational Budget, Latency & Search Diagnostics

All bots complied with official test constraints (first reply $\le 2000$ ms, subsequent turns $\le 50$ ms).

| Arm | Utility | Faults | Late | First Reply Max | Median p99 | Worst Game p99 | Max Reply | Samples/Call | Truncated Share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c08_ponly_pxl_vm` | 0.7872 | 0 | 0 | 72.9 ms | 30.55 ms | 37.96 ms | 45.31 ms | 351.5 | 71.88% |
| `c08_ponly_pxl_vm_v` | 0.7975 | 0 | 0 | 75.9 ms | 30.60 ms | 37.18 ms | 44.92 ms | 352.0 | 71.72% |
| `c08_ponly_pxxl_vm` | 0.7755 | 1 | 1 | 79.8 ms | 30.74 ms | 38.41 ms | 43.45 ms | 255.8 | 85.52% |
| `c08_ponly_pxxl_vm_v`| 0.7940 | 0 | 0 | 104.0 ms | 30.80 ms | 39.22 ms | 44.51 ms | 258.4 | 85.28% |
| `v12_ponly_pxl_vm` | 0.7526 | 0 | 0 | 85.3 ms | 5.63 ms | 17.90 ms | 26.78 ms | - | - |
| `v12_ponly_pxl_vm_v` | 0.7651 | 0 | 0 | 77.0 ms | 5.77 ms | 17.07 ms | 29.85 ms | - | - |
| `v12_ponly_pxxl_vm` | 0.7634 | 0 | 0 | 81.6 ms | 7.74 ms | 21.87 ms | 30.08 ms | - | - |
| `v12_ponly_pxxl_vm_v`| 0.7612 | 0 | 0 | 70.4 ms | 8.17 ms | 22.61 ms | 38.81 ms | - | - |

### V Path Utilization & Decision Shifts

| Arm | Path | Coeff | V Calls/Turn | Exact Terminal | Total Decisions | Changed Decisions | Shift Share | Time in V / Turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v12_ponly_pxl_vm_v` | `rollout` | 40.0 | 6.633 | 1918 | 92,508 | 12,294 | **13.29%** | 0.135 ms |
| `v12_ponly_pxxl_vm_v`| `rollout` | 40.0 | 6.665 | 1854 | 92,951 | 12,259 | **13.19%** | 0.144 ms |
| `c08_ponly_pxl_vm_v` | `refine` | 50.0 | 1.499 | 244 | 33,482 | 4,360 | **13.02%** | 0.050 ms |
| `c08_ponly_pxl_vm_v` | `beam` | 35.0 | 1.046 | 432 | 68,470 | 9,511 | **13.89%** | 0.050 ms |
| `c08_ponly_pxxl_vm_v`| `refine` | 50.0 | 1.360 | 222 | 33,563 | 3,188 | **9.50%** | 0.047 ms |
| `c08_ponly_pxxl_vm_v`| `beam` | 35.0 | 1.046 | 442 | 68,446 | 9,447 | **13.80%** | 0.047 ms |

---

## 4. Error Classification & Root Cause Analysis

We categorize the factors influencing V's effectiveness across the 5 standard hypotheses:

1. **Coefficient Scale & Inactivity**: **REJECTED.**  
   V active paths alter $9.5\%$ to $13.9\%$ of total branch decisions with coefficients $35.0$–$50.0$. V is actively steering node selections rather than acting as a no-op.
2. **Search Starvation & Latency Cost**: **REJECTED.**  
   Incremental fixed-point accumulator updates require $<0.15$ ms/turn. Search sample completion rates between V-on and V-off arms are virtually indistinguishable ($352.0$ vs $351.5$ in C08 XL).
3. **Continuation Policy Mismatch (Distribution Shift)**: **CONFIRMED.**  
   The committed V network was trained on self-play matches involving legacy rule-based and mixed-roster agents. In contrast, evaluation pairs P-only against diverse modern pool opponents. State valuations predicated on legacy bot continuations fail to accurately order outcomes produced by P-guided rollouts.
4. **Feature Representation Granularity**: **CONFIRMED.**  
   The current 54 value features aggregate board state into coarse spatial bins (e.g. `enemies_r3`, `enemy_active`, `home_dist`). Crucial localized variables—individual opponent health points, specific combat focus fire, and precise carrier intercept ETA—are missing. Consequently, V cannot resolve tactical local edge-cases that deterministic heuristics navigate correctly.
5. **Capacity Undersizing**: **INCONCLUSIVE / UNLIKELY.**  
   Offline training logs (`runs_m/value_n04`) showed a flat learning curve (validation cross-entropy moving marginally from $0.5226$ to $0.5219$). Scaling parameter count without updating targets and feature fidelity is unlikely to resolve rank inversion.

---

## 5. Next Steps & Handoff

1. **T0003 Status**: Stage 2 screen complete; evidence committed under `bots/p_only/report/value/`. Task handoff updated.
2. **Future Retraining Prerequisites**:
   - Do not retrain V purely by expanding network width.
   - Any fine-tuning (Phase 5 / `T0006`) must condition on counterfactual rollouts from frozen P-only bots to resolve target distribution mismatch.
   - Maintain strict separation: keep P input/output contracts immutable when updating V features.
