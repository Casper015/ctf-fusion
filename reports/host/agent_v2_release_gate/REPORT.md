# Host Release Gate Screen Report: G vs D (T0083)

**Date:** 2026-10-17  
**Host Machine:** `desktop-g60ojud` (Windows 11, Ryzen 9 7945HX, RTX 4080 Laptop GPU)  
**Task:** T0083 (`Host: release gate, general PPO policy G versus current default D`)  
**Arms:**
- $G$: `agent_v2_rl_general_csp` (general PPO policy, CSP P bank, actor-direct, binary `08d1d5602e4f`, identity `571d1e2ce6051c06`)
- $D$: `agent_v2_p2xxl_direct` (current default, P2 XXL, actor-direct, binary `d5329fc7f69e`, identity `b8414864b71b509a`)
- $G\sim\text{aa}$: `agent_v2_rl_general_csp~aa` (A/A copy of G, identical binary `08d1d5602e4f`)

---

## 1. Executive Summary & Promotion Verdict

- **Screen Budget:** 1,344 paired fixtures across 14 team counts 2..15 = 4,032 formal games under single-tenant CPU lock (8 workers, seed 20262141).
- **Primary Contrast ($G - D$):** **not shown** (+0.00114 / +0.11 pp, 95% map-cluster CI `[-0.01158, +0.01385]`).
- **A/A Consistency:** Identical scores on **1,344 of 1,344 fixtures (100.0%)**, verifying perfect deterministic reproducibility across workers.
- **Reliability:** 0 faults, 0 dead, 0 late across all 4,032 games.
- **Frozen Promotion Rule Decision:**
  > "Promotion rule, frozen before play: G becomes the default only if G-D 'helps' (95% interval above 0); 'not shown' or 'hurts' keeps D."
  
  Because the contrast $G - D$ is **not shown** (the 95% confidence interval straddles zero: `[-1.16%, +1.38%]`), **`agent_v2_p2xxl_direct` ($D$) remains the official production default**. $G$ (`agent_v2_rl_general_csp`) is preserved as a verified competitive RL variant with virtually identical utility (0.7669 vs 0.7658).

---

## 2. Preflight Gate (Seed 20262140)

- **Execution:** 16 maps at each of counts 2, 5, 8, 15 across both variants = 128 games on 8 workers.
- **Results:**
  - `agent_v2_rl_general_csp`: **PASS** (0 faults, 0 dead, 0 late; max reply 15.2 ms, p99 worst game 5.7 ms)
  - `agent_v2_p2xxl_direct`: **PASS** (0 faults, 0 dead, 0 late; max reply 13.4 ms, p99 worst game 6.9 ms)
- **Status:** Passed cleanly; receipt recorded at `reports/host/agent_v2_release_gate/PREFLIGHT.json`.

---

## 3. Formal Paired Screen Results (Seed 20262141)

### Primary Comparison ($G - D$)

| Band | Fixtures | Mean Difference | 95% Map-Cluster CI | Verdict |
|---|---:|---:|---|---|
| **All (2..15)** | **1,344** | **+0.0011** (+0.11 pp) | **[-0.0116, +0.0138]** | **not shown** |
| 2 Teams | 96 | +0.0104 (+1.04 pp) | [-0.0279, +0.0488] | not shown |
| 3–5 Teams | 288 | +0.0130 (+1.30 pp) | [-0.0237, +0.0498] | not shown |
| 6–10 Teams | 480 | -0.0119 (-1.19 pp) | [-0.0330, +0.0092] | not shown |
| 11–15 Teams | 480 | +0.0052 (+0.52 pp) | [-0.0115, +0.0219] | not shown |
| **3–15 Equal Weight** | 1,248 | +0.0004 (+0.04 pp) | — | descriptive |

- **Fixture Outcome Breakdown:**
  - $G > D$ (improved): 457 fixtures
  - $G == D$ (unchanged): 427 fixtures
  - $G < D$ (degraded): 460 fixtures
  - Identical scores: 0 fixtures (both bots actively explore distinct trajectories)

### A/A Consistency Control ($G\sim\text{aa} - G$)

| Band | Fixtures | Mean Difference | 95% CI | Identical Scores |
|---|---:|---:|---|---|
| All (2..15) | 1,344 | +0.0000 | [+0.0000, +0.0000] | **1,344 / 1,344 (100.0%)** |

---

## 4. Operational Arm Statistics

| Arm | Games | Overall Utility | Faults | Dead | Late | First Reply Max | p99 Median | Max Reply | Binary SHA-256 (prefix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `agent_v2_rl_general_csp` ($G$) | 1,344 | 0.76694 | 0 | 0 | 0 | 102.0 ms | 1.35 ms | 15.2 ms | `08d1d5602e4f` |
| `agent_v2_p2xxl_direct` ($D$) | 1,344 | 0.76580 | 0 | 0 | 0 | 94.8 ms | 1.51 ms | 13.4 ms | `d5329fc7f69e` |
| `agent_v2_rl_general_csp~aa` | 1,344 | 0.76694 | 0 | 0 | 0 | 117.2 ms | 1.38 ms | 15.2 ms | `08d1d5602e4f` |

Both variants comfortably operate well below official 2,000 ms startup and 50 ms per-turn limits on 8 concurrent workers.
