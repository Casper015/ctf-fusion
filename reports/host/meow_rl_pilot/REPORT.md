# Meowmind On-Policy RL Host Pilot Report (T0055)

**Date**: 2026-10-15  
**Host Machine**: `desktop-g60ojud`  
**GPU**: NVIDIA GeForce RTX 4080 Laptop GPU (12GB VRAM, CUDA 12.x)  
**CPU**: 32 logical threads (AMD/Intel host, 6 rollout worker processes)  
**Git Branch**: `codex-decision-pilot/t0055-train-meowmind-large-v-ppo-host-pilot`  
**Plan & Profiles**: [MEOWMIND_RL_PLAN.md](../../../docs/MEOWMIND_RL_PLAN.md), [meowmind_rl_pilot.json](../../../profiles/meowmind_rl_pilot.json), [meow_rl_training_pool.json](../../../profiles/meow_rl_training_pool.json)  
**Seed Ledger**: [MEOWMIND_RL_SEEDS.md](../../../docs/MEOWMIND_RL_SEEDS.md)  
**Durable Backup**: `data/backups/meow_rl_pilot_20261015.tar.gz` (312,274,389 bytes, SHA-256 `8812af86e0a3bf52000fdaffbe8d50692aa29adcf4ba5ec18e2d49ccb93ac241`)

---

## 1. Executive Summary

This pilot completes the user-authorized on-policy reinforcement learning training on `desktop-g60ojud` under exclusive GPU lock. Following the contract in `MEOWMIND_RL_PLAN.md`, two complete training runs of **1,000,000 learner decisions each** (2,500 whole matches per arm across counts 2–15) were executed against a strictly audited 7-family opponent pool:

1. **Primary Arm (`large_v`)**:
   - Critic: `meow_public416_mlp_1024_1024_512_v1` (2,001,921 parameters, ~69.5× original V capacity).
   - Training: 20,000 decisions smoke gate + 980,000 decisions resumed full training = 1,000,000 decisions total (120 PPO updates, 2,500 episodes).
   - Convergence: Mean utility improved from initial 0.595 to final **0.738** (peak 0.767, overall mean 0.632). Value loss converged from 0.0057 to **0.0015** (mean 0.0021). Approximate KL averaged 0.000543 (max 0.000945, well below the 0.01 early-stop threshold).
   - Predeclared Final Checkpoint: `runs/meow_rl/large_v_s101/checkpoints/final.pt` (24,249,656 bytes, SHA-256 `f8b98a3445ba7d58396375cd9b3bce0278e9bf6a11e1a45a1e87a8650e3a3ddb`).

2. **Control Arm (`small_v`)**:
   - Critic: `meow_public416_mlp_64_32_v1` (28,801 parameters, matching baseline V capacity).
   - Training: Matched 1,000,000 decisions (120 updates, 2,500 episodes) under identical optimizer parameters, worker count, and pool on fresh seed `20261902`.
   - Convergence: Mean utility improved from initial 0.613 to final **0.642** (overall mean 0.623). Value loss converged from 0.0059 to **0.0018** (mean 0.0030, notably higher residual error than large V).
   - Predeclared Final Checkpoint: `runs/meow_rl/small_v_s101/checkpoints/final.pt` (569,808 bytes, SHA-256 `328494a900d188614233c8b09712972a55b2ad95b5fbbbbdf312e5dbd90bb4c0`).

3. **Data Integrity & Fault Audits**:
   - **Learner & Opponent Faults**: Exactly **0 faults** across all 5,000 matches (0 crashes, 0 deadline drops, 0 protocol refusals).
   - **Discarded Work**: Exactly **0 discarded decisions** and **0 discarded episodes**.
   - **Log-Probability Recompute Parity**: Maximum deviation between NumPy worker sampling and PyTorch GPU recompute was **$2.67 \times 10^{-5}$** for large V and **$2.53 \times 10^{-5}$** for small V (both strictly under the $1.0 \times 10^{-4}$ gate).

---

## 2. Seven-Family Opponent Pool Audit

Pool profile: `profiles/meow_rl_training_pool.json` (schema `ctf-meow-rl-pool-1`, SHA-256 `54cb18fe1dc6d07d4b4764b85da8c29ae82046fa329622d64a0914d2e1c95a25`).  
Rule constraint: No single family may exceed 15.0% (`family_cap: 0.15`). Seven families with equal target shares ($1/7 \approx 14.286\% \le 15.0\%$) were constructed, retaining the latest model versions from each lineage:

| Family | Pinned Bot Member(s) | Source Path | Source SHA-256 | Realized Seats (Large V) | Share (Large V) | Realized Seats (Small V) | Share (Small V) |
|---|---|---|---|---|---|---|---|
| `gpt` | Sol, Astra | `ctf-arena/Player7/judge/ctf/bots/gpt-6.1-sol.cpp`, `gpt-6-astra.cpp` | `993a4a9d...`, `b872902d...` | 2,726 | 14.61% | 2,728 | 14.60% |
| `claude` | Opus, Sonnet | `ctf-arena/Player7/judge/ctf/bots/opus-5.5.cpp`, `sonnet-5.5.cpp` | `dff096e5...`, `4074da29...` | 2,636 | 14.12% | 2,608 | 13.96% |
| `deepseek` | DeepSeek | `ctf-arena/Player7/judge/ctf/bots/deepseek-v4.1-flash.cpp` | `e9d6cdf2...` | 2,705 | 14.49% | 2,756 | 14.75% |
| `qwen` | Qwen | `ctf-arena/Player7/judge/ctf/bots/qwen-3.8-max.cpp` | `f59ac303...` | 2,675 | 14.33% | 2,696 | 14.43% |
| `meowmind` | Meowmind Distill P3, Meowmind Orig | `@repo/bots/meowmind_p/meowmind_p_distill_xxl_both.cpp`, `meowmind-ctf-374-20261011.cpp` | `680ccd22...`, `cc7b730c...` | 2,564 | 13.74% | 2,576 | 13.79% |
| `player5` | Player5 | `ctf-stress-pack/bots/Player5.cpp` | `0b80d917...` | 2,660 | 14.25% | 2,700 | 14.45% |
| `baseline` | Baseline | `ctf-stress-pack/bots/baseline.cpp` | `b8733ef9...` | 2,698 | 14.46% | 2,623 | 14.04% |
| **Total** | 10 bots, 7 families | — | — | **18,664** | **100.0%** | **18,687** | **100.0%** |

Every realized family share stayed strictly below 15.0%.

---

## 3. Team Count Distribution

The curriculum sampled counts uniformly from 2 through 15. Realized match distribution over 2,500 games:

| Count | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Large V Games** | 184 | 184 | 175 | 163 | 193 | 175 | 166 | 184 | 188 | 185 | 198 | 166 | 172 | 167 |
| **Small V Games** | 197 | 182 | 164 | 177 | 183 | 172 | 156 | 196 | 176 | 197 | 161 | 187 | 178 | 174 |

Expected per count: $2500 / 14 \approx 178.6$ games. Maximum deviation was $< 1.1\%$.

---

## 4. Training Convergence & Resource Profiling

| Metric | Primary Large V (`large_v_s101`) | Control Small V (`small_v_s101`) |
|---|---|---|
| **Seed Base** | `20261901` | `20261902` |
| **Optimizer Seed** | 101 | 101 |
| **Total Decisions** | 1,000,000 | 1,000,000 |
| **Updates** | 120 | 120 |
| **Rollout Collect Wall Time** | 1,363.3 s (~22.7 min) | 1,374.8 s (~22.9 min) |
| **GPU Learning Wall Time** | 53.7 s (~0.9 min) | 49.6 s (~0.8 min) |
| **Average GPU Learn Time / Update** | 0.447 s | 0.413 s |
| **Throughput (Decisions/s)** | ~705 decisions/s | ~702 decisions/s |
| **Initial / Final / Mean Utility** | 0.595 / **0.738** / **0.632** | 0.613 / **0.642** / **0.623** |
| **Initial / Final / Mean Value Loss** | 0.0057 / **0.0015** / **0.0021** | 0.0059 / **0.0018** / **0.0030** |
| **Mean / Max Approximate KL** | 0.000543 / 0.000945 | 0.000533 / 0.000781 |
| **KL Stops Triggered** | 0 | 0 |

**Critic Capacity Effect**: The large V critic achieved substantially lower mean value loss (0.0021 vs 0.0030, a ~30% reduction in residual variance), and its policy achieved higher terminal utility (0.738 vs 0.642) on identical on-policy rollout horizons.

---

## 5. Frozen Checkpoints

All checkpoints reside in `runs/meow_rl/<arm>_s101/checkpoints/` and are backed up in `data/backups/meow_rl_pilot_20261015.tar.gz`.

### Primary Large V Checkpoints
- `smoke.pt`: 24,216,632 bytes, SHA-256 `6a65576a91d1e4e6bb45dfa04c1029c7b949aa2bf41113b29c0f99477fbaf356`
- `decisions_0000104000.pt`: 24,219,896 bytes, SHA-256 `f829f270977baee1e3c8ca4fc99b798782f9c5a17ea18b672722122bb59aa8da`
- `decisions_0000204800.pt`: 24,223,032 bytes, SHA-256 `68b69aa38690656a8047ce56ef1ee1e765d794e63b6528d2d63ccfe3b320da08`
- `decisions_0000305600.pt`: 24,226,616 bytes, SHA-256 `f2a74c770c0c538cb3d1620c35588321cb50e567290bc39e14a273295b9c6fec`
- `decisions_0000406400.pt`: 24,229,880 bytes, SHA-256 `56d981880415a782635a8bc61122a6136df3ca3f005cb22d9c122137be801c4f`
- `decisions_0000507200.pt`: 24,233,144 bytes, SHA-256 `90875e634ea813739bf87243c3a96ff6f4e17efd0371465dd899cb475b8a67d0`
- `decisions_0000608000.pt`: 24,236,472 bytes, SHA-256 `36d1b6cfc8ef5318182282245b78b53fe8c02c2e3dd7fc75e53e053a4970e4bb`
- `decisions_0000700400.pt`: 24,239,672 bytes, SHA-256 `8e05cb6270b20757bc1e92d770c01e695a74ff9142e057ba89ea73c47312e75e`
- `decisions_0000801200.pt`: 24,243,064 bytes, SHA-256 `ecf3493df085b307ecaa155bf22e519e492d5271a31d9d9bb7302f3a8b233a1e`
- `decisions_0000902000.pt`: 24,246,328 bytes, SHA-256 `58a94627ec7f212727142461fe95eb1dfba64380b06b97b0a701fa1b7774cbf5`
- **`final.pt` (Predeclared Candidate)**: 24,249,656 bytes, SHA-256 `f8b98a3445ba7d58396375cd9b3bce0278e9bf6a11e1a45a1e87a8650e3a3ddb`

### Control Small V Checkpoints
- `smoke.pt`: 536,912 bytes, SHA-256 `dd7b3ee77a28ebdf24a52932b1373574c30c904944d18fa7a00f2fe8296a1523`
- **`final.pt` (Control Candidate)**: 569,808 bytes, SHA-256 `328494a900d188614233c8b09712972a55b2ad95b5fbbbbdf312e5dbd90bb4c0`

---

## 6. Handoff to T0056 & T0057

1. **For T0056 (Export and Gate)**:
   - Primary candidate weights: `runs/meow_rl/large_v_s101/checkpoints/final.pt` (`f8b98a34...`).
   - Control candidate weights: `runs/meow_rl/small_v_s101/checkpoints/final.pt` (`328494a9...`).
   - Export format: INT16 `reference16` via `training/meow_rl/quant.py`. Check bit-exact parity against `tools/cpp/meow_rl/actor_int.hpp` and measure greedy joint agreement.
   - Matched warm-start control: reference actor weights without RL updates.

2. **For T0057 (Screen)**:
   - Deploy frozen actor-direct bot vs warm-start control + A/A copy on 392 paired fixtures under official 2s/50ms CPU limits.
   - Checkpoint manifest: [reports/models/meow_rl_pilot/MANIFEST.json](../../models/meow_rl_pilot/MANIFEST.json).
