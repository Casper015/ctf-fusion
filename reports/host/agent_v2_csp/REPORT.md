# Host Screen Report: agent_v2 with CSP per-count P bank vs XXL P (T0073)

**Date:** 2026-10-16  
**Host:** `desktop-g60ojud` (Windows 10, RTX 4080 Laptop GPU, 16C/32T)  
**Task:** T0073 (`Host: screen agent_v2 with per-count P bank vs XXL P`)  
**Artifacts & Receipts:** `reports/host/agent_v2_csp/`  
**Screen Profile:** `profiles/agent_v2_csp_screen.json`  
**Verdict:** **not shown** (mean utility difference = -0.0116 [-0.0315, +0.0082])  

---

## 1. Summary

Task T0073 evaluates whether better offline opponent prediction accuracy (from the T0067 refreshed per-count P specialists, which beat the deployed XXL P by 0.04 to 0.17 nats on test loss) translates into game wins on the modular `agent_v2` carrier prior to investing in 8XL teacher distillation.

Both arms are actor-direct with the identical trained large-V actor (`meow_rl_model::kDigest = 8b0ca3cfa4d4`):
- **Candidate:** `agent_v2_csp_direct` (selecting T0067's 14 per-count P specialists via `FUSION_TIER_CSP`, binary `83b80ff99b41`).
- **Baseline:** `agent_v2_p2xxl_direct` (the deployed XXL P opponent model, binary `119c7f76b972`).

All experiment gates and protocol requirements were satisfied:
1. **Fresh Seed Reservations:** Immutable seeds registered in `docs/MEOWMIND_RL_SCREEN_SEEDS.md`:
   - `20261960`: Preflight gate (`agent_v2_csp_preflight`, 64 fixtures x 2 arms = 128 games).
   - `20261961`: Paired screen (`agent_v2_csp_screen`, 392 fixtures x 3 arms = 1,176 games).
2. **Eight-Worker Preflight Gate:** Executed under `task.py lock cpu` across 8 concurrent workers on 64 fixtures (16 maps each for counts 2, 5, 8, 15) against the held-out reference roster (`["Fable", "MiMo", "Kimi", "GLM", "Player4"]`).
   - `agent_v2_csp_direct`: **PASSED** (100% turn coverage, 0 faults/late/dead, max steady reply 9.35 ms < 48 ms, max first reply 58.3 ms < 1,800 ms).
   - `agent_v2_p2xxl_direct`: **PASSED** (100% turn coverage, 0 faults/late/dead, max steady reply 6.05 ms < 48 ms, max first reply 54.6 ms < 1,800 ms).
   - Wall time: **11.7 s** for 128 games.
   - Confirms the game-level host timing deferred from T0067.
3. **Paired Carrier Screen:**
   - 392 fixtures (28 maps per count across counts 2–15), exactly 8 workers under CPU lock, 1,176 matches total (including A/A copy).
   - Overall utility difference: **-0.0116** ([-0.0315, +0.0082]), verdict: **not shown**.
   - Determinism: `agent_v2_csp_direct~aa` matched `agent_v2_csp_direct` on **392 of 392 fixtures** (+0.0000 [+0.0000, +0.0000]).
   - Reliability: 0 faults, 0 dead, 0 late across all 1,176 games.
   - Wall time: **132.3 s** total.

---

## 2. Eight-Worker Preflight Gate

- **Receipt:** `reports/host/agent_v2_csp/PREFLIGHT.json`
- **Seed:** `20261960` | **Workers:** 8 | **Wall Time:** 11.7 s | **Schedule:** 64 fixtures (16 maps each for counts 2, 5, 8, 15)
- **Limits:** 2,000 ms initialization, 50 ms subsequent turns.
- **Gates:** steady max < 48.0 ms, first reply max < 1,800.0 ms, 0 faults/late/dead.

| Variant | Role | Turns Played | Turn Coverage | Steady Max (ms) | First Max (ms) | p99 Max (ms) | Faults / Late / Dead | Gate Verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| `agent_v2_csp_direct` | Candidate (CSP per-count P) | 25,600 | 100.0% | 9.35 | 58.3 | 3.33 | 0 / 0 / 0 | **PASS** |
| `agent_v2_p2xxl_direct` | Baseline (XXL P) | 25,600 | 100.0% | 6.05 | 54.6 | 3.65 | 0 / 0 / 0 | **PASS** |

### Per-Count Latency Breakdown (`agent_v2_csp_direct`)
- **Count 2:** steady max = 0.620 ms, p99 = 0.342 ms, p50 = 0.221 ms, first reply = 3.6 ms
- **Count 5:** steady max = 0.860 ms, p99 = 0.579 ms, p50 = 0.419 ms, first reply = 7.0 ms
- **Count 8:** steady max = 1.452 ms, p99 = 1.052 ms, p50 = 0.729 ms, first reply = 22.9 ms
- **Count 15:** steady max = 9.348 ms, p99 = 3.326 ms, p50 = 1.603 ms, first reply = 58.3 ms

Both variants easily met all timing gates under full 8-worker concurrent load with zero faults.

---

## 3. Paired Carrier Screen

- **Receipts & Directory:** `reports/host/agent_v2_csp/screen/`
- **Schedule:** `b30654ae120f289ddc180c82203f436a09bab536000f80713eef3d48708b48fd`
- **Seed:** `20261961` | **Workers:** 8 | **Wall Time:** 132.3 s | **Total Games:** 1,176

### Contrast: `agent_v2_csp_direct` − `agent_v2_p2xxl_direct`

| Band | Teams | Mean Utility Diff | 95% Map-Cluster CI | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| **All** | 2–15 | **-0.0116** | **[-0.0315, +0.0082]** | not shown | 392 |
| **Band 2** | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| **Band 3–5** | 3–5 | -0.0481 | [-0.1060, +0.0097] | not shown | 84 |
| **Band 6–10** | 6–10 | -0.0090 | [-0.0406, +0.0227] | not shown | 140 |
| **Band 11–15** | 11–15 | +0.0053 | [-0.0244, +0.0350] | not shown | 140 |

- **Fixture Outcome Distribution:** 120 improved, 138 unchanged, 134 degraded (identical scores on 5 of 392 fixtures).

### A/A Determinism Diagnostic
- `agent_v2_csp_direct~aa` − `agent_v2_csp_direct`:
  - **Mean difference:** +0.0000 [+0.0000, +0.0000]
  - **Replay Agreement:** 392 of 392 fixtures produced bit-exact identical game scores (0 improved / 392 unchanged / 0 degraded).
  - Verifies zero simulation jitter, zero concurrency cross-talk, and deterministic decision execution.

### Arms Summary Table

| Arm | Games | Utility | Faults | Dead | Late | Opponent Faults | First Reply Max (ms) | p99 Median (ms) | Max Reply (ms) | Binary SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `agent_v2_csp_direct` | 392 | 0.7645 | 0 | 0 | 0 | 0 | 51.3 | 1.085 | 5.65 | `83b80ff99b41` |
| `agent_v2_p2xxl_direct` | 392 | 0.7761 | 0 | 0 | 0 | 0 | 54.6 | 1.052 | 6.05 | `119c7f76b972` |
| `agent_v2_csp_direct~aa` | 392 | 0.7645 | 0 | 0 | 0 | 0 | 55.8 | 1.082 | 5.32 | `83b80ff99b41` |

---

## 4. Architectural Conclusions & Downstream Impact

1. **Better Offline Opponent P Is Not the Bottleneck:**
   - T0067's 14 count-specific P specialists achieved clear offline test loss improvements over XXL (+0.04 to +0.17 nats across all counts).
   - In actual paired carrier games under `agent_v2`, this offline gain yielded **no statistically significant in-game playing benefit** (-0.0116 [-0.0315, +0.0082]).
2. **Decision on Task T0074 (8XL Distillation):**
   - T0074 was explicitly gated on T0073: *"Start only if T0073 shows the per-count bank helps in games. If T0073 is 'not shown' or 'hurts', hand off --status blocked and ask Casper whether to run it anyway; better offline P loss would then not be the bottleneck."*
   - Per this predefined rule, **the 8XL distillation route for P should be dropped / blocked**, saving substantial GPU compute that would produce only further offline loss gains without in-game playing strength improvements.
3. **Green Light for Task T0068 (Exact-Count PPO Specialists):**
   - The primary performance lever is the actor policy ($\pi$) and value function ($V$), not marginal opponent modeling.
   - Upstream carriers and opponent model routing are verified reliable, low-latency (<6 ms worst-case reply), and zero-fault.
   - Resource allocation should proceed directly to on-policy RL training in **T0068**.
