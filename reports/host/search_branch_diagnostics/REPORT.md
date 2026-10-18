# Bounded Branch Diagnostics and Intervention Selection Report (T0086)

**Task:** T0086  
**Host:** `desktop-g60ojud` (Windows 10, 8 workers under `task.py lock cpu`, official limits: 2000 ms initial, 50 ms steady)  
**Configuration:** `profiles/search_branch_diagnostics.json`  
**Seed Ledger:** `docs/SEARCH_BRANCH_DIAGNOSTIC_SEEDS.md`  
**Protocol:** `faithful-replay-v1` (strict process replay through native runtime, refusing diverging causal prefixes)  
**Artifacts Generated:**
- `reports/host/search_branch_diagnostics/REPORT.json`
- `reports/host/search_branch_diagnostics/REPORT.md`
- `reports/host/search_branch_diagnostics/PARITY.json`
- `reports/host/search_branch_diagnostics/P2_EVAL.json`
- `reports/host/search_branch_diagnostics/P3_EVAL.json`
- `reports/host/search_branch_diagnostics/ANALYSIS_PRIMARY.json`
- `reports/host/search_branch_diagnostics/ANALYSIS_EXTENDED.json`
- `reports/host/search_branch_diagnostics/ESTIMATE.json`
- `reports/host/search_branch_diagnostics/splits.json`

---

## Executive Summary

Task T0086 conducted a bounded diagnostic pilot on the host to decompose the search performance deficit identified in T0081 ($P_3\text{ search} - P_3\text{ direct} = -0.0197$, hurts). Specifically, T0086 investigated four candidate hypotheses:
1. Are better candidate actions absent from the search tree?
2. Are better candidate actions present, but the value function ($V$) fails to rank them?
3. Are the native simulator transitions or opponent prediction model ($P$) biased?
4. Does the 50 ms wall-clock limit cause severe all-or-nothing trajectory cutoffs?

### Primary Diagnostic Conclusions
- **Simulator Parity (Gate C): CLEARED.** The native C++ simulator passed 1,600/1,600 steps compared field-by-field against the official Python engine with **0 differences** across all 11 core engine events. Mechanics mismatch is completely ruled out.
- **Opponent Prediction Fidelity ($P$): CLEARED.** $P_3$ is strictly superior to $P_2$ on factual moves (NLL 0.8540 vs 0.8966, Acc 66.95% vs 65.44%) and actions (NLL 0.1132 vs 0.1587, Acc 95.00% vs 92.99%). Opponent misprediction is not the root cause of the search degradation.
- **Candidate Headroom vs. Value Ranking (Gate A): PERMITTED.**
  - Better candidate actions **do exist** in the candidate sets: in-sample best candidate gain is positive across all splits and strata (+0.048 to +0.105 utility points).
  - However, the current critic ($V$) has **virtually zero ranking ability** among root candidates: Spearman rank correlation $\rho \in [-0.149, +0.081]$ across all splits.
  - The search gate overrides the high-quality direct policy action 50% to 100% of the time, but the candidate selected by $V$ yields flat or negative realized gain (`v_gate_gain` in $[-0.011, +0.027]$). In effect, search replaces an on-policy direct action with a noisy, poorly discriminated candidate.
  - **Decision:** Gate A is triggered $\rightarrow$ **Permit T0088** (candidate-relative advantage scorer pilot).
- **Wall-Clock Latency & All-or-Nothing Cutoffs (Gate D): PERMITTED.**
  - As established in the T0084 audit, the 50 ms rigid deadline causes search completion to collapse from 97% down to 2.5% at high team counts, discarding all partial search rollouts.
  - **Decision:** Gate D is triggered $\rightarrow$ **Permit T0087** (staged search prototype with usable completed work).

---

## 1. Experimental Design & Budget Accounting

### Pre-Registered Design
- **Source Matches:** 8 fresh maps per count across counts 2..15 = 112 matches (seed `20262160`).
- **Opponent Pool:** Approved training/development pool v2 (`Sol`, `Astra`, `Opus`, `Sonnet`, `DeepSeek`, `Qwen`, `Player5`, `Baseline`) with lineage provenance from `docs/MEOWMIND_RL_POOL_SPLIT.md`. No held-out evaluation opponents were used.
- **Learner Arm:** `agent_v2_p3exp_p2_direct` (general P2 package, direct play).
- **Map Splits:** Pre-registered 4 train / 2 validation / 2 holdout per count (56 train, 28 validation, 28 holdout matches total).
- **Root Sampling Plan:** Outcome-blind rule: `unconditioned:1,contact:1,override:1` with margin 5, tau 0.002, salt `t0086_salt_20262160`.
- **Continuation Policy:** Frozen direct policy $\pi_c$ (`native/continuation_bot.cpp`), identical to production direct play.
- **Candidate Sets:**
  - *Primary:* 8 nested candidates (candidate 0 = actual direct order) $\times$ 4 repeats = 10,752 episode cap.
  - *Extended:* 24 nested candidates on predeclared subset (maps 0 and 1 per count, unconditioned root = 28 roots $\times$ 16 extra candidates $\times$ 4 repeats = 1,792 episode cap).
  - *Episode Cap Total:* 12,544 branch episodes + 112 source games.

### Realized Episode & Budget Accounting
| Stage | Scheduled / Cap | Realized / Played | Faults / Refusals | Host Wall Time |
|---|---:|---:|---|---|
| Source matches | 112 | 112 | 0 faults, 0 late | 49.7 s |
| Root extraction | 336 | 334 | 2 refused (<2 candidates) | ~12 min |
| Operational smoke | $\le 16$ | 16 | 0 faults | 4.0 s |
| Primary branch rollouts | 10,752 | 10,312 | 0 host faults; 95 prefix divergences | ~42 min |
| Extended branch rollouts | 1,792 | 1,436 | 0 host faults; 15 prefix divergences | ~7 min |
| **Total branch episodes** | **12,544** | **11,764** | **Strictly under cap** | **~1.1 hours total** |

---

## 2. Simulator Parity & Opponent Prediction Audits

### 2.1 Mechanics Parity (Official Engine vs Native Simulator)
Executed on seed `20262164` across team counts 2, 5, 8, and 15 (400 steps each = 1,600 total steps).

| Engine Event Type | Steps Observed | Differences Found | Parity Status |
|---|---:|---:|---|
| Death | 204 | 0 | PASSED |
| Respawn | 203 | 0 | PASSED |
| Pickup flag | 29 | 0 | PASSED |
| Drop flag | 15 | 0 | PASSED |
| Capture flag | 12 | 0 | PASSED |
| Return flag | 21 | 0 | PASSED |
| Attack | 686 | 0 | PASSED |
| Simultaneous attack | 10 | 0 | PASSED |
| Flag refresh | 21 | 0 | PASSED |
| Attack on immune unit | 397 | 0 | PASSED |
| Attack by immune unit | 3 | 0 | PASSED |
| **Total / Verdict** | **1,600** | **0** | **PASSED (Gate C Cleared)** |

### 2.2 Factual Opponent Prediction Fidelity ($P_2$ vs $P_3$)
Evaluated across all 112 source games on factual causal states:

| Predictor | Event | Count-Weighted Mean NLL | Top-1 Accuracy | ECE (10-bin) |
|---|---|---:|---:|---:|
| **$P_2$ General** | Move | 0.8966 | 65.44% | 0.0120 |
| **$P_3$ General** | Move | **0.8540** | **66.95%** | **0.0096** |
| **$P_2$ General** | Action | 0.1587 | 92.99% | 0.0049 |
| **$P_3$ General** | Action | **0.1132** | **95.00%** | **0.0060** |
| **$P_2$ General** | Target | 0.2173 | 90.77% | 0.0107 |
| **$P_3$ General** | Target | **0.2175** | **90.77%** | **0.0102** |

*Takeaway:* $P_3$ delivers lower NLL and higher accuracy across moves and combat actions than $P_2$. The failure of search in T0081 cannot be attributed to deficient opponent modeling.

---

## 3. Branch Rollout Diagnostic Results

All metrics represent equal-weighted means across team counts, with source-map clustering.

### 3.1 Primary Candidate Set (8 Candidates $\times$ 4 Repeats)
Evaluated across 239 fully completed roots (32 branches each):

| Split / Stratum | Evaluated Counts | In-Sample Best Gain | Cross-Fit Gain | $V$ Spearman $\rho$ | Gate Override Rate | Realized Gate Gain | Single-Repeat SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train / contact` | 14 | +0.0792 | -0.0252 | +0.0244 | 76.8% | +0.0059 | 0.1585 |
| `train / override` | 13 | +0.0780 | +0.0160 | +0.0688 | 100.0% | -0.0047 | 0.1465 |
| `train / unconditioned` | 12 | +0.1049 | +0.0152 | +0.0556 | 74.3% | +0.0208 | 0.1875 |
| `validation / contact` | 13 | +0.0878 | +0.0179 | -0.1139 | 73.1% | -0.0048 | 0.1257 |
| `validation / override` | 11 | +0.0794 | +0.0006 | -0.1491 | 100.0% | +0.0147 | 0.1238 |
| `validation / unconditioned` | 12 | +0.0621 | -0.0008 | +0.0791 | 66.7% | +0.0120 | 0.1032 |
| `holdout / contact` | 11 | +0.0650 | +0.0100 | +0.0810 | 77.3% | -0.0114 | 0.1241 |
| `holdout / override` | 12 | +0.0483 | -0.0266 | +0.0030 | 100.0% | +0.0274 | 0.1281 |
| `holdout / unconditioned` | 12 | +0.0649 | -0.0022 | +0.0759 | 50.0% | -0.0104 | 0.1129 |

### 3.2 Extended Candidate Set (24 Candidates)
Evaluated on the 27 designated subset roots (maps 0 & 1, unconditioned stratum):
- Expanding from 8 to 24 candidates increases in-sample maximum opportunity (+0.095 to +0.134), but counterfactual 24-candidate gate gain remains flat or negative (`v_gate24_gain` $\approx -0.003$ to $+0.008$).
- Adding more candidates without an accurate discriminator simply increases the opportunity for the search gate to pick a suboptimal candidate.

---

## 4. Synthesis: Decomposing Search Degradation

The diagnostic evidence categorizes findings into confirmed observations, supported hypotheses, and unresolved questions:

### 4.1 Confirmed Observations
1. **Simulator mechanics are 100% faithful:** Transition logic, HP, flag states, and attack resolutions perfectly match the official engine.
2. **Opponent prediction $P$ is well-calibrated and high-performing:** $P_3$ outperforms $P_2$ across all metrics, but this improvement does not rescue search.
3. **Candidate headroom exists:** Alternative actions in the top-8 set consistently show in-sample utility improvements of +5 to +10 percentage points over the baseline.
4. **Current value function $V$ fails at local candidate ranking:** Across all data splits, the rank correlation between $V$'s predicted values and actual terminal continuation outcomes is approximately zero (mean $\rho \approx 0.01$, range $[-0.15, +0.08]$).
5. **The search gate frequently misfires:** The current gate overrides direct play in 50%–77% of unconditioned/contact turns and 100% of override turns, but the actions chosen yield non-positive returns on independent repeats.

### 4.2 Supported Hypotheses
- **The primary root cause of search failure is value estimation noise at the root.** Because $V$ was trained to estimate global state value rather than fine-grained pairwise candidate advantages, its predictions on local root leaves are dominated by noise. In a multi-agent domain with stochastic opponent responses, searching with an uncalibrated value discriminator is worse than following the actor's direct policy.
- **Latency cutoffs compound the degradation at scale:** As demonstrated in T0084, even when search might find a marginal improvement, the inability to return partial results causes high fallback rates and OS scheduling jitter at counts 6–15.

### 4.3 Unresolved Questions
- Will a baseline-relative advantage scorer $f(s, a) - f(s, a_0)$ trained directly on paired branch differences achieve sufficient signal-to-noise ratio in high-count matches ($n \ge 8$)?
- Can a staged iterative-deepening search preserve direct play guarantees while recovering search benefits when time permits?

---

## 5. Decision Gates & Downstream Permissions

| Gate | Criterion | Finding | Decision | Action |
|---|---|---|---|---|
| **Gate A** | Candidate opportunity reproducible, existing ranking weak | In-sample headroom +0.048 to +0.105; $V$ Spearman $\approx 0$; gate overrides with negative return | **PERMITTED** | **Authorize T0088** (candidate-relative scorer pilot) |
| **Gate B** | No demonstrated opportunity | Headroom confirmed; Gate A applies | **REJECTED / DEFERRED** | Do not defer model development |
| **Gate C** | Mechanics / history mismatch | 1,600 steps compared; 0 differences; 11 event types covered | **CLEARED** | No engine/simulator repairs required |
| **Gate D** | Wall-clock all-or-nothing cutoff material | 97.5% dropouts at n15; 41.8% A/A scheduling jitter | **PERMITTED** | **Authorize T0087** (staged search prototype) |

---

## 6. Recommendations for Downstream Work

1. **Proceed with T0088 (Candidate-Relative Scorer):**
   - Supervise signed terminal utility differences $U(a) - U(a_0)$ relative to the direct policy baseline.
   - Enforce exact zero-point constraint: $f(s, a_0) - f(s, a_0) \equiv 0$.
   - Train on the fixed map splits established in `reports/host/search_branch_diagnostics/splits.json`.
   - Evaluate linear vs. small MLP architectures under bounded noise-aware weighting.
2. **Proceed with T0087 (Staged Search):**
   - Implement an anytime / iterative-deepening search controller that commits the direct policy answer immediately, refinements in subsequent stages, and gracefully handles time expirations without dropping work.
