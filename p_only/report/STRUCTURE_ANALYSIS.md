# Structure Analysis: Carriers for P-Only and P+V Integration

**Date**: 2026-10-10  
**Status**: Phase 0 Complete  
**Scope**: Offline architectural triage across all 258 eligible bots, deep-dive into the 8–12 strongest candidates from `p_only/report/STRUCTURE_SCAN.md`, synthesis of earlier experimental evidence, and ranked shortlist of integration carriers.

---

## 1. Executive Summary & Ranked Shortlist

The objective of P-only integration is to replace hand-written opponent forecasts, heuristic habit tracking, and hard-coded rival lookahead with a pure trained Policy network $P$, without blending or fallback floors, while preserving our own macro planning, task assignment, and evaluation.

### Ranked Shortlist of Carriers

| Rank | Carrier Candidate | Light Screen Utility [95% low] | Turn p99 Headroom | Opponent Sampling Depth | Where P Adds Value | Where V Adds Value | Integration & Maintenance Cost | Primary Risks |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: | :--- |
| **#1** | **`Player5_v12_rollout_volley` (`v12`)** | **0.770** [0.723] | **>40 ms** (max 8.6 ms) | 2-ply multi-model rollout | Rivals' action/fire distribution in close combat | Terminal evaluation of simulated 2-ply states | **Low** (649 LOC, clean simStep interface) | Over-pruning before rollout; reliance on single-destination assumptions |
| **#2** | **`C08_Player5_Omega_LIT` (`C08`)** | **0.711** [0.651] | **15–20 ms** (max 32.4 ms) | 24-sample expectimax, 2-ply playout | Joint movement sampling in multi-team close combat | Leaf re-ranking in close-range beam search | **Medium** (2,212 LOC, complex state caching) | Wall-clock search truncation; non-deterministic A/A repeats |
| **#3** | **`R3_HABITS_adaptive` / `C20`** | **0.756** [0.691] | **>30 ms** (max 15.0 ms) | 2-ply `stepExact` | Direct neural replacement of hand-crafted habit mixture | State evaluation after combat settlement | **Medium** (1,723 LOC) | Code overlap with C08 foundation without C08's search genes |
| **#4** | **`Control_external_CL`** | **0.743** [0.676] | **>30 ms** (max 14.5 ms) | 2-ply rollout | Rival tactical intentions (`rivalWants`) | State scoring at rollout horizon | **Low-Medium** (778 LOC) | Very close architectural sibling to v12; marginal distinctiveness is low |

**Conclusion**: `v12` and `C08` represent two fundamentally different, complementary carrier paradigms:
- `v12` is a lightweight, high-speed rollout engine with massive timing headroom (<10 ms), where P directly supplies realistic tactical opponent responses.
- `C08` is a heavy, search-centric expectimax planner where P's joint multi-unit distribution directly powers Monte Carlo scenario sampling.
Both carriers are already fully ported and validated in `p_only/bots/`. No third carrier needs to be built at this stage without new empirical justification.

---

## 2. Detailed Code & Architectural Review of Strongest Candidates

### 1. `Player5_v12_rollout_volley` (`v12`)
- **Opponent Model**: Hand-written `forecast` and `rivalWants` in tactical proximity.
- **Lookahead & Simulation**: 2-ply forward simulation using an internal simulator (`simStep`, `refine`) across 2 tactical models.
- **Sampling**: Deterministic 2-ply tactical branching; does not run Monte Carlo sampling, but evaluates tactical volleys under anticipated enemy response.
- **Leaf Evaluation**: Terminal combat utility based on survivor HP, carrier status, and material delta.
- **Time Budget & Headroom**: Blazing fast. Light screen max was 8.61 ms, mean ~1–2 ms. Leaves >40 ms headroom for network inference per turn.
- **P Replacement Scope**: Replaces hand-coded `rivalWants` and enemy fire assignment rules completely.
- **Band Utility**: 2T: 1.00, 3–5T: 0.81, 6–10T: 0.71, 11–15T: 0.77.

### 2. `C08_Player5_Omega_LIT` (`C08`)
- **Opponent Model**: `rivalsFirstStep`, `rivalsNextStep`, `rivalsNear`, `forecastDrops`, and habit tracking.
- **Lookahead & Simulation**: 24-sample expectimax in close-combat gate (`refineGate`), simulating 2 plies with `stepExact`.
- **Sampling**: True joint opponent sampling across teams. P's joint probability distribution $P(\text{joint move})$ directly replaces the uniform/independent assumptions.
- **Leaf Evaluation**: Playout terminal scoring combining material, positioning, and flag distance.
- **Time Budget & Headroom**: Max reply time is 32.36 ms (p99 ~22–26 ms). Fits the 50 ms budget, but expectimax sample count is throttled by a wall-clock timer (often scaling back from 512 samples to 24 samples).
- **P Replacement Scope**: Replaces hand-written 4-turn scripted paths, heuristic drop predictions, and attention weights.
- **Band Utility**: 2T: 1.00, 3–5T: 0.84, 6–10T: 0.62, 11–15T: 0.67.

### 3. `R3_HABITS_adaptive` / `C20_R3_HABITS_adaptive`
- **Opponent Model**: Dynamically updates a per-opponent empirical frequency histogram of moves (`HABITS`).
- **Lookahead**: 2-ply `stepExact` with `refineChecked`.
- **Sampling**: Single-branch expectimax without Monte Carlo dispersion.
- **Time Budget**: Max 10.8–15.0 ms. Headroom >35 ms.
- **Why Not Shortlisted Over C08**: Its core architecture is the exact precursor to C08 (Round 3 foundation). C08 already includes its foundation plus sequential halving (`SH`), asymmetric planning (`ASYM`), and risk management (`RISK`).

### 4. `cand-mc1` and `cand-mc2`
- **Opponent Model**: Proximity-based prediction with `enemyMoves`.
- **Lookahead**: Monte Carlo simulation (`simulate`, `refine`, `settle`).
- **Time Budget**: Max 18.3–18.7 ms. Headroom >30 ms.
- **Verdict**: Very similar rollout logic to `v12`, but with 1,333 LOC (double the size of v12) and slightly lower utility (0.745 vs 0.770).

---

## 3. Inventory of Earlier Experimental Evidence & Limitations

| Test Campaign | Commit | Seed(s) | Team Counts | Maps / Games | Contenders | What it Supports | What it CANNOT Support / Limitations |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **Pilot Pre-test** | `9ee676b` / `e11c45b` | 20261020, 20261009 | 2–15 | 400 maps / 3,000 seat fixtures | Opponent Pool Mix | Verified pool sampling fidelity (69.1% Field, 30.6% Style, 2% Noise); verified zero timeouts and voluntary flag drop capture. | Did not test candidate playing strength or carrier models. |
| **P Integration Pilot** | `e11c45b` | 20261020 | 2, 5 | Smoke games | C08, v12 (P-S, P-M, P-L) | Verified basic C++ carrier compilation and runtime viability under engine. | No paired statistical comparisons; small sample size. |
| **V Ablation Screen** | `0a0badc` | 20261020 | 2, 5, 8, 15 | 20 maps/count (480 games total) | `control`, `pm_no_v`, `pm_with_v` on C08 and v12 | Isolated marginal $\Delta V$ over pure P-M: $+0.0366$ on v12 ($29\text{W}/30\text{T}/21\text{L}$), $+0.0607$ at 15T on C08. | Evaluated Tier M only; tested legacy-blended P rather than pure P-only; maps overlapped earlier seeds. |
| **Standard 1,000-Game BT Tournament** | `99ecbe1` | 20261020 | 2, 3, 4, 5, 6, 8, 10, 15 | 125 games/count (1,000 games) | 16 bots: C08_pvL, v12_pvL, controls, 12 reference bots | Ranked C08_pvL #1 (+36.3 BT) and v12_pvL #2 (+13.3 BT). | 95% bootstrap intervals overlapped zero ($\Delta \text{BT}$ CI for C08 was $[-12.7, +69.8]$, for v12 was $[-18.4, +59.0]$); could not statistically separate gaps from 0; C08 control reproduced score in only 36/80 reruns. |
| **Light Screen** | `HEAD` (`roster/report`) | Legacy | 2, 5, 8, 15 | 70 paired games / bot | 64 eligible candidate bots | Broad ranking triage across all candidate lineages. | Only 70 games per bot; high standard errors; observational only. |

---

## 4. Synthesis & Recommendations for Phase 1 and Phase 2

1. **Carrier Focus**: Concentrate exclusively on `C08` and `v12`. Both are already implemented in `p_only/bots/` with clean P-only headers (`c08_ponly.hpp` and `v12_ponly.hpp`).
2. **Fresh Seed Isolation**: All Phase 1 timing and Phase 2 screen runs must strictly use seed `20261031` to eliminate any map contamination from 20261020 or 20261009.
3. **Control Calibration**: Incorporate both `c08_control` (4-turn scripted) and `c08_control_2ply` (2-turn scripted), along with explicit A/A duplicate arms, to definitively isolate:
   - What portion of C08's score change is due to reducing search depth from 4 plies to 2 plies.
   - What portion is due to replacing hand-written scripts with pure neural $P$.
   - The baseline re-run noise floor.
