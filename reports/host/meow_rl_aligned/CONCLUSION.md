# Aligned Meowmind RL Evaluation: Conclusion and Analysis

## 1. Executive Summary

- **What improved:** Reinforcement learning fine-tuning on pool v2 produces a statistically credible improvement in direct policy strength against the held-out reference roster (`Fable`, `MiMo`, `Kimi`, `GLM`, `Player4`). Under unified 16,000-decision calibration and identical architecture, the primary actor `p2_large` gains **+3.84 percentage points** over `warm` (95% CI `[+1.66, +6.01]`, verdict: **helps**). In addition, WideAct16 integer arithmetic (`w16`) completely resolves the severe 7-bit quantization distortion of `r16`, boosting search utility by **+10.10 percentage points** (Bonferroni `[+6.55, +13.66]`, verdict: **helps**) over `r16`.
- **What failed:** Search itself—even with exact FP32 arithmetic (`fp32`) or WideAct16 (`w16`)—consistently underperforms the direct actor `p2_large` by **-3.91 pp** (`[-6.95, -0.87]`) and **-4.66 pp** (`[-7.88, -1.43]`), both carrying the verdict **hurts**. 2-ply expectimax search degrades performance relative to playing the policy prior directly.
- **What remains unknown:** Whether the residual search deficit (-3.9 to -4.7 pp) is caused by: (1) lookahead overriding the actor on near-ties under sampling noise (~8 leaves/turn, where 81.9% of roots have F margin < 0.01), (2) simulator rollout dynamics deviating from the real engine, (3) the opponent forecasting model mismatching actual multi-agent play during 2-ply lookahead, or (4) critic value scoring failing to correlate with true win probability when rolling out non-stationary tactical branches.
- **Open question on training:** The data demonstrate that direct actor PPO fine-tuning works (+3.84 pp over warm start). Because the direct actor is the deployed default, whether to scale actor training or hold further training until lookahead search dynamics are clarified remains an open question for Casper.

---

## 2. Schedule A: Direct Actor Alignments

Schedule A evaluated 1,960 formal games (392 fixtures × 5 arms, seed 20261954, 8 workers, 2,000 ms / 50 ms limits) with zero timeouts, zero faults, and 100% turn counter capture (156,800 turns per arm).

| Comparison | Mean (pp) | Interval | Multiplicity | Verdict |
|---|---:|:---:|:---:|:---:|
| `p2_large - warm` | **+3.84** | `[+1.66, +6.01]` | 95% (z = 1.9600) | **helps** |
| `p2_large - p1_large` | +1.17 | `[-1.05, +3.39]` | Bonferroni 2 (z = 2.2414) | not shown |
| `p2_large - p2_small` | -0.59 | `[-2.82, +1.65]` | Bonferroni 2 (z = 2.2414) | not shown |

- **RL Effect:** Unlike historical screens where candidate and control had divergent calibration corpora, both arms here share the exact same 16,000-decision calibration set. The +3.84 pp gain is legitimately attributable to PPO fine-tuning, noting that this is one checkpoint from one training seed.
- **Opponent Pools:** Training on pool v2 (incorporating AI_Test lineages) shows a directional +1.17 pp over pool v1 (`p1_large`), but the interval encompasses zero.
- **Critic Capacity in Actor Training:** `p2_large` and `p2_small` actors show virtually identical performance (-0.59 pp, interval spans zero). Training with a large critic (416→1024→1024→512→1, `meow_public416_mlp_1024_1024_512_v1`) vs a small critic (416→64→32→1, `meow_public416_mlp_64_32_v1`) produced no statistically detectable policy divergence on this sample.
- **Re-run (A/A):** `p2_large~aa - p2_large` yielded 392 of 392 identical fixture score vectors (+0.00 pp), showing identical replays on these 392 fixtures (same binary, same seed).

---

## 3. Schedule B: Search, Quantization, and Arithmetic Precision

Schedule B evaluated 1,960 formal games (392 fixtures × 5 arms, seed 20261955, 8 workers, 30 ms search deadline) with zero faults, zero timeouts, and 1,240,000+ tree leaves evaluated.

| Comparison | Mean (pp) | Interval (Bonferroni 5, z = 2.5758) | Verdict |
|---|---:|:---:|:---:|
| `fp32 - direct` | **-3.91** | `[-6.95, -0.87]` | **hurts** |
| `r16 - direct` | **-14.76** | `[-18.11, -11.41]` | **hurts** |
| `w16 - direct` | **-4.66** | `[-7.88, -1.43]` | **hurts** |
| `w16 - r16` | **+10.10** | `[+6.55, +13.66]` | **helps** |
| `fp32 - r16` | **+10.85** | `[+7.54, +14.16]` | **helps** |

### Decomposition of the Search Deficit

The comparison structure cleanly disentangles two distinct phenomena:

1. **Arithmetic Quantization Error:**
   - `r16` used 7-bit hidden activations (the exporter design inherited from T0056), which suffered severe layer saturation; on Schedule B, this produced a **-14.76 pp crash** relative to direct play (`[-18.11, -11.41] pp`, verdict: hurts).
   - Fixed-work diagnostics (1,024 trees, 8,112 leaves) predicted this: `r16` had a 0.1955 MAE against FP32 and only 33.5% top-action agreement.
   - WideAct16 (`w16`) uses 15-bit hidden activations and INT64 accumulators. On the diagnostic leaves, `w16` reduced MAE to 0.00134 and restored 97.3% top-action agreement.
   - On the board, `w16` restored **+10.10 pp** over `r16`. Note that `w16 - fp32` was not a predeclared contrast in Schedule B's Bonferroni family; computing the paired `w16 - fp32` contrast post hoc across the 392 fixtures yields a descriptive difference of -0.75 pp (95% CI `[-2.84, +1.34] pp`, SE 1.07 pp, outside the Bonferroni family), which is statistically indistinguishable from zero.
2. **Search Mechanics Degradation:**
   - Despite perfect arithmetic fidelity, FP32 search is still **-3.91 pp worse** than playing the direct actor without search.
   - The search loss is therefore not an arithmetic or quantization defect. It is a fundamental property of the search setup itself (rollout modeling, simulator fidelity, or value function calibration under rollout states).

---

## 4. Synthesis with Host Preflight Limits

- The eight-worker preflight (`reports/host/meow_rl_aligned/PREFLIGHT.json`, 448 games, seed 20261956) and Schedule A verified CPU execution limits on Windows:
  - Actor-direct latency: across Schedule A, p99 median is about 0.95 ms (ranging 0.92–0.95 ms across arms); preflight steady maximum reply is 2.99 ms at 15 teams (`p2_large`); Schedule A's maximum observed reply reached 31.8 ms (on `p1_large`, well within the 50 ms engine turn limit).
  - Search latency: steady maximum reply < 20 ms in preflight (18.98 ms for FP32 at 15 teams), never exceeding the 30 ms search deadline or 50 ms turn limit.
  - Zero search refusals, zero late replies, and zero process faults across all preflight and formal matches.
- Turn-by-turn counter logging via unbuffered `stderr` (`_IONBF` + explicit `fflush`) operated reliably without pipe truncations.

---

## 5. Explicit Methodological Disclaimers

To maintain strict scientific integrity across the research log:
1. **Do not cite historical T0057 as "RL gain confirmed":** T0057 suffered from disparate calibration corpora between arms (n04 400-row vs candidate 16,000-row batches), confounding calibration drift with policy improvement. The verified RL gain is confirmed only now by Schedule A.
2. **Do not equate A/A identity with "zero evaluation noise":** The 392/392 identical score match on A/A proves runtime determinism and harness reproducibility given a fixed seed; it does not eliminate map sampling variance, opponent seating variance, or seed-to-seed variance.
3. **Do not infer a general "large-V mechanism" from one seed:** The small vs large value network comparison (`p2_large - p2_small` = -0.59 pp, interval spanning zero) reflects one training run checkpoint. General capacity claims require multi-seed trials.
4. **Do not declare a "strongest bot" from raw sample means:** `p2_small` had an unadjusted mean of 0.79058 vs `p2_large` 0.78471; their paired contrast has an interval of `[-2.82, +1.65] pp`, meaning they are statistically indistinguishable.

---

## 6. Forward Proposal: Search Hypotheses and Follow-Up Diagnostics

Because arithmetic quantization is resolved yet search still hurts, the empirical evidence points to several hypotheses:

- **Hypothesis 1 (Sampling Noise on Near-Ties - Most Direct):**
  Schedule B evaluated about 7.9 leaves per turn (1.24M leaves / 156,800 turns), which was nearly identical across `fp32`, `r16`, and `w16`. This confirms the 30 ms deadline did not bind. However, T0063 fixed-work diagnostics show that search overrides the actor's own choice on 73.3% of roots, while 835 of 1,020 evaluated roots (81.9%) have an F top-two value margin under 0.01. With only ~8 samples per turn, search routinely overrides a strong policy prior on negligible value differences, effectively introducing sampling noise into move selection.
  - *Cheaper pre-test proposal (do not run in T0070):* Before undertaking a full real-engine simulator harness, test whether: (a) evaluating more samples per root at fixed work narrows variance, or (b) introducing an override gating threshold (only allowing search to override the actor when the value margin exceeds a minimum threshold) eliminates the search deficit. This test requires a fresh seed ledger.
- **Hypothesis 2 (Simulator Divergence):** The internal C++ simulator used by `search_v2.hpp` differs from the true arena environment in subtle collision, pickup, or tag resolutions.
- **Hypothesis 3 (Opponent Model Misalignment):** Opponent prediction rollouts diverge over multi-ply trajectories when opponents react to tactical aggression.
- **Hypothesis 4 (Out-of-Distribution Rollout States):** The value network evaluates synthetic board states visited during 2-ply rollouts that are out of distribution compared to the on-policy trajectory distribution it was trained on.

### Proposed Real-Engine Continuation Task (Separately Budgeted):
- **Title:** `Host: Real-Engine Continuation Diagnostics and Search Policy Alignment`
- **Scope:**
  - Build a continuation harness that executes candidate rollouts inside the actual match engine rather than the standalone simulator.
  - Evaluate value predictions on true continuation states vs synthetic simulator leaves.
  - Measure opponent policy log-loss conditionally on multi-step rollouts.
  - Require a dedicated fresh seed ledger in `docs/CONTINUATION_SEEDS.md`.
- **Rule:** Do NOT run this proposal within T0064/T0070; schedule as an independent downstream investigation.

---

## 7. Open Strategic Decision on Future Training

- **What the data show:**
  - Direct policy PPO fine-tuning reliably improved playing strength: `p2_large` achieved a statistically credible **+3.84 pp gain** (`[+1.66, +6.01]`, verdict: **helps**) over the BC warm start on the held-out reference roster under unified 16k calibration.
  - The direct actor is the deployed default, where RL demonstrably helps.
  - Lookahead search underperforms direct actor play by -3.91 pp (FP32) and -4.66 pp (WideAct16). This search deficit is an orthogonal property of tree search and rollout modeling, not a defect or failure of the actor policy itself.
- **Decision for Casper:**
  - Whether to proceed with further PPO training (e.g., scaling actor capacity, increasing training iterations, or expanding opponent training lineages) or pause training until search lookahead mechanics and rollout modeling are resolved is an open strategic decision left for Casper.
