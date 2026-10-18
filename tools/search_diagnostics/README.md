# Agent V2 Search Diagnostics (`tools/search_diagnostics/`)

Authoritative, input-bound diagnostic harness for investigating why candidate search underperforms direct policy evaluation in `agent_v2`, delivering faithful causal branching, simulator parity audits, opponent $P$ fidelity verification, and winner's-curse-free evaluation (task T0085).

---

## 1. Background & Empirical Motivation

Recent host experiments established clear diagnostic evidence:
- **T0078 (Meowmind Search Ablation):** Search provided **+8.09** utility points `[+6.31, +9.88]`, proving that tree search can substantially improve tactical play in this game.
- **T0079 (agent_v2 Gated Search):** `agent_v2` search degraded performance by **-3.10** `[-4.47, -1.74]`.
- **T0081 (P3 vs P2 Search Screen, 3,920 games):** Upgrading opponent prediction from $P_2$ to $P_3$ did **not** rescue search ($-0.0040$ `[-0.0230, +0.0151]`), and $P_3\text{ search} - P_3\text{ direct}$ degraded performance by **-0.0197** `[-0.0390, -0.0004]`, reaching **-0.0684** in completed 3–5 team trees.

Prior exploratory scripts contained mock placeholders or synthetic labels. Task **T0085** delivers the real, mathematically rigorous, input-bound diagnostic harness to isolate whether failures arise from:
1. **Simulator mechanics divergence** (engine rules vs native simulator).
2. **Opponent prediction error** ($P$ fidelity on actual next-turn transitions).
3. **Candidate generation failure** (lack of positive-gain actions in shortlists).
4. **Value re-ranking / evaluation error** ($V$ ordering vs realized rollout returns).

---

## 2. Core Architectural Guarantees

### A. Real Causal Root Extraction (`extract.py`)
- **Actually Sampled Baseline:** Captures the true direct joint action $j_{\text{sample}} \in [0, 125)$ sampled by the policy, which is **not** necessarily the top-prior argmax action.
- **Nested Shortlists:** Produces unique candidate sets containing $j_{\text{sample}}$ (sizes 8 and optionally 24).
- **Exact Self-Comparison:** Candidate 0 is designated as the exact direct baseline order, guaranteeing $\Delta(a_{\text{base}}) \equiv 0$ on every repeat.
- **Resolver Completion:** Expands joint moves into complete orders using official `assign_v1` rules (carriers wait, immune units cannot attack, no drops, valid target distances).

### B. Faithful Branch Restoration (`branch.py`)
- **Hidden RNG Preservation:** Restores official `engine.py` `Game` state including hidden RNG and map/flag seeds for evaluator use.
- **Full Causal Prefix Replay:** Feeds recorded turns $1 \dots t-1$ into live bot processes (`fusion.runner.Bot`) to reconstruct internal memory, caches, and history. Divergent prefixes fail closed with `PrefixDivergenceError`.
- **Common Opponent Commands at Turn $t$:** All sibling candidate branches share the exact same submitted opponent commands at root turn $t$.
- **Reactive Live Continuations:** From turn $t+1$ to terminal, opponent bots observe and react live to the counterfactual branch state. Factual future opponent replies are **never** reused as counterfactual ground truth.

### C. Signed Baseline Gain & Regret (`analyze.py`)
- **Signed Baseline Gain:**
  $$\Delta(a) = \mathbb{E}[U_T \mid \text{root } a, \pi_c] - \mathbb{E}[U_T \mid a_{\text{direct}}, \pi_c]$$
  where $U_T \in [0, 1]$ is terminal pairwise rank utility (`fusion.labels.pairwise_outcome`).
- **Non-Negative Regret:**
  $$\hat{R}(a) = \max_{a'} \mathbb{E}[U_T \mid a'] - \mathbb{E}[U_T \mid a] \ge 0$$
  Explicitly distinguished: signed baseline gain is **not** negative regret.
- **Fractional Tie-Handling:** Candidates with tied utilities receive their average fractional rank (e.g. ties for 1st receive rank 1.5).
- **Winner's Curse Elimination (Cross-Fitting):** Evaluates apparent best candidates on disjoint repeat partitions (e.g., choosing on even repeats, evaluating on odd repeats) to prevent order-statistic maximization bias.
- **Map Clustering:** Grouping roots by `map_seed` ensures games remain strictly isolated across cross-validation splits.

### D. Mechanics Parity vs Opponent $P$ Error (`parity.py`, `p_eval.py`)
- **Simulator Parity:** Steps native simulator and `engine.py` on identical full commands, verifying HP, positions, immunity, scores, flag status, and simultaneous attacks.
- **$P$ Model Fidelity:** Evaluates movement distribution calibration (ECE), log-loss, and attack targeting on factual observed opponent moves, properly accounting for eligible unit denominators.

---

## 3. Command Line Interface Reference

The CLI entrypoint is `tools/search_diagnostics/cli.py`:

```bash
# Dry-run pilot cost estimation
python -m tools.search_diagnostics.cli estimate --markdown

# Simulator parity check against official engine.py
python -m tools.search_diagnostics.cli parity --check build/check.exe --ai-test c:/Users/caspe/OneDrive/Code/AI_Test

# Cross-fit statistical analysis of branch continuations
python -m tools.search_diagnostics.cli analyze --roots roots.json --branches branches.json --out report.json
```

---

## 4. Host Pilot Runbook & Cost Estimation

Per repository lease constraints:
- Executed on host machine `desktop-g60ojud`.
- Exactly 8 game workers under the CPU lease (`tools/repo/task.py lock cpu`).
- GPU training must be stopped during CPU execution.
- Bot processes run under official limits (2000 ms init, 50 ms turns).

### Authoritative Pilot Budget Table

Derived formulaically via `estimator.py` across 14 team counts (2..15):

| Parameter | Value | Unit |
|---|---:|---|
| Team Counts | 14 (2..15) | counts |
| Base Matches per Count | 16 | matches |
| Total Base Matches | 224 | matches |
| Total Causal Roots | 224 | roots |
| Candidates per Root ($K$) | 8 | orders |
| Repeats per Candidate ($R$) | 4 | seeds |
| **Total Continuation Games** | **7,168** | **games** |
| Average Continuation Length | 300 | turns |
| Total Engine Steps | 2,150,400 | steps |
| Total Bot Turn Inferences | 18,278,400 | calls |
| Workers (CPU lease) | 8 | workers |
| Step Latency Budget | 50.0 | ms |
| Total Compute Budget | 29.87 | CPU-hours |
| **Estimated Wall-Clock Runtime** | **3.73** | **hours** |
| Row Manifest Disk Size | 3.50 | MB |

---

## 5. Verification & Test Suite

Run the full unit and integration test suite:

```powershell
python -m unittest tools.search_diagnostics.test_search_diagnostics
```

The test suite covers:
1. `test_baseline_exact_self_comparison_requirement`: Validates Candidate 0 is the direct baseline order.
2. `test_baseline_exact_self_comparison_has_zero_gain`: Proves $\Delta(a_{\text{base}}) \equiv 0$ on all repeats and mean.
3. `test_regret_is_distinct_from_negative_signed_gain`: Proves $\hat{R}(a) \ge 0$ is relative to the best candidate, distinct from $\Delta$.
4. `test_fractional_ranking_with_ties`: Verifies proper fractional tie handling.
5. `test_cross_fitting_eliminates_winners_curse_on_noise`: Demonstrates on synthetic pure noise that naive oracle max has positive bias ($> +0.05$), while cross-fitting yields unbiased zero ($\pm 0.035$).
6. `test_wrong_state_history_refused`: Fails closed on missing or corrupted history.
7. `test_corrupted_branch_receipt_rejected`: Fails closed on invalid probabilities or utilities.
8. `test_duplicate_candidate_joints_rejected`: Fails closed on duplicate actions.
9. `test_map_clustering_keeps_same_map_together`: Verifies no map leakage across folds.
10. `test_p_fidelity_metrics_computation`: Validates ECE, cross-entropy, and denominators.
11. `test_estimator_mathematical_consistency`: Verifies exact derivation of rollout steps and hours.
