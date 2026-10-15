# First Own-Action Trial Report: Linear and MLP Training with Frozen Evaluation

**Task:** T0050  
**Agent:** `codex-decision-pilot@desktop-g60ojud`  
**Host Machine:** `desktop-g60ojud` (Windows)  
**Date:** 2026-10-15  
**Accepted Dataset:** `data/decision/host_pilot_dataset` (Schema: `ctf-fusion-decision-2`, Seed: 20261701, PR #46 / T0038)  
**Deliverables Directory:** `reports/models/own_action_first_trial/`

---

## 1. Executive Summary

This report documents the completion of task **T0050**: the first bounded offline own-action trial comparing linear and MLP decision rankers trained on incumbent-relative continuation outcome gains ($A^{\pi_0}(s, a) = \mathbb{E}[U_T \mid s, a] - \mathbb{E}[U_T \mid s, a_0]$).

- **Strict Protocol Compliance:** Exactly six runs were trained inside the host CPU lease using the accepted T0038 host dataset (336 maps, train/val/test splits of 268/37/31 maps). `PLAN.json` was committed prior to optimization.
- **Rule Selection:** Decision rules were tuned exclusively on the 37 validation maps with the proposed incumbent-anchored `zscore` normalization and negligible margin $\pm 0.002$. The frozen selection was committed in `SELECTION.json` before opening the test split:
  - **Primary model (`mlp_s101`):** selected $\lambda = 100.0$ (validation mean $\Delta = +0.0166$).
  - **Secondary comparator (`linear_s101`):** selected $\lambda = 50.0$ (validation mean $\Delta = +0.0604$).
- **Single Test Evaluation:** Evaluated once on the 31 independent test maps:
  - **`mlp_s101` vs Incumbent (Primary):** $\text{Mean } \Delta = -0.0151$, 95% Map-Clustered CI $[-0.0563, +0.0261]$. Verdict: **`insufficient precision`** (CI spans 0 and exceeds the negligible margin).
  - **`linear_s101` vs Incumbent (Secondary):** $\text{Mean } \Delta = -0.0203$, 95% Bonferroni Map-Clustered CI $[-0.0846, +0.0440]$. Verdict: **`insufficient precision`**.
  - **Diagnostic Headroom:** $+0.0487$ $[+0.0003, +0.0970]$, confirming that theoretical headroom exists on the test split, but neither small model captured generalizable positive utility gain on the test maps.

---

## 2. Dataset and Identity Verification

The trial operated strictly on the accepted T0038 host corpus without re-sampling, relabeling, or altering data splits. All file hashes and NPZ digests were verified against `CHECKS.json` and `FINAL_ACCEPTANCE.json` prior to execution:

| Asset | Path | Verified SHA-256 Digest |
|---|---|---|
| Dataset Index | `data/decision/host_pilot_dataset/index.json` | `d3a19bd3de638118b60f367ed1eed68fb02a86b95e1978b50f3bf76b5a1ae0b3` |
| Train Split ($M=268$) | `data/decision/host_pilot_dataset/train/decision.npz` | `70441e0197382c630b2f64f230e54722ae031bb184e5a73e3874e08b00b877bc` |
| Val Split ($M=37$) | `data/decision/host_pilot_dataset/val/decision.npz` | `74b630efa4e80bd2f947c45a816751124b06c155bc6e05533d88adb0255b5ace` |
| Test Split ($M=31$) | `data/decision/host_pilot_dataset/test/decision.npz` | `1b9a0dd6719c27d98e621248dda06c841df0b54a61ea3edb5f8d5d8541cce598` |
| Carrier Config | `bots/p_only/report/decision_dataset/carrier_config.json` | `f0cea701efafd583b9b7a2029149ceaad84b4f3872982ee1c350769660a7636d` |
| Carrier Binary | `v12_ponly_xxl_dlog` | `981a1e3c580021c6b5879e3ca3feb6de44642db4272aba25e5c97be22bac45b3` |
| Feature Digest | `own-action-1` (layout v3) | `9ed719f238cd5d70` |

---

## 3. Training Execution and Results

Training was conducted under the host exclusive CPU lease (`task.py lock cpu --task T0050`) using the existing NumPy CPU Adam trainer (`training.train_decision_ranker`). All runs used identical hyperparameters:
- Loss: `gain` (weighted MSE against continuation outcome differences)
- Batch size: 256
- Learning rate: $1 \times 10^{-3}$
- Weight decay: $1 \times 10^{-4}$
- Max epochs: 50, Early stopping patience: 5 (monitored on validation loss)

### Run Receipts Summary

| Run Name | Architecture | Optimizer Seed | Trained Epochs | Best Val Epoch | Final Train Loss | Best Val Loss | Model SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| `linear_s101` | Linear | 101 | 45 | 40 | 0.01948 | 0.04483 | `5f6b888f22aa...` |
| `linear_s102` | Linear | 102 | 41 | 36 | 0.02012 | 0.05622 | `7c1444d037b0...` |
| `linear_s103` | Linear | 103 | 50 | 49 | 0.02059 | 0.05139 | `670215e89ca4...` |
| `mlp_s101` | MLP [64, 16] | 101 | 13 | 8 | 0.01310 | 0.04786 | `ff01c27e73ce...` |
| `mlp_s102` | MLP [64, 16] | 102 | 22 | 17 | 0.00968 | 0.04726 | `40aacc6a225c...` |
| `mlp_s103` | MLP [64, 16] | 103 | 14 | 9 | 0.01323 | 0.04568 | `ca768042433f...` |

### Overfitting and Optimization Dynamics
1. **Model Expressivity vs. Generalization:** The MLP models achieved markedly lower training losses ($0.0097 - 0.0132$) than the linear models ($0.0195 - 0.0206$). However, their validation losses plateaued rapidly within 8–17 epochs at $\sim 0.046 - 0.048$, indicating quick memorization of training-specific order interactions.
2. **Run-to-Run Variation:** Linear runs showed moderate sensitivity to initialization seeds on validation loss ($0.0448$ to $0.0562$), while MLP runs exhibited tighter best validation losses ($0.0457$ to $0.0479$).

---

## 4. Frozen Validation Selection

Using `training.evaluate_decisions select`, each model's candidate scoring was evaluated on the 37 validation maps across the standard grid $\lambda \in \{0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0\}$ with the `zscore` norm:

- **`mlp_s101` (Pre-declared Primary):**
  - $\lambda = 0.0$: $\Delta = 0.0000$ (Changed share: 0.0%)
  - $\lambda = 20.0$: $\Delta = +0.0088$ (Changed share: 37.8%)
  - $\lambda = 50.0$: $\Delta = +0.0143$ (Changed share: 67.6%)
  - $\lambda = 100.0$ (Selected): $\text{Mean } \Delta = +0.0166$, Changed share: 70.3%
- **`linear_s101` (Main Secondary Comparator):**
  - $\lambda = 0.0$: $\Delta = 0.0000$ (Changed share: 0.0%)
  - $\lambda = 20.0$: $\Delta = +0.0486$ (Changed share: 37.8%)
  - $\lambda = 50.0$ (Selected): $\text{Mean } \Delta = +0.0604$, Changed share: 59.5%
- All selected parameters, model hashes, and dataset digests were locked into `SELECTION.json` (SHA-256: `a571b05a4473eaaa...`) and committed prior to opening the test set.

---

## 5. Single Frozen Test Evaluation

The frozen selection was evaluated exactly once via `training.evaluate_decisions test` against the 31 test maps ($M=31$, 31 roots, 121 candidates).

### Offline Test Results

| Comparison | Role | Mean $\Delta$ | 95% Clustered CI | Independent Maps | Verdict |
|---|---|---:|---|---:|---|
| **`mlp_s101 - incumbent`** | **Primary** | **-0.0151** | **[-0.0563, +0.0261]** | **31** | **`insufficient precision`** |
| `linear_s101 - incumbent` | Secondary | -0.0203 | [-0.0846, +0.0440] | 31 | `insufficient precision` |
| `linear_s102 - incumbent` | Secondary | -0.0263 | [-0.0862, +0.0337] | 31 | `insufficient precision` |
| `linear_s103 - incumbent` | Secondary | -0.0167 | [-0.0714, +0.0380] | 31 | `insufficient precision` |
| `mlp_s102 - incumbent` | Secondary | -0.0339 | [-0.0962, +0.0284] | 31 | `insufficient precision` |
| `mlp_s103 - incumbent` | Secondary | -0.0188 | [-0.0820, +0.0445] | 31 | `insufficient precision` |
| `mlp_s101 - linear_s101` | Secondary | +0.0052 | [-0.0275, +0.0379] | 31 | `insufficient precision` |

*(Note: Secondary comparisons include Bonferroni multiplicity correction over 10 secondary comparisons).*

### Root-Level Rule Behavior on Test Set

| Rule | Evaluable Roots | Changed Share | Mean $\Delta$ Where Changed |
|---|---:|---:|---:|
| `incumbent` (Control) | 31 | 0.0% | — |
| `linear_s101` | 31 | 58.1% | -0.0420 |
| `linear_s102` | 31 | 54.8% | -0.0385 |
| `linear_s103` | 31 | 41.9% | -0.0403 |
| `mlp_s101` | 31 | 61.3% | -0.0334 |
| `mlp_s102` | 31 | 67.7% | -0.0389 |
| `mlp_s103` | 31 | 64.5% | -0.0350 |

- **Diagnostic Baseline (`carrier_top_score`):** $\Delta = +0.0000$ $[+0.0000, +0.0000]$ (incumbent was identical to carrier top score on all 31 test roots).
- **Test Set Diagnostic Headroom:** $+0.0487$ $[+0.0003, +0.0970]$ (evaluated across independent odd/even repeat subsets).

---

## 6. Interpretation and Findings

1. **Validation Over-Optimization vs. Out-of-Distribution Generalization:**  
   Because the validation set had only 37 independent maps, the grid search selected relatively large values ($\lambda = 50$ and $\lambda = 100$). On the test set, these large multipliers caused high change rates ($58.1\%$ for linear, $61.3\%$ for MLP). On roots where the model overrode the incumbent, the substituted orders performed worse than the incumbent by $-0.0334$ to $-0.0420$ on average.
2. **Point Estimates vs. Confidence Bounds:**  
   Point estimates are slightly negative for all seeds (ranging from $-0.015$ to $-0.034$), but the 95% map-clustered confidence intervals all cross zero. Under the pre-declared evaluation rules, the formal statistical verdict is **`insufficient precision`** rather than demonstrated harm or demonstrated benefit.
3. **No Retraining or Tuning:**  
   Per the strict anti-leakage commitments of T0050, no hyperparameter adjustments, seed reselection, or additional test evaluations were performed after inspecting the test split.

---

## 7. Recommendations for Task T0051 (Host Pilot)

Task **T0051** specifies a bounded host pilot with export, deadline gating, and paired matches:
1. **Admissibility Assessment:**  
   Both `mlp_s101` ($\lambda = 100$) and `linear_s101` ($\lambda = 50$) have non-zero lambda selections and valid exported weights. However, neither demonstrated statistically significant positive gain offline on test maps (`insufficient precision`).
2. **Deployment Feasibility & Caution:**  
   If T0051 executes the bounded paired-game pilot (392 fresh maps across team counts 2–15, matched replacement arm, byte-identical A/A control), it must proceed strictly under the pre-declared protocol:
   - Full deadline enforcement ($2,000\,\text{ms} / 50\,\text{ms}$ with hard engine-side reply drop).
   - Equal-time paired replacement with unchanged `v12_ponly_xxl` control.
   - Reporting truthful map-clustered intervals without claiming positive strength gain unless established in live paired games.
   - If timing preflight reveals unsafe deadlines, T0051 must deliver a truthful no-admission report.
