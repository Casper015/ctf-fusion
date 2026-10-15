# Decision Data Quality Report: T0038 Independent Audit of Host Pilot Corpus

**Task:** T0038
**Auditor:** `codex-data-audit@desktop-g60ojud`
**Host Machine:** `DESKTOP-G60OJUD` (Windows)
**Date:** 2026-10-14 (Revised 2026-10-15)
**Source Pilot:** T0037 Windows Host Decision Pilot (PR #44 merged, master seed `20261701`)
**Overall Verdict:** **GO** (Proceed to Small Linear/MLP Own-Action Scorer Training)

---

## 1. Executive Summary & Recommendation

This report presents an independent, read-only audit of the 336-map Windows host decision pilot corpus generated under T0037, and delivers the authoritative versioned decision training dataset (`ctf-fusion-decision-2`).

### Key Audit Findings
- **Match & Continuation Faults: 0.** Every one of the 336 matches and 5,268 continuations completed without match faults, missing reply slots, or continuation faults (prefix replay 0, root discarded 0, live continuation 0).
- **Trajectory Files & Matrix Completeness: 100% Verified.** All 5,268 individual continuation trajectory files on disk were checked; their SHA-256 digests match branch records bit-exactly. The candidate $\times$ repeat matrix is complete ($K \times 4$ across all 336 roots: 327 roots with $K=4$, 9 roots with $K=1$).
- **Exact Utility Recomputation: 100% Agreement.** Pairwise utilities recomputed independently from raw final team score vectors agree with stored match utilities and branch outcomes within $10^{-9}$ numerical tolerance.
- **Shortlist Scores & Real $K$: Verified.** All 336 roots carry valid `ctf-fusion-shortlist-scores-1` payloads with real finite floating-point logit scores for every candidate in order. No future information leaks into candidate or root features.
- **Causal Feature Extraction & Canonicalization: Bit-Exact.** Causal root features extracted from full recorded prefixes (`INIT + TURN 1..t`) match native engine extractions. Zero parse failures, zero illegal moves, zero out-of-cap targets, and zero feature collisions between distinct canonical actions.
- **Opponent Weights & Carrier Identity: Verified.** The carrier binary (`v12_ponly_xxl_dlog`), carrier source file, and all 12 opponent model weight headers in `inference/models_xxl/` were independently hashed and match admitted frozen receipts.
- **Compiled Dataset Alignment: Verified.** The authoritative dataset compiled at `data/decision/host_pilot_dataset/` was independently checked against raw matches and branches: root IDs, map seeds, shortlist orders, carrier scores, utilities, and zero fault arrays match with zero discrepancy.
- **Positive Net Signal Above Noise Floor:**
  - Train: within-action repeat variance $\sigma^2_{\text{within}} = 0.00729$; between-action variance raw $= 0.01223$, net $= 0.01041$.
  - Val: within-action repeat variance $\sigma^2_{\text{within}} = 0.00558$; between-action variance raw $= 0.01729$, net $= 0.01590$.
- **Diagnostic Out-of-Sample Headroom (Split-Repeat Method):**
  - Evaluated on independent repeat subsets (selecting the top alternative on even repeat IDs {0, 2} and measuring its gain over the incumbent on odd repeat IDs {1, 3}):
    - Train Headroom: **$+0.0525$** (95% CI: $[+0.0345, +0.0706]$, 260 clusters) $\to$ **Benefit**
    - Val Headroom: **$+0.0895$** (95% CI: $[+0.0176, +0.1614]$, 37 clusters) $\to$ **Benefit**
  - *Statistical Caveat:* This headroom is a diagnostic upper-bound measurement of the candidate pool's label-space discriminative potential under oracle label selection. It does **not** represent the achievable gain of any trained model or deployed policy.
- **Flag Protocol Defect Documented:** The historical collector tested for invented status `ground` instead of `home`/`dropped`, causing all 56 flag requested roots to fall back to rotation. Reclassified observable contexts yield 52 flag roots (15.5%) and 54 quiet roots (16.1%). Only 10/56 flag requests hit true flag context.

### Final Recommendation: GO
The corpus exhibits genuine predictive signal above noise and verified link/feature integrity. We recommend proceeding to small linear and MLP own-action training on the authoritative dataset. However, because of the flag selector defect and sample size constraints on the held-out test split ($M=31$), subsequent training tasks must explicitly acknowledge context distribution skew, evaluate generalization strictly on the held-out test split, and avoid claiming balanced flag-targeted policy optimization until expanded collection occurs.

---

## 2. Corpus Provenance & Schedule Audit

The audit verified all fixtures and raw archives produced by T0037 on `DESKTOP-G60OJUD`:
- **Master Seed:** `20261701` (reserved and consumed; strictly excluded historical seeds `20261501`, `20261250`, `20261260`).
- **Raw Root:** `data/decision/host_20261014/`
- **Build Root:** `build/decision_host_20261014/`
- **Schedule Structure:** 14 team counts ($n \in [2, 15]$), exactly 24 independent maps per team count = 336 matches.
- **Match Length:** Exactly 400 turns per match ($1,142,400$ total player reply slots).
- **Time Limits:** Official $2,000\,\text{ms}$ first reply, $50\,\text{ms}$ subsequent turns.
- **Seats & Opponents:** Designated candidate seat was verified across all matches; opponents were checked against the registered active pool without reserved lineage exposure or binary aliases.

| Metric | Scheduled / Declared | Audited & Verified | Status |
|---|---:|---:|:---:|
| Matches | 336 | 336 | Passed |
| Turns per match | 400 | 400 | Passed |
| Match faults (all seats) | 0 | 0 | Passed |
| Branch roots | 336 | 336 | Passed |
| Continuations | 5,268 | 5,268 | Passed |
| Continuation trajectory files on disk | 5,268 | 5,268 (SHA256 verified) | Passed |
| Candidate $\times$ repeat matrix ($K \times 4$) | 336 roots | 336 roots complete | Passed |
| Prefix replay faults | 0 | 0 | Passed |
| Discarded root computed faults (`root_discarded`) | 0 | 0 | Passed |
| Live continuation faults | 0 | 0 | Passed |
| Partial continuations | 0 | 0 | Passed |
| Prefix order mismatches retained | 18,972 | 18,972 | Passed |

---

## 3. Designated Carrier & Shortlist Scoring Verification

The designated log-only carrier `v12_ponly_xxl_dlog` (binary SHA256 `981a1e3c580021c6b5879e3ca3feb6de44642db4272aba25e5c97be22bac45b3`) was verified against the frozen preflight receipts:
- **Variant:** `v12_ponly_xxl_dlog`
- **Opponent Policy:** `p2 XXL` (`inference/models_xxl/`)
- **Value Tier:** `off` (pure P-only policy)
- **Search Source:** `bots/p_only/bots/variants/v12_ponly_xxl_dlog.cpp` (SHA256 `98b4e2f0f6d24be5e17b08a39d4f8d67b97bc4af5a686cffdfb8281c1e8d9e78`)
- **Opponent Model Headers:** All 12 header files in `inference/models_xxl/` (`registry.hpp`, `weights_general.hpp`, and `weights_n02.hpp` through `weights_n15.hpp`) match pinned SHA-256 digests bit-exactly.
- **Shortlist Orders & Scores:**
  - Candidate 0 is on-policy reply: `normalize_line(match.replies[turn-1][seat]) == cands[0]`.
  - All candidate orders are normalized, distinct, and match candidate lines in `row["candidates"]`.
  - Scores are real finite floating-point logit scores.
  - Candidate distribution: 327 roots with $K=4$ ($1,308$ candidates) and 9 roots with $K=1$ ($9$ candidates) = $1,317$ total candidates.

---

## 4. Flag Selector Defect & Corrected Context Coverage

During T0037 collection, `host_collect.py` checked for flag status `ground`, whereas the CTF engine protocol uses `home` and `dropped`. Consequently:
- All 56 flag-requested maps fell back to the rotating tactical context selector.
- `host_flag_coverage.py` supplied the corrected protocol predicate and reclassified all 336 selected roots:

| Tactical Context | Maps with Opportunity | Opportunity Turns | Requested Target Hits | Selected Root Count | Selected Map Rate |
|---|---:|---:|---:|---:|---:|
| Carrier | 336 | 38,727 | 56 / 56 (100.0%) | 117 | 34.8% |
| Contact | 336 | 91,523 | 56 / 56 (100.0%) | 240 | 71.4% |
| Defense | 320 | 16,207 | 51 / 56 (91.1%) | 88 | 26.2% |
| Interception | 336 | 52,950 | 56 / 56 (100.0%) | 132 | 39.3% |
| Quiet | 336 | 19,407 | 50 / 56 (89.3%) | 54 | 16.1% |
| Flag | 336 | 24,519 | **10 / 56 (17.9%)** | **52** | **15.5%** |

*Interpretation:* The pilot corpus has adequate tactical diversity (52 ground-flag opportunities and 54 quiet opportunities captured), but flag context was acquired via rotation fallback rather than targeted selection.

---

## 5. Authoritative Dataset & Split Isolation

The dataset was compiled using `training.decision_dataset` under `--authoritative` mode:
- **Output Path:** `data/decision/host_pilot_dataset/`
- **Receipts Bound:**
  - `DATA_AUDIT.json`: SHA256 verified
  - `MANIFEST_METADATA_ADAPTER.json`: Verification scope `selected_frozen_identities_only` accepted
  - `carrier_config.json`: Validated against `inference/models_xxl/weights_general.hpp` and canonical LF manifest/identities hashes
- **Source Digests (`ctf-fusion-decision-source-1`):**
  - `branches_sha256`: `d4b13e1c172caebfbe0b0822fb3ec95cf4f9c81680cb70e1994583cee5b3c38b`
  - `matches_sha256`: `65d979158d2bb8d19db1184c1f3895e5c1e7c9ce157c83b2318ae7879eb643cf`
  - `root_content_manifest_sha256`: `d1a3e89a14354f8e416f93950a3045d9b2a033d4a95530974b99654d86ca35ea`

### Split Statistics

| Split | Maps | Roots | Changeable ($K \ge 2$) | Candidates | Missing Outcomes | Dropped Roots | Split npz SHA256 |
|---|---:|---:|---:|---:|---:|---:|---|
| **train** | 268 | 268 | 260 | 1,048 | 0 | 0 | `70441e0197382c630b2f64f230e54722ae031bb184e5a73e3874e08b00b877bc` |
| **val** | 37 | 37 | 37 | 148 | 0 | 0 | `74b630efa4e80bd2f947c45a816751124b06c155bc6e05533d88adb0255b5ace` |
| **test** | 31 | 31 | 30 | 121 | 0 | 0 | `1b9a0dd6719c27d98e621248dda06c841df0b54a61ea3edb5f8d5d8541cce598` |
| **Total** | **336** | **336** | **327** | **1,317** | **0** | **0** | *(Authoritative)* |

*Isolation:* Map sets are strictly mutually disjoint ($\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$). Zero split contamination.

---

## 6. Noise, Covariance & Headroom Diagnostics

Noise and signal characteristics were evaluated with `training.decision_noise` over noise-root populations:

| Split | Noise Roots | Within-Action Var ($\sigma^2_w$) | Between-Action Raw ($\sigma^2_{b,\text{raw}}$) | Between-Action Net ($\sigma^2_{b,\text{net}}$) | Paired Gap Var ($\sigma^2_{\Delta}$) | Implied Covariance ($\text{Cov}$) |
|---|---:|---:|---:|---:|---:|---:|
| **train** | 260 | 0.00729 | 0.01223 | **0.01041** | 0.01620 | -0.00055 |
| **val** | 37 | 0.00558 | 0.01729 | **0.01590** | 0.01191 | +0.00003 |

### Methodological Observations & Limitations
1. **Net Signal Exceeds Noise:** The net between-action variance ($0.0104$ train, $0.0159$ val) is substantially positive after subtracting within-action sampling variance ($0.0073$). The policy alternatives exhibit real outcome divergence.
2. **Repeat Covariance Interpretation:**
   - Implied covariance between repeats of different candidates is small ($-0.00055$ train, $+0.00003$ val).
   - *Limitation:* Small empirical covariance indicates that cross-candidate variation is not dominated by candidate-correlated branch noise in this sample. However, small covariance does **not** prove statistical independence: repeats branch from identical prefix states against reactive bot policies on the same map, meaning shared trajectory dynamics remain.
3. **Replay Memory Diagnostic:**
   - Train mean difference (original match utility minus mean incumbent continuation): $+0.0107$ ($\text{sd} = 0.1049$).
   - Val mean difference: $+0.0089$ ($\text{sd} = 0.0811$).
   - *Limitation:* The mean difference is small relative to turn-level variance ($s \approx 0.10$), indicating no massive systemic bifurcation across the 400 turns. However, small sample mean differences do **not** prove that continuations are statistically unbiased, and trajectory divergences do occur.

### Out-of-Sample Headroom (Diagnostic Split-Repeat Method)
To avoid the optimistic bias of $\max(\text{noisy means})$, headroom was estimated by selecting the best alternative on even repeat IDs {0, 2} and evaluating its advantage over the incumbent on odd repeat IDs {1, 3}:

$$\Delta_{\text{headroom}} = \mathbb{E}\left[ Y_{c^*}^{\text{odd}} - Y_{\text{inc}}^{\text{odd}} \right], \quad c^* = \arg\max_{c} Y_c^{\text{even}}$$

- **Train Headroom:** **$+0.0525$** (95% CI: $[+0.0345, +0.0706]$, 260 clusters) $\to$ **Benefit**
- **Val Headroom:** **$+0.0895$** (95% CI: $[+0.0176, +0.1614]$, 37 clusters) $\to$ **Benefit**

*Critical Interpretation:* Both confidence intervals exclude zero and exceed the negligible margin ($0.002$). Crucially, this split-repeat metric is a **diagnostic measurement of candidate pool discriminative headroom under oracle continuation-label selection**. It establishes that alternative candidates contain genuine win-rate signal above noise. It does **not** measure or guarantee the realized policy gain of an actual parametric ranker or deployed model.

---

## 7. Power Analysis & Generalization Scope

Using train/val map-level variance ($s^2_{\text{map}} = 0.01452$) and an assumed switch share $f$, the planning approximation for Minimum Detectable Effect is:

$$\text{MDE}(M, f) = 2.8 \times \sqrt{\frac{f \cdot s^2_{\text{map}}}{M}}$$

| Total Independent Maps ($M$) | Switch Share $f = 0.10$ | Switch Share $f = 0.25$ | Switch Share $f = 0.50$ |
|---|---:|---:|---:|
| 300 | 0.0062 (0.62%) | 0.0097 (0.97%) | 0.0138 (1.38%) |
| 400 | 0.0053 (0.53%) | 0.0084 (0.84%) | 0.0119 (1.19%) |
| 500 | 0.0048 (0.48%) | 0.0075 (0.75%) | 0.0107 (1.07%) |

### Held-Out Evaluation Constraints
- **Test Split Sample Power:** Generalization testing on the held-out test split is strictly limited to **$M=31$ independent maps**.
- **No Split Conflation:** The 268 training maps do **not** contribute to test-set sample power. On $M=31$ maps, statistical power is limited, and the MDE for held-out evaluation is substantially larger ($\approx 2.5\% - 4.0\%$ at moderate switch rates).
- **Future Expansion:** Detecting finer effect sizes on held-out evaluation will require future dataset expansion.

---

## 8. Preliminary Bot Evaluations Note

Preliminary smoke checks (such as single-match or multi-bot runs) demonstrate functional compilation, memory safety, and protocol execution compatibility. However, they do **not** constitute formal proof of distillation dominance or official production admission. Formal admission requires rigorous paired A/A and multi-team controls under frozen benchmark protocols.

---

## 9. Reproducibility & Integrity Receipts

### Exact Reproduction Commands

```bash
# 1. Run independent audit and regenerate DATA_AUDIT.json
python bots/p_only/report/decision_dataset/audit_pilot.py

# 2. Build authoritative decision training dataset
python -m training.decision_dataset \
    --branches data/decision/host_20261014/branches \
    --matches data/decision/host_20261014/matches \
    --audit bots/p_only/report/decision_dataset/DATA_AUDIT.json \
    --manifest-receipt bots/p_only/report/decision_dataset/MANIFEST_METADATA_ADAPTER.json \
    --carrier-config bots/p_only/report/decision_dataset/carrier_config.json \
    --accept-selected-scope \
    --authoritative \
    --out data/decision/host_pilot_dataset

# 3. Generate coverage, noise and power diagnostics
python -m training.decision_noise \
    --data data/decision/host_pilot_dataset \
    --out reports/eval/decision_data_quality/noise
```

### Artifact Manifest

All tracked artifacts are canonicalized to UNIX LF line endings per repository configuration (`* text=auto eol=lf`). Checksums are verified bit-exact on LF bytes in `CHECKS.json`:

- `bots/p_only/report/decision_dataset/DATA_AUDIT.json`: Audited match/branch/continuation receipts, trajectory file hashes, and split digests
- `bots/p_only/report/decision_dataset/MANIFEST_METADATA_ADAPTER.json`: Pinned manifest scope adapter
- `bots/p_only/report/decision_dataset/manifest.json`: Collection build manifest for `v12_ponly_xxl_dlog` (LF canonical)
- `bots/p_only/report/decision_dataset/identities.json`: Component identities for `v12_ponly_xxl_dlog` (LF canonical)
- `bots/p_only/report/decision_dataset/carrier_config.json`: Pinned carrier configuration with LF manifest/identity hashes
- `bots/p_only/report/decision_dataset/audit_pilot.py`: Source-bound independent audit script
- `reports/eval/decision_data_quality/DATA_QUALITY_RECEIPTS.json`: Machine-readable summary receipts
- `reports/eval/decision_data_quality/noise.json`: Detailed noise diagnostics
- `reports/eval/decision_data_quality/noise.md`: Tabular noise summary
