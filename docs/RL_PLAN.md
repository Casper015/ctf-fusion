# Decision learning on top of the frozen P and search

This plan was first written for T0033 (2026-10-11). It was revised for T0034 (2026-10-12) after Casper retired the unstarted V pilots. T0034 is code only. **No real training, host measurement, new game, continuation batch, model export or strength improvement happened in it.** The code is tested on small fixtures: real recorded prefixes with synthetic outcomes.

**The direction.**
- **P predicts the opponents** and stays frozen.
- **A separate own-action scorer chooses among our own complete orders** inside a legal shortlist. A later experiment may choose team tasks instead.
- **Its benefit is unproven.** Lower loss, more changed decisions and richer features are not strength claims.

## 1. What changed

**Retired (2026-10-12), unstarted:**
- **T0012**, the V pilot: M/4XL sweeps, the ranking arm, V distillation.
- **T0014**, its deployment screen and leaf-value coefficient tuning.

Their task files and every completed V experiment stay as they are: T0003, T0016, T0022 and the T0010 data. Nothing here revives the sweep, the distillation or the coefficient tuning. **Deployed V stays disabled.** The action scorer is a separate component with its own model kind. It is never written over the V route, and never over an opponent P artifact.

**Dropped from the first version of this plan:**
- **Its automatic kill rules.** r_AA ≥ 0.8 · r_cross and a 50 % feature-collision rate are now diagnostics, nothing more. Section 3 says how noise is actually measured.
- **The V-rank arm.**
- **Its stage-2 and stage-3 commitments.** CQL/IQL and six rounds of expert iteration are not planned. Sections 7–9 state the conditions under which they would be.

**What the evidence says so far:**

| Source | Result |
|---|---|
| T0003, fixed P, 2,240 games | V on against V off: +0.010, +0.018, +0.012, −0.002. All four not shown. |
| T0022, fresh maps, 3,500 games | V3-XL against V off: +0.0028, not shown. V3-XL against the old V2-M: −0.0229 [−0.0411, −0.0047]. |
| T0019 continuation regret | "Incumbent regret against the best sampled candidate": 0.114 (v12), 0.079 (C08). The best of noisy means is biased upward, so this is not available improvement. |

## 2. The code (T0034)

| File | What it does |
|---|---|
| [fusion/decision_features.py](../fusion/decision_features.py) | Canonical complete orders, support flags and duplicate detection. Root and candidate scorer inputs. The feature identity, and the model kind `own_action_scorer`. |
| [training/decision_dataset.py](../training/decision_dataset.py) | The versioned root/candidate dataset `ctf-fusion-decision-1`. It builds from branch roots plus their source matches, with linkage checks, map splits and authority receipts. |
| [training/decision_noise.py](../training/decision_noise.py) | Coverage, noise, replay and power diagnostics. No automatic verdicts. |
| [training/train_decision_ranker.py](../training/train_decision_ranker.py) | A linear baseline and a small MLP Q_θ(history, candidate). It trains on incumbent-relative gains or within-root preferences. |
| [training/evaluate_decisions.py](../training/evaluate_decisions.py) | `select` freezes λ, the normalization and every identity on validation. `test` evaluates exactly that rule, once. Also: the headroom and oracle diagnostics. |
| `training/export_weights.py`, `training/export_mixed.py` | Refuse an `own_action_scorer` run as opponent P or V. |
| [tests/test_decision_learning.py](../tests/test_decision_learning.py) | Linkage, history, leakage, support, duplicates, split contamination, model kind, repeat arithmetic, weights, clustering, ties and frozen identities. |

## 3. Data contract

### What a root and a candidate carry

**A root** is one decision of the designated carrier seat: match, map, team count, seat and turn, its split, and the carrier's binary hash. Its extractor record comes from the **recorded prefix**: INIT plus TURN 1..t of our seat, exactly what the bot had read when it answered turn t.
- A root is never re-encoded as the first turn of a one-turn match.
- No block after t is read for features.
- A test checks that the prefix row equals the same row of a whole-game extraction, so the extractor is causal.
- Another test removes every later block and gets the identical record.

**A candidate** carries:
- its requested order, canonical key and source;
- whether it is on-policy;
- its canonical move, action, target unit and target slot per unit, and its flags;
- its duplicate pointer and support bit;
- its raw final score vector for every repeat, the derived outcome, the repeat count, faults per repeat, mean and sample variance;
- the carrier's own score where a hook logged it.

Missing outcomes stay missing (n = 0, NaN). Dropped roots are listed with their reasons.

### Canonical orders and support

The canonical order follows `fusion.labels`, the same convention P's labels use. Every change from the requested order is kept as a flag:

| Request | Becomes | Flag |
|---|---|---|
| An illegal move | the stay it settles as | `requested_move_illegal` |
| An action the root forbids | a wait | `action_canonicalized` |
| An attack on a target in the eight slots | that slot's features | — |
| An attack on a target hittable only beyond the full slots | **unavailable**: no zero-feature success, no probability clipped to ε | `target_out_of_cap` |
| An attack on an unhittable target | a wait | `target_invalid` |

**Support** means: parsed, every target representable, and the first of its effective duplicates, with the incumbent always first. A root whose own incumbent is outside the support keeps the incumbent and offers no gains. Such roots are counted.

### Linkage checks

The builder stops with the root's file name on any of these failures:
- the match id, team count, map, INIT, designated seat or carrier hash does not match the source match;
- a branch's root block is not the recorded TURN t block;
- candidate 0 is not the recorded reply, or it is not the only on-policy candidate;
- the first stored source is not "recorded";
- a requested order appears twice;
- a stored outcome differs from the pairwise utility of its stored score vector;
- one dataset mixes carrier binaries.

### Authority

`--authoritative` refuses unless the audit receipt records a passed strict source alignment.

A manifest receipt that verified only selected frozen identities is accepted only with `--accept-selected-scope`, and that acceptance is recorded. A freeze-time manifest baseline is never reconstructed from current bytes.

Without `--authoritative`, the dataset is **exploratory**. The trainer then requires `--exploratory`, and every report repeats the reasons.

T0010's data is exploratory today. `DATA_AUDIT.json` has `source_row_alignment_verified: false` (`pending_host_reaudit`), and its manifest receipt scope is `selected_frozen_identities_only`.

### Noise and coverage (`training.decision_noise`)

- **Within-action repeat variance** is the pooled sample variance of repeats of the *same* candidate at the same root. It exists only where a candidate has two or more repeats. One repeat cannot establish it, and the report says so instead of estimating.
- **Between-action variance** is the variance of candidate means inside a root. It is reported raw and, when the within-action share is known, net of it. The spread of effects across *different* roots is never used as continuation noise.
- **The replay diagnostic** compares the original game's outcome with the incumbent's replayed continuations. It measures bot memory and restart effects, not repeat noise.
- **Coverage:**
  - effective map counts and per-count coverage;
  - candidates per root (actual K) and sources;
  - support, label availability and changeable roots;
  - canonicalization flags, effective duplicates and out-of-cap targets;
  - feature collisions (distinct canonical orders with identical scorer inputs).
- **Power** is a planning approximation from train/val **map-level** variance: MDE(M, f) = 2.8 · √(f · s²_map / M) for M maps and switch share f. Rows that are not independent are never treated as if they were.

## 4. The scorer and the rule

**Target** (exploratory): A^π0(s, a) = E[U_T | s, a, then π0] − E[U_T | s, a0, then π0].
- U_T is our seat's final pairwise utility.
- a0 is the incumbent.
- π0 is the carrier that played the continuations.

The scorer predicts g(a) = Q_θ(s, a) − Q_θ(s, a0). So g(a0) = 0 exactly, and only differences between complete orders are learned.

**Inputs.**
- The root's own units and v3 value features.
- Per unit of the candidate: move, action, the move's features, the p3 combat block, and the chosen target slot's features.
- Never an outcome, a later block, an opponent's turn-t reply or a realized successor.

The opponent P is neither read nor trained.

**Models.** Start simple:
- `linear`: the candidate features only;
- `mlp`: clipped-ReLU hidden layers over state and action features together.

**Losses:**
- `gain`: weighted squared error;
- `pairwise`: logistic loss on within-root preferences whose gap exceeds one pairwise step, and two standard errors when repeats allow.

**Weights.** Roots weigh equally within a team count, and team counts weigh equally. With repeats, an item is weighted by the inverse variance of its observed gain. With one repeat everywhere, the weighting is uniform, and the run config records that.

DPO stays a separately gated follow-up. PPO, CQL and IQL are not implemented.

**The decision rule.** Choose argmax S_norm(c) + λ · g(c) among the incumbent and the supported candidates.
- With no logged carrier scores, S_norm is `incumbent_prior`: S = 1 for the incumbent and 0 otherwise. A switch then happens only when λ · g(c) > 1.
- With logged scores, S_norm is `zscore`.
- Ties go to the incumbent, then to the smallest canonical key.
- λ = 0 is the incumbent.

**Evaluation** (`training.evaluate_decisions`):
1. **`select`** chooses λ per model, and the normalization, on **validation** only. It then writes a selection file. The file holds its own digest, the dataset identity (index and per-split hashes, features, carrier, authority), each model's run, `model.npz` and `config.json` hashes, the predeclared primary comparison, the secondary comparisons, and the negligible margin.
2. **`test`** verifies every one of those identities before any test outcome is read. It then evaluates that same rule, never argmax Q. It compares the incumbent, the linear ranker and the learned scorer.

**The primary quantity.** Δ = the chosen candidate's mean outcome minus the incumbent's, over **all** roots whose incumbent has an outcome.
- Unchanged and uncertain roots count, as Δ = 0 or as observed.
- Team counts weigh equally.
- The interval is 95 % and cluster-robust over independent maps.
- Candidates without outcomes cannot be chosen offline. They are counted, and so are unevaluable roots.

**Verdicts.**

| Verdict | When |
|---|---|
| **benefit** | the interval is above 0 |
| **harm** | the interval is below 0 |
| **negligible** | the whole interval lies inside the predeclared ±margin |
| **insufficient precision** | anything else |

An interval containing zero does not show that learning is impossible. The primary comparison is fixed before test. Secondary comparisons use a Bonferroni interval over their number.

**Diagnostics, labeled as such.**
- **Headroom** with independent repeat subsets: select the best candidate on even repeats, evaluate it on odd repeats. It is unavailable below two repeats. A test shows that a candidate selected on a lucky repeat loses on the other one.
- **The `--oracle-value` oracle** ranks by a V on **realized** successors. Those contain the opponents' actual turn-t replies, so the oracle is unavailable at deployment and is not evidence for any deployable predictor.
- **A deployable V or simulator comparison** must choose using P-predicted opponent actions, through the carrier hook (section 5). That is not implemented here.

## 5. Carrier hook (specified, not implemented)

The hook is a later code task in the family-carrier generator (`bots/p_only/tools/derive/`), off by default.

| Requirement | Detail |
|---|---|
| Off means off | With the experiment off or λ = 0, the carrier skips all extra work: no feature encoding, no scoring, no shortlist logging. Its replies are byte-identical to today's (tested on recorded games, as the V-cut equivalence is). |
| Shortlist | Always includes the legal incumbent. Complete orders are deduplicated by the canonical key of `fusion.decision_features`. The actual K is exposed (K = 1 is allowed and means no choice), along with each candidate's source and carrier score. |
| Counters | Decision coverage on stderr: roots offered, K, switches, deadline fallbacks. |
| Time | Feature encoding, scoring, the remaining search, protocol parsing and output must all fit inside the **full 50 ms** reply limit, with the carrier's existing internal budgets. A nominal extra 3 ms is not free. The real cost is measured on the host under representative concurrency before any screen. On a deadline, the carrier's own choice stands. |
| Caches | Reuse immutable P features and views, candidate buffers and scorer caches only where the complete history, the state and the model identity permit. No cached result crosses branches, roots or model versions. |
| Identity | The hook binds the scorer's feature identity and run hashes, and the carrier configuration of its training data: carrier, P family, P models and identity (the original P3, or a distilled P3 admitted from T0032), temperatures, search settings and budget. A mismatch refuses to build. |

## 6. Bounded host pilot (proposed, not run)

**Preconditions.**
- The deployment carrier is frozen: P3 or an admitted distilled P3, with its search settings and budget.
- The hook exists and has passed its byte-identity and timing checks.
- The T0010 strict re-audit has run, if T0010's data is to be cited at all. Without it, T0010 stays exploratory pretraining data.

**Design.**
- **Maps:** about 300–500 independent maps, on a fresh training-only seed. The pilot task records the seed in the ledger when it starts; this plan reserves none.
- **Roots:** one representative root per map.
- **Candidates:** the incumbent plus three legal shortlist candidates from the hook.
- **Continuations:** initially four per candidate.
- **Splits:** separate map splits for learning and testing. Headroom uses independent repeat subsets.
- **Scaling:** repeats and any expansion are sized from the measured within-action and between-action variance, and the measured cost, after the first batch. Never from forecasts.

**Planning cost.** Rates from T0010 and T0022: v12 continuations about 0.31 s wall each at 6 workers; a C08 game about 5 times a v12 game.
- 500 roots × 4 candidates × 4 continuations = 8,000 continuations.
- About 0.7 h on v12, and very roughly 3–4 h on C08.
- These are not evidence.

**Deployment claim.** Only an equal-full-turn-time screen on fresh maps establishes a deployed improvement. It needs:
- paired fixtures and an A/A control;
- every fault and timeout kept;
- the actual valid denominators;
- the unchanged production control.

## 7. Macro decisions (later experiment)

**Team tasks** over the carrier's existing task assignment:
- delivery and escort;
- interception and defence;
- flag selection;
- disengagement.

Each action must be an **executable complete team plan** with explicit replanning and termination conditions, not an informal tactical label. Micro legality, damage allocation and movement stay with the existing mathematical layer.

**First, measure whether the incumbent's macro candidate set contains better plans at all.** Do that with the same headroom design: independent repeat subsets, map clusters. Only then train another model.

## 8. Expert iteration (conditional)

Worth starting only with one of these:
- a **stronger offline expert**, such as a larger-budget search;
- a demonstrated ability of a student to **save useful online search work**.

**Compare** imitation of the original search against imitation of a larger-budget expert.

**Data use.**
- Fresh on-policy data is required to validate an updated deployment policy.
- Correctly labeled old-policy data may serve exploratory pretraining or replay.

A failed DPO arm does not rule out stronger-expert distillation.

## 9. Multi-step offline RL (conditional)

**Required first: complete transitions.**
- the root and its history;
- the actual legal shortlist;
- the selected action;
- the actual behaviour probability;
- the next state and history;
- rewards and termination.

**Exploration** among K distinct candidates uses the actual K and handles K = 1.

**The objective** stays the final pairwise utility. If r_t = U_{t+1} − U_t, the episode must be complete and undiscounted, so that the rewards sum to U_T − U_0. No capture, kill or distance bonus enters the primary objective.

**Opponents** can have adaptive internal memory even when their binaries are frozen.

## 10. Proposed host commands (not run)

On T0010's data (exploratory). These commands only check the plumbing at scale. T0010 holds p2 continuations, so this is not a P3 decision-policy gate.

```bash
T=tools/repo/task.py
# The carrier configuration of T0010 (write it from bots/p_only/report/v_data/manifest.json and the profile; do not invent fields)
#   build/decision/t0010_carrier.json: {"carrier": "v12_ponly_xxl", "binary_sha256": "6cbd5d34...", "p": {"family": "p2", "models": "inference/models_xxl"}, "v": "off", ...}
python $T lock cpu --task TXXXX -- python -m training.decision_dataset \
    --branches data/branches/v12_ponly_xxl --matches data/p_only/v_data/matches \
    --audit bots/p_only/report/v_data/DATA_AUDIT.json \
    --manifest-receipt bots/p_only/report/v_data/MANIFEST_METADATA_ADAPTER.json \
    --carrier-config build/decision/t0010_carrier.json --successors --out data/decision/t0010_p2
python -m training.decision_noise --data data/decision/t0010_p2 --out reports/models/decision/t0010_noise
python -m training.train_decision_ranker --data data/decision/t0010_p2 --model linear --exploratory --seed 1 \
    --out runs/decision_linear_s1
python -m training.train_decision_ranker --data data/decision/t0010_p2 --model mlp --hidden 64,16 --exploratory --seed 1 \
    --out runs/decision_mlp_s1
python -m training.evaluate_decisions select --data data/decision/t0010_p2 --run linear=runs/decision_linear_s1 \
    --run mlp=runs/decision_mlp_s1 --primary mlp --negligible 0.002 --out reports/models/decision/SELECTION.json
# commit SELECTION.json before this step; it verifies every identity, then reads test once
python -m training.evaluate_decisions test --data data/decision/t0010_p2 \
    --selection reports/models/decision/SELECTION.json --out reports/models/decision/t0010_test
```

**For the pilot:** the same commands on the pilot's dataset, with `--authoritative` and `--carrier-config` of the frozen deployment carrier, after its collection. The negligible margin and the primary model are set in the pilot task **before** its data exists.

## 11. Repository facts that constrain this plan

1. **The stage 2 report's "CONFIRMED" causes are hypotheses.** No data swap and no feature ablation was run. v3 already has the per-carrier hits, volley and escort features that report calls missing, and V3-XL with them was not shown (T0022).
2. **T0010's continuations are p2** (`v12_ponly_xxl`). T0011 made p3 the finalist. Anything deployed on a P3 carrier needs data from that carrier.
   - **Refresh rule, proposed for FEATURE_FAMILIES.md:** "A carrier is a new policy whenever its P, V, actor, search settings or budgets change. Data labelled by one carrier version never trains or evaluates a component deployed in another without being collected again."
   - Old-policy data may still pretrain, labeled as such.
3. **T0010 used one repeat**, so its within-action noise is unknown. Its "resolved" pairs are single-continuation differences.
4. **Branch files keep only the root and t + 1 blocks.** They are not trajectories, so multi-step offline RL needs the complete transitions of section 9.
5. **T0010's candidates are hashed one-unit deviations, not the carrier's shortlist.** The deployment candidate set comes from the hook.
6. **Labels are not deployment conditions.** Branch labels follow the opponents' **recorded** turn-t replies; deployment sees P's predictions. True-successor rankings, including `evaluate_value`'s, are therefore diagnostics, not deployment gates.
7. **Seeds.** The ledger in [DIGEST.md](DIGEST.md#seeds) lists every consumed seed, including 20261009, 20261019, 20261020, 20261029, 20261031, 20261101, 20261102, 20261107 (reserved), 20261108, 20261201 (training only), 20261210, 20261211, 20261220, 20261230, 20261240 and 20261301–20261364. T0028's seeds were never recorded. Never reuse a measurement seed.
