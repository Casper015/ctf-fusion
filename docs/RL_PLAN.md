# RL plan: P predicts opponents, a learned component moves us

Task T0033, written on 2026-10-11 against main at bba5dcb. This is a plan. Nothing in it is implemented, trained or played, and it spends no seed. The follow-on tasks it proposes are listed in [section 7](#7-proposed-tasks); none has been created yet.

**The question.** Supervised V has not shown a benefit in games:
- **T0003** (fixed P, 2,240 games): +0.010, +0.018, +0.012 and −0.002 across the four carrier and P-size combinations. All four are *not shown*.
- **T0022** (fresh maps, 3,500 games): V3-XL against V off is +0.0028, *not shown*. V3-XL against the old V2-M is −0.0229 [−0.0411, −0.0047], a deployed-bot *harm*.

Can a component trained on decision outcomes, rather than on state-value regression, move our own seat better than the carriers' hand-written choice? The plan answers in three stages, from cheapest to most expensive. Each stage has a numeric kill rule.

| Stage | What learns | Data | New host games | Cost to its kill decision |
|---|---|---|---|---|
| 1 | A preference scorer over our own candidate orders (two arms: V-rank and policy DPO) | T0010's 13,702 branch roots, already collected | none for the offline gate | about 1 GPU-hour plus a small code task |
| 1-R / 1-S | The same scorer, re-collected under the deployed P3 carrier, then screened | new: 7,000 games and 112k continuations | about 12 h (v12) or about 55 h (C08) | about 1 host-day |
| 2 | An offline-RL actor over the carrier's shortlist (CQL/IQL) | ε-shortlist games with logged propensities, plus horizon branches | about 4–14 h | about 1–2 host-days |
| 3 | Expert iteration: search as expert, actor as apprentice, V as critic | re-collected every iteration | about 4 h (v12) or about 14 h (C08) per iteration | 6 iterations at most, plus T0006 |

Read [section 6](#6-repository-facts-that-contradict-or-constrain-the-request) first. Several premises of the request do not match the repository, and two of the mismatches change what stage 1 can be.

## 1. Facts this plan is built on

**Measured host wall times** (DESKTOP-G60OJUD; official 2,000/50 ms limits):

| Run | Work | Wall | Workers | Per item |
|---|---|---:|---:|---|
| T0010 collection (v12_ponly_xxl, p2, V off) | 7,000 games | 2,249.8 s | 6 | 0.32 s wall, 1.93 worker-s per game |
| T0010 branches | 54,808 continuations from 13,702 roots | 16,879.8 s | 6 | 0.31 s wall, 1.85 worker-s per continuation (prefix replay included) |
| T0010 ranking dataset | 13,702 roots | 255.3 s | 1 | |
| T0022 screen (C08 family carrier) | 3,500 games | 8,536.0 s | 4 | 2.44 s wall, 9.76 worker-s per game |
| T0011 P XXL training (RTX 4080 Laptop) | 24.15M rows per epoch | 185–202 s per epoch | GPU | |

A C08 game costs about 5 times a v12 game (9.76 against 1.93 worker-s). No C08 branch continuations were ever timed. Every C08 continuation cost below is that same factor of 5 applied to the v12 figure: **an estimate, not a measurement**.

**T0010's ranking data** ([bots/p_only/report/v_data/](../bots/p_only/report/v_data/BRANCH_STATUS.md), [DATA_AUDIT.json](../bots/p_only/report/v_data/DATA_AUDIT.json)):
- 7,000 games of `v12_ponly_xxl` (P family p2, V off) on training seed 20261201.
- 14,000 roots, of which 298 were dropped (fewer than two legal candidates). Each of the remaining 13,702 roots has 4 candidates and **1 repeat**.
- Candidates: the recorded reply plus three hashed one-unit move deviations (`--candidate-mode single`).
- 82,212 pairs: 29,629 resolved and 52,583 uncertain. Resolved means the outcome difference exceeds one pairwise step; with one repeat, no noise test applies.
- Splits are by map:

| Split | Roots | Roots with ≥ 1 resolved pair | Maps |
|---|---:|---:|---:|
| train | 10,992 | 6,018 | 5,612 |
| val | 1,356 | 727 | 692 |
| test | 1,354 | 740 | 688 |

**Continuation noise** ([T0019 regret](../bots/p_only/report/search_portfolio/RESULTS.md)): C08 roots where two arms chose different orders, with 2 repeats per candidate, gave a difference of +0.0093 [−0.0202, +0.0389] over 47 roots. That implies a per-root standard deviation of the 2-repeat mean difference of about **0.10**, so about **0.15** with one repeat. This is the only committed estimate, and it comes from a different carrier. Stage 1 recomputes it on its own data before anything is trained (check K1 below).

**How V enters the carriers today.** V adds `coef × (p − 0.5)` to a heuristic score:
- **v12:** at the end of its two-turn rollout, `PONLY_V_ROLLOUT` 40;
- **C08:** at the refine end (`PONLY_V_REFINE` 50) and on beam successors (`PONLY_V_BEAM` 35).

Our own moves come from hand-written task assignment and candidate generation. v12's rollout moves our seat by script, not by P.

## 2. Stage 1: preference optimization on the branch pairs

### 2.1 What is learned, and why two arms

A branch root gives, for one state s, several of our orders c with sampled outcomes y_c. These are common-prefix counterfactuals: the opponents' recorded replies at turn t, then everything live. Stage 1 learns a score q(s, c) whose argmax over a root's candidates beats the carrier's own choice. Two parametrizations use the same pairs and the same gate.

**Arm 1A: V-rank (existing machinery, no new trainer code).**
- **Model:** `training.train_value --rank --rank-weight w`. The pairwise logistic loss acts on V's outcome logits of each candidate's t + 1 successor (v3, 108 inputs): P(a ≻ b) = σ(z(s_a) − z(s_b)).
- **What it is:** a critic ranking. It is a Bradley–Terry reward model with a uniform reference, not DPO.
- **What it is not:** "DPO on the existing machinery". Section 6, item 5 explains why that phrase does not hold.
- **Warm start:** `--init` from T0012's best arm. The outcome loss stays in the objective as an anchor.

**Arm 1B: policy DPO (small new trainer).** An own-seat policy π_θ over our team's complete order:

- **Inputs:** the p3 policy inputs of our own seat at the root, read with the same encoder P uses for every team.
- **Initialization:** θ starts from the frozen P3 XXL seed-1 run of T0011.
- **Reference policy:** π_ref is that frozen P3 run, evaluated on our seat.
- **Loss** over resolved pairs (w preferred to l), each root weighted 1 as in `branch_dataset`:

  L = −Σ w_root · log σ(β · [(log π_θ(w|s) − log π_ref(w|s)) − (log π_θ(l|s) − log π_ref(l|s))])

  where log π(order) is the joint move log-probability, plus each unit's action log-probability, plus the target-slot log-probability for attacks.
- **Score used to choose:** the implicit reward q(s, c) = β · (log π_θ(c|s) − log π_ref(c|s)).
- **Why this arm exists:** this is the DPO the request asks for.
- **Why it may see what 1A cannot:** the p3 combat block is per (unit, destination). It sees one-unit deviations directly, through kill-in-one, volley, exposure and lethal-reach features. A one-unit deviation changes the v3 successor aggregates only when it changes a count, and when it does not, arm 1A cannot tell the candidates apart (check K1b).
- **Caveat on π_ref:** P3 was trained on roster opponents' moves (designated carrier rows are refused as P labels). So π_ref is "what a roster bot would do here", not the carrier's policy. That is acceptable for a log-ratio reranker over the carrier's own shortlist. A carrier-imitation reference (behaviour cloning on T0010's 7,000 designated games) is the alternative. Train it only if 1B passes K2 and its val Δ is near zero: a reference far from the behaviour policy weakens the KL anchor.

**Guard.** A 1B model is an own-seat preference model. It must never load as an opponent P. Its run `config.json` carries `kind: "own_pref"`. `export_weights`, `export_mixed` and the registries must refuse that kind, and a test checks this. This mirrors the existing refusal of designated shards in `train_policy`.

### 2.2 Checks before any training (no games, host CPU only)

| Check | Data | Statistic | Rule |
|---|---|---|---|
| **K0** integrity | T0010 retained artifacts | `DATA_AUDIT.json`: `source_row_alignment_verified` | It is `false` ("pending_host_reaudit") today. Stage 1 does not start until the host re-audit sets it `true`. |
| **K1** label noise | all 13,702 roots | Two resolution rates at the same tie margin: r_AA = the share of roots where candidate 0's continuation and the recorded `root_outcome` (same order at turn t, an A/A pair up to the carrier-memory caveat) differ by more than one pairwise step; r_cross = 29,629 / 82,212 = **0.360**. Also r_AA from T0019's two repeats per candidate (`data/branches/search_v12`, `search_c08`). | **Kill stage 1 on this data if r_AA ≥ 0.8 · r_cross (0.288).** Most "resolved" labels would then be noise. Re-collect with `--repeats 3` before training. |
| **K1b** feature identity | train resolved pairs | Share of pairs whose two successors have identical v3 value columns | **Over 50 %: drop arm 1A** (it cannot learn them; the tie rule in `rank_metrics` falls back to the on-policy candidate). Report v2 too. |
| **K1c** power | test split, no model | σ_d = SD of (y_c − y_on) over (root, c ≠ on); MDE = 2.8 · √f · σ_d / √n_roots, with f = 0.25 (share of roots a model changes) | Recorded before training. **If MDE > 0.010, the T0010 test split is declared underpowered**, and stage 1's gate moves to the 1-R data (section 2.5), sized for MDE ≤ 0.004. With σ_d ≈ 0.15 and n = 1,354, the expected MDE is about 0.0056. |

### 2.3 Training and selection (val only)

**Arm 1A.** Four rank weights, each warm-started from T0012's selected arm. Everything else is identical to T0012's step 3.

**Arm 1B.** β ∈ {0.1, 0.3, 1.0} in one run directory. 10 epochs, Adam 1e-4, batch 256 pairs. A KL(π_ref ‖ π_θ) anchor on the root's legal orders, weight κ = 0.1.

**Selection.** For each arm, pick the setting with the largest val Δ: the mean of y(argmax q) − y(on-policy) over all val roots. Break ties toward the smaller weight or β. Write the choice to `reports/models/pref/SELECTION.json`, with its hash, **before** the test split is read.

**K2 learnability (val).** An arm whose pair accuracy on val resolved pairs has a 95 % map-clustered interval against 0.5 that includes zero (`pair_vs_random`) is killed. It learned nothing that transfers across maps.

### 2.4 The offline gate (test split, read once)

**Statistic.** Δ = y(argmax_c q(s, c)) − y(on-policy). It is computed over **all** test roots, including roots without resolved pairs and roots where the model keeps the on-policy order (Δ = 0 there). Team counts get equal weight, and the 95 % interval is clustered by map.

**Why this statistic.** It is unbiased with one repeat: the model's choice is fixed before test outcomes are seen. It needs no pair labels. It is `−regret_vs_incumbent` in `training.evaluate_value`, plus count weighting. The "regret against the best sampled candidate" that T0019 reports is a different quantity. It selects the maximum of noisy means and is biased upward (winner's curse), so it is never used as a gate.

| Result | Verdict | Next |
|---|---|---|
| lower bound > 0 | **pass** | 1-R and 1-S |
| upper bound < 0 | **harm: kill** | stop the RL line; write down why before proposing anything else |
| interval includes 0 | **not shown: kill on this data** | 1-R only if K1c declared the test underpowered *before training*; otherwise stop |

Also reported, never gating: Δ restricted to changed roots, the share of changed roots, per-count and per-band Δ, and pair accuracy.

### 2.5 After a pass: 1-R (refresh) and 1-S (screen)

**Why re-collect.** T0010's games were played by the **p2** v12 carrier. T0011 made p3 the finalist (`FINALIST_DECISION.json`: `p3_finalist: true`). Labels under the p2 continuation policy are exactly the mismatch the stage 2 report blames for V's failure. Any game screen of a scorer therefore requires data from the carrier it will run in. Section 6, item 2 proposes the Refresh rule text.

**1-R collection** (training only, seed **20261401**):
- **Carrier:** the deployment carrier as T0013 freezes it: P3 XXL, V off, fixed time. This plan recommends v12_p3, because its continuations cost about a fifth of C08's.
- **Games:** 7,000, 500 maps per count, keep_matches.
- **Branches:** 2 roots per match, 4 candidates, **2 repeats**, candidates from the carrier's own shortlist (section 2.6) plus one-unit deviations.
- **Size:** 112k continuations.
- **Cost (v12):** collection about 0.6 h, branches about 9.6 h, ranking about 0.1 h. For C08, about 3.2 h + about 48 h (estimate).
- **Stage 1 then re-runs on this data:** 2.2 through 2.4, test MDE ≈ 0.0039.

**1-S screen** (host, CPU lock):

| | |
|---|---|
| Arms | carrier off, carrier + 1A, carrier + 1B (only the arms that passed), and an A/A copy of off |
| λ tuning | λ (section 2.6) tuned on a 280-fixture pre-screen at seed **20261411**, then frozen |
| Measurement | 1,400 fixtures (100 maps × 14 counts) at seed **20261410** |
| Size | 4 × 1,400 = 5,600 games, plus 1,120 for tuning |
| Cost | about 0.9 h for v12 at 4 workers, about 4.6 h for C08 |
| **Pass** | the primary equal-count-weight paired interval is wholly above 0 **and** the A/A interval includes 0. The arm goes to T0006 as a finalist. |
| **Kill** | otherwise. A wholly positive single band is not a pass. |

### 2.6 Deployment hook (bots code task, off by default)

The family carriers get one compile-time interface, `PONLY_PREF`, default 0.

**When 1:**
- The carrier exposes its final root shortlist: C08's top K beam candidates by `c.value`, v12's top K root alternatives by rollout score, with K = 4.
- It scores each with q, and plays argmax_c [S(c) + λ · (q(c) − mean q)]. S is the carrier's own score, normalized per root.
- The actor never leaves the carrier's support: a behaviour constraint in the BCQ sense.
- **Deadline fallback:** past `PONLY_PREF_DEADLINE_MS` (default 3 ms of scoring), the carrier's own choice stands.
- **1A** scores the shortlisted successors. These are built with P's predicted opponent replies (in training, opponents' replies were recorded). **1B** needs one own-seat p3 forward pass at the root.

**Tests:**
- With `PONLY_PREF=0`, and with λ = 0, replies are byte-identical to the current carriers on recorded games. This is the pattern of the existing V-cut equivalence test.
- Root-history and deadline contracts are kept.
- In two-team games, C08 bypasses this path, so the hook's coverage must be counted and reported.

The same interface carries `PONLY_ROOT_LOG=1`, which writes the shortlist and its scores to stderr, used for collection only. Stages 2 and 3 need it.

### 2.7 Commands (stage 1, offline)

These run on DESKTOP-G60OJUD after K0. Arm 1A also needs T0012 merged. Each `TXXXX` is the stage-1 task id.

```bash
T=tools/repo/task.py
# K1, K1b, K1c: noise, feature identity, power (no games)
python $T lock cpu --task TXXXX -- python -m training.branch_noise \
    --branches data/branches/v12_ponly_xxl --rank data/rank/v12_ponly_xxl \
    --repeat-branches data/branches/search_v12 data/branches/search_c08 \
    --families v2 v3 --mde-share 0.25 --out reports/models/pref/noise
# 1B data: own-seat root records and candidate labels (layout 3)
python -m training.branch_dataset --branches data/branches/v12_ponly_xxl \
    --out data/pref/v12_ponly_xxl --root-records
# 1A: rank-weight sweep, warm-started from T0012's chosen arm (<arm>, <hidden>, <B> from its report)
for w in 0.5 1 2 4; do
  python $T lock gpu --task TXXXX -- python -m training.train_value --model general \
      --cache data/vcache/v12xxl_v3 --value-family v3 --hidden <hidden> --max-width 8192 \
      --backend torch --device cuda --epochs 40 --patience 3 --plateau-decay 2 --lr 3e-4 --min-lr 1e-5 \
      --batch <B> --seed 1 --checkpoint --init runs/v_pilot_<arm>_s1 \
      --rank data/rank/v12_ponly_xxl --rank-weight $w --out runs/pref1a_w${w}_s1
done
# 1B: DPO against the frozen P3 reference
python $T lock gpu --task TXXXX -- python -m training.dpo_policy --pairs data/pref/v12_ponly_xxl \
    --reference runs/pf_p3_xxl_s1 --beta 0.1 0.3 1.0 --anchor-kl 0.1 --epochs 10 --lr 1e-4 \
    --batch 256 --backend torch --device cuda --seed 1 --out runs/pref1b_s1
# Selection on val, then the gate on test (refuses to run without a committed SELECTION.json)
python -m training.pref_gate --rank data/rank/v12_ponly_xxl --pref data/pref/v12_ponly_xxl --split val \
    --run 1A_w0.5=runs/pref1a_w0.5_s1 --run 1A_w1=runs/pref1a_w1_s1 --run 1A_w2=runs/pref1a_w2_s1 \
    --run 1A_w4=runs/pref1a_w4_s1 --run 1B=runs/pref1b_s1 --select --out reports/models/pref/val
python -m training.pref_gate --rank data/rank/v12_ponly_xxl --pref data/pref/v12_ponly_xxl --split test \
    --selection reports/models/pref/SELECTION.json --out reports/models/pref/test
```

**Files written.**

| Kind | Paths |
|---|---|
| Code (stage-1 code task, PR) | `training/branch_noise.py`, `training/pref_gate.py`, `training/dpo_policy.py`, `training/branch_dataset.py` (`--root-records`), the `own_pref` refusal in `training/export_weights.py` and `training/export_mixed.py`, `tests/test_pref.py` |
| Never committed | `data/pref/`, `runs/pref1a_*`, `runs/pref1b_*` |
| Committed (reports PR) | `reports/models/pref/{noise,val,test}.{md,json}`, `SELECTION.json`, and each run's `config.json` and `log.jsonl` under `reports/models/pref/runs/` |

**What the new code has to do.**
- **`branch_noise`:** reads the root JSONs (`root_outcome`, per-candidate outcomes) and `rank.npz`. It computes K1, K1b and K1c, per split, per team count and in total.
- **`pref_gate`:** scores candidates with either kind of run (a V run: successor logits; an `own_pref` run: implicit reward). It reports pair accuracy against 0.5 and against the majority label, and Δ with equal count weights, map-clustered. With `--select`, it writes `SELECTION.json`; on test, it requires that file and checks its hash.
- **`dpo_policy`:** builds on the torch policy forward and batching of `fusion/torch_backend.py` and `training/distill_policy.py` (T0031).
- **`--root-records`:** produces each candidate's labels with `fusion.labels.label_match` on a one-turn view of the match, with that candidate's reply substituted.

## 3. Stage 2: offline RL over the carrier's shortlist (sketch, costed)

**What it adds over stage 1.** Stage 1 is one policy-improvement step: it chooses better at one turn, assuming the carrier plays every later turn. An actor deployed at every turn changes the states it later meets. Stage 2 learns a Q over the shortlist from multi-step data, so that compounding is in the data.

**Data.** Branch files cannot provide it: they keep only the root block and the t + 1 block (section 6, item 6). Two collections on the refreshed carrier, both training-only:

1. **ε-shortlist games** (seed **20261402**, 7,000 games):
   - With probability ε = 0.05 per turn, the carrier plays a uniformly drawn member of its top-4 shortlist (other than its choice) instead of its choice.
   - The draw is a hash of (match, turn), so it is reproducible.
   - The shortlist, the scores and the propensity μ(c|s) = (1 − ε) · 1[c = choice] + ε/3 · 1[c ≠ choice] are logged through `PONLY_ROOT_LOG`.
   - Cost: v12 about 0.6 h, C08 about 3.2 h.
2. **Horizon branches:**
   - `play_branch` gains `--keep-blocks 1,10,20,40`, so a branch stores its state at t + H.
   - That allows n-step targets y ≈ V(s_{t+H}) with bootstrap, next to the Monte-Carlo outcome.
   - Cost: 1-R's branch run with 4 extra blocks per continuation, no extra games.

**Algorithm.** Discrete-action CQL or IQL over the K = 4 shortlist:
- **Q head:** Q_φ(s, c) reads the 1B own-seat p3 features of c and the v3 successor of c.
- **Penalty:** CQL's α · (logsumexp_c Q(s, c) − Q(s, c_behaviour)) keeps Q pessimistic off-support.
- **Actor:** advantage-weighted, π(c|s) ∝ μ(c|s) · exp(A(s, c)/β). A uses V(s) from the stage 3 critic definition as its baseline.
- **Decision transformer: not proposed.** Return-conditioned imitation in a stochastic environment conditions on luck, not on choices. With a single behaviour policy, it learns state-return correlations, not improvements.

**Kill rule.** Same Δ statistic, on a fresh branch test set of the refreshed carrier, paired on the same roots. The stage-2 actor is compared with the stage-1 actor, not only with the carrier.
- Continue to a screen only if the interval is wholly above 0.
- The screen is the 1-S design (stage-2 actor against stage-1 actor, plus A/A) on a new seed, 20261412.
- Stop otherwise; the stage-1 actor stays.

**Cost to the kill decision:**
- v12: collection 0.6 h + horizon branches about 2 h for 2,000 test roots + training < 1 GPU-hour + screen about 0.9 h. **About 4–5 host hours.**
- C08: **about 14 host hours** (estimate).

## 4. Stage 3: expert iteration with V as critic (sketch, costed)

### 4.1 The loop

Iteration k, with carrier_k = frozen P3 carrier + actor_k (+ V_k only where its own gate passed):

1. **Collect.** 3,500 games (250 maps per count) at training seed 2026142k (k = 1..6), with `PONLY_ROOT_LOG=1`. The search's root scores over its shortlist are the expert targets.
2. **Train the actor** (apprentice): cross-entropy to softmax(S/T) over each logged shortlist, plus stage-1 DPO pairs from 2,000 branch roots of these games.
3. **Train the critic.** V_{k+1} = E[u | s, carrier_k], trained on carrier_k's own games. The Refresh rule is satisfied by construction: V never sees another policy's continuations.
4. **Offline gate.** Δ_k = actor_{k+1} against carrier_k's choice, on 1,000 fresh branch roots (4 candidates × 2 repeats).
5. **Screen.** carrier_{k+1} − carrier_k plus A/A, 700 fixtures, seed 2026143k.

**Expert iteration needs the search to consume the actor.** Without that, the loop is stage 1 repeated. The actor must be:
- C08's candidate ordering prior for our own orders;
- v12's own-seat move policy in its two-turn rollout. Today v12 moves our seat there by script; see the v12_ponly.hpp header.

That is a carrier code change behind its own off-by-default switch, `PONLY_ACTOR`.

**Stop** at the first of:
- two consecutive iterations whose Δ_k is not shown;
- any screen interval wholly below zero;
- k = 6.

The surviving carrier goes to T0006.

### 4.2 Fifteen-team general-sum play

The opponents are the frozen reference roster plus its P model. They do not learn. Our bot holds one designated seat, as in every collection so far. So this is single-agent RL in a stationary environment, not multi-agent self-play. Equilibrium questions arise only if copies of the learner fill other seats, and the plan never does that.

The objective is our seat's pairwise utility only: the share of rivals beaten, ties ½. It never includes other seats' utility, so kingmaking and collusion terms do not exist in it.

Two risks remain:
- **Roster overfitting.** The check is T0025's two reserved opponent lineages, as an out-of-distribution read after the final iteration. T0028-style Plackett–Luce placement is descriptive, never a gate.
- **Diluted credit in large games.** In a 15-team game, one decision moves the utility in steps of 1/28. Branch rows keep the full final `score` vector, so per-rival outcomes (beat rival j or not) are recoverable from existing data. Use them as per-rival auxiliary targets for the critic. The objective stays the average.

### 4.3 Credit over 400 turns

1. **Common-prefix counterfactuals.** These are already the design. A branch fixes the prefix and the opponents' turn-t replies, so a candidate difference is an advantage estimate with common random numbers up to turn t. The variance comes from the remaining turns: σ ≈ 0.15 per single-repeat difference (section 1).
2. **Repeats and fixed-work continuations.** Repeats divide the variance. Fixed-work continuations (`search_diag --fixed-samples`) remove our bots' timing nondeterminism, but they change the continuation policy away from the deployed one. Measure r_AA both ways (K1) before choosing.
3. **The critic as bootstrap.** n-step targets y_c ≈ V(s_{t+H}) at H ∈ {10, 20, 40} use the horizon blocks of stage 2. V is useful here only to the extent that it reduces the variance of within-root differences. That is measured on val as the explained variance of (y_c − y_on) by V(s_{t+H}) differences. V-as-critic is dropped if bootstrapped Δ selection is worse on val than Monte-Carlo Δ selection.
4. **No shaped reward in the objective.** Capture-within-20 and death-within-10 stay auxiliary heads, as in V today.

### 4.4 What V does

**V is the critic. The actor moves us.** V's measured failure is as a leaf term in search: not shown at four combinations and two seeds. Its proposed role is a baseline and a bootstrap for the actor's advantages, retrained on each iteration's own carrier. Neither role is assumed useful. Each has its own offline check (4.3, item 3), and V stays off as a search term unless a screen shows it.

### 4.5 Budget

| Item | v12 carrier | C08 carrier |
|---|---:|---:|
| Collect 3,500 games | 0.3 h | 1.6 h |
| Branches, 3,000 roots × 8 continuations | 2.1 h | about 10 h (estimate) |
| Train actor + critic | < 1 h host GPU, or ≤ 6.8 Colab A100 units | the same |
| Export, build, preflight | about 0.5 h (unmeasured) | about 0.5 h |
| Screen, 700 fixtures × 3 arms | 0.3 h at 4 workers | 1.4 h |
| **Per iteration** | **about 4 h** | **about 14 h** |
| **6 iterations** | **about 24 h** | **about 84 h** |
| T0006 confirmation: 10,000 fixtures × (control + 3 seeds) = 40,000 games | 5.4 h | 27 h |

**Colab.** These nets are V- and P-XXL sized, and the host RTX 4080 trains them in minutes. Measured: a P XXL epoch over 24M rows takes about 190 s, so about 1.4M own-seat rows take about 12 s per epoch. Colab's value is parallelism, not speed. Training on the host while it collects would share the CPU with an evidence run, and AGENTS.md forbids that. So iteration k + 1's training can run on Colab while the host collects iteration k's screen. That costs at most about 6.8 units per iteration, about 41 units for 6 iterations, plus the upload of kept matches (size not measured).

**What "three independent training seeds" means here** is a decision for Casper (section 8). It is cheap if it means three seeds of the final iteration's training on the same collected data. It is three times the whole table if it means three independent iteration chains.

## 5. Seeds

**Never touched for training or tuning.** The request's list is incomplete. The ledger in [DIGEST.md](DIGEST.md#seeds) also has:
- 20261009, 20261019, 20261020, 20261029 (proposed collection), 20261101;
- 20261107 (reserved for T0002, still unplayed);
- 20261210 (T0013) and 20261211 (T0014), both reserved;
- 20261301–20261364 (showcase games);
- T0028's seeds, which were never recorded.

20261201 is training data. Stage 1 uses its map-split **test split** only as an offline gate, which never becomes a game screen.

**Proposed here.** None of these appears anywhere in the repository on 2026-10-11. Each is added to the ledger by the task that first uses it, not before.

| Seed | Use |
|---|---|
| 20261401 | 1-R refresh collection (training only) |
| 20261402 | Stage 2 ε-shortlist collection (training only) |
| 20261410 | 1-S screen |
| 20261411 | 1-S λ tuning pre-screen |
| 20261412 | Stage 2 screen |
| 20261421–20261426 | Stage 3 iteration data (training only) |
| 20261431–20261436 | Stage 3 iteration screens |

The confirmation seed stays unchosen until T0006 freezes finalists.

## 6. Repository facts that contradict or constrain the request

1. **"Confirmed causes" are hypotheses, and v3 already has the missing features.**
   - The stage 2 report labels distribution mismatch and coarse features as CONFIRMED. The report runs no experiment that isolates either: no data-swap and no feature ablation.
   - The features it calls missing are v2 gaps. v3 already has enemy-carrier hits, volley-killable, our shooters now and by reach, escort and delivery distance ([FEATURE_FAMILIES.md §4](FEATURE_FAMILIES.md#4-the-v3-extension)).
   - V3-XL with those features was not shown in T0022. Its training cache, `data/vcache/designated_v3`, predates T0010's frozen-carrier data. So the mismatch hypothesis is still untested, and T0012 is its test.
2. **Stage-1 data is p2.**
   - T0010 played the p2 `v12_ponly_xxl`, and T0011 made p3 the finalist. Every game screen of anything trained on T0010 needs the 1-R re-collection.
   - The Refresh rule ([FEATURE_FAMILIES.md §5](FEATURE_FAMILIES.md#5-v-data-under-a-frozen-policy)) mentions only V. Proposed text: *"A carrier is a new policy whenever its P, V, actor, search settings or budgets change. Data labelled by continuations of one carrier version never trains or evaluates a component deployed in another without being collected again."*
3. **The T0010 rank data is not fully audited.** `DATA_AUDIT.json` has `source_row_alignment_verified: false` and `pending_host_reaudit`. That is check K0.
4. **"Resolved" pairs carry no noise test.**
   - With `--repeats 1`, `--tie-z` does not apply. A pair is resolved by a single-continuation difference larger than one pairwise step.
   - The 29,629 resolved pairs are counted over all three splits; the train split holds 80 % of the roots.
   - K1 measures how much of this is noise.
5. **`--rank-weight` is not DPO, and it is already scheduled.**
   - It is a pairwise logistic loss on V's outcome logits of successor states, with the outcome loss as the stopping criterion. There is no policy and no reference.
   - T0012 step 4 already runs it at weight 0.5.
   - Arm 1A therefore extends T0012 (a sweep, plus selection by Δ instead of log loss) and must run after it, not beside it. Arm 1B is the DPO.
6. **Branch files are not trajectories.** `play_branch` keeps `root_block` and `next_block` only. "Offline RL on full branch trajectories" is impossible on existing data; stage 2 adds `--keep-blocks`. The 7,000 kept matches are full trajectories, but of a single near-deterministic policy, with almost no action coverage beyond the carrier's choices.
7. **The candidate set is not the deployment set.** T0010's candidates are hashed one-unit deviations. A deployed scorer chooses among the carrier's own shortlist. `branch_portfolio.py` (T0019) has richer sources (move2, target, drop, transport, another arm's reply) but no "carrier top-K" source. 1-R adds one through `PONLY_ROOT_LOG`.
8. **The point where V is applied differs from where it was trained.**
   - The branch labels are for the t + 1 successor after the opponents' recorded replies.
   - v12 applies V at turn + 2 after a P-predicted rollout. C08 applies it at the refine end and on beam successors; only the beam successors are at t + 1.
   - In deployment, the opponents' turn-t replies are P's predictions, not the truth. That gap is inherent to "P predicts opponents". The hook in 2.6 confines it to the root.
9. **T0019's regret is biased upward.** Its "incumbent regret against the best candidate" (0.114 v12, 0.079 C08) maximizes over noisy means. It is not the available improvement, which is why stage 1 gates on Δ against the on-policy order instead.
10. **The board is behind the evidence.**
    - T0010 shows `open` and T0011 `active`, though both reports are merged (PR #23, PR #21).
    - The host agent works without board writes, by the user's instruction.
    - Stage 1's dependencies (T0012 waiting on T0010) cannot be read from the board until it is updated.

## 7. Proposed tasks

None of these has been created. Each needs Casper's go-ahead, then `task.py new`.

| # | Task | Needs | Lanes | Land | Depends |
|---|---|---|---|---|---|
| A | Stage-1 code: `branch_noise`, `pref_gate`, `dpo_policy`, `branch_dataset --root-records`, the `own_pref` export refusal, tests | any | `training/`, `tests/` (named files) | pr | — |
| B | Stage-1 offline run on T0010 data: K0–K2, selection, test gate | cuda (host) | `reports/models/pref/` | pr | A, T0012, the T0010 host re-audit |
| C | Carrier root hook: `PONLY_PREF`, `PONLY_ROOT_LOG`, `PONLY_EXPLORE_EPS`, off by default with byte-identity tests | any | `bots/p_only/tools/derive/`, `bots/p_only/include/`, `bots/p_only/tests/`, generated family carriers | direct | T0030 (it regenerates the same family carriers) |
| D | 1-R refresh collection + shortlist branches with 2 repeats, seed 20261401 | host | `bots/p_only/report/pref_data/` | direct | B passed or declared underpowered, C, T0013 frozen carrier |
| E | 1-S screen, seeds 20261411 then 20261410 | host | `bots/p_only/report/pref_screen/` | direct | D, then B re-run on D's data |
| F, G | Stages 2 and 3, written only after E | host, cuda | to be set | to be set | E passed |

## 8. Decisions for Casper

1. **The RL carrier.** This plan recommends v12_p3: continuations cost about a fifth of C08's, and T0010's pipeline is proven on v12. C08 is the alternative if T0013 shows C08-p3 clearly ahead at equal time.
2. **"Three independent training seeds" for expert iteration** (T0006): three seeds of the final iteration on shared data, or three independent chains (three times the cost).
3. **Refresh rule text** (section 6, item 2): adopt it in FEATURE_FAMILIES.md through a docs task.
4. **Board state:** whether the host agent should record T0010 and T0011 as done, so dependent tasks show as ready.
