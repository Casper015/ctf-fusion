# Fusing heuristics, policy networks and search

> The original design brief for this repository, reformatted. It targets the AI_Test `player7-nextgen` code
> (cdb45c5) under a 50 ms per-turn CPU limit. It was written before any implementation or benchmark.
> [REVIEW.md](REVIEW.md) records which parts the implementation adopted, which it changed and why. Literature is in
> [SOURCES.md](SOURCES.md).

## 1. Keep three quantities distinct

| Symbol | Meaning | Role |
|---|---|---|
| $q(b \mid s)$ | Opponent requested-action distribution | How the opponent is expected to act |
| $\pi(a \mid s)$ | Ego action proposal / prior | Which moves the search looks at first |
| $Q(s, a)$ | Expected outcome of an ego action | Comparative ranking |

The heuristic layer splits into hard rules, objective assignment, opponent prediction, move ordering and board evaluation.

- Code keeps enforcing the hard rules.
- Objective assignment and move ordering supply the candidates.
- Probabilistic prediction and value evaluation connect to neural networks separately.

A probability of 0.8 cannot be added to a heuristic score of 30.

## 2. Opponent prediction: a mixture

For team count $N$, mix two normalized distributions over the squad's **joint** action space:

$$q_\alpha(b \mid s) = (1 - \alpha_N)\, q_H(b \mid s) + \alpha_N\, q_\theta(b \mid s)$$

with the weight fitted on held-out data:

$$\alpha_N^* = \arg\min_{0 \le \alpha \le 1} \; -\sum_i w_i \log\big[(1-\alpha)\, q_H(b_i \mid s_i) + \alpha\, q_\theta(b_i \mid s_i)\big], \qquad w_i \ge 0$$

**The fit is convex.** With both predictors fixed this is a one-dimensional convex problem, since its second derivative is

$$\sum_i w_i \frac{(q_{\theta,i} - q_{H,i})^2}{q_{\alpha,i}^2} \ge 0 .$$

**What α\* does and does not promise.** It is the best log-loss within this mixture family on this dataset. It does not guarantee Bradley–Terry optimality, nor worst-case behavior against unseen opponents.

**Rules for the predictors:**
- Evaluate probabilities over the actual legal support; rule-violating actions keep probability zero.
- Calibrate by player count and a few key tactical situations, not with a complex gating network up front.
- In rare situations, shrink the weight toward the heuristic. A sharp softmax does not by itself mean the network is reliable.
- Verify downstream gains in independent matches.

**Residual learning**, when needed:

$$q_{\text{res}} = \operatorname{softmax}(\log q_H + \delta_\theta)$$

Initialize its final layer to zero so the start equals the baseline on $q_H$'s support. Legal moves the baseline omitted keep prior support. Supervised logit loss and regularization are a different objective from online control performance.

**Joint coordination must be preserved.** Blend the squad's joint distribution rather than three independent marginals. Opponent predictions are shared by all ego candidates for a given state. The prediction at the root must never see the ego team's actual decision for the current turn.

## 3. Candidates and value

The first version keeps the heuristic top action. The candidate pool holds:
- the top heuristic actions;
- the high-probability moves of an optional ego policy head;
- critical delivery, evasion, flag capture, attack and flag-drop actions.

$a$ is the joint action of all three units. 125 is only the upper bound on joint movement combinations, before attacks and flag interactions.

**Evaluating Q.** Under shared opponent scenarios, $Q$ comes from exact forward simulation plus a value network:
- If the value network predicts the **residual** future return, add the accumulated step rewards to its prediction.
- If it predicts the **final pairwise outcome**, pass it the board score after any flag delivery directly, without also adding a hand-coded delivery reward.

**Order of responsibility:** heuristics preserve candidates → networks expand candidates → search supplies one comparable return. Proven 1v1 tactical responses stay in the candidate set as fallbacks when inference fails or the budget runs out. Keeping a candidate does not by itself prevent regression, because $Q$ can still be wrong.

## 4. KL-regularized fusion (when an ego policy head exists)

**Building the prior.** Turn heuristic scores $H$ into a supported prior $p_H$, for example a temperature-calibrated softmax. Combine it with the ego policy $\pi_\theta$ to form $p_0$. Opponent predictions $q$ play no part in this step.

**Closed form.** Given values, priors and a temperature $\tau$:

$$\mu^* = \arg\max_\mu \Big\{\sum_a \mu(a)\, \hat Q(s,a) - \tau\, D_{\mathrm{KL}}(\mu \,\|\, p_0)\Big\}, \qquad
\mu^*(a) = \frac{p_0(a)\, e^{\hat Q(s,a)/\tau}}{\sum_c p_0(c)\, e^{\hat Q(s,c)/\tau}}$$

- A larger $\tau$ stays closer to $p_0$; a smaller $\tau$ trusts $\hat Q$ more.
- Compute it stably as $\operatorname{softmax}(\log p_0 + \hat Q/\tau - \max)$.
- Where $p_0(a) = 0$ this KL direction gives $a$ probability zero. So a deterministic heuristic must first become a non-degenerate distribution over the candidates before any deviation is possible.

**What it does not promise.** This is the exact solution of the stated regularized objective, structurally the MPO E-step. It does not guarantee monotonic improvement in an approximate CTF environment. Even with a perfect $Q$, bounds on the expected return of $\mu$ do not imply that $\arg\max_a \mu(a)$ beats the existing heuristics. Use $\mu$ first as a training target or search prior, benchmarked independently against plain search selection.

## 5. Limits of baseline safeguards

If the absolute value error is bounded by $\epsilon$ for any two actions,

$$\hat Q(a) - \hat Q(a_H) > 2\epsilon \;\Rightarrow\; Q(a) > Q(a_H).$$

In practice no uniform $\epsilon$ is available:
- Paired scenario evaluation lowers sampling variance, but it does not remove systematic errors in the opponent model or the value.
- Conservative switching thresholds stay useful.
- Standard errors and heuristic softmax confidences are not formal error bounds.

SPIBB and its reference code preserve a baseline under limited data. Their guarantees do not transfer directly to multi-team play with neural approximators.

## 6. Adjustments to the ten-tier plan

**Two model families first:**
- **Opponent policy P**, blended with the legacy predictor through player-count-calibrated weights. Legacy objective assignment and candidate generation stay.
- **Board value V**, which supplies terminal search evaluations.

An ego proposal head can later share P's encoder. It must train on search-improved move labels, not on opponent behavior traces.

**First ablation round:**
- legacy baseline;
- prediction fusion only;
- value-network replacement only;
- both together.

**What to track:**
- log-loss by player count;
- calibration on critical events;
- coverage of the top move by the candidates;
- final-move divergence rate;
- strict latency distributions;
- paired match outcomes.

MPO priors and residual logits belong to the second round, to isolate their contributions.

---

# Team-count-specific design and training plan

Base: `player7-nextgen` (cdb45c5b946151d01d46f976d7e981c9dc5db3aa).

**Target team counts:** 2, 3, 4, 5, 6, 7, 8, 10, 12, 15.

**Constraints:**
- turn 1, including initialization, ≤ 2 s;
- later turns ≤ 50 ms from input to complete output;
- one persistent process;
- unconstrained memory;
- portable to mobile WASM later.

The parameters below are starting points for validation, not confirmed optima.

## 1. Shared core, independent exports

**Shared core:** game rules, board representation, inference and caches.

**Per team count:** configuration files, model parameters and benchmark logs.

**Exports:** ten standalone executables (`bot_n02.cpp` … `bot_n15.cpp`), then one unified dispatch binary.

```text
profiles/n02.json ... n15.json   # player count, search budgets, module flags, model hashes
inference/policy.hpp             # joint squad action inference
inference/value.hpp              # incremental board value evaluation
training/collect.py              # trajectories: states, requested actions, events, returns
training/train_policy.py         # supervised joint policy
training/train_value.py          # value (TD / Monte Carlo)
training/export_weights.py       # quantized C++ arrays / fixed-point parameters
```

**What the current search lacks:**
- `search/types.hpp::configFor(teams)` returns one configuration for all team counts.
- `search/rollout.hpp` simplifies opponent casualties and flag transitions after step one.

Before exploring deeper plies, extract the verified rules from `stepExact` into a reusable simulator. More plies without full transition mechanics is not deeper search.

**What "exact" means here.** "Exact" forward simulation means exact rule transitions under fixed actions and stochastic outcomes; opponent moves remain probabilistic predictions. Distant opponents may use cheaper predictions, but still take part in collisions, combat damage and flag state updates.

## 2. Initial configuration by team count

**How to read the table:**
- Depth counts full game turns.
- Return DP is an approximate long-horizon heuristic, not a global adversarial search of the same depth.
- Candidate counts are post-pruning pool sizes.
- Scenario counts are starting ranges, capped dynamically by the per-turn deadline.

| Teams | Focus | Main depth and extensions | Root candidates | Opponent scenarios | Return DP horizon |
|---:|---|---|---:|---:|---:|
| 2 | 3-unit coordination, focus fire, no overkill, flag interception; keep the 1v1 heuristics | 2–3 plies; 4 on explicit delivery/kill | 16–24 | 64–256 | 10–12 |
| 3 | Crossfire between opponents, pickup/intercept transitions | 2; 3 for close flag fights | 16–24 | 64–192 | 8–10 |
| 4 | Flag capping, support, return-path roles; key tile contention | 2; 3 for top candidates | 12–20 | 64–192 | 8–10 |
| 5 | Return vs combat exposure; penalize over-conservative detours | 2; 3 on critical events | 12–20 | 48–160 | 6–8 |
| 6 | Flank and base-adjacent flags; fewer unrewarded skirmishes | 2; 3 on local events | 10–16 | 32–128 | 6–8 |
| 7 | Mid-scale; calibrate congestion and crossfire separately | 2 | 10–16 | 32–128 | 6–8 |
| 8 | Opponent tendencies, focus-fire assignment, multi-team signals | 2; extend only on pivotal flag battles | 10–16 | 32–128 | 6–8 |
| 10 | Coverage density, neighbor relations, throughput; avoid over-conservative DP | 2 | 8–12 | 32–96 | 4–6 |
| 12 | Larger maps; separate distant opportunities from local threats | quiet: 1 ply + V; contact: 2 | 8–12 | 24–96 | 4–6 |
| 15 | Local focus fire / flag contention, delivery probability, strict latency cap | quiet: 1 ply + V; contact: 2 | 6–10 | 24–64 | 4–6 |

**Calibration.** Calibrate these on the target CPU. At runtime, resource allocation is driven jointly by:
- local engagement distance;
- flag status;
- remaining turn time;
- candidate score gaps;
- model uncertainty.

Compare competing candidates on identical opponent scenario sets, in bounded mini-batches.

**Turn time budget:**
- **Turn 1:** initialization within 1.5–1.7 s, leaving time for the first move.
- **Later turns:** computation done by about 35 ms and output by about 40 ms.

| Time | Activity |
|---|---|
| 0–3 ms | Parse input, update history, stage a legal fallback move |
| 3–8 ms | Infer opponent joint actions, build the shared turn cache |
| 8–12 ms | Coarse heuristic evaluation of up to 125 ego movement combinations |
| 12–32 ms | Exact forward rollouts + network evaluation of the top candidates |
| 32–38 ms | Remaining time: verify delivery, focus fire and collision edge cases |
| ~40 ms | Emit the best verified action and flush |

**Caveats from earlier evidence:**
- 7- and 12-team settings had no independent evaluation data in the v3 suite, so their settings are working hypotheses.
- 5- and 10-team settings showed negative signals from return DP earlier, so return DP must stay switchable.
- Positive overall BT estimates across 3–15 teams do not imply per-tier significance, because opponent rosters varied with team count.

## 3. Models

**Per team count:** two compact networks, opponent squad policy $P_N$ and board value $V_N$. That makes 20 specialized parameter sets.

**General models:** two more, $P_{\text{general}}$ and $V_{\text{general}}$, for pretraining and fallback. That makes 22 logical sets in all.

**Loading rules:**
- Exactly one pair is loaded per match.
- Several random seeds are trained to measure stability, not for runtime ensembling.

**P architecture:**
- A shared entity/squad encoder with two small hidden layers (e.g. 64 → 32).
- 125 joint movement logits, plus conditional pickup, drop, attack and target heads.
- Rolling history statistics first; GRU or attention later, as ablations.

**V architecture:**
- A sparse 128-dimensional accumulator → 16 → 1.
- 256 dimensions only if it shows value within the 50 ms limit.
- Inputs: true team count, map metrics and interaction geometry, not raw slot indices.

**Training and fallback:**
- Pretrain on all team sizes, then fine-tune per tier.
- Data naturally clusters by map scale into 2, 3–5, 6–10 and 12–15 teams.
- Sizes 9, 11, 13 and 14 get no dedicated model; the general models cover them so the bot never refuses a size.

## 4. Training

### Data

**Volume.** 1,000–2,000 matches per team count, 10,000–20,000 in all. At 400 turns that is up to 4–8 million correlated states.

**Sampling.** Stratify across quiet states, skirmishes, pickups, deliveries, respawns and manual drops. Deduplicate global states across teams to save storage.

**Opponent diversity** is mandatory:
- 2-team matches cover Astra, Sol, Fable, Omega, Player2_v005 and Baseline.
- Other sizes randomize seats and include diverse styles.

**Splits.** Split train/validation/test strictly by game/map id and opponent family, so no information leaks between adjacent turns.

**What to log:**
- public pre-action state;
- requested moves and actions;
- resolved transitions;
- event flags;
- final scores;
- engine hashes.

**Leakage rules:**
- Store requested moves separately from post-collision positions.
- The opponents' current actions are offline labels only.
- Inference inputs must never see current opponent actions or future random flag respawns.

### Policy

**Objective.** Supervised learning on the actual requested actions: joint movement plus conditional action probabilities.

**Evaluation**, beyond per-unit accuracy:
- joint focus-fire alignment;
- collision rates;
- flag attack accuracy;
- log-loss calibration.

**Rules:**
- Learn mixed opponent distributions offline; no online gradient updates within 50 ms.
- If the 2-team network does not beat the heuristic predictor, keep the heuristic.

### Value

**Targets:**
- Main: the normalized pairwise outcome (win 1.0, draw 0.5, loss 0.0).
- Auxiliary heads: delivery and possession transitions.

If V predicts final outcomes, the search compares terminal states without adding a hand-made delivery reward again. If it predicts residual returns, use realized step rewards plus the remaining value.

**Training method:**
- Train with Monte Carlo returns and short-horizon TD targets.
- Verified strong offline search teachers may be used, but do not distill unvalidated heuristic evaluations.
- Validation must include states reached through suboptimal candidate paths, so V learns to evaluate off-policy branches.

### Iterations and budget

- Train small models for 3–10 epochs, with early stopping on validation loss and downstream play.
- Screen with one seed; confirm the final configuration with three.
- Scale to 5,000–10,000 games per tier only if underfitting persists with clear validation gains.
- GPUs may train offline; inference is CPU-only.

## 5. Implementation and verification sequence

1. Unify the exact simulator (`stepExact`), per-turn deadline checks, mini-batch search interruption and trajectory logging. Freeze existing bots as controls.
2. Validate inference interfaces, feature caches and training objectives on 2, 5, 8 and 15 teams. Integrate V first, keeping the legacy opponent predictor as the control.
3. Extend V to all ten tiers, then ablate P alone and P+V. Four variants per tier: baseline, V-only, P-only, P+V.
4. Final validation: at least 10,000 independent held-out matches per tier candidate (≥ 100,000 across ten tiers) on identical fixtures.
5. Publish:
   - game logs;
   - pairwise win rates;
   - BT ratings;
   - map-cluster breakdowns;
   - timeout rates;
   - p95/p99 latency.

   Then verify the unified 2–15 bot on randomized multi-team schedules.

**Deliverables:**
- ten specialized C++ files and one unified binary;
- a manifest of model and commit hashes;
- training data manifests.

Weights are embedded as C++ arrays, with no runtime dependency on files, Python, GPUs or network.

---

# Prompt for the implementing agent (as given)

> Implement the end-to-end data collection, training, evaluation and C++ export pipeline in `training/`.
>
> 1. `collect.py`: matches against a diverse pool (Astra, Sol, Fable, Omega, Player2_v005, Baseline). It records:
>    - public state;
>    - raw requested actions, separate from resolved positions;
>    - events (pickup, drop, delivery, kill);
>    - final pairwise outcomes.
>
>    Sampling is stratified, and data is stored by match id and team count.
> 2. `dataset.py`: tensors matching the C++ inference structures, and 125-way joint move labels plus action/target labels. Splits are by match id, never by turn.
> 3. `train_policy.py`: a 64 → 32 MLP with 125 joint logits and flag/attack heads; cross-entropy with label smoothing and temperature calibration; offline convex α fit against the heuristic; log-loss, calibration and joint accuracy.
> 4. `train_value.py`: accumulator → 128 → 16 → 1 on pairwise outcomes with MC returns and auxiliary event targets; reports MSE, ranking accuracy and calibration.
> 5. `export_weights.py`: int8/int16 fixed point or float arrays, as zero-dependency C++ headers (`inference/weights_n02.hpp` …), with checksum and architecture metadata.
> 6. `evaluate_offline.py`: 2, 5, 8 and 15 teams first. It reports policy log-loss versus the heuristic, α per team count, value MSE and ranking accuracy, parameter sizes and FLOPs.
>
> Requested moves must be kept apart from resolved positions; a collision stop is not an intentional stay. Inference inputs never see current opponent actions or future flag spawns. Document a small smoke test (e.g. 10 two-player matches).
