# Review of the design, and what was built

This page reviews [DESIGN.md](DESIGN.md) against the benchmark evidence and the AI_Test code. It records what this repository implements, what it changed from the brief, and the order of the next experiments.

**In short:** the brief's core is sound and is implemented as written:
- separate opponent prediction q, proposals π and value Q;
- a convex mixture weight per team count;
- requested-action labels;
- integer weights embedded in C++.

Five changes matter most:
1. stay inside the existing 30 ms search budget;
2. cover all 14 rated team counts (the 22-artifact matrix stands; see 3.2);
3. compute every feature in one C++ header shared by training and the bot;
4. integrate the opponent policy before the value network;
5. treat duels like every other team count: the elite-duel data shows opponent modeling matters there too (§3.6).

A source review of this repository (2026-10-08) found gaps between the gate and the deployed artifact; §7 lists them and the fixes.

## 1. Evidence this review uses

| Source | Size | Finding that matters here |
|---|---|---|
| C-series (`ctf-benchmark-v3/track2_2k_c_series`) | 44,000 matches, 22 variants × 2,000, 8 team counts | `Player5_Omega_LIT` +354.5, genes +346.9…+354.0, `Player5_Omega` +343.8, `Control_Hybrid` +331.5, `R3_HABITS_adaptive` +299.9 (21st of 22) |
| Same | 50 ms timeouts per 2,000 matches | every search variant **479–657**, every non-search variant **306–335** |
| Track 1 (`ctf-benchmark-v3/track1_10k_suite`) | 90,207 matches, 9 candidates × ~10,000 | `B2_Player2_v005` +357.8, `B1_Player5_Omega_r3` +353.0, `Control_Omega` +352.6; A/A pair 325.5 vs 325.1 |
| Same, 2 teams | | every candidate without opponent habit modeling wins **92.2%**; `Astra_Habits` / `Astra_Full` drop to **83.4%** |
| Same, 8/10/15 teams | 45,180 matches at 6–15 teams | `Astra_Habits` / `Astra_Full` (online opponent habit models) lead: 15 teams 22.9% / 22.5% win rate; 10 teams `Astra_Full` first |
| `RULES.md` | | "Covers 2 to 15 teams with equal weight per team size" |
| This repository, 2 local matches | 2,800 rows, 4 judge bots, in-sample | joint log-loss: uniform 3.96–4.00, legacy forecast q_H 3.46–3.73, "repeat last step" 3.19–3.26 nats |

The last row is two matches and only an indication. It says the legacy forecast is poorly calibrated for these opponents: a one-parameter inertia rule beats it.

## 2. Adopted as designed

| Brief | Where |
|---|---|
| q, π, Q kept separate; probabilities never added to scores | `inference/policy.hpp` returns probabilities; nothing in this repository mixes them into heuristic points |
| Joint 125-way move distribution, legal support only | mover + squad joint logits masked by wall/off-board legality (`fusion/models.py`, `inference/policy.hpp`) |
| Conditional pickup / drop / attack heads, attack target head | `act` (4-way, masked to actions the engine would execute) and `target` (pointer over hittable enemies within 4) |
| α\* per team count by held-out weighted log-loss (convex) | `fusion/mixture.py::fit_alpha`: exact bisection on the monotone derivative, tested against a grid |
| Temperature calibration | `fit_temperature` (convex in the inverse temperature); folded into the exported last layers |
| KL-regularized fusion closed form | `kl_fusion`, tested as the argmax of the regularized objective. Not used at runtime, as the brief advises |
| Requested moves separate from resolved positions | labels come only from reply lines; `blocked` marks collision stops, and the test checks they really did not move |
| No leakage | features see only the public stream up to the current block; replies are labels only; splits are by match |
| Value target = pairwise final outcome; auxiliary event heads | `labels.py`: outcome, capture within 20, own death within 10 |
| Stratified sampling with weights | per-stratum keep rates; importance weights restore the natural distribution in every metric |
| Integer weights in C++ arrays, checksums, metadata | `export_weights.py` → `inference/models/weights_<model>.hpp` + `MANIFEST.json` + `registry.hpp` |
| Small smoke test | README "Smoke test" |

## 3. Changed, and why

### 3.1 Keep the 30 ms search budget; inference must displace work, not add it

**The brief asks for more time than the search has.** It budgets about 35 ms of compute and output by 40 ms. The search budget is 30 ms, the AI_Test operating rule. The C-series already shows that the 30 ms search variants have about **twice the 50 ms timeouts** of the non-search variants (479–657 against 306–335 per 2,000 matches).

**What the policy costs** (sizes from the exporter; an estimate, not a measurement):
- One opponent team's joint policy is about 30k multiply-accumulates (MACs).
- At 15 teams that is 14 × 30k ≈ 0.42M MACs per turn. That should take well under a millisecond on the cluster CPU, but it is a hypothesis until the host check times it, and it excludes the parsing, search, output and startup of an integrated bot.
- It runs once per turn, before the search, like today's forecast.

**What the value net costs.** At about 9k MACs per state, V is too expensive to call per rollout sample: 24 plans × up to 512 samples ≈ 12k calls ≈ 100M MACs. It can only run at plan level (24 calls) or on the survivors of sequential halving. Profile its feature update, accumulator and tail on real branch states before fixing how often it runs.

**Ranking sampled plans with V.** A plan's value is `mean_j V(s'_j)` over its sampled successor states, not `V(mean_j features(s'_j))`: V is nonlinear, so the two differ. At a terminal state (after the last settlement) use the exact pairwise outcome instead of V (`inference/value.hpp` `nodeValue`).

**Latency is unmeasured.** None of this has been timed; latency must be measured on the cluster, not here.

### 3.2 The model matrix: 22 artifacts, all 14 team counts

**First proposal.** This review first proposed one model per map size (four tiers), for data efficiency.

**The decision.** The follow-up brief fixed the matrix at 22 artifacts:
- dedicated P and V for 2, 3, 4, 5, 6, 7, 8, 10, 12 and 15 teams;
- a general P and V pretrained on all counts, serving 9, 11, 13 and 14.

The pipeline implements that ([MODEL_MATRIX.md](../training/specs/MODEL_MATRIX.md)).

**The concern survives, as warm-starting.** Each dedicated model warm-starts from `general`, which has seen every count. A small dedicated dataset therefore refines the general model instead of starting cold.

**Where the matrix lives.** Profiles (`profiles/nXX.json`, `general.json`) define it, so changing it is a configuration edit. Point a count's `model` at `general` (or a shared id) to merge counts again.

**Why every count matters.** The rating weights every count 2–15 equally, so data is collected for all 14, including the four served by `general`.

**Source size.** About 200 KiB of header per exported model (the smoke export was 196 KiB), so about 2.2 MiB for all 11 models in one unified binary. Whether the judge limits source size is still an open question (section 6).

### 3.3 One feature implementation, in C++

**One implementation for training and inference.** The extractor (`tools/cpp/extract_features.cpp`) and the bot call the same `inference/features.hpp`, so training/inference skew is impossible by construction. Python never reimplements game geometry.

**Checked against the engine.** The engine-derived rules (spawn order and phase-1 respawns, `can_attack`, flag availability) are checked against the official `engine.py` in `tests/test_pipeline.py`. Respawn predictions are compared with the engine's actual respawn events.

**Bit-exact inference.** Features are clipped small integers, and the network is integer end to end (int16 weights, int32 accumulators with a proven overflow bound). The C++ output is therefore bit-identical to the Python emulation, which the tests check, and independent of FMA contraction or compiler.

### 3.4 Dense relational value features, not an NNUE sparse accumulator (v1)

**Why not sparse cells.** Maps are random per seed (pillars), so absolute-cell one-hot inputs — the NNUE pattern — would learn map-specific noise. A 49 × 49 × relation sparse table is also millions of weights, several MiB of source per model.

**What V uses instead.** V reads 54 relational features (feature spec v2): BFS distances, threats, carriers, and the score state. Spec v1 kept only own score, best rival, rank and margin, which cannot tell [5,6,4] from [5,6,5] for team 0 although the outcome target differs (0.5 against 0.25 if nobody scores again). v2 adds the tied count, the rivals one or two captures ahead and behind, and the current pairwise outcome. The network is 54 → 128 → 16 → 1. An incrementally updated accumulator only pays off once V runs inside rollouts, which section 3.1 rules out for now.

### 3.5 Policy before value

The brief proposes integrating V first. This repository recommends P first:

1. **Evidence.** Opponent modeling is what moved the large-team results. `Astra_Habits` / `Astra_Full` lead at 8–15 teams. In the search, q is the sampling distribution, so its errors compound over every sample.
2. **Interface.** P enters at one place: `search::Rival::prob` and `fire`, the forecast the search already samples. No value scale changes.

   V replaces the evaluator, whose terms are heuristic points blended 0.3 analytic / 0.7 sampled. That is exactly the probability-versus-score mixing the brief warns about, and every blend weight would need retuning.
3. **A built-in null.** With α = 0 the bot is bit-identical to today's. The AI_Test stream-exactness harness can prove it before a single match.

### 3.6 Duels: not saturated after all

The first version of this section called the duel saturated: every top candidate won 92.2% of track-1 duels, and habit models lost 8.8 points. Those duels were all against Baseline. Against strong opponents the picture is different ([DUEL_REVIEW.md](benchmark/DUEL_REVIEW.md); AI_Test `ctf-s1-controls/results/ANALYSIS.md`):
- the duel ceiling is Astra and Sol (the control wins about 46–48%);
- duel planners differ by several points against single opponents. CFL scores +6.2 over Player2's duel play against Player5 (pooled over 7,626 duels). CFL + expectimax scores +4.1 against Astra and +2.7 against Sol over CFL alone, but −5.1 against Baseline;
- online habit models move duels opponent by opponent (+5.4 against Player5, −3.0 against Astra), for a net of zero.

Opponent modeling clearly changes duel results, in both directions, so a learned 2-team opponent model (`n02`) is worth an experiment. It is gated like every other count: α = 0 unless it beats the legacy forecast on held-out duels, then paired duels against Astra and Sol specifically.

### 3.7 The deployment adapter: joint team samples (v2)

**The first plan mixed marginals.** The search samples each rival unit independently from its forecast marginals (`Rival.cell/prob`, `JointSampler`), so v1 planned to mix q_H and q_θ unit by unit: `q(m) = Π_k [(1-α) qH_k + α qNN_k]`.

**The gate scored a different distribution.** It fitted α for the joint mixture `(1-α) Π_k qH_k + α qNN(m)`. The two differ, and v1 throws away exactly what the squad network adds. Example: two units choose 0 or 1, the third waits; q_H is uniform over the four joint outcomes; the network puts ½ on (0,0) and ½ on (1,1). On labels split between (0,0) and (1,1), the joint mixture gains ln 2 nats per example at α = 1. Yet the network's unit marginals are uniform, so v1 reproduces q_H for every α. A positive gate would have approved a change that does nothing.

**The contract now** ([`inference/adapter.hpp`](../inference/adapter.hpp), [`fusion/mixture.py`](../fusion/mixture.py)):
- **`joint-v2` (default).** One mixture choice per opponent team and scenario. With probability α the whole team move comes from the network's 125-way joint, and the team's actions from the network's act and target heads conditioned on those moves (`drawAct`). Otherwise every unit draws from q_H and fires as the legacy planner says.
- **`marginal-v1`.** Kept as an option, and scored as itself: the gate then fits the product of the integer model's mixed marginals.
- Both give each unit the same marginal, `(1-α) qH_k + α qNN_k` (`mixedMarginal`), so analytic search terms (expected incoming fire) use it under either.
- The gate fits α for one adapter, the report records it, the exporter refuses a different one, and the header carries `kAdapter`. `tools/cpp/adapter_check.cpp` checks that the C++ probabilities equal the Python ones and that `drawTeam` samples them.

### 3.8 Things deliberately not built yet

(The decoupled simulator that the brief asks for before deeper plies now exists: `inference/sim.hpp`. It is verified turn by turn against the engine; see [SIMULATOR_AUDIT.md](../training/specs/SIMULATOR_AUDIT.md).)

- **Off-policy value states.** The brief requires states from suboptimal branches. Real counterfactual branching needs bot processes restarted from altered streams, and the search bots are clock-dependent.

  A cheap substitute is a proxy that perturbs a seat's replies with probability ε (random legal move for one unit). Not built yet.
- **Online adaptation.** The offline α_N is a prior. A per-opponent posterior (exponential weights over q_H, q_θ and inertia, updated from each opponent's observed moves) is the natural next step. It is also what the strong `Astra_Habits` result suggests.

  Note that this repository's own online habit module (`R3_HABITS_adaptive`) ranked 21st of 22, so it must be measured alone.
- **Deeper search.** The brief's per-count depth table is a hypothesis. `G_PLAYOUT_4ply` (+346.9) did not beat the other genes, so depth does not look like the bottleneck. Keep two plies; tune scenario counts per size where measured.
- **Ten per-count exports.** One unified bot with `fusion::models::forTeams(teams)` (generated `registry.hpp`) replaces them; `profiles/` maps counts to models.
- **A delivery value.** V's targets are the final pairwise outcome (exported), a team capture within 20 turns and an own death within 10 (auxiliary, not exported). A team's near-term capture is not a given carrier's chance to deliver before death, drop or takeover. Per-flag transport episodes, possession-transition targets and counterfactual branch values are not built, so the carrier DP stays until a delivery head has labels, tests and a consumer.

## 4. Integration plan (AI_Test `nextgen`)

1. **Inline the headers.** Inline `inference/state.hpp`, `features.hpp`, `mlp.hpp`, `policy.hpp`, `value.hpp`, `sim.hpp`, the `weights_*.hpp` headers and `registry.hpp` into `master.cpp` exports, the way `build_exports.py` inlines `search/`. They use only `<algorithm> <array> <cmath> <cstdint> <cstdlib> <utility> <vector>`.
2. **Build the state.** In `multi::Planner::forecast`:
   - build `fusion::State` from `Game` (a few lines: units, flags, attack events, scores);
   - build `Board` once and `TurnView` once per turn;
   - run `encodePolicy` + `jointPolicy` for each opponent team with a unit on the board, using `fusion::models::forTeams(teams)`.
3. **Sample with the adapter the gate approved** (`kAdapter`, `alpha = forTeams(teams)->alpha`):
   - **joint-v2:** in each scenario, draw each opponent team's move with `drawTeam`. A learned draw also takes the team's pickup / drop / attack from `drawAct`, conditioned on the sampled moves. A legacy draw keeps today's per-unit sampling and fire model. Analytic terms use `mixedMarginal`.
   - **marginal-v1:** replace each unit's forecast by `mixedMarginal`, and fire by the matching mixture of the act head's P(attack | destination) and today's `MOD_FLAGSHOOT` conditioning.

   The switch is `MOD_FUSION_P`, with a team-size mask like every other module.
4. **Prove the null.** With `MOD_FUSION_P_SIZES = 0`, or α = 0, the stream-exactness check must report zero differences against the foundation. α only switches P. Any use of V goes behind its own switch (`MOD_FUSION_V`), off by default, with its own null check, because no gate approves V (`kValueApproved = false`).
5. **Measure** on the cluster:
   - first-turn startup and p50/p99 input-to-reply time per size, for the integrated bot;
   - paired matches, `Player5_Omega_LIT` against `Player5_Omega_LIT + P`, per size.

## 5. Experiment order (cluster)

| Step | What | Decides |
|---|---|---|
| E0 | Collect 1,500 matches × 14 team counts (`profiles/`); build shards; legacy baseline on val/test | Data |
| E1 | Train the general P, then a few representative dedicated P models (`n02`, `n05`, `n15`); offline report, strict holdout included | Does the integer q_θ beat q_H per team count, through the joint adapter? α per count. Stop for any count where it does not; scale to all 22 artifacts only after it does. |
| E2 | `Omega_LIT + P(α)` against `Omega_LIT`, paired, per size; also α = 1 | Match strength of P |
| E3 | Train V; offline Brier and ranking against a score-margin baseline | Whether V is worth integrating |
| E4 | V at plan level (24 calls) inside `search::rank`, then on halving survivors | Match strength of V |
| E5 | `n02` duel integration, paired against Astra and Sol | Whether opponent modeling lifts the duel ceiling (§3.6) |

**Statistical design.** Paired fixtures make comparisons tight: the track-1 A/A pair agreed within 0.4 BT. Sequential stopping per size (stop once the CI excludes zero, or once it excludes the smallest effect worth shipping) will need far fewer than the brief's 10,000 matches per tier for clear results.

## 6. Open questions for the user

- **Source size.** Is there a judge-side size limit for `bot.cpp`? All 11 models add about 2.2 MiB of source (about 200 KiB each). `RULES.md` names none.
- **The official field.** Are the official opponents the judge bots (plus other submissions)? The opponent model is only as good as the overlap between the training pool and the real field. The pool (`profiles/pool.json`) now holds the judge bots without Doubao, the original Player5 and Player4, our lineage, Baseline, and a random-movement Noise bot whose own moves are not policy labels.

## 7. Source review (2026-10-08) and what changed

A static review of commit `34a83b0` checked the data and deployment contracts before the expensive training run. Every finding was confirmed in the code and fixed:

| Finding | Fix | Checked by |
|---|---|---|
| **P1. Windows corrupts raw tool output.** `extract_features` and `infer_check` wrote binary records to a text-mode stdout; on Windows every `0x0A` byte becomes `0x0D 0x0A`. | Both switch stdout to `_O_BINARY` (`tools/cpp/stream.hpp` `binaryStdout`). `fusion.native` checks the byte count and every record's turn and team columns. | `test_contracts`: a stream whose records contain `0x0A`, `0x0D` and `0x1A`; `env_check.py` repeats the check on the host |
| **The gate scored a different distribution from the planned adapter.** Joint mixture offline, independent unit mixtures at runtime (§3.7). | One declared adapter per artifact (`joint-v2` by default; `marginal-v1` scored as itself), in C++ and Python, recorded in the report and the header. | `test_contracts`: C++ probabilities equal Python's to 1e-12; 400k draws match them; the coordination example |
| **Missing q_H switched the comparison to inertia**, so a model could be approved against the wrong baseline. | Without a complete, single-build q_H the gate stays closed; inertia is only a diagnostic. `heuristic_baseline` records the baseline's source digest per shard. | `test_pipeline`: an incomplete baseline closes the gate |
| **The gate used float predictions.** | `evaluate_offline` quantizes the runs exactly as the exporter does and gates the integer model (temperatures folded in). | Report columns `learned` (float, diagnostic) and `quantized` |
| **Any report could authorize any weights.** | The report carries the artifact identity (model, checkpoint hashes, integer weight digest, schema, adapter), plus dataset and baseline. The exporter recomputes it and stops on any difference. | `test_contracts`; `test_pipeline`: another model's report, or another adapter, is refused |
| **V features collided.** [5,6,4] and [5,6,5] gave team 0 identical inputs. | Feature spec v2 adds tied, 1–2 ahead, 1–2 behind and the current pairwise outcome; terminal states use the exact outcome (`nodeValue`). Datasets, warm starts, evaluation and export refuse other spec versions. | `test_contracts`: the two score vectors now differ in exactly those features |
| **The family holdout leaked.** Only the held-out family's own rows left training. | `--holdout-mode fixture` (default) reserves whole matches; `actor-row` remains, named as the weaker test. Splits group by map; gate errors are clustered by map; the gate needs 20 maps and a gain above two standard errors. | `test_pipeline`: fixture and actor-row holdouts |
| **Wide layers overran the C++ buffers.** | Trainers and the exporter refuse widths above 256; headers `static_assert` every layer. | `test_contracts` |
| **Simulator checks could pass without coverage.** | `sim_check` exits 3 when no turn was compared; the tests require every turn compared; `env_check` reports compared and skipped turns. | `test_pipeline` |
| **Scope.** V is not a delivery value; latency is unmeasured; the duel was not saturated. | §3.8, §3.1 and §3.6 now say so; α gates only P (`kValueApproved = false`). | — |

Order from here:
1. Run `tools/host/env_check.py` on the Windows host: binary records, integer parity, runtime contract, simulator coverage, timing.
2. Collect E0.
3. Train the general P and a few dedicated models.
4. Scale only what beats q_H through the declared adapter.
