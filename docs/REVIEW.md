# Review of the design, and what was built

This page reviews [DESIGN.md](DESIGN.md) against the benchmark evidence and the AI_Test code. It records what this repository implements, what it changed from the brief, and the order of the next experiments.

**In short:** the brief's core is sound and is implemented as written:
- separate opponent prediction q, proposals π and value Q;
- a convex mixture weight per team count;
- requested-action labels;
- integer weights embedded in C++.

Five changes matter most:
1. stay inside the existing 30 ms search budget;
2. train one model per map size instead of 22 parameter sets, covering all 14 rated team counts;
3. compute every feature in one C++ header shared by training and the bot;
4. integrate the opponent policy before the value network;
5. leave the duel planner alone at first.

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
| Integer weights in C++ arrays, checksums, metadata | `export_weights.py` → `inference/models/<tier>.hpp` + `MANIFEST.json` |
| Small smoke test | README "Smoke test" |

## 3. Changed, and why

### 3.1 Keep the 30 ms search budget; inference must displace work, not add it

**The brief asks for more time than the search has.** It budgets about 35 ms of compute and output by 40 ms. The search budget is 30 ms, the AI_Test operating rule. The C-series already shows that the 30 ms search variants have about **twice the 50 ms timeouts** of the non-search variants (479–657 against 306–335 per 2,000 matches).

**What the policy costs** (sizes from the exporter):
- One opponent team's joint policy is about 30k multiply-accumulates (MACs).
- At 15 teams that is 14 × 30k ≈ 0.42M MACs per turn, well under a millisecond on the cluster CPU.
- It runs once per turn, before the search, like today's forecast.

**What the value net costs.** At about 8.5k MACs per state, V is too expensive to call per rollout sample: 24 plans × up to 512 samples ≈ 12k calls ≈ 100M MACs. It can only run at plan level (24 calls) or on the survivors of sequential halving.

**Latency is unmeasured.** None of this has been timed; latency must be measured on the cluster, not here.

### 3.2 One model per map size, all 14 team counts

**Every team count counts equally.** The rating weights every count from 2 to 15 equally, so 9, 11, 13 and 14 count as much as 8 or 10.

**Map size moves at exactly three boundaries.** `engine.map_size` is 23 for 2 teams, 31 for 3–5, 41 for 6–10 and 49 for 11–15.

**So: four tiers, data for every count:**
- one model pair per map size (tiers `m23`, `m31`, `m41`, `m49`), with the team count as an input feature;
- data from all 14 counts;
- one α per team count.

Compared with 22 parameter sets, each model sees about four times more data. Each tier's source is about 200 KiB (the smoke export was 196 KiB), against several MiB for 22 sets.

A per-count model is still one line away: set `"tier": "n08"` in `profiles/n08.json`. Do it only if that count's held-out loss shows the shared model underfits.

### 3.3 One feature implementation, in C++

**One implementation for training and inference.** The extractor (`tools/extract_features.cpp`) and the bot call the same `inference/features.hpp`, so training/inference skew is impossible by construction. Python never reimplements game geometry.

**Checked against the engine.** The engine-derived rules (spawn order and phase-1 respawns, `can_attack`, flag availability) are checked against the official `engine.py` in `tests/test_pipeline.py`. Respawn predictions are compared with the engine's actual respawn events.

**Bit-exact inference.** Features are clipped small integers, and the network is integer end to end (int16 weights, int32 accumulators with a proven overflow bound). The C++ output is therefore bit-identical to the Python emulation, which the tests check, and independent of FMA contraction or compiler.

### 3.4 Dense relational value features, not an NNUE sparse accumulator (v1)

**Why not sparse cells.** Maps are random per seed (pillars), so absolute-cell one-hot inputs — the NNUE pattern — would learn map-specific noise. A 49 × 49 × relation sparse table is also millions of weights, several MiB of source per tier.

**What v1 uses instead.** V v1 reads 50 relational features: BFS distances, threats, carriers, score state. The network is 50 → 128 → 16 → 1. An incrementally updated accumulator only pays off once V runs inside rollouts, which section 3.1 rules out for now.

### 3.5 Policy before value

The brief proposes integrating V first. This repository recommends P first:

1. **Evidence.** Opponent modeling is what moved the large-team results. `Astra_Habits` / `Astra_Full` lead at 8–15 teams. In the search, q is the sampling distribution, so its errors compound over every sample.
2. **Interface.** P enters at one place: `search::Rival::prob` and `fire`, the forecast the search already samples. No value scale changes.

   V replaces the evaluator, whose terms are heuristic points blended 0.3 analytic / 0.7 sampled. That is exactly the probability-versus-score mixing the brief warns about, and every blend weight would need retuning.
3. **A built-in null.** With α = 0 the bot is bit-identical to today's. The AI_Test stream-exactness harness can prove it before a single match.

### 3.6 Leave the duel alone at first

**The duel is saturated:**
- Every top candidate wins 92.2% of duels.
- The only variants that model opponent habits in duels lost 8.8 points.

**What to do.** Collect 2-team data (cheap) and train `m23`, but integrate it only after 3–15 teams show a gain. The brief's own rule — "if the 2-team network does not beat the heuristic predictor, keep the heuristic" — likely applies.

### 3.7 Marginals first, joint samples second

**The sampler can't take a joint distribution yet.** The search samples each rival unit independently from its forecast marginals (`Rival.cell/prob`, `JointSampler`). A joint q_θ can therefore enter only through its marginals until the sampler draws joint per-team samples.

**Plan:**
- **v1:** marginals. `PolicyOutput.marginal` already provides them.
- **v2:** joint team samples, as a separately measured change.

The brief's warning against factorizing applies in full to v2.

### 3.8 Things deliberately not built yet

- **Off-policy value states.** The brief requires states from suboptimal branches. Real counterfactual branching needs bot processes restarted from altered streams, and the search bots are clock-dependent.

  A cheap substitute is a proxy that perturbs a seat's replies with probability ε (random legal move for one unit). Not built yet.
- **Online adaptation.** The offline α_N is a prior. A per-opponent posterior (exponential weights over q_H, q_θ and inertia, updated from each opponent's observed moves) is the natural next step. It is also what the strong `Astra_Habits` result suggests.

  Note that this repository's own online habit module (`R3_HABITS_adaptive`) ranked 21st of 22, so it must be measured alone.
- **Deeper search.** The brief's per-count depth table is a hypothesis. `G_PLAYOUT_4ply` (+346.9) did not beat the other genes, so depth does not look like the bottleneck. Keep two plies; tune scenario counts per size where measured.
- **Ten per-count exports.** One unified bot with a per-size tier switch replaces them; `profiles/` already maps counts to tiers.

## 4. Integration plan (AI_Test `nextgen`)

1. **Inline the headers.** Inline `inference/state.hpp`, `features.hpp`, `mlp.hpp`, `policy.hpp` and the tier headers into `master.cpp` exports, the way `build_exports.py` inlines `search/`. They use only `<algorithm> <array> <cmath> <cstdint> <cstdlib> <deque> <vector>`.
2. **Build the state.** In `multi::Planner::forecast`:
   - build `fusion::State` from `Game` (a few lines: units, flags, attack events, scores);
   - build `Board` once and `TurnView` once per turn;
   - run `encodePolicy` + `jointPolicy` for each opponent team with a unit on the board.
3. **Mix into the forecast**, per enemy unit and destination:

   `prob = (1 - a) * prob + a * marginal[k][move]`, with `a = kAlpha[teams]`.

   Do the same for `fire` with the act head's P(attack | destination), conditioned the way `MOD_FLAGSHOOT` conditions it today. The switch is `MOD_FUSION_P`, with a team-size mask like every other module.
4. **Prove the null.** With `MOD_FUSION_P_SIZES = 0`, or α = 0, the stream-exactness check must report zero differences against the foundation.
5. **Measure** on the cluster:
   - p50/p99 turn time per size;
   - paired matches, `Player5_Omega_LIT` against `Player5_Omega_LIT + P`, per size.

## 5. Experiment order (cluster)

| Step | What | Decides |
|---|---|---|
| E0 | Collect 1,500 matches × 14 team counts (`profiles/`); build shards; legacy baseline on val/test | Data |
| E1 | Train `m31`, `m41`, `m49` policies; offline report | Does q_θ beat q_H and inertia per team count? α per count. Stop here for any count where it does not. |
| E2 | `Omega_LIT + P(α)` against `Omega_LIT`, paired, per size; also α = 1 | Match strength of P |
| E3 | Train V; offline Brier and ranking against a score-margin baseline | Whether V is worth integrating |
| E4 | V at plan level (24 calls) inside `search::rank`, then on halving survivors | Match strength of V |
| E5 | `m23` duel integration, only if E2 is positive | |

**Statistical design.** Paired fixtures make comparisons tight: the track-1 A/A pair agreed within 0.4 BT. Sequential stopping per size (stop once the CI excludes zero, or once it excludes the smallest effect worth shipping) will need far fewer than the brief's 10,000 matches per tier for clear results.

## 6. Open questions for the user

- **Source size.** Is there a judge-side size limit for `bot.cpp`? Four tiers add about 0.8 MiB of source. `RULES.md` names none.
- **The official field.** Are the official opponents the 11 judge bots (plus other submissions)? The opponent model is only as good as the overlap between the training pool (`profiles/pool.json`) and the real field.
