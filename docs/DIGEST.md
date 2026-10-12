# Digest of every Markdown file

Every tracked Markdown file is summarised here once. Documents on the same topic share one section. Written 2026-10-11 (UTC) against `main` at 958d6e9.

- **The originals stay authoritative.** The numbers below are quoted from them; the cross-tier means in [Models](#models-capacity-features-offline-gates) were computed for this page from the per-count reports.
- **Each section opens with a Sources line** naming the files and folders it covers. [MARKDOWN_INDEX.md](MARKDOWN_INDEX.md) lists every file with its title and its section here; `tools/repo/md_index.py` generates that list from the Sources lines.
- **Utility** in the screens is the share of the other teams beaten (ties count half). A difference is reported as **helps** or **hurts** only when its 95% interval excludes zero; otherwise it is **not shown**.

**Contents:**
- [Where things stand](#where-things-stand)
- [Guides](#guides-how-the-repository-works)
- [The task board](#the-task-board)
- [Design and its review](#design-and-its-review)
- [Training contracts](#training-contracts)
- [Models](#models-capacity-features-offline-gates)
- [The host](#the-host-environment-speed-capacity)
- [Opponent roster](#opponent-roster)
- [AI_Test benchmarks](#ai_test-heuristic-benchmarks-and-duels)
- [Hybrid carriers](#hybrid-carriers-the-first-p-and-v-pilots)
- [P-only bots](#p-only-bots)
- [Archived briefs](#archived-handoff-briefs)
- [Triage](#triage-of-the-pre-board-prompts)
- [Seeds](#seeds)
- [Known defects](#known-defects-in-the-records)
- [Open questions](#open-questions)

## Where things stand

**Built:**
- **Pipeline.** Collection, shards and training of the opponent policy P and the value net V. The export writes integer C++ that is bit-identical to its Python emulation.
- **Models.** Tiers S to XXL for all 14 team counts.
- **Simulator.** An exact simulator, `inference/sim.hpp`, matches the engine on every compared turn.
- **Bots.** The P-only C08 and v12 bots take every opponent prediction from P.
- **Tooling.** Roster tools cover 258 eligible opponent bots. Alongside them: the task board, and CI that runs against the public engine (chen-006/meow-ai-arena, pinned to 54139d4).

**Shown:**
- **P predicts opponent moves better than the legacy forecast q_H.** It wins at every count and every tier, offline on held-out maps. The mean gain is 1.12 nats per joint move at S and 1.28 at XXL.
- **In paired play** (280 fixtures per screen):
  - C08 P-only XXL beats its frozen control by +0.040 on the tuning seed.
  - v12 P-only XL and XXL beat theirs by +0.052 and +0.070 on a second fresh seed.
  - Each of these effects is *not shown* on the other seed.

**Not shown:**
- Any overall benefit from V. This covers stage 2, the V3-XL screens and the 5XL pilot.
- A best carrier: C08 against v12 is *not shown* on either seed.
- Any result at confirmation scale (10,000 or more map groups).
- How P does against opponent lineages it never saw.

**Now:**
- T0022 is active: a fresh-map V3-XL screen on the host.
- Casper deferred the four costly pre-board steps in T0027 (2026-10-11): none runs until a task needs it.
- The critical path to a frozen submission:
  - T0010 → T0012 → T0014: the V pilot.
  - T0019 → T0005: four-turn lookahead.
  - T0004: near-large / far-small P.
  - Then T0006: freeze and confirm.
- T0002 waits on T0001. T0001's check can no longer pass, and its log asks whether to drop it.

## Guides: how the repository works

**Sources:** [README.md](../README.md), [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md), [CONTRIBUTING.md](../CONTRIBUTING.md), [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md), [PATH_MAP.md](PATH_MAP.md), [tools/README.md](../tools/README.md), [reports/README.md](../reports/README.md), [inference/models/README.md](../inference/models/README.md), [DIGEST.md](DIGEST.md), [MARKDOWN_INDEX.md](MARKDOWN_INDEX.md)

| File | Read it for |
|---|---|
| README | What the project is, the quick start, and the folder map. |
| AGENTS (CLAUDE.md only imports it) | The protocol for agents: identity (`tool@machine`), claiming tasks, worktrees, lanes, machine locks, landing, handoffs. |
| CONTRIBUTING | Where each kind of file goes, what is committed and what is not, generated files, frozen paths, branch and commit names, and what CI runs. |
| TECHNICAL_OVERVIEW | How collection, training, export and the bots fit together. Also the requirements, where the engine comes from (an AI_Test checkout or the public arena), the host steps and the E0–E5 commands. |
| PATH_MAP | Where every file was before the 2026-10-10 reorganization. Older reports cite those paths. |
| tools/README, reports/README, inference/models/README | Folder guides. The last one is out of date (see [Known defects](#known-defects-in-the-records)). |
| DIGEST (this page), MARKDOWN_INDEX | All documents compressed by topic, and the generated list of every Markdown file. |

## The task board

**Sources:** [tasks/README.md](../tasks/README.md), [tasks/SUMMARY.md](../tasks/SUMMARY.md), [tasks/](../tasks/)

- **How it works.** Each task is one file (`T####-slug.md`): YAML front matter (status, owner, needs, writes, depends, land) and the sections Goal, Done when, Handoff and Log.
  - `tools/repo/task.py` makes every change as one compare-and-swap commit on `main`, so two agents can never both claim a task.
  - `writes` are lanes: a claim, amend or merge is refused while another active task writes there.
  - `merge` folds duplicate tasks into one. Every change regenerates [SUMMARY.md](../tasks/SUMMARY.md), the live state of the board.

**The tasks by theme:**

| Theme | Tasks |
|---|---|
| P-only line | T0001 (Windows host check; drop it?) → T0002 (stage 1, capacity and sampling). T0003 (done: V at fixed P, not shown). T0003 → T0004 (near-large / far-small P) and T0005 (four-turn lookahead, also after T0019) → T0006 (freeze, confirm). |
| Feature families and V | T0009 (done: p3/v3 families, 4XL infrastructure) → T0010 (V data) → T0012 (V pilot, GPU) → T0014 (V pilot on the host) → T0006. T0011 (train p2 and p3) → T0013 (p3 screen). T0016 (done: 5XL V). T0022 (active: fresh-map V3-XL screen). |
| Search | T0018 (done: action portfolio, per-branch ply 2; T0017 merged into it) → T0019 (host evaluation) → T0005. T0020 (done: deadline and regret fixes). |
| Evidence gaps | T0007 (more seeds or per-epoch logs for L to XXL). T0025 (P on the two reserved lineages). T0026 (calibrate pure P). T0027 (done: all four steps deferred). |
| Infrastructure | T0021 (done: CI/CD). T0023 (done: the public arena as the engine source). T0024 (this consolidation; T0008 and T0015 merged into it). |

## Design and its review

**Sources:** [DESIGN.md](DESIGN.md), [REVIEW.md](REVIEW.md), [SOURCES.md](SOURCES.md)

**DESIGN, the original brief.** It targeted AI_Test `player7-nextgen` (cdb45c5) under a 50 ms CPU limit.
- **Three quantities stay separate.** q is the opponent's action distribution, π is our move proposal and Q is the action value. A probability is never added to a heuristic score.
- **Opponent prediction is a mixture** over the squad's joint action: q = (1−α_N)·q_H + α_N·q_θ. The weight α_N is fitted per team count by held-out log-loss, a convex problem.
- **KL-regularized fusion** has the closed form μ* ∝ p₀·exp(Q/τ). It is meant as a training target, not as a runtime rule.
- **A depth and candidate table per team count**, and a turn budget: compute done by about 35 ms, output by about 40 ms.
- **22 model sets:** P and V for 2–8, 10, 12 and 15 teams, plus a general pair.
- **Data:** 1,000–2,000 matches per count, split by map and by opponent family, with leakage rules. V's target is the pairwise outcome.
- **Validation:** at least 10,000 held-out matches per tier.

**REVIEW, what was built.** It adopted the core: q/π/Q kept apart, the convex α, requested-action labels and integer weights in C++. It made five changes:
1. Keep the 30 ms search budget: inference must replace work, not add to it.
2. Cover all 14 rated counts. The general model serves 9, 11, 13 and 14.
3. One C++ feature implementation, shared by training and the bot, so inference is bit-exact.
4. Integrate P before V.
5. Duels are not saturated (from the elite-duel data).

Other points from the review:
- **Deployment adapter `joint-v2`:** one mixture draw per opponent team and scenario.
- **Experiment order:** E0 data → E1 train P → E2 play P → E3/E4 V → E5 the duel model.
- **Source review of 34a83b0 (2026-10-08), all fixed:** Windows binary stdout, the gate scoring a different distribution than the adapter, a missing q_H, a float gate, report-to-artifact identity, colliding V features, a leaking family holdout, layer widths, and simulator coverage.
- **Open questions:** the judge's source-size limit (since answered: [bots/p_only/TASK.md](../bots/p_only/TASK.md) says size and memory are free) and the composition of the official field.

**SOURCES** lists the literature: MPO, RLax, SPIBB, temperature scaling, NNUE.

## Training contracts

**Sources:** [training/specs/](../training/specs/)

- **ACTIONS.**
  - 125 joint moves per squad (5³).
  - A per-unit `act` head: wait, pickup, drop or attack, masked to what the engine would execute.
  - A `target` head: a pointer over at most 8 hittable enemies within Manhattan 4.
  - Labels are the requested moves; collision-blocked moves are kept.
- **DATA_FORMAT.** Raw match records become shards. Features see only the public stream, and replies are labels only. Splits are by match and map; the holdout mode `fixture` is the default and `actor-row` is the weaker test.
- **ENVIRONMENT.** The training host and inference target: torch backend parity and the host checks.
- **FEATURES** (generated). Families and their digests:
  - p2: 40 / 222 / 7 inputs, digest `213481118d8a6263`.
  - p3: p2 plus 15 combat features per unit and move, `326a4299784b00bd`.
  - v2: 54 inputs, `ff0f693ef8e95ff0`.
  - v3: 108 inputs, `b58aaa174f156003`.

  The extractor record is layout 3: 923 int16 values per turn and team. Legacy digests are frozen, so old weights never meet new features.
- **MODEL_MATRIX.**
  - 22 artifacts; profiles map team counts to models.
  - Turn budget: search 30 ms, compute 35, output 40, engine 50, first turn 2,000 ms.
  - P must take at most 5 ms per turn on the host.
  - The P gate scores the integer model on at least 20 maps and needs a gain above two standard errors.
  - Every header says `kValueApproved = false`: no gate approves V.
- **SIMULATOR_AUDIT.**
  - AI_Test's `stepExact` gives up on turns where a flag returns: 9.9% of turns in two recorded matches.
  - Its rollout plies beyond the first were scripted.
  - `inference/sim.hpp` is a standalone exact step. Compared against the engine, every turn is identical: 0 mismatches.

## Models: capacity, features, offline gates

**Sources:** [LARGER_MODEL_EXPERIMENTS.md](LARGER_MODEL_EXPERIMENTS.md), [FEATURE_FAMILIES.md](FEATURE_FAMILIES.md), [V_CAPACITY_5XL.md](V_CAPACITY_5XL.md), [RL_PLAN.md](RL_PLAN.md), [P_MIXED_PRECISION.md](P_MIXED_PRECISION.md), [reports/models/](../reports/models/), [MODEL_EXPORT_REVIEW_8595c36.md](../reports/reviews/MODEL_EXPORT_REVIEW_8595c36.md), [TIER_M_REVIEW_AND_L_XL_XXL_TEST_PLAN_7c3f26d.md](../reports/reviews/TIER_M_REVIEW_AND_L_XL_XXL_TEST_PLAN_7c3f26d.md)

**Tiers and the offline gate.** The test split holds out maps. Each count uses its dedicated model, and the general model for 9, 11, 13 and 14. The means are over all 14 counts. The gain is q_H's loss minus that of the gated mixture, which gives P a weight α between 0.94 and 0.99. Every count passes the gate in every tier.

| Tier | Squad / V hidden | Parameters | P MACs per opponent team | V MACs | Mean integer NLL | Mean gain over q_H | Gain at 2 / 15 teams |
|---|---|---:|---:|---:|---:|---:|---|
| S | 64,32 / 128,16 | 31,068 | 30,096 | 8,976 | 2.0955 | 1.117 | 1.110 / 0.902 |
| M | 128,64 / 256,32 | 70,196 | 64,288 | 22,048 | 2.0361 | 1.174 | 1.215 / 0.943 |
| L | 256,64 / 512,64 | 148,436 | 120,576 | 60,480 | 1.9982 | 1.212 | 1.286 / 0.971 |
| XL | 512,128 / 1024,64 | 329,380 | 273,920 | 120,896 | 1.9531 | 1.257 | 1.366 / 1.002 |
| XXL | 1024,256 / 2048,128 | 920,484 | 678,912 | 372,864 | 1.9252 | 1.284 | 1.413 / 1.020 |

**Limits of these numbers:**
- **One training seed.** 8 epochs and no per-epoch logs (T0007).
- **Mixed backends.** M was trained with NumPy, L and up with PyTorch, so keep M out of the size trend.
- **Count weighting.** The general models use `inv_teams` count weighting; the XL and XXL summaries correct an earlier wrong note on this.
- **Calibration does not follow NLL.** M beats S by 0.0595 nats (2.86%) at every count, yet its ECE is worse at 10 of 14 counts.
- **No out-of-distribution report.** It needs the reserved lineages (T0025).
- **V has no gate.**

**The export review (8595c36).**
- 360,162 int32 comparisons of the C++ against the NPZ emulator: 0 differences.
- The reported gains, 0.90–1.44 nats for S, are internally consistent but cannot be recomputed: the raw data is not in the repository.
- V needs a stronger baseline (score margin and time) and a continuation contract.

**Wider shapes** (LARGER_MODEL_EXPERIMENTS). Measured times for one 15-team stream:

| Shape | 1 process | 16 processes |
|---|---:|---:|
| XXL | 1.68 ms | 2.74 ms |
| 3XL | 2.93 ms | 23.04 ms |
| 4XL | 9.73 ms | 191.5 ms |

The presets therefore stop at XXL. Wider nets can only be offline teachers. Capacity is judged on the same data with repeated seeds, not on training loss.

**Feature families** (FEATURE_FAMILIES, T0009). Built and tested on a development machine; nothing is trained or played yet (T0010–T0014).
- **Diagnosis.** Only the target head sees a target's HP, a carried flag or a volley; mover, squad and act never do. v2 also confuses pairs of states:
  - an enemy carrier at 100 hp and at 34 hp;
  - a dropped flag returning in 14 turns and one returning in 1.
- **Fix.** p3 adds a 15-feature combat block per unit and move; v3 adds 54 V features. Families are versioned by digest.
- **V data.** Games of the frozen carrier on seed 20261201, used for training only.
- **4XL pilot arms.** A v2/M (22,403 parameters), B v3/M, C v2/4XL, D v3/4XL (5,089,283 parameters).
- **Two ways to deploy a large V:**
  - Route A calls it directly, bounded by `PONLY_V_MAX_CALLS` and `PONLY_V_DEADLINE_MS`.
  - Route B distills it into a small student.
- **Screens.** Seeds 20261210 (P features) and 20261211 (V pilot).

**5XL V** (V_CAPACITY_5XL, T0016). v3 V sizes: XL 177,411, 4XL 5,089,283, 5XL 18,567,171 parameters.
- **Validation log loss:** 5XL − XL is −0.00158 [−0.00272, −0.00044] over 31 map groups: lower.
- **Selection regret:** +0.0083 [−0.044, +0.060] over 8 groups: not shown.
- **Bounded screens** (4 map groups): every comparison not shown.
- **Cost.** One 5XL V call takes 1.44 ms alone and 2.91 ms at p99 with 15 copies running. The production reference stays `inference/models_pxxl_v3xl`.

**V pilot reports** (`reports/models/v_pilot/`). V3-XL test log loss is 0.543. Its selection regret is 0.087 [0.012, 0.162], against 0.097 for the on-policy choice and 0.130 for uniform. That is over 15 roots: 37 resolved and 53 uncertain pairs.

**Mixed-precision P** (P_MIXED_PRECISION, T0030; code only).
- **Three profiles.** reference16 is the legacy INT16 export. uniform8 is 8-bit everywhere. mixed4_8 puts 4-bit weights in the squad head's hidden layers.
- **The format.** 32-column blocks, explicit integer bounds, and a Python emulation that the C++ kernels match exactly (reference, scalar, SSE2, AVX2, NEON).
- **The training side:** activation-aware calibration, a distillation and QAT interface (CE plus teacher KL, 50/50 to start), and precision-aware cache keys with low-precision screening and high-precision verification.
- **Off by default.** Everything is behind `PONLY_PRECISION` and `PONLY_PRECISION_ROUTE`.
- **Storage** of the XXL P relative to INT16: uniform8 about 0.52, mixed4_8 about 0.42.
- **Not done:** nothing is trained, exported for a carrier, timed or played.

**Decision learning** (RL_PLAN; T0033 plan, revised by T0034, code only: nothing trained, measured, played or exported).
- **Retired, unstarted (2026-10-12):** the V pilot T0012 and its screen T0014. Deployed V stays disabled.
- **The component.** P predicts the opponents and stays frozen. A separate own-action scorer (kind `own_action_scorer`, never exportable as P or V) ranks our own complete orders by their gain over the incumbent.
- **The data.** A versioned root/candidate dataset is built from the recorded prefix. Its builder checks linkage, support, duplicates and map splits. Authority binds an audit receipt's source digests to the exact bytes read, a carrier config to its collection receipts, and logged carrier scores to their orders. T0010 stays exploratory: its re-audit is pending, it has no source digests and no logged carrier scores.
- **The diagnostics.** Noise (within- against between-action), coverage and power are reported without automatic kill rules.
- **The rule.** The deployed rule (z-scored logged carrier score + λ·gain; incumbent override only as a labeled fallback) is frozen on validation and evaluated once on test. Verdicts: benefit, harm, negligible or insufficient precision.
- **The carrier.** The pilot freezes the currently admitted carrier, P2 after T0013, or a later admitted model.
- **Still specified only:** the carrier hook, a 300–500-map host pilot, and the macro, expert-iteration and offline-RL conditions.

## The host: environment, speed, capacity

**Sources:** [reports/host/](../reports/host/)

- **Host.** DESKTOP-G60OJUD: Ryzen 9 7945HX (two 8-core CCDs, 32 MB L3 each), GCC 15.2, RTX 4080 Laptop.
- **Environment check.** Every check passes: bots build and play, protocol, first turn, no timeouts, p99 under 40 ms, simulator exact, record alignment, kernel and integer parity, runtime contract, policy p99 under 5 ms.
- **Training on CUDA** (L tier). Per epoch, P is 3.8× and V 1.9× faster than NumPy. Whole runs are only 1.27× and 1.31× faster.
- **Load.** 16 seats against 1 seat costs about 2× in inference time. XXL needs 1.29 ms with one seat and 2.83 ms with 16, at 15 teams with a 1.5× safety factor, so every tier up to XXL fits the 5 ms budget at every count.
- **SIMD.** At XXL under load, the time fell from 12.02 to 2.83 ms (4.25×): the data layout gave 3.29× and explicit AVX2 kernels 1.29×.
- **The consolidated report.** It recommends M by default and XXL as the ceiling. 3XL (23.0 ms) and 4XL (191.5 ms) fail under load. Its explanation, L3 cache spill once 8 copies exceed 32 MB, is inferred, not measured.

## Opponent roster

**Sources:** [ROSTER.md](ROSTER.md), [FIRST_RUN.md](FIRST_RUN.md), [roster/](../roster/), [FINGERPRINT_EFFECTIVENESS_REVIEW.md](../reports/reviews/FINGERPRINT_EFFECTIVENESS_REVIEW.md), [SAMPLING_DATA_READINESS_REVIEW.md](../reports/reviews/SAMPLING_DATA_READINESS_REVIEW.md)

**Plan and tools.**
- [ROSTER.md](ROSTER.md) covers inventory, probes, behavior distances, strength, assessments, an open training registry with no 15-bot cap, and a separate frozen evaluation roster. [FIRST_RUN.md](FIRST_RUN.md) is the host runbook.
- **The first inventory:** 191 candidates, 122 eligible, 29 lineage groups at 0.7. 130 files are archived (8.5 MiB) under `roster/bots/`.
- **A scan for new bots** found 136 more programs (`roster/new_bots/`).
- **Not built yet:**
  - fingerprint stages 3–5;
  - context-level sampling;
  - response profiles;
  - a continuing-policy declaration for V;
  - validation of the selection itself.

**Host evidence.**
- **Fingerprints.** 258 bots × 84 probes (6 maps × 14 counts) × 2 repeats = 43,344 runs. 253 bots are measured at every count; 5 performance forks crash. Complete linkage gives 180 behavior groups.
- **Shortlist.** 78 families: 32 core, 19 distinct, 27 near.
- **Light strength screen** (seed 20261009). 64 bots × 70 games split 30 above parity, 11 level and 23 below, with no faults. The top is Player5_v12_rollout_volley at 0.770 [0.723, 0.818].
- **The full screen of every eligible bot was not run**; Casper deferred it (T0027).

**Training selection (2026-10-09).**
- **The pool.** An open, behavior-diverse pool of 58 programs: 56 P-label policies plus 2 augmentation programs. It holds the 11 judge bots plus 7 challengers, and it raises the judge bots' expected share of P-label rows from 34% to 56%.
- **Sampling weights.** field 0.7, style 0.3, lineage cap 0.1, augmentation 0.02.
- **Reserved lineages.** Two whole lineages are kept out for evaluation: AI_Arena_CTF_Player02_bot and ctf-arena_Player4_v2.
- **The pilot.** 400 matches (seed 20261019; pool `85f96f37…`) ran clean under official limits.
- **Pilot review.** It supports collecting in stages. It does not show that the data will be enough. Its proposed next collection uses seed 20261029.

**Reviews.**
- **Fingerprint review.** It found that unmeasured counts looked distinct and that A/A and A/B used different populations. It wrote the v2 plan; stages 0–2 are built.
- **Sampling review.** Three of its four P2 findings are fixed. Still open: collection resumes by file name, with no immutable run manifest.

## AI_Test heuristic benchmarks and duels

**Sources:** [benchmark/](benchmark/)

**What helps** (track 1 and the C-series; 1 pp ≈ 8–9 BT):
- tuned weights +1.6 pp;
- sampled two-ply expectimax +1.6 / +2.2 pp;
- Omega against Hybrid +2.6 pp (track 1);
- the RISK gene +2.6 pp against Hybrid, the most consistent gene;
- CFL in duels only: +9.6 pp, winning all 250 duels against Baseline.

**What hurts:**
- FLAGS −6.3 pp: remove it.
- Any habit model in duels: −8.4 and −9.5 pp.
- Player2 v005, searching for 15 ms, beats Omega's 30 ms search by 0.7 pp.
- The genes don't add up: LIT is no better than RISK alone.

**Caveats:**
- Every size had a fixed roster.
- Sizes 7, 12, 13 and 14 were never benchmarked.
- The 15-team timeouts were run in record mode: late replies were still played, so the benchmarks overrate 30 ms search at 15 teams.
- Crowding halves delivery (conversion 0.6 → 0.32 from 5 to 15 teams).

**Duels** (31 bots × 2,000 duels):
- Duel play is deterministic: the 31 bots are 9 behaviors.
- CFL is the best duel behavior: 77.6%, +2.5 points.
- R3 expectimax is the only change that helps against Astra (+6.4).
- Habits help against Player5 (+5.4) and hurt against Astra (−3.0).
- The ceiling is Astra and Sol: the control wins about 46–48% against them.
- This is why REVIEW §3.6 calls duels not saturated. It supersedes BENCHMARK_REVIEW's "duels untouched".

## Hybrid carriers: the first P and V pilots

**Sources:** [bots/hybrid/README.md](../bots/hybrid/README.md), [C08_VS_V12_FOUNDATION_REVIEW_e11c45b.md](../reports/reviews/C08_VS_V12_FOUNDATION_REVIEW_e11c45b.md), [CARRIER_INTEGRATION_REPORT.md](../reports/reviews/CARRIER_INTEGRATION_REPORT.md), [review_bundle_20261009/README.md](../reports/reviews/review_bundle_20261009/README.md), [CURRENT_STATUS_AND_V_ABLATION_REVIEW.md](../reports/reviews/review_bundle_20261009/CURRENT_STATUS_AND_V_ABLATION_REVIEW.md), [tournament_1000_bt_report.md](../reports/eval/tournament_1000_bt_report.md), [v_ablation_report.md](../reports/eval/v_ablation_report.md)

`bots/hybrid/` holds the two carriers, C08 Player5_Omega_LIT and Player5 v12. Each mixes the legacy forecast q_H with P, and optionally adds V. The P-only bots are derived from these headers.

| Run (seed 20261020) | Published claim | What the records support |
|---|---|---|
| P pilot, e11c45b: 100 fixtures per carrier at 2, 5, 8 and 15 teams | v12 M and L both significantly beat the control; L unbeaten in duels (4 / 21 / 0) | M's interval crosses zero. W/T/L count paired utility changes: v12 L's duels were 17 wins, 2 draws, 6 losses. C08 with M has the highest point utility; v12 gains more from P and has far more CPU headroom. No best carrier. |
| V ablation, 0a0badc: 80 fixtures, reusing the P pilot's maps | V "settled conclusively"; under 0.2 ms; no timeouts | Marginal V: +0.0011 for C08, +0.0366 for v12. Both intervals include zero. The timing fields are empty in the records. |
| 1,000-game tournament, 8 counts | C08 P+V L ranks #1 at +36.3 BT; v12 +21.7 over its control | Each candidate's BT interval overlaps its control's, and the deltas carry no paired interval. |

**The review bundle's verdict.** Neither a best carrier nor a V benefit is confirmed. The blockers were:
- C08's branch planning advanced the persistent P history;
- the consumers did not use the distribution the gate approved;
- V was evaluated on unsettled branch states;
- results were not bound to artifact identities or complete timing.

The P-only line fixed these. The three write-ups now open with a correction note (T0024).

## P-only bots

**Sources:** [bots/p_only/](../bots/p_only/), [screen_v3xl_matched/SCREEN.md](../reports/eval/screen_v3xl_matched/SCREEN.md), [test_c08_v3xl/SCREEN.md](../reports/eval/test_c08_v3xl/SCREEN.md), [tournament_50k_report.md](../reports/eval/tournament_50k_report.md), [SEARCH_PORTFOLIO.md](SEARCH_PORTFOLIO.md), [handoffs/p_only_distance_tiers/NEXT_AGENT_PROMPT.md](handoffs/p_only_distance_tiers/NEXT_AGENT_PROMPT.md)

**What changed.**
- C08 and v12 take every opponent prediction from P. The hand-written forecasts, habits and scripted opponent moves are deleted, not switched off.
- C08 searches two plies: `stepExact` settles turn 1.
- V is off unless a variant turns it on.
- A screen runner binds every row to an identity: bundle, binary, models, runner, schedule and limits. It interleaves the arms on the same fixtures, adds A/A arms, and counts improved, unchanged and degraded fixtures (these are not wins).

**50,000-game tournament** ([tournament_50k_report.md](../reports/eval/tournament_50k_report.md), T0028). The run had 21 bots, team counts 2–15 and 16 workers.
- **Ratings.** Plackett-Luce ratings as the run reported them. The fitter, the ordered results and a run manifest are not committed, so the ratings cannot be refitted and carry no intervals.
- **Win rates** (approximate intervals, averaged over counts). C08 P-only beats its control by +3.2 points (XL) and +3.5 (XXL), v12 P-only by +2.7 (both). XL against XXL is not shown on either carrier; why sizes would differ is a hypothesis.
- **Limits.** Late replies were played, not dropped, so the official 50 ms rule was not applied. The report is final for this run and descriptive: no per-game data was kept, so it is not confirmation evidence.

**Phases 0 and 1.**
- **Phase 0.** STRUCTURE_SCAN (generated, 258 bots) and STRUCTURE_ANALYSIS rank v12 first and C08 second as carriers.
- **Timing** (seed 20261101). C08 XXL's worst-game p99 is 31.2 ms. v12 XL and XXL take 8.1 and 10.6 ms.
- **The size rule** therefore picks C08 at XXL, and v12 at both XL and XXL.

**Phase 2 screens.** 280 fixtures each (20 maps × 14 counts); no faults and no late replies.

| Comparison | Seed 20261031 (tuning) | Seed 20261102 (second fresh seed) |
|---|---|---|
| C08 P-only XXL − C08 control | +0.040 [+0.008, +0.071] helps | +0.029 not shown |
| C08 P-only XXL − 2-ply scripted control | +0.019 not shown | +0.026 not shown |
| C08 P-only XXL − hybrid M | +0.034 helps | +0.039 helps |
| v12 P-only XL − v12 control | +0.020 not shown | +0.052 [+0.019, +0.084] helps |
| v12 P-only XXL − v12 control | +0.030 not shown | +0.070 [+0.037, +0.103] helps |
| C08 XXL − v12 XL | +0.020 not shown | +0.016 not shown |
| A/A (identical binary), C08 / v12 | +0.001 / 0.000 | −0.012 / +0.013 |

**Stage 2: V at fixed P** (T0003, seed 20261108).
- Overall V effect: C08 XL +0.010, C08 XXL +0.018, v12 XL +0.012, v12 XXL −0.002. All not shown.
- Locally, C08 XL's V helps at 11–15 teams (+0.037), while C08's P-size × V interaction hurts there (−0.051).

**V3-XL screens.** Both reused seeds.
- `test_c08_v3xl` (seed 20261102, 70 fixtures): +0.029, not shown.
- `screen_v3xl_matched` (seed 20261108, 140 fixtures): +0.004, not shown. Its C08 A/A row came out at −0.036, so re-run noise is as large as the effects these screens look for.
- T0022 repeats the screen on a fresh seed.

**Not run yet.** Stage 1, capacity and sampling: seed 20261107, 2,520 games (T0002).

**Search experiment** (SEARCH_PORTFOLIO, T0018 and T0020). It runs in the family carriers and is off by default.
- **New options.**
  - `PONLY_PORTFOLIO_K` adds alternative action assignments for the best movements.
  - `PONLY_PLY2_MODE` 1 predicts ply 2 from each plan's own successor. Mode 2 plays as mode 0 but also scores every plan with its own prediction, as a diagnostic.
  - Successor views are cached.
- **The audit found:**
  - one action assignment per movement;
  - C08's checked lookahead compared movement only;
  - one shared ply-2 prediction per sample.
- **Fixes.** T0020 fixed the per-branch deadlines and the validity of the regret comparison.
- **Next.** The host evaluation is T0019.

The old `docs/handoffs/p_only_distance_tiers` prompt only points here.

## Archived handoff briefs

**Sources:** [handoffs/README.md](handoffs/README.md), [handoffs/diverse15_distance_tiered_policy/](handoffs/diverse15_distance_tiered_policy/)

The "diverse15" package (reviewed against 34a83b0 through 93fa7e6). Its eight briefs:
- **BOT_SELECTION_PLAN:** an open training pool with no 15-bot cap, and a separately frozen evaluation roster.
- **BOT_GROUP_AND_SAMPLING_REVIEW:** the 29 groups are source lineages, not styles. Sample by measured behavior; use lineage only to control redundancy and to hold out whole lineages.
- **DISTANCE_TIERED_POLICY_PLAN:** larger P for opponents born near us, smaller for distant ones, with tactical overrides; memory-aware inference.
- **ADAPTIVE_COMPUTE_RESEARCH:** spend compute selectively, starting with independent small, base and large models.
- **FIXED_POINT_PROBABILITY_PLAN:** a proposal for integer softmax, value and sampling.
- **CUDA_TRAINING_PLAN:** optional PyTorch training, CPU-only deployment. It is built.
- **EVALUATION_PLAN:** evaluate the actual integer model and adapter. Every integrated version needs at least 10,000 independent map groups, at least 500 per count, and three seeds.
- **SOURCE_BASELINE:** what the package reviewed.

The steps of its execution prompt are triaged below.

## Triage of the pre-board prompts

**Sources:** [roster/report/NEXT_AGENT_PROMPT.md](../roster/report/NEXT_AGENT_PROMPT.md), [review_bundle_20261009/NEXT_AGENT_EXECUTION_PROMPT.md](../reports/reviews/review_bundle_20261009/NEXT_AGENT_EXECUTION_PROMPT.md), [diverse15 NEXT_AGENT_PROMPT.md](handoffs/diverse15_distance_tiered_policy/NEXT_AGENT_PROMPT.md)

Each step of these prompts is done, covered by a task, or deferred by Casper (T0027, 2026-10-11). New work starts from the board, not from these files.

**Roster prompt** (finish the first roster evidence run):

| Step | Status |
|---|---|
| A. Setup and checks; C. commit the evidence | Done: `roster/report/results/` (RUN.md, env check, distances, assessments, admission). |
| B1–B4. Probe suite of 84 probes; fingerprints with 2 repeats; distances | Done for 258 bots, not only 122 (ecc45a0). |
| B5. Strength screen of every eligible bot | Not done: only the 64-bot light screen ran. Deferred (T0027). |
| B6–B7. Live profiles; assessments and admission template | Done; live profiles exist for the light screen's 64 bots. |
| D. Report generated from the files; fix `matches_summary.csv` | Done: `tools/reporting/generate_roster_report.py` with its test. The CSV now has scores and winners. |

**Review-bundle prompt** (repair the P/V integration, then confirm):

| Step | Status |
|---|---|
| 1. History isolation, full-P consumer, settled V states, V off by default, content identities | Done in the P-only bots ([EXPERIMENTS.md](../bots/p_only/EXPERIMENTS.md)). The hybrid carriers were superseded, not repaired. |
| 1.3. Carrier-specific α and temperature on tune data | α is obsolete: P-only has no blend. Temperature: T0026. |
| 2. Trustworthy records: logs, timing, A/A, improved/unchanged/degraded | Done in the P-only screen runner. |
| 2. Correct the original write-ups' claims | Done in T0024: correction notes on the carrier report, the V ablation and the tournament report. |
| 3. Fresh tuning comparison | Done for P (seeds 20261031 and 20261102) and for V at fixed P (T0003). Capacity: T0002. V pilot: T0010–T0014 and T0022. |
| 4. Freeze and confirm | T0006. |

**diverse15 prompt:**

| Step | Status |
|---|---|
| A1–A3, A5–A9. Inventory, aliases, assessments, probes, distances, admission, separate admission and scheduling | Done: `roster/`, [ROSTER.md](ROSTER.md), `training_selection_20261009`. |
| A4. 10,000 map groups per evaluated bot | Not done. Deferred (T0027). |
| A8. Evaluation on the reserved OOD lineages | Lineages reserved, never evaluated: T0025. |
| A10. Validate the selection at equal training budget | Not done. Deferred (T0027). |
| B. Capacity tiers | S–XXL are trained and exported. Distance tiers at the root were dropped: all 14 opponents at XXL cost 1.34 ms per turn ([TASK.md](../bots/p_only/TASK.md)). Inside the search: T0004. |
| C. CPU budget and memory | Whole-turn deadlines are in the P-only bots and `capacity_bench` is reused. The fixed-point kernels: deferred (T0027). |
| D. CUDA training | Done: the torch backend, measured in the training bench. |
| E. Confirm and deliver | T0006, with T0025 and T0026. |

## Seeds

Never reuse a seed for a new comparison. Pick one that is not in this table, and add it here and to the task.

| Seed | Used by |
|---|---|
| 20261501 | T0037 Colab stage-1 decision collection: 306 maps, 2–10 teams; training-only, consumed; branches pending |
| 20261009 | Roster light strength screen |
| 20261019 | Recommended-pool pilot (400 matches) |
| 20261020 | Carrier P pilot, V ablation, 1,000-game tournament |
| 20261029 | Proposed training collection (training_selection_20261009) |
| 20261031 | P-only phase 2 screen (tuning) |
| 20261101 | P-only timing run |
| 20261102 | P-only second screen (`screen_ood_20261102`); `test_c08_v3xl` |
| 20261107 | Stage 1, capacity and sampling: reserved for T0002 |
| 20261108 | Stage 2, V at fixed P (T0003); `screen_v3xl_matched` |
| 20261201 | V training data from the frozen carrier (training only) |
| 20261210 | P feature screen (T0013) |
| 20261211 | V pilot screen (T0014) |
| 20261220 | T0016's 5XL screens |
| 20261230 | Search experiment on the host (T0019) |
| 20261240 | Fresh-map V3-XL screen (T0022) |
| (not recorded) | 50,000-game tournament (T0028): its seeds are not committed |
| 20261301–20261364 | 64 showcase games of the newest P-only bots on the Mac (2026-10-11), 2–15 teams; nine are in the replay viewer. Not evidence |
| (unchosen) | Confirmation (T0006): chosen only after the finalists are frozen |

## Known defects in the records

These records are kept as they are. Fix them through their generator or owner task.

- **[inference/models/README.md](../inference/models/README.md)** still says nothing is exported. Tiers S–XXL are exported under `inference/models*/`.
- **[training/specs/DATA_FORMAT.md](../training/specs/DATA_FORMAT.md)** gives the shard record as `(R, 644)`, which is layout 2. The extractor writes layout 3, 923 values wide ([FEATURES.md](../training/specs/FEATURES.md)).
- **Broken artifact links.** [FINGERPRINT_BEHAVIOR_REPORT.md](../roster/report/FINGERPRINT_BEHAVIOR_REPORT.md) links its protocol through a `file:///c:/Users/...` path on the host. The V ablation report links its artifacts as `file:///reports/...`. Neither link resolves.
- **The V ablation report has a non-English phrase in its objective.** Repository files are English only.
- **`docs/benchmark/BENCHMARK_REVIEW.md`** concludes "duels untouched". The later DUEL_REVIEW and REVIEW §3.6 reverse that.

## Open questions

- **Deferred by Casper (T0027, 2026-10-11):** the full roster strength screen, 10,000-map confirmations per opponent bot, validating the opponent selection at equal training budget, and the fixed-point kernels. Reopen one with a new task when a result needs it.
- **For Casper, in T0001's log:** drop T0001? Its move check can no longer pass, and T0002 waits on it.
- **The official field.** Are the judge bots plus other submissions the real opponents? P is only as good as the overlap between the training pool and that field (REVIEW §6).
- **Whether V helps at all.** No V configuration has shown a benefit so far (T0003; T0022's fresh-map V3-XL was not shown). The V pilots T0012 and T0014 were retired unstarted on 2026-10-12; deployed V stays off, and decision learning (RL_PLAN) is the conditional route.

## T0037 Colab stage-1 collection

**Sources:** [stage-1 report](../bots/p_only/report/decision_collection/colab_20261012/RESULTS.md).

306 whole matches (34 maps each at 2–10 teams), all with usable roots, are backed up and verified. Two retained faults occurred in one 9-team fixture. Counts 11–15 failed the bounded Colab load gate and remain uncollected. Colab results are not host timing or strength evidence. Branch continuations wait for T0040 / PR #36; candidate/repeat labels, noise and learnability remain unmeasured. This partial report does not close T0037 or unlock T0038. Seed 20261501 is consumed.
