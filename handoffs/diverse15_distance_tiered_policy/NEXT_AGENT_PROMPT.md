# Execution prompt for the ctf-fusion implementation agent

Implement the following project in the current ctf-fusion and AI_Test repositories. This is an execution request, not a request for another architecture essay. Evaluate the available bots, choose exactly 15 useful and behaviorally distinct data opponents, implement distance-tiered opponent prediction and memory-aware CPU inference, and add an optional CUDA training backend. Keep every production bot CPU-only. Read this folder's plans and evaluation tools before editing.

## Constraints and baseline

- Start from the actual current repository, and record its commit. This package was synchronized with ctf-fusion `93fa7e604ca9a05e48b8e04d787dcb387401fba8`, following the contract fixes at `7bbeeb0`.
- Preserve the recently fixed binary-output, declared-adapter, integer-artifact gate, feature-v2, map/family split, bounded-buffer and simulator-coverage contracts. Do not reintroduce a float-only gate or marginal/joint mismatch.
- Competition limits: first input/reply including startup within 2 seconds; each later complete input/reply within 50 milliseconds. RAM is generous. No GPU, file access, networking or subprocesses inside the submitted bot. GPU training is an offline option only.
- Support every team count 2–15. Keep the existing dedicated counts 2,3,4,5,6,7,8,10,12,15 and general fallback for 9,11,13,14.
- All new code, prompts, documentation, comments and reports must be English.
- Perform expensive matches, training and authoritative latency checks on the designated host. Do not substitute development-machine timings for target-host evidence. Do not install or run on an unapproved external/cloud GPU service.

## Phase A: evaluate every candidate, then select the 15

1. Inventory all newly written bots as well as official and previous controls. Use `tools/inventory_bots.py` as a starting point and expand its directories to include every new export. Do not restrict discovery to the old `profiles/pool.json`. Exclude test drivers and broken builds, but retain an explicit reason for each exclusion.
2. Identify byte aliases, compiled-policy aliases, shared strategy lineage and behavior equivalence **per team count**. Similar names or a shared duel path do not establish equivalence at 3–15 teams. Use source/dependency/compiler hashes and fixed-stream fingerprints.
3. Give each candidate an assessment: protocol reliability, startup and reply timing, strength per team count, strengths/weaknesses, distinctive mechanisms, joint movement/action behavior, and evidence confidence. Static inspection is mechanism evidence, not a strength result.
4. Every distinct evaluated bot needs at least **10,000 independent map groups** in its confirmation record, using a common frozen representative opponent roster and covering 2–15 teams. Stage cheap screening first; aliases may point to a genuinely equivalent representative's evidence rather than pretending to have independent runs. Keep map/seat/flag-seed clusters and do not count 400 turns as 400 matches. A source alias with a different compilation contract needs separate checking.
5. Freeze a public-stream probe suite containing quiet movement, collisions, focus fire, overkill avoidance, pickup/drop/flagshoot, escort, interception, respawn, return-risk and endgame cases. Include live trajectories and legal alternate-state probes. Bot processes must receive a valid historical stream; an arbitrary isolated state is not a fair test of a persistent bot.
6. Build a behavior distance matrix over requested joint moves, conditional actions, attack targets and role changes. Give rare tactical events their own coverage score; silence on quiet states must not dominate the fingerprint. Repeat clock-dependent probes and measure A/A noise.
7. Select exactly 15, balancing credible strength, behavior coverage and demonstrated unique skills. Use `tools/select_roster.py` as a transparent heuristic, then review its result. The same behavior cluster cannot occupy two slots. Family quotas and distance thresholds are pre-registered, not tuned on the final confirmation set. A uniquely useful specialist may be weaker overall, but record its actual marginal value instead of filling a slot with a random or broken bot.
8. Produce `bot_inventory.json`, `all_bot_assessments.csv`, `behavior_distance.json`, `selected15.json`, `selection_report.md` and a source-hash-locked collection pool. Each selected bot must have one sentence saying which behavior it contributes that the other 14 do not adequately cover. If the search cannot find 15 qualified distinct behaviors, report the shortage and missing evidence honestly; do not invent results or rename near-duplicates.
9. Validate the selection: compare a 15-strongest pool, the selected diverse-15 pool and a broader pool at the same sample/training budget on unseen opponent families. Do not claim that diversity improves prediction solely from the selection objective.

## Phase B: train and export capacity tiers

Follow `DISTANCE_TIERED_POLICY_PLAN.md`. Opponents born nearer our initial spawn region receive a larger P network by default; more distant opponents receive a smaller one. Dynamic contact, carriers and imminent scoring override that prior. Capacity means network architecture, not an arbitrary increase in alpha or probability sharpness.

Read `ADAPTIVE_COMPUTE_RESEARCH.md` before implementing cascades or shared-width networks. The two-level router uses spawn proximity first, then estimated tactical decision-risk reduction per admitted CPU cost. Do not promote solely from entropy or always pay for both small and large inference. Independent tiers are the reference; early exit/slimmable variants need their own measured parity, quality and end-to-end budget tests.

Add small/base/large architectures behind explicit configuration. Start with a general tier family before expanding dedicated models. Each capacity has its own calibration, gate, integer identity and fallback. Three capacities across 11 logical P models plus 11 V models means **44 network instances**, not the original 22; document the actual artifact count and RAM/source size.

Train on the selected 15 with lineage-balanced sampling. Treat P as opponent prediction, not imitation of our policy. Define the continuing policy for V labels and keep exploratory/random episodes separate from the main target distribution. Prevent whole-map, whole-fixture and held-out-lineage leakage.

## Phase C: implement the CPU budget and memory plan

Precompute static map geometry during the 2-second startup window. Share a root turn view and feature primitives, cache exact feature/weight computations, allocate bounded scratch/cache storage ahead of turns, and use measured per-tier costs to schedule inference. Do not reuse a root forecast for changed search branches unless explicitly modeled as a frozen-forecast approximation. Avoid full product lookup tables unless end-to-end measurements beat vectorized integer multiplies.

Use a whole-turn deadline, reserve output time, and admit expensive inference only when its measured cost fits. If a tier cannot fit, choose a **separately evaluated** smaller tier or the legacy predictor. Complete a valid action on time. Memory must improve measured useful search work, not merely increase cache capacity.

Reuse the newly added `tools/capacity_bench.py` rather than writing a duplicate capacity profiler. On a development machine use correctness-only `--verify`; run actual timings on the target host. Supply custom P tier widths with V held fixed when isolating P cost. Its synthetic-weight/native workload establishes capacity cost, not trained strength or a complete integrated-bot deadline.

## Phase D: CUDA training, CPU deployment

Follow `CUDA_TRAINING_PLAN.md`. Implement a parity-tested optional PyTorch backend preserving the current labels, masks, clipped activation, objective, optimizer semantics and export format. Float32 is the first reference. Pre-extract C++ features, cache batches in RAM/VRAM where practical, benchmark transfers and optimizer steps separately, and enable AMP only when it improves real throughput without losing acceptable quality. Measure the overall pipeline bottleneck, not just GPU matrix throughput. No CUDA dependency may enter the generated bot.

## Phase E: confirm and deliver

Use `evaluation/EVALUATION_PLAN.md`, `scorecard.py` and the existing artifact gate. Evaluate the **actual integer model and actual selected adapter**, including the distance-tier scheduler and fallbacks. Fit temperatures/alpha on tune, freeze them, and confirm on new data.

Ablate: legacy; fixed-small P; fixed-large P; spawn-distance tiers; distance+dynamic tiers; the same tiers with caches; CUDA versus NumPy training parity. Hold search budget, candidates, RNG/fixtures and opponent roster fixed where applicable. Then separately measure useful search expansion under the real 50ms budget.

For each integrated version, retain at least 10,000 independent confirmation map groups, report every team count and three independently trained seeds for the final configuration. Use paired map-cluster intervals and a new final set after tuning. Ordinary repeated 95% intervals are not a valid sequential stopping rule. Missing or inconclusive evidence leaves the corresponding size/tier disabled.

Required final deliverables: source changes and focused tests; frozen 15-bot registry; acquisition/evaluation manifests; per-tier integer weights and identities when trained; actual target-host complete-turn latency/fault reports; offline and paired-play results; representative single-file C++ exports; and an English explanation of what improved and what remains unverified. Commit only relevant changes. Do not claim completion from a plan, synthetic test or estimated MAC count.

## Delegation seams

If subagents are available, give separate ownership to (1) inventory/behavior probes and roster evidence, (2) CUDA trainer parity and export, (3) tier scheduling/cache/runtime tests, and (4) evaluation and leakage audit. Keep one coordinator responsible for the common schema, frozen fixtures, artifact identity and final integration. Agents may suggest code, but only measured evidence can enable a deployment tier.
