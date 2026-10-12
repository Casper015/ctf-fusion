# P3 marginal information: existing strata, cost and a new branch swap

The proposed far-heavy prediction-gain explanation is not supported by the saved T0011 strata. P3 has clear near/contact/pursuit/retreat joint-loss gains and no shown distance-9+ joint gain. The new fixed-work C08 branch intervention changes many more near replies than far replies. C08 responds to P3 marginal information; this does not establish better decisions or playing strength. Deadline cost offset remains plausible, and the bounded T0013 strength screen remains inconclusive.

## Saved test-set prediction strata

These are existing T0011 seed1 artifact results, read without new fitting or scoring. Negative P3 minus P2 joint NLL is better. Seeds2/3 context is retained in EXISTING_EVIDENCE.json. Distance is nearest-enemy Manhattan at labeled roots, not the branch BFS distance used in the swap. Overlapping behavioral strata cannot be added.

| Stratum | P3 minus P2 joint NLL | 95% map interval |
|---|---:|---|
| all | -0.088042 | [-0.089199, -0.086885] |
| dist 0-2 | -0.109396 | [-0.111192, -0.107600] |
| dist 3-4 | -0.116492 | [-0.118051, -0.114934] |
| dist 5-8 | -0.018546 | [-0.019843, -0.017249] |
| dist 9+ | +0.001676 | [-0.000095, +0.003447] |
| contact | -0.112706 | [-0.114106, -0.111307] |
| pursuit | -0.129090 | [-0.131032, -0.127148] |
| retreat | -0.137626 | [-0.139699, -0.135553] |
| own carrier | -0.135380 | [-0.137637, -0.133124] |
| carrier shot | -0.114506 | [-0.116654, -0.112358] |

Do not transfer the S-to-L/XXL capacity result from T0004 to the P2-to-P3 feature-family intervention. They are different comparisons on different data, with different distance definitions and root model dispatch. An offline loss reduction does not mathematically guarantee a changed argmax, chosen order or positive utility.

## Existing deployment cost and realized work

C08 P3 has 276.0 samples/call versus P2 360.1, a 23.35% lower count, with requested count 512 unchanged. Truncated-call shares are 0.8384 versus 0.6996. This is descriptive of paired games: changed forecasts/actions/trajectories can change the work, so the entire 23.35% cannot be attributed causally to feature extraction alone. NLL, sample count and utility have different units and are not an additive net-effect equation. Actual same-stream cost probes support that P3 inference is more expensive.

v12 has measured cost data, not a complete work-cut trace:

| v12 metric | P2 | P3 |
|---|---:|---:|
| Median of per-game p99 reply ms | 5.417 | 7.257 |
| Worst game p99 reply ms | 10.896 | 14.765 |
| Maximum successful post-first reply ms | 12.215 | 16.843 |
| Views per turn | 7.66 | 7.68 |
| Team encodes per turn | 57.26 | 57.29 |
| Act heads per turn | 246.73 | 247.31 |
| Target heads per turn | 83.78 | 86.26 |

The full replies are slower, but recorded view/encode counts do not show a C08-like reduction. v12 has a 24 ms refinement guard; completed alternatives/scenarios and guard-cut counts are not separately logged by this T13 build. Thus its exact deadline tax remains unmeasured. Its more positive point estimate is not evidence for a different carrier effect: the paired interaction CI includes zero.

## New zero-game fixed-work intervention

Use two lexicographically first retained c08_pf_p2xxl matches per count3–15 from T13: 26 maps, all400 observed turns, 10400 reference states per mode. No engine match is advanced or re-played; only retained seat input streams are fed to native diagnostic copies. Common fixed128 samples per call, Voff, deadlineoff. Keep P2 root predictions/history, original scoring/depth/random keys and wanted-team mask fixed. Change only the model family of already-requested active opponent teams in branch Views. Pin that family per View/team for joint, mover, action and target caches. P3 gets the actual unchanged exported seed1 weights, proper combat encoder and INIT-derived rules.

Team distance is the minimum wall-aware BFS distance from active own units to active units of that opponent team: near0–3, far7–finite, with middle4–6. A team classification does not select every individual opponent-label row by the same distance as T11. The full denominator includes turns with no eligible intervention, rather than claiming a turn-conditional change rate.

| Mode versus original P2 | Changed replies / 10400 states | P3-routed team encodes |
|---|---:|---:|
| disabled | 0 | 0 |
| branch_p2 | 0 | 0 |
| far_p3 | 20 | 378424 |
| near_p3 | 465 | 970844 |
| all_p3 | 601 | 2656696 |

All modes have zero root movement-digest mismatches and zero truncation. Both disabled and branch-P2 controls match original full replies/root digests/search samples/calls across all26 maps. Direct versus DLL-loaded P3 modules match29344 integer outputs and scaling exponents across14 count dispatches, four heads and16 synthetic input vectors per head. This is module/ABI parity, not a new loss evaluation or an exhaustive real-state raw/confidence parity check. The actual P3 game carrier had already passed T11 real-weight parity and its T13 binary hash was identical.

The separate independent checker re-read all156 mode/map receipts and stdout/stderr hashes, binaries, sources and selected inputs; recomputed reply changes and route counts; and confirmed control/search-work parity and fixed128/truncated0 counters. Root digests cover move marginals, not a separately dumped four-head tensor. See INDEPENDENT_CHECK.json and frozen/driver receipts.

These are C08 branch-only fixed-work sensitivity results, not the full P3 deployment (which also changes the root), not v12 sensitivity, and not counterfactual outcome/strength or latency. Different decisions can be better or worse. Removing deadlines holds per-call requested work fixed, but different decisions can take different total call paths. Do not compare 20/465 numerically to T4's 14/385 as a controlled difference: models, states and data differ.

## Structural and next-step interpretation

v12's simulated own-seat next move uses heuristic goal/move scoring and ctfcombat::choose; C08 uses a separate duel path at two teams. Those are real architectural constraints, but do not prove broken prediction use, a search ceiling or a causal explanation of this screen. T19 confidence intervals containing zero do not establish a ceiling.

T32 student/QAT variants could test whether cheaper deployment preserves useful prediction, after exact checkpoint/export/arithmetic identity and same-host cost/parity are frozen. Mixed4_8 speed is not assumed, and removing this cost does not guarantee a strength gain. The offline 8XL teacher is not deployed.

RL_PLAN arm1B is a separate own-seat preference reranker initialized from P3, scored over the carrier shortlist; it is not an opponent-P export or a direct measured rescue of opponent prediction gains. Attention/search heuristics and caching can also be changed behind controlled switches. An actor is not the sole logically possible remedy. No actor/RL training, T32 mutation, model export, production switch, new game or new seed was performed here.
