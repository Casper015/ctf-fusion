# agent_v2 search override gate (T0076)

The bounded two-ply search of agent_v2 (`inference/agent_v2/search.hpp`) may replace the actor's joint only through an override gate. This file freezes the rule for choosing the gate's threshold tau and the tree breadth. **Section 1 was committed before any diagnostic margin was computed**; sections 2 and later record what the frozen rule produced. No game was played for this task, and nothing here measures time or strength.

Background: on the meow_rl carrier, search overrode the actor on 73% of roots, 82% of them with a top-two margin under 0.01 ([DIAGNOSTICS.md](../reports/host/meow_rl_aligned/diagnostics/DIAGNOSTICS.md)), and lost 3.9–4.7 pp to direct play ([VERDICTS.md](../reports/host/meow_rl_aligned/VERDICTS.md)).

## The gate

`gated_choice(mean, n, tau)`: `mean[0]` is the actor's joint, `mean[i]` root i's mean value over the same completed scenarios. The best root (the first with the highest mean) replaces the actor only when `mean[best] - mean[0] > tau`.

- A tie, a margin at or below tau, or a non-finite mean anywhere in the tree keeps the actor's joint.
- Incomplete work (any admission refusal, a late tree) keeps the actor's joint and its complete order, resolved before the search started.
- tau = 0 is the ungated choice, identical to the search before T0076 whenever every mean is finite.
- Build setting: `AGENT_V2_SEARCH_TAU` (default 0). The candidate count and scenario count stay `AGENT_V2_SEARCH_CANDIDATES` / `AGENT_V2_SEARCH_SAMPLES`.

## 1. The frozen rule (pre-registered)

### States

- **Source:** the three recorded seat streams `bots/p_only/tests/fixtures/stream_n02/n04/n15.txt.gz` (one game each; no other recorded state corpus exists offline).
- **Every seat:** each game is replayed once from every seat (the INIT `me` field rewritten, the TURN blocks unchanged), so the states are 2 + 4 + 15 = 21 seat streams.
- **Turns:** every 4th turn (4, 8, …, 400), 100 per seat stream, 2,100 states.
- **Game-disjoint splits:** selection = n02 and n15 (17 seat streams, 1,700 states); validation = n04 (4 seat streams, 400 states). tau and breadth are chosen on selection only.

### Trees

- **Configurations:** N (narrow) = 4 candidates × 2 scenarios, the current default; W (wide) = 8 candidates × 4 scenarios. Everything else is the deployed search: the existing actor, the FP32 critic of `inference/meow_rl/aligned/p2_large`, P2 XXL (`FUSION_TIER_PXXL_VM`), 4 branch scenarios.
- **Roots:** the deployed root list (the actor's sampled joint, then the next feasible joints by accumulator). N's roots are the first four of W's.
- **Repeats:** r = 0..3, each an independent draw of the scenarios: `setup(..., salt_r)` with `salt_r = splitmix64(20262120 + r)`. Within a repeat, N uses scenarios 0–1 and W scenarios 0–3 of the same draw, so N and W are paired.
- **Reference:** one more independent draw, `salt_ref = splitmix64(20262120 + 1000)`, with 16 scenarios for each of W's roots.
- **Values:** a terminal leaf's reward 2U − 1, otherwise the FP32 critic of the leaf; each non-terminal leaf also gets a float64 forward pass of the same weights.

### Quantities, per configuration c and split

- `m_{c,r}(i)`: root i's mean over c's scenarios in repeat r.
- `pick_{c,r}(t) = gated_choice(m_{c,r}, n_c, t)`.
- `d_ref(i)`: the reference mean of V(i) − V(actor) over the 16 reference scenarios.
- Overrides `O(t)`: the (state, repeat) pairs with `pick != 0`; `n_over(t)` = their number; the override rate = `n_over(t)` / (4 × states).
- Confirmation `C(t)`: the share of `O(t)` with `d_ref(pick) > 0`.
- Gain `G(t)`: the mean over every (state, repeat) of `d_ref(pick)` (0 where the actor is kept). It is the reference-estimated actor-relative margin of the gated choice, in V units. It is not a win rate.
- Backend error `eps`: the largest |FP32 − float64| over every non-terminal leaf of the selection split. A margin is a difference of two means, so its backend error is at most `2 eps`.
- Intervals: 2,000 bootstrap resamples of states (each with all its repeats); percentile 95% intervals. The bootstrap stream is numpy PCG64 seeded with the first 8 bytes (little-endian) of SHA-256(`"20262120|bootstrap"`).

### Choosing tau and breadth (selection split)

1. **Grid:** t ∈ {0, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2}.
2. **Eligible t for configuration c:** `t >= 2 eps`, `n_over(t) >= 30`, and the lower end of C(t)'s 95% interval `>= 0.75`.
3. **tau_c** = the smallest eligible t. A configuration without an eligible t is unsupported.
4. **Breadth:** the supported configuration with the larger `G(tau_c)`; a difference under 0.0001, or W unsupported, picks N (less work).
5. **Stop** (blocked handoff with the receipts, no host screen) when neither configuration is supported.

### Validation (n04, frozen breadth and tau, looked at once)

- **Pass:** `n_over >= 10`, the lower end of C's 95% interval `> 0.5`, and `G > 0`.
- **Fail:** stop. Nothing is retuned on validation.

### Integrity stops

- **Parity:** for selected states, the diagnostic tree's values must equal, bit for bit, those of a build at the frozen breadth (`check` compiled with that candidate and scenario count) for the same salt.
- **Isolation:** `check isolation` at the frozen breadth on all three streams: search never changes the root prediction, the history or the default order.
- **Simulator:** `check step` parity with the engine is covered by `tests/test_agent_v2.py` where an arena checkout exists; it is not re-run here without one.

### Variants delivered

- **Gated:** the frozen breadth and `AGENT_V2_SEARCH_TAU = tau`.
- **Ungated:** the same breadth, tau = 0.
- Both use the actor, critic, P and admission settings of `agent_v2_p2xxl_search_fp32`.

## 2. What the frozen rule produced

Receipts: [DIAGNOSTICS.md](../bots/agent_v2/search_gate/diagnostics/DIAGNOSTICS.md) (generated; every table), `DIAGNOSTICS.json`, `INTEGRITY.json` and `TREES.jsonl.gz` (2,100 states, sha256 `57edea1a…`) in the same folder. Produced at commit `01a6abe` with the commands in section 4.

**Frozen: breadth W (8 candidates × 4 scenarios), tau = 0.002.**

| Step | Result |
|---|---|
| Backend error | eps = 2.9e-7 (FP32 against float64), so 2 eps = 5.8e-7 does not bind |
| tau_N (selection) | 0.010: the first grid value with C's lower bound ≥ 0.75 (C = 0.798 [0.775, 0.821], 1,880 overrides) |
| tau_W (selection) | 0.002: C = 0.783 [0.766, 0.800], 4,630 overrides, G = 0.00478 [0.00430, 0.00529] |
| Breadth | W: G(tau_W) − G(tau_N) = 0.00478 − 0.00232 = +0.0025, above the 0.0001 tie |
| Validation (n04, looked at once) | **pass**: 1,088 overrides, C = 0.754 [0.718, 0.788] (needs > 0.5), G = 0.00366 (needs > 0) |
| Integrity | `check gate` 42 cases; tree-prefix parity of the 8×4 and 4×2 builds with the diagnostic build, bit for bit, on 10 states per game; `check isolation` at both breadths on all three streams |

The delivered variants:
- `agent_v2_p2xxl_gw8x4_fp32` (gated, T0079 arm GW): tau 0.002;
- `agent_v2_p2xxl_uw8x4_fp32` (ungated, arm UW): tau 0.

Both are listed in [bots/agent_v2/search_gate/variants.json](../bots/agent_v2/search_gate/variants.json), which includes the agent_v2 variants, so one build gives T0079's D, GW and UW.

## 3. How to read it

**Numerical stability (what this measures).**
- The values are reproducible to the bit, and the backend contributes nothing measurable (2.9e-7 against margins of 1e-3 and up).
- Single repeats are unstable at the level of the exact pick. Two independent 4-scenario repeats of W choose the same root only 42% of the time.
- The direction is more stable than the pick. Under an independent 16-scenario reference, W's override challenger beats the actor 75% of the time at tau 0, 88% at tau 0.01 and 94% at tau 0.02 (selection). Larger margins are more reliable, as expected.
- Narrow trees are less stable. N at tau 0 has C = 0.70, against 0.75 for W.

**The gate is mild.**
- At tau 0.002, W still overrides on 68% of (state, repeat) pairs, against 78% ungated.
- The reference-estimated gain G is highest at the smallest tau and falls as tau grows. Under the search's own model, overrides on small margins are mostly right, not noise.
- So T0079's GW and UW arms will differ only on the roughly 10% of trees whose margin falls between 0 and 0.002.

**Winning (what this does not measure).**
- The 16-scenario reference uses the same simulator, opponent model P and critic as the tree. "Confirmed" means self-consistent under the search's model, not correct about the real game.
- T0064's deficit is not explained by this. These numbers rule out backend arithmetic and show that overrides are not mainly scenario noise, but they cannot test hypotheses 2–4 of [CONCLUSION.md](../reports/host/meow_rl_aligned/CONCLUSION.md): simulator fidelity, opponent-model mismatch, and V on rollout states.
- Nothing here predicts whether GW or UW beats direct play. That is T0079's question.
- The corpus is three games (one per team count 2, 4, 15) seen from every seat. The seats of a game share one trajectory, so the effective sample is smaller than 2,100 states. The bootstrap resamples states, not games, and its intervals are optimistic about between-game variation.

**Completed work on the host is the open risk.** The gate only acts on completed trees: incomplete work plays the actor's order. Two parts of the admission arithmetic matter here; neither changed in T0076.
- *No trajectory starts after 16.5 ms.* The variants keep `agent_v2_p2xxl_search_fp32`'s settings: deadline 30 ms, trajectory prior 8 ms, safety 150%, reply reserve 1.5 ms. An admitted trajectory needs `max(8 ms, worst seen) × 1.5 + 1.5 ms` before the deadline. So 32 trajectories plus setup must fit in about 16.5 ms, roughly 0.5 ms per trajectory.
- *The worst-seen estimate never decays.* After one trajectory costs about 16 ms or more (for example a cold first critic call), setup is refused on every later turn of the game.
- *On this Mac (not evidence, a mechanism check only):* both effects show. The existing 4×2 FP32 variant completed 1 tree in 400 turns of n04. GW completed 150 of 400 trees on n02 and 1 of 400 on n15. UW completed 100 on n02 and 17 on n15.
- *What T0079 must check:* host costs are unknown. Its preflight has to report completed trees (`search_used`) per arm and count, not only reply times. If few trees complete, GW and UW play the direct actor's order and the screen compares D with D.

## 4. Reproduce

```bash
python3 tools/screens/search_gate_offline/run.py --jobs 8     # about 40 min on 8 Mac cores; trees, parity, isolation
python3 tools/screens/search_gate_offline/analyze.py          # applies section 1, writes DIAGNOSTICS.json/.md
python -m unittest tests.test_agent_v2_search_gate            # gate and fallback contracts, the frozen variants
```
