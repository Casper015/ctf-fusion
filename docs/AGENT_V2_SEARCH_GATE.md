# agent_v2 search override gate (T0076)

The bounded two-ply search of agent_v2 (`inference/agent_v2/search.hpp`) may replace the actor's joint only through an override gate. This file freezes the rule for choosing the gate's threshold tau and the tree breadth. **Section 1 was committed before any diagnostic margin was computed**; sections 2 and later record what the frozen rule produced. No game was played for this task, and nothing here measures time or strength.

Background: on the meow_rl carrier, search overrode the actor on 73% of roots, 82% of them with a top-two margin under 0.01 ([DIAGNOSTICS.md](../reports/host/meow_rl_aligned/diagnostics/DIAGNOSTICS.md)), and lost 3.9–4.7 pp to direct play ([VERDICTS.md](../reports/host/meow_rl_aligned/VERDICTS.md)).

## The gate

`gated_choice(mean, n, tau)`: `mean[0]` is the actor's joint, `mean[i]` root i's mean value over the same completed scenarios. The best root (the first with the highest mean) replaces the actor only when `mean[best] - mean[0] > tau`.

- A tie, a margin at or below tau, or a NaN keeps the actor's joint.
- Incomplete work (any admission refusal, a late tree) keeps the actor's joint and its complete order, resolved before the search started.
- tau = 0 is the ungated choice, identical to the search before T0076.
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
