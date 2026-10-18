---
id: T0087
title: "Prototype staged search with usable completed work and matched controls"
status: active
owner: claude-fervent@cloud
needs: []
writes: [inference/agent_v2/search.hpp, bots/agent_v2/staged_search/, tools/search_diagnostics/staged/]
depends: [T0086, T0082]
land: pr
branch: claude-fervent/t0087-prototype-staged-search-with-usable
updated: 2026-10-18
---

## Goal

Conditional code-only intervention, selected from the diagnostic pilot. If mechanics defects or failed evidence integrity block it, close with a documented defer decision rather than quietly implementing around them. Keep P, actor, critic, resolver coefficients, features, tau and outcome objective fixed. Do not simultaneously add shaping, a policy-prior penalty, a learned ranker, risk weights or new candidate generation.

Prototype an opt-in search that retains decisions from completed comparable rounds: baseline plus frozen candidate set, same scenario IDs and equal completed samples per compared action, immutable checkpoint after each full round, explicit fallback before any round completes. Begin with a small predeclared candidate subset including the direct order; expand or refine only under a frozen schedule. Do not compare partially evaluated candidates, select on unequal sample counts, or count incomplete work as a completed tree. Separate one-ply screening/two-ply refinement only if the score contract and depth-comparable stage boundary are specified; keep this separate from simply retaining complete two-ply rounds.

Instrument per-count completed rounds, effective candidate/scenario counts, overrides, baseline retention, discarded work and admission causes. Cache only pure quantities under complete state/parent/model/rules keys; prove branch isolation. Any cache or admission-cost-estimator changes are separate switches/controls, not bundled causal claims. Keep a conservative reply reserve; returning an earlier result cannot undo time already overspent. No claim of a hard real-time guarantee from sampled latency.

Supply deterministic fixed-work mode to isolate arithmetic/selection changes from timing effects and equal-time opt-in variants for the host. Test forced interruptions at every stage: before first round -> exact complete fallback; after a round -> last valid comparable decision; fixed-work repeats -> identical decisions. Include NaNs, refused predictions, ties and nondefault rules. Use local tests within the new directory to avoid reserved test lanes; amend lanes explicitly for any additional shared CI registration. Existing production variants remain behaviorally unchanged by default.

Shared execution contract: this continues an architecture redesign, not a legacy-behavior refactor. Keep official rules exact; do not require whole-bot parity with Meowmind. Preserve agent_v2_p2xxl_direct as production default. P (opponent prediction), pi (our actor), and V/advantage scorer are distinct components. Follow AGENTS.md, claim with your own identity, respect active lanes (especially T0082/PR69 and T0051), and use isolated worktrees. All new timing, collection and strength evidence comes from desktop-g60ojud with exactly eight game workers under the CPU lease, GPU training stopped, 2000/50 ms complete replies. GPU training uses the GPU lease and is scheduled separately. Pre-register fresh map/flag/actor/scenario seeds and check derived-seed collisions; never replay consumed seeds as new evidence. Offline reanalysis of existing immutable artifacts is allowed. Freeze source, engine, model, runner, binary and profile hashes. Missing evidence is unavailable, never zero or a successful check. Do not extend budgets after inspecting outcomes. No default promotion in these tasks; no new giant P/V, CNN, count specialists, or full learned attack policy. All repository prose is English.


## Done when

A reviewed opt-in implementation and frozen variants have complete interruption/cache/branch tests, source-bound controls and a host preflight runbook; no strength/timing claim from the development machine. Or a reasoned defer record if its upstream diagnostic gate fails. Record a decision so final screening does not wait forever on an intentionally skipped arm.

## Handoff

Not started.

## Log

- 2026-10-18 03:36Z codex-search-plan@caspers-macbook: created
- 2026-10-18 17:39Z claude-fervent@cloud: claimed
- 2026-10-18 18:04Z claude-fervent@cloud: note: PR https://github.com/Casper015/ctf-fusion/pull/77 (49a1794): opt-in StagedSearch, variants r8x4 / w4to8x4 with T0081 P2 gw8x4 and direct as controls, 8 offline tests (fixed-work), host runbook in bots/agent_v2/staged_search/README.md.
