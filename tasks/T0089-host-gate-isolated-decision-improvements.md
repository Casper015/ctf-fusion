---
id: T0089
title: "Host gate isolated decision improvements against matched direct play"
status: open
owner: ""
needs: [host, ai-test]
writes: [tools/screens/search_decision_pilot/, reports/host/search_decision_pilot/, profiles/search_decision_pilot.json, docs/SEARCH_DECISION_PILOT_SEEDS.md]
depends: [T0084, T0087, T0088]
land: direct
branch: ""
updated: 2026-10-18
---

## Goal

Run only after upstream branches deliver implementation OR an explicit defer decision. Do not invent an arm to satisfy a dependency. Select at most two isolated candidate arms using validation evidence before any formal games: staged search and/or candidate-relative scorer. If neither is eligible, publish no-run and stop. Do not test their combination until isolated evidence supports it in a later task. Fix common P/actor/resolver, with a same-package direct baseline; include the current production D as an additional reference if that baseline differs. Never attribute P/actor differences to a scorer or scheduling change.

Freeze exact arm list, identities, complete turn limits, primary contrasts and family correction before preflight. Exactly eight workers, official host/CPU lease, no GPU training; 16 fixtures/count at 2/5/8/15 per unique arm for preflight (64 times arm count). Require complete all-seat counters, no faults/dead/late, first <1800ms and steady <48ms, usable completed work per count and full raw logs. A failure stops admission; never silently lower workers, prune the candidate or lengthen deadlines. Register a new configuration/seed only as a separately reviewed experiment.

If admitted, use 96 fresh maps/count 2..15 = 1344 paired fixtures per arm, plus one A/A copy of the validation-selected primary candidate. At most five arms including controls/A-A = 6720 games; register exact cost before play. These are paired independent map fixtures, NOT a claim of ten thousand independent games per variant. Use a frozen roster audited against all P, actor, diagnostic and ranker data; held-out status must be stated per model component. Interleave arms, equal-count weight, map-cluster intervals, multiplicity over all preregistered primary comparisons. Report A/A, all-seat faults and per-count/band descriptive results. Publish intention-to-treat results even with faults; fault-free sensitivity is secondary. No favorable early stopping, sample extension or final-screen tuning.

Candidate admission requires a positive adjusted interval against its matched direct baseline and operational gates, not merely beating the old losing search. If it differs from production D, require separately preregistered evidence against D before recommending deployment. No automatic promotion here; preserve the frozen default. Report not-shown as uncertainty, not equality; a higher completed-tree count or lower prediction loss is not strength.

Shared execution contract: this continues an architecture redesign, not a legacy-behavior refactor. Keep official rules exact; do not require whole-bot parity with Meowmind. Preserve agent_v2_p2xxl_direct as production default. P (opponent prediction), pi (our actor), and V/advantage scorer are distinct components. Follow AGENTS.md, claim with your own identity, respect active lanes (especially T0082/PR69 and T0051), and use isolated worktrees. All new timing, collection and strength evidence comes from desktop-g60ojud with exactly eight game workers under the CPU lease, GPU training stopped, 2000/50 ms complete replies. GPU training uses the GPU lease and is scheduled separately. Pre-register fresh map/flag/actor/scenario seeds and check derived-seed collisions; never replay consumed seeds as new evidence. Offline reanalysis of existing immutable artifacts is allowed. Freeze source, engine, model, runner, binary and profile hashes. Missing evidence is unavailable, never zero or a successful check. Do not extend budgets after inspecting outcomes. No default promotion in these tasks; no new giant P/V, CNN, count specialists, or full learned attack policy. All repository prose is English.


## Done when

A reproducible gated screen (or documented no-run/preflight failure) binds every result to raw logs and hashes, reports isolated comparisons and uncertainty, and recommends keep/reject/defer without modifying defaults. No measured claim is a source constant, and no opponent overlap or failure is omitted.

## Handoff

Not started.

## Log

- 2026-10-18 03:36Z codex-search-plan@caspers-macbook: created
