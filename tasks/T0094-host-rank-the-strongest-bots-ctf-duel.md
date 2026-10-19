---
id: T0094
title: "Host: rank the strongest bots (ctf_duel_challenger, Meowmind, Meowmind+P3, agent_v2) in one paired screen"
status: open
owner: ""
needs: [host, ai-test]
writes: [reports/host/strongest_bot/, profiles/strongest_bot_screen.json, docs/STRONGEST_BOT_SEEDS.md]
depends: [T0093]
land: direct
branch: ""
updated: 2026-10-19
---

## Goal

Rank the strongest bots on the host with one paired screen. Contenders, all built from bots/challengers/variants.json (T0093): ctf_duel_challenger (uploaded 2026-10-19, no host evidence), meowmind_search_on (original Meowmind; T0078: 0.835, above agent_v2_p2xxl_direct by +0.066 [+0.049, +0.083]), meowmind_p3_xxl and meowmind_p3_distill_xxl (Meowmind with our P3 in its opponent mixture; no host evidence), agent_v2_p2xxl_direct (production default). Plus an A/A copy of ctf_duel_challenger.

Design, frozen before any game (profiles/strongest_bot_screen.json, the layout of reports/host/meowmind_gap/screen/RUN.json's config; bots/p_only/tools/run_screen.py):
- 56 fresh maps per team count 2..15 (784 fixtures per arm, 6 arms, 4,704 games); the frozen held-out reference roster Fable, MiMo, Kimi, GLM, Player4, as in T0078/T0081, so the numbers are comparable; one candidate seat per fixture, the same fixtures for every arm.
- Exactly 8 game workers under python tools/repo/task.py lock cpu, GPU training stopped, 2,000/50 ms complete replies.
- Seeds: reserve fresh preflight and screen seeds in docs/STRONGEST_BOT_SEEDS.md before any game, check derived-seed collisions against every docs/*SEEDS*.md ledger, never reuse a consumed seed.
- Preflight first (16 maps at counts 2, 5, 8, 15, every arm): any fault, dead game or late reply in an arm stops the task with a report; no budget extension after seeing results.
- One Bonferroni family (alpha 0.05) of four primary comparisons: ctf_duel_challenger - meowmind_search_on; ctf_duel_challenger - agent_v2_p2xxl_direct; meowmind_p3_xxl - meowmind_search_on; meowmind_p3_distill_xxl - meowmind_search_on. A/A: ctf_duel_challenger~aa - ctf_duel_challenger. Equal weight per team count, map clusters, bands 2 / 3-5 / 6-10 / 11-15.
- Ranking rule: report every arm's equal-count utility with its interval. Call an arm stronger than another only where a primary comparison's Bonferroni interval excludes zero; any other pairwise difference is descriptive (95%, no claim). "Not shown" is not "equal". If no arm is separated from the rest, say so.
- Do not pool with T0078's games (other seeds); compare with T0078 only descriptively.

Generated reports only: reports/host/strongest_bot/ (the run_screen receipts plus a REPORT rendered from screen.json by a script; no hand-written figures; missing evidence unavailable, never zero). No default promotion and no submission decision: any promotion needs its own release gate. Note the licenses in the report: ctf_duel_challenger is MIT (bots/challengers/README.md); the original Meowmind carries no license notice (bots/meowmind_p/README.md).

## Done when

Seeds reserved and checked; preflight receipt; the frozen config; the 4,704-game screen receipts (or a documented stop) in reports/host/strongest_bot/; a generated report that ranks the contenders under the predeclared rule, with the A/A noise, per-band results and every unavailable item named; landed on main.

## Handoff

Not started.

## Log

- 2026-10-19 17:11Z claude-fervent@cloud: created
