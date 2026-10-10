---
id: T0019
title: "Search experiment on the host: fixed-work diagnostics, equal-time screen, branch regret"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/report/search_portfolio/]
depends: [T0018]
land: direct
branch: ""
updated: 2026-10-10
---

## Goal

Evaluate T0018's search changes (docs/SEARCH_PORTFOLIO.md) with models, features, depth, scoring and V (off) fixed. On DESKTOP-G60OJUD after T0018 is merged (git pull --ff-only; no pull, rebuild or build/ cleanup while runs are going; record workers and host load). (1) Fixed-work diagnostics on recorded games of an earlier screen: python tools/repo/task.py lock cpu --task T0019 -- python bots/p_only/tools/search_diag.py --matches data/p_only/value/matches --arm v12_ponly_pxxl_vm --games 60 --fixed-samples 128 --out bots/p_only/report/search_portfolio/diag_fixed_work  (root digest mismatches must be 0 for every config; report changed-reply shares, prediction work, cache/memo hits, the diagnostic's draw differences and top-plan changes). (2) Equal full-turn-time games: python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_search.json --dry-run, then python tools/repo/task.py lock cpu --task T0019 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_search.json  (C08 A/B/C/D, v12 A/B, A/A of both A arms; 8 x 280 = 2,240 games; fresh tuning seed 20261220; official 2 s / 50 ms). Read paired utility per band with map-group intervals, the D-C-B+A interaction, the A/A rows, faults, late replies, p99/max replies, and the search experiment counters (samples displaced, truncations, predictions per call, states built, view and memo hits, alternatives searched/refined, cuts). (3) Ranking and regret against the incumbent from the screen's kept matches: python tools/repo/task.py lock cpu --task T0019 -- python bots/p_only/tools/branch_portfolio.py --matches data/p_only/search/matches --arm v12_search_a --other-arms c08_search_a c08_search_b c08_search_c c08_search_d v12_search_b --compare-arm v12_search_b --out data/branches/search_v12 --roots-per-match 1 --candidates 8 --repeats 2 --max-matches 200 --workers 6, then python bots/p_only/tools/branch_regret.py --branches data/branches/search_v12 --arm v12_search_a --compare-arm v12_search_b --out bots/p_only/report/search_portfolio/regret_v12; the same for C08 with --arm c08_search_a --compare-arm c08_search_d (other arms: the rest) -> data/branches/search_c08, regret_c08. Copy each data/branches/*/summary.json into the report folder (prefix mismatches, faults, unresolved pairs). Intervals that include zero are 'not shown'; more candidates, changed decisions or one favourable run are not a strength result. Leave the production default unchanged unless an arm improves within the same whole-turn budget; a positive tuning result goes to confirmation on an unseen seed (T0006).

## Done when

bots/p_only/report/search_portfolio/ committed (fixed-work diagnostics, screen report with counters, branch summaries and regret reports); Handoff states per arm whether an improvement is shown, not shown or a harm, and whether anything goes to confirmation.

## Handoff

Not started.

## Log

- 2026-10-10 23:26Z claude@caspers-macbook: created
