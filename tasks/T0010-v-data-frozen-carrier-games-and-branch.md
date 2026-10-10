---
id: T0010
title: "V data: frozen-carrier games and branch continuations for the V pilot"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/report/v_data/]
depends: [T0009]
land: direct
branch: ""
updated: 2026-10-10
---

## Goal

Collect the value data of docs/FEATURE_FAMILIES.md section 5 under the frozen carrier v12_ponly_xxl (P XXL from inference/models_xxl, V off, fixed time) against the frozen reference roster, data seed 20261201 (training only; never a screen seed). On DESKTOP-G60OJUD after T0009 is merged (git pull --ff-only; no pull or rebuild during the runs): (1) python tools/repo/task.py lock cpu --task T0010 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/collect_v_data.json   (500 maps x 14 counts = 7,000 games, keep_matches; resumable). (2) python tools/repo/task.py lock cpu --task T0010 -- python -m training.branch_collect --matches data/p_only/v_data/matches --carrier v12_ponly_xxl --out data/branches/v12_ponly_xxl --roots-per-match 2 --candidates 4 --repeats 1 --workers 6   (official limits from the branch turn on; do not rebuild build/p_only between (1) and (2): the carrier binary hash must match the matches). (3) python -m training.branch_dataset --branches data/branches/v12_ponly_xxl --out data/rank/v12_ponly_xxl. Data stays in data/ (never committed). Commit only the screen's by-product report bots/p_only/report/v_data/ and record in the Handoff: games, faults, dropped roots and why, prefix_mismatch distribution (nondeterminism), resolved/uncertain pair counts from data/rank/v12_ponly_xxl/index.json, wall times.

## Done when

bots/p_only/report/v_data/ committed; data/p_only/v_data/matches, data/branches/v12_ponly_xxl and data/rank/v12_ponly_xxl exist on the host; Handoff lists counts, faults, dropped roots, prefix mismatches and pair counts; T0012 can start.

## Handoff

Not started.

## Log

- 2026-10-10 19:23Z claude@caspers-macbook: created
