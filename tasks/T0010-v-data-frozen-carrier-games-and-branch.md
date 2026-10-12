---
id: T0010
title: "V data: frozen-carrier games and branch continuations for the V pilot"
status: done
owner: codex@desktop-g60ojud
needs: [host]
writes: [bots/p_only/report/v_data/]
depends: [T0009]
land: direct
branch: ""
updated: 2026-10-12
---

## Goal

Collect the value data of docs/FEATURE_FAMILIES.md section 5 under the frozen carrier v12_ponly_xxl (P XXL from inference/models_xxl, V off, fixed time) against the frozen reference roster, data seed 20261201 (training only; never a screen seed). On DESKTOP-G60OJUD after T0009 is merged (git pull --ff-only; no pull or rebuild during the runs): (1) python tools/repo/task.py lock cpu --task T0010 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/collect_v_data.json   (500 maps x 14 counts = 7,000 games, keep_matches; resumable). (2) python tools/repo/task.py lock cpu --task T0010 -- python -m training.branch_collect --matches data/p_only/v_data/matches --carrier v12_ponly_xxl --out data/branches/v12_ponly_xxl --roots-per-match 2 --candidates 4 --repeats 1 --workers 6   (official limits from the branch turn on; do not rebuild build/p_only between (1) and (2): the carrier binary hash must match the matches). (3) python -m training.branch_dataset --branches data/branches/v12_ponly_xxl --out data/rank/v12_ponly_xxl. Data stays in data/ (never committed). Commit only the screen's by-product report bots/p_only/report/v_data/ and record in the Handoff: games, faults, dropped roots and why, prefix_mismatch distribution (nondeterminism), resolved/uncertain pair counts from data/rank/v12_ponly_xxl/index.json, wall times.

## Done when

bots/p_only/report/v_data/ committed; data/p_only/v_data/matches, data/branches/v12_ponly_xxl and data/rank/v12_ponly_xxl exist on the host; Handoff lists counts, faults, dropped roots, prefix mismatches and pair counts; T0012 can start.

## Handoff

*2026-10-12 05:20Z codex@desktop-g60ojud:*

Completed and merged PR #23 https://github.com/Casper015/ctf-fusion/pull/23; final reviewed head 0e82a279898e8b569ee0d45f0507b16b203a5c0b. Original measurement source ebda2c4867fb6503f8e28f3bd0a8b434e6affb96: 7000 games, 14000 roots (298 legal-shortage drops), 13702 retained roots/54808 continuations, zero faults; split/NPZ/label/weight audits and local checks complete. Original submitted head CI passed; final review changes were made by others, and PR is now confirmed merged. Reports bots/p_only/report/v_data/; user outputs/T0010. No replay or automatic retraining. T0012 data dependency satisfied. User authorized task status reconciliation on 2026-10-12.

## Log

- 2026-10-10 19:23Z claude@caspers-macbook: created
- 2026-10-12 05:20Z codex@desktop-g60ojud: done
