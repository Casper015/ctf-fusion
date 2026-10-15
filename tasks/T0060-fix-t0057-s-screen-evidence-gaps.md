---
id: T0060
title: "Fix T0057's screen evidence gaps: multiplicity, search telemetry, preflight receipts, all-seat faults"
status: open
owner: ""
needs: []
writes: [bots/p_only/tools/analyze.py, bots/p_only/tools/run_screen.py, bots/p_only/tools/preflight_meow_rl.py, bots/meow_rl/bot.hpp, roster/strength.py, bots/p_only/tests/test_p_only.py, tests/test_meow_rl_export.py, tools/reporting/generate_meow_rl_screen_multiplicity.py, reports/host/meow_rl_screen/secondary/MULTIPLICITY.md, docs/MEOWMIND_RL_DEPLOYMENT.md]
depends: []
land: pr
branch: ""
updated: 2026-10-15
---

## Goal

Casper asked (2026-10-15) to fix the gaps the Claude+Codex audit found in T0057 (commit 36b8489): (1) the secondary screen predeclared Bonferroni (z=2.638) but analyze.py always uses 1.96 and run_screen.py never forwards multiplicity, so 'large_v - v12_ponly_xxl' was published as hurts though its corrected interval contains 0; add multiplicity to analyze.py/run_screen.py and publish a corrected, generated secondary verdict table from the committed screen.json. (2) MEOWRL search counters never reached the receipts (the bot printed them once after its final reply and the runner kills it): print cumulative counters every turn, parse them in analyze.py and preflight_meow_rl.py, and make missing counters a failure, not zeros. (3) preflight_meow_rl.py: record per-game details, timestamps, bundle/binary hashes and the search deadline read from the bundled source; require --seed. (4) Fault accounting covers only our seat: record all-seat and opponent faults in roster.strength rows and report them. No new games, no change to the T0057 measurements.

## Done when

PR merged: tests for each fix pass; MULTIPLICITY.md generated from screen.json; a host re-run is not part of this task.

## Handoff

Not started.

## Log

- 2026-10-15 18:05Z claude-review@caspers-macbook: created
