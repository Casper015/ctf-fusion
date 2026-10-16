---
id: T0063
title: "Host: aligned fixed-work search diagnostics, eight-worker preflight and freeze (T0061 runbook 4-8)"
status: done
owner: codex-decision-pilot@desktop-g60ojud
needs: [host, ai-test]
writes: [bots/meow_rl/bot.hpp, docs/MEOWMIND_RL_SCREEN_SEEDS.md, docs/MEOWMIND_RL_SEEDS.md, profiles/meow_rl_aligned_screen_a.json, profiles/meow_rl_aligned_screen_b.json, reports/host/meow_rl_aligned/FREEZE.json, reports/host/meow_rl_aligned/PREFLIGHT.json, reports/host/meow_rl_aligned/diagnostics/, reports/host/meow_rl_aligned/trees/, tests/test_meow_rl_aligned.py]
depends: [T0062]
land: direct
branch: codex-decision-pilot/t0063-host-aligned-fixed-work-search
updated: 2026-10-16
---

## Goal

Run steps 4-8 of the T0061 host runbook (docs/MEOWMIND_RL_ALIGNED.md) on desktop-g60ojud. Stop CUDA training and benchmarks first, and wrap every game-playing step in "python tools/repo/task.py lock cpu --task T0063 -- ...".

1. Build: python bots/p_only/tools/build.py --stats --variants bots/meow_rl/variants_aligned.json
2. Reserve fresh seeds before any game. Check docs/DIGEST.md, and never reuse a consumed or training seed.
   - In docs/MEOWMIND_RL_SEEDS.md: one seed base for the diagnostic root collection.
   - In docs/MEOWMIND_RL_SCREEN_SEEDS.md: one preflight seed, plus the schedule A and schedule B seeds, all distinct.
3. Fixed-work diagnostics:
       bots/meow_rl/aligned/trees.py collect --seed-base <S_roots> --out data/meow_rl_aligned/trees      (exactly 8 workers; 224 games vs training pool v2)
       bots/meow_rl/aligned/trees.py select ...
       bots/meow_rl/aligned/trees.py build ...
       bots/meow_rl/aligned/diagnose.py --trees data/meow_rl_aligned/trees --corpus data/meow_rl_aligned/corpus --out reports/host/meow_rl_aligned/diagnostics
   Commit COLLECTION.json, ROOTS.json and TREES.json (copied to reports/host/meow_rl_aligned/trees/), plus DIAGNOSTICS.{json,md}.
4. Eight-worker preflight:
       python bots/meow_rl/aligned/preflight.py --seed <S_preflight>
   This is 448 games, exactly 8 workers. If it is BLOCKED:
   - commit the receipt;
   - do NOT lower the concurrency or drop an arm;
   - hand off blocked with the failure and a proposed fix. A fix needs new frozen settings and a fresh preflight seed.
5. Only if it passed: python bots/meow_rl/aligned/freeze.py --seed-a <S_A> --seed-b <S_B>. This writes the seeds into profiles/meow_rl_aligned_screen_{a,b}.json and writes FREEZE.json. A roster-audit hit blocks the freeze: hand off blocked for a human decision.
6. Record host state in the log: power plan, GPU idle, other processes.

## Done when

DIAGNOSTICS.{json,md}, PREFLIGHT.json (passed at 8 workers, or a blocked handoff) and FREEZE.json committed

## Handoff

*2026-10-16 00:00Z codex-decision-pilot@desktop-g60ojud:*

Aligned search diagnostics completed (1024 fixed trees, 8112 leaves; R 7-bit quantization caused 66.5% top-action degradation and 42.1% discordance, H WideAct16 restored 97.3% agreement and 0.0000 regret). Windows stderr pipe buffering resolved in bot.hpp. Eight-worker preflight passed (448 games, seed 20261956, 0 faults/timeouts). Freeze passed with 0 roster audit hits, freezing seeds 20261954 and 20261955 into schedules A and B.

## Log

- 2026-10-15 22:19Z claude-review@caspers-macbook: created
- 2026-10-15 22:20Z claude-review@caspers-macbook: amended goal
- 2026-10-15 23:12Z codex-decision-pilot@desktop-g60ojud: claimed
- 2026-10-15 23:31Z codex-decision-pilot@desktop-g60ojud: amended writes -> bots/meow_rl/bot.hpp, docs/MEOWMIND_RL_SCREEN_SEEDS.md, docs/MEOWMIND_RL_SEEDS.md, profiles/meow_rl_aligned_screen_a.json, profiles/meow_rl_aligned_screen_b.json, reports/host/meow_rl_aligned/FREEZE.json, reports/host/meow_rl_aligned/PREFLIGHT.json, reports/host/meow_rl_aligned/diagnostics/, reports/host/meow_rl_aligned/trees/
- 2026-10-15 23:51Z codex-decision-pilot@desktop-g60ojud: amended writes -> bots/meow_rl/bot.hpp, docs/MEOWMIND_RL_SCREEN_SEEDS.md, docs/MEOWMIND_RL_SEEDS.md, profiles/meow_rl_aligned_screen_a.json, profiles/meow_rl_aligned_screen_b.json, reports/host/meow_rl_aligned/FREEZE.json, reports/host/meow_rl_aligned/PREFLIGHT.json, reports/host/meow_rl_aligned/diagnostics/, reports/host/meow_rl_aligned/trees/, tests/test_meow_rl_aligned.py
- 2026-10-15 23:59Z codex-decision-pilot@desktop-g60ojud: landed: 76557d36db on main, 14 file(s)
- 2026-10-16 00:00Z codex-decision-pilot@desktop-g60ojud: done
