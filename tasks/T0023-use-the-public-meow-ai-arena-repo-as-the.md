---
id: T0023
title: "Use the public meow-ai-arena repo as the engine and judge-bot source (CI without a secret)"
status: done
owner: claude-ci@caspers-macbook
needs: []
writes: [fusion/paths.py, fusion/native.py, roster/strength.py, bots/p_only/tools/screen_runner.py, tests/test_pipeline.py, tests/test_roster.py, tests/test_value_pipeline.py, tests/test_arena_layout.py, ".github/", README.md, CONTRIBUTING.md, AGENTS.md, docs/TECHNICAL_OVERVIEW.md, docs/FIRST_RUN.md, training/specs/ENVIRONMENT.md, tools/README.md, tasks/README.md]
depends: []
land: pr
branch: claude-ci/t0023-use-the-public-meow-ai-arena-repo-as-the
updated: 2026-10-11
---

## Goal

chen-006/meow-ai-arena (public) ships the official engine (ctf/workspace/engine.py) and the 10 judge bots (ctf/bots/). Make it a first-class source: fusion finds the engine and judge bots in either that layout or the AI_Test layout, CI runs the end-to-end tests against it with no secret, and the docs point there. AI_Test stays the source of what the public repo does not have: the legacy forecast (nextgen), the Player5 reference bot, the stress-pack, benchmark and archive bots.

## Done when

- The 22 end-to-end tests pass against a clean clone of the public repo in CI.
- The tests that need the legacy planner or Player5 skip with a clear reason instead of erroring.
- README, CONTRIBUTING, FIRST_RUN and the CI job name the public repo and say what still needs AI_Test.

## Handoff

*2026-10-11 01:27Z claude-ci@caspers-macbook:*

Merged as PR #18 (83af82b). CI runs the end-to-end tests against chen-006/meow-ai-arena pinned to 54139d4; no secret.

## Log

- 2026-10-11 00:33Z claude-ci@caspers-macbook: created
- 2026-10-11 00:33Z claude-ci@caspers-macbook: claimed
- 2026-10-11 01:08Z claude-ci@caspers-macbook: handoff -> review
- 2026-10-11 01:27Z claude-ci@caspers-macbook: done
