---
id: T0023
title: "Use the public meow-ai-arena repo as the engine and judge-bot source (CI without a secret)"
status: review
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

*2026-10-11 01:08Z claude-ci@caspers-macbook:*

PR https://github.com/Casper015/ctf-fusion/pull/18, all checks green (incl. the end-to-end job against chen-006/meow-ai-arena pinned to 54139d4, and Windows).

After merge: (1) remove 'continue-on-error: true' from the pipeline job once it has been green on main; (2) delete the worktree ../ctf-fusion.worktrees/T0023; (3) the host keeps the AI_Test working tree for the legacy planner (q_H), Player5 and the archived opponents, so nothing changes there.

Open question for Casper: should the roster/inventory scan 'ctf/bots' as well? Not done: ids would differ from the committed pools by path.

## Log

- 2026-10-11 00:33Z claude-ci@caspers-macbook: created
- 2026-10-11 00:33Z claude-ci@caspers-macbook: claimed
- 2026-10-11 01:08Z claude-ci@caspers-macbook: handoff -> review
