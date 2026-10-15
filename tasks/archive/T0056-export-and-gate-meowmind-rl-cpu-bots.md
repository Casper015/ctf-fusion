---
id: T0056
title: "Export and gate Meowmind RL CPU bots"
status: done
owner: claude-review@caspers-macbook
needs: []
writes: [bots/meow_rl/, inference/meow_rl/, tests/test_meow_rl_export.py, profiles/meow_rl_screen.json, docs/MEOWMIND_RL_DEPLOYMENT.md]
depends: [T0055]
land: pr
branch: claude-review/t0056-export-and-gate-meowmind-rl-cpu-bots
updated: 2026-10-15
---

## Goal

Export the frozen pilot actor into an actor-direct CPU bot with unchanged resolver and a matched warm-start control. Add a separately switchable bounded two-ply large-V search arm using consistent utility reward units, explicit cache identities and deadline fallback. Follow docs/MEOWMIND_RL_PLAN.md. No default replacement.

## Done when

Python/native output parity, model/feature/reward-unit identity checks, protocol and fallback tests pass; frozen primary actor-direct vs warm-start plus A/A screen specification prepared with fresh seeds left for host allocation. Separate search arm and small-V ablation remain distinct. Host must subsequently measure full-turn 2s/50ms and run 392 paired fixtures; do not invent host timing on Mac.

## Handoff

*2026-10-15 10:09Z claude-review@caspers-macbook:*

PR #53 merged into main (1976364). Candidates meow_rl_large_v_s101, meow_rl_small_v_s101, meow_rl_large_v_s101_search exported with 40-episode calibration and added to variants.json.

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
- 2026-10-15 08:13Z claude-review@caspers-macbook: claimed (forced)
- 2026-10-15 08:13Z claude-review@caspers-macbook: note: Claimed with --force past unfinished T0055 on Casper's explicit instruction (2026-10-15, chat: 'you do t0056 first'). Scope now: the export/bot pipeline, the matched warm-start control (the warm-start actor), the switchable search arm as plumbing, and the screen spec incl. secondary arms vs P3-distilled (meowmind_p_distill_xxl_both) and P2 production (v12_ponly_xxl). The trained candidate plugs in after T0055; no host timing invented.
- 2026-10-15 09:04Z claude-review@caspers-macbook: handoff -> review
- 2026-10-15 09:04Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/53 (targets main; carries T0054's session.hpp follow-ups).
- 2026-10-15 09:29Z claude-review@caspers-macbook: note: PR 53 screen spec now uses a held-out opponent roster (Fable, MiMo, Kimi, GLM, Player2_v005, Player4): T0055's training pool held 8 of the 12 default screen opponents.
- 2026-10-15 10:09Z claude-review@caspers-macbook: done
