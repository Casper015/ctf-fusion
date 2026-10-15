---
id: T0056
title: "Export and gate Meowmind RL CPU bots"
status: review
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

*2026-10-15 09:04Z claude-review@caspers-macbook:*

**Done** (PR https://github.com/Casper015/ctf-fusion/pull/53, branch claude-review/t0056-export-and-gate-meowmind-rl-cpu-bots):
- bots/meow_rl/: the bot (sampled actor-direct, greedy diagnostic), the two-turn value-search arm, export.py, variants.json;
- inference/meow_rl/warm_start: the matched control;
- profiles/meow_rl_screen.json: primary and secondary blocks, seeds null;
- tests/test_meow_rl_export.py (12 tests) and docs/MEOWMIND_RL_DEPLOYMENT.md.

Verified on caspers-macbook, no PyTorch:
- bot = Python pipeline on every recorded reply;
- recovery replayed exactly;
- byte-reproducible export;
- identity, architecture and NaN refusals;
- critic C++ = NumPy;
- exact search fallback;
- all 5 variants build.

Codex reviewed twice; every finding is fixed.

**Next, after T0055's final checkpoints** (on the host):
1. Export the three candidates:
   - python bots/meow_rl/export.py --checkpoint runs/meow_rl/large_v_s101/checkpoints/final.pt --name large_v_s101 --calibration-batches runs/meow_rl/large_v_s101/batches
   - the same with small_v_s101;
   - large_v_s101 again with --critic --search --name large_v_s101_search.
2. Add meow_rl_large_v_s101, meow_rl_small_v_s101 and meow_rl_large_v_s101_search to bots/meow_rl/variants.json.
3. Check each MANIFEST's agreement_held_out. If greedy agreement is low, a QAT step against reference16 is a separate recorded step before screening.
4. Commit the exports via PR.

Then T0057:
- preflight the actor-direct bot and the large-V search arm at 2 s / 50 ms;
- freeze workers and MEOW_RL_SEARCH_DEADLINE_MS;
- write each profiles/meow_rl_screen.json block to a run config with a fresh seed;
- run bots/p_only/tools/run_screen.py --config <block>.

**Not checked:**
- trained artifacts and reading .pt files;
- Windows;
- host timing;
- the large critic header's compile time;
- all games.

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
- 2026-10-15 08:13Z claude-review@caspers-macbook: claimed (forced)
- 2026-10-15 08:13Z claude-review@caspers-macbook: note: Claimed with --force past unfinished T0055 on Casper's explicit instruction (2026-10-15, chat: 'you do t0056 first'). Scope now: the export/bot pipeline, the matched warm-start control (the warm-start actor), the switchable search arm as plumbing, and the screen spec incl. secondary arms vs P3-distilled (meowmind_p_distill_xxl_both) and P2 production (v12_ponly_xxl). The trained candidate plugs in after T0055; no host timing invented.
- 2026-10-15 09:04Z claude-review@caspers-macbook: handoff -> review
- 2026-10-15 09:04Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/53 (targets main; carries T0054's session.hpp follow-ups).
- 2026-10-15 09:29Z claude-review@caspers-macbook: note: PR 53 screen spec now uses a held-out opponent roster (Fable, MiMo, Kimi, GLM, Player2_v005, Player4): T0055's training pool held 8 of the 12 default screen opponents.
