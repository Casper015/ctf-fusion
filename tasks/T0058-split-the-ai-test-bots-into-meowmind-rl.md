---
id: T0058
title: "Split the AI_Test bots into Meowmind RL training and held-out test lineages"
status: review
owner: claude-review@caspers-macbook
needs: []
writes: [profiles/meow_rl_training_pool_v2.json, docs/MEOWMIND_RL_POOL_SPLIT.md, tests/test_meow_rl_pool_split.py]
depends: []
land: pr
branch: claude-review/t0058-split-the-ai-test-bots-into-meowmind-rl
updated: 2026-10-15
---

## Goal

Casper asked (2026-10-15) to add the original test repo's (AI_Test) bots to Meowmind RL training while keeping testing on unseen bots. Split whole lineages: training pool v2 = T0055's pool plus AI_Test lineages (Astra variants; the Player5 lineage incl. Control_Hybrid/Omega, C-series/R3, Player2_v005, S1/S2 controls; Player1; Player3), held out for testing = Fable, MiMo, Kimi, GLM, Player2 (original), Player4, Player6. Pinned sha256s, family cap 0.15, a test that the split is disjoint by lineage and that the screen roster is held out.

## Done when

Pool v2, split doc and test merged via PR; the host training task with pool v2 is registered.

## Handoff

*2026-10-15 09:51Z claude-review@caspers-macbook:*

PR https://github.com/Casper015/ctf-fusion/pull/55: profiles/meow_rl_training_pool_v2.json (50 bots, 9 lineage families at 11.1%, pinned from AI_Test 3bab99c; 44 pins verified on caspers-macbook), docs/MEOWMIND_RL_POOL_SPLIT.md, tests/test_meow_rl_pool_split.py. Held out: Fable, MiMo, Kimi, GLM, Player2, Player4, Player6. Next: merge; T0059 (host) re-verifies pins on its AI_Test checkout and trains. Not checked: the host checkout's bytes, the six judge copies.

## Log

- 2026-10-15 09:49Z claude-review@caspers-macbook: created
- 2026-10-15 09:49Z claude-review@caspers-macbook: claimed
- 2026-10-15 09:51Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/55
- 2026-10-15 09:51Z claude-review@caspers-macbook: handoff -> review
