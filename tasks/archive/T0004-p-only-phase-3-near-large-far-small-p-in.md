---
id: T0004
title: "P-only phase 3: near-large / far-small P in C08's second-turn views"
status: done
owner: codex@desktop-g60ojud
needs: [host]
writes: [bots/p_only/report/near_far/]
depends: [T0003]
land: direct
branch: ""
updated: 2026-10-12
---

## Goal

Do larger P models help more for opponents near our units? Offline first (held-out records on the host). Build a tiered arm only if the gains concentrate near us. Spec: [bots/p_only/TASK.md](../../bots/p_only/TASK.md), Phase 3.

## Done when

- An offline report in `bots/p_only/report/near_far/` with the NLL gain by distance band and the decision-change counts.
- A clear decision: build the tiered arm, or one size is enough.
- If the arm is built: that needs tier-specific identities through `training.export_weights`, which is shared core. Open a separate task with `land: pr` for it.

## Handoff

*2026-10-12 05:20Z codex@desktop-g60ojud:*

Completed and merged PR #26 https://github.com/Casper015/ctf-fusion/pull/26; head eadf0f4c66f37e8f59ae2e867e3d31afc2034cc8. Exact-head nine verification checks passed; release skipped as expected. Frozen source 8a4eab4bef5bf65e971ec6712eebb89821738130; 260 retained maps, native parity/NLL, six decision modes and independent raw/paired-map audits complete. Larger P improves prediction more in far views; evidence does not support near-large/far-small routing and is not playing strength/50ms latency. Production defaults unchanged. Reports/user outputs/T0004 DELIVERY and CI retained; no measurement/check/audit restart. User authorized task status reconciliation on 2026-10-12.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-12 05:20Z codex@desktop-g60ojud: done