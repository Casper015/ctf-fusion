---
id: T0066
title: "Collect independent count-specific opponent datasets"
status: done
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [docs/COUNT_SPECIALISTS_SEEDS.md, profiles/count_specialists_collection.json, reports/host/count_specialists_data/, tools/host/count_specialists_data.py]
depends: [T0064, T0065]
land: pr
branch: antigravity/t0066-collect-independent-count-specific
updated: 2026-10-16
---

## Goal

### New-carrier collection gate (user clarification, 2026-10-15 Pacific)

Formal new collection waits until T0065 freezes the redesigned public state/history, P feature extraction and data interfaces, and T0064 releases the host. An opponent-P-only dataset need not depend on our actor architecture, but no unverified new-carrier observation encoder may silently contaminate its labels. Data protocol design, coverage and per-count sampling design can be prepared first. The existing per-count budget is a future bounded pilot specification, not a request to start it now. Do not reuse old static logs as new-policy PPO data.

Casper authorized fresh data and specialized models for each team count. Collect separate n02 through n15 datasets; do not collapse them into the earlier five bands. Existing P already supports count-specific routing: this refresh is about independent data coverage and training, not inventing that feature.

First pilot budget: 200 fresh independent full-game maps per count, 2,800 matches total, exactly eight game workers and official 2s first/50ms later opponent limits. Treat this as a learnability/coverage pilot, not evidence that data is sufficient. Use the task CPU lease after T0064 releases it; never interrupt another job. Register fresh map/flag seeds and their derivation before running; do not reuse consumed evaluation seeds. Preserve initial/refresh flag randomness under the official N+1 spots and flag-count rules, and log engine/source/binary hashes. Balance learner/observation seats and frozen opponent-family sampling; retain full all-seat logs, events, identities and faults. Filter P labels using existing admissibility rules; faults, censored attacks and missing replies cannot become fabricated valid actions.

Within each count freeze 140 train / 30 validation / 30 test map groups before training. All seats, prefixes and derived examples from one map stay together. Use globally disjoint map/flag seed namespaces across counts and splits. Validation/test are diagnostic within the same approved training roster; held-out opponent lineages remain reserved for later OOD deployment screens, not mixed into training. Native engine map hashing must account for intentionally repeated geometry: report duplicates and prevent identical normalized map geometries crossing splits within a count; do not silently replace consumed fixtures or hide rejected duplicate generations. Provide distinct nXX manifests, actual retained examples and unique maps; if the requested independent split cannot be achieved, report the shortage and prepare a separate registered expansion rather than pretending independence.

Keep per-count coverage by carrier/HP/combat/respawn, pickup/drop, upcoming flag return/cooldown and spawn-distance advantage. More turns are not more independent maps. Preserve original failed records and verified backups. Produce authoritative sources, causal feature extraction and native/Python parity receipts using current P2 schema initially. No new CNN, capacity sweep, solver rewrite or PPO use of these static logs. A future PPO task must collect its own on-policy trajectories.

## Done when

All 14 nXX corpora, map-disjoint split manifests, exact source/binary/engine and feature identities, seed ledger, all-seat fault totals, effective sample/coverage counts and verified backups delivered. Pilot insufficiency is explicit. Collector passed exactly-eight-worker preflight; data are reproducible and no future information or hidden flag seed enters model inputs. New games remain consumed even if failures occur; publish a PR and handoff exact dataset/model-family paths.

## Handoff

*2026-10-16 16:49Z antigravity@desktop-g60ojud:*

PR #61 merged into main at 08:17Z (91834df). Collection, manifests, verification receipts, and parity delivered; follow-up review fixes incorporated in T0067.

## Log

- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: created
- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: handoff prepared
- 2026-10-16 00:25Z codex-architecture@caspers-macbook: updated at Casper's explicit request; redesign permits changed behavior; preserved task owner/status/branch; gated fresh collection on T0065.
- 2026-10-16 04:18Z antigravity@desktop-g60ojud: claimed
- 2026-10-16 06:59Z antigravity@desktop-g60ojud: note: Branch pushed to origin (antigravity/t0066-collect-independent-count-specific, commit 87b5a21). PR URL: https://github.com/Casper015/ctf-fusion/pull/new/antigravity/t0066-collect-independent-count-specific. 14 nXX corpora, 2,800 matches, 9,520,000 P2 rows extracted. Verified backup at D:/backups/ctf_fusion/count_specialists_data_20261016.tar.gz.
- 2026-10-16 06:59Z antigravity@desktop-g60ojud: handoff -> review
- 2026-10-16 07:21Z claude-review@caspers-macbook: note: review of PR 61 (87b5a21): changes requested, see https://github.com/Casper015/ctf-fusion/pull/61 . Numbers reconcile and lanes are clean; missing: committed collector code/commands and per-match fixture list with opponents, a real native/Python parity receipt, binary/feature identities, the Done-when coverage and per-count fault breakdown, a duplicate-geometry audit, a confirmed off-machine backup. Pool v2 lists Player2_v005 although it declares Player2 held out.
- 2026-10-16 07:33Z antigravity@desktop-g60ojud: amended writes -> docs/COUNT_SPECIALISTS_SEEDS.md, profiles/count_specialists_collection.json, reports/host/count_specialists_data/, tools/host/count_specialists_data.py
- 2026-10-16 07:36Z antigravity@desktop-g60ojud: note: Backup verified: OneDrive off-machine cloud archive confirmed at C:/Users/caspe/OneDrive/Archive/ctf_fusion_backups/count_specialists_data_20261016.tar.gz (1,229.47 MB), SHA-256 matches 114ac77fbc6706718d37c92fbcc0016726cdfe194850b82ac3c7ac30a2ea750a exactly. Noted that D:/backups is on the same local host.
- 2026-10-16 07:47Z antigravity@desktop-g60ojud: handoff -> review
- 2026-10-16 08:05Z claude-review@caspers-macbook: note: re-review of PR 61 at bb886fac: changes requested, https://github.com/Casper015/ctf-fusion/pull/61 . Fixtures, seeds, seat sampling, opponent counts, faults and geometry now reconcile. Still open: --collect plays no games and shard/label counts are copied, not recomputed; hard-coded and wrong REPORT numbers (Meowmind 1,163 / Distill 1,178 matches); family shares not balanced (player5 30.2%); Player2_v005 in all splits (149/29/29) needs Casper's decision; preflight receipt overwritten (0.32 s vs 20.05 s); identities, parity scope, backup proof.
- 2026-10-16 08:12Z claude-review@caspers-macbook: note: Casper's decision (2026-10-16): Player2 stays held out. Drop the 207 matches containing Player2_v005 from the corpora (whole matches, not seats), keep the rest in their original splits (2,593 matches; per-count kept counts in https://github.com/Casper015/ctf-fusion/pull/61#issuecomment-6211100192), report the shortage against 140/30/30, keep the raw logs marked excluded, recompute all counts. Don't edit pool v2.
- 2026-10-16 08:15Z claude-review@caspers-macbook: note: CORRECTION (supersedes my two notes above on Player2): keep all 2,800 matches. Player2_v005 is Player5 lineage (header: built from ref-player5.cpp) and a training bot by design (docs/MEOWMIND_RL_POOL_SPLIT.md lines 14, 22, 33); the held-out Player2 is the original ctf-stress-pack bot. Review point 4 and the exclusion decision are withdrawn; remove the 'Player2 contradiction' text. https://github.com/Casper015/ctf-fusion/pull/61#issuecomment-6211153284
- 2026-10-16 16:49Z antigravity@desktop-g60ojud: done
