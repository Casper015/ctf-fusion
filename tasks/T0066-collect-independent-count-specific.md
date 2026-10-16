---
id: T0066
title: "Collect independent count-specific opponent datasets"
status: open
owner: ""
needs: [host, ai-test]
writes: [profiles/count_specialists_collection.json, reports/host/count_specialists_data/, docs/COUNT_SPECIALISTS_SEEDS.md]
depends: [T0064]
land: pr
branch: ""
updated: 2026-10-16
---

## Goal

Casper authorized fresh data and specialized models for each team count. Collect separate n02 through n15 datasets; do not collapse them into the earlier five bands. Existing P already supports count-specific routing: this refresh is about independent data coverage and training, not inventing that feature.

First pilot budget: 200 fresh independent full-game maps per count, 2,800 matches total, exactly eight game workers and official 2s first/50ms later opponent limits. Treat this as a learnability/coverage pilot, not evidence that data is sufficient. Use the task CPU lease after T0064 releases it; never interrupt another job. Register fresh map/flag seeds and their derivation before running; do not reuse consumed evaluation seeds. Preserve initial/refresh flag randomness under the official N+1 spots and flag-count rules, and log engine/source/binary hashes. Balance learner/observation seats and frozen opponent-family sampling; retain full all-seat logs, events, identities and faults. Filter P labels using existing admissibility rules; faults, censored attacks and missing replies cannot become fabricated valid actions.

Within each count freeze 140 train / 30 validation / 30 test map groups before training. All seats, prefixes and derived examples from one map stay together. Use globally disjoint map/flag seed namespaces across counts and splits. Validation/test are diagnostic within the same approved training roster; held-out opponent lineages remain reserved for later OOD deployment screens, not mixed into training. Native engine map hashing must account for intentionally repeated geometry: report duplicates and prevent identical normalized map geometries crossing splits within a count; do not silently replace consumed fixtures or hide rejected duplicate generations. Provide distinct nXX manifests, actual retained examples and unique maps; if the requested independent split cannot be achieved, report the shortage and prepare a separate registered expansion rather than pretending independence.

Keep per-count coverage by carrier/HP/combat/respawn, pickup/drop, upcoming flag return/cooldown and spawn-distance advantage. More turns are not more independent maps. Preserve original failed records and verified backups. Produce authoritative sources, causal feature extraction and native/Python parity receipts using current P2 schema initially. No new CNN, capacity sweep, solver rewrite or PPO use of these static logs. A future PPO task must collect its own on-policy trajectories.

## Done when

All 14 nXX corpora, map-disjoint split manifests, exact source/binary/engine and feature identities, seed ledger, all-seat fault totals, effective sample/coverage counts and verified backups delivered. Pilot insufficiency is explicit. Collector passed exactly-eight-worker preflight; data are reproducible and no future information or hidden flag seed enters model inputs. New games remain consumed even if failures occur; publish a PR and handoff exact dataset/model-family paths.

## Handoff

*2026-10-16 00:12Z codex-count-plan@caspers-macbook:*

User-authorized future host work. Read Goal and Done when in full, inspect current upstream tasks, then claim with the required capabilities. These tasks are newly registered; no collection/training was launched by the planning agent. Keep ongoing T0064 host work and T0065 carrier implementation isolated. Specialized data and policies use exact counts 2-15, not band labels.

## Log

- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: created
- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: handoff prepared
