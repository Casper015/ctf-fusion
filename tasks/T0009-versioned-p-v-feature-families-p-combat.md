---
id: T0009
title: "Versioned P/V feature families: P combat context, richer V, 4XL V infrastructure"
status: review
owner: claude@caspers-macbook
needs: []
writes: [inference/combat.hpp, inference/policy_p3.hpp, inference/value_v3.hpp, fusion/, training/, tools/cpp/, tools/pipeline/, tools/host/, tests/, profiles/v_pilot.json, profiles/p_features.json, docs/FEATURE_FAMILIES.md, bots/p_only/tools/, bots/p_only/bots/, bots/p_only/profiles/, bots/p_only/tests/, bots/p_only/include/]
depends: []
land: pr
branch: claude/t0009-versioned-p-v-feature-families-p-combat
updated: 2026-10-10
---

## Goal

Implement (no training or evidence here) the staged P/V upgrade. (1) Separate feature families: legacy P p2 / V v2 stay bit-identical and usable; new P p3 = p2 + a per-destination combat-context block (one-hit targets and carriers, coordinated volleys, carrier required hits and delivery distance, threat exposure), new V v3 = v2 + public carrier/flag/rival information. Record layout versions are distinct from family versions; runs, exports, registries and manifests check family compatibility and refuse mismatches; mixed exports may pair a frozen P with a new V. (2) Training/evaluation support for a controlled four-arm V pilot (v2/v3 x M 256/32 / 4XL 8192/512): V-only caches, batch sweeps with VRAM and throughput, validation stopping, optional candidate-pair ranking loss, outcome/Brier/calibration/phase/count/subset metrics. (3) Frozen-carrier own-seat collection and legal candidate-branch continuation records. (4) CPU deployment support: P-only carriers that consume p3/v3 sets, a bounded direct-4XL route with fallback, a distillation route, and V cost measurement tools. (5) Contract tests. Training (cuda) and evaluation (host) are separate dependent tasks created by this task. Spec: the 2026-10-10 request recorded in docs/FEATURE_FAMILIES.md.

## Done when

PR merged with: new headers and extractor layout; legacy outputs bit-identical on fixtures; family checks in run/export/registry/manifest; tests for HP visibility (legacy P unit features, p3 combat block), the V alias separation, extractor/runtime agreement, masks/immunity/respawn/missing candidates, incompatible combinations failing, quantized parity, branch evaluation not mutating the root; both test suites pass; dependent cuda/host tasks exist with exact commands in their Goal or this task's Handoff.

## Handoff

*2026-10-10 19:27Z claude@caspers-macbook:*

Status: in review as https://github.com/Casper015/ctf-fusion/pull/9 (branch claude/t0009-versioned-p-v-feature-families-p-combat, commits 0e51c2c, 0e1c07e, eda32c9, 48e84a4). Implemented and tested on caspers-macbook; nothing trained; no timing or strength measured.

Done:
- Feature families p2/p3/v2/v3 with digests (fusion/families.py), record layout 3, family checks in runs, shards, warm starts, evaluation, exports, mixed exports, registries and carriers; legacy outputs bit-identical (tests/test_feature_families.py).
- V data and training tooling: dataset --rows designated, branch_collect / branch_dataset, value_cache, train_value (--cache, --rank, --checkpoint, --soft-targets), value_sweep, evaluate_value, distill_value, compare_policy, feature_audit.
- Carriers: make_families.py -> ponly_p3.hpp, v12_ponly_f.hpp, c08_ponly_f.hpp; Route A bound (PONLY_V_MAX_CALLS, PONLY_V_DEADLINE_MS); export_mixed --value-run / --policy-run (shared V); tools/host/value_cost.py.
- Profiles: profiles/v_pilot.json, profiles/p_features.json; bots/p_only/profiles/collect_v_data.json (20261201), screen_p_features.json (20261210), screen_v_pilot.json (20261211), variants_families.json.
- docs/FEATURE_FAMILIES.md.

Checked: core suite 148 OK (with and without AI_TEST_ROOT); torch tests OK on CPU PyTorch 2.14.1 (scratch venv); bots/p_only suite 20 OK.

Next:
1. A human reviews and merges PR #9 (merge commit). Then: python tools/repo/task.py done T0009 --text "merged as <sha>".
2. Then T0010 (host) -> T0012 (cuda) -> T0014 (host); T0011 (cuda) -> T0013 (host); T0015 (docs). Exact commands are in each task's Goal.

Unchecked:
- The CUDA path (FUSION_TEST_DEVICE=cuda python -m unittest tests.test_torch_backend on the host).
- branch_collect with real P-only bots under the official limits (only seeded test bots locally).
- The prefix nondeterminism of the roster bots (T0010 records prefix_mismatch).
- Whether p3 summaries per C08 branch view cost too many samples (T0013).
- The Route A deadline value of 30 ms (T0014).
- The AGENTS.md machine table still says caspers-macbook has no capabilities; this machine has an AI_Test checkout (CTF_CAN=ai-test used).

## Log

- 2026-10-10 18:32Z claude@caspers-macbook: created
- 2026-10-10 18:32Z claude@caspers-macbook: claimed
- 2026-10-10 19:01Z claude@caspers-macbook: note: milestone: families p2/p3/v2/v3 with compatibility checks (0e51c2c), frozen-carrier V data, branch continuations, V pilot training/eval tooling (0e1c07e); pushed to claude/t0009-versioned-p-v-feature-families-p-combat. Next: P-only carriers consuming p3/v3, V deployment routes and bench, profiles, docs, PR.
- 2026-10-10 19:27Z claude@caspers-macbook: handoff -> review
