---
id: T0016
title: "5XL V capacity: width contracts and reproducible experiment"
status: active
owner: codex@desktop-g60ojud
needs: []
writes: [fusion/, training/, inference/mlp.hpp, inference/value_v3.hpp, tests/, tools/cpp/, tools/host/value_cost.py, profiles/v_capacity_5xl.json, docs/V_CAPACITY_5XL.md, reports/models/v_capacity_5xl/]
depends: []
land: pr
branch: codex/t0016-5xl-v-capacity-width-contracts-and
updated: 2026-10-10
---

## Goal

Implement and validate explicit v3 XL 1024,64; 4XL 8192,512; 5XL 16384,1024 capacity comparison. Preserve production and T0012 four-arm scope. Audit native buffers, safe quantization, deterministic chunked export, bounded GPU profiles, offline paired decision metrics and hand off host-dependent evidence.

## Done when

PR with passing width/export/parity contracts, reproducible same-data FP32 pilot commands, parameter/MAC counts, capability and evidence status; real training/host evidence where available without changing production.

## Handoff

Not started.

## Log

- 2026-10-10 22:37Z codex@desktop-g60ojud: created
- 2026-10-10 22:37Z codex@desktop-g60ojud: claimed
- 2026-10-10 22:45Z codex@desktop-g60ojud: note: Confirmed host,cuda,ai-test via supplied Python 3.14 / torch 2.11.0+cu128 and AI_Test path. Actual 5XL shape and native integer/incremental/chunk contracts pass. Separate mixed-policy exploratory pilot uses cache ccf6f46945b83ad3 with fresh XL/4XL/5XL controls; production and T0012 untouched.
