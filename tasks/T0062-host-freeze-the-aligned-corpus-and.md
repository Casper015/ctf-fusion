---
id: T0062
title: "Host: freeze the aligned corpus and export the aligned Meowmind RL arms (T0061 runbook 1-3)"
status: active
owner: codex-decision-pilot@desktop-g60ojud
needs: [host]
writes: [bots/meow_rl/variants/aligned/, inference/meow_rl/aligned/, reports/host/meow_rl_aligned/CORPUS.json, tests/test_meow_rl_aligned.py]
depends: [T0061]
land: pr
branch: codex-decision-pilot/t0062-host-freeze-the-aligned-corpus-and
updated: 2026-10-15
---

## Goal

Run steps 1-3 of the T0061 host runbook (docs/MEOWMIND_RL_ALIGNED.md) on desktop-g60ojud. Casper asked for this on 2026-10-15. No training and no games.

1. Inventory runs/meow_rl_pool2/{large_v,small_v}_s101/batches. Every update_*.npz must have feats and value_feats, and a companion update_*.json. Record the file count and a batch inventory in the log.
2. Freeze the corpus. This never reselects:
       python bots/meow_rl/aligned/corpus.py --batches runs/meow_rl_pool2/large_v_s101/batches runs/meow_rl_pool2/small_v_s101/batches --out data/meow_rl_aligned/corpus
   Commit its MANIFEST.json as reports/host/meow_rl_aligned/CORPUS.json. Report the shortages and missing strata exactly as written.
3. Export, with PyTorch:
       python bots/meow_rl/aligned/export.py --corpus data/meow_rl_aligned/corpus --require-simd avx2
   It refuses a checkpoint whose bytes differ from its committed pin (bots/meow_rl/aligned.json). Never edit a pin to make it pass: stop and hand off blocked. Commit inference/meow_rl/aligned/ (headers, MANIFEST.json, EXPORT.json) and bots/meow_rl/variants/aligned/ through a PR.
4. Back up data/meow_rl_aligned/corpus off this machine. A second disk on the same host is not off-machine. Record the destination and the verified sha256 of the copy.
5. In the log, report:
   - per actor: greedy agreement, TV, KL, sampled agreement, saturation, native parity (AVX2 and scalar);
   - per critic (F, R, H, R0): MAE, RMSE, max, bias, saturation.
   These are diagnostics on the diagnostic split, not strength evidence.

## Done when

PR merged with CORPUS.json, inference/meow_rl/aligned/ (EXPORT.json) and the aligned variants; off-machine corpus backup receipt in the log

## Handoff

Not started.

## Log

- 2026-10-15 22:19Z claude-review@caspers-macbook: created
- 2026-10-15 22:20Z claude-review@caspers-macbook: amended goal
- 2026-10-15 22:38Z codex-decision-pilot@desktop-g60ojud: claimed
- 2026-10-15 22:56Z codex-decision-pilot@desktop-g60ojud: amended writes -> bots/meow_rl/variants/aligned/, inference/meow_rl/aligned/, reports/host/meow_rl_aligned/CORPUS.json, tests/test_meow_rl_aligned.py
