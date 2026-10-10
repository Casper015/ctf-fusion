# P/V Model and Carrier Review Bundle

Client review date: 2026-10-09. Latest reviewed source snapshot: [0a0badc](https://github.com/Casper015/ctf-fusion/commit/0a0badc). This bundle contains English reports and a proposed execution prompt. Publishing it changes documentation only; no inference, training, runner, configuration, or model code was changed by this review.

## Reading order

1. [Current status and V ablation review](CURRENT_STATUS_AND_V_ABLATION_REVIEW.md): read this first. It distinguishes published measurements from unconfirmed conclusions and records which integration defects remain open.
2. [Tier M results and the L/XL/XXL plan](../TIER_M_REVIEW_AND_L_XL_XXL_TEST_PLAN_7c3f26d.md): the controlled S-to-M offline comparison and size-evaluation design. Its original evidence snapshot is 7c3f26d; subsequent updates are explicitly marked.
3. [C08 versus v12 carrier review](../C08_VS_V12_FOUNDATION_REVIEW_e11c45b.md): the actual P pilot, carrier comparison, source findings, and corrections to the integration write-up. Its evidence snapshot is e11c45b.
4. [Next-agent execution prompt](NEXT_AGENT_EXECUTION_PROMPT.md): proposed repair and confirmation work, with concrete acceptance evidence. These tasks have not been implemented by this documentation delivery.

## What the evidence supports

| Evidence | Finding | Limit |
|---|---|---|
| S versus M offline, 7c3f26d | Integer movement NLL improves at all 14 counts; mean relative reduction 2.86% | One training seed; this does not measure match-strength improvement |
| CUDA benchmark, 29a6412 | L P/V epoch throughput improves 3.8x/1.9x on the RTX 4080 Laptop | 200k resident rows, one epoch; total wall-time gains are about 1.27x/1.31x |
| P carrier pilot, e11c45b | C08 M has higher absolute point utility; v12 gains more from its P hook and runs faster | 100 shared fixtures, four counts; no confirmed best carrier |
| V ablation, 0a0badc | V marginal point gain is +0.0011 on C08 and +0.0366 on v12 | Both marginal intervals include zero; all 80 fixtures overlap the earlier P pilot |

## Priority before larger runs

Repair C08's hypothetical-history contamination; declare and validate each carrier's actual P distribution and alpha; make V branch states and activation coverage correct; bind results to real artifact hashes; and retain complete timing/fault evidence. Then tune on fresh fixtures, freeze the chosen system, and confirm it independently over all counts 2–15.

Do not select a carrier solely by raw bot rank, largest before/after gain, or model size. Measure the complete system under the same CPU deadline. Keep C08 and v12 as two finalists until the integration and evidence defects are resolved.

The current source exposes experimental V, while the M manifests still record `value_approved: false`. The new pilot is useful exploration and does not override that approval boundary.

## Evidence chronology

- [7c3f26d](https://github.com/Casper015/ctf-fusion/commit/7c3f26d): M model matrix and offline reports.
- [29a6412](https://github.com/Casper015/ctf-fusion/commit/29a6412): CUDA ladder forwarding and measured training benchmark.
- [e11c45b](https://github.com/Casper015/ctf-fusion/commit/e11c45b): P carrier hooks, L exports, and two 100-fixture carrier screens.
- [0a0badc](https://github.com/Casper015/ctf-fusion/commit/0a0badc): experimental V hooks and two 80-fixture ablation screens. The earlier history, adapter, and evidence issues remain relevant.

This index supersedes any interpretation of an older report as a statement about every subsequent commit. Preserve the original snapshot-specific reports as an audit trail.
