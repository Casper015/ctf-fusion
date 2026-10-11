# Authorized follow-on queue

The user explicitly resumed this experiment on 2026-10-11 and authorized
continuing through the following task queue until interrupted. This is an
execution record, not a replacement for the task specifications. Refresh remote
ownership, lanes, dependencies and seed ledger before starting each task.
Deliver every update through an independent branch and PR; direct main writes,
including automatic task-board mutations, remain disallowed by the user.

## Current execution

Finish T0011 under its frozen source, data, arguments and finalist rule.
Completed seeds1/2 and validation comparisons are retained. P2 seed3 restarts
from its original seed because the interrupted calibration had no checkpoint.
Then finish P3 seed3, validation3, held-out test comparisons and seed1 exports.
Check exports and required tests, produce the evidence report and PR.
Never select models, temperatures or routing settings from held-out results.

## Ready host work after the current locks release

| Task | Frozen input and measurement | Required result |
|---|---|---|
| T0010 | `collect_v_data.json`: old p2 `v12_ponly_xxl`, V off, seed20261201, 500 maps/count2-15; continue from existing receipts if present | 7000 kept games, continuation roots2/candidates4/repeats1, ranking dataset, faults/dropped roots/prefix mismatches/pair counts |
| T0019 | Recorded value games, then `screen_search.json` at corrected seed20261230; prior seed20261220 is consumed | Fixed-work diagnostics, 2240 equal-time games, interaction/A-A/counters, v12 and C08 regret with valid comparison replies |
| T0004 | Held-out rows and frozen small/large/XXL P; near/far distance, two-turn contact and carrier strata | Clustered NLL differences and decision-change counts; implement a tiered arm only if near concentration supports it and its separate code task is available |
| T0025 | Fresh maps against the two reserved opponent lineages; XL/XXL p2 | OOD-only offline losses/calibration/count intervals alongside ID test; freeze a new unseen data seed before collection |

T0010's carrier must not be silently upgraded to P3. T0019's old task-body seed
must not override its corrected log and configuration. T0022 seed20261240 must
never be repeated. Existing value matches are available on the host; neither
the new T0010 data nor its rank dataset existed at the preflight inspection.

## Dependencies and current lane reservation

- P3: T0011 -> T0013, real extraction/inference cost and 1960 games with V off,
  seed20261210, both carriers and an A/A reference.
- Search: T0019 -> T0005, validate four-turn state/history/deadline contracts.
- V: T0010 -> T0012 -> T0014, training then deployment cost and controlled screen.
- T0006: only after the finalist settings freeze; choose an unseen confirmation
  seed afterwards.

At the 2026-10-11 09:39 UTC remote-board inspection, T0030 was active under
`claude-ci@caspers-macbook`, reserving shared model/trainer/inference/test and
P-only code lanes. It prepares quantization/QAT/precision-search code separately
from T0011. Respect that reservation: do not modify its source or existing
production defaults. T0005's broad P-only code lane and T0025's layout-test change
need that reservation resolved before overlapping edits. Read-only analysis and
nonoverlapping host reports may proceed in isolated worktrees under the lock
protocol. Quantization experiments wait for code and model dependencies; their
deployment baseline follows T0013. Four-turn precision routing follows T0005.

## Resource and provenance discipline

No host timing, match, collection, fingerprint or GPU work may overlap this
T0011 CPU/GPU lock. During training, preparation is lightweight and read-only.
Preserve model, configuration, dataset, runner, compiler and binary identities;
freeze the actual measurement source before each run. Use bounded input loading
and inspect available physical/commit memory before choosing workers. Do not
close user applications or change system paging settings. The 40 GiB working-set
cap changes throughput; do not equate timings across resource policies.
