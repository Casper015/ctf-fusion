# T0019 validation

The measurements use frozen source `05f4aca0acd2f4d31b101b9b71b4afa95c888330`; this branch adds only the task report lane.

| Check | Result |
|---|---|
| Independent raw audit under exclusive CPU lease | Passed:2240 raw games,400 selected branch roots, exact scheduled coverage, frozen identities, utilities, faults, candidate/repeat/source/root blocks, resolved pairs and clustered regret intervals |
| Paired statistics | Passed: all overall/band/per-count contrasts, both A/A studies and interaction match generated JSON |
| Aggregates | Passed: all eight arms' primary timing, PONLY/SEARCH/PLY2/PORTFOLIO counters and consistent settings |
| Core unittest suite |189 tests run:188 passed,1 platform skip |
| P-only unittest suite |43 tests passed |
| Single-file release package check | Passed;29 bundles |

Local tests and package checks ran sequentially under a separate exclusive CPU lease after formal measurement completion. They are verification fixtures, not additional tuning data or strength measurements. The recorded stage/check durations are wall time receipts, not same-condition capacity comparisons. Runtime warnings about test fixture file handles are retained in ignored logs; commands returned0.

No main/task-board mutation, model export, production default change, formal game replay or interference with the user desktop applications occurred. T0010 PR23 is handled by other reviewers and is not merged by this task.

`RECEIPT_TRANSPORT.json` maps report files' original raw bytes to logical LF bytes for Git transport. It does not alter the frozen ignored data/build files or redefine their measurement hashes. The independent audit uses the original frozen checkout and retained data as documented in REPRODUCE.md.

## Check command receipts

- `python -m unittest discover -s tests -t .`: exit0, 288.847 wall seconds.
- `python -m unittest discover -s bots/p_only/tests`: exit0, 1059.209 wall seconds.
- `python tools/repo/release_bots.py --out build/t0019/check_bundles --check`: exit0, 101.353 wall seconds.
