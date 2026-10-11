# Fixed-work search diagnostics

60 recorded games (team counts [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]) of arm v12_ponly_pxxl_vm; C08 at 128 samples per search call, deadlines removed. Changed: replies that differ from the same carrier's A. Root digest mismatches must be 0. Replay seconds: wall time of the replay processes (run --workers at a time) on this machine, not reply latency.

| Config | Turns | Changed | Root digest mismatches | Truncated | Ply-2 predictions | States built | View hits | Memo hits | Alternatives | Draws differ (diag) | Top changed first / final (diag) | Replay s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| c08_A | 24000 | 0.0 | 0 | 0 | 2903001 | 2628694 | 0 | 0 | 0 |  |  | 244.3 |
| c08_B | 24000 | 0.0832 | 0 | 0 | 2904084 | 2629777 | 0 | 3297485 | 127228 |  |  | 253.6 |
| c08_C | 24000 | 0.0775 | 0 | 0 | 28025261 | 24855868 | 1319478 | 0 | 0 |  |  | 2010.8 |
| c08_D | 24000 | 0.1461 | 0 | 0 | 30514330 | 25612937 | 3038637 | 3283953 | 127228 |  |  | 2644.3 |
| c08_diag | 24000 | 0.0 | 0 | 0 | 30928262 | 24856028 | 3948012 | 28025261 | 0 | 0.1089 | 0.102 / 0.060 | 2809.5 |
| v12_A | 24000 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | 86.0 |
| v12_B | 24000 | 0.044 | 0 | 0 | 0 | 0 | 0 | 0 | 131159 |  |  | 139.4 |
