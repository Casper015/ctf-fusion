# T0011 delivery checks

Checks ran on integrated branch source `be8e96b408e5c20709c9090537ebd17c49d3c6b9`. Frozen measurement and real-weight native verification use `9a78345172010a207fe89c54af12704b8cc3fddf`. Report-only commits after this check do not change the tested code or models.

Both exported seed1 policies passed exact regeneration, copied V2/M array equality, actual registry dispatch for every count2-15, integer-accumulator parity with Python, and native runtime contracts on n02/n04/n15 recorded fixtures. See EXPORT_VERIFICATION.json. The manifest served_team_counts field lists explicit legacy profile routes9/11/13/14; the general-only registry also provides the fallback for all counts2-15. Compiler paths use ${REPO} for portability.

Two preliminary verification-only attempts failed on script import setup and an incorrect assumption about that profile metadata. They did not alter or invalidate training, checkpoints, comparisons or exports; their logs remain in build/t0011/artifact_attempt1 and artifact_attempt2. The corrected verifier is embedded in REPRODUCE.md.

| Command | Exit | Seconds |
|---|---:|---:|
| `python -m unittest discover -s tests -t .` | 0 | 198.501 |
| `python -m unittest discover -s bots/p_only/tests` | 0 | 769.165 |
| `python tools/repo/release_bots.py --out dist --check` | 0 | 73.818 |
| `python tools/repo/task.py check` | 0 | 0.101 |

Exact check summaries and log digests are in CHECKS.json. Full local check logs remain in build/t0011/final_checks and are not committed. Tests supply correctness evidence; their runtimes are not host inference-cost measurements.

The four T0013 carriers c08/v12 x p2/p3 also compile with the actual exported seed1 models. Binary, bundle and inlined-input identities are in CARRIER_BUILD.json. This compilation uses the integrated source above and spends no match seed.
