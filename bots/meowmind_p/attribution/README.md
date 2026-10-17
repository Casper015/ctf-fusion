# bots/meowmind_p/attribution/

Original Meowmind with its search on and off, for T0078's gap attribution (task T0075).

| Arm | Wrapper | NN_SEARCH |
|---|---|---|
| ON | [`meowmind_search_on.cpp`](meowmind_search_on.cpp) | 1, the source's own default: the root search over the best 32 candidates and the beam |
| OFF | [`meowmind_search_off.cpp`](meowmind_search_off.cpp) | 0: the prior-ordered first candidate, with its shots |

Each wrapper defines `NN_SEARCH` and includes [`../meowmind-ctf-374-20261011.cpp`](../meowmind-ctf-374-20261011.cpp) (sha256 `cc7b730c…`, unmodified; see [`../README.md`](../README.md)). The source reads the macro at about line 680 (`#ifndef NN_SEARCH`). Its predictor, value net, time budget (1,600 ms on the first turn, 12 ms after) and output are untouched. With search off, the bot still runs the opponent prediction and computes shots for the first 32 candidates; it only skips their evaluation. The two wrappers, and therefore their bundles, differ in exactly one line. [`tools/screens/next_controls/MANIFEST.json`](../../../tools/screens/next_controls/MANIFEST.json) records that diff and the source hash. The tests also check that ON preprocesses (`-E -P`) to exactly the original source.

Meowmind writes nothing to stderr, so these arms have no work counters. The preflight adapter ([`tools/screens/next_controls/telemetry.py`](../../../tools/screens/next_controls/telemetry.py)) reports them as absent, never as zero. Their telemetry is the runner's reply count and timing.

```bash
python bots/p_only/tools/build.py --variants bots/meowmind_p/attribution/variants.json
```
