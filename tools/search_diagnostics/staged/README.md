# Staged-search checks (T0087)

Offline correctness checks of `agent_v2::StagedSearch` (`inference/agent_v2/search.hpp`) and of the staged bots in
[bots/agent_v2/staged_search/](../../../bots/agent_v2/staged_search/README.md), which documents the design, the
counters and the host runbook.

```
python -m unittest discover -s tools/search_diagnostics/staged/tests -t .
```

`staged_check.cpp` (modes `match`, `interrupt`, `nan`, `repeat`, `schedule`; usage in its header) runs in
fixed-work mode on the recorded streams of `bots/p_only/tests/fixtures`. Its results are correctness checks, never
timing or strength evidence.
