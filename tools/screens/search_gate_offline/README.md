# Offline search gate (T0076)

Replays the recorded n02/n04/n15 streams from every seat, records fixed-tree FP32 values and float64 backend error, and checks gate, tree-prefix parity and isolation. The analyzer applies the frozen rule in `docs/AGENT_V2_SEARCH_GATE.md` section 1. This measures numerical stability on fixed trees; it says nothing about winning games. Requires Python 3, NumPy and the repo's C++ compiler.

Quick check (seat 0, stride 100; all generated files stay under the ignored work directory):

```bash
python3 tools/screens/search_gate_offline/run.py --smoke --jobs 3
python3 tools/screens/search_gate_offline/analyze.py --trees build/search_gate_offline/TREES.jsonl.gz --out build/search_gate_offline
```

Full diagnostic (run by the caller, not by the smoke check):

```bash
python3 tools/screens/search_gate_offline/run.py
python3 tools/screens/search_gate_offline/analyze.py
```

Full outputs go to `bots/agent_v2/search_gate/diagnostics/`: reproducible `TREES.jsonl.gz`, command/provenance receipts in `INTEGRITY.json`, and generated `DIAGNOSTICS.json`/`.md`. A smoke/subset corpus cannot freeze a gate. The driver exits nonzero on an integrity failure; analysis records unsupported or failed validation as `frozen.blocked`. Options: `--games n02 n04 n15`, `--stride 4`, `--jobs N`, `--work PATH`; analysis also accepts `--trees PATH`, `--integrity PATH`, `--out DIR`.
