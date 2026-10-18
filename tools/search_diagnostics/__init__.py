"""Real agent_v2 branch diagnostics with faithful continuations (task T0085).

Input-bound diagnostic harness providing:
  - Causal root extraction with actually sampled direct baselines and nested candidate sets.
  - Faithful branch restoration in official engine.py and live Bot continuations.
  - Signed baseline gain Delta(a), pairwise rank utility, within-root ranking with ties.
  - Cross-fitting over repeats to eliminate winner's curse.
  - Simulator parity audit against official engine.py.
  - Opponent P fidelity audit against actual causal transitions.
  - Authoritative host pilot cost estimator.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "schemas",
    "extract",
    "branch",
    "parity",
    "p_eval",
    "analyze",
    "estimator",
]
