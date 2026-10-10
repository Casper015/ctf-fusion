# Moved: P-only bots and near-large / far-small P

This handoff (b3cddf5) is superseded by the self-contained package in [`bots/p_only/`](../../../bots/p_only/README.md):

- [`bots/p_only/README.md`](../../../bots/p_only/README.md): the P-only C08 and v12 bots (every opponent prediction from the policy network; the hand-written forecasts, habits and scripted opponent moves in the lookahead and playouts removed), how to build, test and screen them, and what is verified.
- [`bots/p_only/TASK.md`](../../../bots/p_only/TASK.md): the host agent's phases. Phase 0 is the offline structure analysis of every bot; then verification, the paired screen, near-large / far-small P, V, and confirmation.
