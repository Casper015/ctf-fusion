# Meowmind RL opponents: training lineages and held-out test lineages (T0058)

Casper asked on 2026-10-15 to train the Meowmind RL agent against the bots of the original test repository, AI_Test, while testing stays on bots it never saw. The split is by **whole lineage**: a bot derived from another (copied code, the same macro layer, a tuned variant) belongs to the same side. Otherwise a "held-out" opponent would be a near copy of a training opponent.

## The split

| Side | Family (lineage) | Bots | Source |
|---|---|---|---|
| train | gpt | Sol, Astra; Astra_Carrier, Astra_Full, Astra_Habits, Astra_Joint | judge copies (T0055); `ctf-benchmark-v3/track1_10k_suite` |
| train | claude | Opus, Sonnet | judge copies (T0055) |
| train | deepseek | DeepSeek | judge copy (T0055) |
| train | qwen | Qwen | judge copy (T0055) |
| train | meowmind | Meowmind, Meowmind + P3 distilled | this repository (T0055) |
| train | player5 | Player5; Control_Hybrid, Control_Omega, Player5_Omega_r3, Player2_v005; the 21 C-series/R3/G bots; the 9 S1/S2 controls | `ctf-stress-pack`, `ctf-benchmark-v3`, `ctf-s1-controls` |
| train | baseline | Baseline | `ctf-stress-pack` (T0055) |
| train | player1 | Player1 | `ctf-stress-pack` |
| train | player3 | Player3 | `ctf-stress-pack` |
| **test** | fable, mimo, kimi, glm | Fable, MiMo, Kimi, GLM | judge copies, and the stress-pack copies of Fable, GLM and Kimi |
| **test** | player2, player4, player6 | Player2 (the original), Player4, Player6 | `ctf-stress-pack` |

Why the Player5 family is so large:
- **Player2_v005 and Control_Hybrid** say in their headers that they were "built from … ref-player5.cpp".
- **The S1/S2 controls** are built on the Omega fusion with "plain Player5 macro layer".
- **The C-series and R3 bots** are Player7's experiments on Player5 Omega.

That makes them one lineage, so they share one family's 11.1% of opponent seats: the cap counts lineages, not files. `C01_Control_Hybrid` and `A0_Control_Hybrid_AA` are byte-identical to `B0_Control_Hybrid` and appear once.

The stress-pack Player1, Player2, Player3, Player4 and Player6 are independent bots (their headers describe their own designs and no Player5 code).

## Files

- `profiles/meow_rl_training_pool_v2.json`: 50 bots, 9 families of equal share (1/9 = 11.1%, under the 0.15 cap). It is T0055's ten bots unchanged, plus 40 AI_Test bots pinned by sha256 from AI_Test commit `3bab99c`. Its `held_out` field names the test lineages.
- The screens (`profiles/meow_rl_screen.json`, T0056) draw their opponents only from held-out bots that `profiles/pool.json` lists: Fable, MiMo, Kimi, GLM and Player4. Player2_v005 left the roster, because it is Player5 lineage. Player2 (original) and Player6 are held out too but not in `profiles/pool.json`; registering them there is a separate change.
- `tests/test_meow_rl_pool_split.py` checks:
  - the pool loads under the cap;
  - no training bot is a held-out bot or one of their files;
  - the screen roster is held out;
  - T0055's pool is contained in v2 (once merged);
  - with an arena checkout, every pin.

## Training on it

This is a new host training run (the next task on the board): fresh seeds reserved in the ledger, the same recipe, both arms. The candidate it produces is a separately named arm. Whether it replaces T0055's predeclared final checkpoint in the primary screen must be decided before any screen game, never after one.
