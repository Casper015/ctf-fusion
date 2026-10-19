# bots/challengers/

External bots entered for the strongest-bot ranking, kept unmodified with their provenance. None is a production default; none has host evidence yet.

## ctf_duel_challenger

| | |
|---|---|
| File | [`ctf_duel_challenger.cpp`](ctf_duel_challenger.cpp), unmodified |
| sha256 | `79bb510681ce0ca1958c96f3eee423e8e89af11b58b7e150def2db3dfa2c3f5d` |
| Received | uploaded by Casper on 2026-10-19 as `ctf_duel_challenger.zip` (sha256 `31f3c8725842a6ee8e25a2ed9c0e0a56fc448850d1c6d4c19289ecbe6c94f3ae`, one file) |
| License | MIT, "Copyright (c) 2026 meow-ai-arena contributors"; the notice is in the file |
| What it says it is | "MeowMind: CPU-only learned policy/value search and online opponent beliefs. Based on ctf/bots/gpt-6-astra.cpp protocol; trained on official bot policies." Defaults: `NN_WIDTH 32`, `NN_SAMPLES 4`, `NN_BEAM 5`, `NN_BUDGET_MS 12` |

It is a different program from the original Meowmind in [`bots/meowmind_p/`](../meowmind_p/README.md): outside the weight arrays, about 1,600 lines against about 740. It includes only the C++ standard library and has no file, process or network calls. `bots/p_only/tools/build.py` bundles and compiles it with the arena's `g++ -std=c++20 -O2`.

The development-machine check (T0093) replayed the 2-, 4- and 15-team fixture streams of `bots/p_only/tests/fixtures`, and the bot answered all 400 turns of each. That shows only that it runs. It is not timing or strength evidence.

## Building the contenders

[`variants.json`](variants.json) adds the challenger and includes the lists that hold the other contenders:

```
python bots/p_only/tools/build.py --variants bots/challengers/variants.json \
    ctf_duel_challenger meowmind_search_on meowmind_p3_xxl meowmind_p3_distill_xxl agent_v2_p2xxl_direct
```

| Variant | What it is | Host evidence so far |
|---|---|---|
| `ctf_duel_challenger` | this upload | none |
| `meowmind_search_on` | the original Meowmind, unmodified, search on | T0078: 0.835 utility, above `agent_v2_p2xxl_direct` by +0.066 [+0.049, +0.083] (Bonferroni) |
| `meowmind_p3_xxl`, `meowmind_p3_distill_xxl` | Meowmind with our P3 added to its opponent mixture | none on the host (a Mac pilot showed no difference) |
| `agent_v2_p2xxl_direct` | our production default | T0078: 0.769 |
