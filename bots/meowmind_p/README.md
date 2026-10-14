# bots/meowmind_p/

The meowmind CTF bot with our P3 XXL opponent policy added to its search. Two bundled single-file bots, the generator that builds them, and the development-machine pilot results (task T0041). **The pilot results are not host evidence:** they come from a Mac, and they show no difference from the unchanged bot.

meowmind (`meowmind-ctf-374-20261011.cpp`, sha256 `cc7b730cf77c087c8c7f415b3b7d07101c3f003924d3a845a2ea3b81588794bc`) is another author's bot. Casper chose to publish its source here, inside the two bundles. The source carries no license notice.

## The bots

| File | Policy P | sha256 |
|---|---|---|
| `meowmind_p_t0011_xxl_both.cpp` | T0011 P3 XXL, seed 1 (`inference/models_pf_p3xxl_vm`, header `a1c70248…`) | `a27b73001fd42c96bb9a76f0c7b4f317c7acf9038ad57b2941671b1fe98614d3` |
| `meowmind_p_distill_xxl_both.cpp` | T0032 `distill_s101` (8XL teacher, XXL student; model.npz `68d2e776…`, best val 2.07186) | `680ccd22501f99c59c0f4ee7cab1d170d1b1aa1078eecf563fc17d26785e3e14` |

Each is one C++20 file (standard library only) that the arena's `g++ -std=c++20 -O2` compiles. "both" means P is used in both places below, with `--own 0.5 --pexp 1 --prior 0.10`:

- **Own candidates.** meowmind enumerates our squad's 125 joint moves, orders them by a prior and searches the best 32. The prior becomes `0.5 * (meowmind's sum of unit log-probabilities) + 0.5 * log P(joint)`. A joint move is a candidate when every unit's move is legal (no wall, a dead unit stays) and no two units share a destination.
- **Opponent model.** P's unit marginals are one more expert in meowmind's opponent mixture (prior weight 0.10; meowmind's own experts sum to 1.0). meowmind reweights its experts each turn by how well they predicted the observed moves, so P keeps weight only if it predicts well. P is computed once per real turn for every team, from the raw TURN block with the previous block as history. Deeper search states reuse meowmind's head 0 for this expert.

Everything else is meowmind's: its value net, rollouts, shots, time budget (12 ms per turn after the first) and output.

## Rebuilding

```bash
python bots/meowmind_p/make.py meowmind-ctf-374-20261011.cpp models_pf_p3xxl_vm meowmind_p_t0011_xxl_both.cpp
```

This reproduces `meowmind_p_t0011_xxl_both.cpp` byte for byte. `MODEL_DIR` is relative to `inference/`. The patch stops if any anchor in the meowmind source is missing or repeated.

The distilled bot needs a model set that is not committed:

1. Take `distill_s101/config.json` and `model.npz` from T0032's backup (`reports/models/p_distill/RUNS.json` pins the model sha256).
2. Add `"features": fusion.families.record(fusion.schema.layout().schema, "p3")` to the config. Its digest is `326a4299784b00bd`, the same as T0011's export. `training/distill_policy.py` does not write this record yet, and the exporter refuses a run without it.
3. Export and build with `FUSION_REPO` pointing at the checkout holding the export:

```bash
python -m training.export_mixed --policy-run <distill_s101> --value-tier m --out inference/models_pdistill_xxl_vm
python bots/meowmind_p/make.py meowmind-ctf-374-20261011.cpp models_pdistill_xxl_vm meowmind_p_distill_xxl_both.cpp
```

The export's header sha256 was `1b1999b60f7ef89d597a14d0dd3c6fad8936b053bc210fcbdd4f0ce3df7e2e4f`. Only the policy is used; the set's V (models_m) is not called.

## Pilot results (Mac, not evidence)

These games used the meow-ai-arena `ctf` engine and its tournament scripts on a MacBook (10 cores, `g++-16`), 2 games at a time. Ratings are the arena's Bradley–Terry scale: baseline 0, +400 means 10 times stronger, 95% intervals from 200 map resamples. "Flags vs meowmind" is the mean per-game difference in flags captured, in the same games as unchanged meowmind, with a 95% map-bootstrap interval. In round 2 every bot, including the trivial baseline, timed out on at least one turn in 4% to 12% of its games (the meowmind variants 6% to 10%): that is Mac scheduling, not these bots.

Round 2: 1,408 games (640 two-player, 768 four-player), schedule seed 20261013.

| Bot | Rating | 95% interval | Flags vs meowmind |
|---|---:|---|---|
| distilled XXL, both | 428 | 380 to 472 | +0.71 (−0.01 to +1.50) |
| T0011 XXL, both | 426 | 381 to 466 | +1.00 (+0.17 to +1.84) |
| T0011 XXL, own candidates only | 397 | 360 to 438 | −0.10 (−0.77 to +0.49) |
| distilled XXL, opponent expert only | 387 | 346 to 426 | +0.24 (−0.45 to +0.88) |
| T0011 XXL, opponent expert only | 375 | 332 to 421 | −0.07 (−0.88 to +0.73) |
| meowmind unchanged | 374 | 338 to 414 | |
| distilled XXL, own candidates only | 356 | 322 to 393 | −0.11 (−1.14 to +0.92) |
| GPT-6 Astra | 221 | 184 to 258 | |
| GPT-6.1 Sol | 150 | 107 to 191 | |

Round 1 (352 games, two- and four-player, T0011 XXL only, schedule seed 20261012): the same T0011 "both" bot rated 293 (204 to 374) against meowmind's 294 (220 to 368).

Round 3, fifteen players (stopped at 132 of 240 games, 9 maps, schedule seed 20261014): the two "both" bots, meowmind, the 11 arena AIs and the baseline in every game, one game at a time. Fifteen bots on 10 cores time out more often than in rounds 1 and 2, which cuts the search of all three meowmind variants alike.

| Bot | Flags per game | Mean finish (of 15) | Flags vs meowmind |
|---|---:|---:|---|
| T0011 XXL, both | 8.84 | 3.98 | +0.29 (−1.48 to +1.56) |
| meowmind unchanged | 8.55 | 4.20 | |
| distilled XXL, both | 8.52 | 4.00 | −0.04 (−1.73 to +1.66) |
| GPT-6.1 Sol | 7.77 | 4.78 | |
| GPT-6 Astra | 7.07 | 5.26 | |

**Reading.** A gain over unchanged meowmind is not shown. In round 2 the T0011 "both" bot's own interval excludes zero (+1.00, +0.17 to +1.84) and the distilled one's touches it (+0.71, −0.01 to +1.50). But six variants were compared with one control and no correction for multiple comparisons, and the same T0011 "both" binary tied meowmind in round 1 (293 against 294) and in the fifteen-player games (+0.29, −1.48 to +1.56). The distilled and T0011 policies do not differ here. Using P in only one place shows nothing. Strength and timing on the host (50 ms per turn, representative concurrency, paired against an unchanged control with A/A copies) were not measured.
