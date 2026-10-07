# Benchmarks at 5+ teams, size by size

**Sources:**
- **Data:** the `ctf-benchmark-v3` per-game logs from GitHub, as fetched by `tools/fetch_benchmark.py`:
  - commit `a2da2a48cb08`;
  - track 1, `games.jsonl`: 90,207 games;
  - C-series, `games_c_2000.jsonl`: 44,000 games.
  The `.jsonl` and `.jsonl.gz` copies hold identical records.
- **Tables:** in [BENCHMARK_5PLUS_TABLES.md](BENCHMARK_5PLUS_TABLES.md); raw numbers in [benchmark_5plus.json](benchmark_5plus.json). Both are produced by `tools/benchmark_5plus.py`.
- **Cleaned games:** 83,930 games with 5+ teams, one row each, in `benchmarks/games_5plus.csv.gz`.

**Coverage:**

| Requested size | 5 | 6 | 7 | 8 | 10 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| Played | yes | yes | **no** | yes | yes | **no** | **no** | **no** | yes |
| Fixtures per variant | T1 ~1,250 / C 250 | same | – | same | same | – | – | – | same |

Neither track ever played 7, 12, 13 or 14 teams, so nothing can be said about those sizes from this data. The official rating weights all 14 sizes equally, so a future benchmark should include them.

**What the logs lack.** They carry captures, pickups, carrier kills, carrier deaths, drops and timeouts per game. They have no per-turn latency and no kills or deaths outside carrier fights, so "latency impact" below means timeouts.

**Noise floor.** At one size, a C-series cell has a 95% interval of about ±2.5 pp; a track-1 cell about ±1.2 pp. `+`/`-` marks an interval excluding zero, `++`/`--` |z| > 3. Across ~200 per-size cells, about 10 single `+`/`-` marks are expected by chance.

## 1. Key comparisons by size (pairwise score, pp)

| Comparison | 5 | 6 | 8 | 10 | 15 |
|---|---:|---:|---:|---:|---:|
| Omega vs Control_Hybrid (T1) | **+5.8++** | -0.4 | +1.9+ | **+2.8++** | +0.9 |
| Omega vs Control_Hybrid (C) | +5.5+ | -1.3 | +0.9 | +2.4 | +1.5 |
| RISK gene vs Omega (C) | -0.5 | +1.3 | +0.7 | +2.5 | +2.0 |
| Astra habits + carrier DP vs Omega (T1) | +0.6 | +1.2 | +1.8 | +0.8 | +0.8 |
| Astra habits + carrier DP vs Player2 v005 (T1) | +0.7 | -0.1 | +0.6 | +0.7 | -0.6 |
| Expectimax vs foundation, both copies (C) | +1.8 | +4.5+ | -0.0 | +4.0+ | +1.6 |
| Player2 v005 vs Control_Hybrid (T1) | **+5.6++** | +0.9 | **+3.1++** | **+2.9++** | +2.4+ |
| Omega vs Player2 v005 (T1) | +0.1 | -1.4 | -1.2 | -0.2 | -1.4 |

### Reading it size by size

- **5 teams.**
  - The fusion pays off most here: Omega +5.8 and Player2 +5.6 over Hybrid.
  - RISK adds nothing (-0.5).
  - Tuned weights *on the foundation* rank 20th of 21 (-3.1), although inside Omega they are +0.7.
- **6 teams.**
  - Nothing beats Hybrid by much: Omega -0.4/-1.3, Player2 +0.9.
  - Gains come only from plain-foundation additions: tuned weights +5.8++, expectimax +4.5+, our carrier DP +3.8+.
  - LIT is -2.0.
- **8 teams.**
  - Player2 leads (+3.1++ over Hybrid); Omega trails it by 1.2.
  - Expectimax gives exactly 0.0 here.
  - The habit models and SH lead the single factors (Astra habits +1.4, SH +1.4). CFL (-3.6-) and PLAYOUT (-3.4-) are the worst after FLAGS.
- **10 teams.**
  - Search helps most here: expectimax +4.0+, tuned weights in Omega +3.5+, RISK +2.5.
  - Astra habits + DP: +1.9++ over the joint sampler.
- **15 teams.**
  - RISK ranks first among single factors (+2.0); TERMINAL second (+1.9).
  - LIT is -3.1- even though most of its genes are positive alone. **The genes interfere** when the board is crowded.
  - Omega trails Player2 by 1.4, its largest gap. That is where its timeouts explode (section 2).

### How the ranking shifts as the board fills (rank among 21 single factors)

| Factor | 5 | 6 | 8 | 10 | 15 | Trend |
|---|---:|---:|---:|---:|---:|---|
| Gene RISK | 10 | 6 | 5 | 3 | **1** | rises steadily with crowding |
| Expectimax | **1** | 2 | 10 | **1** | 5 | strong except at 8 |
| Astra habits | 3 | 18 | **1** | 6 | 6 | strong at 5 and 8–15, not 6 |
| Astra habits + DP | 2 | 11 | 3 | 4 | 11 | strong 5–10 |
| Tuned weights (foundation) | 20 | **1** | 4 | 9 | 3 | erratic: worst at 5, best at 6 |
| TERMINAL | 6 | 17 | 8 | 14 | 2 | only at 15 (end-of-game pressure) |
| Genes LIT | 9 | 20 | 18 | 16 | 20 | falls with crowding |
| PLAYOUT | 19 | 13 | 19 | 20 | 7 | poor nearly everywhere |
| CFL | 14 | 5 | 20 | 18 | 12 | harmful at 8–10 (keep it for duels only) |
| FLAGS | 21 | 21 | 21 | 21 | 21 | last everywhere (-3.8 to -10.7 pp) |

**The pattern.** Risk aversion (RISK) and opponent modeling (habits) gain value as the board gets more crowded. Combining every search gene (LIT), deeper playouts (PLAYOUT) and our CFL prediction lose value.

## 2. Timeouts as the team count grows

**Timeouts per 100 games:**

| Bot | 5 | 6 | 8 | 10 | 15 |
|---|---:|---:|---:|---:|---:|
| Control_Hybrid (T1, no search) | 1.1 | 0.2 | 2.3 | 5.8 | **102.9** |
| Player2 v005 (15 ms search) | 1.0 | 0.2 | 3.7 | 6.9 | **127.1** |
| Control_Omega (30 ms search) | 0.7 | 0.1 | 3.0 | 7.0 | **237.7** |
| Player5_Omega_r3 (old rollout) | 0.9 | 0.1 | 2.8 | 6.5 | **263.9** |
| C-series search variants | 0–0.8 | 0.8–4.0 | 4–10 | 14–22 | **168–232** |
| C-series no-search variants | 0–1.6 | 0–3.6 | 4–10 | 12–23 | **95–104** |

**15 teams is a different regime:**
- In 96–100% of 15-team games every bot had at least one timeout, the no-search Control_Hybrid included (about 1 per game). That points at the harness or host load with 15 processes, not the bots' own logic.
- On top of that baseline, the 30 ms search bots add about 1.2–1.6 timeouts per game: twice Hybrid's rate, and 1.9× Player2's.
- Sequential halving (SH, 168) is the cheapest search variant, plausibly because it ranks fewer plans to the end.
- Below 15 teams, timeouts are rare (under 0.25 per game) for every bot.

**Omega's 30 ms budget costs it at 15 teams.** Omega trails Player2 most at 15 teams (-1.4) and 6 and 8 teams (-1.4, -1.2); at 5 and 10 the two are even. At 15 Omega has +1.1 timeouts per game more than Player2 (`++`). The earlier finding — more samples gave no gain (round-4 speed-up 0.0) — means the 30 ms budget buys nothing that offsets this. **Recommended test:** a 15 ms budget at 15 teams, or for every size.

## 3. In-game events (track 1, absolute)

| Size | Bot | Pickups | Captures | Conversion | Carrier deaths | Carrier survival | Carrier kills | K/D (carrier) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | Hybrid / Omega / Player2 | 21.6 / 21.1 / 20.9 | 12.1 / 12.5 / 12.5 | .56 / .59 / .60 | 9.3 / 8.4 / 8.2 | .57 / .60 / .61 | 7.3 / 9.7 / 9.9 | 0.79 / 1.16 / 1.20 |
| 6 | | 14.0 / 13.0 / 13.2 | 8.9 / 8.9 / 9.0 | .63 / .68 / .68 | 5.0 / 4.0 / 4.0 | .65 / .70 / .70 | 3.9 / 4.9 / 5.4 | 0.79 / 1.23 / 1.33 |
| 8 | | 17.0 / 16.2 / 16.7 | 8.5 / 8.6 / 8.8 | .50 / .53 / .53 | 8.3 / 7.5 / 7.7 | .51 / .54 / .54 | 7.1 / 8.8 / 9.4 | 0.85 / 1.18 / 1.22 |
| 10 | | 18.5 / 17.9 / 18.2 | 8.1 / 8.2 / 8.3 | .43 / .46 / .46 | 10.3 / 9.5 / 9.7 | .44 / .47 / .47 | 8.9 / 10.9 / 11.2 | 0.86 / 1.15 / 1.16 |
| 15 | | 18.6 / 18.0 / 18.4 | 5.7 / 5.8 / 5.9 | .31 / .32 / .32 | 12.7 / 12.2 / 12.3 | .32 / .33 / .33 | 10.1 / 12.4 / 12.6 | 0.79 / 1.02 / 1.03 |

Conversion = captures / pickups. Carrier survival = 1 − carrier deaths / pickups. K/D = enemy carriers killed / own carriers lost.

**What the event data shows:**
- **Crowding halves delivery.** Conversion falls from about 0.6 at 5 teams to 0.32 at 15 for every bot, and carrier deaths per game rise from ~8–9 to ~12–13. At 15 teams two of three carries die.
- **Carrier fights are where search pays.** Omega and Player2 cut carrier deaths by 0.6–1.0 per game against Hybrid at every size (`--`). They kill more carriers too: K/D 1.0–1.3 against Hybrid's 0.8. The paired carrier-death deltas are significant at every size for Omega, Player2 and expectimax.
- **The edge shrinks with crowding.** Carrier K/D advantage falls from +0.4 (5–6 teams) to +0.2 at 15. More opponents means more third-party fire that no two-ply model sees.
- **Astra_Full** has the lowest carrier deaths among track-1 bots at 5 teams (7.3) and their best K/D at 5–8 teams (1.27–1.42).
- **LIT** has the best K/D of any bot at 5–10 teams (1.38–1.60; 1.09 at 15) but fewer pickups and captures. It plays carrier fights safely and passively, which the ranking does not reward at 15 teams.

## 4. Conclusions for the next build

1. **Per-size module masks are justified by data:**
   - RISK at 6–15;
   - Astra habits + DP at 5 and 8–15, not 6;
   - expectimax at 5–6 and 10–15 (its 8-team zero is worth a second look);
   - no FLAGS, no CFL above 2 teams, no LIT.
2. **Cut the search budget at 15 teams first.** The timeout regime there is the clearest explanation of Omega's deficit to Player2.
3. **5–6 teams behave differently:**
   - at 5 every fusion gains a lot;
   - at 6 nothing beats Hybrid except plain-foundation additions;
   - neither track's rosters vary within a size (each size has one fixed opponent set), so this may be roster-specific.
4. **Run 7, 12, 13 and 14 teams.** They have never been benchmarked, and the official rating weights them as much as the others.
