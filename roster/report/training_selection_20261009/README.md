# Recommended training opponents after the host light screen

Reviewed snapshot: `e85ff1a91f6113e7f5ea179ed47bdf4b51c19409`. Client review date: 2026-10-09. This delivery selects opponents and changes a **separate proposed collection configuration**. It does not replace `profiles/open_training_pool.json`, run matches, train models, or establish a learned-strength improvement.

## Decision

Keep an open, behavior-diverse pool rather than the top 15 or top 30 by score. The proposal retains **56 P-label policies**, adds two separately marked augmentation programs, merges three redundant operational representatives, excludes one inert program, and reserves two whole source lineages for fresh-map evaluation. The resulting 58-program pool is a pilot, not a permanent size limit.

Use the official benchmark bots as the main prediction targets, while keeping a bounded share of our stronger descendants and weaker useful styles. The existing `field` component is used here for **benchmark targets plus selected challengers**; this does not assert that every official bot is a strong teacher. Every retained policy also enters the style component.

The proposed field has the eleven judge bots plus these seven challengers:

| Challenger | Screen utility [marginal 95% interval] | Selection reason |
|---|---|---|
| `Player5_v12_rollout_volley` | 0.770 [0.723, 0.818] | Strong continuation candidate; one representative for the rollout family, rather than every fork |
| `cand-mc1` | 0.745 [0.680, 0.810] | Strong separate observed family; add mc2 only at its count-specific enrichment |
| `Control_external_CL` | 0.743 [0.676, 0.811] | Strong observed alternative; retain under the related-family cap |
| `R3_EXPECTIMAX_sampled` | 0.739 [0.677, 0.800] | Strong observed alternative; screen timing passed, without treating it as a deployment approval for a larger integrated bot |
| `C04_G_PLAYOUT_4ply` | 0.722 [0.653, 0.792] | Strong alternative that also supplies observed voluntary-drop events |
| `cand-012` | 0.711 [0.639, 0.783] | Strong additional source lineage and distinct observed behavior |
| `New_AI_Test/Player5/workspace/submission/bot.cpp` | 0.677 [0.609, 0.744] | Important rare-drop source even though several newer variants score higher |

These are not statistically ordered against one another. The point estimates justify provisional inclusion; they do not establish the best seven or the globally strongest bot. Stable program IDs and source hashes, rather than shortened display names, are in [candidate_selection.json](candidate_selection.json) and [candidate_selection.csv](candidate_selection.csv).

Retain Sol, Astra, Sonnet, Fable and Opus as important strong/distinct benchmark targets; retain DeepSeek, Kimi, MiMo, Qwen, GLM and Doubao as real prediction targets with explicit benchmark-oriented exposure. All eleven share the benchmark field: this proposal does not strength-downweight the weaker judge bots, and their exposure increases relative to the active pool. Their weaker screen utility does not make their requests useless for P. Preserve Baseline and measured Player2/3/4, fortress, sparring, flags/volley/macro representatives as style coverage. Their role is not to imitate them as our own optimal policy.

## Rare behavior should survive strength-based pruning

Candidate-only results from each bot's 70 screen games show the following voluntary-drop coverage. This excludes hundreds of repeated reference appearances, which would inflate the apparent evidence for official bots.

| Source | Voluntary drop events | Drops followed by any friendly hit within 3 turns |
|---|---:|---:|
| New_AI_Test Player5 submission | 52 | 36 |
| `bots_v2/Player5.cpp` | 18 | 13 |
| Fable | 15 | 14 |
| `C04_G_PLAYOUT_4ply` | 7 | 5 |
| `C08_Player5_Omega_LIT` | 5 | 3 |

Preserve all five. Several leading newer rollout/macro bots showed no voluntary drops in this small screen. Zero observed is not proof the behavior is impossible. The follow-up hit may come from another teammate: it is not proof that the dropping unit executed a successful drop-and-shoot tactic. The ledger intentionally records event counts without using the currently suspect carrying-episode denominator for ranking.

Collect diverse legal opportunities for these behaviors and inspect event/map coverage. Merely keeping these programs does not ensure enough drop examples. The current admission `contexts` field is metadata, not a collection quota; a future targeted regime must be implemented explicitly. Never interpret the legacy pool `weight` as effective priority: the registry collector uses component/group sampling instead.

## Count enrichment and redundancy

Keep the existing count restrictions for additional versions, as **redundancy-control decisions**, not claims that those bots are weak outside the listed counts:

| Representative | Additional admission counts |
|---|---|
| Export `R3_HABITS_adaptive` | 5 |
| Archive `R3_HABITS_adaptive` | 2, 3, 4 |
| Export `R3_WEIGHTS_tuned` | 4, 5 |
| `cand-mc2` | 3 |
| `small_runner` | 2, 3, 4 |
| `round3_macro_candidate` | 2, 3, 4, 5 |
| `S1_C_Adaptive6p_Fast15` | 6 |
| `S1_D_CrowdHabit` | 3, 4, 5, 6, 7 |
| `Player4_v9_latest` | 6, 7, 9, 11, 12, 15 |
| `candidate-v002-risk` | 15 |

Merge C20 into export HABITS and C22 into export WEIGHTS: each pair has zero observed fingerprint disagreement and shares groups. Use v9 instead of v8 as an operational representative; their overall corrected excess is only 0.00087 and they are equivalent under the current instrument. This is not proof v9 wins more. Removing physical duplicate representatives also prevents them occupying two seats just because `max_copies=1` is per file.

Exclude the New_AI_Test Player0 submission from the primary pool: it has zero attacks, kills and captures in this screen, despite being highly distinctive. Random and Noise remain augmentation only and never supply P labels. Five unmeasured crashing perf forks remain excluded.

Reserve complete `lineage:AI_Arena_CTF_Player02_bot` and `lineage:ctf-arena_Player4_v2` families before main collection. Their measured novelty makes them useful unseen-policy evaluation targets. Existing selection/probe/screen maps have already been inspected: obtain fresh maps and do not call those existing maps an untouched final test. Do not later admit a renamed alias of a reserved lineage without a new evaluation design.

## Proposed sampling composition

```json
{
  "lambdas": {"field": 0.7, "style": 0.3, "hard": 0.0},
  "lineage_cap": 0.1,
  "augmentation": 0.02,
  "focus": 0.0
}
```

No unverified bot is labeled a hard case. Augmentation means 2% of **fixtures**, with one augmentation seat, not 2% of all seats. Final shares differ from initial component caps because sampling is without replacement. More related-family control may eventually require a shared-parent or per-fixture budget; the current settings are not an exact global cap.

Existing committed diagnostics and 10,000 new sampler draws per count give:

| Expected share, equal games/count and acting-team rows weighted by N | Active pool | Proposed pool |
|---|---:|---:|
| Eleven judge bots | 34.44% | 56.10% |
| Player5 + Control_Hybrid | 43.21% | 22.18% |
| Other styles and augmentation | 22.35% | 21.72% |

In the proposed pool the related-family share ranges from about 19.1% at 2 teams to 23.7% at 15; judge share ranges from about 58.5% to 53.9%. No count has forced repeat seats or a fixed composition warning. [sampling_comparison.json](sampling_comparison.json) records all counts, both equal-count and row-weighted summaries, simulation sample sizes and the proposed pool identity. This is **composition arithmetic**, not evidence that training or play improves.

The general model's count prior must still be declared: equal matches/count generate N team rows/turn. Independent dedicated-count models avoid that particular pooled-count imbalance, but do not remove within-count opponent/rare-event sampling choices.

## What the latest evidence does and does not establish

- Fingerprint v2 materially repairs the earlier action-target, A/A support and unmeasured-count issues. It reports 253 measured candidates, 84 probes on six maps/count, zero stale runs and no flagged stochastic bots. Six histories/count and two repeats still do not prove universal strategy equivalence.
- The light screen is complete: 64 candidates x 70 games = 4,480 games, 30 above / 11 level / 23 below the reference parity criterion, zero recorded faults/late/dead games, maximum later reply 32.911ms, median marginal 95% half-width about 0.0663. There are only five maps/count per candidate, sharing 70 fixtures across candidates, not 4,480 independent maps.
- It tested only a preselected subset of 253 bots. Its ranks are not Bradley-Terry ratings or head-to-head results among all 64. Confirm close differences using paired candidate deltas on fresh maps and more seats.
- Duel references are Qwen, GLM, Baseline, Qwen and Opus. A perfect duel result is not evidence against Astra or Sol. At 15 teams, candidate seats 0..4 were used; 5..14 remain untested in this screen.
- The coarse shortlist groups by overall mean excess <= 0.05; it does not preserve every strict joint/context/count distinction. Use it for pruning and targeted follow-up, not as a proof all members supply identical training states.
- Engine-total agreement validates kills/carrier kills/deaths/pickups/captures. It does not validate every derived profile: `trajectory.py` can misbook a same-settlement pickup/capture episode as unfinished. Threatened survival is a geometric proxy and drop-follow-up counts any friendly hit. Fix/recompute episode profiles before ranking delivery or claimed escape/handoff skill.
- Fingerprint validation still needs the full dependency/build/limit regime identity; a bounded check accepts two repeats with differing dependency hashes and limits. No contamination of this committed corpus was established. Fix the join before a larger reused corpus.
- The prior rebuilt pool used pre-screen assessment metadata, so its evidence fields could remain `fingerprinted` with no utility even though contribution prose mentions the screen. This delivery joins the matching screen fields before generating the proposed pool. All retained entries remain provisional, not confirmed by 10,000 independent games each.

Primary sources: [host run](../screen_light/RUN.json), [screen utilities](../screen_light/strength.json), [screen live profiles](../screen_light/live_profiles.json), [frozen fixtures](../screen_light/schedule.json), [fingerprint results](../results/behavior_distance.json), [shortlist](../selection/shortlist.json), and [active pool](../../../profiles/open_training_pool.json).

## Files and next execution

- [candidate_selection.csv](candidate_selection.csv): every one of the 64 screened programs, source hash, retained/merged/held-out/excluded status, count limits, roles and utility bounds.
- [candidate_selection.json](candidate_selection.json): same ledger plus controlling input hashes and script provenance. Original and generated pool names are both recorded; stable program IDs govern joins.
- [admission_recommended.json](admission_recommended.json): editable proposed admission with reasons, reserved lineages and sampling settings.
- [assessments_screened.json](assessments_screened.json): assessment metadata joined to the exact matching light-screen sources.
- [proposed collection pool](../../../profiles/open_training_pool_recommended.json): compatible with the existing registry collector, separate from the active pool.
- [reproduction tool](../../../tools/recommend_training_pool.py): read-only source analysis and sampler draws; no bot execution.

Reproduce the proposal from this evidence snapshot:

The reproduction tool checks canonical JSON identities and normalized-code hashes before writing. A changed corpus or controlling sampler requires a renewed review instead of retaining the old source-revision claim. Line-ending conversion alone does not change its canonical JSON identity; the active pool's original Windows input-byte hashes were verified to match the committed files after CRLF conversion.

```sh
python3 tools/recommend_training_pool.py --games 10000
```

On the designated host, first collect a small fresh-map pilot in a **new directory**:

```sh
python -m training.collect --ai-test AI_TEST_ROOT --pool profiles/open_training_pool_recommended.json --teams 2 5 8 15 --matches 100 --seed 20261019 --out data/raw_recommended_pilot_20261009 --workers 1
```

Replace `AI_TEST_ROOT` with the checkout path. Tune workers on the host without multiplying workers x teams beyond the measured competition load. This command was not run here. Verify the pool/source/engine identities, actual seat and group masses, valid P labels, independent maps, faults and rare-event coverage before expanding. Collector resume still skips files by name, so do not reuse another pool's output directory. Preserve these identities when constructing shards; the training manifest must distinguish the continuation mixture used for V.

Retain the observed drop producers in targeted legal follow-ups, and screen one omitted control per large near-family rather than every filename. Compare active-pool versus proposed-pool training at equal independent-map, optimizer and full-turn budgets. Evaluate integer P NLL/calibration/joint-target errors on fresh maps and reserved lineages, V error under the declared continuation regime, and equal-time bot outcomes. P source quality and V continuation quality are related but different: stronger recorded opponents are not our optimal-policy imitation labels, and V from mixed weaker continuations is not a bot-independent optimal value.

No production inference/search/trajectory code changed in this selection delivery. The known episode and identity issues are findings for the next implementation task, not silently repaired here.
