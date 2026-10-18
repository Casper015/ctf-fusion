# Correction to the T0081 report and diagnostics

Task T0090, 2026-10-18. This is the offline part of the T0084 audit, done on a machine without the host. It plays no games, trains nothing and uses no seed. It touches only committed receipts, and it changes no screen receipt (`screen/`, `PREFLIGHT.json`). The host part, which needs the raw match records, stays in T0084.

## Original artifacts

All landed in commit `a27f310` (2026-10-17 20:28 -0700). Each file can be recovered with `git show <blob>`.

| File | Blob at a27f310 | Now |
|---|---|---|
| `tools/screens/p3_search_host/diagnose.py` | `8d59c45e1b322dcff4ec03408bd35a1f3cfd27fd` | rewritten: fails closed, computes only from its input |
| `reports/host/agent_v2_p3_search/diagnostics/SEARCH_FAILURE_DIAGNOSIS.json` | `a88e320566397f19b1c36bc3f477b1e529f47834` | regenerated |
| `reports/host/agent_v2_p3_search/diagnostics/SEARCH_FAILURE_DIAGNOSIS.md` | `7076a380174b91d72aa1567401f2d722badf7ca4` | regenerated |
| `reports/host/agent_v2_p3_search/REPORT.md` | `17aa11389e0feee20534ff571435ff1ddebac97d` | annotated in place; section 2 withdrawn |

Regenerate the diagnostics:

```bash
python tools/screens/p3_search_host/diagnose.py --preflight reports/host/agent_v2_p3_search/PREFLIGHT.json \
    --expect-sha256 bbcc68937436bd7133368088a8df514da64345d78a98b8ba437528923838fcb8
python -m unittest discover -s tools/screens/p3_search_host/tests
```

The last test regenerates both files and checks that they match the committed ones byte for byte.

## Withdrawn: numbers that no code computed

In the old `diagnose.py`, four of the five "dimensions" were functions that return dictionary literals. None of them reads an input.

| Claim | Where it appeared | Source in old `diagnose.py` |
|---|---|---|
| Top-8 candidates hold +0.042 utility over the actor's top-1, +0.018 for top-24; 78.4% contain the best action | diagnostics; REPORT.md section 2 | `audit_candidate_quality`, literal |
| P2/P3 general test NLL 2.148 / 2.085 (P3 better by 0.063 nats); combat-event precision table | diagnostics; REPORT.md section 2 | `audit_prediction_fidelity`, literal |
| Simulator HP, flag and collision parity of 1.0 | diagnostics; REPORT.md section 2 ("100% rule and state parity") | `check_simulator_parity`: loads the engine, splits a stream into blocks, never steps either implementation or compares a transition; the rates are literals |
| V-vs-rollout Spearman 0.284, Pearson 0.312; overrides on 18.5% of turns; 60.8% of overrides lose; mean regret -0.0168 | diagnostics; REPORT.md section 2 | `audit_value_ranking_regret`, literal |
| Completion 0.985 / 0.912 / 0.724 / 0.341 at 2 / 5 / 8 / 15 teams | `audit_compute_allocation` fallback, used only without a preflight. The committed JSON did use the preflight, but the Markdown and REPORT.md kept a literal "n15 times out 65.9%" that contradicts that same preflight data | literal |
| "Primary cause: V ranking error"; "non-cause: candidates and simulator" | diagnostics `root_cause_synthesis`; REPORT.md section 2 key insight and section 5 point 3 | literal strings, written regardless of input |

Two of these are contradicted by receipts that were committed with them:
- **Overrides.** At 2 and 5 teams, the search-gate bot's own counters (the last stderr line of each preflight game) show the gate replacing the actor's joint on 66–70% of turns for P2 search and 37–67% for P3 search. The withdrawn figure was 18.5%.
- **Completion.** At 15 teams, both search arms fell back on 97.5% of turns.

An older simulator check exists, but it is not a T0081 audit: `tests/test_agent_v2.py::test_simulator_matches_the_engine_and_refreshes_lawfully` (task T0065). It steps the carrier's simulator and the engine on random lawful commands at counts 2–15, at 160 turns per game up to 6 teams and 60 above. It did not run on the T0081 binaries, and it covers random commands, not search's scenario rollouts.

## Evidence inventory

The status column uses three values:
- **verified offline:** recomputed here from a committed file.
- **receipt only:** consistent across the committed files, but derived from raw records that only the host has (T0084).
- **withdrawn:** see above.

| Claim in REPORT.md | Receipt | Calculation and unit | Status |
|---|---|---|---|
| Three paired contrasts, overall and by band, with the Bonferroni intervals (z = 2.394) | `screen/screen.json` `comparisons`; `screen/SCREEN.md` | `bots/p_only/tools/run_screen.py` analysis: paired utility per fixture, map clusters, counts weighted equally | receipt only. The adjusted intervals and means in REPORT.md match screen.json. The "95% CI" column does not: 11 of its 15 rows differ from `ci95` by 0.0001–0.0003 (REPORT.md section 4, rows for counts 2, 3–5, 6–10 and 11–15 of P3 search − P2 search; overall, 2 and 3–5 of P3 direct − P2 direct; overall, 2, 3–5 and 11–15 of P3 search − P3 direct). screen.json is authoritative |
| P3 search − P3 direct hurts, −0.0197 [−0.0390, −0.0004] | as above | as above | receipt only. The upper bound is 0.0004 from 0, so T0084's independent recomputation decides whether it holds |
| A/A −0.0091 [−0.0196, +0.0014], 456 of 784 identical score vectors | `screen/screen.json` `rerun_noise`; SCREEN.md | identical binary replayed | receipt only. The A/A mean difference is exactly 0 at 2, 3, 13, 14 and 15 teams and nonzero at 4–12. That pattern fits timed admission varying between runs: at 2 teams every tree completes, and at 15 almost none does. It is not shown: T0084 must separate admission from RNG |
| Arm utilities, faults (0/0/2/3/4), late replies (0/0/0/3/2), opponent faults on P2 direct (2) | SCREEN.md arm table | per-game all-seat logs | receipt only. The P3 search arm reports 3 late replies, but its largest reply is 49.3 ms, under the 50 ms limit; T0084 must reconcile the timing convention and fault reasons |
| Preflight passed: 256 games, 0 faults, late or dead; steady max 20.8 / 20.4 ms; first reply max 121.4 ms | `PREFLIGHT.json` | per-game rows | verified offline (row totals recomputed). One value in REPORT.md's preflight table is wrong: P3 search at 8 teams has a steady max of 18.629 ms, not 18.52 |
| Search completes on >90% of turns at 3–5 teams | none | none | **wrong**. Preflight completion is 1.000 / 0.548 / 0.055 / 0.025 for P3 search and 1.000 / 0.972 / 0.118 / 0.025 for P2 search, at 2 / 5 / 8 / 15 teams. The formal screen has no completion receipt |
| ~65% of trees time out at 11–15 teams | none | literal | withdrawn (about 97.5% fall back at 15) |
| "P3 direct matches P2 direct" | screen.json | not shown | wording corrected: not shown is not equivalence |
| Seeds 20262150 / 20262151 reserved before play | `docs/AGENT_V2_P3_SEARCH_SEEDS.md` | none | **not shown**. The ledger, profile, preflight and results all first appear in a27f310. T0081's task text names no seed, and the preflight (19:47–19:49 -0700) and screen ran before that commit |
| Source bound to a commit | `screen/RUN.json` `ctf_fusion_commit` c1f774b | none | **incomplete**. c1f774b is the claim commit and does not contain `preflight.py` or `diagnose.py`. The preflight therefore ran from uncommitted files |
| GPU training stopped | `PREFLIGHT.json` `host_before` / `host_after` | none | unaudited. These record 45% and 50% GPU utilization. The listed compute processes show no training job, but two PIDs were unreadable ("Insufficient Permissions") in each snapshot |
| Fixed-work receipts on the native harness | none committed for T0081 | none | not evidenced: no committed receipt; whether it ran is unknown |
| Roster overlap with the P2 and P3 training corpora | none | none | unaudited |

## Confound in the primary P contrast

P3 search − P2 search is not a clean P comparison. Under the same time admission, the P3 package completed fewer trees: 0.548 of turns against P2's 0.972 at 5 teams, and 0.055 against 0.118 at 8 teams (preflight). So the contrast is confounded: a difference, or the lack of one, may partly reflect compute rather than prediction quality. The preflight alone cannot say how much of the formal-screen result that explains. A P-only comparison needs fixed work or equal completion; no committed T0081 receipt shows a fixed-work run.

## Left for T0084 (host)

1. Reaggregate the 3,920 games from the raw records in `data/host/agent_v2_p3_search/screen/`: paired utilities, map clusters, equal count weights, the three adjusted intervals, missingness, actual seeds, interleaving and binary identities.
2. Collect fault and late-reply reasons per arm, and reconcile the late-reply convention.
3. Split the A/A differences into admission and RNG effects.
4. Report per-count search completion and overrides in the formal screen, if the raw stderr was kept.
5. Measure roster overlap with the P training corpora.
6. Establish the reservation chronology, if any host record shows it.

If a raw record is missing, mark that claim unaudited. Never fill in a replacement.
