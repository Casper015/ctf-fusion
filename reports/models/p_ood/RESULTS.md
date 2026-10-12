# T0025: deployed P2 on two reserved lineages

Both deployed integer P sizes predict the two recorded reserved lineages better than the unchanged legacy q_H on this fresh-map collection. XXL has a small lower joint NLL than XL. These are prediction results; production defaults and model exports remain unchanged.

| Reserved lineage | XL joint NLL | XXL joint NLL | XL − q_H, 95% map CI | XXL − q_H, 95% map CI |
|---|---:|---:|---|---|
| lineage:AI_Arena_CTF_Player02_bot | 1.832970 | 1.823553 | -0.939256 [-0.959608, -0.918903] | -0.948672 [-0.969009, -0.928335] |
| lineage:ctf-arena_Player4_v2 | 2.394960 | 2.383549 | -1.219012 [-1.250157, -1.187867] | -1.230423 [-1.261435, -1.199412] |
| both reserved lineages | 2.113965 | 2.103551 | -1.079134 [-1.098780, -1.059488] | -1.089548 [-1.109174, -1.069921] |

Equal-count XXL − XL: -0.010414 nats [-0.012779, -0.008049].

All figures above give each count 2–15 equal weight. Lower joint NLL is better. Per-count/per-lineage comparisons with uniform, ID-validation-fitted inertia and q_H, 10-bin calibration with map-bootstrap intervals, top-1 and active-only cohorts are in [OOD_RESULTS.json](OOD_RESULTS.json) and the compact [per-count table](OOD_RESULTS.md). A confidence interval containing zero is labelled `not shown`.

The collection used fresh master seed `20261250`, 300 distinct maps, six workers and official 2000/50 ms limits. At two teams, 40 separate maps balance the lineages (20 each); at counts 3–15, 20 maps per count contain one of each. R3_EXPECTIMAX_sampled is the fixed admitted reference and fillers have distinct recorded lineages within each fixture. Both lineages have 20 maps per count, and share map clusters at counts 3–15.

All 300 games and all 968,000 seat-turn replies are retained. There were zero all-seat faults and zero missing/invalid protocol replies. The primary cohort contains 224,000 canonical valid reserved-lineage rows, with one row per observed team turn and weight 1. q_H covers all 224,000 joint rows and 514,151 active units, with zero missing or off-grid active forecasts. Success latency summaries are collection diagnostics; this report does not measure P deployment latency or show a 50 ms strength gain.

The independent training provenance audit checked 21,000 original match headers, every recorded source hash against the lineage/alias inventory, and 22 original XL/XXL checkpoint/config identities. No reserved-source/name hit or unresolved inventory source was found. Exclusion is relative to that frozen lineage assignment, not a claim about unknowable historical authorship. Fresh-map checks include the original train/val/test index, inspected prior schedules and all 50,000 actual T0028 records; the formerly unrecorded tournament master was located as `20261010`. See [seed provenance](SEED_PROVENANCE.json).

The P2 weights and folded temperatures are unchanged. Native versus actual NPZ parity checked 320 raw P-head accumulators and joint NLL/confidence at tolerance 1e-12 in 28 count/size cases before full scoring. No mixture or OOD-fitted parameter enters primary scores. Inertia was recomputed on original ID validation rows and matched the original report parameters. ID test NLL/top-1/calibration numbers are existing reports for exactly matching artifacts, shown alongside each OOD cohort; their stride/importance-weighted trajectories differ from these complete OOD trajectories. This is not a causal estimate of an OOD penalty.

The independent [metric audit](INDEPENDENT_AUDIT.json) rechecked all score-array hashes and 168 size/cohort entries, including paired means, map-clustered SE/CI, coverage and ECE points. ECE bootstrap endpoints use the declared deterministic aggregator; the separate auditor does not independently repeat those percentile endpoints. No game or model prediction was repeated for that audit.

The first scoring entry stopped before audit/inference because a local variable shadowed the raw-audit function. The failure log is retained; the name was fixed before any affected computation. Official games were never replayed. No training, OOD temperature fit, artifact export, main/task-board change or production routing change occurred.

Reproduction sources and commands are in [REPRODUCE.md](REPRODUCE.md). Local checks and PR CI are separate delivery steps.
