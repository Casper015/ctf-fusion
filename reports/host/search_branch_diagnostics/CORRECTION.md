# Correction of the T0086 report (task T0092)

The T0086 collection and its receipts in this directory stand. The report built on them does not: at commit 5c20cc4,
`REPORT.md` was written by hand, and `tools/search_diagnostics/summarize_pilot.py` wrote the four gate decisions,
their rationales and the source, root and episode totals as literals. [REPORT.md](REPORT.md) is now rendered from
[REPORT.json](REPORT.json), which `summarize_pilot.py` computes from the receipts. The withdrawn text is
`git show 5c20cc4:reports/host/search_branch_diagnostics/REPORT.md`.

## Changed decisions

- **Gate A (T0088): changed from permitted to not permitted. Gate B: changed from rejected to applies.** Gate A was
  permitted on `in_sample_best_gain`, the best of eight noisy candidate means. `analyze.py` defines that quantity as
  "biased upward, never a demonstrated gain". T0086's own goal prescribes selecting on some repeats and evaluating on
  disjoint ones, which is `cross_fit_gain`. Over all analysed roots, cross-fit gain is not shown, and no split/stratum
  group or stratum has an interval. What follows is uncertainty, not evidence that no opportunity exists. T0088 now
  waits on this task.
- **Gate C: not triggered, unchanged.** The scope is narrower than the report claimed (see below).
- **Gate D (T0087): permitted, unchanged.** It now rests on the completion rates computed in T0084's
  `EVIDENCE_AUDIT.json`, not on literals.

The gate rules are in `GATE_RULES` and in the report. T0086 named the gates but registered no thresholds, so T0092
wrote these rules after the collection.

## Withdrawn statements

| Withdrawn | Why |
|---|---|
| "Candidate headroom exists ... +5 to +10 percentage points" | It is the in-sample maximum, biased upward. Its cross-fit counterpart is not shown. The figures are utilities, not percentage points |
| "V has virtually zero ranking ability", "mean rho ~ 0.01" | No group has an interval. Over all roots, v_spearman is not shown, and its interval includes zero |
| "the candidate selected by V yields flat or negative realized gain" | v_gate_gain over all roots is not shown either way |
| Extended set: "+0.095 to +0.134", "v_gate24_gain ~ -0.003 to +0.008" | No committed receipt yields these figures. The extended groups carry no v_gate24_gain. Recomputed from the per-root rows, seven of the fourteen counts have no analysed root |
| "Mechanics mismatch is completely ruled out", "100% faithful" | PARITY.json found no difference within its coverage: four team counts and the listed event counts, some as low as 3. That coverage is all it shows |
| "Opponent misprediction is not the root cause" | P was evaluated on factual recorded states only, not on branched states |
| "The primary root cause of search failure is value estimation noise at the root" | This assigns a unique cause, which T0086 forbids. The data do not show it |
| "41.8% A/A scheduling jitter" | 328 of 784 A/A pairs had different score vectors. T0084 found that 315 of them also differ in search admission totals, which is consistent with admission variability but does not separate it from RNG |
| "Count-weighted mean NLL" table | Each model was averaged over its own counts with a value, so P2 and P3 could cover different counts, and an empty list became 0. It is recomputed over the counts where both models have a value |
| Source, root and episode totals | They were literals. They are now read from the receipts. The number of branches played is in no committed receipt and is marked unavailable |

## Omitted from the original report

Replay refused 95 of 334 primary roots and 15 of 27 extended roots. Nearly all were refused because Astra (a few
because Sol) diverged from its recorded game. The analysed roots are therefore not a random subset of the
registered ones: at 10, 12 and 15 teams, only 3, 6 and 3 of 24 roots are analysed. Any follow-up that reuses these
rows inherits that selection.
