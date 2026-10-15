# Decision data: coverage, noise and power

No rule here kills or passes anything; these numbers size the next collection.

| Split | Roots | Maps | Changeable | Without outcome | Failed repeats | Noise roots | Within-action var | Between-action var (raw / net) | Paired gap var (implied cov) | Faults prefix / root / continuation | Collisions |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| train | 268 | 268 | 260 | 0 | 0 | 260 | 0.00729 | 0.01223 / 0.01041 | 0.01620 (-0.00055) | 0 / 0 / 0 | 0 |
| val | 37 | 37 | 37 | 0 | 0 | 37 | 0.00558 | 0.01729 / 0.01590 | 0.01191 (0.00003) | 0 / 0 / 0 | 0 |

Assumptions:

- Every statistic below is computed over one population: the noise roots, i.e. roots with at least two noise candidates (the incumbent and supported candidates with an outcome). Unsupported candidates, candidates without an outcome and roots with fewer than two noise candidates enter none of them, so adding such data changes nothing.
- between_action_variance_net = raw minus within-action variance x the root's mean 1/n, averaged over roots. It assumes one within-action variance for every noise candidate, independent repeats inside a candidate, and no covariance between the repeat noise of different candidates at a root; positive covariance would make the correction too large and the net value too small.
- Repeats are paired by repeat id (the same column of cand_outcome). The collector has no controlled seed (decision_branch_collect.RNG_NOTE): matching repeat ids neither create nor prove correlation, and a small implied covariance is consistent with, but does not prove, independent identically distributed repeats.
