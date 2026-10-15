# Own-action decision rule: test split

Rule: proposed rule: z-scored logged carrier scores + lambda * gain.

Delta = chosen minus incumbent mean continuation outcome, all evaluable roots, equal weight per team count, 95% intervals clustered by map. No game was played to make this report; it is not a strength result.

| Comparison | Role | Mean | Interval | Maps | Verdict |
|---|---|---:|---|---:|---|
| mlp_s101 - incumbent | primary | -0.0151 | [-0.0563, +0.0261] | 31 | insufficient precision |
| linear_s101 - incumbent | secondary | -0.0203 | [-0.0846, +0.0440] | 31 | insufficient precision |
| linear_s102 - incumbent | secondary | -0.0263 | [-0.0862, +0.0337] | 31 | insufficient precision |
| linear_s103 - incumbent | secondary | -0.0167 | [-0.0714, +0.0380] | 31 | insufficient precision |
| mlp_s102 - incumbent | secondary | -0.0339 | [-0.0962, +0.0284] | 31 | insufficient precision |
| mlp_s103 - incumbent | secondary | -0.0188 | [-0.0820, +0.0445] | 31 | insufficient precision |
| mlp_s101 - linear_s101 | secondary | +0.0052 | [-0.0275, +0.0379] | 31 | insufficient precision |
| mlp_s101 - linear_s102 | secondary | +0.0112 | [-0.0345, +0.0568] | 31 | insufficient precision |
| mlp_s101 - linear_s103 | secondary | +0.0016 | [-0.0201, +0.0234] | 31 | insufficient precision |
| mlp_s101 - mlp_s102 | secondary | +0.0188 | [-0.0301, +0.0676] | 31 | insufficient precision |
| mlp_s101 - mlp_s103 | secondary | +0.0037 | [-0.0263, +0.0336] | 31 | insufficient precision |

| Rule | Evaluable / unevaluable roots | Changed share | Mean where changed |
|---|---|---:|---:|
| incumbent | 31 / 0 | 0.000 | - |
| linear_s101 | 31 / 0 | 0.581 | -0.0420 |
| linear_s102 | 31 / 0 | 0.548 | -0.0385 |
| linear_s103 | 31 / 0 | 0.419 | -0.0403 |
| mlp_s101 | 31 / 0 | 0.613 | -0.0334 |
| mlp_s102 | 31 / 0 | 0.677 | -0.0389 |
| mlp_s103 | 31 / 0 | 0.645 | -0.0350 |

carrier_top_score - incumbent: +0.0000 [+0.0000, +0.0000] negligible. DIAGNOSTIC BASELINE: argmax of the logged carrier score, no scorer; not the disabled learned rule (lambda = 0 is the incumbent); its runtime equivalent is a carrier that plays its top logged score.

Headroom: estimated with independent repeat subsets (select: even repeat ids, evaluate: odd repeat ids); +0.0487 [+0.0003, +0.0970]
