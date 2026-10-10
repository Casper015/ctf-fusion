# Value evaluation: test split

Rows 305326 from 19 maps. Probabilities are calibrated (sigmoid(beta z)). Intervals are 95%, clustered by map; an interval that includes zero is "not shown". Lower loss alone does not establish stronger decisions.

| Run | Family | Hidden | Params | Log loss | Brier | ECE | Pair acc. | Regret | Epochs (best) | Seconds | Peak GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| V3_XL | v3 | [1024, 64] | 177411 | 0.5428 | 0.0352 | 0.0021 | - | - | 17 (14) | 38.9 | 0.998 |

## By phase (turns)

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | 1-100 | 74614 | 0.6354 | 0.0733 | 0.0059 |
| V3_XL | 101-200 | 76651 | 0.5540 | 0.0385 | 0.0064 |
| V3_XL | 201-300 | 77134 | 0.5120 | 0.0220 | 0.0043 |
| V3_XL | 301-400 | 76927 | 0.4700 | 0.0071 | 0.0018 |

## By team count

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | 4 | 6416 | 0.5159 | 0.0580 | 0.0355 |
| V3_XL | 5 | 25629 | 0.5135 | 0.0434 | 0.0169 |
| V3_XL | 7 | 34269 | 0.5244 | 0.0373 | 0.0238 |
| V3_XL | 8 | 40414 | 0.5511 | 0.0422 | 0.0107 |
| V3_XL | 9 | 15679 | 0.5737 | 0.0491 | 0.0213 |
| V3_XL | 11 | 18293 | 0.5392 | 0.0302 | 0.0087 |
| V3_XL | 13 | 66856 | 0.5474 | 0.0301 | 0.0080 |
| V3_XL | 14 | 71659 | 0.5432 | 0.0281 | 0.0138 |
| V3_XL | 15 | 26111 | 0.5623 | 0.0352 | 0.0067 |

## Subsets

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | carrier | 286983 | 0.5401 | 0.0334 | 0.0023 |
| V3_XL | drop | 147120 | 0.5428 | 0.0319 | 0.0049 |
| V3_XL | interception | 27856 | 0.5469 | 0.0355 | 0.0072 |
| V3_XL | close | 239300 | 0.5771 | 0.0406 | 0.0024 |

## Decision regret

| Run | Roots | Resolved / uncertain pairs | Regret (V choice) | On-policy | Uniform |
|---|---:|---|---:|---:|---:|

## Learning curves

- **V3_XL** (val outcome loss by epoch): 1: 0.5402, 2: 0.5381, 3: 0.5382, 4: 0.5387, 5: 0.5379, 6: 0.5376, 7: 0.5377, 8: 0.5376, 9: 0.5378, 10: 0.5377, 11: 0.5375, 12: 0.5374, 13: 0.5376, 14: 0.5373, 15: 0.5377, 16: 0.5374, 17: 0.5373
