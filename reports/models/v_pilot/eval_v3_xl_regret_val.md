# Value evaluation: val split

Rows 52728 from 31 maps. Probabilities are calibrated (sigmoid(beta z)). Intervals are 95%, clustered by map; an interval that includes zero is "not shown". Lower loss alone does not establish stronger decisions.

| Run | Family | Hidden | Params | Log loss | Brier | ECE | Pair acc. | Regret | Epochs (best) | Seconds | Peak GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| V3_XL | v3 | [1024, 64] | 177411 | 0.4674 | 0.0354 | 0.0075 | 0.458 | 0.0869 | 13 (9) | 7.8 | 0.229 |

## By phase (turns)

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | 1-100 | 12948 | 0.5330 | 0.0589 | 0.0148 |
| V3_XL | 101-200 | 13313 | 0.4815 | 0.0404 | 0.0122 |
| V3_XL | 201-300 | 13261 | 0.4496 | 0.0284 | 0.0118 |
| V3_XL | 301-400 | 13206 | 0.4053 | 0.0139 | 0.0117 |

## By team count

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | 2 | 7020 | 0.2173 | 0.0654 | 0.0432 |
| V3_XL | 3 | 1603 | 0.7933 | 0.0423 | 0.1414 |
| V3_XL | 4 | 6286 | 0.4137 | 0.0293 | 0.0319 |
| V3_XL | 5 | 1799 | 0.6106 | 0.0377 | 0.0627 |
| V3_XL | 7 | 4963 | 0.5079 | 0.0403 | 0.0453 |
| V3_XL | 8 | 5056 | 0.3764 | 0.0225 | 0.0696 |
| V3_XL | 9 | 7092 | 0.4890 | 0.0346 | 0.0312 |
| V3_XL | 11 | 1592 | 0.6599 | 0.0150 | 0.0362 |
| V3_XL | 12 | 5064 | 0.5900 | 0.0207 | 0.0401 |
| V3_XL | 13 | 1699 | 0.7570 | 0.0738 | 0.1686 |
| V3_XL | 14 | 5257 | 0.5003 | 0.0347 | 0.0345 |
| V3_XL | 15 | 5297 | 0.4206 | 0.0199 | 0.0311 |

## Subsets

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | carrier | 45908 | 0.4780 | 0.0330 | 0.0071 |
| V3_XL | drop | 19528 | 0.4997 | 0.0312 | 0.0063 |
| V3_XL | interception | 4424 | 0.4643 | 0.0387 | 0.0175 |
| V3_XL | close | 36392 | 0.5353 | 0.0421 | 0.0118 |

## Decision regret

| Run | Roots | Resolved / uncertain pairs | Regret (V choice) | On-policy | Uniform |
|---|---:|---|---:|---:|---:|
| V3_XL | 15 | 37 / 53 | 0.0869 [0.0116, 0.1622] | 0.0966 | 0.1298 |

## Learning curves

- **V3_XL** (val outcome loss by epoch): 1: 0.4747, 2: 0.4722, 3: 0.4683, 4: 0.4694, 5: 0.4686, 6: 0.4689, 7: 0.4690, 8: 0.4681, 9: 0.4675, 10: 0.4676, 11: 0.4675, 12: 0.4676, 13: 0.4682
