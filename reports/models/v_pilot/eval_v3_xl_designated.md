# Value evaluation: test split

Rows 32703 from 19 maps. Probabilities are calibrated (sigmoid(beta z)). Intervals are 95%, clustered by map; an interval that includes zero is "not shown". Lower loss alone does not establish stronger decisions.

| Run | Family | Hidden | Params | Log loss | Brier | ECE | Pair acc. | Regret | Epochs (best) | Seconds | Peak GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| V3_XL_Designated | v3 | [1024, 64] | 177411 | 0.4970 | 0.0399 | 0.0328 | - | - | 13 (9) | 7.8 | 0.229 |

## By phase (turns)

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL_Designated | 1-100 | 8003 | 0.5843 | 0.0735 | 0.0307 |
| V3_XL_Designated | 101-200 | 8170 | 0.5204 | 0.0496 | 0.0381 |
| V3_XL_Designated | 201-300 | 8289 | 0.4639 | 0.0273 | 0.0410 |
| V3_XL_Designated | 301-400 | 8241 | 0.4198 | 0.0095 | 0.0230 |

## By team count

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL_Designated | 4 | 1621 | 0.5619 | 0.0301 | 0.0462 |
| V3_XL_Designated | 5 | 5127 | 0.5370 | 0.0440 | 0.0453 |
| V3_XL_Designated | 7 | 4988 | 0.3138 | 0.0233 | 0.0629 |
| V3_XL_Designated | 8 | 5075 | 0.6564 | 0.0787 | 0.2081 |
| V3_XL_Designated | 9 | 1710 | 0.4874 | 0.0565 | 0.0790 |
| V3_XL_Designated | 11 | 1707 | 0.5323 | 0.0282 | 0.0512 |
| V3_XL_Designated | 13 | 5309 | 0.3899 | 0.0251 | 0.0212 |
| V3_XL_Designated | 14 | 5353 | 0.4880 | 0.0218 | 0.0389 |
| V3_XL_Designated | 15 | 1813 | 0.6967 | 0.0640 | 0.1864 |

## Subsets

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL_Designated | carrier | 29921 | 0.4914 | 0.0389 | 0.0340 |
| V3_XL_Designated | drop | 13716 | 0.4960 | 0.0361 | 0.0324 |
| V3_XL_Designated | interception | 2420 | 0.5013 | 0.0422 | 0.0354 |
| V3_XL_Designated | close | 23864 | 0.5318 | 0.0469 | 0.0384 |

## Decision regret

| Run | Roots | Resolved / uncertain pairs | Regret (V choice) | On-policy | Uniform |
|---|---:|---|---:|---:|---:|

## Learning curves

- **V3_XL_Designated** (val outcome loss by epoch): 1: 0.4747, 2: 0.4722, 3: 0.4683, 4: 0.4694, 5: 0.4686, 6: 0.4689, 7: 0.4690, 8: 0.4681, 9: 0.4675, 10: 0.4676, 11: 0.4675, 12: 0.4676, 13: 0.4682
