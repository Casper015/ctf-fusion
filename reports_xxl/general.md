# Offline evaluation: model general, split test

Log-loss is nats per joint opponent move (lower is better), weighted back to the natural turn distribution. alpha is fitted on the fit split and applied here. `quantized` is the integer model the exporter ships; `mixture` is the joint-v2 adapter on it. Gain = baseline minus mixture loss, ± one map-clustered standard error. Offline numbers rank predictors; only paired matches measure strength.

## Opponent joint-move prediction

| teams | rows | maps | uniform | inertia | heuristic | learned | quantized | mixture | alpha | gain ± SE | use | why not | top-1 | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 9 | 220067 | 144 | 3.6285 | 3.0659 | 3.1234 | 1.9921 | 1.9923 | 1.9908 | 0.9767 | +1.1325 ± 0.0149 | yes | - | 0.4392 | 0.0076 |
| 11 | 266532 | 145 | 3.7396 | 3.0501 | 3.1656 | 1.9480 | 1.9484 | 1.9475 | 0.9832 | +1.2181 ± 0.0145 | yes | - | 0.4525 | 0.0046 |
| 13 | 360569 | 163 | 3.6600 | 3.0406 | 3.0973 | 1.9741 | 1.9744 | 1.9733 | 0.9813 | +1.1240 ± 0.0108 | yes | - | 0.4441 | 0.0070 |
| 14 | 293635 | 123 | 3.5865 | 3.0217 | 3.0536 | 1.9787 | 1.9789 | 1.9775 | 0.9811 | +1.0761 ± 0.0112 | yes | - | 0.4427 | 0.0080 |

Mixture baseline: heuristic. Inertia rates: 9: 0.528, 11: 0.560, 13: 0.546, 14: 0.535. Unit move accuracy (marginal argmax): 0.6647. Blocked requested moves: 0.0130. Units without a heuristic forecast: 0.0000.

### By stratum

| stratum | rows | uniform | inertia | heuristic | learned | quantized |
|---|---:|---:|---:|---:|---:|---:|
| flag | 535296 | 3.8797 | 3.3356 | 3.2204 | 2.1482 | 2.1484 |
| contact | 392997 | 3.7756 | 3.1424 | 3.4687 | 2.2290 | 2.2294 |
| respawn | 161136 | 2.4428 | 2.0225 | 1.9967 | 1.2309 | 1.2312 |
| quiet | 51374 | 4.7219 | 3.5755 | 3.7594 | 2.0022 | 2.0025 |

### Action heads (at the requested move)

| head | rows | log-loss | accuracy |
|---|---:|---:|---:|
| act | 2663160 | 0.1606 | 0.9276 |
| target | 727399 | 0.2143 | 0.9080 |

## Board value (diagnostic; no gate approves V)

Rows 1142196. Brier 0.0307 (constant 0.5: 0.0967), BCE 0.5418, binned calibration error 0.0028, pairwise ranking accuracy 0.8595 over 4944095 pairs. Quantized Brier 0.0307, max |p_q - p| 1.44e-02.

| phase | rows | Brier | ranking accuracy |
|---|---:|---:|---:|
| turns 1-133 | 374250 | 0.0589 | 0.7448 |
| turns 134-266 | 382940 | 0.0257 | 0.8714 |
| turns 267-400 | 385006 | 0.0077 | 0.9581 |

## Model size

| net | parameters | MACs per call |
|---|---:|---:|
| mover | 10753 | 10496 |
| squad | 522877 | 521472 |
| act | 11524 | 11264 |
| target | 289 | 256 |
| value | 375041 | 372864 |

Joint policy for one opponent team: 678912 MACs (15 mover calls + 1 squad call).
