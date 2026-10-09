# Offline evaluation: model general, split test

Log-loss is nats per joint opponent move (lower is better), weighted back to the natural turn distribution. alpha is fitted on the fit split and applied here. `quantized` is the integer model the exporter ships; `mixture` is the joint-v2 adapter on it. Gain = baseline minus mixture loss, ± one map-clustered standard error. Offline numbers rank predictors; only paired matches measure strength.

## Opponent joint-move prediction

| teams | rows | maps | uniform | inertia | heuristic | learned | quantized | mixture | alpha | gain ± SE | use | why not | top-1 | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 9 | 220067 | 144 | 3.6285 | 3.0659 | 3.1234 | 2.1380 | 2.1383 | 2.1343 | 0.9535 | +0.9890 ± 0.0132 | yes | - | 0.4149 | 0.0061 |
| 11 | 266532 | 145 | 3.7396 | 3.0501 | 3.1656 | 2.1004 | 2.1020 | 2.0988 | 0.9640 | +1.0668 ± 0.0129 | yes | - | 0.4265 | 0.0020 |
| 13 | 360569 | 163 | 3.6600 | 3.0406 | 3.0973 | 2.1206 | 2.1220 | 2.1188 | 0.9611 | +0.9785 ± 0.0096 | yes | - | 0.4193 | 0.0024 |
| 14 | 293635 | 123 | 3.5865 | 3.0217 | 3.0536 | 2.1192 | 2.1203 | 2.1167 | 0.9596 | +0.9370 ± 0.0101 | yes | - | 0.4171 | 0.0036 |

Mixture baseline: heuristic. Inertia rates: 9: 0.528, 11: 0.560, 13: 0.546, 14: 0.535. Unit move accuracy (marginal argmax): 0.6454. Blocked requested moves: 0.0130. Units without a heuristic forecast: 0.0000.

### By stratum

| stratum | rows | uniform | inertia | heuristic | learned | quantized |
|---|---:|---:|---:|---:|---:|---:|
| flag | 535296 | 3.8797 | 3.3356 | 3.2204 | 2.3138 | 2.3152 |
| contact | 392997 | 3.7756 | 3.1424 | 3.4687 | 2.3679 | 2.3690 |
| respawn | 161136 | 2.4428 | 2.0225 | 1.9967 | 1.3196 | 1.3202 |
| quiet | 51374 | 4.7219 | 3.5755 | 3.7594 | 2.2029 | 2.2044 |

### Action heads (at the requested move)

| head | rows | log-loss | accuracy |
|---|---:|---:|---:|
| act | 2663160 | 0.1672 | 0.9248 |
| target | 727399 | 0.2153 | 0.9070 |

## Board value (diagnostic; no gate approves V)

Rows 1142196. Brier 0.0309 (constant 0.5: 0.0967), BCE 0.5424, binned calibration error 0.0039, pairwise ranking accuracy 0.8586 over 4944095 pairs. Quantized Brier 0.0309, max |p_q - p| 1.47e-02.

| phase | rows | Brier | ranking accuracy |
|---|---:|---:|---:|
| turns 1-133 | 374250 | 0.0592 | 0.7430 |
| turns 134-266 | 382940 | 0.0258 | 0.8708 |
| turns 267-400 | 385006 | 0.0078 | 0.9578 |

## Model size

| net | parameters | MACs per call |
|---|---:|---:|
| mover | 673 | 656 |
| squad | 20477 | 20256 |
| act | 724 | 704 |
| target | 73 | 64 |
| value | 9121 | 8976 |

Joint policy for one opponent team: 30096 MACs (15 mover calls + 1 squad call).
