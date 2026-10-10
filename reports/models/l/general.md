# Offline evaluation: model general, split test

Log-loss is nats per joint opponent move (lower is better), weighted back to the natural turn distribution. alpha is fitted on the fit split and applied here. `quantized` is the integer model the exporter ships; `mixture` is the joint-v2 adapter on it. Gain = baseline minus mixture loss, ± one map-clustered standard error. Offline numbers rank predictors; only paired matches measure strength.

## Opponent joint-move prediction

| teams | rows | maps | uniform | inertia | heuristic | learned | quantized | mixture | alpha | gain ± SE | use | why not | top-1 | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 9 | 220067 | 144 | 3.6285 | 3.0659 | 3.1234 | 2.0578 | 2.0591 | 2.0563 | 0.9651 | +1.0670 ± 0.0142 | yes | - | 0.4278 | 0.0055 |
| 11 | 266532 | 145 | 3.7396 | 3.0501 | 3.1656 | 2.0168 | 2.0169 | 2.0152 | 0.9752 | +1.1503 ± 0.0138 | yes | - | 0.4407 | 0.0026 |
| 13 | 360569 | 163 | 3.6600 | 3.0406 | 3.0973 | 2.0413 | 2.0417 | 2.0398 | 0.9740 | +1.0575 ± 0.0102 | yes | - | 0.4321 | 0.0029 |
| 14 | 293635 | 123 | 3.5865 | 3.0217 | 3.0536 | 2.0429 | 2.0432 | 2.0410 | 0.9735 | +1.0126 ± 0.0106 | yes | - | 0.4306 | 0.0036 |

Mixture baseline: heuristic. Inertia rates: 9: 0.528, 11: 0.560, 13: 0.546, 14: 0.535. Unit move accuracy (marginal argmax): 0.6558. Blocked requested moves: 0.0130. Units without a heuristic forecast: 0.0000.

### By stratum

| stratum | rows | uniform | inertia | heuristic | learned | quantized |
|---|---:|---:|---:|---:|---:|---:|
| flag | 535296 | 3.8797 | 3.3356 | 3.2204 | 2.2210 | 2.2213 |
| contact | 392997 | 3.7756 | 3.1424 | 3.4687 | 2.2948 | 2.2954 |
| respawn | 161136 | 2.4428 | 2.0225 | 1.9967 | 1.2722 | 1.2725 |
| quiet | 51374 | 4.7219 | 3.5755 | 3.7594 | 2.0940 | 2.0947 |

### Action heads (at the requested move)

| head | rows | log-loss | accuracy |
|---|---:|---:|---:|
| act | 2663160 | 0.1623 | 0.9270 |
| target | 727399 | 0.2143 | 0.9076 |

## Board value (diagnostic; no gate approves V)

Rows 1142196. Brier 0.0307 (constant 0.5: 0.0967), BCE 0.5419, binned calibration error 0.0020, pairwise ranking accuracy 0.8593 over 4944095 pairs. Quantized Brier 0.0307, max |p_q - p| 1.18e-02.

| phase | rows | Brier | ranking accuracy |
|---|---:|---:|---:|
| turns 1-133 | 374250 | 0.0589 | 0.7442 |
| turns 134-266 | 382940 | 0.0257 | 0.8713 |
| turns 267-400 | 385006 | 0.0077 | 0.9582 |

## Model size

| net | parameters | MACs per call |
|---|---:|---:|
| mover | 2689 | 2624 |
| squad | 81661 | 81216 |
| act | 2884 | 2816 |
| target | 145 | 128 |
| value | 61057 | 60480 |

Joint policy for one opponent team: 120576 MACs (15 mover calls + 1 squad call).
