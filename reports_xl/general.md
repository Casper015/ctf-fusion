# Offline evaluation: model general, split test

Log-loss is nats per joint opponent move (lower is better), weighted back to the natural turn distribution. alpha is fitted on the fit split and applied here. `quantized` is the integer model the exporter ships; `mixture` is the joint-v2 adapter on it. Gain = baseline minus mixture loss, ± one map-clustered standard error. Offline numbers rank predictors; only paired matches measure strength.

## Opponent joint-move prediction

| teams | rows | maps | uniform | inertia | heuristic | learned | quantized | mixture | alpha | gain ± SE | use | why not | top-1 | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 9 | 220067 | 144 | 3.6285 | 3.0659 | 3.1234 | 2.0190 | 2.0199 | 2.0179 | 0.9714 | +1.1055 ± 0.0145 | yes | - | 0.4340 | 0.0063 |
| 11 | 266532 | 145 | 3.7396 | 3.0501 | 3.1656 | 1.9769 | 1.9772 | 1.9759 | 0.9787 | +1.1897 ± 0.0143 | yes | - | 0.4463 | 0.0048 |
| 13 | 360569 | 163 | 3.6600 | 3.0406 | 3.0973 | 2.0030 | 2.0034 | 2.0018 | 0.9772 | +1.0955 ± 0.0106 | yes | - | 0.4388 | 0.0052 |
| 14 | 293635 | 123 | 3.5865 | 3.0217 | 3.0536 | 2.0059 | 2.0063 | 2.0045 | 0.9776 | +1.0491 ± 0.0110 | yes | - | 0.4365 | 0.0061 |

Mixture baseline: heuristic. Inertia rates: 9: 0.528, 11: 0.560, 13: 0.546, 14: 0.535. Unit move accuracy (marginal argmax): 0.6607. Blocked requested moves: 0.0130. Units without a heuristic forecast: 0.0000.

### By stratum

| stratum | rows | uniform | inertia | heuristic | learned | quantized |
|---|---:|---:|---:|---:|---:|---:|
| flag | 535296 | 3.8797 | 3.3356 | 3.2204 | 2.1788 | 2.1795 |
| contact | 392997 | 3.7756 | 3.1424 | 3.4687 | 2.2586 | 2.2591 |
| respawn | 161136 | 2.4428 | 2.0225 | 1.9967 | 1.2478 | 1.2481 |
| quiet | 51374 | 4.7219 | 3.5755 | 3.7594 | 2.0375 | 2.0377 |

### Action heads (at the requested move)

| head | rows | log-loss | accuracy |
|---|---:|---:|---:|
| act | 2663160 | 0.1608 | 0.9275 |
| target | 727399 | 0.2142 | 0.9076 |

## Board value (diagnostic; no gate approves V)

Rows 1142196. Brier 0.0308 (constant 0.5: 0.0967), BCE 0.5420, binned calibration error 0.0041, pairwise ranking accuracy 0.8594 over 4944095 pairs. Quantized Brier 0.0308, max |p_q - p| 1.30e-02.

| phase | rows | Brier | ranking accuracy |
|---|---:|---:|---:|
| turns 1-133 | 374250 | 0.0590 | 0.7444 |
| turns 134-266 | 382940 | 0.0257 | 0.8713 |
| turns 267-400 | 385006 | 0.0077 | 0.9581 |

## Model size

| net | parameters | MACs per call |
|---|---:|---:|
| mover | 5377 | 5248 |
| squad | 195965 | 195200 |
| act | 5764 | 5632 |
| target | 289 | 256 |
| value | 121985 | 120896 |

Joint policy for one opponent team: 273920 MACs (15 mover calls + 1 squad call).
