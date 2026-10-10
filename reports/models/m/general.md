# Offline evaluation: model general, split test

Log-loss is nats per joint opponent move (lower is better), weighted back to the natural turn distribution. alpha is fitted on the fit split and applied here. `quantized` is the integer model the exporter ships; `mixture` is the joint-v2 adapter on it. Gain = baseline minus mixture loss, ± one map-clustered standard error. Offline numbers rank predictors; only paired matches measure strength.

## Opponent joint-move prediction

| teams | rows | maps | uniform | inertia | heuristic | learned | quantized | mixture | alpha | gain ± SE | use | why not | top-1 | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 9 | 220067 | 144 | 3.6285 | 3.0659 | 3.1234 | 2.0916 | 2.0925 | 2.0898 | 0.9643 | +1.0335 ± 0.0136 | yes | - | 0.4215 | 0.0094 |
| 11 | 266532 | 145 | 3.7396 | 3.0501 | 3.1656 | 2.0520 | 2.0525 | 2.0507 | 0.9732 | +1.1148 ± 0.0135 | yes | - | 0.4348 | 0.0025 |
| 13 | 360569 | 163 | 3.6600 | 3.0406 | 3.0973 | 2.0752 | 2.0758 | 2.0738 | 0.9718 | +1.0235 ± 0.0100 | yes | - | 0.4269 | 0.0035 |
| 14 | 293635 | 123 | 3.5865 | 3.0217 | 3.0536 | 2.0752 | 2.0758 | 2.0735 | 0.9706 | +0.9801 ± 0.0104 | yes | - | 0.4247 | 0.0050 |

Mixture baseline: heuristic. Inertia rates: 9: 0.528, 11: 0.560, 13: 0.546, 14: 0.535. Unit move accuracy (marginal argmax): 0.6513. Blocked requested moves: 0.0130. Units without a heuristic forecast: 0.0000.

### By stratum

| stratum | rows | uniform | inertia | heuristic | learned | quantized |
|---|---:|---:|---:|---:|---:|---:|
| flag | 535296 | 3.8797 | 3.3356 | 3.2204 | 2.2604 | 2.2611 |
| contact | 392997 | 3.7756 | 3.1424 | 3.4687 | 2.3270 | 2.3277 |
| respawn | 161136 | 2.4428 | 2.0225 | 1.9967 | 1.2912 | 1.2916 |
| quiet | 51374 | 4.7219 | 3.5755 | 3.7594 | 2.1382 | 2.1393 |

### Action heads (at the requested move)

| head | rows | log-loss | accuracy |
|---|---:|---:|---:|
| act | 2663160 | 0.1644 | 0.9264 |
| target | 727399 | 0.2143 | 0.9076 |

## Board value (diagnostic; no gate approves V)

Rows 1142196. Brier 0.0308 (constant 0.5: 0.0967), BCE 0.5421, binned calibration error 0.0038, pairwise ranking accuracy 0.8592 over 4944095 pairs. Quantized Brier 0.0308, max |p_q - p| 1.50e-02.

| phase | rows | Brier | ranking accuracy |
|---|---:|---:|---:|
| turns 1-133 | 374250 | 0.0590 | 0.7440 |
| turns 134-266 | 382940 | 0.0258 | 0.8711 |
| turns 267-400 | 385006 | 0.0077 | 0.9581 |

## Model size

| net | parameters | MACs per call |
|---|---:|---:|
| mover | 1345 | 1312 |
| squad | 44925 | 44608 |
| act | 1444 | 1408 |
| target | 145 | 128 |
| value | 22337 | 22048 |

Joint policy for one opponent team: 64288 MACs (15 mover calls + 1 squad call).
