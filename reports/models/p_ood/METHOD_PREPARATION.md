# T0025 proposed offline methodology

This is preparation, not a measured result or a frozen collection schedule.

Collect new independent maps under the official2000/50ms limits, using a seed
verified absent from the committed ledger and existing host measurement
receipts. Preserve the evaluated source identities of both reserved lineages.
Use a declared roster with a fixed admitted reference carrier, exactly one
instance of each held-out lineage when both fit, and admitted in-distribution
fillers. At two teams, balance the two held-out lineages across distinct maps.
Randomize seats deterministically and keep lineage caps explicit. Freeze the
complete fixture schedule, engine, programs, binaries, runner, compiler and
limits before games. Retain all faults and raw records; do not replay completed
fixtures or exclude faulty games silently. Valid canonical policy-label rows
and unavailable replies have separate denominators.

Evaluate deployed XL/XXL P2 artifacts on held-out-lineage rows only. Verify
every served model ID, NPZ/export identity and complete lineage exclusion
from original training/calibration data. Use the same frozen original models
for their in-distribution test numbers alongside OOD results. All map overlap
checks are against original train/val/test partitions and inspected prior
selection/probe/screen inputs, not just the training partition.

Use `training.evaluate_offline` metric/baseline routines with actual committed
integer `QuantizedTier` artifacts. The CLI re-quantizes floating checkpoints
and also fits a mixture diagnostic; it is not by itself a pure-P-only report.
Prove any reconstructed checkpoint matches the deployed NPZ exactly before
using it. Primary results are the pure quantized joint distribution, compared
with actual q_H, inertia and uniform. Mixtures, if included by the unchanged
evaluator, stay explicitly diagnostic and do not redefine the deployed P-only
predictor. No artifact re-export or production change occurs.

The inertia rate and every fitted baseline parameter must come from the
original in-distribution validation data, never the new OOD evaluation rows.
Keep folded model temperatures unchanged. Verify q_H source identity,
baseline coverage and missingness; absent forecasts cannot become a claimed
legacy baseline. Report joint NLL, calibration, per-count and per-lineage
results with independent-map clustered intervals and explicit valid-row/map
denominators. Prediction scores are not game strength. Registering the study
in the repository layout remains a separate lane constraint to resolve before
final PR delivery.
