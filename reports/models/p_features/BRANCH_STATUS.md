# T0011 branch update record

All subsequent result and status updates are recorded on
`codex/t0011-p-feature-ablation-train-p2-and-p3-at` and delivered through a pull request.
Do not push directly to main, including task-board note/handoff commands that
atomically update main. Earlier task-board records are retained.

Training and evaluation continue with source commit
`9a78345172010a207fe89c54af12704b8cc3fddf` frozen. Commit report changes only after
the running pipeline completes, so source provenance remains stable.

Latest verified state: P2/P3 seed1 training and validation comparison completed.
P3 passed the prespecified validation finalist rule: paired joint-NLL difference
-0.08838297, map-clustered 95% interval [-0.08959273, -0.08717321], 2112 clusters.
Both families therefore continue at seeds 2 and 3 without test-based selection.
P2 seed2 finished eight epochs and is undergoing calibration/evaluation before
checkpoint save. The memory guard remains active. Prediction improvement alone
does not establish stronger play.

2026-10-11 06:50 UTC: P2 seed2 completed successfully; checkpoint and identical
arguments/data/hidden widths/source verified in P2_S2_VERIFIED.json. P3 seed2
started at 06:45:29 UTC, verified child PID46460, and completed epoch1. Guard
PID9872 remains healthy, CPU advances, available physical13.10 GiB and commit
38.50 GiB. No changes to frozen source or main.

2026-10-11 07:25 UTC: P3 seed2 completed successfully and artifacts were
verified in P3_S2_VERIFIED.json. Paired validation comparison began07:23:29 UTC
as verified child PID40008. CPU advances and guard9872 remains healthy.
Available physical47.78 GiB/commit49.02 GiB. Both seed3 runs follow automatically;
test evaluation remains pending. No main updates or source commits.

2026-10-11 07:30 UTC: Seed2 paired validation completed successfully, 2981919
rows and2112 map clusters. P3-P2 joint-NLL difference -0.08635656,95% interval
[-0.08756729,-0.08514582], again lower. P2 seed3 began07:26:07 UTC, verified
child PID36408, and completed epoch1. Guard9872 applied the unchanged40 GiB
limit; CPU advances, available physical13.17 GiB/commit39.28 GiB. Test remains
pending; no playing-strength claim or updates to main.

2026-10-11 07:55 UTC: User requested a temporary training pause to release RAM and
GPU pressure. Stopped the verified T0011 P2 seed3 trainer PID36408 after epoch8
(188704 optimizer steps) while calibration was pending; no seed3 checkpoint/config
was saved. The training pipeline exited at07:55:55 UTC and its CPU/GPU lock owners
released. GPU memory fell from about10.8 GiB to1.5 GiB; available host physical
memory rose to about53 GiB. To resume this experiment later, rerun P2 seed3 from
the beginning with the frozen source/data/arguments; retain all completed seed1
and seed2 artifacts and comparisons. Do not resume training until the user asks.

2026-10-11 09:37 UTC: User explicitly resumed training and authorized continuing
through the ready/dependent task queue until interrupted. Verified all four
completed seed1/seed2 checkpoint hashes, identical arguments/data/hidden widths,
validation receipts, and frozen source/input-loading receipts before resuming.
Preserved the interrupted P2 seed3 log and failed pipeline step; P2 seed3 restarts
from its original seed, followed by P3 seed3, validation3, test1/2/3 and seed1
exports. RESUME_AFTER_PAUSE.json records verification and recovery provenance.
The CPU/GPU locks belong to T0011; verified pipeline20668 and trainer28592.
Guard32708 checks every15 seconds with the existing40 GiB working-set policy.
At09:38 UTC: available physical12.24 GiB/commit38.10 GiB, private16.61 GiB,
working38.51 GiB; CPU advances. The 40 GiB cap affects throughput, so timings
across resource policies are not an equal-resource capacity comparison.

Follow-on execution order: T0010 (old p2 v12_ponly_xxl with V off), T0019
(corrected seed20261230), T0004 and T0025; dependency branches T0011->T0013,
T0019->T0005, T0010->T0012->T0014, then T0006 after finalists freeze.
Quantization work stays separate and waits for its code/model dependencies.
Respect independent worktrees, latest remote ownership/lanes and CPU/GPU locks.
No direct main writes, including task-board commands; report updates and task
work are delivered through independent branches and pull requests.

2026-10-11 09:49 UTC: Resumed P2 seed3 completed epochs1/2. Learning-rate,
train/validation loss and optimizer-step counts match the interrupted run
exactly (P2_S3_RESUME_PREFIX_VERIFIED.json). First epoch434.824 seconds and
second188.659 seconds; cache residency/resource differences prevent a timing
comparison. Verified pipeline20668/trainer28592/guard32708 identities and CPU
progress. Guard sample: physical10.97 GiB/commit37.33 GiB available, private17.14
GiB, working39.99 GiB; no guard or pipeline errors. Source remains frozen.

2026-10-11 10:09 UTC: Resumed P2 seed3 finished all8 epochs/188704 steps at
10:06:41 UTC and entered calibration/evaluation. All8 learning rates and
train/validation losses match the interrupted run exactly, verified in
P2_S3_RESUME_PREFIX_VERIFIED.json. Checkpoint/config remain pending until this
stage completes. Pipeline20668, trainer28592 and guard32708 identity unchanged;
CPU advances during calibration. Guard at10:08:45: physical12.48 GiB/commit28.58
GiB available, private28.75 GiB, working37.72 GiB; no error or attention flag.
No changes to frozen source, training arguments or main.

2026-10-11 10:29 UTC: P2 seed3 completed calibration and saved its checkpoint at
10:19:47 UTC; driver stage succeeded in2544.755 seconds. Verified identical
arguments except seed/output, data, hidden widths, row counts, features and
source commit against P2 seed1, all8 epochs plus calibration and copied report
artifacts (P2_S3_VERIFIED.json). Validation calibrated joint1.9421687617,
action0.3480603031 and target0.1134760251. This is a prediction metric.
P3 seed3 started10:19:49 UTC as verified direct child42100, completed epoch1,
and continues. Guard32708 applied its unchanged40 GiB limit at10:19:59 UTC.
Latest10:28:52 sample: physical9.90 GiB/commit37.42 GiB available, private17.16
GiB, working39.96 GiB; CPU advances and no errors. Test remains pending.
Source and main remain unchanged.

2026-10-11 11:09 UTC: P3 seed3 completed all8 epochs/188704 optimizer steps at
11:02:43 UTC and is undergoing calibration/evaluation. Final training loss
2.0614634205, smoothed combined validation loss2.0846966508. Completed epoch
seconds267.001,319.766,316.523,316.063,343.909,327.054,330.459,335.298;
these timings include the40 GiB resident policy and are not an equal-resource
comparison against earlier runs. Checkpoint/config are not yet saved.
Verified pipeline20668, direct trainer42100 and guard32708 identities; CPU
advances during calibration. At11:08:44 available physical14.61 GiB/commit28.55
GiB, private26.01 GiB, working35.03 GiB; no errors or attention flag. Test and
exports follow automatically. Frozen source and main unchanged.

2026-10-11 11:25 UTC: All six policy runs completed. P3 seed3 calibration saved
at11:15:54 UTC; checkpoint/config, same-family seed1 arguments/data/hidden widths,
feature/source identity and copied report artifacts verified in P3_S3_VERIFIED.json.
Its calibrated trainer losses joint1.8620226828/act0.2421160599/target0.1135550360.
Paired validation3 completed: P3-P2 joint-NLL -0.0861525577 with map-clustered
95% interval[-0.0873369243,-0.0849681912]; target-NLL +0.0001867729
[+0.0001572224,+0.0002163235], a small prediction harm for that head.
Test1/2 completed, test3 running since11:23:42 UTC; seed1 exports follow.
Queued artifact verification behind the CPU lock (wrapper43848): regenerated
arrays/headers/registry, exact V2/M copies, real-weight native/Python parity and
runtime contracts on recorded n02/n04/n15 streams. No concurrent CPU evidence
run is started; post checks wait for the frozen pipeline lock release.
Core/bundle tests and PR remain to be completed. No commit or main update.

2026-10-11 11:35 UTC: Frozen pipeline completed at11:26:16 UTC with all
six runs, all six paired comparisons and both seed1 exports successful.
The pipeline and memory guard exited normally; GPU lock was released.
At11:31 UTC physical51.84 GiB/commit53.92 GiB were available, GPU utilization4%
and total GPU memory1502 MiB. No training process was restarted.
Both exports passed exact regeneration, unchanged V2/M arrays, registry
dispatch for counts2-15, integer-accumulator native/Python parity and runtime
contracts on recorded n02/n04/n15 streams. Verification receipt compiler paths
are portable. Preliminary verification-script failures were preserved locally;
they did not change measurement artifacts. RESULTS.md/SUMMARY.json summarize
all three held-out seed comparisons: joint and action prediction improve,
while overall target prediction slightly worsens. No playing-strength claim.
Measurement source remains9a78345172010a207fe89c54af12704b8cc3fddf.
Training is complete, so integration and required delivery checks can now
proceed on this independent branch. No direct main or task-board update.

2026-10-11 11:56 UTC: Integrated latest main on this branch only, source
be8e96b408e5c20709c9090537ebd17c49d3c6b9. Core189 tests passed with1 platform
skip, P-only43 tests passed, all29 default release bundles compiled, and
task.py check passed. The four actual T0013 carriers (C08/V12 x P2/P3)
also compiled against the seed1 exports; no games or match seeds were spent.
CHECKS.json/md and CARRIER_BUILD.json retain verification identities.
Git's staged LF bytes match both final model manifests/native verification,
all six paired comparison digests and immutable historical receipt transport.
Measurement source remains9a78345172010a207fe89c54af12704b8cc3fddf; this
integration and subsequent report commits do not redefine the experiment.
Ready to push this independent branch and open its PR. T0010 is prepared in
its own worktree on codex/t0010-frozen-carrier-v-data at sourceebda2c4867fb6503f8e28f3bd0a8b434e6affb96.
