# Reproduce the T0004 offline evidence

Measurement checkout: `8a4eab4bef5bf65e971ec6712eebb89821738130`.
Restore that checkout in an independent worktree and copy this report directory there.
Use GCC15.2, Python3.14, NumPy, and the unchanged repository inference headers.
The source models, data and code hashes are recorded in the frozen receipts.
Do not run a command against a report commit expecting its HEAD to match the measurement source.

Set `CXX`, `AI_TEST_ROOT`, `T0004_RETAINED_MATCHES` and `T0004_TRAIN_INDEX` to the
verified local compiler, arena checkout, original retained value-match directory
and original training dataset index. All external inputs are read-only. Use
per-worktree Git safe.directory configuration and OMP/MKL/OpenBLAS8. Never change
global Git trust, write the main branch, claim a task, or modify a reserved source lane.

Run substantive host work under an exclusive CPU lease with no formal host GPU
training. A long background run must have a passive observer bound to its actual
wrapper command, creation time and child chain. It must not change host resources
or interfere with user applications. Commands below are foreground equivalents;
the recorded host used separate hidden helpers and a15-second passive observer.

```powershell
python tools/repo/task.py lock cpu --task T0004 -- python bots/p_only/report/near_far/input_inventory.py
python tools/repo/task.py lock cpu --task T0004 -- python bots/p_only/report/near_far/score_records.py
python tools/repo/task.py lock cpu --task T0004 -- python bots/p_only/report/near_far/aggregate_nll.py
python tools/repo/task.py lock cpu --task T0004 -- python bots/p_only/report/near_far/swap_decisions.py
python tools/repo/task.py lock cpu --task T0004 -- python bots/p_only/report/near_far/audit_evidence.py
```

The inventory and completed measurement receipts already exist for this delivery.
Do not repeat those computations just to check a report. Resume only unfinished
work after verifying its frozen identities and preserved successful outputs.
Fresh independent reproduction should use a new output area; native build hashes
and host include metadata can differ across checkouts. Never overwrite an original
frozen receipt to force compatibility. Report transport mappings preserve original
measurement identities while making published include paths portable.

The native scanner compiles S/L/XXL separately, uniformly with
`FUSION_MAX_WIDTH=2048`; unchanged XXL V headers require that width even though
scoring is policy-only. Its raw mode writes320 int32 policy accumulators. The
scorer returns joint NLL, conditional action sum/count, target sum/count, top1,
complete modeled-order NLL or NaN, and actual serving model ID. Bounded probes
compare those accumulators exactly with QuantizedTier and stable floating losses
within1e-12 before the full scan. Independent Python BFS checks reference/flag/
carrier distances and potential two-turn line-of-fire geometry.

Decision diagnostics generate carrier/View/replay copies only under ignored
`build/t0004/swapsrc`. They do not edit source lanes or weight exports. Unchanged
S/L registries are isolated in DLLs; raw policy outputs from their actual loaded
pointers are checked against direct native executables. Original root C08 PXXL_VM
and routing-disabled controls must match complete replies, root digests and
sample counts before interventions. All modes use the same128 fixed samples,
depth, scores, random keys, history and wanted-team mask, with deadlines removed.
Only second-turn team policy changes. A fresh View pins one immutable model for
all heads and owns its conditional caches. Different modes use separate processes.

Per-match scan NPZ/JSON and per-match/mode replay JSON/stdout/stderr remain under
ignored build. Published receipts bind them by SHA256, without committing raw
games, binaries, private paths or generated weight arrays. Independent audit reads
the preserved outputs; it never replays games or reevaluates policy inference.
Its checks include loss and label denominators, independent paired joint intervals,
source/binary/input hashes, protocol-valid complete orders, change categories and
root invariance. Turn-level branch eligibility was not traced; actual branch-call
counters and the complete10,400-state denominator are explicitly separate.

Delivery checks run sequentially under a separate CPU lease:

```powershell
python -m unittest discover -s tests -t .
python -m unittest discover -s bots/p_only/tests
python tools/repo/release_bots.py --out build/t0004/check_bundles --check
```

These checks use the frozen T0004 source, which includes the merged precision code.
Record the actual test and bundle totals from its logs; do not assume the older
T0019 checkout's totals apply. Successful check receipts are retained and skipped
when resuming delivery. No production routing or model export is part of this task.
