# T0004 verification

Frozen measurement source:8a4eab4bef5bf65e971ec6712eebb89821738130. Changes are limited to this report lane.

- Native P2 scorer:39 size/count cases; exact320 integer policy-head accumulators per probe; NLL within1e-12; independent BFS/line-of-fire geometry validation.
- Loaded S/L model modules:26 size/count cases,16 probes each, exact320 policy accumulators, same compiler/2048 scratch ABI.
- Full scan:260 frozen maps,832000 valid opponent rows,789836 primary; canonical labels and primary/conditional denominators verified.
- Decisions: six modes x26maps x400states; original/disabled control identical; root digests invariant; no truncation or invalid complete replies. Complete-order movement/action/target changes independently audited.
- Independent audit: raw/source/binary/array/replay hashes, paired joint means and map-cluster intervals, label coverage and complete protocol orders passed.
- Core suite:218 tests run; P-only suite:46 tests run; all returned0. Platform skips, if any, remain in the original logs.
- Single-file package syntax:29 bundles passed.

Checks ran sequentially under a separate exclusive CPU lease after measurement completion. Test fixtures are verification, not new tuning data. Wall time is not a capacity or latency comparison. No model export, main/task-board change or production-default change occurred.

- `python -m unittest discover -s tests -t .`: exit0, 309.24 wall seconds.
- `python -m unittest discover -s bots/p_only/tests`: exit0, 1260.659 wall seconds.
- `python tools/repo/release_bots.py --out build/t0004/check_bundles --check`: exit0, 100.998 wall seconds.

NATIVE_METADATA_TRANSPORT.json makes the published include flag portable while preserving original receipt hashes and measurement identity. RECEIPT_TRANSPORT.json describes report CRLF-to-LF transport only. Original arrays, game records, binaries and replay bytes are unchanged.
