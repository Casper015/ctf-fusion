"""Command-line interface for agent_v2 branch diagnostics (T0085).

Subcommands:
  extract   Extract causal roots, actually sampled baselines, and candidate shortlists.
  branch    Execute faithful branch rollouts holding turn-t opponent commands common.
  parity    Audit simulator mechanics parity against official engine.py.
  p-eval    Audit opponent P model fidelity (ECE, log-loss, accuracy) against factual transitions.
  analyze   Compute signed baseline gain Delta(a), regret, ties, and cross-fit metrics.
  estimate  Print exact pilot cost and worker-hour dry-run estimates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fusion.paths import REPO, ai_test_root, load_engine
from tools.search_diagnostics import (
    analyze,
    branch,
    estimator,
    extract,
    p_eval,
    parity,
)
from tools.search_diagnostics.schemas import (
    BranchResult,
    RootRecord,
    ValidationError,
)


def cmd_estimate(args: argparse.Namespace) -> int:
    counts = [int(c) for c in args.counts.split(",")] if args.counts else None
    est = estimator.estimate_pilot_cost(
        team_counts=counts,
        matches_per_count=args.matches,
        candidates_per_root=args.k,
        repeats=args.repeats,
        workers=args.workers,
        ms_per_step=args.ms_per_step,
    )
    if args.markdown:
        print(estimator.format_pilot_cost_markdown(est))
    else:
        print(json.dumps(est.to_dict(), indent=2))
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    arena = ai_test_root(args.ai_test)
    eng = load_engine(arena)
    check_exe = Path(args.check).resolve()
    if not check_exe.is_file():
        sys.stderr.write(f"check binary not found: {check_exe}\n")
        return 1

    counts = [int(c) for c in args.counts.split(",")] if args.counts else [2, 3, 4, 8, 15]
    report = parity.run_simulator_parity_audit(
        engine=eng,
        check_exe=check_exe,
        team_counts=counts,
        steps_per_team=args.steps,
    )
    res_dict = report.to_dict()
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res_dict, indent=2) + "\n", encoding="utf-8")
        print(f"Parity report written to {out_p}")
    else:
        print(json.dumps(res_dict, indent=2))
    return 0 if report.passed else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    roots_p = Path(args.roots)
    branches_p = Path(args.branches)

    roots_data = json.loads(roots_p.read_text(encoding="utf-8"))
    if isinstance(roots_data, dict) and "roots" in roots_data:
        roots_list = [RootRecord.from_dict(r) for r in roots_data["roots"]]
    elif isinstance(roots_data, list):
        roots_list = [RootRecord.from_dict(r) for r in roots_data]
    else:
        roots_list = [RootRecord.from_dict(roots_data)]

    branches_data = json.loads(branches_p.read_text(encoding="utf-8"))
    if isinstance(branches_data, dict) and "branches" in branches_data:
        branches_list = [BranchResult.from_dict(b) for b in branches_data["branches"]]
    elif isinstance(branches_data, list):
        branches_list = [BranchResult.from_dict(b) for b in branches_data]
    else:
        branches_list = [BranchResult.from_dict(branches_data)]

    summary = analyze.compute_cross_fit_evaluation(roots_list, branches_list)
    out_dict = dataclasses_to_dict(summary)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_dict, indent=2) + "\n", encoding="utf-8")
        print(f"Analysis summary written to {out_p}")
    else:
        print(json.dumps(out_dict, indent=2))
    return 0


def dataclasses_to_dict(obj: Any) -> Any:
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # estimate
    p_est = sub.add_parser("estimate", help="Compute dry-run pilot cost estimate")
    p_est.add_argument("--counts", type=str, default="", help="Comma-separated team counts (e.g. 2,4,8,15)")
    p_est.add_argument("--matches", type=int, default=16, help="Matches per team count")
    p_est.add_argument("-k", type=int, default=8, help="Candidates per root")
    p_est.add_argument("--repeats", type=int, default=4, help="Repeats per candidate")
    p_est.add_argument("--workers", type=int, default=8, help="CPU workers (default 8)")
    p_est.add_argument("--ms-per-step", type=float, default=50.0, help="Latency per engine step in ms")
    p_est.add_argument("--markdown", action="store_true", help="Print Markdown report")
    p_est.set_defaults(func=cmd_estimate)

    # parity
    p_par = sub.add_parser("parity", help="Audit simulator parity against official engine.py")
    p_par.add_argument("--check", type=str, required=True, help="Path to compiled check.exe")
    p_par.add_argument("--ai-test", type=str, default=None, help="Path to arena checkout (or AI_TEST_ROOT)")
    p_par.add_argument("--counts", type=str, default="", help="Comma-separated team counts")
    p_par.add_argument("--steps", type=int, default=40, help="Steps per team count")
    p_par.add_argument("--out", type=str, default="", help="Output report JSON")
    p_par.set_defaults(func=cmd_parity)

    # analyze
    p_ana = sub.add_parser("analyze", help="Compute signed gain, regret, ties, and cross-fit metrics")
    p_ana.add_argument("--roots", type=str, required=True, help="Roots JSON file")
    p_ana.add_argument("--branches", type=str, required=True, help="Branches JSON file")
    p_ana.add_argument("--out", type=str, default="", help="Output analysis JSON")
    p_ana.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
