"""Command line of the agent_v2 branch diagnostics (task T0085). The runbook is in README.md.

    python -m tools.search_diagnostics.cli build    p2general --out build/search_diagnostics
    python -m tools.search_diagnostics.cli extract  --matches DIR --build build/search_diagnostics/p2general.json --plan unconditioned:1,contact:1,carrier:1 --salt S --out ROOTS
    python -m tools.search_diagnostics.cli collect  --roots ROOTS --matches DIR --build ... --set primary --repeats 4 --scenario-seed N --seed-ledger LEDGER --workers 8 --out ROWS
    python -m tools.search_diagnostics.cli peval    --matches DIR --build ... --out P.json
    python -m tools.search_diagnostics.cli parity   --counts 2 5 8 15 --steps 160 --seed N --out PARITY.json
    python -m tools.search_diagnostics.cli analyze  --roots ROOTS --rows ROWS [--splits SPLITS.json] [--tau 0.002] --out ANALYSIS.json
    python -m tools.search_diagnostics.cli estimate [--measured ROWS] [--markdown]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fusion import matchlog
from fusion.paths import REPO, ai_test_root, load_engine
from tools.search_diagnostics.schemas import ValidationError, write_gz


def _write(path: str, obj) -> None:
    p = Path(path)
    if p.exists():
        raise ValidationError(f"{p} exists: outputs are never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def _build(path: str) -> dict:
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(b) != {"root_tool", "continuation_bot"}:
        raise ValidationError(f"{path} is not a package build record (cli build)")
    return b


def _matches(directory: str, counts) -> list[Path]:
    paths = list(matchlog.iter_matches(directory, counts))
    if not paths:
        raise ValidationError(f"no matches under {directory}")
    return paths


def cmd_build(a) -> dict:
    from tools.search_diagnostics.build import build_package  # noqa: PLC0415
    ids = build_package(a.package, Path(a.out))
    record = Path(a.out) / f"{a.package}.json"
    record.write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")
    return {"record": str(record), **{k: v["binary_sha256"] for k, v in ids.items()}}


def cmd_extract(a) -> dict:
    from tools.search_diagnostics.build import binary_path  # noqa: PLC0415
    from tools.search_diagnostics.extract import RootRefused, designated_seat, extract_root, select_roots  # noqa: PLC0415
    build = _build(a.build)
    tool = binary_path(build["root_tool"])
    plan = [(s.split(":")[0], int(s.split(":")[1])) for s in a.plan.split(",")]
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise ValidationError(f"{out} is not empty: roots are extracted once")
    rule = f"plan {a.plan}; salt {a.salt}; margin 5; outcome-blind (blocks up to the root and reply presence only)"
    kept, refused = 0, []
    for p in _matches(a.matches, a.counts):
        match = matchlog.load(p)
        seat = designated_seat(match, a.seat_name)
        for row in select_roots(match, seat, plan, a.salt):
            try:
                r = extract_root(p, match, seat, row, build["root_tool"], tool, rule)
            except RootRefused as e:
                refused.append({"match": match["match_id"], "teams": match["teams"], "turn": row["turn"],
                                "stratum": row["stratum"], "reason": str(e)})
                continue
            write_gz(out / f"n{match['teams']:02d}" / f"{r['root_id']}.json.gz", r)
            kept += 1
    _write(str(out / "SELECTION.json"), {"rule": rule, "plan": plan, "salt": a.salt, "kept": kept, "refused": refused,
                                         "root_tool": build["root_tool"]["binary_sha256"]})
    return {"kept": kept, "refused": len(refused)}


def cmd_collect(a) -> dict:
    from tools.search_diagnostics.collect import collect  # noqa: PLC0415
    return collect(Path(a.roots), Path(a.matches), _build(a.build), Path(a.out), ai_test_root(a.ai_test), a.set,
                   a.repeats, a.scenario_seed, Path(a.seed_ledger), a.workers, a.dev)


def cmd_peval(a) -> dict:
    from tools.search_diagnostics.build import binary_path  # noqa: PLC0415
    from tools.search_diagnostics.p_eval import evaluate  # noqa: PLC0415
    build = _build(a.build)
    report = evaluate(build["root_tool"], binary_path(build["root_tool"]), _matches(a.matches, a.counts), a.seat_name)
    _write(a.out, report)
    return {"out": a.out, "games": len(report["games"])}


def cmd_parity(a) -> dict:
    from tools.search_diagnostics.parity import audit, build_check  # noqa: PLC0415
    engine = load_engine(ai_test_root(a.ai_test))
    report = audit(engine, build_check(REPO / "build" / "search_diagnostics" / "check"), a.counts, a.steps, a.seed)
    _write(a.out, report)
    return {"out": a.out, "status": report["status"], "steps": report["steps_compared"]}


def cmd_analyze(a) -> dict:
    from tools.search_diagnostics.analyze import analyze  # noqa: PLC0415
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8")) if a.splits else None
    report = analyze(Path(a.roots), Path(a.rows), a.tau, splits)
    _write(a.out, report)
    return {"out": a.out, "analysed": report["analysed"], "excluded": report["excluded"]}


def cmd_estimate(a):
    from tools.search_diagnostics.estimator import estimate, markdown, measured_seconds  # noqa: PLC0415
    e = estimate(measured=measured_seconds(Path(a.measured)) if a.measured else None, model=a.model)
    return markdown(e) if a.markdown else e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="bundle and compile a package's root tool and continuation bot")
    p.add_argument("package")
    p.add_argument("--out", default=str(REPO / "build" / "search_diagnostics"))
    for name, helptext in (("extract", "select and extract roots"), ("peval", "factual P fidelity")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--matches", required=True)
        p.add_argument("--build", required=True, help="the record `build` wrote")
        p.add_argument("--seat-name", default="@candidate")
        p.add_argument("--counts", type=int, nargs="*")
        p.add_argument("--out", required=True)
        if name == "extract":
            p.add_argument("--plan", required=True, help="stratum:count,... (unconditioned, contact, carrier)")
            p.add_argument("--salt", required=True, help="the registered selection salt")
    p = sub.add_parser("collect", help="play the branches of extracted roots")
    for k in ("--roots", "--matches", "--build", "--seed-ledger", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--set", default="primary", choices=["primary", "extended"])
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--scenario-seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dev", action="store_true", help="allow another worker count; the manifest says not evidence")
    p.add_argument("--ai-test")
    p = sub.add_parser("parity", help="native simulator step against the engine")
    p.add_argument("--counts", type=int, nargs="+", default=[2, 5, 8, 15])
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ai-test")
    p = sub.add_parser("analyze", help="gains, regret, cross-fitting, critic ranking, map-cluster aggregation")
    p.add_argument("--roots", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--splits")
    p.add_argument("--tau", type=float, default=0.002)
    p.add_argument("--out", required=True)
    p = sub.add_parser("estimate", help="exact episode counts; measured time or none")
    p.add_argument("--measured", help="a collection's rows directory (a smoke)")
    p.add_argument("--model", choices=["linear-teams"], help="model unmeasured counts (labelled as modelled)")
    p.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    try:
        out = globals()[f"cmd_{a.cmd}"](a)
    except ValidationError as e:
        print(f"[search_diagnostics] refused: {e}", file=sys.stderr)
        return 2
    print(out if isinstance(out, str) else json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
