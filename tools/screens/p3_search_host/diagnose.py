"""Search diagnostics for task T0081, computed only from committed receipts (task T0090; audit task T0084).

    python tools/screens/p3_search_host/diagnose.py --preflight reports/host/agent_v2_p3_search/PREFLIGHT.json \\
        --out reports/host/agent_v2_p3_search/diagnostics/

The version that T0081 landed (a27f310) returned constants for candidate headroom, P fidelity, simulator parity and V
regret without reading any input. Those are withdrawn (reports/host/agent_v2_p3_search/CORRECTION.md). Here each
dimension either computes its numbers from an input it read and hashed, or reports status "unavailable" with the
reason and the input it would need. Nothing is ever filled in with a default.

Computable today: search work per variant and team count (completed trees, fallbacks, refusal causes, overrides,
gate-kept trees) from the preflight's per-game counters. The preflight covers counts 2, 5, 8 and 15 only; the formal
screen's committed receipts carry no per-game counters, so the screen's own completion stays unavailable here.

The JSON and the Markdown are rendered from one object. Neither carries a timestamp, so a rerun on the same input
reproduces both byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "ctf-search-failure-diagnosis-2"
PREFLIGHT_SCHEMA = "ctf-agent-v2-p3-search-preflight-1"
WITHDRAWN = "a27f310"

# The search-gate bot's cumulative stderr line (bots/agent_v2/search_gate/bot.hpp): the counters preflight.py parses,
# then "overrides <n> gate_kept <n>", which preflight.py does not parse.
TAIL_RE = re.compile(r"AGENTV2 turns (\d+) .*\boverrides (\d+) gate_kept (\d+)\s*$")
REFUSALS = ("refuse_setup", "refuse_trajectory", "refuse_expand", "refuse_critic", "refuse_late")

UNAVAILABLE = {
    "dim1_candidate_quality": (
        "Needs terminal utilities of continuations from each root candidate under a frozen continuation policy. "
        "T0081 collected none; the harness is T0085 and the collection T0086.",
        ["branch continuation rows with candidate orders, scenario keys and terminal utilities"]),
    "dim2_opponent_prediction": (
        "Needs P2 and P3 evaluated on the same held-out causal labels, with event definitions and denominators. "
        "No such evaluation was run or read for T0081.",
        ["held-out P evaluation receipt for both general packages on identical labels"]),
    "dim3_simulator_parity": (
        "Needs both the native simulator and the engine run on identical full commands and refresh outcomes. The "
        "withdrawn version loaded the engine and counted stream blocks, but compared nothing. Existing older "
        "coverage, not a T0081 audit: tests/test_agent_v2.py::test_simulator_matches_the_engine_and_refreshes_lawfully "
        "(T0065; random lawful commands, counts 2-15).",
        ["paired native/engine transition records on the T0081 binaries"]),
    "dim4_value_ranking_regret": (
        "Needs the critic's root scores next to realized continuation utilities for the same candidates. T0081 "
        "recorded neither; the screen's committed receipts also carry no override counters.",
        ["per-root critic scores and continuation utilities (T0086)"]),
}


class InputError(ValueError):
    """The input is missing, changed or malformed; nothing is written."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preflight(path: Path, expect_sha256: str | None) -> tuple[dict, dict]:
    if not path.is_file():
        raise InputError(f"preflight {path} not found")
    digest = sha256_file(path)
    if expect_sha256 and digest != expect_sha256:
        raise InputError(f"preflight {path} has sha256 {digest}, expected {expect_sha256}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InputError(f"preflight {path} is not JSON: {e}") from None
    if not isinstance(data, dict) or data.get("schema") != PREFLIGHT_SCHEMA:
        raise InputError(f"preflight schema is not {PREFLIGHT_SCHEMA!r}")
    if not isinstance(data.get("games"), list) or not data["games"]:
        raise InputError("preflight has no per-game rows")
    try:
        shown = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        shown = path.name
    return data, {"path": shown, "sha256": digest, "bytes": path.stat().st_size, "seed": data.get("seed")}


def game_work(row: dict) -> tuple[dict | None, str | None]:
    """One game's search work, or None and the reason it cannot be read."""
    c = row.get("counters")
    if not c:
        return None, "no counters"
    if c.get("turns") != row.get("turns_played"):
        return None, f"counters cover {c.get('turns')} of {row.get('turns_played')} turns"
    if any(k not in c for k in ("search_used", "search_fallback", *REFUSALS)):
        return None, "search or refusal counters missing"
    if c["search_used"] + c["search_fallback"] != c["turns"]:
        return None, "search_used + search_fallback != turns"
    work = {"turns": c["turns"], "search_used": c["search_used"], "search_fallback": c["search_fallback"]}
    work |= {k: c[k] for k in REFUSALS}
    tail = (row.get("stderr_tail") or [""])[-1]
    m = TAIL_RE.search(tail)
    if not m or int(m.group(1)) != c["turns"]:
        return None, "override counters missing from the last stderr line"
    work["overrides"], work["gate_kept"] = int(m.group(2)), int(m.group(3))
    if work["overrides"] + work["gate_kept"] > work["search_used"]:
        return None, "overrides + gate_kept exceed completed trees"
    return work, None


def compute_allocation(data: dict, source: dict) -> dict:
    """Dimension 5: search work per variant and count, from the preflight's per-game counters."""
    search = sorted(n for n, v in data.get("variants", {}).items() if v.get("search"))
    if not search:
        raise InputError("preflight lists no search variant")
    rows, problems = {}, []
    for name in search:
        games = [g for g in data["games"] if g.get("variant") == name]
        if not games:
            raise InputError(f"preflight has no games for {name}")
        for count in sorted({g["teams"] for g in games}):
            cell = [g for g in games if g["teams"] == count]
            works, bad = [], []
            for g in cell:
                w, why = game_work(g)
                if w:
                    works.append(w)
                else:
                    bad.append(f"{g.get('fixture_id')}: {why}")
            key = f"{name} n{count:02d}"
            if bad:
                problems += [f"{key} {b}" for b in bad]
                rows[key] = {"variant": name, "teams": count, "games": len(cell), "status": "unavailable",
                             "reason": f"{len(bad)} of {len(cell)} games unreadable"}
                continue
            tot = {k: sum(w[k] for w in works) for k in works[0]}
            used = tot["search_used"]
            rows[key] = {
                "variant": name, "teams": count, "games": len(cell), "status": "computed", **tot,
                "completed_share": round(used / tot["turns"], 4),
                "override_share_of_turns": round(tot["overrides"] / tot["turns"], 4),
                "override_share_of_completed": round(tot["overrides"] / used, 4) if used else None,
            }
    return {
        "status": "computed" if not problems else "partial",
        "source": source,
        "scope": "preflight only: counts 2, 5, 8 and 15, 16 fixtures each. The formal screen's committed receipts "
                 "carry no per-game counters, so the screen's own completion is unavailable.",
        "definitions": {
            "completed_share": "search_used / turns: turns whose tree completed (the gate then kept or replaced the "
                               "actor); the rest fell back to the actor's order",
            "override_share_of_turns": "overrides / turns: completed trees whose best root replaced the actor",
            "gate_kept": "completed trees whose best root beat the actor by at most tau",
        },
        "rows": rows,
        "problems": problems,
    }


def diagnose(preflight: Path, expect_sha256: str | None = None) -> dict:
    data, source = load_preflight(preflight, expect_sha256)
    dims = {k: {"status": "unavailable", "reason": r, "needs": n} for k, (r, n) in UNAVAILABLE.items()}
    dims["dim5_compute_allocation"] = compute_allocation(data, source)
    return {
        "schema": SCHEMA,
        "task": "T0081 diagnostics, corrected by T0090",
        "withdrawn": {"commit": WITHDRAWN, "record": "reports/host/agent_v2_p3_search/CORRECTION.md"},
        "inputs": {"preflight": source},
        "dimensions": dict(sorted(dims.items())),
        "root_cause": "not determined: four of five dimensions have no measurement",
    }


def fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def render_markdown(report: dict) -> str:
    src = report["inputs"]["preflight"]
    out = [
        "# T0081 search diagnostics (corrected)",
        "",
        f"Generated by `tools/screens/p3_search_host/diagnose.py` from `{src['path']}` (seed {src['seed']}, sha256 "
        f"`{src['sha256']}`). Do not edit by hand. The earlier version of this file ({report['withdrawn']['commit']}) "
        f"reported constants that no code computed; see `{report['withdrawn']['record']}`.",
        "",
        f"Root cause: {report['root_cause']}.",
        "",
        "## Dimensions",
        "",
        "| Dimension | Status | Reason or scope |",
        "|---|---|---|",
    ]
    for k, d in report["dimensions"].items():
        out.append(f"| {k} | {d['status']} | {d.get('reason', d.get('scope', ''))} |")
    alloc = report["dimensions"]["dim5_compute_allocation"]
    cols = ("games", "turns", "search_used", "search_fallback", "completed_share", "overrides", "gate_kept",
            "override_share_of_turns", "override_share_of_completed")
    out += ["", "## Search work by variant and count (preflight)", "",
            "| Variant | Teams | " + " | ".join(cols) + " | Refusals (setup/trajectory/expand/critic/late) |",
            "|---|---:|" + "---:|" * len(cols) + "---|"]
    for r in alloc["rows"].values():
        if r["status"] != "computed":
            out.append(f"| {r['variant']} | {r['teams']} | {r['games']} | unavailable: {r['reason']} |")
            continue
        refusals = "/".join(str(r[k]) for k in REFUSALS)
        out.append(f"| {r['variant']} | {r['teams']} | " + " | ".join(fmt(r[k]) for k in cols) + f" | {refusals} |")
    out += ["", "Definitions:", ""] + [f"- `{k}`: {v}." for k, v in alloc["definitions"].items()]
    if alloc["problems"]:
        out += ["", "Unreadable games:", ""] + [f"- {p}" for p in alloc["problems"]]
    return "\n".join(out) + "\n"


def write(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "SEARCH_FAILURE_DIAGNOSIS.json", out_dir / "SEARCH_FAILURE_DIAGNOSIS.md"
    jp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    mp.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return jp, mp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", required=True, help="the T0081 PREFLIGHT.json")
    ap.add_argument("--expect-sha256", default=None, help="refuse unless the preflight has this sha256")
    ap.add_argument("--out", default=str(REPO / "reports" / "host" / "agent_v2_p3_search" / "diagnostics"))
    a = ap.parse_args(argv)
    try:
        report = diagnose(Path(a.preflight), a.expect_sha256)
    except InputError as e:
        print(f"[diagnose] refused: {e}; nothing written", file=sys.stderr)
        return 2
    jp, mp = write(report, Path(a.out))
    print(f"[diagnose] wrote {jp} and {mp} (dim5 {report['dimensions']['dim5_compute_allocation']['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
