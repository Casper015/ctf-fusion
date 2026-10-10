"""Phase 1 size rule: which P sizes enter the paired screen, from the timing run.

    python p_only/tools/run_screen.py --config p_only/profiles/timing.json   # host, screen load
    python p_only/tools/size_rule.py                                        # report the decision
    python p_only/tools/size_rule.py --apply                                # also rewrite p_only/profiles/screen.json

Rule (fixed before the timing run):
  C08  the largest P-only size, from XXL down to M, whose worst per-game p99
       reply is at most 40 ms (10 ms headroom under the 50 ms limit) with no
       fault and no late reply. If none qualifies, the screen keeps M and the
       report says so: reduce C08's ply-2 work before screening.
  v12  XL and XXL by default; a size with a fault or a late reply drops out.
The worst per-game p99 is conservative: a game's p99 bounds 99% of its replies,
and the rule takes the worst game. --apply sets the screen's C08 P-only arm and
regenerates the comparisons; everything else in screen.json stays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIMING = REPO / "p_only" / "report" / "timing" / "screen.json"
SCREEN = REPO / "p_only" / "profiles" / "screen.json"
LIMIT_MS = 40.0


def clean(arm: dict) -> bool:
    return arm["games"] > 0 and arm["faults"] == 0 and arm["late_replies"] == 0


def decide(timing: dict) -> dict:
    arms = timing["arms"]
    c08, why = None, []
    for size in ("xxl", "xl", "l", "m"):
        a = arms.get(f"c08_ponly_{size}")
        if not a:
            why.append(f"c08_ponly_{size}: not measured")
            continue
        ok = clean(a) and a["p99_ms_worst"] is not None and a["p99_ms_worst"] <= LIMIT_MS
        why.append(f"c08_ponly_{size}: worst-game p99 {a['p99_ms_worst']} ms, max {a['max_ms']} ms, "
                   f"faults {a['faults']}, late {a['late_replies']} -> {'admitted' if ok else 'rejected'}")
        if ok and c08 is None:
            c08 = f"c08_ponly_{size}"
    v12 = []
    for size in ("xl", "xxl"):
        a = arms.get(f"v12_ponly_{size}")
        ok = bool(a) and clean(a)
        why.append(f"v12_ponly_{size}: " + (f"faults {a['faults']}, late {a['late_replies']}, worst-game p99 {a['p99_ms_worst']} ms"
                                            if a else "not measured") + f" -> {'admitted' if ok else 'rejected'}")
        if ok:
            v12.append(f"v12_ponly_{size}")
    return {"c08": c08, "v12": v12, "reasons": why}


def screen_arms(c08: str, v12: list[str]) -> tuple[list[str], list[list[str]]]:
    arms = ["c08_control", "c08_control_2ply", "c08_hybrid_m", c08, "v12_control", "v12_hybrid_m", *v12]
    comp = [[c08, "c08_control"], [c08, "c08_control_2ply"], ["c08_control_2ply", "c08_control"], [c08, "c08_hybrid_m"]]
    comp += [[v, "v12_control"] for v in v12]
    if len(v12) == 2:
        comp.append([v12[1], v12[0]])
    if v12:
        comp.append([v12[0], "v12_hybrid_m"])
    comp += [[c08, v] for v in v12]
    return arms, comp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing", default=str(TIMING))
    ap.add_argument("--apply", action="store_true", help="rewrite the arms and comparisons of p_only/profiles/screen.json")
    a = ap.parse_args(argv)
    d = decide(json.loads(Path(a.timing).read_text(encoding="utf-8")))
    c08 = d["c08"] or "c08_ponly_m"
    lines = ["# Phase 1 size rule", "", f"Rule: C08 gets the largest P-only size with worst-game p99 <= {LIMIT_MS:g} ms, no fault and "
             "no late reply; v12 gets XL and XXL unless one faults or replies late.", ""] + [f"- {r}" for r in d["reasons"]]
    lines += ["", f"Decision: C08 {c08}" + ("" if d["c08"] else " (NO size qualified: reduce C08's ply-2 work before the screen)")
              + f"; v12 {', '.join(d['v12']) or 'none'}."]
    out = Path(a.timing).parent / "SIZE_RULE.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if a.apply:
        cfg = json.loads(SCREEN.read_text(encoding="utf-8"))
        cfg["arms"], cfg["comparisons"] = screen_arms(c08, d["v12"])
        SCREEN.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
        print(f"applied to {SCREEN.relative_to(REPO)}: arms {cfg['arms']}")


if __name__ == "__main__":
    main()
