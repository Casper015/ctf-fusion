"""Offline structure scan of every bot we have: which ones could gain most from P (and V).

    python p_only/tools/structure_scan.py            # -> p_only/report/STRUCTURE_SCAN.md and structure_scan.json

No games. For every eligible representative of roster/new_bots/inventory.json (the
archived source, else the repository or AI_Test file), the scan reads the code
and records structural signals, then joins the light screen (roster/report/
screen_light/tiers.json: utility, tier, bands, reply times).

Signals (regular expressions over the source with comments removed; a triage
aid, not a proof; read the code of the candidates you shortlist):

  opponent model   functions that forecast or predict opponents (forecast,
                   predict, rival..., enemy...move, opponent...): where P plugs in
  lookahead        simulation of future turns (simulate, rollout, playout,
                   expectimax, minimax, lookahead, step/settle functions): where
                   P is consumed beyond the root and V can score leaves
  sampling         random or hashed scenario sampling (several opponent futures)
  time budget      a wall-clock deadline (steady_clock with a budget/deadline):
                   a bigger network costs search work
  habits           opponent history or habit tracking (to be replaced by P)
  duel split       a separate planner for two teams

PV readiness (triage score, 0-4): +1 opponent model, +1 lookahead, +1 sampling,
+1 reply headroom (screen max reply below 25 ms). Screened bots are listed by
screen utility, the others by readiness. The two carriers already integrated
(C08, v12) are marked.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from roster.fingerprint import bots_from_inventory  # noqa: E402

INVENTORY = REPO / "roster" / "new_bots" / "inventory.json"   # the light screen's inventory (archive + new bots)
TIERS = REPO / "roster" / "report" / "screen_light" / "tiers.json"
OUT = REPO / "p_only" / "report"
INTEGRATED = {"C08_Player5_Omega_LIT": "C08 (P-only built)", "Player5_v12_rollout_volley": "v12 (P-only built)"}

CALL = r"\b({})\s*\("
PATTERNS = {
    "opponent_model": re.compile(CALL.format(r"forecast\w*|predict\w*|rival\w*|enemy\w*(?:Move|Moves|Want|Wants|Order|Orders|Pred)\w*|opponent\w*|foe\w*(?:Move|Pred)\w*"), re.I),
    "lookahead": re.compile(CALL.format(r"simulate\w*|rollout\w*|playout\w*|expectimax\w*|minimax\w*|lookahead\w*|refine\w*|step(?:Exact|Sim|Game)\w*|simStep\w*|settle\w*|search\w*|rank"), re.I),
    "sampling": re.compile(r"mt19937|\brand\s*\(|uniform_real|uniform_int|0x9e3779b9|0\.618\d*|\bsample\w*\s*\(|drawTeam|jointDraw", re.I),
    "time_budget": re.compile(r"steady_clock|high_resolution_clock|perf_counter|time\.time\(\)"),
    "budget_word": re.compile(r"deadline|budget|time_?limit", re.I),
    "habits": re.compile(r"habit\w*|observe\w*\s*\(|lasttarget|tendenc\w*|dropCounts|history\w*\s*\[", re.I),
    "duel_split": re.compile(r"teams\s*==\s*2\s*\?|teams\s*==\s*2\s*\)|n_teams\s*==\s*2|num_teams\s*==\s*2"),
}


def strip_comments(text: str, lang: str) -> str:
    if lang == "py":
        return re.sub(r"#.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def names(rx: re.Pattern, text: str, limit: int = 6) -> list[str]:
    seen = []
    for m in rx.finditer(text):
        n = m.group(1) if m.groups() else m.group(0)
        if n not in seen:
            seen.append(n)
    return seen[:limit]


def scan(path: Path) -> dict:
    lang = "py" if path.suffix == ".py" else "cpp"
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = strip_comments(raw, lang)
    loc = sum(1 for line in text.splitlines() if line.strip())
    out = {"lang": lang, "loc": loc}
    for key in ("opponent_model", "lookahead"):
        out[key] = names(PATTERNS[key], text)
    out["sampling"] = bool(PATTERNS["sampling"].search(text))
    out["time_budget"] = bool(PATTERNS["time_budget"].search(text) and PATTERNS["budget_word"].search(text))
    out["habits"] = names(PATTERNS["habits"], text, 4)
    out["duel_split"] = bool(PATTERNS["duel_split"].search(text))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default=str(INVENTORY))
    ap.add_argument("--tiers", default=str(TIERS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    inv = json.loads(Path(a.inventory).read_text(encoding="utf-8"))
    names_by_id = {b["id"]: b.get("pool_name") or Path(b["path"]).stem for b in inv["bots"]}
    lineage = {b["id"]: b.get("lineage_group") for b in inv["bots"]}
    tiers = {r["bot"]: r for r in json.loads(Path(a.tiers).read_text(encoding="utf-8"))["bots"]} if Path(a.tiers).is_file() else {}
    rows = []
    for b in bots_from_inventory(inv, "archive"):
        p = Path(b["path"])
        r = {"id": b["id"], "name": names_by_id.get(b["id"]), "lineage": lineage.get(b["id"]), "from": b["from"]}
        if not p.is_file():
            r["status"] = "source missing"
            rows.append(r)
            continue
        r.update(scan(p))
        t = tiers.get(b["id"])
        if t:
            r.update({"utility": t["utility"], "lcb": t["lcb"], "tier": t["tier"], "bands": t["bands"],
                      "max_ms": t["max_ms"], "p99_ms_median": t["p99_ms_median"], "screen_rank": t["rank"]})
        headroom = r.get("max_ms") is not None and r["max_ms"] < 25
        r["readiness"] = int(bool(r["opponent_model"])) + int(bool(r["lookahead"])) + int(r["sampling"]) + int(headroom)
        for key, label in INTEGRATED.items():
            if key in (r["name"] or "") or key in b["id"]:
                r["integrated"] = label
        rows.append(r)
    rows.sort(key=lambda r: (-(r.get("utility") is not None), -(r.get("utility") or 0), -r.get("readiness", -1)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "structure_scan.json").write_text(json.dumps({"inventory": str(Path(a.inventory).relative_to(REPO)),
                                                         "bots": rows}, indent=1) + "\n", encoding="utf-8")
    screened = [r for r in rows if r.get("utility") is not None]
    md = ["# Structure scan: where P and V could plug in", "",
          f"{len(rows)} eligible bots from `{Path(a.inventory).relative_to(REPO)}`; {len(screened)} have light-screen results "
          "(70 paired games each, `roster/report/screen_light`). Signals come from regular expressions over the code: "
          "a triage aid, not a verdict. Readiness (0-4): opponent model, lookahead, sampling, max reply below 25 ms. "
          "Generated by `p_only/tools/structure_scan.py`; no games were played.", "",
          "## Screened bots, by screen utility", "",
          "| # | Bot | Utility [95% low] | Tier | 2 / 3-5 / 6-10 / 11-15 | Max ms | LOC | Ready | Opponent model | Lookahead | Sampling | Budget | Habits | Duel split |",
          "|---:|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for i, r in enumerate(screened, 1):
        bands = " / ".join(f"{r['bands'].get(k, float('nan')):.2f}" for k in ("2", "3-5", "6-10", "11-15"))
        name = (r["name"] or Path(r["id"]).stem) + (f" **[{r['integrated']}]**" if r.get("integrated") else "")
        md.append(f"| {i} | {name} | {r['utility']:.3f} [{r['lcb']:.3f}] | {r['tier']} | {bands} | {r['max_ms']} | {r['loc']} | "
                  f"{r['readiness']} | {', '.join(r['opponent_model']) or '-'} | {', '.join(r['lookahead']) or '-'} | "
                  f"{'yes' if r['sampling'] else '-'} | {'yes' if r['time_budget'] else '-'} | {', '.join(r['habits']) or '-'} | "
                  f"{'yes' if r['duel_split'] else '-'} |")
    rest = [r for r in rows if r.get("utility") is None]
    md += ["", f"## Not screened ({len(rest)})", "", "| Bot | LOC | Ready | Opponent model | Lookahead | Status |", "|---|---:|---:|---|---|---|"]
    for r in rest:
        md.append(f"| {r['name'] or Path(r['id']).stem} | {r.get('loc', '')} | {r.get('readiness', '')} | "
                  f"{', '.join(r.get('opponent_model') or []) or '-'} | {', '.join(r.get('lookahead') or []) or '-'} | {r.get('status', '')} |")
    (out / "STRUCTURE_SCAN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(rows)} bots ({len(screened)} screened) -> {out / 'STRUCTURE_SCAN.md'}")


if __name__ == "__main__":
    main()
