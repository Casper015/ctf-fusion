"""Paired analysis of a P-only screen: what each arm gains over another on identical fixtures.

    python bots/p_only/tools/analyze.py --results data/p_only/screen/results.jsonl --schedule data/p_only/screen/schedule.json \\
        --manifest build/p_only/manifest.json --arms c08_control c08_ponly_m --aa c08_control \\
        --compare c08_ponly_m:c08_control --out bots/p_only/report/screen [--matches data/p_only/screen/matches]

Rows count only when their binary SHA-256 is the manifest's for that arm (older
builds are reported as stale) and their schedule is this schedule. Each fixture
is one independent map group; the result per game is the arm's pairwise utility
(each other team beaten 1, tied 0.5, divided by teams - 1).

For "A:B" the per-fixture difference u_A - u_B is averaged per team count, the
counts are weighted equally (overall) or within the bands 2, 3-5, 6-10 and 11-15,
and the 95% interval is mean +- 1.96 * sqrt(sum_c var_c / n_c) / C. Verdict:
"helps" when the interval is above zero, "hurts" when below, else "not shown".
improved/unchanged/degraded count fixtures by the sign of the paired difference;
they are not game wins. An aa arm ("X~aa") replays X's identical binary: X~aa:X
measures re-run noise (and its rate of identical final scores).

Timing per arm: faults (late replies count as faults in fusion/runner.py), dead
games, late replies, the largest first reply, the median and the worst of the
per-game p99 (the size rule of bots/p_only/tools/size_rule.py uses the worst), and the
largest single reply (max over games of the per-game max). With --matches (kept
logs of a --stats build) the network call counters per turn are added, and for
C08 arms the expectimax calls and joint samples per call.

Exit status 2 when any arm misses a scheduled fixture (the report says INCOMPLETE).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from roster.strength import schedule_identity  # noqa: E402

AA = "~aa"
BANDS = {"2": (2, 2), "3-5": (3, 5), "6-10": (6, 10), "11-15": (11, 15)}
COUNTER = re.compile(r"PONLY builds (\d+) teams (\d+) act (\d+) target (\d+) fallback (\d+)(?: inconsistent (\d+) skipped (\d+))?")
LINE = re.compile(r"^(SEARCH|VALUE|PLY2|PORTFOLIO|VBUDGET) (.*)$")
SETTINGS = {"fixed", "mode", "cache", "k", "moves", "max_calls", "deadline_ms"}  # build settings, not counters
PAIR = re.compile(r"(\w+) (-?[0-9.]+(?:e-?\d+)?)")
TERM = re.compile(r"([+-]?)\s*([A-Za-z0-9_~]+)")


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binary_of(arm: str, manifest: dict) -> str:
    return manifest[arm.removesuffix(AA)]["binary_sha256"]


def parse_contrast(spec: str) -> tuple[str, list[tuple[int, str]]]:
    """'name=D-C-B+A' -> (name, [(+1, D), (-1, C), (-1, B), (+1, A)])."""
    name, _, expr = spec.partition("=")
    terms = [(-1 if s == "-" else 1, arm) for s, arm in TERM.findall(expr)]
    if not terms:
        raise SystemExit(f"empty contrast {spec!r}")
    return name.strip(), terms


def contrast(terms: list[tuple[int, dict]], fixtures: dict, teams: list[int]) -> dict:
    """Per fixture sum_i w_i * u_i over the fixtures every arm played (A - B for two terms)."""
    per = defaultdict(list)
    same = 0
    common = set(fixtures)
    for _, rows in terms:
        common &= set(rows)
    for fid in sorted(common):
        per[fixtures[fid]["teams"]].append(sum(w * rows[fid]["utility"] for w, rows in terms))
        if len(terms) == 2:
            same += terms[0][1][fid]["score"] == terms[1][1][fid]["score"]
    def summary(counts):
        counts = [c for c in counts if per.get(c)]
        if not counts:
            return None
        means = [sum(per[c]) / len(per[c]) for c in counts]
        var = 0.0
        for c, m in zip(counts, means):
            n = len(per[c])
            if n > 1:
                var += sum((x - m) ** 2 for x in per[c]) / (n - 1) / n
        mean = sum(means) / len(counts)
        se = math.sqrt(var) / len(counts)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        n = sum(len(per[c]) for c in counts)
        return {"mean": round(mean, 5), "ci95": [round(lo, 5), round(hi, 5)], "se": round(se, 5), "fixtures": n,
                "verdict": "helps" if lo > 0 else "hurts" if hi < 0 else "not shown"}
    diffs = [x for c in per for x in per[c]]
    out = {"overall": summary(teams), "bands": {b: summary([c for c in teams if lo <= c <= hi]) for b, (lo, hi) in BANDS.items()},
           "per_count": {str(c): round(sum(per[c]) / len(per[c]), 5) for c in teams if per.get(c)},
           "improved": sum(x > 0 for x in diffs), "unchanged": sum(x == 0 for x in diffs), "degraded": sum(x < 0 for x in diffs),
           "identical_scores": same, "fixtures": len(diffs)}
    return out


def paired(rows_a: dict, rows_b: dict, fixtures: dict, teams: list[int]) -> dict:
    """Paired difference A - B over the fixtures both played."""
    return contrast([(1, rows_a), (-1, rows_b)], fixtures, teams)


def counters(matches: Path, arm_keys: dict) -> dict:
    """Network calls per turn of each arm's seat, from kept match logs (PONLY stderr line)."""
    acc = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 5 counters, turns, inconsistent, skipped, -, -, games
    extra = defaultdict(lambda: defaultdict(float))  # SEARCH and VALUE lines: summed counters, plus turns
    for p in matches.rglob("*.json.gz"):
        key = p.name.split("-")[-1].split(".")[0]
        arm = arm_keys.get(key)
        if arm is None:
            continue
        m = json.loads(gzip.decompress(p.read_bytes()))
        for seat, bot in zip(m["seats"], m["bots"]):
            if not seat["name"].startswith("@candidate"):
                continue
            a = acc[arm]
            a[10] += 1
            for line in bot.get("stderr_tail", []):
                c = COUNTER.search(line)
                if c:
                    for i in range(5):
                        a[i] += int(c.group(i + 1))
                    a[5] += len(m.get("blocks", [])) or 1
                    a[6] += int(c.group(6) or 0)
                    a[7] += int(c.group(7) or 0)
                s = LINE.search(line.strip())
                if s:
                    e = extra[arm]
                    for k, v in PAIR.findall(s.group(2)):
                        if not k.startswith("coef_") and k not in SETTINGS:
                            e[f"{s.group(1).lower()}.{k}"] += float(v)
                        else:
                            e[f"{s.group(1).lower()}.{k}"] = float(v)
                    e[f"{s.group(1).lower()}.turns"] += len(m.get("blocks", [])) or 1
    out = {}
    for arm, a in acc.items():
        row = {}
        if a[5]:
            row.update({"turns": a[5], "views_per_turn": round(a[0] / a[5], 2), "team_encodes_per_turn": round(a[1] / a[5], 2),
                        "act_heads_per_turn": round(a[2] / a[5], 2), "target_heads_per_turn": round(a[3] / a[5], 2),
                        "fallbacks": a[4], "inconsistent": a[6], "skipped": a[7]})
        e = extra.get(arm, {})
        calls = e.get("search.calls", 0)
        if calls:
            row.update({"search_calls": int(calls), "samples_per_search": round(e["search.samples"] / calls, 1),
                        "requested_per_search": round(e.get("search.requested", 0) / calls, 1),
                        "truncated_share": round(e.get("search.truncated", 0) / calls, 4),
                        "rollouts_per_search": round(e["search.rollouts"] / calls, 1) if "search.rollouts" in e else None,
                        "plans_per_search": round(e.get("search.plans", 0) / calls, 1),
                        "survivors_per_search": round(e["search.survivors"] / calls, 2) if "search.survivors" in e else None,
                        "fixed_samples": int(e.get("search.fixed", 0)),
                        "refine_cut_share": round(e["search.refine_cuts"] / e["search.refines"], 4) if e.get("search.refines") else None})
        if e.get("ply2.predictions") is not None and calls:
            row["ply2"] = {"mode": int(e.get("ply2.mode", 0)), "cache": int(e.get("ply2.cache", 0)),
                           "predictions_per_search": round(e["ply2.predictions"] / calls, 1),
                           "prediction_ms_per_search": round(1000 * e.get("ply2.seconds", 0) / calls, 3),
                           "view_builds_per_search": round(e.get("ply2.view_builds", 0) / calls, 1),
                           "view_hit_share": round(e.get("ply2.view_hits", 0) / e["ply2.view_calls"], 4) if e.get("ply2.view_calls") else None,
                           "memo_hit_share": round(e.get("ply2.memo_hits", 0) / (e.get("ply2.memo_hits", 0) + e.get("ply2.memo_misses", 0)), 4)
                           if e.get("ply2.memo_hits", 0) + e.get("ply2.memo_misses", 0) else None}
            if e.get("ply2.diag_searches"):
                n = e["ply2.diag_searches"]
                row["ply2"] |= {"diag_move_diff_share": round(e.get("ply2.diag_move_diff", 0) / e["ply2.diag_pairs"], 4) if e.get("ply2.diag_pairs") else None,
                                "diag_aim_l1_per_pair": round(e.get("ply2.diag_aim_l1", 0) / e["ply2.diag_pairs"], 4) if e.get("ply2.diag_pairs") else None,
                                "diag_first_top_changed_share": round(e.get("ply2.diag_first_top_changed", 0) / n, 4),
                                "diag_final_top_changed_share": round(e.get("ply2.diag_top_changed", 0) / n, 4),
                                "diag_first_swaps_per_search": round(e.get("ply2.diag_first_swaps", 0) / n, 2)}
        if e.get("portfolio.calls"):
            turns = e.get("portfolio.turns", 0) or 1
            row["portfolio"] = {"k": int(e.get("portfolio.k", 0)), "moves": int(e.get("portfolio.moves", 0)),
                                "assignments_per_call": round(e.get("portfolio.assignments", 0) / e["portfolio.calls"], 2),
                                "waits_per_call": round(e.get("portfolio.waits", 0) / e["portfolio.calls"], 2),
                                "searched_per_turn": round(e.get("portfolio.searched", 0) / turns, 2),
                                "refined_per_turn": round(e.get("portfolio.refined", 0) / turns, 2),
                                "same_move_sets": int(e.get("portfolio.same_move", 0)), "cuts": int(e.get("portfolio.cut", 0)),
                                "overkill_skipped": int(e.get("portfolio.overkill_skipped", 0))}
        if e.get("vbudget.skipped_calls") is not None:
            row["vbudget"] = {"skipped_calls": int(e["vbudget.skipped_calls"]), "fallback_decisions": int(e.get("vbudget.fallback_decisions", 0))}
        if any(k.startswith("value.") for k in e):
            turns = e.get("value.turns", 0) or 1
            v = {"value_seconds_per_turn_ms": round(1000 * e.get("value.seconds", 0) / turns, 3)}
            for path in ("refine", "beam", "rollout"):
                d = e.get(f"value.{path}_decisions", 0)
                if f"value.{path}_calls" in e:
                    v[path] = {"calls_per_turn": round(e[f"value.{path}_calls"] / turns, 3), "exact": int(e.get(f"value.{path}_exact", 0)),
                               "decisions": int(d), "changed": int(e.get(f"value.{path}_changed", 0)),
                               "changed_share": round(e.get(f"value.{path}_changed", 0) / d, 4) if d else None,
                               "coefficient": e.get(f"value.coef_{path}")}
            row["value"] = v
        if row:
            out[arm] = row
    return out


def fmt_ci(s):
    if not s:
        return "n/a", "n/a", ""
    return f"{s['mean']:+.4f}", f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]", s["verdict"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--aa", nargs="*", default=[])
    ap.add_argument("--compare", action="append", default=[], help="A:B, the paired difference A - B")
    ap.add_argument("--contrast", action="append", default=[], help="name=D-C-B+A: a signed sum of arms per fixture (interactions)")
    ap.add_argument("--identities", help="identities.json from bots/p_only/tools/screen_runner.py: rows of another identity are stale")
    ap.add_argument("--matches", help="kept match logs (network call counters)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sched = json.loads(Path(a.schedule).read_text(encoding="utf-8"))
    sid = schedule_identity(sched)
    fixtures = {fx["fixture_id"]: fx for fx in sched["fixtures"]}
    teams = sorted({fx["teams"] for fx in fixtures.values()})
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))["variants"]
    arms = list(a.arms) + [x + AA for x in a.aa]
    ids = json.loads(Path(a.identities).read_text(encoding="utf-8")) if a.identities and Path(a.identities).is_file() else None
    rows = {arm: {} for arm in arms}
    stale = defaultdict(int)
    for r in load_rows(Path(a.results)):
        arm = r["candidate"]
        if arm not in rows:
            continue
        current = (r.get("identity") == ids[arm]["digest"]) if ids and arm in ids else r["source_sha256"] == binary_of(arm, manifest)
        if r.get("schedule") != sid or not current:
            stale[arm] += 1
            continue
        rows[arm][r["fixture_id"]] = r
    missing = {arm: sorted(set(fixtures) - set(rows[arm])) for arm in arms}
    incomplete = {arm: len(m) for arm, m in missing.items() if m}

    per_arm = {}
    for arm in arms:
        rs = list(rows[arm].values())
        by = defaultdict(list)
        for r in rs:
            by[r["teams"]].append(r["utility"])
        means = [sum(v) / len(v) for v in by.values()]
        p99 = sorted(r["p99_ms"] for r in rs if r.get("p99_ms") is not None)
        per_arm[arm] = {"games": len(rs), "binary_sha256": binary_of(arm, manifest),
                        "utility": round(sum(means) / len(means), 5) if means else None,
                        "faults": sum(r["faults"] for r in rs), "dead_games": sum(1 for r in rs if r.get("dead")),
                        "late_replies": sum(r.get("late") or 0 for r in rs),
                        "first_ms_max": max((r["first_ms"] for r in rs if r.get("first_ms") is not None), default=None),
                        "p99_ms_median": p99[len(p99) // 2] if p99 else None,
                        "p99_ms_worst": p99[-1] if p99 else None,
                        "max_ms": max((r["max_ms"] for r in rs if r.get("max_ms") is not None), default=None)}
    comparisons = {}
    for spec in a.compare:
        x, _, y = spec.partition(":")
        if x in rows and y in rows:
            comparisons[f"{x} - {y}"] = paired(rows[x], rows[y], fixtures, teams)
    for spec in a.contrast:
        name, terms = parse_contrast(spec)
        if all(arm in rows for _, arm in terms):
            comparisons[name] = contrast([(w, rows[arm]) for w, arm in terms], fixtures, teams)
    noise = {f"{x}{AA} - {x}": paired(rows[x + AA], rows[x], fixtures, teams) for x in a.aa if x in rows}
    calls = {}
    if a.matches and Path(a.matches).is_dir():
        import hashlib
        keys = {hashlib.sha256(arm.encode()).hexdigest()[:10]: arm for arm in arms}
        calls = counters(Path(a.matches), keys)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"schedule": sid, "seed": sched.get("seed"), "fixtures": len(fixtures), "team_counts": teams,
              "incomplete": incomplete, "stale_rows": dict(stale), "arms": per_arm, "comparisons": comparisons,
              "rerun_noise": noise, "network_calls": calls,
              "identities": {arm: ids[arm]["digest"] for arm in arms if ids and arm in ids}}
    (out / "screen.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    md = [f"# P-only paired screen{' (INCOMPLETE)' if incomplete else ''}", "",
          f"Schedule `{sid}`, seed {sched.get('seed')}, {len(fixtures)} fixtures over team counts {teams[0]}-{teams[-1]} "
          f"({len(fixtures) // max(1, len(teams))} maps per count). Utility: share of the other teams beaten (ties half). "
          "Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by "
          "the sign of the paired difference; they are not wins.", ""]
    if incomplete:
        md += ["**Missing fixtures:** " + ", ".join(f"{k} {v}" for k, v in incomplete.items()), ""]
    if stale:
        md += ["Stale rows ignored (another identity, binary or schedule): " + ", ".join(f"{k} {v}" for k, v in stale.items()), ""]
    md += ["## Paired differences", "", "| Comparison | Band | Mean | 95% interval | Verdict | Fixtures |", "|---|---|---:|---|---|---:|"]
    for name, c in list(comparisons.items()) + list(noise.items()):
        for band, s in [("all", c["overall"])] + list(c["bands"].items()):
            m, ci, v = fmt_ci(s)
            md.append(f"| {name} | {band} | {m} | {ci} | {v} | {s['fixtures'] if s else 0} |")
        md.append(f"| {name} | improved / unchanged / degraded | {c['improved']} / {c['unchanged']} / {c['degraded']} | "
                  f"identical scores {c['identical_scores']} of {c['fixtures']} | | |")
    md += ["", "## Arms", "", "| Arm | Games | Utility | Faults | Dead | Late | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, s in per_arm.items():
        md.append(f"| {arm} | {s['games']} | {s['utility'] if s['utility'] is not None else 'n/a'} | {s['faults']} | {s['dead_games']} | "
                  f"{s['late_replies']} | {s['first_ms_max']} | {s['p99_ms_median']} | {s['p99_ms_worst']} | {s['max_ms']} | `{s['binary_sha256'][:12]}` |")
    if calls:
        md += ["", "## Network calls per turn and search samples (our seat)", "",
               "| Arm | Turns | Views | Team encodes | Act heads | Target heads | Fallbacks | Inconsistent | Flag-return states | Expectimax calls | Samples per call |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, c in calls.items():
            md.append(f"| {arm} | {c.get('turns', '')} | {c.get('views_per_turn', '')} | {c.get('team_encodes_per_turn', '')} | "
                      f"{c.get('act_heads_per_turn', '')} | {c.get('target_heads_per_turn', '')} | {c.get('fallbacks', '')} | "
                      f"{c.get('inconsistent', '')} | {c.get('skipped', '')} | {c.get('search_calls', '')} | {c.get('samples_per_search', '')} |")
    searched = {arm: c for arm, c in calls.items() if c.get("search_calls")}
    if searched:
        md += ["", "## Search work (C08, our seat)", "",
               "Truncated: a deadline stopped the call before its requested samples. A fixed-sample arm (fixed > 0) is an "
               "equal-work comparison only for its untruncated calls.", "",
               "| Arm | Fixed | Calls | Samples per call | Requested per call | Truncated share | Rollouts per call | Plans per call | Survivors per call | Lookahead cut share |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, c in searched.items():
            md.append(f"| {arm} | {c['fixed_samples']} | {c['search_calls']} | {c['samples_per_search']} | {c['requested_per_search']} | "
                      f"{c['truncated_share']} | {c['rollouts_per_search'] if c['rollouts_per_search'] is not None else 'n/a'} | "
                      f"{c['plans_per_search']} | {c['survivors_per_search'] if c['survivors_per_search'] is not None else 'n/a'} | "
                      f"{c['refine_cut_share'] if c['refine_cut_share'] is not None else 'n/a'} |")
    experimental = {arm: c for arm, c in calls.items() if c.get("ply2") or c.get("portfolio")}
    if experimental:
        md += ["", "## Search experiment counters (our seat; docs/SEARCH_PORTFOLIO.md)", "",
               "Ply 2 (C08): predictions and their time per search call, distinct successor states built, P-view cache and "
               "settlement-memo hit shares; diagnostic arms (mode 2) also report how often a plan's own ply-2 draw differs from "
               "the shared one and how often the best plan changes. Portfolio: complete assignments per shortlist call, "
               "alternatives entering the sampled search and the refinement per turn, candidate sets holding two orders of one "
               "movement, deadline cuts inside the alternatives. Timings are host timings only when this report comes from the host.", "",
               "| Arm | Ply-2 mode | Predictions per call | Prediction ms per call | States built per call | View hits | Memo hits | "
               "Draws differ | Top changed (first / final) | Assignments per call | Searched per turn | Refined per turn | Same-move sets | Cuts |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
        for arm, c in experimental.items():
            p, q = c.get("ply2", {}), c.get("portfolio", {})
            top = (f"{p['diag_first_top_changed_share']} / {p['diag_final_top_changed_share']}" if "diag_first_top_changed_share" in p else "")
            md.append(f"| {arm} | {p.get('mode', '')} | {p.get('predictions_per_search', '')} | {p.get('prediction_ms_per_search', '')} | "
                      f"{p.get('view_builds_per_search', '')} | {p.get('view_hit_share', '')} | {p.get('memo_hit_share', '')} | "
                      f"{p.get('diag_move_diff_share', '')} | {top} | {q.get('assignments_per_call', '')} | {q.get('searched_per_turn', '')} | "
                      f"{q.get('refined_per_turn', '')} | {q.get('same_move_sets', '')} | {q.get('cuts', '')} |")
    valued = {arm: c["value"] for arm, c in calls.items() if c.get("value")}
    if valued:
        md += ["", "## Value term (V arms, our seat)", "",
               "Changed: decisions where V picked differently from the same path without V. Zero calls on a path mean the "
               "path was not exercised, not that V has no effect there.", "",
               "| Arm | Path | Coefficient | V calls per turn | Exact terminal | Decisions | Changed | Changed share | V ms per turn (all paths) |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, v in valued.items():
            for path in ("refine", "beam", "rollout"):
                if path in v:
                    p = v[path]
                    md.append(f"| {arm} | {path} | {p['coefficient']} | {p['calls_per_turn']} | {p['exact']} | {p['decisions']} | "
                              f"{p['changed']} | {p['changed_share']} | {v['value_seconds_per_turn_ms']} |")
    md += ["", "Max reply is the largest single reply over every game; p99 is per game (not pooled). "
           "A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary."]
    (out / "SCREEN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    sys.exit(2 if incomplete else 0)


if __name__ == "__main__":
    main()
