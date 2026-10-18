"""Host audit of the T0081 screen from its raw records (task T0084).

    python tools/screens/p3_search_host/audit_host.py [--data DIR] [--receipt DIR] [--shards-index PATH ...] [--out-dir DIR]

Inputs, all read-only:
  --data          the screen's work directory (default: work_dir of the receipt's RUN.json): results.jsonl and
                  matches/n<teams>/<fixture_id>-<sha256(arm)[:10]>.json.gz, screen_runner's naming. The A/A copy
                  of an arm runs the same binary, so match records are told apart by this arm key, never by hash.
  --receipt       the committed screen receipt (default reports/host/agent_v2_p3_search/screen): RUN.json,
                  screen.json, schedule.json.
  --shards-index  a P training corpus index (data/shards/index.json layout); repeat it for each corpus.

Writes EVIDENCE_AUDIT.json and renders EVIDENCE_AUDIT.md from that JSON alone (render()). Every figure is computed
from the inputs above. The rows must cover the receipt's schedule and arms exactly, or the audit refuses; a later
section whose inputs are missing or incomplete is written as unavailable with its reason, never as zero. What these
records cannot settle is listed under not_audited.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "ctf-agent-v2-p3-host-audit-2"
RECEIPT = REPO / "reports" / "host" / "agent_v2_p3_search" / "screen"
OUT = REPO / "reports" / "host" / "agent_v2_p3_search"
CANDIDATE = "@candidate"
TOLERANCE = 6e-5          # the receipt rounds to 5 decimals
NOT_AUDITED = {
    "fixed_work_receipts": "whether the committed fixed-work receipts ran the native harness: no receipt input here",
    "job_interleaving": "the order games ran in and the reservation chronology: the rows carry no wall-clock times",
    "admission_vs_rng_per_turn": "the final counters show whether two runs admitted search on a different number of "
                                 "turns, not on which turns; separating admission from RNG needs per-turn admission logs",
}


def arm_key(arm: str) -> str:
    return hashlib.sha256(arm.encode()).hexdigest()[:10]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---- rows -------------------------------------------------------------------------------------------------------

def load_rows(results: Path, schedule: dict, screen: dict) -> dict[str, dict[str, dict]]:
    """{fixture_id: {arm: row}}, refused unless every scheduled fixture has exactly one current row per arm."""
    fixtures = {f["fixture_id"] for f in schedule["fixtures"]}
    arms, idents = set(screen["arms"]), screen["identities"]
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    problems = []
    for n, line in enumerate(results.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        fx, arm = r.get("fixture_id"), r.get("arm")
        if fx not in fixtures or arm not in arms:
            problems.append(f"line {n}: {fx} / {arm} is not in the receipt's schedule and arms")
        elif r.get("identity") != idents[arm]:
            problems.append(f"line {n}: {fx} / {arm} identity differs from the receipt's")
        elif arm in out[fx]:
            problems.append(f"line {n}: duplicate row {fx} / {arm}")
        else:
            out[fx][arm] = r
    missing = [(fx, arm) for fx in sorted(fixtures) for arm in sorted(arms) if arm not in out.get(fx, {})]
    if missing:
        problems.append(f"{len(missing)} scheduled games have no row, first {missing[:3]}")
    if problems:
        raise SystemExit("results do not cover the receipt exactly:\n  " + "\n  ".join(problems[:20]))
    return dict(out)


# ---- paired comparisons -----------------------------------------------------------------------------------------

def band_counts(name: str) -> list[int]:
    lo, _, hi = name.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def equal_count(diffs: dict[int, list[float]], counts: list[int], z: float | None) -> dict:
    """Mean of per-count means; se = sqrt(sum_c var_c / n_c) / C, var with n - 1."""
    if any(len(diffs.get(c, [])) < 2 for c in counts):
        raise ValueError(f"fewer than two fixtures at some count of {counts}")
    mean = statistics.fmean(statistics.fmean(diffs[c]) for c in counts)
    se = math.sqrt(sum(statistics.variance(diffs[c]) / len(diffs[c]) for c in counts)) / len(counts)
    out = {"mean": mean, "se": se, "fixtures": sum(len(diffs[c]) for c in counts),
           "ci95": [mean - 1.96 * se, mean + 1.96 * se]}
    if z is not None:
        out["z"], out["ci"] = z, [mean - z * se, mean + z * se]
    lo, hi = out["ci"] if z is not None else out["ci95"]
    out["verdict"] = "helps" if lo > 0 else "hurts" if hi < 0 else "not shown"
    return out


def compare(rows: dict, teams: dict[str, int], a: str, b: str, bands: list[str], z: float | None) -> dict:
    diffs: dict[int, list[float]] = defaultdict(list)
    for fx, arms in rows.items():
        diffs[teams[fx]].append(arms[a]["utility"] - arms[b]["utility"])
    counts = sorted(diffs)
    return {"overall": equal_count(diffs, counts, z),
            "bands": {name: equal_count(diffs, band_counts(name), z) for name in bands},
            "per_count": {str(c): statistics.fmean(diffs[c]) for c in counts}}


def against_receipt(mine: dict, theirs: dict) -> dict:
    """Largest difference between a recomputed comparison and the receipt's, over every number both carry."""
    diffs = []
    for part, got in [("overall", mine["overall"])] + [(f"band {k}", v) for k, v in mine["bands"].items()]:
        ref = theirs["overall"] if part == "overall" else theirs["bands"].get(part[5:])
        if ref is None:
            diffs.append((part, "missing in receipt", math.inf))
            continue
        for key in ("mean", "se", "ci95", "ci"):
            if key in ref and key in got:
                pairs = zip(got[key], ref[key]) if isinstance(got[key], list) else [(got[key], ref[key])]
                diffs += [(part, key, abs(x - y)) for x, y in pairs]
        if ref.get("fixtures") != got["fixtures"] or ref.get("verdict") != got["verdict"]:
            diffs.append((part, "fixtures/verdict", math.inf))
    worst = max(d[2] for d in diffs)
    return {"reproduces": worst <= TOLERANCE, "max_abs_diff": worst if math.isfinite(worst) else None,
            "mismatches": sorted({f"{p} {k}" for p, k, d in diffs if d > TOLERANCE})}


def comparisons(rows: dict, teams: dict[str, int], screen: dict) -> dict:
    out = {}
    for family, z in (("comparisons", screen["multiplicity"]["z"]), ("rerun_noise", None)):
        for name, ref in screen[family].items():
            a, b = name.split(" - ")
            got = compare(rows, teams, a, b, list(ref["bands"]), z)
            out[name] = got | {"family": family, "receipt": against_receipt(got, ref)}
    return out


# ---- match records ----------------------------------------------------------------------------------------------

def match_name(fx: str, teams: int, arm: str) -> str:
    return f"n{teams:02d}/{fx}-{arm_key(arm)}.json.gz"


def read_matches(rows: dict, teams: dict[str, int], matches: Path) -> dict | None:
    """{(fixture, arm): match or None}; None as a whole when the directory is absent."""
    if not matches.is_dir():
        return None
    out = {}
    for fx, arms in rows.items():
        for arm in arms:
            p = matches / match_name(fx, teams[fx], arm)
            out[(fx, arm)] = json.loads(gzip.decompress(p.read_bytes())) if p.is_file() else None
    return out


def counters(match: dict | None) -> dict[str, int] | None:
    """The candidate's last cumulative AGENTV2 counter line (bots/agent_v2/bot.hpp), integer fields only."""
    if match is None:
        return None
    seat = next(i for i, s in enumerate(match["seats"]) if s["name"] == CANDIDATE)
    for line in reversed(match["bots"][seat].get("stderr_tail") or []):
        tok = line.split()
        if tok[:2] == ["AGENTV2", "turns"]:
            return {k: int(v) for k, v in zip(tok[1::2], tok[2::2]) if v.lstrip("-").isdigit()}
    return None


# ---- faults -----------------------------------------------------------------------------------------------------

FAULT_FIELDS = (("faults", "faults"), ("late", "late_replies"), ("opponent_faults", "opponent_faults"),
                ("opponent_late", "opponent_late"))          # (row field, screen.json arm field)


def faults(rows: dict, screen: dict, games: dict | None, turn_ms: int) -> dict:
    per_arm = {arm: {f: 0 for f, _ in FAULT_FIELDS} | {"games_with_any": 0, "by_count": Counter()} for arm in screen["arms"]}
    details = []
    for fx, arms in sorted(rows.items()):
        for arm, r in sorted(arms.items()):
            if not any(r.get(f) for f, _ in FAULT_FIELDS):
                continue
            s = per_arm[arm]
            for f, _ in FAULT_FIELDS:
                s[f] += r.get(f) or 0
            s["games_with_any"] += 1
            s["by_count"][str(r["teams"])] += 1
            d = {"fixture_id": fx, "arm": arm, "teams": r["teams"], **{f: r.get(f) or 0 for f, _ in FAULT_FIELDS},
                 "max_ms": r.get("max_ms"), "match_record": None}
            m = None if games is None else games[(fx, arm)]
            if m is not None:
                seats = m["seats"]
                d["match_record"] = match_name(fx, r["teams"], arm)
                d["events"] = [{"turn": t, "seat": i, "bot": seats[i]["name"], "reason": why,
                                "role": "candidate" if seats[i]["name"] == CANDIDATE else "opponent"}
                               for t, i, why in m.get("faults", [])]
                d["late_by_seat"] = {seats[i]["name"]: b["late"] for i, b in enumerate(m["bots"]) if b.get("late")}
            details.append(d)
    for arm, s in per_arm.items():
        s["by_count"] = dict(sorted(s["by_count"].items(), key=lambda kv: int(kv[0])))
        s["receipt"] = {k: screen["arms"][arm][k] for _, k in FAULT_FIELDS}
        s["matches_receipt"] = all(s[f] == s["receipt"][k] for f, k in FAULT_FIELDS)
    linked = [d for d in details if d["match_record"]]
    events = [e for d in linked for e in d["events"]]
    return {"turn_ms": turn_ms, "per_arm": per_arm, "details": details,
            "match_records": "unavailable: no matches directory" if games is None else
                             f"{len(linked)} of {len(details)} games with a fault or late reply have their record",
            "event_reasons": dict(Counter(e["reason"] for e in events)),
            "event_roles": dict(Counter(e["role"] for e in events)),
            "candidate_fault_games": sum(1 for d in details if d["faults"]),
            "candidate_fault_games_max_ms_below_deadline": sum(
                1 for d in details if d["faults"] and d["max_ms"] is not None and d["max_ms"] < turn_ms)}


# ---- search telemetry -------------------------------------------------------------------------------------------

def ratio(num: int | None, den: int | None) -> float | None:
    return num / den if num is not None and den else None


def telemetry(rows: dict, teams: dict[str, int], screen: dict, games: dict | None) -> dict:
    if games is None:
        return {"status": "unavailable", "reason": "no matches directory"}
    out = {}
    for arm in screen["arms"]:
        by = defaultdict(lambda: {"games": 0, "with_counters": 0, "turns": 0, "search_used": 0, "search_fallback": 0,
                                  "overrides": None, "gate_kept": None, "max_search_us": []})
        for fx in rows:
            d = by[teams[fx]]
            d["games"] += 1
            c = counters(games[(fx, arm)])
            if c is None:
                continue
            d["with_counters"] += 1
            for k in ("turns", "search_used", "search_fallback"):
                d[k] += c[k]
            for k in ("overrides", "gate_kept"):
                if k in c:
                    d[k] = (d[k] or 0) + c[k]
            if "max_search_us" in c:
                d["max_search_us"].append(c["max_search_us"])
        out[arm] = {}
        for cnt in sorted(by):
            d = by[cnt]
            ok = d["with_counters"] == d["games"]          # a partial count is not reported as a rate
            out[arm][str(cnt)] = {
                "games": d["games"], "with_counters": d["with_counters"], "complete": ok, "turns": d["turns"],
                "search_used": d["search_used"], "search_fallback": d["search_fallback"],
                "overrides": d["overrides"], "gate_kept": d["gate_kept"],
                "completion": ratio(d["search_used"], d["search_used"] + d["search_fallback"]) if ok else None,
                "override_rate": ratio(d["overrides"], d["search_used"]) if ok else None,
                "max_search_us_median": statistics.median(d["max_search_us"]) if ok and d["max_search_us"] else None}
    return {"status": "computed", "per_arm": out}


# ---- A/A --------------------------------------------------------------------------------------------------------

def aa_rerun(rows: dict, teams: dict[str, int], screen: dict, games: dict | None) -> dict:
    """Score-vector identity per count and, per pair, whether the two runs' search admission totals differ."""
    out = {}
    for name in screen["rerun_noise"]:
        copy, orig = name.split(" - ")
        by: dict[int, Counter] = defaultdict(Counter)
        for fx, arms in rows.items():
            same = "identical" if arms[copy]["score"] == arms[orig]["score"] else "differing"
            by[teams[fx]][same] += 1
            if games is None:
                continue
            ca, cb = counters(games[(fx, copy)]), counters(games[(fx, orig)])
            if ca is None or cb is None:
                by[teams[fx]]["counters_unavailable"] += 1
            else:
                key = ("search_used", "search_fallback")
                equal = all(ca[k] == cb[k] for k in key)
                by[teams[fx]][f"{same}_counters_{'equal' if equal else 'differ'}"] += 1
        out[name] = {"counters": "unavailable: no matches directory" if games is None else "computed",
                     "total": dict(sorted(sum(by.values(), Counter()).items())),
                     "per_count": {str(c): {"games": by[c]["identical"] + by[c]["differing"], **dict(sorted(by[c].items()))}
                                   for c in sorted(by)}}
    return out


# ---- corpus overlap ---------------------------------------------------------------------------------------------

def corpus_overlap(schedule: dict, indexes: list[Path]) -> dict:
    maps = {f["map_seed"] for f in schedule["fixtures"]}
    seats = Counter(o for f in schedule["fixtures"] for o in f["opponents"])
    out = {"screen_distinct_map_seeds": len(maps), "screen_opponent_seats": dict(sorted(seats.items())), "corpora": []}
    for path in indexes:
        if not path.is_file():
            out["corpora"].append({"index": str(path), "status": "unavailable", "reason": "not found"})
            continue
        idx = load_json(path)
        corpus_maps = {m["map_seed"] for m in idx.get("matches", [])}
        by_bot = idx.get("manifest", {}).get("seats_by_bot")
        out["corpora"].append({
            "index": str(path), "status": "computed", "matches": len(idx.get("matches", [])),
            "distinct_map_seeds": len(corpus_maps), "map_seed_overlap": len(corpus_maps & maps),
            "seats_by_opponent": None if by_bot is None else {b: by_bot.get(b, 0) for b in sorted(seats)}})
    return out


# ---- report -----------------------------------------------------------------------------------------------------

def audit(data: Path, receipt: Path, indexes: list[Path]) -> dict:
    run, screen, schedule = (load_json(receipt / f) for f in ("RUN.json", "screen.json", "schedule.json"))
    rows = load_rows(data / "results.jsonl", schedule, screen)
    teams = {f["fixture_id"]: f["teams"] for f in schedule["fixtures"]}
    games = read_matches(rows, teams, data / "matches")
    return {"schema": SCHEMA, "seed": screen["seed"], "schedule": screen["schedule"], "fixtures": len(rows),
            "games": sum(len(a) for a in rows.values()), "arms": sorted(screen["arms"]),
            "match_records": {"status": "unavailable", "reason": "no matches directory"} if games is None else
                             {"status": "computed", "expected": len(games),
                              "missing": sum(1 for m in games.values() if m is None)},
            "comparisons": comparisons(rows, teams, screen),
            "faults": faults(rows, screen, games, run["config"]["turn_ms"]),
            "search_telemetry": telemetry(rows, teams, screen, games),
            "aa_rerun": aa_rerun(rows, teams, screen, games),
            "corpus_overlap": corpus_overlap(schedule, indexes),
            "not_audited": NOT_AUDITED}


def num(x, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render(doc: dict) -> str:
    """EVIDENCE_AUDIT.md from EVIDENCE_AUDIT.json; every figure comes from doc."""
    L = ["# T0081 host evidence audit", "",
         f"Task T0084. Generated by `tools/screens/p3_search_host/audit_host.py` from the raw rows and match records "
         f"of seed {doc['seed']} (schedule `{doc['schedule'][:12]}`): {doc['fixtures']} fixtures, {doc['games']} games, "
         f"{len(doc['arms'])} arms. Do not edit by hand.", "",
         "Means give each team count equal weight. Comparisons use the receipt's Bonferroni z, the A/A row 95%. "
         "An interval that includes zero is \"not shown\", which is not evidence of equivalence.", "",
         "## Paired comparisons against the receipt", "",
         "| Comparison | Mean | SE | Interval | Verdict | Bands: mean, verdict | Reproduces screen.json |",
         "|---|---:|---:|---|---|---|---|"]
    for name, c in doc["comparisons"].items():
        o, rc = c["overall"], c["receipt"]
        lo, hi = o.get("ci", o["ci95"])
        bands = "; ".join(f"{k}: {b['mean']:+.4f} {b['verdict']}" for k, b in c["bands"].items())
        rep = "yes" if rc["reproduces"] else f"**no** ({', '.join(rc['mismatches'])})"
        L.append(f"| `{name}` | {o['mean']:+.4f} | {o['se']:.4f} | [{lo:+.4f}, {hi:+.4f}] | {o['verdict']} | {bands} | {rep} |")

    f = doc["faults"]
    L += ["", "## Faults and late replies", "", f"Counted from the rows; match records: {f['match_records']}.", "",
          "| Arm | Games with any | Faults | Late | Opponent faults | Opponent late | Games by team count | Matches screen.json |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    for arm, s in f["per_arm"].items():
        by = ", ".join(f"n{k}: {v}" for k, v in s["by_count"].items()) or "-"
        L.append(f"| `{arm}` | {s['games_with_any']} | {s['faults']} | {s['late']} | {s['opponent_faults']} | "
                 f"{s['opponent_late']} | {by} | {'yes' if s['matches_receipt'] else '**no**'} |")
    if f["event_reasons"]:
        L += ["", "Fault events in the linked records by reason: "
              + ", ".join(f"{k} {v}" for k, v in sorted(f["event_reasons"].items()))
              + "; by role: " + ", ".join(f"{k} {v}" for k, v in sorted(f["event_roles"].items())) + "."]
    L += ["", f"In {f['candidate_fault_games_max_ms_below_deadline']} of {f['candidate_fault_games']} games with a "
              f"candidate fault, the row's max_ms is below the {f['turn_ms']} ms deadline.", "",
          "| Fixture | Arm | Faults | Late | Opponent faults | Opponent late | max_ms | Fault events: turn, bot, reason |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for d in f["details"]:
        ev = "record missing" if d["match_record"] is None else \
            "; ".join(f"{e['turn']} {e['bot']} {e['reason']}" for e in d["events"]) or "none listed"
        L.append(f"| {d['fixture_id']} | `{d['arm']}` | {d['faults']} | {d['late']} | {d['opponent_faults']} | "
                 f"{d['opponent_late']} | {num(d['max_ms'], '.1f')} | {ev} |")

    L += ["", "## A/A rerun", ""]
    for name, a in doc["aa_rerun"].items():
        t = a["total"]
        L += [f"`{name}`: {t.get('identical', 0)} identical and {t.get('differing', 0)} differing score vectors. "
              f"Search admission totals per pair: {a['counters']}.", "",
              "| Teams | Games | Identical | Differing | Differing, totals differ | Differing, totals equal | "
              "Identical, totals differ | Totals unavailable |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c, d in a["per_count"].items():
            L.append(f"| {c} | {d['games']} | {d.get('identical', 0)} | {d.get('differing', 0)} | "
                     f"{d.get('differing_counters_differ', 0)} | {d.get('differing_counters_equal', 0)} | "
                     f"{d.get('identical_counters_differ', 0)} | {d.get('counters_unavailable', 0)} |")
        L += ["", "Totals are the candidate's final search_used and search_fallback. A differing pair whose totals also "
                  "differ admitted search on a different number of turns, consistent with wall-clock admission "
                  "variability. Equal totals do not show identical per-turn admission, so neither column separates "
                  "admission from RNG (see Not audited)."]

    tel = doc["search_telemetry"]
    L += ["", "## Search completion and overrides in the screen", ""]
    if tel["status"] != "computed":
        L.append(f"Unavailable: {tel['reason']}.")
    else:
        searched = [a for a, per in tel["per_arm"].items()
                    if any(d["search_used"] + d["search_fallback"] for d in per.values())]
        L += ["Completion = search_used / (search_used + search_fallback), override rate = overrides / search_used, "
              "both from the candidate's last counter line of each game. n/a: some game of that count has no "
              "counters, or there is no denominator.", ""]
        for a in searched:
            L += [f"`{a}`:", "", "| Teams | Completion | Override rate | Median max_search_us | Games with counters |",
                  "|---:|---:|---:|---:|---:|"]
            for c, d in tel["per_arm"][a].items():
                L.append(f"| {c} | {pct(d['completion'])} | {pct(d['override_rate'])} | "
                         f"{num(d['max_search_us_median'], '.0f')} | {d['with_counters']}/{d['games']} |")
            L.append("")

    o = doc["corpus_overlap"]
    L += ["", "## Overlap with P training corpora", "",
          f"The screen uses {o['screen_distinct_map_seeds']} distinct map seeds; opponent seats: "
          + ", ".join(f"{b} {n}" for b, n in o["screen_opponent_seats"].items()) + ".", ""]
    if not o["corpora"]:
        L.append("No corpus index was given: overlap not audited.")
    for c in o["corpora"]:
        if c["status"] != "computed":
            L.append(f"- `{c['index']}`: unavailable ({c['reason']}).")
            continue
        seats = "seats per opponent unavailable" if c["seats_by_opponent"] is None else \
            "seats per opponent: " + ", ".join(f"{b} {n}" for b, n in c["seats_by_opponent"].items())
        L.append(f"- `{c['index']}`: {c['matches']} matches, {c['distinct_map_seeds']} distinct map seeds, "
                 f"{c['map_seed_overlap']} shared with the screen; {seats}.")
    L += ["", "## Not audited", ""] + [f"- {k}: {v}." for k, v in doc["not_audited"].items()]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipt", type=Path, default=RECEIPT)
    ap.add_argument("--data", type=Path, help="default: the receipt's RUN.json work_dir")
    ap.add_argument("--shards-index", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args(argv)
    data = a.data or REPO / load_json(a.receipt / "RUN.json")["config"]["work_dir"]
    if not (data / "results.jsonl").is_file():
        sys.stderr.write(f"results not found: {data / 'results.jsonl'} (host-only input); nothing written\n")
        return 1
    doc = audit(data, a.receipt, a.shards_index)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "EVIDENCE_AUDIT.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (a.out_dir / "EVIDENCE_AUDIT.md").write_text(render(doc), encoding="utf-8")
    print(f"[audit] wrote {a.out_dir / 'EVIDENCE_AUDIT.json'} and EVIDENCE_AUDIT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
