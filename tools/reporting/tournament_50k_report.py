"""The 50,000-game tournament report (task T0028), generated from its committed sources; no number is typed by hand.

    python tools/reporting/tournament_50k_report.py --replays DIR   # DIR holds the published replays and their viewer
                                                                    # metadata (below): writes the highlights JSON
    python tools/reporting/tournament_50k_report.py                 # writes reports/eval/tournament_50k_report.md
    python tools/reporting/tournament_50k_report.py --check         # exit 1 when the committed report is stale

Sources: reports/eval/tournament_50k_summary.json (the run's per-size aggregates and its Plackett-Luce ratings) and
reports/eval/tournament_50k_highlights.json (participants, scores and timing read from the three published replays,
each bound to its SHA-256). The replays are at https://casper015.github.io/ctf-replay-viewer/data/matches/<id>.json;
their viewer metadata (game id, title) is content/matches/<id>.json in Casper015/ctf-replay-viewer, saved in DIR as
<id>.meta.json.

What this script cannot do: refit the ratings. The run's fitter, its ordered match results and its run manifest are
not committed, so the ratings are reported as the run gave them. The comparisons below use the per-size win rates
only, with an approximate interval (each bot's games as independent Bernoulli trials; see the report's caveats).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "reports" / "eval" / "tournament_50k_summary.json"
HIGHLIGHTS = REPO / "reports" / "eval" / "tournament_50k_highlights.json"
REPORT = REPO / "reports" / "eval" / "tournament_50k_report.md"
REPLAY_URL = "https://casper015.github.io/ctf-replay-viewer/data/matches/{}.json"
VIEWER_URL = "https://casper015.github.io/ctf-replay-viewer/match.html?m={}"
HIGHLIGHT_IDS = ["g4_ponly_pxl_comeback", "g2_ponly_duel", "g8_ponly_pxxl_chaos"]
COMPARISONS = [  # (label, a, b): a minus b
    ("C08 P-only XL - C08 control", "c08_ponly_pxl", "c08_control"),
    ("C08 P-only XXL - C08 control", "c08_ponly_pxxl", "c08_control"),
    ("C08 P-only XL - XXL", "c08_ponly_pxl", "c08_ponly_pxxl"),
    ("C08 P-only XL - hybrid M", "c08_ponly_pxl", "c08_hybrid_m"),
    ("v12 P-only XL - v12 control", "v12_ponly_pxl", "v12_control"),
    ("v12 P-only XXL - v12 control", "v12_ponly_pxxl", "v12_control"),
    ("v12 P-only XL - XXL", "v12_ponly_pxl", "v12_ponly_pxxl"),
]
Z = 1.96


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_highlights(replay_dir: Path) -> list[dict]:
    """Participants, final scores, winners and per-bot timing of each published replay, bound to its file hash."""
    out = []
    for mid in HIGHLIGHT_IDS:
        path = replay_dir / f"{mid}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads((replay_dir / f"{mid}.meta.json").read_text(encoding="utf-8"))
        res, head = data["result"], data["header"]
        names, scores = list(res["names"]), list(res["score"])
        top = max(scores)
        out.append({
            "id": mid, "game_id": meta["id"], "viewer_title": meta["title"]["en"], "viewer_desc": meta["desc"]["en"],
            "url": REPLAY_URL.format(mid), "sha256": sha256(path), "bytes": path.stat().st_size,
            "teams": head["teams"], "map_size": head["size"], "seed": res["seed"], "flag_seed": res["flag_seed"],
            "turns": res["turns"], "names": names, "scores": scores,
            "winners": [n for n, s in zip(names, scores) if s == top],
            "bots": [{k: b.get(k) for k in ("name", "dead", "calls", "first_ms", "p99_ms", "max_ms", "timeouts_50ms",
                                             "dropped_official", "hard_limit_hits")} for b in res["bots"]],
        })
    return out


def title_check(title: str, scores: list[int]) -> str:
    """Whether the score a viewer title (or description) states fits the replay: 'A-B' must be the top two scores, 'N points' the unique
    top score. Anything else is not checked."""
    top = sorted(scores, reverse=True)
    m = re.search(r"(\d+)-(\d+)", title)
    if m:
        return "match the replay" if [int(m.group(1)), int(m.group(2))] == top[:2] else "do not match the replay"
    m = re.search(r"(\d+) points", title)
    if m:
        unique_top = len(top) < 2 or top[0] > top[1]
        return "match the replay" if int(m.group(1)) == top[0] and unique_top else "do not match the replay"
    return "state no score, not checked"


def overall(per_size: dict) -> dict[str, dict]:
    """Games summed over sizes; win rate, pairwise win rate and score weighted by each size's games."""
    tot: dict[str, dict] = {}
    for rows in per_size.values():
        for bot, r in rows.items():
            t = tot.setdefault(bot, {"games": 0, "wins": 0.0, "pair": 0.0, "score": 0.0})
            t["games"] += r["games"]
            t["wins"] += r["games"] * r["win_rate"]
            t["pair"] += r["games"] * r["pair_win_rate"]
            t["score"] += r["games"] * r["avg_score"]
    return {b: {"games": t["games"], "win_rate": t["wins"] / t["games"], "pair_win_rate": t["pair"] / t["games"],
                "avg_score": t["score"] / t["games"]} for b, t in tot.items()}


def diff(per_size: dict, size: str, a: str, b: str) -> tuple[float, float]:
    """Win-rate difference a - b at one size and its approximate standard error (independent binomials)."""
    ra, rb = per_size[size][a], per_size[size][b]
    pa, pb = ra["win_rate"], rb["win_rate"]
    return pa - pb, math.sqrt(pa * (1 - pa) / ra["games"] + pb * (1 - pb) / rb["games"])


def verdict(d: float, se: float) -> str:
    lo, hi = d - Z * se, d + Z * se
    return "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0"


def later_late(bot: dict) -> int:
    """Replies after the first one that took over 50 ms. The replays' `timeouts_50ms` also counts the first reply,
    which may take 2 s (see first_counted)."""
    return (bot["timeouts_50ms"] or 0) - (1 if (bot["first_ms"] or 0) > 50 else 0)


def first_counted(highlights: list[dict]) -> bool:
    """True when `timeouts_50ms` evidently counts the first reply: every bot whose later replies all stayed under
    50 ms (max_ms <= 50) has exactly one timeout if its first reply took over 50 ms, and none otherwise."""
    quiet = [b for h in highlights for b in h["bots"] if (b["max_ms"] or 0) <= 50]
    return bool(quiet) and all((b["timeouts_50ms"] or 0) == (1 if (b["first_ms"] or 0) > 50 else 0) for b in quiet)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def signed_pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def render(summary: dict, highlights: list[dict]) -> str:
    per_size = summary["per_size_summary"]
    sizes = sorted(per_size, key=int)
    ratings = summary["ratings_plackett_luce"]
    tot = overall(per_size)
    bots = sorted(ratings, key=lambda b: -ratings[b])
    games_total = sum(r["games"] for rows in per_size.values() for r in rows.values())
    out = [
        "# 50,000-game tournament: Plackett-Luce ratings reported by the run", "",
        "Generated by `tools/reporting/tournament_50k_report.py` from "
        "[tournament_50k_summary.json](tournament_50k_summary.json) and "
        "[tournament_50k_highlights.json](tournament_50k_highlights.json); do not edit by hand. Regenerate with "
        "`python tools/reporting/tournament_50k_report.py` (task T0028).", "",
        f"**The run** (as its summary records): {summary['total_games']:,} games, status `{summary['status']}`, "
        f"{summary['workers']} concurrent workers on the host, {summary['total_time_seconds']:,.1f} s wall time, "
        f"{len(ratings)} bots, team counts {', '.join(sizes)}. The per-size tables hold {games_total:,} seat-games.", "",
        "**What these numbers are, and are not:**",
        "- **The ratings are the run's own Plackett-Luce estimates, anchored at `"
        f"{summary['anchor']}` = 0.** They are not the official pairwise Bradley-Terry scorer, and they cannot be "
        "reproduced from the repository: the fitter, its likelihood and tie handling, the rating scale, the count "
        "weights and the ordered match results are not committed. They carry no intervals, so the order of close "
        "ratings is unresolved.",
        "- **Win rate, pairwise win rate and average score** are the run's per-size fields. The run did not record "
        "how `win_rate` treats a shared first place.",
        "- **No run manifest is committed:** engine, runner, binary and model hashes, seeds, schedule and result "
        "digests, duplicate checks, and fault, late-reply and timing coverage are missing. See "
        "[What is missing](#what-is-missing).",
        "- **P predicts what the opponents do,** inside the C08 or v12 heuristic search. It does not choose our own "
        "moves.", "",
        "## Ratings", "",
        "Games, win rate, pairwise win rate and score are summed over the ten team counts, each count weighted by the "
        "bot's games there.", "",
        "| # | Bot | PL rating (reported) | Games | Win rate | Pairwise win rate | Avg score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, b in enumerate(bots, 1):
        t = tot[b]
        out.append(f"| {i} | `{b}` | {ratings[b]:+.1f} | {t['games']:,} | {pct(t['win_rate'])} | "
                   f"{pct(t['pair_win_rate'])} | {t['avg_score']:.2f} |")
    gaps = [(a, b, ratings[a] - ratings[b]) for a, b in (("c08_ponly_pxl", "c08_ponly_pxxl"),
                                                         ("v12_ponly_pxl", "v12_ponly_pxxl"))]
    out += ["", "Reported rating gaps between P-only XL and XXL: " + "; ".join(
        f"`{a}` - `{b}` = {g:+.1f}" for a, b, g in gaps) + ". Without intervals they do not order the two sizes.", ""]

    out += ["## Comparisons by team count", "",
            "Win-rate differences in percentage points, with an approximate 95% interval. The interval treats each "
            "bot's games at one count as independent Bernoulli trials. It ignores the different opponent mix of each "
            "game and the games two bots share (only one of them can win such a game, so the true interval is "
            "somewhat wider). The last column is the equal-count mean over the ten counts, with the same approximation; "
            "**above 0** or **below 0** means its interval excludes zero, **includes 0** that the difference is not "
            "shown.", "",
            "| Comparison | " + " | ".join(f"{s}T" for s in sizes) + " | Equal-count mean |",
            "|---|" + "---:|" * len(sizes) + "---|"]
    for label, a, b in COMPARISONS:
        cells, ds, vs = [], [], []
        for s in sizes:
            d, se = diff(per_size, s, a, b)
            ds.append(d)
            vs.append(se * se)
            cells.append(signed_pp(d) + ("*" if verdict(d, se) != "includes 0" else ""))
        m, mse = sum(ds) / len(ds), math.sqrt(sum(vs)) / len(vs)
        out.append(f"| {label} | " + " | ".join(cells) +
                   f" | {signed_pp(m)} [{signed_pp(m - Z * mse)}, {signed_pp(m + Z * mse)}] {verdict(m, mse)} |")
    out += ["", "\\* The interval at that count excludes zero.", ""]

    for label, a, b in (("C08", "c08_ponly_pxl", "c08_ponly_pxxl"), ("v12", "v12_ponly_pxl", "v12_ponly_pxxl")):
        xxl_ahead = [s for s in sizes if per_size[s][b]["win_rate"] > per_size[s][a]["win_rate"]]
        out.append(f"- **{label} XL against XXL.** XXL has the higher point win rate at "
                   f"{', '.join(s + 'T' for s in xxl_ahead) or 'no count'}; XL at the others.")
    out += ["- **Why the sizes would differ is a hypothesis.** A larger P could predict combat better; a smaller one "
            "leaves more search time per turn. This run measured neither prediction quality nor search samples, so "
            "neither explanation is shown.", ""]

    for title, key, fmt in (("Win rate by team count", "win_rate", pct),
                            ("Average score by team count", "avg_score", lambda x: f"{x:.2f}"),
                            ("Games by team count", "games", lambda x: f"{x:,}")):
        out += [f"## {title}", "", "| Bot | " + " | ".join(f"{s}T" for s in sizes) + " |",
                "|---|" + "---:|" * len(sizes)]
        for b in bots:
            out.append(f"| `{b}` | " + " | ".join(fmt(per_size[s][b][key]) if b in per_size[s] else "-"
                                                  for s in sizes) + " |")
        out.append("")

    out += ["## Replay highlights", "",
            "Read from the published replay files; each is bound to its SHA-256. The game id is the one the replay "
            "viewer's metadata gives; the summary has no per-game rows to check it against. The score each viewer title or "
            "description states is checked against the replay. They are three games out of "
            "50,000: illustrations, not evidence.", ""]
    for h in highlights:
        score = ", ".join(f"`{n}` {s}" for n, s in zip(h["names"], h["scores"]))
        win = " and ".join(f"`{w}`" for w in h["winners"])
        out += [f"**[{h['id']}]({VIEWER_URL.format(h['id'])})**: {h['teams']} teams, {h['turns']} turns, map seed "
                f"{h['seed']}, flag seed {h['flag_seed']}, game id `{h['game_id']}`. [Replay file]({h['url']}), "
                f"sha256 `{h['sha256'][:16]}`.", "",
                f"- Final score: {score}.",
                f"- {'Winners (tied)' if len(h['winners']) > 1 else 'Winner'}: {win}.",
                f"- The viewer's title and description (\"{h['viewer_title']}\" / \"{h['viewer_desc']}\") "
                f"{title_check(h['viewer_title'] + ' ' + h['viewer_desc'], h['scores'])}."]
        late = [b for b in h["bots"] if later_late(b) > 0]
        out.append("- Later replies over 50 ms: " + (", ".join(
            f"`{b['name']}` {later_late(b)} of {b['calls'] - 1} (p99 {b['p99_ms']} ms, max {b['max_ms']} ms)"
            for b in late) if late else "none") +
            f". Replies dropped by the referee: {sum(b['dropped_official'] or 0 for b in h['bots'])}.")
        out.append("")

    over = [b for h in highlights for b in h["bots"] if later_late(b) > 0]
    dropped = sum(b["dropped_official"] or 0 for h in highlights for b in h["bots"])
    bot_games = sum(len(h["bots"]) for h in highlights)
    late_list = ", ".join(f"`{b['name']}` {later_late(b)}" for b in over)
    out += ["## Time limits", "",
            "The summary records no faults, late replies or reply times; only the three replays do. "
            + ("Their `timeouts_50ms` field also counts the first reply, which may take up to 2 s: every bot whose "
               "later replies all stayed under 50 ms has exactly one such timeout when its first reply took over "
               "50 ms, and none otherwise. The counts here leave the first reply out. "
               if first_counted(highlights) else "")
            + f"In {len(over)} of their {bot_games} bot-games, later replies took over 50 ms "
            f"({late_list}), "
            f"and the referee dropped {dropped} replies. Late replies were therefore recorded and still played, at "
            "least in these games: the official rule that drops a late reply was not applied, so timeouts cost no "
            "games, and compliance with the 2 s / 50 ms limits cannot be inferred for the run. The run used "
            f"{summary['workers']} workers, a heavier CPU load than the 6 of the P-only screens.", "",
            "## What is missing", "",
            "This report is final for the run (T0028): it uses all the data that was kept. The following were never "
            "committed, so the results above stay descriptive and are not confirmation evidence. A future tournament "
            "should commit them with its summary:",
            "- the ratings fitter and its exact command: likelihood, tie handling, scale, anchor, count weights;",
            "- the ordered results of every game, with a digest, so the ratings can be refitted and given intervals "
            "clustered by game;",
            "- a run manifest: engine, runner, every bot binary and model header with hashes, seeds, schedule digest, "
            "independent and duplicate game counts;",
            "- per-game faults, dead bots, late replies and reply times (first reply, p50, p99, max), with how many "
            "games each field covers; faulted games stay in the denominators;",
            "- per-game rows, so that game ids such as the highlights' can be checked.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replays", type=Path, help="directory with the published replay JSONs: rewrite the highlights")
    ap.add_argument("--check", action="store_true", help="exit 1 when the committed report differs from a fresh render")
    args = ap.parse_args(argv)
    if args.replays:
        HIGHLIGHTS.write_text(json.dumps(build_highlights(args.replays), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {HIGHLIGHTS.relative_to(REPO)}")
    text = render(json.loads(SUMMARY.read_text(encoding="utf-8")), json.loads(HIGHLIGHTS.read_text(encoding="utf-8")))
    if args.check:
        if REPORT.read_text(encoding="utf-8") != text:
            print(f"{REPORT.relative_to(REPO)} is stale: python tools/reporting/tournament_50k_report.py", file=sys.stderr)
            return 1
        return 0
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
