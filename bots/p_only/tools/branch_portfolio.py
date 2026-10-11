"""Candidate-branch continuations with complete-order alternatives (host only: real bots under the official limits).

    python bots/p_only/tools/branch_portfolio.py --matches data/p_only/v_pilot/matches --manifest build/p_only/manifest.json \\
        --arm v12_ponly_xxl --compare-arm v12_search_k3 --out data/branches/v12_portfolio --roots-per-match 2 \\
        --candidates 8 --repeats 2 --workers 6

The play itself is training.branch_collect's (task T0009), reused unchanged: the root is rebuilt from the
match's seeds and recorded orders (every block must match), every bot replays the recorded prefix, opponents keep
their recorded orders at the branch turn, then everything is live to the end, and the outcome is our seat's final
pairwise utility. This tool (task T0018, docs/SEARCH_PORTFOLIO.md) changes what is played:

Matches. Only matches of the declared carrier arm are used: a screen names each match <fixture>-<key>, key =
sha256(arm)[:10] (bots/p_only/tools/screen_runner.py), so other arms (and A/A copies) are skipped deliberately and
counted. A selected match whose carrier binary differs from the build manifest's stops the run with its id: a
rebuilt carrier is a different policy.

Candidates (bounded by --candidates, the recorded order first, every one a complete legal order that the engine
decodes; deduplicated by complete order; each with its source):
  recorded      the carrier's own reply
  arm:<name>    --compare-arm's reply at the same root: that variant replays the same recorded inputs (its history
                is the root's) and answers turn t under the official limit
  move1         one unit moves to another open neighbour, its action kept
  move2         two units change moves together (coordinated)
  target        an unladen unit attacks another enemy it could hit from its intended destination
  pickup_wait   a unladen unit on a flag cell: pickup and wait swapped
  drop          a carrier drops its flag (X) where it stands after its move
  transport     a carrier takes another open step (toward home or not)
Sources take turns, in this order after the recorded order, each in a hash order of (match, turn), so a small
budget still covers every kind. Candidate generation assumes enemies stay where they stand; it is a candidate
set, not a prediction, and no P output is used.

Writes one JSON per root (the branch_collect schema plus `sources`, `arm`, `compare_arm`), readable by
training.branch_dataset, and <out>/summary.json: matches seen, selected and skipped, roots played and dropped (by
reason), candidates by source, prefix mismatches (nondeterminism of the bots' replies to the recorded prefix),
faults, repeats, and the resolved and unresolved candidate pairs at the branch_dataset tie margin.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fusion import matchlog, runner  # noqa: E402
from fusion.labels import pairwise_outcome  # noqa: E402
from fusion.paths import ai_test_root, bot_path, load_engine, load_pool, sha256  # noqa: E402
from fusion.protocol import ATTACK, DELTAS, DROP, MOVES, PICKUP, WAIT, parse_block, parse_reply  # noqa: E402
from training.branch_collect import (PREFIX_LIMIT, designated_seat, play_branch, recorded_orders, reproduce,  # noqa: E402
                                     root_turns, unit_hash)

CANDIDATE = "@candidate"
SOURCES = ("move1", "target", "pickup_wait", "drop", "transport", "move2")


def arm_key(arm: str) -> str:
    return hashlib.sha256(arm.encode()).hexdigest()[:10]


def match_arm(match_id: str, arms: dict[str, str]) -> str | None:
    """The arm whose key ends this match id (screen_runner naming), or None."""
    return next((arm for arm, key in arms.items() if match_id.endswith("-" + key)), None)


def parse_flags(block_text: str) -> list[dict]:
    lines = block_text.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("FLAGS "))
    out = []
    for line in lines[i + 1:i + 1 + int(lines[i].split()[1])]:
        tok = line.split()
        out.append({"id": int(tok[1]), "status": tok[2], "x": int(tok[3]), "y": int(tok[4]), "return_at": int(tok[6])})
    return out


def act_text(req) -> str:
    return {ATTACK: str(req.target), PICKUP: "P", DROP: "X"}.get(req.act, "-")


def line_of(turn: int, reqs) -> str:
    return " ".join([str(turn)] + [f"{MOVES[r.move]} {act_text(r)}" for r in reqs])


def canonical(line: str | None, turn: int) -> str | None:
    """The canonical complete order of a reply (notes, spacing and action spelling removed), or None when the engine
    would reject it. Two replies are the same order exactly when their canonical forms are equal; engine no-ops
    (an attack on a unit out of range) are not merged, so such orders stay separate candidates."""
    reqs = parse_reply(line, turn) if line is not None else None
    return None if reqs is None else line_of(turn, reqs)


# A comparison arm's reply at a root: the only statuses with a usable outcome are `identical` (the incumbent's
# canonical order, so the incumbent's branches are its outcome) and `different` (a candidate of its own).
COMPARE_STATUSES = ("identical", "different", "malformed", "timeout", "missing_candidate", "missing_branch", "none")


def classify_compare(reply: str | None, turn: int, orders: list[str], played: set[int] | None = None,
                     has_compare: bool = True) -> tuple[str, str | None, int | None]:
    """(status, canonical order, candidate index) of a comparison reply against a root's canonical candidate
    orders (orders[0] is the incumbent's). `played`: candidate indices with at least one branch outcome."""
    if not has_compare:
        return "none", None, None
    if reply is None:
        return "timeout", None, None
    canon = canonical(reply, turn)
    if canon is None:
        return "malformed", None, None
    if not orders or canonical(orders[0], turn) != canon:
        index = next((i for i, o in enumerate(orders) if i > 0 and canonical(o, turn) == canon), None)
        if index is None:
            return "missing_candidate", canon, None
        if played is not None and index not in played:
            return "missing_branch", canon, index
        return "different", canon, index
    if played is not None and 0 not in played:
        return "missing_branch", canon, 0
    return "identical", canon, 0


def alternatives(game, block_text: str, line: str, seat: int, turn: int) -> dict[str, list[str]]:
    """Complete legal orders per source, derived from the recorded order (see the module docstring)."""
    reqs = parse_reply(line, turn)
    if reqs is None:
        return {}
    blk = parse_block(block_text)
    units = {u[0]: u for u in blk.units}
    size, walls = game.size, game.walls
    flags = parse_flags(block_text)
    flag_cells = {(f["x"], f["y"]) for f in flags if f["status"] == "home" or (f["status"] == "dropped" and turn < f["return_at"])}

    def open_cell(x, y):
        return 0 <= x < size and 0 <= y < size and (x, y) not in walls

    def dest(k, move):
        _, _, x, y, *_ = units[3 * seat + k]
        return x + DELTAS[move][0], y + DELTAS[move][1]

    def with_(changes: dict) -> list:
        out = [copy.copy(r) for r in reqs]
        for k, (move, act, target) in changes.items():
            out[k].move, out[k].act, out[k].target = move, act, target
        return out

    active = [k for k in range(3) if units[3 * seat + k][2] >= 0]
    carrying = {k: units[3 * seat + k][5] >= 0 for k in active}
    moves = {k: [m for m in range(5) if m != reqs[k].move and open_cell(*dest(k, m))] for k in active}
    pools: dict[str, list] = {s: [] for s in SOURCES}
    for k in active:
        r = reqs[k]
        for m in moves[k]:
            pools["transport" if carrying[k] else "move1"].append(with_({k: (m, r.act, r.target)}))
        if carrying[k]:
            pools["drop"].append(with_({k: (r.move, DROP, -1)}))
            continue
        d = dest(k, r.move)
        if not open_cell(*d):  # a move into a wall stays where it is
            d = tuple(units[3 * seat + k][2:4])
        for e in blk.units:
            if e[1] != seat and e[2] >= 0 and e[0] != r.target and game.can_attack(d, (e[2], e[3])):
                pools["target"].append(with_({k: (r.move, ATTACK, e[0])}))
        if d in flag_cells:
            pools["pickup_wait"].append(with_({k: (r.move, WAIT if r.act == PICKUP else PICKUP, -1)}))
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            for ma in moves[a]:
                for mb in moves[b]:
                    if dest(a, ma) != dest(b, mb):
                        pools["move2"].append(with_({a: (ma, reqs[a].act, reqs[a].target), b: (mb, reqs[b].act, reqs[b].target)}))
    return {s: [line_of(turn, x) for x in pool] for s, pool in pools.items()}


def candidates(game, block_text: str, line: str, seat: int, turn: int, budget: int, key: str,
               extra: list[tuple[str, str]] = ()) -> list[tuple[str, str]]:
    """(order, source) pairs: the recorded order, `extra` (another arm's reply), then the sources in turn."""
    recorded = parse_reply(line, turn)
    if recorded is None:
        return []
    out, seen = [], set()

    def add(order: str, source: str):
        norm = canonical(order, turn)
        if norm is None or norm in seen or len(out) >= budget:
            return
        seen.add(norm)
        out.append((norm, source))

    add(line_of(turn, recorded), "recorded")
    for order, source in extra:
        add(order, source)
    pools = alternatives(game, block_text, line, seat, turn)
    for s in pools:
        pools[s].sort(key=lambda x: unit_hash(key, s, x))
    while len(out) < budget and any(pools.values()):
        for s in SOURCES:
            if pools.get(s):
                add(pools[s].pop(0), s)
    return out


def compare_reply(engine, match: dict, seat: int, turn: int, command: list[str], limits) -> tuple[str | None, int]:
    """Another arm's reply at the root: it replays our seat's recorded inputs of turns 1..turn-1 (relaxed limits;
    its replies there are discarded) and answers turn `turn` under the official limit. (reply, prefix mismatches)."""
    game = engine.Game(match["teams"], match["map_seed"], flag_seed=match["flag_seed"])
    bot = runner.Bot(command, "compare", (limits[0], PREFIX_LIMIT))
    mismatch, reply = 0, None
    try:
        for t in range(1, turn + 1):
            if t == turn:
                bot.turn_limit, bot.strikes = limits[1], 0
            text = game.encode(seat)
            got = bot.receive(bot.send(text), t)
            if t < turn:
                mismatch += got != match["replies"][t - 1][seat]
                game.step(recorded_orders(game, match, t))
            else:
                reply = got
    finally:
        bot.close()
    return reply, mismatch


def _root_job(job) -> dict:
    path, turn, budget, repeats, commands, compare, limits, root, arm, compare_arm = job
    engine = load_engine(Path(root))
    match = matchlog.load(path)
    seat = designated_seat(match)
    game = reproduce(engine, match, turn)
    base = {"match_id": match["match_id"], "teams": match["teams"], "turn": turn, "arm": arm}
    if game is None:
        return base | {"dropped": "recorded blocks not reproduced"}
    snapshot = copy.deepcopy(game)
    line = match["replies"][turn - 1][seat]
    extra, compare_mismatch, reply = [], 0, None
    if compare is not None:
        reply, compare_mismatch = compare_reply(engine, match, seat, turn, compare, limits)
        if reply is not None:
            extra.append((reply, f"arm:{compare_arm}"))
    cands = candidates(game, match["blocks"][turn - 1], line, seat, turn, budget, f"{match['match_id']}/{turn}", extra)
    if len(cands) < 2:
        return base | {"dropped": "fewer than two legal candidates"}
    for order, _ in cands:  # every candidate is an order the engine accepts as written
        game.decode(seat, order)
    branches = []
    for c, (order, source) in enumerate(cands):
        for r in range(repeats):
            row = play_branch(engine, match, seat, turn, order, commands, limits)
            branches.append({"candidate": c, "repeat": r, "on_policy": c == 0, "source": source} | row)
    if snapshot.encode(0) != game.encode(0):
        raise RuntimeError("the reproduced root changed while branches were played")
    orders = [o for o, _ in cands]
    status, compare_order, compare_index = classify_compare(reply, turn, orders, played={b["candidate"] for b in branches},
                                                            has_compare=compare is not None)
    return base | {"schema": "ctf-fusion-branches-1", "map_seed": match["map_seed"], "seat": seat, "init": match["init"],
                   "carrier_sha256": match["seats"][seat]["sha256"], "candidates": [o for o, _ in cands],
                   "sources": [s for _, s in cands], "compare_arm": compare_arm, "compare_prefix_mismatch": compare_mismatch,
                   "compare_reply": None if reply is None else " ".join(reply.partition("#")[0].split()),
                   "compare_status": status, "compare_order": compare_order, "compare_candidate": compare_index,
                   "branches": branches, "root_outcome": float(pairwise_outcome(match["result"]["score"])[seat])}


def select(files, manifest: dict, arm: str, arms: list[str]):
    """(selected match paths, counts): matches of `arm` only; a changed carrier binary stops the run."""
    keys = {a: arm_key(a) for a in set(arms) | {arm}}
    selected, counts = [], Counter()
    for f in files:
        match = matchlog.load(f)
        counts["seen"] += 1
        owner = match_arm(match["match_id"], keys)
        if owner != arm:
            counts["skipped_other_arms" if owner else "skipped_unknown_arm"] += 1
            continue
        seat = designated_seat(match)
        if match["seats"][seat]["sha256"] != manifest[arm]["binary_sha256"]:
            raise SystemExit(f"{match['match_id']}: carrier {arm} was played by binary {match['seats'][seat]['sha256'][:12]}, "
                             f"the build manifest has {manifest[arm]['binary_sha256'][:12]}: the carrier was rebuilt")
        selected.append((f, match))
        counts["selected"] += 1
    return selected, counts


def seat_commands(match: dict, carrier_cmd: list[str], root: Path) -> list[list[str]]:
    pool = {e["name"]: e for e in load_pool()["bots"]}
    out = []
    for s in match["seats"]:
        if s["name"] == CANDIDATE:
            out.append(carrier_cmd)
            continue
        path = bot_path(pool[s["name"]], root)
        if sha256(path) != s["sha256"]:
            raise SystemExit(f"{match['match_id']}: bot {s['name']} changed since the match was played")
        out.append(runner.command_for(path, REPO / "build" / "roster"))
    return out


def summarize(rows: list[dict], counts: Counter, tie_steps: float = 1.0) -> dict:
    out = {"matches": dict(counts), "roots": 0, "dropped": Counter(), "candidates_by_source": Counter(), "branches": 0,
           "prefix_mismatch": {"sum": 0, "max": 0, "branches_with_any": 0}, "prefix_faults": 0, "live_faults": 0,
           "pairs": {"resolved": 0, "unresolved": 0}, "compare_differs": 0, "compare_status": Counter()}
    for row in rows:
        if "dropped" in row:
            out["dropped"][row["dropped"]] += 1
            continue
        out["roots"] += 1
        out["compare_differs"] += any(s.startswith("arm:") for s in row["sources"])
        out["compare_status"][row.get("compare_status", "unrecorded")] += 1
        for s in row["sources"]:
            out["candidates_by_source"][s] += 1
        means = {}
        for b in row["branches"]:
            out["branches"] += 1
            pm = out["prefix_mismatch"]
            pm["sum"] += b["prefix_mismatch"]; pm["max"] = max(pm["max"], b["prefix_mismatch"]); pm["branches_with_any"] += b["prefix_mismatch"] > 0
            out["prefix_faults"] += b["prefix_faults"]; out["live_faults"] += len(b["faults"])
            means.setdefault(b["candidate"], []).append(b["outcome"])
        margin = tie_steps * 0.5 / max(row["teams"] - 1, 1)
        ys = [sum(v) / len(v) for _, v in sorted(means.items())]
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                out["pairs"]["resolved" if abs(ys[i] - ys[j]) > margin else "unresolved"] += 1
    out["dropped"], out["candidates_by_source"] = dict(out["dropped"]), dict(out["candidates_by_source"])
    out["compare_status"] = dict(out["compare_status"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True, help="match records of a screen with keep_matches")
    ap.add_argument("--manifest", default=str(REPO / "build" / "p_only" / "manifest.json"))
    ap.add_argument("--arm", required=True, help="the carrier arm whose matches give the roots")
    ap.add_argument("--other-arms", nargs="*", default=[], help="the screen's other arms (their matches are skipped and counted)")
    ap.add_argument("--compare-arm", help="a variant whose reply at each root joins the candidates (source arm:<name>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teams", type=int, nargs="*")
    ap.add_argument("--roots-per-match", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-matches", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--first-ms", type=float, default=2000.0)
    ap.add_argument("--turn-ms", type=float, default=50.0)
    ap.add_argument("--ai-test")
    a = ap.parse_args(argv)
    root = ai_test_root(a.ai_test)
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))["variants"]
    files = sorted(matchlog.iter_matches(a.matches, a.teams))
    selected, counts = select(files, manifest, a.arm, a.other_arms + ([a.compare_arm] if a.compare_arm else []))
    if a.max_matches:
        selected = sorted(selected, key=lambda fm: unit_hash("pick", fm[0].name))[:a.max_matches]
    carrier = runner.command_for(REPO / manifest[a.arm]["binary"], REPO / "build" / "p_only")
    compare = runner.command_for(REPO / manifest[a.compare_arm]["binary"], REPO / "build" / "p_only") if a.compare_arm else None
    limits = (a.first_ms / 1000, a.turn_ms / 1000)
    jobs = []
    for f, match in selected:
        commands = seat_commands(match, carrier, root)
        for t in root_turns(match, a.roots_per_match):
            out = Path(a.out) / f"n{match['teams']:02d}" / f"{match['match_id']}-t{t:03d}.json.gz"
            if not out.exists():
                jobs.append((str(f), t, a.candidates, a.repeats, commands, compare, limits, str(root), a.arm, a.compare_arm))
    print(f"[branch_portfolio] {dict(counts)}; {len(jobs)} roots to play, up to {a.candidates} candidates x {a.repeats} repeats",
          flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) if a.workers > 1 else _Serial() as p:
        for i, row in enumerate(p.imap_unordered(_root_job, jobs), 1):
            out = Path(a.out) / f"n{row['teams']:02d}" / f"{row['match_id']}-t{row['turn']:03d}.json.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(out, "wt", encoding="utf-8") as fh:
                json.dump(row, fh)
            print(f"[branch_portfolio] {i}/{len(jobs)} {row['match_id']} t{row['turn']}"
                  + (f" dropped: {row['dropped']}" if "dropped" in row else f" sources {row['sources']}") + f" ({time.time() - t0:.0f}s)",
                  flush=True)
    rows = []
    for f in sorted(Path(a.out).rglob("*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            row = json.load(fh)
        if row.get("arm") == a.arm:
            rows.append(row)
    summary = summarize(rows, counts) | {"arm": a.arm, "compare_arm": a.compare_arm, "candidates": a.candidates, "repeats": a.repeats}
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"[branch_portfolio] summary: {json.dumps(summary)}")


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def imap_unordered(fn, items):
        return map(fn, items)


if __name__ == "__main__":
    main()
