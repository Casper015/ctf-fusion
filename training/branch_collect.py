"""Candidate-branch continuations for value ranking (host only: it plays real bots under the official limits).

    python -m training.branch_collect --matches data/p_only/v_data/matches --manifest build/p_only/manifest.json \\
        --carrier v12_ponly_xxl --out data/branches/v12_ponly_xxl --roots-per-match 2 --candidates 4 --workers 6

A root is one turn t of a match that the frozen carrier played in its designated seat (a screen run with
keep_matches, bots/p_only/profiles/collect_v_data.json). For each root, K legal candidate orders for that seat
are each played to the end of the match:

  1. the engine is rebuilt from the match's seeds and stepped with the *recorded* orders of every seat up to
     turn t - 1; every reproduced TURN block must equal the recorded one, or the root is dropped (the root
     match record itself is only read, never changed);
  2. every bot (the carrier and each opponent) is a fresh process that receives exactly the recorded inputs
     of turns 1 .. t - 1, so its history is the root's; its replies there are discarded (they are compared
     with the recorded ones: `prefix_mismatch` counts differences, a measure of nondeterminism);
  3. at turn t the opponents' recorded replies are used for every candidate (they could not see our order),
     and our seat plays the candidate;
  4. from turn t + 1 everything is live under the official limits, to the end; the outcome is our seat's
     final pairwise utility (fusion.labels.pairwise_outcome).

Candidates: the carrier's own recorded reply (on-policy), plus K - 1 distinct one-unit deviations of it (one
active unit moves to another open neighbour, its action kept), drawn by a hash of (match, turn). They are
legal orders, not predictions; nothing here asks P what anyone will do. Caveat: the carrier's internal memory
at turn t reflects its own choice, not the forced candidate.

Rows (one JSON per root under --out/nXX/<match>-t<turn>.json.gz) keep the INIT text, the root block (turn t)
and each branch's next block (turn t + 1), so training.branch_dataset can extract the successor's features
with the same encoder the bot runs. Roots of one match share its map, and every split is by map
(training.dataset.split_of), so a root and its branches never straddle two splits.
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
from pathlib import Path

from fusion import matchlog, runner
from fusion.labels import pairwise_outcome
from fusion.paths import REPO, ai_test_root, bot_path, load_engine, load_pool, sha256
from fusion.protocol import DELTAS, MOVES, parse_block, parse_reply

CANDIDATE = "@candidate"
PREFIX_LIMIT = 5.0  # seconds per reply while a bot replays the recorded prefix (no timing is measured there)


def unit_hash(*parts) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little")


def designated_seat(match: dict) -> int:
    seats = [i for i, s in enumerate(match["seats"]) if s["name"] == CANDIDATE or s.get("family") == "candidate"]
    if len(seats) != 1:
        raise ValueError(f"{match['match_id']}: needs exactly one designated seat, found {seats}")
    return seats[0]


def root_turns(match: dict, count: int, margin: int = 5) -> list[int]:
    """`count` distinct turns, away from the first and last `margin` turns, by a hash of the match id."""
    turns = len(match["blocks"])
    pool = list(range(1 + margin, max(turns - margin, 2 + margin)))
    pool.sort(key=lambda t: unit_hash(match["match_id"], "root", t))
    return sorted(pool[:count])


def reproduce(engine, match: dict, upto: int):
    """A fresh engine game stepped with the recorded orders of turns 1 .. upto - 1; None if a block differs."""
    game = engine.Game(match["teams"], match["map_seed"], flag_seed=match["flag_seed"])
    for turn in range(1, upto):
        block = game.encode(0)
        if turn > 1 and block != match["blocks"][turn - 1]:
            return None
        game.step(recorded_orders(game, match, turn))
    if upto > 1 and game.encode(0) != match["blocks"][upto - 1]:
        return None
    return game


def recorded_orders(game, match: dict, turn: int) -> dict:
    orders = {}
    for i, line in enumerate(match["replies"][turn - 1]):
        try:
            orders[i] = game.decode(i, line) if line is not None else {}
        except ValueError:
            orders[i] = {}
    return orders


def candidates(game, block_text: str, line: str | None, seat: int, turn: int, k: int, key: str) -> list[str]:
    """The recorded reply first (on-policy), then distinct one-unit move deviations of it."""
    reqs = parse_reply(line, turn)
    if reqs is None:
        return []
    blk = parse_block(block_text)
    walls, size = game.walls, game.size
    tok = line.partition("#")[0].split()
    out, seen = [" ".join(tok)], {" ".join(tok)}
    options = []
    for u in range(3):
        _, _, x, y, *_ = blk.units[3 * seat + u]
        if x < 0:
            continue
        for m, (dx, dy) in enumerate(DELTAS):
            nx, ny = x + dx, y + dy
            if m != reqs[u].move and 0 <= nx < size and 0 <= ny < size and (nx, ny) not in walls:
                alt = list(tok)
                alt[1 + 2 * u] = MOVES[m]
                options.append(" ".join(alt))
    options.sort(key=lambda s: unit_hash(key, s))
    for s in options:
        if len(out) >= k:
            break
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def play_branch(engine, match: dict, seat: int, turn: int, line: str, commands: list[list[str]],
                limits: tuple[float, float]) -> dict:
    """Replay the recorded prefix into fresh bots, force `line` for our seat at `turn`, play live to the end."""
    game = engine.Game(match["teams"], match["map_seed"], flag_seed=match["flag_seed"])
    names = [s["name"] for s in match["seats"]]
    bots = [runner.Bot(cmd, name, (limits[0], PREFIX_LIMIT)) for cmd, name in zip(commands, names)]
    mismatch, prefix_faults, faults = 0, 0, []
    root_block = next_block = None
    try:
        while not game.done:
            t = game.turn + 1
            inputs = [game.encode(i) for i in range(match["teams"])]
            if t == turn:
                root_block = inputs[0] if t > 1 else matchlog.split_init(inputs[0])[1]
                for b in bots:   # official limits from the branch turn on
                    b.turn_limit = limits[1]
                    b.strikes = 0
            if t == turn + 1:
                next_block = inputs[0]
            starts = [b.send(text) for b, text in zip(bots, inputs)]
            lines = [b.receive(t0, t) for b, t0 in zip(bots, starts)]
            if t < turn:
                mismatch += sum(a != b for a, b in zip(lines, match["replies"][t - 1]))
                prefix_faults += sum(b.faults[-1][0] == t for b in bots if b.faults)
                orders = recorded_orders(game, match, t)
            else:
                if t == turn:
                    lines = list(match["replies"][t - 1])
                    lines[seat] = line
                orders = {}
                for i, (b, ln) in enumerate(zip(bots, lines)):
                    if ln is None:
                        orders[i] = {}
                        continue
                    try:
                        orders[i] = game.decode(i, ln)
                        b.ok()
                    except ValueError as exc:
                        orders[i] = {}
                        if t > turn:
                            b.strike(t, f"malformed output: {exc}")
                faults += [[t, i, reason] for i, b in enumerate(bots) for tt, reason in b.faults if tt == t and t > turn]
            game.step(orders)
        score = game.result()["score"]
    finally:
        for b in bots:
            b.close()
    return {"line": line, "score": score, "outcome": float(pairwise_outcome(score)[seat]), "prefix_mismatch": mismatch,
            "prefix_faults": prefix_faults, "faults": faults, "root_block": root_block, "next_block": next_block}


def _root_job(job) -> dict | None:
    path, turn, k, repeats, commands, limits, root = job
    engine = load_engine(Path(root))
    match = matchlog.load(path)
    seat = designated_seat(match)
    game = reproduce(engine, match, turn)
    if game is None:
        return {"match_id": match["match_id"], "teams": match["teams"], "turn": turn, "dropped": "recorded blocks not reproduced"}
    snapshot = copy.deepcopy(game)  # the reproduced root; branches below never touch it
    line = match["replies"][turn - 1][seat]
    cands = candidates(game, match["blocks"][turn - 1], line, seat, turn, k, f"{match['match_id']}/{turn}")
    if len(cands) < 2:
        return {"match_id": match["match_id"], "teams": match["teams"], "turn": turn, "dropped": "fewer than two legal candidates"}
    branches = []
    for c, cand in enumerate(cands):
        for r in range(repeats):
            row = play_branch(engine, match, seat, turn, cand, commands, limits)
            branches.append({"candidate": c, "repeat": r, "on_policy": c == 0} | row)
    if snapshot.encode(0) != game.encode(0):
        raise RuntimeError("the reproduced root changed while branches were played")
    return {"schema": "ctf-fusion-branches-1", "match_id": match["match_id"], "teams": match["teams"],
            "map_seed": match["map_seed"], "seat": seat, "turn": turn, "init": match["init"],
            "carrier_sha256": match["seats"][seat]["sha256"], "candidates": cands, "branches": branches,
            "root_outcome": float(pairwise_outcome(match["result"]["score"])[seat])}


def resolve_commands(match: dict, manifest: dict, carrier: str, root: Path) -> list[list[str]]:
    """Commands for every seat: the frozen carrier binary (its SHA-256 must equal the match's), the roster bots by
    pool entry (their sources must be unchanged)."""
    pool = {e["name"]: e for e in load_pool()["bots"]}
    m = manifest[carrier]
    out = []
    for s in match["seats"]:
        if s["name"] == CANDIDATE:
            if s["sha256"] != m["binary_sha256"]:
                raise SystemExit(f"{match['match_id']}: the carrier binary changed since the match was played (rebuilt?)")
            out.append([str(REPO / m["binary"])])
            continue
        path = bot_path(pool[s["name"]], root)
        if sha256(path) != s["sha256"]:
            raise SystemExit(f"{match['match_id']}: bot {s['name']} changed since the match was played")
        out.append(runner.command_for(path, REPO / "build" / "roster"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True, help="match records of the frozen carrier (screen keep_matches dir)")
    ap.add_argument("--manifest", default=str(REPO / "build" / "p_only" / "manifest.json"))
    ap.add_argument("--carrier", required=True, help="variant name of the frozen carrier in the build manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teams", type=int, nargs="*", help="team counts (default: all found)")
    ap.add_argument("--roots-per-match", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=1, help="continuations per candidate (bots may be nondeterministic)")
    ap.add_argument("--max-matches", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--first-ms", type=float, default=2000.0)
    ap.add_argument("--turn-ms", type=float, default=50.0)
    ap.add_argument("--ai-test")
    a = ap.parse_args(argv)
    root = ai_test_root(a.ai_test)
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))["variants"]
    files = list(matchlog.iter_matches(a.matches, a.teams))
    if a.max_matches:
        files = sorted(files, key=lambda p: unit_hash("pick", p.name))[:a.max_matches]
    limits = (a.first_ms / 1000, a.turn_ms / 1000)
    jobs = []
    for f in files:
        match = matchlog.load(f)
        commands = resolve_commands(match, manifest, a.carrier, root)
        for t in root_turns(match, a.roots_per_match):
            out = Path(a.out) / f"n{match['teams']:02d}" / f"{match['match_id']}-t{t:03d}.json.gz"
            if not out.exists():
                jobs.append((str(f), t, a.candidates, a.repeats, commands, limits, str(root)))
    print(f"[branches] {len(jobs)} roots to play from {len(files)} matches, {a.candidates} candidates x {a.repeats} repeats",
          flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) if a.workers > 1 else _Serial() as p:
        for i, row in enumerate(p.imap_unordered(_root_job, jobs), 1):
            if row is None:
                continue
            out = Path(a.out) / f"n{row['teams']:02d}" / f"{row['match_id']}-t{row['turn']:03d}.json.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(out, "wt", encoding="utf-8") as fh:
                json.dump(row, fh)
            print(f"[branches] {i}/{len(jobs)} {row['match_id']} t{row['turn']}"
                  + (f" dropped: {row['dropped']}" if "dropped" in row else
                     f" outcomes {[round(b['outcome'], 3) for b in row['branches']]}") + f" ({time.time() - t0:.0f}s)", flush=True)


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def imap_unordered(fn, items):
        return map(fn, items)


if __name__ == "__main__":
    sys.exit(main())
