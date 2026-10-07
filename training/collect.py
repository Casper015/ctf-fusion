"""Play matches and record every seat's view and requests (training/specs/DATA_FORMAT.md).

    python -m training.collect --ai-test ~/code/AI_Test --teams 2 3 4 --workers 8
    python -m training.collect --ai-test ... --teams 2 --matches 10 --bots Random,Random   # smoke test

Matches run under the official engine (AI_Test engine.py) with the real time
limits unless --turn-ms says otherwise. Bot processes run through
fusion/runner.py, a thread-based mirror of the original arena core that also
works on Windows. Fixtures are a
pure function of (--seed, team count, match index), so an interrupted run
resumes by skipping files that already exist, and another machine can
produce exactly the same fixture list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from fusion import matchlog, runner
from fusion.paths import REPO, ai_test_root, bot_path, git_commit, load_engine, load_json, load_profile, sha256


@dataclass
class Fixture:
    match_id: str
    teams: int
    map_seed: int
    flag_seed: int
    seats: list[str]


def derive(*parts, bits=31) -> int:
    h = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") % (1 << bits)


def resolve_pool(pool: dict, root: Path, names=None) -> dict[str, dict]:
    """Pool entries by name, with absolute paths; drops missing optional bots."""
    out = {}
    for entry in pool["bots"]:
        path = bot_path(entry, root)
        if not path.exists():
            if entry.get("optional"):
                print(f"[collect] skipping optional bot {entry['name']}: {path} not found", file=sys.stderr)
                continue
            if names is None or entry["name"] in names:
                raise SystemExit(f"bot {entry['name']} not found at {path}")
            continue
        out[entry["name"]] = {**entry, "abs": str(path)}
    return out


def schedule(teams: int, count: int, start: int, seed: int, profile: dict, pool: dict[str, dict], bots=None) -> list[Fixture]:
    collect = profile["collect"]
    if bots:
        names = bots
    elif collect["pool"] == "default":
        names = [n for n, e in pool.items() if e.get("weight", 1) > 0]
    else:
        names = [n for n in collect["pool"] if n in pool]
    anchors = [a for a in collect.get("anchors", []) if a in names]
    fixtures = []
    for i in range(start, start + count):
        rng = random.Random(derive(seed, teams, i, "seats"))
        if bots:
            seats = [bots[j % len(bots)] for j in range(teams)]
        else:
            seats = []
            if anchors and rng.random() < collect.get("anchor_share", 0.0):
                seats.append(rng.choice(anchors))
            while len(seats) < teams:
                fresh = [n for n in names if n not in seats] or names
                weights = [pool[n].get("weight", 1) for n in fresh]
                seats.append(rng.choices(fresh, weights)[0])
        rng.shuffle(seats)
        fixtures.append(Fixture(f"n{teams:02d}-s{seed}-{i:06d}", teams, derive(seed, teams, i, "map"),
                                derive(seed, teams, i, "flag", bits=63), seats))
    return fixtures


def play(job) -> dict:
    """Run one fixture and return its match record (schema ctf-fusion-match-1)."""
    fx, commands, info, root, limits = job
    engine = load_engine(Path(root))
    game = engine.Game(fx.teams, fx.map_seed, flag_seed=fx.flag_seed)
    bots = [runner.Bot(commands[name], name, tuple(limits)) for name in fx.seats]
    init_text, blocks, replies, faults = None, [], [], []
    try:
        while not game.done:
            turn = game.turn + 1
            inputs = [game.encode(i) for i in range(fx.teams)]
            if turn == 1:
                init_text, block = matchlog.split_init(inputs[0])
                for i, text in enumerate(inputs):
                    if matchlog.init_for_seat(init_text, i) + block != text:
                        raise RuntimeError("seat input is not INIT(me) + shared TURN block")
            else:
                block = inputs[0]
                if any(text != block for text in inputs):
                    raise RuntimeError("TURN blocks differ between seats")
            blocks.append(block)
            starts = [b.send(text) for b, text in zip(bots, inputs)]
            lines = [b.receive(t0, turn) for b, t0 in zip(bots, starts)]
            replies.append(lines)
            orders = {}
            for i, (b, line) in enumerate(zip(bots, lines)):
                if line is None:
                    orders[i] = {}
                    continue
                try:
                    orders[i] = game.decode(i, line)
                    b.ok()
                except ValueError as exc:
                    orders[i] = {}
                    b.strike(turn, f"malformed output: {exc}")
            game.step(orders)
            faults += [[turn, i, reason] for i, b in enumerate(bots) for t, reason in b.faults if t == turn]
        final_block = game.encode(0)
    finally:
        for b in bots:
            b.close()
    result = game.result()
    score = result["score"]
    return {
        "schema": matchlog.SCHEMA, "match_id": fx.match_id, "teams": fx.teams,
        "map_seed": fx.map_seed, "flag_seed": fx.flag_seed,
        "seats": [{"name": n, "family": info[n]["family"], "sha256": info[n]["sha256"]} for n in fx.seats],
        "engine_sha256": info["__engine__"], "ai_test_commit": info["__commit__"],
        "limits_ms": [1000 * limits[0], 1000 * limits[1]], "platform": sys.platform,
        "init": init_text, "blocks": blocks, "final_block": final_block, "replies": replies, "faults": faults,
        "result": {"score": score, "ranks": [1 + sum(s > x for s in score) for x in score], "stats": result["stats"]},
        "bots": [b.report() for b in bots], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _work(job):
    fx, out = job[0], Path(job[-1])
    match = play(job[:-1])
    matchlog.save(out, match)
    return fx.match_id, match["result"]["score"], sum(st["faults"] for st in match["bots"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai-test", help="AI_Test checkout root (or AI_TEST_ROOT)")
    ap.add_argument("--teams", type=int, nargs="+", required=True)
    ap.add_argument("--matches", type=int, help="matches per team count (default: profile)")
    ap.add_argument("--start", type=int, default=0, help="first match index")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bots", help="comma-separated seat list, overriding the profile pool (cycled over seats)")
    ap.add_argument("--pool", default=str(REPO / "profiles" / "pool.json"))
    ap.add_argument("--out", default=str(REPO / "data" / "raw"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--turn-ms", type=float, help="per-turn limit (default: engine, 50 ms)")
    ap.add_argument("--first-ms", type=float, help="turn-1 limit (default: engine, 2000 ms)")
    a = ap.parse_args(argv)
    root = ai_test_root(a.ai_test)
    engine = load_engine(root)
    bots = a.bots.split(",") if a.bots else None
    pool = resolve_pool(load_json(a.pool), root, set(bots) if bots else None)
    limits = list(getattr(engine.Game, "TIME_LIMITS", (2.0, 0.05)))
    if a.first_ms:
        limits[0] = a.first_ms / 1000
    if a.turn_ms:
        limits[1] = a.turn_ms / 1000

    jobs, needed = [], set()
    for teams in a.teams:
        profile = load_profile(teams)
        count = a.matches if a.matches is not None else profile["collect"]["matches"]
        for fx in schedule(teams, count, a.start, a.seed, profile, pool, bots):
            out = Path(a.out) / f"n{teams:02d}" / f"{fx.match_id}.json.gz"
            if not out.exists():
                jobs.append((fx, out))
                needed |= set(fx.seats)
    # Compile once in this process; workers reuse the content-hashed binaries.
    commands = {n: runner.command_for(Path(pool[n]["abs"]), REPO / "build" / "bots") for n in needed}
    info = {n: {"family": pool[n]["family"], "sha256": sha256(pool[n]["abs"])} for n in needed}
    info["__engine__"] = sha256(Path(engine.__file__))
    info["__commit__"] = git_commit(root)
    work = [(fx, commands, info, str(root), limits, str(out)) for fx, out in jobs]
    print(f"[collect] {len(work)} matches to play ({len(a.teams)} team counts), {a.workers} workers", flush=True)
    t0, done = time.time(), 0
    with mp.Pool(a.workers) if a.workers > 1 else _Serial() as p:
        for match_id, score, faults in p.imap_unordered(_work, work):
            done += 1
            print(f"[collect] {done}/{len(work)} {match_id} score={score} faults={faults} ({time.time() - t0:.0f}s)", flush=True)
    manifest = Path(a.out) / "fixtures.jsonl"
    with manifest.open("a") as f:
        for fx, out in jobs:
            f.write(json.dumps(asdict(fx) | {"file": out.relative_to(a.out).as_posix()}) + "\n")


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
