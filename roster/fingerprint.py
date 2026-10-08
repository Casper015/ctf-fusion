"""Replay the frozen probe suite to every candidate bot and record what it requests.

    python -m roster.fingerprint --suite data/roster/probe_suite.json --inventory data/roster/bot_inventory.json
    python -m roster.fingerprint --suite ... --bot Random=@repo/tools/random_bot.py --repeat 2

Each run feeds one bot one probe stream turn by turn through fusion.runner
(the collection runner), as the probe's seat, and keeps every reply line. The
replies are canonicalized exactly like training labels (fusion.labels): a
move into a wall is a stay, a pickup or drop the engine would ignore is a
wait, an attack on an unhittable unit is a wait. Each run file holds the raw
lines, per-turn reply times, faults, and per turn and unit the requested
move, action and attack target (unit id).

Limits are relaxed (--limits, default 10 s first turn, 1 s later): a
fingerprint asks what the bot wants to do, not whether it meets 50 ms. A bot
that searches until its own clock runs out answers differently on a busy
machine, so keep --workers well below the core count, and use --repeat 2 or
more: roster.distance turns repeated runs into each bot's A/A noise. Reply
times here are informational only; deadlines are measured on the host.

Inventory bots used: eligible (not excluded) representatives; a byte alias
reuses its representative's evidence and is never replayed. Runs are stored
one file per (bot, probe, repeat) under --out and are resumable.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from fusion import matchlog, runner
from fusion.labels import label_match
from fusion.paths import REPO
from fusion.protocol import ATTACK, parse_reply
from fusion.schema import layout as feature_layout
from roster.common import SCHEMA_FINGERPRINT, bot_key, load_json, sha256_file
from roster.probes import probe_records


def resolve(path: str, ai_test_root: str | None) -> Path:
    if path.startswith("@repo/"):
        return REPO / path[len("@repo/"):]
    p = Path(path)
    return p if p.is_absolute() or not ai_test_root else Path(ai_test_root) / p


def bots_from_inventory(inv: dict) -> list[dict]:
    out = []
    for b in inv["bots"]:
        if b["excluded"]:
            continue
        out.append({"id": b["id"], "path": str(resolve(b["path"], inv.get("ai_test_root"))),
                    "source_sha256": b["source_sha256"], "binary_sha256": b["build"].get("binary_sha256")})
    return out


def command(bot: dict) -> list[str]:
    return runner.command_for(Path(bot["path"]), REPO / "build" / "roster")


@lru_cache(maxsize=8)
def _probe(file: str, sha: str, turns: int):
    if sha256_file(file) != sha:
        raise RuntimeError(f"{file} changed since the suite was frozen")
    match = matchlog.load(file)
    m, rec = probe_records(match, turns)
    return m, rec


def canonical(m: dict, rec: np.ndarray, seat: int, lines: list) -> dict:
    """Requested move, action and target per turn and unit, canonicalized like the training labels."""
    layout = feature_layout()
    replies = [row[:seat] + [line] + row[seat + 1:] for row, line in zip(m["replies"], lines)]
    lab = label_match(m | {"replies": replies}, rec, layout)
    rows = rec[:, 1] == seat
    move, act, valid = lab["y_move"][rows], lab["y_act"][rows], lab["valid"][rows]
    target = np.full(move.shape, -1, np.int64)
    for t, line in enumerate(lines):
        reqs = parse_reply(line, t + 1)
        if reqs is None:
            continue
        for k, req in enumerate(reqs):
            if act[t, k] == ATTACK:
                target[t, k] = req.target
    return {"valid": valid.astype(int).tolist(), "move": move.astype(int).tolist(), "act": act.astype(int).tolist(),
            "target": target.tolist()}


def replay(job) -> dict:
    """One bot over one probe stream; returns the run record (also written to disk)."""
    bot, probe, rep, limits, out_file = job
    m, rec = _probe(probe["file"], probe["match_sha256"], probe["turns"])
    seat = probe["seat"]
    init = matchlog.init_for_seat(m["init"], seat)
    proc = runner.Bot(command(bot), bot["id"], limits)
    lines, ms = [], []
    try:
        for t, block in enumerate(m["blocks"], 1):
            started = proc.send(init + block if t == 1 else block)
            line = proc.receive(started, t)
            ms.append(None if line is None or started is None else round(1000 * (time.perf_counter() - started), 3))
            if line is not None:
                if parse_reply(line, t) is None:
                    proc.strike(t, "malformed reply")
                else:
                    proc.ok()
            lines.append(line)
    finally:
        proc.close()
    run = {"schema": SCHEMA_FINGERPRINT, "bot": bot, "probe_id": probe["probe_id"], "teams": probe["teams"], "seat": seat,
           "rep": rep, "limits_ms": [1000 * limits[0], 1000 * limits[1]], "lines": lines, "reply_ms": ms,
           "faults": [[t, r] for t, r in proc.faults], "dead": proc.dead, "host_cpus": os.cpu_count(),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")} | canonical(m, rec, seat, lines)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(run, f, separators=(",", ":"))
    tmp.replace(out_file)
    return {"bot": bot["id"], "probe": probe["probe_id"], "rep": rep, "faults": len(run["faults"]), "dead": run["dead"]}


def run_file(out: Path, bot_id: str, probe_id: str, rep: int) -> Path:
    return out / bot_key(bot_id) / f"{probe_id}-r{rep}.json.gz"


def load_all(out: Path) -> dict[str, list[dict]]:
    """Every stored run under `out`, by bot id."""
    runs = {}
    for p in sorted(Path(out).glob("*/*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            run = json.load(f)
        runs.setdefault(run["bot"]["id"], []).append(run)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--inventory", help="bot_inventory.json: replay every eligible representative")
    ap.add_argument("--bot", action="append", default=[], help="extra bot as ID=PATH (PATH may use @repo/)")
    ap.add_argument("--only", nargs="*", help="restrict to these bot ids")
    ap.add_argument("--repeat", type=int, default=2, help="runs per bot and probe (A/A noise needs 2 or more)")
    ap.add_argument("--limits", default="10000,1000", help="first,later reply limits in ms (relaxed on purpose)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "fingerprints"))
    a = ap.parse_args(argv)
    suite = load_json(a.suite)
    bots = bots_from_inventory(load_json(a.inventory)) if a.inventory else []
    for spec in a.bot:
        bid, _, path = spec.partition("=")
        p = resolve(path, None)
        bots.append({"id": bid, "path": str(p), "source_sha256": sha256_file(p), "binary_sha256": None})
    if a.only:
        bots = [b for b in bots if b["id"] in set(a.only)]
    first, later = (float(x) / 1000 for x in a.limits.split(","))
    out = Path(a.out)
    for b in bots:  # compile before forking, so workers never race on a build
        if b["path"].endswith(".cpp"):
            command(b)
    jobs = [(b, p, r, (first, later), run_file(out, b["id"], p["probe_id"], r))
            for b in bots for p in suite["probes"] for r in range(a.repeat)]
    todo = [j for j in jobs if not j[-1].exists()]
    print(f"[fingerprint] {len(bots)} bots x {len(suite['probes'])} probes x {a.repeat} = {len(jobs)} runs, "
          f"{len(todo)} to do, {a.workers} workers", flush=True)
    pool = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
    results = pool.map(replay, todo) if pool else map(replay, todo)
    for i, r in enumerate(results, 1):
        if r["faults"] or r["dead"]:
            print(f"[fingerprint] {r['bot']} on {r['probe']} r{r['rep']}: {r['faults']} faults, dead: {r['dead']}", flush=True)
        if i % 50 == 0:
            print(f"[fingerprint] {i}/{len(todo)}", flush=True)
    if pool:
        pool.shutdown()


if __name__ == "__main__":
    main()
