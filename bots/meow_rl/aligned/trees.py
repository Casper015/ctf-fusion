"""Fixed-work search trees of the aligned Meowmind RL search diagnosis (task T0061, docs/MEOWMIND_RL_ALIGNED.md).

    python bots/meow_rl/aligned/trees.py collect --seed-base <ledgered> --out data/meow_rl_aligned/trees   (host, 8 workers)
    python bots/meow_rl/aligned/trees.py select --out data/meow_rl_aligned/trees
    python bots/meow_rl/aligned/trees.py build  --out data/meow_rl_aligned/trees

collect: diagnostic games (never evidence of strength) of the deployed actor-direct bot D (meow_rl_aligned_p2_large, the
built binary of build/p_only/manifest.json, checked against its bundle) against the training lineages of
profiles/meow_rl_training_pool_v2.json (never the held-out test roster), --games per team count 2-15, on exactly eight
worker processes at the official limits. Episode specs come from training.meow_rl.pool (a fresh ledgered seed base,
--seed-ledger docs/MEOWMIND_RL_SEEDS.md), one team count per game. Each match record is kept; a seat's stream is
rebuilt from it exactly (its INIT and the shared TURN blocks, which the runner checked against what that seat received).

select: before any tree is scored, 1,024 roots: 74 per count for 2 and 3, 73 for 4-15; within a count, an even split
over the phases (turns 2-133, 134-266, 267-400; turn 1 has its own 1,500 ms deadline and is excluded) and, within a
phase, a cycle over the corpus's tactical signatures (bots/meow_rl/aligned/corpus.py), at most --per-game roots per
game, each signature's candidates ordered by sha256("<seed>|<match>|<turn>"). Shortages are filled within the count,
then reported; nothing is duplicated.

build: compiles tools/cpp/meow_rl/tree.cpp with D's exported actor header and materializes the search_v2.hpp tree of
every selected root (4 roots x 2 scenarios, common random numbers, no deadline): trees.jsonl (one line per root, with
its tree hash and stream sha256) and leaves.f32 (416 float32 value features per non-terminal leaf). MANIFEST.json
binds the inputs (header and source sha256, streams, roots) to the outputs. bots/meow_rl/aligned/diagnose.py scores them.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "meow_rl" / "aligned"))

import corpus as K  # noqa: E402
from fusion import matchlog, runner  # noqa: E402

WORKERS = 8
COUNTS = list(range(2, 16))
ROOTS = 1024
PHASES = ((2, 133), (134, 266), (267, 400))
SEED = "meow_rl_aligned_roots_v1"
VARIANT = "meow_rl_aligned_p2_large"
LIMITS = (2.0, 0.050)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_quotas(total: int = ROOTS) -> dict[int, int]:
    base, extra = divmod(total, len(COUNTS))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(COUNTS)}


def seat_stream(match: dict, seat: int) -> str:
    return matchlog.init_for_seat(match["init"], seat) + "".join(match["blocks"])


# ---- collect ------------------------------------------------------------------------------------------------------

def _play(job):
    from training.collect import Fixture, play  # noqa: PLC0415
    spec, commands, info, root, keep = job
    seats = list(spec["opponents"])
    seats.insert(spec["seat"], "@candidate")
    fx = Fixture(spec["match_id"], spec["teams"], spec["map_seed"], spec["flag_seed"], seats)
    t0 = time.time()
    match = play((fx, commands, info, root, LIMITS))
    path = Path(keep) / f"{spec['match_id']}.json.gz"
    matchlog.save(path, match)
    me = match["bots"][spec["seat"]]
    return {"match_id": spec["match_id"], "teams": spec["teams"], "seat": spec["seat"], "map_seed": spec["map_seed"],
            "opponents": spec["opponents"], "faults": me["faults"], "dead": me["dead"], "late": me["late"],
            "opponent_faults": sum(b["faults"] for i, b in enumerate(match["bots"]) if i != spec["seat"]),
            "wall_s": round(time.time() - t0, 1), "record_sha256": sha256(path)}


def collect(a) -> dict:
    from fusion.paths import ai_test_root, engine_path, git_commit, load_engine  # noqa: PLC0415
    from training.meow_rl.pool import Pool  # noqa: PLC0415
    from training.meow_rl.train import check_ledger  # noqa: PLC0415
    sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
    from preflight_meow_rl import checked_bundle  # noqa: PLC0415
    if a.workers != WORKERS:
        raise SystemExit(f"the diagnostic collection runs on exactly {WORKERS} workers")
    ledger = check_ledger(a.seed_base, a.seed_ledger, False)
    out = Path(a.out)
    keep = out / "matches"
    if keep.exists() and any(keep.iterdir()):
        raise SystemExit(f"{keep} is not empty: a collection is never resumed or mixed")
    keep.mkdir(parents=True, exist_ok=True)
    root = ai_test_root(a.ai_test)
    os.environ["AI_TEST_ROOT"] = str(root)
    manifest = json.loads((REPO / "build" / "p_only" / "manifest.json").read_text(encoding="utf-8"))["variants"]
    if VARIANT not in manifest:
        raise SystemExit(f"{VARIANT} is not built (python bots/p_only/tools/build.py --variants bots/meow_rl/variants_aligned.json)")
    entry = manifest[VARIANT]
    checked_bundle(VARIANT, entry, REPO)
    pool = Pool.load(Path(a.pool))
    commands, info = pool.commands(root, REPO / "build" / "meow_rl" / "bots")
    commands["@candidate"] = [str(REPO / entry["binary"])]
    info["@candidate"] = {"family": "meow_rl", "sha256": entry["binary_sha256"]}
    from fusion.paths import sha256 as fsha  # noqa: PLC0415
    info |= {"__engine__": fsha(engine_path(root)), "__commit__": git_commit(root)}
    load_engine(root)
    specs = []
    for c in COUNTS:
        for g in range(a.games):
            e = pool.episode(a.seed_base, c * 1000 + g, [c])
            specs.append({"match_id": f"roots-s{a.seed_base}-n{c:02d}-g{g:03d}", "teams": e.teams, "seat": e.seat,
                          "map_seed": e.map_seed, "flag_seed": e.flag_seed, "opponents": list(e.opponents)})
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(_play, [(s, commands, info, root, str(keep)) for s in specs]))
    receipt = {"schema": "ctf-meow-rl-aligned-roots-collection-1", "generator": "bots/meow_rl/aligned/trees.py collect",
               "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "workers": WORKERS,
               "limits_ms": [1000 * LIMITS[0], 1000 * LIMITS[1]], "seed_base": a.seed_base, "seed_ledger": ledger,
               "pool": a.pool, "pool_sha256": sha256(Path(a.pool)), "games_per_count": a.games, "variant": VARIANT,
               "binary_sha256": entry["binary_sha256"], "bundle_sha256": entry["bundle_sha256"],
               "engine_sha256": info["__engine__"], "games": rows,
               "faults": int(sum(r["faults"] for r in rows)), "opponent_faults": int(sum(r["opponent_faults"] for r in rows))}
    (out / "COLLECTION.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    return receipt


# ---- select -------------------------------------------------------------------------------------------------------

def candidates(match_path: Path) -> list[dict]:
    """Every replayable turn of the D seat in one match record, with its strata."""
    from training.meow_rl import bridge as B  # noqa: PLC0415
    match = matchlog.load(match_path)
    seat = next(i for i, s in enumerate(match["seats"]) if s["name"] == "@candidate")
    text = seat_stream(match, seat)
    out = []
    with B.Session(B.load()) as s:
        for block in B.split_stream(text):
            o = s.observe(block)
            if PHASES[0][0] <= o.turn <= PHASES[-1][1]:
                sig = int(K.signature(o.feats[None], o.value_feats[None])[0])
                phase = next(i for i, (lo, hi) in enumerate(PHASES) if lo <= o.turn <= hi)
                out.append({"match": match["match_id"], "teams": match["teams"], "turn": int(o.turn), "phase": phase, "sig": sig})
            s.resolve(0)                                  # all stay: the history does not depend on the joint
    return out


def select(a) -> dict:
    out = Path(a.out)
    if (out / "ROOTS.json").exists():
        raise SystemExit(f"{out}/ROOTS.json exists: roots are selected once, before any tree is scored")
    paths = sorted((out / "matches").glob("*.json.gz"))
    pool = [c for p in paths for c in candidates(p)]
    q = root_quotas(a.roots)
    cells = defaultdict(lambda: defaultdict(deque))
    for c in sorted(pool, key=lambda c: hashlib.sha256(f"{a.seed}|{c['match']}|{c['turn']}".encode()).hexdigest()):
        cells[(c["teams"], c["phase"])][c["sig"]].append(c)
    per_game = defaultdict(int)
    chosen, short = [], {}

    def take(cell: dict, n: int) -> list[dict]:
        got = []
        while len(got) < n:
            moved = False
            for s in sorted(cell):
                while cell[s] and per_game[cell[s][0]["match"]] >= a.per_game:
                    cell[s].popleft()
                if cell[s] and len(got) < n:
                    c = cell[s].popleft()
                    per_game[c["match"]] += 1
                    got.append(c)
                    moved = True
            if not moved:
                break
        return got

    for c in COUNTS:
        per = [q[c] // 3 + (1 if p < q[c] % 3 else 0) for p in range(3)]
        got = []
        for p, n in enumerate(per):
            got += take(cells[(c, p)], n)
        for p in range(3):
            got += take(cells[(c, p)], q[c] - len(got))
        short[str(c)] = q[c] - len(got)
        chosen += got
    by_match = defaultdict(list)
    for c in chosen:
        by_match[c["match"]].append(c["turn"])
    roots = {"schema": "ctf-meow-rl-aligned-roots-1", "generator": "bots/meow_rl/aligned/trees.py select", "seed": a.seed,
             "requested": a.roots, "selected": len(chosen), "per_game_cap": a.per_game, "shortage_by_count": short,
             "candidates": len(pool), "strata": K.strata([{**c, "teams": c["teams"]} for c in chosen]),
             "matches": {m: {"turns": sorted(t), "record_sha256": sha256(out / "matches" / f"{m}.json.gz")}
                         for m, t in sorted(by_match.items())}}
    (out / "ROOTS.json").write_text(json.dumps(roots, indent=1) + "\n", encoding="utf-8", newline="\n")
    return roots


# ---- build --------------------------------------------------------------------------------------------------------

def compile_tree(actor_header: Path, exe: Path) -> Path:
    cxx = runner.find_compiler()
    if not cxx:
        raise SystemExit("no C++ compiler")
    subprocess.run([cxx, "-std=c++20", "-O2", "-w", "-I", str(REPO / "inference"), "-I", str(REPO / "tools" / "cpp" / "meow_rl"),
                    f'-DACTOR_HEADER="{Path(actor_header).resolve().as_posix()}"', str(REPO / "tools" / "cpp" / "meow_rl" / "tree.cpp"),
                    "-o", str(exe)], check=True, capture_output=True)
    return exe


def build(a) -> dict:
    out = Path(a.out)
    roots = json.loads((out / "ROOTS.json").read_text(encoding="utf-8"))
    if (out / "trees.jsonl").exists():
        raise SystemExit(f"{out}/trees.jsonl exists: trees are built once")
    exe = compile_tree(Path(a.actor_header), out / f"tree{runner.EXE}")
    streams = out / "streams"
    streams.mkdir(exist_ok=True)
    lines, offset = [], 0
    with open(out / "leaves.f32", "wb") as leaves:
        for m, spec in roots["matches"].items():
            rec = out / "matches" / f"{m}.json.gz"
            if sha256(rec) != spec["record_sha256"]:
                raise SystemExit(f"{rec} changed after the roots were selected")
            match = matchlog.load(rec)
            seat = next(i for i, s in enumerate(match["seats"]) if s["name"] == "@candidate")
            sp = streams / f"{m}.txt"
            sp.write_text(seat_stream(match, seat), encoding="utf-8", newline="\n")
            lf = streams / f"{m}.f32"
            r = subprocess.run([str(exe), str(sp), *map(str, spec["turns"])], capture_output=True, text=True,
                               env=dict(os.environ, MEOW_RL_TREE_LEAVES=str(lf)), check=True)
            got = [json.loads(x) for x in r.stdout.splitlines()]
            if [g["turn"] for g in got] != spec["turns"]:
                raise SystemExit(f"{m}: built {[g['turn'] for g in got]}, selected {spec['turns']}")
            block = np.fromfile(lf, np.float32)
            for g in got:
                for leaf in g["leaves"]:
                    if leaf["row"] >= 0:
                        leaf["row"] += offset
                lines.append(g | {"match": m, "teams": match["teams"], "stream_sha256": sha256(sp)})
            offset += len(block) // 416
            leaves.write(block.tobytes())
            lf.unlink()
    (out / "trees.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8", newline="\n")
    manifest = {"schema": "ctf-meow-rl-aligned-trees-1", "generator": "bots/meow_rl/aligned/trees.py build",
                "actor_header": str(a.actor_header), "actor_header_sha256": sha256(Path(a.actor_header)),
                "tree_source_sha256": sha256(REPO / "tools" / "cpp" / "meow_rl" / "tree.cpp"),
                "search_source_sha256": sha256(REPO / "bots" / "meow_rl" / "search_v2.hpp"),
                "roots_sha256": sha256(out / "ROOTS.json"), "trees": len(lines),
                "reduced_trees": sum(1 for x in lines if len(x["roots"]) < 4), "leaves": offset,
                "trees_sha256": sha256(out / "trees.jsonl"), "leaves_sha256": sha256(out / "leaves.f32")}
    (out / "TREES.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--seed-base", type=int, required=True)
    c.add_argument("--seed-ledger", default=str(REPO / "docs" / "MEOWMIND_RL_SEEDS.md"))
    c.add_argument("--pool", default="profiles/meow_rl_training_pool_v2.json")
    c.add_argument("--games", type=int, default=16, help="games per team count")
    c.add_argument("--workers", type=int, default=WORKERS)
    c.add_argument("--ai-test")
    c.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--roots", type=int, default=ROOTS)
    s.add_argument("--per-game", type=int, default=6)
    s.add_argument("--seed", default=SEED)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--actor-header", default=str(REPO / "inference" / "meow_rl" / "aligned" / "p2_large" / "actor.hpp"))
    a = ap.parse_args(argv)
    r = {"collect": collect, "select": select, "build": build}[a.cmd](a)
    print(json.dumps({k: v for k, v in r.items() if k not in ("games", "matches")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
