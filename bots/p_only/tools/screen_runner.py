"""Play a paired P-only screen with interleaved arms and identity-bound, resumable results (host only).

    python bots/p_only/tools/screen_runner.py --schedule <schedule.json> --manifest build/p_only/manifest.json \\
        --arms c08_control_2ply c08_ponly_xl --aa c08_ponly_xl --results <results.jsonl> --workers 6 [--keep-matches DIR]
    python bots/p_only/tools/screen_runner.py ... --dry-run      # identities and the job order, no games

Every arm plays every fixture of the schedule (map, flag seed, seat and opponents
from the frozen reference roster); only our bot changes. Order is interleaved,
not one arm after another: fixture i runs all arms, rotated by i, so each arm
takes every position equally often and slow drift of the host's load spreads
over all arms. An aa arm ("X~aa") replays X's identical binary.

Identity. Each arm's identity digest covers its bundle and binary SHA-256, every
inlined source with its SHA-256 (carrier, P-only model, inference headers and
model weights: the model and adapter identity), the runner code (this file,
roster/strength.py, training/collect.py, fusion/runner.py), the schedule and the
limits. Rows carry it; resuming skips only rows with the identical digest, and
bots/p_only/tools/analyze.py ignores rows of any other digest as stale. identities.json
(written next to the results) records the components.

Rows are roster.strength rows (utility, faults, late, dead, first/p99/max reply,
engine hash, live outcomes) plus arm, identity and job order. Faults and late
replies stay in the outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fusion import runner  # noqa: E402
from fusion.paths import ai_test_root, bot_path, engine_path, git_commit, load_engine, load_pool, sha256  # noqa: E402
from roster.strength import play_one, schedule_identity  # noqa: E402

AA = "~aa"
RUNNER_FILES = ["bots/p_only/tools/screen_runner.py", "roster/strength.py", "training/collect.py", "fusion/runner.py"]


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def runner_digest() -> str:
    return digest({f: sha256(REPO / f) for f in RUNNER_FILES})


def identities(arms: list[str], manifest: dict, sched: dict, limits: tuple[float, float]) -> dict:
    """arm -> {digest, components}; an aa arm has exactly its base arm's components."""
    out = {}
    run_id = runner_digest()
    for arm in arms:
        m = manifest[arm.removesuffix(AA)]
        comp = {"variant": arm.removesuffix(AA), "bundle_sha256": m["bundle_sha256"], "binary_sha256": m["binary_sha256"],
                "inputs_sha256": digest(sorted((i["path"], i["sha256"]) for i in m["inputs"])),
                "models": sorted({str(Path(i["path"]).parent) for i in m["inputs"] if "/models" in i["path"]}),
                "stats": m.get("stats"), "runner_sha256": run_id, "schedule": schedule_identity(sched),
                "limits_ms": [round(1000 * limits[0]), round(1000 * limits[1])]}
        out[arm] = {"digest": digest(comp), "components": comp}
    return out


def job_order(fixtures: list[dict], arms: list[str]) -> list[tuple[dict, str]]:
    """Fixture by fixture; fixture i runs every arm, rotated by i."""
    jobs = []
    for i, fx in enumerate(fixtures):
        k = i % len(arms)
        jobs += [(fx, arm) for arm in arms[k:] + arms[:k]]
    return jobs


def reference_commands(sched: dict, root: Path) -> tuple[dict, dict]:
    pool = {e["name"]: e for e in load_pool()["bots"]}
    build = REPO / "build" / "roster"
    commands, info = {}, {}
    for ref in sched["reference"]:
        path = bot_path(pool[ref["name"]], root)
        if sha256(path) != ref["source_sha256"]:
            raise SystemExit(f"reference bot {ref['name']} changed since the schedule was frozen")
        commands[ref["name"]] = runner.command_for(path, build)
        info[ref["name"]] = {"family": pool[ref["name"]]["family"], "sha256": ref["source_sha256"]}
    return commands, info


def _play(job):
    order, arm, ident, game = job
    row = play_one(game)
    return row | {"arm": arm, "identity": ident, "order": order}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--aa", nargs="*", default=[])
    ap.add_argument("--results", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--first-ms", type=float, default=2000.0)
    ap.add_argument("--turn-ms", type=float, default=50.0)
    ap.add_argument("--keep-matches")
    ap.add_argument("--ai-test")
    ap.add_argument("--dry-run", action="store_true", help="write identities.json and print the job order; play nothing")
    a = ap.parse_args(argv)
    sched = json.loads(Path(a.schedule).read_text(encoding="utf-8"))
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))["variants"]
    arms = list(a.arms) + [x + AA for x in a.aa]
    missing = [x for x in arms if x.removesuffix(AA) not in manifest]
    if missing:
        raise SystemExit(f"arms not in the build manifest: {missing}")
    limits = (a.first_ms / 1000, a.turn_ms / 1000)
    ids = identities(arms, manifest, sched, limits)
    out = Path(a.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "identities.json").write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("arm") in ids and r.get("identity") == ids[r["arm"]]["digest"]:
                    done.add((r["arm"], r["fixture_id"]))
    order = job_order(sched["fixtures"], arms)
    todo = [(i, fx, arm) for i, (fx, arm) in enumerate(order) if (arm, fx["fixture_id"]) not in done]
    print(f"[screen] {len(arms)} arms x {len(sched['fixtures'])} fixtures = {len(order)} games; {len(done)} already done, "
          f"{len(todo)} to play, {a.workers} workers", flush=True)
    if a.dry_run:
        for i, fx, arm in todo[:2 * len(arms)]:
            print(f"  {i:5d} {fx['fixture_id']} {arm}")
        for arm in arms:
            print(f"  {arm:24s} identity {ids[arm]['digest'][:16]} binary {ids[arm]['components']['binary_sha256'][:12]}")
        return
    root = ai_test_root(a.ai_test)
    os.environ["AI_TEST_ROOT"] = str(root)
    commands, info = reference_commands(sched, root)
    info |= {"__engine__": sha256(engine_path(root)), "__commit__": git_commit(root)}
    sid = schedule_identity(sched)
    jobs = []
    for i, fx, arm in todo:
        m = manifest[arm.removesuffix(AA)]
        cand = {"id": arm, "key": hashlib.sha256(arm.encode()).hexdigest()[:10], "source_sha256": m["binary_sha256"], "schedule": sid}
        cmd = [str(REPO / m["binary"])]
        game = (fx, cand, commands | {"@candidate": cmd}, info | {"@candidate": {"family": "candidate", "sha256": m["binary_sha256"]}},
                str(root), limits, a.keep_matches)
        jobs.append((i, arm, ids[arm]["digest"], game))
    load_engine(root)
    pool = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
    results = pool.map(_play, jobs, chunksize=1) if pool else map(_play, jobs)
    with out.open("a", encoding="utf-8") as f:
        for n, r in enumerate(results, 1):
            f.write(json.dumps(r) + "\n")
            if n % 50 == 0:
                f.flush()
                print(f"[screen] {n}/{len(jobs)}", flush=True)
    if pool:
        pool.shutdown()


if __name__ == "__main__":
    main()
