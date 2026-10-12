"""Designated-carrier whole-match collection against the expanded opponent registry (task T0035).

    # 1. availability: every member, how its source resolves, disabled/excluded/alias states (no build, no game)
    python -m training.decision_collect list --out build/decision/registry.json --simulate 2000
    # 2. a schedule on an unused seed (refused if the DIGEST ledger lists it); realized shares are reported
    python -m training.decision_collect schedule --seed SEED --maps-per-count 25 --out build/decision/SCHEDULE.json
    # 3. build every bot the schedule needs and freeze all identities (host, before any game)
    python -m training.decision_collect prepare --schedule build/decision/SCHEDULE.json --out-dir build/decision/run1
    # 4. play (resumable): complete raw trajectories, rows and error records
    python tools/repo/task.py lock cpu --task TXXXX -- python -m training.decision_collect collect \\
        --frozen build/decision/run1/FROZEN.json --out data/decision/run1 --workers 6

Every fixture holds exactly one @candidate seat (family "candidate", its binary hash): the frozen designated
incumbent, whose own rows are the decision data (training.dataset --rows designated); every other seat is an opponent
whose rows are opponent-policy labels. Limits are the official 2,000 ms first reply and 50 ms per turn; they are not a
parameter. Matches keep every seat, reply, fault and the sampling metadata (component, group and lineage per seat,
and the pool identity).

Freezing (prepare) pins: the registry and its pool identity (sources, aliases, exclusions), the schedule, every
binary with its bundle inputs (sources, includes, weights and their folded calibration) or its source hash, the
compiler and flags, the engine, the runner and collector code, and the limits. prepare refuses while a collection
holds the output's lock. collect recomputes all of them and refuses on any difference, so a rebuild, an edited
source, another engine or another schedule cannot resume. A fixture is skipped only when its row carries the frozen
identity and its match file exists with the recorded SHA-256 and matches the fixture; a partial match is moved aside
(partial/), never silently reused; a failure writes errors/<fixture>.json and the fixture stays to do. Files are
written atomically.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fusion.paths import REPO
from training import decision_opponents as O

SCHEMA_FROZEN = "ctf-fusion-decision-frozen-1"
SCHEMA_ROW = "ctf-fusion-decision-row-1"
LIMITS = (2.0, 0.05)          # official first-reply and per-turn limits, seconds
RUNNER_FILES = ("training/decision_collect.py", "training/decision_opponents.py", "training/collect.py",
                "fusion/runner.py", "fusion/matchlog.py")
CANDIDATE = "@candidate"


def write_atomic(path: Path, data: bytes | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_atomic(path, json.dumps(obj, indent=1, sort_keys=True) + "\n")


def rel(p) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)


def build_tools():
    spec = importlib.util.spec_from_file_location("p_only_build", REPO / "bots" / "p_only" / "tools" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- identities

def runner_identity() -> dict:
    return {f: O.sha256_file(REPO / f) for f in RUNNER_FILES}


def engine_identity(root: Path) -> dict:
    from fusion.paths import engine_path, git_commit  # noqa: PLC0415
    return {"sha256": O.sha256_file(engine_path(root)), "commit": git_commit(root)}


def compiler_identity() -> dict:
    from fusion.runner import CXX_FLAGS, find_compiler  # noqa: PLC0415
    cxx = find_compiler()
    if not cxx:
        raise SystemExit("no C++ compiler found")
    r = subprocess.run([cxx, "--version"], capture_output=True, text=True)
    flags = [f if not f.startswith("-I") else "-I" + rel(f[2:]) for f in CXX_FLAGS]
    return {"path": cxx, "version": ((r.stdout or r.stderr).splitlines() or [cxx])[0], "flags": flags}


def frozen_identity(frozen: dict) -> str:
    return O.digest({k: v for k, v in frozen.items() if k not in ("created", "identity")})


# ---------------------------------------------------------------- list and schedule

def strip_private(reg: dict) -> dict:
    return {k: v for k, v in reg.items() if not k.startswith("_")}


def cmd_list(a):
    root = Path(a.ai_test) if a.ai_test else (Path(os.environ["AI_TEST_ROOT"]) if os.environ.get("AI_TEST_ROOT") else None)
    reg = O.build_registry(root)
    out = strip_private(reg)
    if a.simulate:
        out["simulation"] = O.simulate(reg, a.simulate)
    write_json(Path(a.out), out)
    print(f"[decision] pool {reg['pool_identity'][:16]}: {reg['counts']}")
    for m in reg["members"]:
        if m["status"] != "enabled" or (m.get("resolution") or {}).get("substituted"):
            how = (m.get("resolution") or {}).get("how", "")
            print(f"  {m['component']:9s} {m['name']:36s} {m['status']:15s} {how:10s} {m.get('reason', '')[:70]}")
    for w in reg["warnings"]:
        print(f"  warning: {w}")


def schedule_shares(sched: dict) -> dict:
    out = {}
    for n in sched["team_counts"]:
        fx = [f for f in sched["fixtures"] if f["teams"] == n]
        seats = sum(len(f["components"]) for f in fx) or 1
        comp = {}
        for f in fx:
            for c in f["components"]:
                comp[c] = comp.get(c, 0) + 1
        out[str(n)] = {c: round(k / seats, 4) for c, k in sorted(comp.items())}
    return out


def cmd_schedule(a):
    reg = O.build_registry(Path(a.ai_test) if a.ai_test else None)
    sched = O.make_schedule(reg, a.seed, a.maps_per_count, a.teams)
    sched["realized_component_shares"] = schedule_shares(sched)
    sched["mixture"] = {k: v for k, v in O.mixture(reg).items() if k not in ("rho", "units")}
    write_json(Path(a.out), sched)
    print(f"[decision] schedule {sched['schedule_identity'][:16]}: {len(sched['fixtures'])} fixtures, seed {a.seed}, "
          f"pool {reg['pool_identity'][:16]}")
    for n, sh in sched["realized_component_shares"].items():
        print(f"  n{int(n):02d} {sh}")


# ---------------------------------------------------------------- prepare and freeze

def catalog_entry(name: str) -> dict:
    vcfg = O.load_json(O.VARIANTS)
    for v in O.merge_catalogs(vcfg["catalogs"]):
        if v["name"] == name:
            return {k: x for k, x in v.items() if k != "_catalog"}
    raise SystemExit(f"{name} is not in the catalogs")


def default_builders():
    tools = build_tools()
    from fusion.runner import compile_cpp, find_compiler  # noqa: PLC0415

    def variant(name: str, out: Path, stats: bool) -> dict:
        e = tools.build_one(catalog_entry(name), find_compiler(), out, stats)
        return {"binary": e["binary"], "binary_sha256": e["binary_sha256"], "bundle_sha256": e["bundle_sha256"],
                "inputs": e["inputs"], "stats": stats}

    def source(path: Path, out: Path) -> dict:
        exe = compile_cpp(Path(path), out)
        return {"binary": rel(exe), "binary_sha256": O.sha256_file(exe)}
    return variant, source


def lock_path(out: Path) -> Path:
    return Path(out) / "COLLECTING.lock"


def prepare(schedule_path: Path, out_dir: Path, root: Path | None, builders=None, engine=None, compiler=None,
            registry=None) -> dict:
    """Build every bot the schedule needs and write FROZEN.json; refuses while a collection holds the lock."""
    out_dir = Path(out_dir)
    if lock_path(out_dir).exists():
        raise SystemExit(f"{lock_path(out_dir)} exists: a collection is running against these builds; do not rebuild")
    sched = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    reg = registry or O.build_registry(root)
    if reg["pool_identity"] != sched["pool_identity"]:
        raise SystemExit("the registry's pool identity differs from the schedule's: make a new schedule")
    variant, source = builders or default_builders()
    members = {m["name"]: m for m in reg["members"]}
    d = reg["designated"]
    # The designated carrier has its own build directory: the same variant can also be an opponent (built without
    # PONLY_STATS), and one directory would let one bundle overwrite the other's source file.
    designated = variant(d["variant"], out_dir / "designated", bool(d.get("stats")))
    needed = sorted({n for f in sched["fixtures"] for n in f["opponents"]})
    opponents = {}
    for name in needed:
        m = members[name]
        if m.get("status") != "enabled":
            raise SystemExit(f"schedule seats {name}, whose status is {m.get('status')}")
        base = {"component": m["component"], "group": m["group"], "lineage": m["lineage"], "unit": m["unit"],
                "family": m["family"], "aliases": m["aliases"]}
        if m["kind"] == "variant":
            opponents[name] = base | variant(name, out_dir / "bots", False) | {"code_sha256": m["code_sha256"]}
        else:
            opponents[name] = base | source(Path(m["resolution"]["path"]), out_dir / "roster") | {
                "source_sha256": m["source_sha256"], "resolution": m["resolution"]}
    frozen = {"schema": SCHEMA_FROZEN, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "registry": strip_private(reg), "pool_identity": reg["pool_identity"],
              "schedule": {"path": rel(schedule_path), "sha256": O.sha256_file(schedule_path),
                           "identity": sched["schedule_identity"]},
              "designated": d | designated, "opponents": opponents,
              "compiler": compiler or compiler_identity(), "engine": engine or engine_identity(root),
              "runner": runner_identity(), "limits_ms": [int(LIMITS[0] * 1000), int(LIMITS[1] * 1000)],
              "platform": platform.platform()}
    frozen["identity"] = frozen_identity(frozen)
    write_json(out_dir / "FROZEN.json", frozen)
    print(f"[decision] frozen {frozen['identity'][:16]}: designated {d['variant']} {designated['binary_sha256'][:12]}, "
          f"{len(opponents)} opponents, schedule {sched['schedule_identity'][:16]}")
    return frozen


def cmd_prepare(a):
    root = Path(a.ai_test) if a.ai_test else Path(os.environ.get("AI_TEST_ROOT", ""))
    if not a.ai_test and not os.environ.get("AI_TEST_ROOT"):
        raise SystemExit("prepare pins the engine: pass --ai-test or set AI_TEST_ROOT")
    prepare(Path(a.schedule), Path(a.out_dir), root)


# ---------------------------------------------------------------- verification and collection

def verify(frozen: dict, root: Path | None, engine=None, registry=None) -> list[str]:
    """Every difference between the frozen identities and the present ones (empty: safe to collect)."""
    bad = []
    if frozen.get("schema") != SCHEMA_FROZEN or frozen_identity(frozen) != frozen.get("identity"):
        return ["FROZEN.json was edited after it was written"]
    if tuple(x / 1000 for x in frozen["limits_ms"]) != LIMITS:
        bad.append("limits are not the official 2000/50 ms")
    if runner_identity() != frozen["runner"]:
        bad.append("collector or runner code changed")
    sp = REPO / frozen["schedule"]["path"] if not Path(frozen["schedule"]["path"]).is_absolute() else Path(frozen["schedule"]["path"])
    if not sp.is_file() or O.sha256_file(sp) != frozen["schedule"]["sha256"]:
        bad.append("schedule file changed or missing")
    for name, b in [("@candidate", frozen["designated"]), *frozen["opponents"].items()]:
        p = REPO / b["binary"] if not Path(b["binary"]).is_absolute() else Path(b["binary"])
        if not p.is_file() or O.sha256_file(p) != b["binary_sha256"]:
            bad.append(f"binary of {name} changed or missing")
    reg = registry or O.build_registry(root)
    if reg["pool_identity"] != frozen["pool_identity"]:
        bad.append("registry pool identity changed (a source, alias, exclusion or quarantine)")
    eng = engine or engine_identity(root)
    if eng != frozen["engine"]:
        bad.append("engine changed")
    return bad


def fixture_seats(fx: dict) -> tuple[list[str], list[str]]:
    seats = list(fx["opponents"])
    seats.insert(fx["seat"], CANDIDATE)
    comps = [f"{c}|{g}|{l}" for c, g, l in zip(fx["components"], fx["groups"], fx["lineages"])]
    comps.insert(fx["seat"], "designated")
    if seats.count(CANDIDATE) != 1 or len(set(seats)) != len(seats) or len(seats) != fx["teams"]:
        raise SystemExit(f"{fx['fixture_id']}: needs exactly one {CANDIDATE} and distinct opponents for {fx['teams']} seats")
    return seats, comps


def commands_and_info(frozen: dict) -> tuple[dict, dict]:
    absolute = lambda p: str(REPO / p) if not Path(p).is_absolute() else p  # noqa: E731
    commands = {CANDIDATE: [absolute(frozen["designated"]["binary"])]}
    info = {CANDIDATE: {"family": "candidate", "sha256": frozen["designated"]["binary_sha256"]}}
    for name, b in frozen["opponents"].items():
        commands[name] = [absolute(b["binary"])]
        info[name] = {"family": b["family"], "sha256": b.get("source_sha256", b["binary_sha256"])}
    info["__engine__"], info["__commit__"] = frozen["engine"]["sha256"], frozen["engine"]["commit"]
    return commands, info


def match_path(out: Path, fx: dict) -> Path:
    return Path(out) / "matches" / f"n{fx['teams']:02d}" / f"{fx['fixture_id']}.json.gz"


def complete(out: Path, fx: dict, identity: str) -> bool:
    """True only for a row with the frozen identity whose match file has the recorded hash and the fixture's seats."""
    row_p = Path(out) / "rows" / f"{fx['fixture_id']}.json"
    if not row_p.is_file():
        return False
    row = json.loads(row_p.read_text(encoding="utf-8"))
    if row.get("identity") != identity:
        raise SystemExit(f"{row_p}: a row of another frozen identity ({str(row.get('identity'))[:16]}); "
                         "stale results cannot be resumed: collect into a new --out")
    mp = match_path(out, fx)
    if not mp.is_file() or O.sha256_file(mp) != row.get("match_sha256"):
        return False
    from fusion import matchlog  # noqa: PLC0415
    m = matchlog.load(mp)
    seats, _ = fixture_seats(fx)
    return m["match_id"] == fx["fixture_id"] and [s["name"] for s in m["seats"]] == seats


def play_fixture(job) -> dict:
    fx, commands, info, root, pool_identity, play_fn = job
    from training.collect import Fixture, play  # noqa: PLC0415
    seats, comps = fixture_seats(fx)
    fixture = Fixture(fx["fixture_id"], fx["teams"], fx["map_seed"], fx["flag_seed"], seats, regime="decision-expanded",
                      components=comps, registry=pool_identity)
    try:
        return {"match": (play_fn or play)((fixture, commands, info, root, LIMITS))}
    except Exception:  # noqa: BLE001  (recorded, never hidden)
        return {"error": traceback.format_exc()}


def collect(frozen_path: Path, out: Path, root: Path | None, workers: int = 1, play_fn=None, engine=None,
            registry=None, dry_run: bool = False) -> dict:
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    bad = verify(frozen, root, engine, registry)
    if bad:
        raise SystemExit("refusing to collect: " + "; ".join(bad))
    sp = Path(frozen["schedule"]["path"])
    sched = json.loads((sp if sp.is_absolute() else REPO / sp).read_text(encoding="utf-8"))
    out = Path(out)
    ident = frozen["identity"]
    todo = []
    for fx in sched["fixtures"]:
        if complete(out, fx, ident):
            continue
        mp = match_path(out, fx)
        if mp.exists():   # a match without a valid row: keep it for inspection, never reuse it
            dest = out / "partial" / f"{fx['fixture_id']}.{int(time.time())}.json.gz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mp), dest)
        todo.append(fx)
    print(f"[decision] {len(sched['fixtures'])} fixtures, {len(sched['fixtures']) - len(todo)} complete, {len(todo)} to play "
          f"(frozen {ident[:16]})", flush=True)
    if dry_run or not todo:
        return {"todo": [f["fixture_id"] for f in todo]}
    lock = lock_path(out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: another collection is running into {out}") from None
    os.write(fd, json.dumps({"pid": os.getpid(), "frozen": ident, "started": time.time()}).encode())
    os.close(fd)
    commands, info = commands_and_info(frozen)
    from fusion import matchlog  # noqa: PLC0415
    from fusion.labels import pairwise_outcome  # noqa: PLC0415
    jobs = [(fx, commands, info, str(root) if root else None, frozen["pool_identity"], play_fn) for fx in todo]
    done = errors = 0
    try:
        pool = ProcessPoolExecutor(workers) if workers > 1 and play_fn is None else None
        results = pool.map(play_fixture, jobs, chunksize=1) if pool else map(play_fixture, jobs)
        for fx, res in zip(todo, results):
            if "error" in res:
                write_json(out / "errors" / f"{fx['fixture_id']}.json", {"fixture": fx, "identity": ident, "error": res["error"]})
                errors += 1
                continue
            match = res["match"]
            mp = match_path(out, fx)
            matchlog.save(mp, match)
            seat = fx["seat"]
            score = match["result"]["score"]
            row = {"schema": SCHEMA_ROW, "identity": ident, "fixture_id": fx["fixture_id"], "teams": fx["teams"],
                   "map_seed": fx["map_seed"], "seat": seat, "match_sha256": O.sha256_file(mp), "score": score,
                   "utility": float(pairwise_outcome(score)[seat]), "engine_sha256": match["engine_sha256"],
                   "limits_ms": match["limits_ms"], "seats": [s["name"] for s in match["seats"]],
                   "components": match["sampling"]["components"],
                   "faults": [{"seat": i, "name": b["name"], "faults": b["faults"], "late": b["late"], "dead": b["dead"]}
                              for i, b in enumerate(match["bots"])]}
            write_json(out / "rows" / f"{fx['fixture_id']}.json", row)
            done += 1
            if done % 50 == 0:
                print(f"[decision] {done}/{len(todo)} played, {errors} errors", flush=True)
        if pool:
            pool.shutdown()
    finally:
        lock.unlink(missing_ok=True)
    print(f"[decision] played {done}, errors {errors} (error records in {out / 'errors'})")
    return {"played": done, "errors": errors}


def cmd_collect(a):
    root = Path(a.ai_test) if a.ai_test else (Path(os.environ["AI_TEST_ROOT"]) if os.environ.get("AI_TEST_ROOT") else None)
    if root is None and not a.dry_run:
        raise SystemExit("collect plays the engine: pass --ai-test or set AI_TEST_ROOT")
    if root is not None:
        os.environ["AI_TEST_ROOT"] = str(root)
    collect(Path(a.frozen), Path(a.out), root, a.workers, dry_run=a.dry_run)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="registry availability (no build, no game)")
    s.add_argument("--out", required=True)
    s.add_argument("--simulate", type=int, default=0, help="games per count to simulate realized seat shares")
    s.add_argument("--ai-test")
    s = sub.add_parser("schedule", help="deterministic fixtures on an unused seed")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--maps-per-count", type=int, required=True)
    s.add_argument("--teams", type=int, nargs="*", default=list(O.TEAM_COUNTS))
    s.add_argument("--out", required=True)
    s.add_argument("--ai-test")
    s = sub.add_parser("prepare", help="build and freeze every identity (host)")
    s.add_argument("--schedule", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--ai-test")
    s = sub.add_parser("collect", help="play the frozen schedule (resumable)")
    s.add_argument("--frozen", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--ai-test")
    s.add_argument("--dry-run", action="store_true", help="verify identities and list what is left; play nothing")
    a = ap.parse_args(argv)
    {"list": cmd_list, "schedule": cmd_schedule, "prepare": cmd_prepare, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
