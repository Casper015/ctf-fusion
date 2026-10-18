"""Resumable, input-bound branch collection (task T0085).

    python tools/repo/task.py lock cpu --task T0086 -- python -m tools.search_diagnostics.cli collect \\
        --roots ROOTS --matches SOURCES --build build/search_diagnostics/p2general.json --set primary --repeats 4 \\
        --scenario-seed <reserved> --seed-ledger docs/SEARCH_BRANCH_DIAGNOSTIC_SEEDS.md --workers 8 --out ROWS

The first run writes <out>/MANIFEST.json, the collector's identity: its source files' hashes, the package build,
the engine, the limits, the scenario seed with its ledger, the candidate set, the repeats and, for an extended set,
the primary collection it reuses. A later run with another identity is refused, and the manifest is never rewritten.

Seeds. On a fresh start the seed must be reserved in the ledger (seeds.check_seed), and no other collection under
the same parent directory may already use it (an extended set's own primary excepted).

Rows. There is one row per root (<out>/nXX/<root_id>.json.gz). It holds every (candidate, repeat) branch with its
keys and outcome or refusal reason, and it is bound to the root record's hash, the candidate set and the registered
keys. Resume keeps a stored row only when it validates against all of that. A row that does not is a stop, never a
replay: its scenarios are consumed.

Extended set. The 24-candidate subset reuses the primary collection's verified branches (same package, engine, seed
and repeats, hence the same keys) and plays only the candidates beyond the primary set. T0086's cap counts only those.

Evidence runs use exactly eight workers (AGENTS.md). --dev allows another count and marks the manifest as not
evidence.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fusion import matchlog
from fusion.paths import REPO, engine_path, load_engine, sha256
from tools.search_diagnostics import seeds
from tools.search_diagnostics.branch import LIMITS, commands_for, play_root
from tools.search_diagnostics.build import binary_path
from tools.search_diagnostics.schemas import (
    PROTOCOL, SCHEMA_MANIFEST, SCHEMA_ROW, ValidationError, digest, read_gz, sha256_file, validate_branch, validate_root,
    verify_leaves, write_gz,
)

WORKERS = 8
COLLECTOR_FILES = ("tools/search_diagnostics/collect.py", "tools/search_diagnostics/branch.py",
                   "tools/search_diagnostics/extract.py", "tools/search_diagnostics/seeds.py",
                   "tools/search_diagnostics/schemas.py", "fusion/runner.py", "fusion/labels.py")


def identity(build: dict, ai_test: Path, cand_set: str, repeats: int, seed: int, ledger: Path, dev: bool,
             reuse: dict | None) -> dict:
    return {
        "schema": SCHEMA_MANIFEST,
        "collector_files": {f: sha256_file(REPO / f) for f in COLLECTOR_FILES},
        "package": build["continuation_bot"]["package"],
        "continuation_bot": {k: build["continuation_bot"][k] for k in ("binary_sha256", "bundle_sha256")},
        "root_tool": {k: build["root_tool"][k] for k in ("binary_sha256", "bundle_sha256")},
        "engine_sha256": sha256(engine_path(ai_test)),
        "limits_ms": [1000 * LIMITS[0], 1000 * LIMITS[1]], "protocol": PROTOCOL,
        "candidate_set": cand_set, "repeats": repeats, "scenario_seed": seed, "seed_ledger": Path(ledger).as_posix(),
        "evidence": not dev, "reuses": reuse,
    }


def load_manifest(out: Path) -> dict:
    p = Path(out) / "MANIFEST.json"
    if not p.is_file():
        raise ValidationError(f"{p} not found: not a collection")
    man = json.loads(p.read_text(encoding="utf-8"))
    body = {k: v for k, v in man.items() if k != "identity_digest"}
    if man.get("schema") != SCHEMA_MANIFEST or digest(body) != man.get("identity_digest"):
        raise ValidationError(f"{p} does not match its identity digest")
    return man


def row_path(out: Path, root: dict) -> Path:
    return Path(out) / f"n{root['source']['teams']:02d}" / f"{root['root_id']}.json.gz"


def binding(root: dict, cand_set: str, repeats: int) -> dict:
    return {"root_sha256": digest(root), "candidate_set": cand_set, "candidates": root["candidate_sets"][cand_set],
            "repeats": repeats}


def validate_row(row: dict, ident: str, root: dict, cand_set: str, repeats: int, seed: int) -> dict:
    """A row, checked against the collection identity, its root record, the candidate set and the registered keys."""
    w = f"row {row.get('root_id', '?') if isinstance(row, dict) else '?'}"
    if not isinstance(row, dict) or row.get("schema") != SCHEMA_ROW:
        raise ValidationError(f"{w}: schema is not {SCHEMA_ROW}")
    if row.get("identity") != ident:
        raise ValidationError(f"{w}: another collector identity")
    if row.get("root_id") != root["root_id"]:
        raise ValidationError(f"{w}: bound to root {row.get('root_id')}, not {root['root_id']}")
    if row.get("binding") != binding(root, cand_set, repeats):
        raise ValidationError(f"{w}: its binding (root hash, candidate set, repeats) does not match")
    keys = seeds.repeat_keys(seed, root["root_id"], repeats)
    if row.get("keys") != keys:
        raise ValidationError(f"{w}: keys are not the registered ones")
    branches = row.get("branches")
    if not isinstance(branches, list):
        raise ValidationError(f"{w}: no branches")
    seen = set()
    teams, seat = root["source"]["teams"], root["source"]["seat"]
    for i, b in enumerate(branches):
        validate_branch(b, f"{w} branch {i}", teams, seat)
        k = (b["candidate"], b["repeat"])
        if k in seen:
            raise ValidationError(f"{w}: duplicate branch candidate {k[0]} repeat {k[1]}")
        seen.add(k)
        if not 0 <= b["repeat"] < repeats or (b["env_key"], b["actor_key"]) != (keys[b["repeat"]]["env_key"],
                                                                               keys[b["repeat"]]["actor_key"]):
            raise ValidationError(f"{w}: branch {i} does not use its repeat's registered keys")
        if b["status"] == "complete" and b.get("protocol") != PROTOCOL:
            raise ValidationError(f"{w}: branch {i} was played under protocol {b.get('protocol')!r}")
    want = {(c, r) for c in root["candidate_sets"][cand_set] for r in range(repeats)}
    if seen != want:
        raise ValidationError(f"{w}: branches do not cover the bound candidates x repeats")
    return row


def load_roots(roots_dir: Path) -> dict[str, dict]:
    roots = {}
    for p in sorted(Path(roots_dir).rglob("*.json.gz")):
        r = validate_root(read_gz(p))
        if r["root_id"] in roots:
            raise ValidationError(f"duplicate root {r['root_id']}")
        verify_leaves(r, p.parent / r["leaf_features"]["file"])
        roots[r["root_id"]] = r
    if not roots:
        raise ValidationError(f"no roots under {roots_dir}")
    return roots


def _job(job) -> dict:
    root, match_path, cand_set, play, repeats, seed, build, ai_test, ident, reused = job
    engine = load_engine(Path(ai_test))
    match = matchlog.load(match_path)
    if sha256_file(match_path) != root["source"]["match_sha256"]:
        raise ValidationError(f"{match_path}: changed since root {root['root_id']} was extracted")
    keys = seeds.repeat_keys(seed, root["root_id"], repeats)
    bot = binary_path(build["continuation_bot"])
    commands = commands_for(match, root["source"]["seat"], bot, Path(ai_test), REPO / "build" / "roster")
    branches = (reused["branches"] if reused else []) + play_root(engine, match, root, play, keys, commands)
    return {"schema": SCHEMA_ROW, "identity": ident, "root_id": root["root_id"],
            "binding": binding(root, cand_set, repeats), "keys": keys, "branches": branches,
            "reused": {"row_sha256": digest(reused), "candidates": sorted({b["candidate"] for b in reused["branches"]})}
            if reused else None}


def collect(roots_dir: Path, matches_dir: Path, build: dict, out: Path, ai_test: Path, cand_set: str, repeats: int,
            seed: int, ledger: Path, workers: int, dev: bool = False, reuse_dir: Path | None = None) -> dict:
    if workers != WORKERS and not dev:
        raise ValidationError(f"evidence collection runs on exactly {WORKERS} workers; {workers} needs --dev")
    if repeats < 1:
        raise ValidationError("repeats must be >= 1")
    out = Path(out)
    primary = None
    reuse = None
    if cand_set != "primary":
        if reuse_dir is None:
            raise ValidationError(f"the {cand_set} set reuses a primary collection: pass --reuse")
        primary = load_manifest(reuse_dir)
        for k, want in (("repeats", repeats), ("scenario_seed", seed), ("evidence", not dev),
                        ("limits_ms", [1000 * LIMITS[0], 1000 * LIMITS[1]]), ("protocol", PROTOCOL)):
            if primary[k] != want:
                raise ValidationError(f"the primary collection's {k} is {primary[k]!r}, not {want!r}")
        if primary["candidate_set"] != "primary":
            raise ValidationError("--reuse must name a primary collection")
        reuse = {"dir": Path(reuse_dir).as_posix(), "identity_digest": primary["identity_digest"]}
    ident_obj = identity(build, ai_test, cand_set, repeats, seed, ledger, dev, reuse)
    if primary:
        for k in ("package", "continuation_bot", "root_tool", "engine_sha256", "collector_files"):
            if primary[k] != ident_obj[k]:
                raise ValidationError(f"the primary collection's {k} differs from this run's")
    ident = digest(ident_obj)
    man = out / "MANIFEST.json"
    if man.exists():
        if load_manifest(out)["identity_digest"] != ident:
            raise ValidationError(f"{man} has another collector identity: start a new output directory")
    else:
        seeds.check_seed(seed, ledger)
        for other in sorted(out.parent.glob("*/MANIFEST.json")):
            if other.parent.resolve() in {out.resolve(), Path(reuse_dir).resolve() if reuse_dir else None}:
                continue
            if json.loads(other.read_text(encoding="utf-8")).get("scenario_seed") == seed:
                raise ValidationError(f"scenario seed {seed} is already consumed by {other.parent}")
        out.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps({**ident_obj, "identity_digest": ident}, indent=1) + "\n", encoding="utf-8")
    roots = load_roots(roots_dir)
    seeds.check_unique([k for r in roots.values() for k in seeds.repeat_keys(seed, r["root_id"], repeats)])
    todo, kept = [], 0
    for r in roots.values():
        if r["tool"]["binary_sha256"] != build["root_tool"]["binary_sha256"]:
            raise ValidationError(f"root {r['root_id']} was extracted by another root tool")
        if cand_set not in r["candidate_sets"]:
            raise ValidationError(f"root {r['root_id']} has no candidate set {cand_set}")
        p = row_path(out, r)
        if p.exists():
            validate_row(read_gz(p), ident, r, cand_set, repeats, seed)
            kept += 1
            continue
        reused, play = None, r["candidate_sets"][cand_set]
        if primary:
            q = row_path(reuse_dir, r)
            if not q.exists():
                raise ValidationError(f"root {r['root_id']} has no row in the primary collection")
            reused = validate_row(read_gz(q), primary["identity_digest"], r, "primary", repeats, seed)
            play = [c for c in play if c not in r["candidate_sets"]["primary"]]
        match_path = Path(matches_dir) / f"n{r['source']['teams']:02d}" / r["source"]["match_file"]
        todo.append((r, match_path, cand_set, play, repeats, seed, build, str(ai_test), ident, reused))
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for row in ex.map(_job, todo):
            r = roots[row["root_id"]]
            validate_row(row, ident, r, cand_set, repeats, seed)
            write_gz(row_path(out, r), row)
            done += 1
    return {"roots": len(roots), "kept": kept, "played": done, "identity": ident,
            "episodes_played": sum(len(j[3]) for j in todo) * repeats}
