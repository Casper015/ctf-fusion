"""Resumable, input-bound branch collection (task T0085).

    python tools/repo/task.py lock cpu --task T0086 -- python -m tools.search_diagnostics.cli collect \\
        --roots data/search_diagnostics/roots --matches data/search_diagnostics/source --package-build build/search_diagnostics/p2general.json \\
        --set primary --repeats 4 --scenario-seed <reserved> --seed-ledger docs/SEARCH_BRANCH_DIAGNOSTIC_SEEDS.md \\
        --workers 8 --out data/search_diagnostics/branches --ai-test $AI_TEST_ROOT

The first run writes <out>/MANIFEST.json: the collector's identity (its source files' hashes, the package build,
the engine, the limits, the scenario seed and its ledger, the candidate set, the repeats). A later run with another
identity is refused; the manifest is never rewritten. One row per root (<out>/nXX/<root_id>.json.gz) holds every
(candidate, repeat) branch, its keys and its outcome or refusal reason, bound to the root record's hash. Resume keeps
a stored row only when its identity and binding are equal; any other row is moved to <out>/stale/ and replayed.
Evidence runs use exactly eight workers (AGENTS.md); --dev allows another count and marks the manifest as not
evidence.
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fusion import matchlog
from fusion.paths import REPO, engine_path, load_engine, sha256
from tools.search_diagnostics import seeds
from tools.search_diagnostics.branch import LIMITS, commands_for, play_root
from tools.search_diagnostics.build import binary_path
from tools.search_diagnostics.schemas import (
    SCHEMA_MANIFEST, SCHEMA_ROW, ValidationError, digest, read_gz, sha256_file, validate_branch, validate_root, write_gz,
)

WORKERS = 8
COLLECTOR_FILES = ("tools/search_diagnostics/collect.py", "tools/search_diagnostics/branch.py",
                   "tools/search_diagnostics/extract.py", "tools/search_diagnostics/seeds.py",
                   "tools/search_diagnostics/schemas.py", "fusion/runner.py", "fusion/labels.py")


def identity(build: dict, ai_test: Path, cand_set: str, repeats: int, seed_info: dict, dev: bool) -> dict:
    return {
        "schema": SCHEMA_MANIFEST,
        "collector_files": {f: sha256_file(REPO / f) for f in COLLECTOR_FILES},
        "package": build["continuation_bot"]["package"],
        "continuation_bot": {k: build["continuation_bot"][k] for k in ("binary_sha256", "bundle_sha256")},
        "root_tool": {k: build["root_tool"][k] for k in ("binary_sha256", "bundle_sha256")},
        "engine_sha256": sha256(engine_path(ai_test)),
        "limits_ms": [1000 * LIMITS[0], 1000 * LIMITS[1]],
        "candidate_set": cand_set, "repeats": repeats,
        "scenario_seed": seed_info["seed"], "seed_ledger": seed_info["ledger"],
        "seed_ledger_sha256": sha256_file(seed_info["ledger"]),
        "evidence": not dev,
    }


def row_path(out: Path, root: dict) -> Path:
    return Path(out) / f"n{root['source']['teams']:02d}" / f"{root['root_id']}.json.gz"


def binding(root: dict, cand_set: str, repeats: int) -> dict:
    return {"root_sha256": digest(root), "candidates": root["candidate_sets"][cand_set], "repeats": repeats}


def validate_row(row: dict, ident: str | None = None) -> dict:
    w = f"row {row.get('root_id', '?') if isinstance(row, dict) else '?'}"
    if not isinstance(row, dict) or row.get("schema") != SCHEMA_ROW:
        raise ValidationError(f"{w}: schema is not {SCHEMA_ROW}")
    if ident is not None and row.get("identity") != ident:
        raise ValidationError(f"{w}: another collector identity")
    bind = row.get("binding") or {}
    branches = row.get("branches")
    if not isinstance(branches, list):
        raise ValidationError(f"{w}: no branches")
    seen = set()
    for i, b in enumerate(branches):
        validate_branch(b, f"{w} branch {i}")
        k = (b["candidate"], b["repeat"])
        if k in seen:
            raise ValidationError(f"{w}: duplicate branch candidate {k[0]} repeat {k[1]}")
        seen.add(k)
    want = {(c, r) for c in bind.get("candidates", []) for r in range(bind.get("repeats", 0))}
    if seen != want:
        raise ValidationError(f"{w}: branches do not cover the bound candidates x repeats")
    return row


def _job(job) -> dict:
    root, match_path, cand_set, repeats, seed, build, ai_test, ident = job
    engine = load_engine(Path(ai_test))
    match = matchlog.load(match_path)
    if sha256_file(match_path) != root["source"]["match_sha256"]:
        raise ValidationError(f"{match_path}: changed since root {root['root_id']} was extracted")
    keys = seeds.repeat_keys(seed, root["root_id"], repeats)
    bot = binary_path(build["continuation_bot"])
    commands = commands_for(match, root["source"]["seat"], bot, Path(ai_test), REPO / "build" / "roster")
    branches = play_root(engine, match, root, root["candidate_sets"][cand_set], keys, commands)
    return {"schema": SCHEMA_ROW, "identity": ident, "root_id": root["root_id"],
            "binding": binding(root, cand_set, repeats), "keys": keys, "branches": branches}


def collect(roots_dir: Path, matches_dir: Path, build: dict, out: Path, ai_test: Path, cand_set: str, repeats: int,
            seed: int, ledger: Path, workers: int, dev: bool = False) -> dict:
    if workers != WORKERS and not dev:
        raise ValidationError(f"evidence collection runs on exactly {WORKERS} workers; {workers} needs --dev")
    if repeats < 1:
        raise ValidationError("repeats must be >= 1")
    seed_info = seeds.check_seed(seed, ledger)
    ident_obj = identity(build, ai_test, cand_set, repeats, seed_info, dev)
    ident = digest(ident_obj)
    out = Path(out)
    man = out / "MANIFEST.json"
    if man.exists():
        old = json.loads(man.read_text(encoding="utf-8"))
        if old.get("identity_digest") != ident:
            raise ValidationError(f"{man} has another collector identity: start a new output directory")
    else:
        out.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps({**ident_obj, "identity_digest": ident}, indent=1) + "\n", encoding="utf-8")
    roots = [validate_root(read_gz(p)) for p in sorted(Path(roots_dir).rglob("*.json.gz"))]
    if not roots:
        raise ValidationError(f"no roots under {roots_dir}")
    ids = [r["root_id"] for r in roots]
    if len(set(ids)) != len(ids):
        raise ValidationError("duplicate root ids")
    all_keys = [k for r in roots for k in seeds.repeat_keys(seed, r["root_id"], repeats)]
    seeds.check_unique(all_keys)
    for r in roots:
        if r["tool"]["binary_sha256"] != build["root_tool"]["binary_sha256"]:
            raise ValidationError(f"root {r['root_id']} was extracted by another root tool")
        if cand_set not in r["candidate_sets"]:
            raise ValidationError(f"root {r['root_id']} has no candidate set {cand_set}")
    todo, kept, stale = [], 0, 0
    for r in roots:
        p = row_path(out, r)
        if p.exists():
            try:
                row = validate_row(read_gz(p), ident)
                if row["binding"] == binding(r, cand_set, repeats):
                    kept += 1
                    continue
            except ValidationError:
                pass
            dest = out / "stale" / p.relative_to(out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            stale += 1
        match_path = Path(matches_dir) / f"n{r['source']['teams']:02d}" / r["source"]["match_file"]
        todo.append((r, match_path, cand_set, repeats, seed, build, str(ai_test), ident))
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for row in ex.map(_job, todo):
            validate_row(row, ident)
            write_gz(row_path(out, next(r for r in roots if r["root_id"] == row["root_id"])), row)
            done += 1
    return {"roots": len(roots), "kept": kept, "stale": stale, "played": done, "identity": ident}
