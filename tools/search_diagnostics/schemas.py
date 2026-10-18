"""Record schemas and validation of the agent_v2 branch diagnostics (task T0085).

Every record is plain JSON. The validators below fail closed on what they check (the fields each one names, their
types, finiteness, and the bindings between records): a violation raises ValidationError, and nothing is filled in
with a default (missing evidence is never zero).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA_ROOT = "ctf-diag-root-2"
SCHEMA_ROW = "ctf-diag-branch-row-2"
SCHEMA_MANIFEST = "ctf-diag-collection-manifest-2"
SCHEMA_ANALYSIS = "ctf-diag-analysis-2"
SCHEMA_PARITY = "ctf-diag-parity-2"
SCHEMA_P_FIDELITY = "ctf-diag-p-fidelity-2"
SCHEMA_ESTIMATE = "ctf-diag-estimate-2"
NATIVE_ROOT = "ctf-diag-native-root-1"
NATIVE_P = "ctf-diag-native-p-1"
PROTOCOL = "faithful-replay-v1"


class ValidationError(ValueError):
    """Input or a record fails an integrity, schema or contract check."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj: Any) -> str:
    """The SHA-256 of a JSON value in canonical form."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def write_gz(path: Path, obj: Any) -> None:
    """Atomic gzip JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def read_gz(path: Path) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValidationError(f"{path}: unreadable ({e})") from None


def need(obj: dict, key: str, kind, where: str):
    """obj[key], which must exist and have type `kind` (a bool is never accepted as a number)."""
    if not isinstance(obj, dict) or key not in obj:
        raise ValidationError(f"{where}: missing {key}")
    v = obj[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if isinstance(v, bool) and bool not in kinds:
        raise ValidationError(f"{where}: {key} is a bool")
    if not isinstance(v, kinds):
        raise ValidationError(f"{where}: {key} is {type(v).__name__}, expected {'/'.join(k.__name__ for k in kinds)}")
    if isinstance(v, float) and not math.isfinite(v):
        raise ValidationError(f"{where}: {key} is not finite")
    return v


def finite(values, where: str) -> list[float]:
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValidationError(f"{where}: {v!r} is not a finite number")
        out.append(float(v))
    return out


def validate_root(root: dict) -> dict:
    """A root record (extract.py); returns it."""
    w = f"root {root.get('root_id', '?') if isinstance(root, dict) else '?'}"
    if not isinstance(root, dict) or root.get("schema") != SCHEMA_ROOT:
        raise ValidationError(f"{w}: schema is not {SCHEMA_ROOT}")
    need(root, "root_id", str, w)
    src = need(root, "source", dict, w)
    for k, kind in (("match_id", str), ("match_sha256", str), ("teams", int), ("map_seed", int), ("flag_seed", int),
                    ("seat", int), ("seat_sha256", str), ("engine_sha256", str)):
        need(src, k, kind, w + " source")
    turn = need(root, "turn", int, w)
    if not 2 <= turn <= 399:
        raise ValidationError(f"{w}: turn {turn} out of range")
    sel = need(root, "selection", dict, w)
    need(sel, "stratum", str, w + " selection")
    need(sel, "rule", str, w + " selection")
    if need(sel, "weight", (int, float), w + " selection") <= 0:
        raise ValidationError(f"{w}: selection weight must be positive")
    hist = need(root, "history", dict, w)
    for k in ("seat_stream", "seat_stream_sha256", "prefix_digest", "root_block_sha256"):
        need(hist, k, str, w + " history")
    if sha256_text(hist["seat_stream"]) != hist["seat_stream_sha256"]:
        raise ValidationError(f"{w}: seat stream does not match its hash")
    tool = need(root, "tool", dict, w)
    need(tool, "binary_sha256", str, w + " tool")
    need(tool, "package", str, w + " tool")
    tree = need(tool, "tree", list, w + " tool")
    if len(tree) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in tree):
        raise ValidationError(f"{w}: tool tree {tree!r}")
    samples = tree[1]
    model = need(root, "model", dict, w)
    for k in ("carrier", "actor", "actor_digest", "predictor_kind", "predictor", "critic", "resolver"):
        need(model, k, str, w + " model")
    policy = need(root, "policy", dict, w)
    for k in ("seed", "state_key", "parent_key", "policy_feature_fnv", "mask", "tree_T"):
        need(policy, k, str, w + " policy")
    need(policy, "sampled_joint", int, w + " policy")
    leaves = need(root, "leaf_features", dict, w)
    for k in ("file", "sha256"):
        need(leaves, k, str, w + " leaf_features")
    n_rows = need(leaves, "rows", int, w + " leaf_features")
    if need(leaves, "dim", int, w + " leaf_features") != 416:
        raise ValidationError(f"{w}: leaf feature dim {leaves['dim']}")
    base = need(root, "baseline", dict, w)
    need(base, "order", str, w + " baseline")
    if base.get("verified") is not True:
        raise ValidationError(f"{w}: baseline identity not verified")
    cands = need(root, "candidates", list, w)
    if len(cands) < 2:
        raise ValidationError(f"{w}: fewer than two candidates")
    seen, rows_seen = set(), []
    for i, c in enumerate(cands):
        cw = f"{w} candidate {i}"
        if need(c, "index", int, cw) != i:
            raise ValidationError(f"{cw}: index {c['index']}")
        order = need(c, "order", str, cw)
        if order in seen:
            raise ValidationError(f"{cw}: duplicate order {order!r}")
        seen.add(order)
        need(c, "joint", int, cw)
        vals = finite(need(c, "v_values", list, cw), cw + " v_values")
        term = need(c, "v_terminal", list, cw)
        at = need(c, "leaf_rows", list, cw)
        if not (len(vals) == len(term) == len(at) == samples):
            raise ValidationError(f"{cw}: {len(vals)} values, {len(term)} terminal flags, {len(at)} leaf rows for {samples} samples")
        for v, x, r in zip(vals, term, at):
            if x not in (0, 1) or isinstance(r, bool) or not isinstance(r, int) or (r == -1) != (x == 1):
                raise ValidationError(f"{cw}: leaf row {r!r} does not match terminal flag {x!r}")
        rows_seen.extend(r for r in at if r >= 0)
    if sorted(rows_seen) != list(range(n_rows)):
        raise ValidationError(f"{w}: leaf rows are not 0 .. {n_rows - 1}, each once")
    if cands[0]["order"] != base["order"] or cands[0]["joint"] != policy["sampled_joint"]:
        raise ValidationError(f"{w}: candidate 0 is not the baseline")
    sets = need(root, "candidate_sets", dict, w)
    for name, idx in sets.items():
        if not isinstance(idx, list) or not idx or idx != list(range(len(idx))) or len(idx) > len(cands):
            raise ValidationError(f"{w}: candidate set {name} is not a prefix starting at the baseline")
    return root


def verify_leaves(root: dict, path: Path) -> None:
    """The root's leaf-feature sidecar must exist with the recorded hash and size."""
    lf = root["leaf_features"]
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"root {root['root_id']}: leaf features {path} missing")
    if path.stat().st_size != lf["rows"] * lf["dim"] * 4 or sha256_file(path) != lf["sha256"]:
        raise ValidationError(f"root {root['root_id']}: leaf features {path} changed")


def validate_branch(b: dict, where: str, teams: int | None = None, seat: int | None = None) -> dict:
    """One branch result inside a row; with teams and seat, the score's length and the utility it implies."""
    for k, kind in (("candidate", int), ("repeat", int), ("env_key", int), ("actor_key", int), ("status", str)):
        need(b, k, kind, where)
    if b["status"] == "complete":
        score = finite(need(b, "final_score", list, where), where + " final_score")
        u = need(b, "utility", (int, float), where)
        if not 0.0 <= u <= 1.0:
            raise ValidationError(f"{where}: utility {u} outside [0, 1]")
        if teams is not None:
            if len(score) != teams or not 0 <= seat < teams:
                raise ValidationError(f"{where}: {len(score)} scores for {teams} teams")
            want = sum(1.0 if score[seat] > x else 0.5 if score[seat] == x else 0.0
                       for i, x in enumerate(score) if i != seat) / (teams - 1)
            if abs(u - want) > 1e-9:
                raise ValidationError(f"{where}: utility {u} is not the score's pairwise outcome {want}")
        if need(b, "final_turn", int, where) != 400 or b.get("terminal") is not True:
            raise ValidationError(f"{where}: not played to the end")
        faults = need(b, "faults", dict, where)
        for k in ("prefix", "root", "continuation"):
            need(faults, k, list, where + " faults")
        if faults["prefix"] or faults["root"]:
            raise ValidationError(f"{where}: a complete branch with prefix or root faults")
        need(b, "wall_s", (int, float), where)
    elif b["status"] == "refused":
        need(b, "reason", str, where)
    else:
        raise ValidationError(f"{where}: status {b['status']!r}")
    return b
