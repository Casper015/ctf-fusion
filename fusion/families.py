"""Feature families: which encoder output a model consumes, versioned apart from the record layout.

A model never depends on the whole extractor record, only on the feature groups its family reads:

    p2  legacy opponent policy   inference/features.hpp encodePolicy (mover/act 40, squad 222, target 7)
    p3  policy + combat context  inference/policy_p3.hpp encodePolicyP3 (mover/act 55, squad 447, target 7)
    v2  legacy board value       inference/features.hpp encodeValue (54)
    v3  value + extension        inference/value_v3.hpp encodeValueV3 (108)

A family's identity is a digest of exactly what it reads: the group specs (names, bounds, scales, in order),
the block shapes and the head-input composition, plus the combat-summary version for p3/v3. Offsets and other
families' features are not part of it, so a new value family (or a wider record layout) leaves every policy
artifact valid, and vice versa. The record layout is versioned separately (`layout_version` in the schema; a
schema without it is layout 2, which predates p3/v3).

Checks (fail closed, never weakened to pass):
  - a training run records `features: {family, digest, layout_version}`; export, warm starts and evaluation
    recompute the family digest from the current extractor and refuse a mismatch;
  - shards (training/dataset.py) are accepted for a family when their stored schema provides it with the
    same digest, whatever their layout version;
  - an exported header carries the family in its types and static_asserts its input widths, so new features
    cannot be compiled into old weights or the reverse;
  - a registry holds one policy family and one value family; manifests record both.

Runs and manifests from before families (`spec_version: 2`, no `features`) are the legacy families p2 and v2.
Their digests are frozen in LEGACY_DIGESTS; tests check the extractor still reproduces them, so the legacy
encoders cannot drift silently.
"""
from __future__ import annotations

import hashlib
import json

POLICY = ("p2", "p3")
VALUE = ("v2", "v3")
LEGACY = {"policy": "p2", "value": "v2"}
TARGET_COLS = (0, 1, 2, 3, None, 9, 10)  # same as fusion.models.TARGET_COLS (the target head is unchanged in p3)

# Family digests of the legacy encoders at spec v2 (layout 2), frozen when families were introduced.
LEGACY_DIGESTS = {"p2": "213481118d8a6263", "v2": "ff0f693ef8e95ff0"}

_POLICY_BLOCKS = ("g", "u", "m", "legal", "act_legal", "origin", "cand", "cand_f")


def kind(name: str) -> str:
    if name in POLICY:
        return "policy"
    if name in VALUE:
        return "value"
    raise ValueError(f"unknown feature family {name!r} (known: {', '.join(POLICY + VALUE)})")


def _blocks(schema: dict) -> dict[str, list[int]]:
    return {name: list(shape) for name, _, shape in schema["layout"]}


def definition(schema: dict, name: str) -> dict | None:
    """What family `name` reads from a record of this schema; None if the schema does not provide it."""
    blocks = _blocks(schema)
    need_specs = {"p2": ("global", "unit", "move", "cand_f"), "p3": ("global", "unit", "move", "cand_f", "combat"),
                  "v2": ("global", "value_unit", "value_agg"), "v3": ("global", "value_unit", "value_agg", "value_ext")}[name]
    need_blocks = {"p2": _POLICY_BLOCKS, "p3": _POLICY_BLOCKS + ("pc",), "v2": ("v",), "v3": ("v", "vx")}[name]
    if any(g not in schema for g in need_specs) or any(b not in blocks for b in need_blocks):
        return None
    d = {"family": name, "specs": {g: schema[g] for g in need_specs}, "blocks": {b: blocks[b] for b in need_blocks}}
    if kind(name) == "policy":
        d |= {"moves": schema["moves"], "cand": schema["cand"], "acts": schema["acts"], "target_cols": list(TARGET_COLS),
              "inputs": {"mover": ["global", "unit", "move"] + (["combat"] if name == "p3" else []),
                         "squad": ["global", "unit x3", "move x15"] + (["combat x15"] if name == "p3" else []),
                         "act": "mover", "target": "target_cols"}}
    else:
        d |= {"inputs": ["global", "value_unit x3", "value_agg"] + (["value_ext"] if name == "v3" else [])}
    if name in ("p3", "v3"):
        d["combat_version"] = schema.get("combat_version")
    return d


def digest(schema: dict, name: str) -> str | None:
    d = definition(schema, name)
    if d is None:
        return None
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def available(schema: dict) -> dict[str, str]:
    """family -> digest for every family this schema's records provide."""
    return {n: dg for n in POLICY + VALUE if (dg := digest(schema, n)) is not None}


def layout_version(schema: dict) -> int:
    return int(schema.get("layout_version", 2))


def run_features(cfg: dict, want_kind: str) -> dict:
    """The feature family a training run used: its recorded `features`, or the legacy family for a run made
    before families existed (spec_version 2)."""
    feats = cfg.get("features")
    if feats:
        if kind(feats["family"]) != want_kind:
            raise SystemExit(f"run trained a {kind(feats['family'])} family ({feats['family']}), not a {want_kind} family")
        return feats
    if cfg.get("spec_version") == 2:
        name = LEGACY[want_kind]
        return {"family": name, "digest": LEGACY_DIGESTS[name], "layout_version": 2, "legacy": True}
    raise SystemExit(f"run has neither feature-family metadata nor the legacy spec_version 2 "
                     f"(spec_version {cfg.get('spec_version')}): retrain it")


def record(schema: dict, name: str) -> dict:
    """The `features` entry a new run writes into its config."""
    dg = digest(schema, name)
    if dg is None:
        raise SystemExit(f"the extractor does not provide feature family {name}")
    return {"family": name, "digest": dg, "layout_version": layout_version(schema)}


def check(schema: dict, feats: dict, what: str) -> None:
    """Refuse `what` (a run, a report, a model) whose family the current extractor does not reproduce."""
    name = feats["family"]
    now = digest(schema, name)
    if now is None:
        raise SystemExit(f"{what} uses feature family {name}, which this extractor's records do not provide")
    if now != feats["digest"]:
        raise SystemExit(f"{what} was trained on feature family {name} with digest {feats['digest']}; the extractor's "
                         f"{name} is {now}: the encoder changed, so retrain (or rebuild with the encoder it was trained on)")


def check_shards(schema: dict, names, where: str) -> None:
    """Refuse shards that do not provide `names` with the current extractor's digests."""
    have = available(schema)
    from .native import schema as current  # noqa: PLC0415
    now = current()
    for name in names:
        if name not in have:
            raise SystemExit(f"{where} (record layout {layout_version(schema)}) has no feature family {name}: "
                             f"rebuild the dataset into a new --out with the current extractor (layout {layout_version(now)})")
        if have[name] != digest(now, name):
            raise SystemExit(f"{where} stores feature family {name} with digest {have[name]}, the extractor's is "
                             f"{digest(now, name)}: rebuild the dataset into a new --out")
        # Records are sliced with the current layout: every block the family reads must sit at the same offset.
        old = {b: (o, list(s)) for b, o, s in schema["layout"]}
        cur = {b: (o, list(s)) for b, o, s in now["layout"]}
        moved = [b for b in definition(now, name)["blocks"] if old.get(b) != cur.get(b)]
        if moved:
            raise SystemExit(f"{where} places blocks {moved} of family {name} elsewhere than the extractor's layout "
                             f"{layout_version(now)}: rebuild the dataset into a new --out")
