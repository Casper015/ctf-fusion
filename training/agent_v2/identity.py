"""Identity of the agent_v2 carrier: what a run on it, and every later export and bot, must name (task T0065).

A run on the new carrier is a NEW run. Its identity names, besides training.meow_rl's actor and critic architectures,
feature schemas, reward units and temperature (kept: the existing actors and critics can initialize it because these
input contracts are unchanged), everything this carrier changes: the carrier, the action schema (the actor's sampled
joint movement + resolver assign_v1, whose orders are a deterministic function of the state, that joint and the
frozen scenario seeds), the resolver rule, profile and scenario count, the opponent predictor (our P: registry tier,
the pinned MANIFEST and every weight header sha256, no mixture) and a digest of every source the native library
compiles. training.meow_rl.train refuses --resume across any change of this identity (its run config differs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.meow_rl import identity as MI

REPO = MI.REPO
CARRIER = "agent_v2_v1"
TIER = "FUSION_TIER_PXXL_VM"
MODELS = REPO / "inference" / "models_pxxl_vm"
ACTION_SCHEMA = "agent_v2_actions_v1"
RESOLVER = {"rule": "assign_v1", "profile": "dp_v1", "scenarios": 8,
            "coefficients": {"hit": 0.045, "kill": 0.20, "carrier_kill": 0.65, "pickup_capture": 1.2, "pickup": 0.32}}
BRIDGE_SOURCE = REPO / "tools" / "cpp" / "agent_v2" / "bridge.cpp"
DEFAULT_ACTOR = REPO / "inference" / "meow_rl" / "aligned" / "p2_large" / "actor.hpp"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(models: Path = MODELS) -> list[Path]:
    """Every file the bridge (and the bot) compiles from this repository: the carrier, the reused P stack, the
    integer kernels, the actor helpers and the model registry with all its weight headers."""
    files = [BRIDGE_SOURCE, *sorted((REPO / "inference" / "agent_v2").glob("*.hpp")),
             REPO / "bots" / "p_only" / "include" / "ponly.hpp", REPO / "tools" / "cpp" / "stream.hpp",
             *(REPO / "inference" / f for f in ("adapter.hpp", "features.hpp", "policy.hpp", "state.hpp", "sim.hpp", "mlp.hpp",
                                                "simd.hpp", "value.hpp")),
             REPO / "tools" / "cpp" / "meow_rl" / "actor_int.hpp",
             models / "registry.hpp", *sorted(models.glob("weights_*.hpp")), models / "MANIFEST.json"]
    return [f for f in files if f.is_file()]


def sources(models: Path = MODELS) -> dict:
    files = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_files(models)}
    return {"files": files, "code_digest": MI.digest(files)}


def predictor(models: Path = MODELS) -> dict:
    manifest = json.loads((models / "MANIFEST.json").read_text(encoding="utf-8"))
    return {"kind": "P", "role": "opponent model", "tier": TIER, "registry": models.relative_to(REPO).as_posix(),
            "manifest_sha256": sha256_file(models / "MANIFEST.json"),
            "models": {k: v.get("header_sha256") for k, v in manifest.items() if isinstance(v, dict)},
            "mixture": "none (ponly::View: pure P, no hand-written forecast, the registry alpha is not applied)"}


def identity(arm: str, models: Path = MODELS) -> dict:
    """The meow_rl identity of `arm` with the agent_v2 carrier, predictor and resolver."""
    return MI.identity(arm) | {"carrier": CARRIER, "action_schema": ACTION_SCHEMA, "resolver": RESOLVER,
                               "predictor": predictor(models)}
