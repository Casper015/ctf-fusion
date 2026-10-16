"""Identity of the agent_v2 carrier: what a run on it, and every later export and bot, must name (task T0065).

A run on the new carrier is a NEW run. Its identity names, besides training.meow_rl's actor and critic architectures,
feature schemas, reward units and temperature (kept: the existing actors and critics can initialize it because these
input contracts are unchanged), everything this carrier changes: the carrier, the action schema (the actor's sampled
joint movement + resolver assign_v1, whose orders are a deterministic function of the state, that joint and the
frozen scenario seeds), the resolver rule, profile and scenario count, the opponent predictor (our P: its bank, registry
tier and buffer width, the pinned MANIFEST and every weight header sha256, no mixture) and a digest of every source the
native library compiles. training.meow_rl.train refuses --resume across any change of this identity (its run config
differs), so a run never continues on another P bank.

P banks (BANKS; the C++ side is inference/agent_v2/p_bank.hpp): "models_pxxl_vm" (P2 XXL, the T0065 default) and
"count_specialists_p" (T0067's per-count specialists, one per exact team count 2-15; FUSION_TIER_CSP, width 2048).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.meow_rl import identity as MI

REPO = MI.REPO
CARRIER = "agent_v2_v1"
# bank -> (FUSION_TIER_* define, FUSION_MAX_WIDTH); the registry is inference/<bank>/.
BANKS = {"models_pxxl_vm": ("FUSION_TIER_PXXL_VM", 1024), "count_specialists_p": ("FUSION_TIER_CSP", 2048)}
DEFAULT_BANK = "models_pxxl_vm"
TIER = BANKS[DEFAULT_BANK][0]
MODELS = REPO / "inference" / DEFAULT_BANK
ACTION_SCHEMA = "agent_v2_actions_v1"
RESOLVER = {"rule": "assign_v1", "profile": "dp_v1", "scenarios": 8,
            "coefficients": {"hit": 0.045, "kill": 0.20, "carrier_kill": 0.65, "pickup_capture": 1.2, "pickup": 0.32}}
BRIDGE_SOURCE = REPO / "tools" / "cpp" / "agent_v2" / "bridge.cpp"
DEFAULT_ACTOR = REPO / "inference" / "meow_rl" / "aligned" / "p2_large" / "actor.hpp"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bank(name: str = DEFAULT_BANK) -> dict:
    """A P bank: its tier define, buffer width and registry directory. An unknown bank is refused."""
    if name not in BANKS:
        raise ValueError(f"unknown P bank {name!r}; known: {', '.join(BANKS)}")
    tier, width = BANKS[name]
    return {"bank": name, "tier": tier, "max_width": width, "models": REPO / "inference" / name}


def manifest_models(models: Path) -> dict:
    """{model id: header sha256} from a registry's MANIFEST.json, checked against the headers on disk. Two layouts:
    export_mixed's (models_*: one top-level entry per model) and T0067's (count_specialists_p: under "models")."""
    manifest = json.loads((models / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest["models"] if isinstance(manifest.get("models"), dict) else manifest
    out = {k: v.get("header_sha256") for k, v in entries.items() if isinstance(v, dict)}
    if not out or not all(out.values()):
        raise ValueError(f"{models}/MANIFEST.json: no models, or a model without header_sha256")
    for k, v in entries.items():
        if isinstance(v, dict) and v.get("header") and sha256_file(models / v["header"]) != v["header_sha256"]:
            raise ValueError(f"{models / v['header']}: sha256 differs from MANIFEST.json")
    return out


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


def sources(p_bank: str = DEFAULT_BANK) -> dict:
    files = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_files(bank(p_bank)["models"])}
    return {"files": files, "code_digest": MI.digest(files)}


def predictor(p_bank: str = DEFAULT_BANK) -> dict:
    b = bank(p_bank)
    models = b["models"]
    return {"kind": "P", "role": "opponent model", "bank": b["bank"], "tier": b["tier"], "max_width": b["max_width"],
            "registry": models.relative_to(REPO).as_posix(), "registry_sha256": sha256_file(models / "registry.hpp"),
            "manifest_sha256": sha256_file(models / "MANIFEST.json"), "models": manifest_models(models),
            "model_ids": "csp_<registry id>" if b["tier"] == "FUSION_TIER_CSP" else "<registry id>",
            "mixture": "none (ponly::View: pure P, no hand-written forecast, the registry alpha is not applied)"}


def identity(arm: str, p_bank: str = DEFAULT_BANK) -> dict:
    """The meow_rl identity of `arm` with the agent_v2 carrier, predictor (P bank `p_bank`) and resolver."""
    return MI.identity(arm) | {"carrier": CARRIER, "action_schema": ACTION_SCHEMA, "resolver": RESOLVER,
                               "predictor": predictor(p_bank)}
