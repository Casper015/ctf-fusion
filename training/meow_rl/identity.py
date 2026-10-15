"""Fixed identities of the Meowmind RL pilot: architectures, feature schemas, reward units and source hashes.

A run records all of them (training.meow_rl.train); a resumed run, an export and a later bot refuse a mismatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "bots" / "meowmind_p" / "meowmind-ctf-374-20261011.cpp"
REFERENCE_SHA256 = "cc7b730cf77c087c8c7f415b3b7d07101c3f003924d3a845a2ea3b81588794bc"
BRIDGE_SOURCE = REPO / "tools" / "cpp" / "meow_rl" / "bridge.cpp"
PROFILE = REPO / "profiles" / "meowmind_rl_pilot.json"

POLICY_FEATURES = "meow_policy112_v1"          # learning::Geometry::policyMoves of the reference
VALUE_FEATURES = "meow_public416_v1"           # learning::Geometry::value (RICH_VALUE: 80 + 3 x 112)
ACTOR_ARCH = "meow_actor_112_64_32_1_joint125_t105_v1"
CRITIC_ARCH = {"large_v": "meow_public416_mlp_1024_1024_512_v1", "small_v": "meow_public416_mlp_64_32_v1"}
REWARD_UNITS = "terminal_2_pairwise_utility_minus_1"
RESOLVER = "meowmind_reference_shots_frozen_v1"
PF, VF, UNITS, MOVES, JOINT = 112, 416, 3, 5, 125
TEMPERATURE = 1.05                             # Planner::policy's softmax temperature, kept for the warm start
MOVE_TEXT = "SUDLR"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def load_profile(path: Path = PROFILE) -> dict:
    """The declarative pilot profile, checked against the architecture this code implements."""
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    actor, critic, small = p["actor"], p["critic"], p["small_critic"]
    expect = [(actor["input_per_move"], PF), (actor["hidden"], [64, 32]), (actor["joint_actions"], JOINT),
              (critic["features"], VF), (critic["hidden"], [1024, 1024, 512]), (critic["units"], REWARD_UNITS),
              (small["hidden"], [64, 32])]
    for got, want in expect:
        if got != want:
            raise ValueError(f"{path}: profile says {got}, this implementation is {want}")
    if critic_parameters(critic["hidden"]) != critic["parameters"] or critic_parameters(small["hidden"]) != small["parameters"]:
        raise ValueError(f"{path}: critic parameter counts do not match the architecture")
    return p


def critic_parameters(hidden: list[int], features: int = VF) -> int:
    sizes = [features, *hidden, 1]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def source_files() -> list[Path]:
    here = Path(__file__).resolve().parent
    return sorted([*here.glob("*.py"), BRIDGE_SOURCE, *(REPO / "tools" / "cpp" / "meow_rl").glob("*.hpp")])


def sources() -> dict:
    """Hashes of the reference, the bridge and this package: a run is bound to exactly this code."""
    ref = sha256_file(REFERENCE)
    if ref != REFERENCE_SHA256:
        raise RuntimeError(f"{REFERENCE} changed: sha256 {ref}, the reference is {REFERENCE_SHA256}")
    files = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_files()}
    return {"reference": REFERENCE.relative_to(REPO).as_posix(), "reference_sha256": ref, "files": files,
            "code_digest": digest(files)}


def identity(arm: str) -> dict:
    """What a checkpoint or export must match."""
    if arm not in CRITIC_ARCH:
        raise ValueError(f"unknown arm {arm!r}: {sorted(CRITIC_ARCH)}")
    return {"actor": ACTOR_ARCH, "critic": CRITIC_ARCH[arm], "policy_features": POLICY_FEATURES,
            "value_features": VALUE_FEATURES, "reward_units": REWARD_UNITS, "resolver": RESOLVER,
            "temperature": TEMPERATURE, "reference_sha256": REFERENCE_SHA256}
