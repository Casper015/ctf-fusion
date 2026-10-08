"""Shared helpers for the roster tools: identities, JSON I/O, event tags."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

SCHEMA_INVENTORY = "ctf-bot-inventory-2"
SCHEMA_SUITE = "ctf-probe-suite-1"
SCHEMA_FINGERPRINT = "ctf-fingerprint-1"
SCHEMA_DISTANCE = "ctf-behavior-distance-1"
SCHEMA_STRENGTH = "ctf-strength-screen-1"

# Pre-registered tactical event tags of a probe turn, for the probing team (roster/probes.py).
# The stratum tags partition every turn; the others mark rarer situations so that quiet turns
# cannot dominate a behavior distance.
TAGS = ("quiet", "contact", "flag", "respawn",              # fusion.labels.strata, one per turn
        "pickup_option", "carrier_own", "escort", "interception", "focus_fire",
        "overkill_risk", "collision_risk", "return_risk", "endgame")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike) -> str:
    return sha256_bytes(Path(path).read_bytes())


def bot_key(bot_id: str) -> str:
    """A filesystem-safe, collision-free key for a bot id (paths, ref:path, pool names)."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", bot_id.rsplit("/", 1)[-1])[:48]
    return f"{stem}-{hashlib.sha256(bot_id.encode()).hexdigest()[:10]}"


def load_json(path: os.PathLike):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: os.PathLike, obj, indent=1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def tag_masks(bits: list[int]) -> np.ndarray:
    """(len(TAGS), T) bool from per-turn tag bitmasks."""
    b = np.asarray(bits, np.int64)
    return np.stack([(b >> i) & 1 for i in range(len(TAGS))]).astype(bool)
