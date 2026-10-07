"""Locate this repository and the AI_Test checkout it trains against.

The game engine, the match runner and every opponent bot live in the AI_Test
repository (github.com/Casper015/AI_Test). Point to its root with --ai-test or
the AI_TEST_ROOT environment variable.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "build"
PROFILES = REPO / "profiles"
WORKSPACE_REL = Path("ctf-arena") / "Player7" / "workspace"
NEXTGEN_REL = WORKSPACE_REL / "experiments" / "round3" / "nextgen"


def ai_test_root(explicit: str | os.PathLike | None = None) -> Path:
    value = explicit or os.environ.get("AI_TEST_ROOT")
    if not value:
        raise SystemExit("Point to the AI_Test checkout with --ai-test or AI_TEST_ROOT")
    root = Path(value).expanduser().resolve()
    if not (root / WORKSPACE_REL / "engine.py").is_file():
        raise SystemExit(f"{root / WORKSPACE_REL / 'engine.py'} not found; is {root} the AI_Test root?")
    return root


def workspace(root: Path) -> Path:
    return root / WORKSPACE_REL


def load_engine(root: Path):
    """Import the official engine and match runner (engine.py, core.py) from the workspace."""
    ws = str(workspace(root))
    if ws not in sys.path:
        sys.path.insert(0, ws)
    import core  # noqa: PLC0415
    import engine  # noqa: PLC0415
    return engine, core


def sha256(path: os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: os.PathLike):
    return json.loads(Path(path).read_text())


def load_profile(teams: int) -> dict:
    return load_json(PROFILES / f"n{teams:02d}.json")


def load_pool() -> dict:
    return load_json(PROFILES / "pool.json")


def bot_path(entry: dict, root: Path) -> Path:
    """Pool paths are relative to the AI_Test root; '@repo/' marks a path inside this repository."""
    rel = entry["path"]
    return REPO / rel[len("@repo/"):] if rel.startswith("@repo/") else root / rel
