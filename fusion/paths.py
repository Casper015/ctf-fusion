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
    """Import the official rules engine (engine.py) from the workspace.

    Bots run through fusion.runner, not the workspace core.py (whose selectors-based
    reads do not work on Windows pipes)."""
    ws = str(workspace(root))
    if ws not in sys.path:
        sys.path.insert(0, ws)
    import engine  # noqa: PLC0415
    return engine


def git_commit(root: os.PathLike) -> str:
    """HEAD of a git checkout, or 'unknown'."""
    import subprocess  # noqa: PLC0415
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: os.PathLike):
    return json.loads(Path(path).read_text(encoding="utf-8"))


TEAM_COUNTS = tuple(range(2, 16))


def load_profile(teams: int) -> dict:
    return load_json(PROFILES / f"n{teams:02d}.json")


def models() -> list[str]:
    """Every model id of the matrix: dedicated nXX ids plus 'general'."""
    return sorted({load_profile(t)["model"] for t in TEAM_COUNTS} | {"general"})


def model_counts(model: str) -> list[int]:
    """Team counts whose data trains `model`: every count for 'general', else the counts it serves."""
    if model == "general":
        return list(load_json(PROFILES / "general.json")["team_counts"])
    counts = [t for t in TEAM_COUNTS if load_profile(t)["model"] == model]
    if not counts:
        raise SystemExit(f"no profile uses model {model!r} (models: {', '.join(models())})")
    return counts


def served_counts(model: str) -> list[int]:
    """Team counts that load `model` at runtime."""
    return [t for t in TEAM_COUNTS if load_profile(t)["model"] == model]


def init_model(model: str) -> str | None:
    """The model a dedicated model warm-starts from (profiles: init_from)."""
    if model == "general":
        return None
    return load_profile(model_counts(model)[0]).get("init_from")


def load_pool() -> dict:
    return load_json(PROFILES / "pool.json")


def bot_path(entry: dict, root: Path) -> Path:
    """Pool paths are relative to the AI_Test root; '@repo/' marks a path inside this repository."""
    rel = entry["path"]
    return REPO / rel[len("@repo/"):] if rel.startswith("@repo/") else root / rel
