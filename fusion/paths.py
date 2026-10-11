"""Locate this repository and the arena checkout it trains against.

The game engine and the ten judge bots come from either of two checkouts, called the arena root:

* **chen-006/meow-ai-arena** (public, https://github.com/chen-006/meow-ai-arena): `ctf/workspace/engine.py`,
  `ctf/bots/*.cpp`. Enough for the end-to-end tests, smoke games and data collection against the judge bots.
* **AI_Test** (private, the original working tree): `ctf-arena/Player7/workspace/engine.py` and the rest of what the
  public repository does not have: the legacy planner behind q_H (`nextgen`), the Player5 reference bot, the
  stress-pack, benchmark and archived opponents.

Point to the root with --ai-test or the AI_TEST_ROOT environment variable (the name is kept: scripts and hosts set it).
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
WORKSPACE_REL = Path("ctf-arena") / "Player7" / "workspace"          # the AI_Test layout
NEXTGEN_REL = WORKSPACE_REL / "experiments" / "round3" / "nextgen"   # AI_Test only
ARENA_WORKSPACE_REL = Path("ctf") / "workspace"                      # the meow-ai-arena layout
ARENA_BOTS_REL = Path("ctf") / "bots"
JUDGE_BOTS_REL = "ctf-arena/Player7/judge/ctf/bots/"                 # where pool.json names the judge bots
IDENTICAL_BOTS = {"ctf-stress-pack/bots/baseline.cpp": "ctf/workspace/baseline.cpp"}   # byte-identical (same SHA-256)
ARENA_URL = "https://github.com/chen-006/meow-ai-arena"
WORKSPACES = (WORKSPACE_REL, ARENA_WORKSPACE_REL)


def workspace_rel(root: Path) -> Path | None:
    """Which layout `root` has (the one whose workspace holds engine.py), or None."""
    return next((w for w in WORKSPACES if (Path(root) / w / "engine.py").is_file()), None)


def ai_test_root(explicit: str | os.PathLike | None = None) -> Path:
    """The arena root: a meow-ai-arena checkout or an AI_Test checkout (see the module docstring)."""
    value = explicit or os.environ.get("AI_TEST_ROOT")
    if not value:
        raise SystemExit(f"Point to an arena checkout with --ai-test or AI_TEST_ROOT: a clone of {ARENA_URL} "
                         "or the AI_Test working tree")
    root = Path(value).expanduser().resolve()
    if workspace_rel(root) is None:
        arena, ai_test = (w.joinpath("engine.py").as_posix() for w in (ARENA_WORKSPACE_REL, WORKSPACE_REL))
        raise SystemExit(f"no engine.py under {root}: expected {arena} (meow-ai-arena, {ARENA_URL}) or {ai_test} (AI_Test)")
    return root


def workspace(root: Path) -> Path:
    return Path(root) / (workspace_rel(root) or WORKSPACE_REL)


def engine_path(root: Path) -> Path:
    return workspace(root) / "engine.py"


def has_legacy_planner(root: Path) -> bool:
    """The nextgen planner behind q_H (tests/forecast_dump.cpp): only the AI_Test working tree has it."""
    return (Path(root) / NEXTGEN_REL / "tests" / "forecast_dump.cpp").is_file()


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


AUGMENTATION_FAMILY = "augmentation:"   # family prefix of open-pool augmentation members (roster/registry.py)


def unlabeled_families(pool: dict | None = None) -> set[str]:
    """Families whose own requests are not opponent-policy labels (pool.json "policy_labels": false), e.g. noise.
    Families starting with AUGMENTATION_FAMILY are never labels either (is_unlabeled)."""
    pool = pool or load_pool()
    return {b["family"] for b in pool["bots"] if b.get("policy_labels", True) is False}


def is_unlabeled(family: str, unlabeled: set[str] | None = None) -> bool:
    return family.startswith(AUGMENTATION_FAMILY) or family in (unlabeled if unlabeled is not None else unlabeled_families())


SUBSTITUTED: dict[str, str] = {}      # pool path -> the public-repo file used instead (reported, never silent)


def bot_path(entry: dict, root: Path) -> Path:
    """Pool paths are relative to the arena root; '@repo/' marks a path inside this repository.

    `Baseline` missing at its stress-pack path is taken from `ctf/workspace/baseline.cpp`: byte-identical, so nothing
    changes. A judge bot missing at its AI_Test path is taken from `ctf/bots/` of a meow-ai-arena checkout instead. That file
    is the original with its comments, so its SHA-256 differs from the archived copy the evidence was keyed to (9 of
    the 11 compile to identical code; Kimi and Doubao differ in string labels). The substitution is recorded in
    SUBSTITUTED and printed once; checks that bind a bot to its recorded SHA-256 (screen_runner, roster.strength)
    still refuse it."""
    rel = entry["path"]
    if rel.startswith("@repo/"):
        return REPO / rel[len("@repo/"):]
    path = Path(root) / rel
    if not path.exists() and rel in IDENTICAL_BOTS and (Path(root) / IDENTICAL_BOTS[rel]).is_file():
        return Path(root) / IDENTICAL_BOTS[rel]          # the same file: no hash changes, nothing to report
    if not path.exists() and rel.startswith(JUDGE_BOTS_REL):
        alt = Path(root) / ARENA_BOTS_REL / rel[len(JUDGE_BOTS_REL):]
        if alt.is_file():
            if rel not in SUBSTITUTED:
                SUBSTITUTED[rel] = str(alt)
                print(f"[paths] judge bot {rel} is not in this checkout; using {alt} (same bot, public copy: its hash and "
                      "comments differ from the archived one)", file=sys.stderr, flush=True)
            return alt
    return path
