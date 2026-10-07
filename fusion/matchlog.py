"""Raw match files (schema ctf-fusion-match-1, training/specs/DATA_FORMAT.md).

One gzip JSON file per match. It keeps exactly what crossed the protocol: the
INIT block, every TURN block (identical for all seats), every seat's reply
line as sent, plus faults and the final result. Features are derived later,
so a feature change never requires replaying matches.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

SCHEMA = "ctf-fusion-match-1"


def save(path: os.PathLike, match: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(match, f, separators=(",", ":"))
    tmp.replace(path)


def load(path: os.PathLike) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        match = json.load(f)
    if match.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unknown schema {match.get('schema')}")
    return match


def init_for_seat(init_text: str, me: int) -> str:
    """The INIT block differs between seats only in its third token (`me`)."""
    first, rest = init_text.split("\n", 1)
    tok = first.split(" ")
    tok[2] = str(me)
    return " ".join(tok) + "\n" + rest


def seat_stream(match: dict, me: int, final: bool = False) -> str:
    """The exact stdin stream seat `me` received (plus the post-game block if `final`)."""
    parts = [init_for_seat(match["init"], me), *match["blocks"]]
    if final:
        parts.append(match["final_block"])
    return "".join(parts)


def split_init(first_input: str) -> tuple[str, str]:
    """Split turn 1's input into the INIT block and the TURN block."""
    i = first_input.index("\nTURN ") + 1
    return first_input[:i], first_input[i:]


def iter_matches(root: os.PathLike, teams=None):
    """Yield match file paths under data/raw/nXX/, optionally for some team counts."""
    root = Path(root)
    dirs = [root / f"n{t:02d}" for t in teams] if teams else sorted(root.glob("n[0-9][0-9]"))
    for d in dirs:
        yield from sorted(d.glob("*.json.gz"))
