"""Registered scenario seeds and the common random keys of a collection (task T0085).

A collection takes one scenario seed. When a collection starts, the seed must have a line marked "reserved" (and none
marked "used" or "consumed") in the ledger the run names (for T0086, docs/SEARCH_BRANCH_DIAGNOSTIC_SEEDS.md), and must
not appear in any other seed ledger; collect.py also refuses it when another collection beside the output already
used it. Mark it used in the ledger once the collection is complete. Every branch of root r, repeat
k, whatever its candidate, uses the same two keys:

  env_key   = key(seed, root, k, "env")    reseeds the engine's flag-refresh RNG (engine.Game.flag_rng) just before
                                           the root turn is stepped, so every candidate of a repeat meets the same
                                           refresh draws, and repeats meet independent ones;
  actor_key = key(seed, root, k, "actor")  keys our continuation policy from turn t + 1 (native/continuation_bot.cpp).

The opponents' own randomness is not controlled (they are unmodified judge processes): it is recorded as such.
Keys are 63-bit, nonzero, and checked for collisions within a collection.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fusion.paths import REPO
from tools.search_diagnostics.schemas import ValidationError

KINDS = ("env", "actor")


def key(seed: int, root_id: str, repeat: int, kind: str) -> int:
    if kind not in KINDS:
        raise ValidationError(f"unknown key kind {kind}")
    h = hashlib.sha256(f"ctf-diag-key-1/{seed}/{root_id}/{repeat}/{kind}".encode()).digest()
    v = int.from_bytes(h[:8], "little") & ((1 << 63) - 1)
    return v or 1


def repeat_keys(seed: int, root_id: str, repeats: int) -> list[dict]:
    return [{"repeat": r, **{f"{k}_key": key(seed, root_id, r, k) for k in KINDS}} for r in range(repeats)]


def check_unique(keys: list[dict]) -> None:
    for kind in KINDS:
        values = [k[f"{kind}_key"] for k in keys]
        if len(set(values)) != len(values):
            raise ValidationError(f"{kind} key collision in this collection")


def check_seed(seed: int, ledger: Path, ledgers_dir: Path = REPO / "docs") -> dict:
    """The seed must be reserved in `ledger` and absent from every other seed ledger."""
    ledger = Path(ledger)
    if not ledger.is_file():
        raise ValidationError(f"seed ledger {ledger} not found")
    word = re.compile(rf"(?<!\d){seed}(?!\d)")
    lines = [ln for ln in ledger.read_text(encoding="utf-8").splitlines() if word.search(ln)]
    if not any(re.search(r"\breserved\b", ln, re.I) for ln in lines):
        raise ValidationError(f"seed {seed} has no line marked reserved in {ledger}")
    if any(re.search(r"\b(used|consumed)\b", ln, re.I) for ln in lines):
        raise ValidationError(f"seed {seed} is marked used or consumed in {ledger}")
    others = []
    for p in sorted(Path(ledgers_dir).glob("*SEEDS*.md")):
        if p.resolve() == ledger.resolve():
            continue
        if word.search(p.read_text(encoding="utf-8")):
            others.append(p.name)
    if others:
        raise ValidationError(f"seed {seed} is already used in {', '.join(others)}")
    return {"seed": seed, "ledger": ledger.as_posix(), "other_ledgers_checked": True}
