"""Durable run storage: atomic files, the episode journal, rollout batches and resumable state.

Layout of a run directory (under runs/, never committed):

    config.json              identities, profile, pool, CLI; a resumed run must match (except mode/budget)
    journal.jsonl            append-only, fsynced: episodes started, batches committed, episodes discarded
    batches/update_000012.npz   per update: sampled actions, old log probs, old values, advantages, returns,
                             masks, turns, episode index per row (+ actor/value features with --keep-features)
    batches/update_000012.json  the update's episode records and diagnostics
    state.pt                 the commit point: learner (models + optimizers), RNG states, counters, next episode
    checkpoints/decisions_0000100000.pt   fixed-interval copies of state.pt; checkpoints/final.pt at the budget

An update is committed when state.pt is replaced; its named checkpoint, if any, is written just before, so a
committed checkpoint mark always has its file (a crash in between leaves a checkpoint the resumed run rewrites). On
resume, episodes the journal shows as started but not committed are recorded as discarded and their indices (seeds)
are never reused; batch files of an uncommitted update and stray temporary files move to discarded/. A torn last
journal line (a crash during the append) is ignored and reported. Opponent processes cannot be restored, so an
interrupted collection is discarded, not continued.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np


def fsync_dir(path: Path):
    if os.name == "nt":
        return                                   # Windows has no directory fsync; os.replace is atomic there
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_json(path: Path, obj):
    atomic_write(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def atomic_npz(path: Path, arrays: dict):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_write(path, buf.getvalue())


class RunStore:
    def __init__(self, out: Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "batches").mkdir(exist_ok=True)
        (self.out / "checkpoints").mkdir(exist_ok=True)
        self.journal_path = self.out / "journal.jsonl"

    # journal
    def log(self, event: str, **fields):
        line = json.dumps({"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}, sort_keys=True)
        sep = ""
        if self.journal_path.is_file() and self.journal_path.stat().st_size:
            with open(self.journal_path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                sep = "" if f.read(1) == b"\n" else "\n"            # a record whose newline was torn off
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(sep + line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def journal(self) -> list[dict]:
        """The journal's events; an unparsable last line (a torn append) is skipped, anything else raises."""
        if not self.journal_path.is_file():
            return []
        lines = [l for l in self.journal_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = []
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                self.torn = line
        return out

    def recover(self, committed_next: int, committed_update: int) -> dict:
        """After a restart: discard what was started after the last commit. Returns the next free episode index and
        the run's discard totals (from the journal, so a crash after a discard loses nothing)."""
        events = self.journal()
        torn = getattr(self, "torn", None)
        if torn is not None:                                      # drop the torn bytes before anything is appended
            (self.out / "journal.torn").write_text(torn + "\n", encoding="utf-8")
            atomic_write(self.journal_path, "".join(json.dumps(e, sort_keys=True) + "\n" for e in events).encode())
        started = {i for e in events if e["event"] == "started" for i in e["indices"]}
        lost = sorted(i for i in started if i >= committed_next)
        if lost:
            self.log("discarded", indices=lost, decisions=0, reason="interrupted before the update was committed")
        moved = []
        for p in sorted((self.out / "batches").iterdir()):
            m = re.fullmatch(r"update_(\d{6})\.(npz|json)", p.name)
            if m is None or int(m.group(1)) > committed_update:         # uncommitted or a stray temporary file
                dest = self.out / "discarded" / p.name
                dest.parent.mkdir(exist_ok=True)
                shutil.move(str(p), dest)
                moved.append(p.name)
        for p in [*self.out.glob("*.tmp*"), *(self.out / "checkpoints").glob("*.tmp*")]:
            p.unlink()
        discards = [e for e in self.journal() if e["event"] == "discarded"]
        return {"next_index": max([committed_next, *[i + 1 for i in started]]), "discarded": lost, "moved": moved,
                "torn_journal_line": torn, "discarded_episodes": len({i for e in discards for i in e["indices"]}),
                "discarded_decisions": sum(e.get("decisions", 0) for e in discards)}

    # config
    def config(self) -> dict | None:
        p = self.out / "config.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def write_config(self, cfg: dict):
        atomic_json(self.out / "config.json", cfg)

    # batches
    def batch_paths(self, update: int) -> tuple[Path, Path]:
        stem = self.out / "batches" / f"update_{update:06d}"
        return stem.with_suffix(".npz"), stem.with_suffix(".json")

    def write_batch(self, update: int, arrays: dict, meta: dict):
        npz, js = self.batch_paths(update)
        atomic_npz(npz, arrays)
        atomic_json(js, meta)

    # state
    def save_state(self, state: dict, checkpoint_name: str | None = None):
        import torch  # noqa: PLC0415  (the store itself is torch-free for rollout workers)
        buf = io.BytesIO()
        torch.save(state, buf)
        data = buf.getvalue()
        if checkpoint_name:                                        # before the commit point (see the module doc)
            atomic_write(self.out / "checkpoints" / checkpoint_name, data)
        atomic_write(self.out / "state.pt", data)

    def ensure_checkpoint(self, name: str):
        """Recreate a committed checkpoint whose copy is missing (from state.pt, which is that very state)."""
        dest = self.out / "checkpoints" / name
        if not dest.is_file() and (self.out / "state.pt").is_file():
            atomic_write(dest, (self.out / "state.pt").read_bytes())

    def load_state(self) -> dict | None:
        import torch  # noqa: PLC0415
        p = self.out / "state.pt"
        if not p.is_file():
            return None
        return torch.load(p, map_location="cpu", weights_only=False)
