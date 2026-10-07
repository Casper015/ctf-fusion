"""Cross-platform bot runner (Windows, Linux, macOS).

Mirrors the original arena core (`ctf-stress-pack/arena/core.py` in AI_Test):
a reader thread per process feeds a queue, so timeouts work on Windows pipes,
where `select` only accepts sockets (the translated workspace `core.py` uses
`selectors` and cannot run there). Semantics kept from the original:

  * time is measured from the write of the turn input to the arrival of a line;
  * a reply later than the limit is a strike; late lines are discarded when
    they arrive, and the bot keeps waiting for the current turn's line;
  * a malformed line is a strike; a parsed line resets the strike count;
  * three consecutive strikes, or a dead process, end the bot for the match.

Compilation: g++/clang++ (CXX overrides; w64devkit is found on Windows), flags
`-std=c++20 -O2`, plus `-static` on Windows as the official rules require.
Binaries are cached by a hash of source and flags.
"""
from __future__ import annotations

import hashlib
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

MAX_STRIKES = 3
MAX_LINE = 65536
WINDOWS = os.name == "nt"
CXX_FLAGS = ["-std=c++20", "-O2"] + (["-static"] if WINDOWS else [])
EXE = ".exe" if WINDOWS else ""


def find_compiler() -> str | None:
    override = os.environ.get("CXX")
    if override and shutil.which(override):
        return shutil.which(override)
    for name in ("g++-16", "g++", "clang++"):
        if shutil.which(name):
            return shutil.which(name)
    if WINDOWS:
        here = Path(__file__).resolve()
        for base in [*(p / "tools" for p in here.parents), Path("C:/")]:
            guess = base / "w64devkit" / "bin" / "g++.exe"
            if guess.is_file():
                return str(guess)
    return None


def compile_cpp(source: Path, build_dir: Path, extra_flags=()) -> Path:
    source = Path(source)
    cxx = find_compiler()
    if not cxx:
        raise RuntimeError("no C++ compiler found (install g++/clang++ or w64devkit, or set CXX)")
    flags = [*CXX_FLAGS, *extra_flags]
    digest = hashlib.sha256(source.read_bytes() + " ".join(flags).encode() + cxx.encode()).hexdigest()[:16]
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / f"{source.stem}-{digest}{EXE}"
    if exe.is_file():
        return exe
    tmp = exe.with_name(f"{exe.stem}.tmp{os.getpid()}{EXE}")
    r = subprocess.run([cxx, *flags, str(source), "-o", str(tmp)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed for {source}:\n{(r.stderr or r.stdout)[-3000:]}")
    os.replace(tmp, exe)
    return exe


def command_for(program: Path, build_dir: Path) -> list[str]:
    p = Path(program)
    if p.suffix.lower() in (".cpp", ".cc", ".cxx"):
        return [str(compile_cpp(p, build_dir))]
    if p.suffix.lower() == ".py":
        return [sys.executable, "-X", "utf8", str(p)]
    return [str(p)]


class Bot:
    """One bot process with a fresh temporary working directory."""

    def __init__(self, command: list[str], name: str, limits: tuple[float, float]):
        self.name = name
        self.first_limit, self.turn_limit = limits
        self.strikes = 0
        self.dead = None
        self.faults: list[tuple[int, str]] = []
        self.calls = 0
        self.stale = 0
        self.stderr = deque(maxlen=20)
        self.latencies: list[float] = []   # seconds, turns after the first, replies that arrived in time
        self.late: list[float] = []        # seconds, replies that arrived after the limit
        self.first_seconds = 0.0
        self.lines: queue.Queue = queue.Queue()
        self.dir = Path(tempfile.mkdtemp(prefix="fusion_bot_"))
        try:
            self.proc = subprocess.Popen(command, cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                         bufsize=1)
        except OSError as exc:
            self.proc = None
            self.dead = f"cannot start: {exc}"
            return
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        try:
            while True:
                line = self.proc.stdout.readline(MAX_LINE + 1)
                if not line:
                    break
                self.lines.put((time.perf_counter(), line))
        except (OSError, ValueError):
            pass
        self.lines.put((time.perf_counter(), None))

    def _read_stderr(self):
        try:
            while True:
                line = self.proc.stderr.readline(4096)
                if not line:
                    break
                self.stderr.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def send(self, text: str) -> float | None:
        if self.dead:
            return None
        try:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self._kill(f"process exited (code {self.proc.poll()})")
            return None
        return time.perf_counter()

    def receive(self, started: float | None, turn: int) -> str | None:
        """The reply line for this turn, or None (timeout, dead process) after recording a strike."""
        if self.dead or started is None:
            return None
        limit = self.first_limit if self.calls == 0 else self.turn_limit
        self.calls += 1
        deadline = started + limit
        while True:
            try:
                arrived, line = self.lines.get(timeout=max(0.0, deadline - time.perf_counter()))
            except queue.Empty:
                self.stale += 1
                return self.strike(turn, f"timeout (>{limit * 1000:g} ms)")
            if line is None:
                self._kill(f"process exited (code {self.proc.poll()})", turn)
                return None
            if self.stale:
                self.stale -= 1
                continue
            if arrived > deadline:
                self.late.append(arrived - started)
                return self.strike(turn, f"timeout (>{limit * 1000:g} ms)")
            break
        spent = arrived - started
        if self.calls == 1:
            self.first_seconds = spent
        else:
            self.latencies.append(spent)
        if len(line) > MAX_LINE:
            return self.strike(turn, "line longer than 64 KiB")
        return line.rstrip("\r\n")

    def ok(self):
        self.strikes = 0

    def strike(self, turn: int, reason: str):
        self.strikes += 1
        self.faults.append((turn, reason))
        if self.strikes >= MAX_STRIKES:
            self._kill(f"{MAX_STRIKES} consecutive failures, last: {reason}", turn)
        return None

    def _kill(self, reason: str, turn: int | None = None):
        if not self.dead:
            self.dead = reason
            if turn is not None:
                self.faults.append((turn, reason))
        self.close()

    def close(self):
        p = self.proc
        if p is not None and p.poll() is None:
            p.kill()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if p is not None:
            for pipe in (p.stdin, p.stdout, p.stderr):
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def report(self) -> dict:
        lat = sorted(self.latencies)

        def pct(q):
            return round(1000 * lat[min(len(lat) - 1, int(q * len(lat)))], 3) if lat else None

        return {"name": self.name, "dead": self.dead, "faults": len(self.faults),
                "first_ms": round(1000 * self.first_seconds, 1), "turns_timed": len(lat),
                "avg_ms": round(1000 * sum(lat) / len(lat), 3) if lat else None,
                "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
                "max_ms": round(1000 * lat[-1], 3) if lat else None, "late": len(self.late),
                "stderr_tail": list(self.stderr)[-5:]}
