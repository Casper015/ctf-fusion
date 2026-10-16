"""Build the agent_v2 native bridge (tools/cpp/agent_v2/bridge.cpp) for PPO rollouts (task T0065).

    lib = build()                                   # content-addressed in build/agent_v2/
    lib = build(p_bank="count_specialists_p")       # P = T0067's per-count specialists (FUSION_TIER_CSP)
    s = training.meow_rl.bridge.Session(training.meow_rl.bridge.load(lib))
    obs = s.observe(block); line = s.resolve(joint)  # the agent_v2 carrier's P + resolver, the deployed path

The library exports training.meow_rl's ABI 1, so training.meow_rl.bridge.load/Session, env.run_episode(lib_path=...)
and the collector use it unchanged. The build hash covers every transitive source (training.agent_v2.identity), the
flags and the compiler, so a changed P, resolver or kernel is a different library. `p_bank` (training.agent_v2.identity
BANKS) selects P's model bank: its FUSION_TIER_* define, its FUSION_MAX_WIDTH and its weight headers; mr_p_bank names
the bank compiled in, and load(p_bank=...) refuses a library built with another.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from fusion import runner
from training.meow_rl import bridge as MB

from . import identity as I

BUILD = I.REPO / "build" / "agent_v2"


def build_flags(actor_header: Path = I.DEFAULT_ACTOR, p_bank: str = I.DEFAULT_BANK) -> list[str]:
    b = I.bank(p_bank)
    return [*MB.build_flags(), "-I", str(I.REPO / "inference"), "-I", str(I.REPO / "tools" / "cpp" / "meow_rl"),
            f"-DFUSION_MAX_WIDTH={b['max_width']}", f"-D{b['tier']}=1",
            f'-DACTOR_HEADER="{Path(actor_header).resolve().as_posix()}"']


def build(out_dir: Path = BUILD, actor_header: Path = I.DEFAULT_ACTOR, p_bank: str = I.DEFAULT_BANK) -> Path:
    cxx = runner.find_compiler()
    if not cxx:
        raise RuntimeError("no C++ compiler found")
    flags = build_flags(actor_header, p_bank)
    h = hashlib.sha256()
    for p in [*I.source_files(I.bank(p_bank)["models"]), Path(actor_header)]:
        h.update(p.read_bytes())
    h.update(" ".join(flags).encode()), h.update(cxx.encode())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lib = out / f"agent_v2_bridge-{h.hexdigest()[:16]}{MB.SUFFIX}"
    if lib.is_file():
        return lib
    tmp = lib.with_name(f"{lib.stem}.tmp{os.getpid()}{MB.SUFFIX}")
    r = subprocess.run([cxx, *flags, "-w", str(I.BRIDGE_SOURCE), "-o", str(tmp)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"agent_v2 bridge compile failed:\n{(r.stderr or r.stdout)[-3000:]}")
    os.replace(tmp, lib)
    return lib


def load(path: Path | None = None, p_bank: str | None = None):
    """training.meow_rl.bridge.load on the agent_v2 library (it checks ABI 1 and the 112/416 schema); with `p_bank`,
    also that the library was built with that P bank (built for it when `path` is None)."""
    lib = MB.load(path or build(p_bank=p_bank or I.DEFAULT_BANK))
    lib.mr_carrier.restype = __import__("ctypes").c_char_p
    if lib.mr_carrier() != I.CARRIER.encode():
        raise RuntimeError(f"{path}: not the agent_v2 bridge")
    if p_bank is not None and bank_of(lib) != p_bank:
        raise RuntimeError(f"{path}: built with P bank {bank_of(lib)}, not {p_bank}")
    return lib


def bank_of(lib) -> str:
    """The P bank a loaded agent_v2 library was compiled with (mr_p_bank)."""
    lib.mr_p_bank.restype = __import__("ctypes").c_char_p
    return lib.mr_p_bank().decode()
