"""ctypes wrapper of the native Meowmind bridge (tools/cpp/meow_rl/bridge.cpp).

    lib = load()                          # compiles once into build/meow_rl/, content-addressed
    s = Session(lib)                      # one episode of one seat
    obs = s.observe(block_text)           # INIT + TURN on the first call, then each TURN block
    line = s.resolve(joint)               # the reference's resolver and reply format for that joint move

The library is the unmodified reference compiled with a C ABI, so features, feasibility, the opponents' online
belief and the attack/pickup resolver are the reference's own code. Windows: MinGW g++ (w64devkit) builds a DLL with
the C++ runtime linked statically, so loading it needs no other DLL; the path is absolute and content-hashed.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fusion import runner
from .identity import BRIDGE_SOURCE, JOINT, MOVES, PF, REFERENCE, REPO, UNITS, VF

ABI = 1
WINDOWS = sys.platform == "win32"
SUFFIX = ".dll" if WINDOWS else ".dylib" if sys.platform == "darwin" else ".so"
BUILD = REPO / "build" / "meow_rl"


def build_flags() -> list[str]:
    flags = ["-std=c++20", "-O2", "-shared"]
    if WINDOWS:
        flags += ["-static"]               # libstdc++/libgcc/winpthread inside the DLL
    else:
        flags += ["-fPIC"]
    return flags


def build(out_dir: Path = BUILD) -> Path:
    """Compile the bridge (once per source/flags/compiler); returns the library path."""
    cxx = runner.find_compiler()
    if not cxx:
        raise RuntimeError("no C++ compiler found (install g++/clang++ or w64devkit, or set CXX)")
    flags = build_flags()
    h = hashlib.sha256()
    for part in (BRIDGE_SOURCE.read_bytes(), REFERENCE.read_bytes(), " ".join(flags).encode(), cxx.encode()):
        h.update(part)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = out_dir / f"meow_rl_bridge-{h.hexdigest()[:16]}{SUFFIX}"
    if lib.is_file():
        return lib
    tmp = lib.with_name(f"{lib.stem}.tmp{os.getpid()}{SUFFIX}")
    r = subprocess.run([cxx, *flags, str(BRIDGE_SOURCE), "-o", str(tmp)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"bridge compile failed:\n{(r.stderr or r.stdout)[-3000:]}")
    os.replace(tmp, lib)
    return lib


_LIBS: dict[str, ctypes.CDLL] = {}


def load(path: Path | None = None) -> ctypes.CDLL:
    """Load (building if needed) the bridge library; one handle per path and process."""
    path = Path(path or build()).resolve()
    key = str(path)
    if key in _LIBS:
        return _LIBS[key]
    lib = ctypes.CDLL(key)
    f32, u8, i32 = (ctypes.POINTER(t) for t in (ctypes.c_float, ctypes.c_uint8, ctypes.c_int))
    sig = {"mr_abi_version": ([], ctypes.c_int), "mr_policy_features": ([], ctypes.c_int),
           "mr_value_features": ([], ctypes.c_int), "mr_create": ([], ctypes.c_void_p),
           "mr_destroy": ([ctypes.c_void_p], None), "mr_error": ([ctypes.c_void_p], ctypes.c_char_p),
           "mr_observe": ([ctypes.c_void_p, ctypes.c_char_p, f32, u8, u8, f32, f32, i32], ctypes.c_int),
           "mr_resolve": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, i32], ctypes.c_int),
           "mr_scalar_features": ([ctypes.c_void_p, f32], ctypes.c_int),
           "mr_weights": ([ctypes.c_char_p, f32, ctypes.c_int], ctypes.c_int)}
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    if lib.mr_abi_version() != ABI or lib.mr_policy_features() != PF or lib.mr_value_features() != VF:
        raise RuntimeError(f"{path}: bridge ABI/feature sizes differ from training.meow_rl")
    _LIBS[key] = lib
    return lib


def reference_weights(lib: ctypes.CDLL, prefix: str = "policy") -> dict[str, np.ndarray]:
    """The reference's actor (prefix 'policy') or V ('value') as [out, in] matrices and bias vectors."""
    shapes = {"w0": (64, PF if prefix == "policy" else VF), "b0": (64,), "w1": (32, 64), "b1": (32,), "w2": (1, 32), "b2": (1,)}
    out = {}
    for key, shape in shapes.items():
        name = f"{prefix}_{key}".encode()
        size = lib.mr_weights(name, None, 0)
        if size != int(np.prod(shape)):
            raise RuntimeError(f"reference weight {name.decode()}: {size} values, expected {shape}")
        buf = np.zeros(size, np.float32)
        lib.mr_weights(name, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size)
        out[key] = buf.reshape(shape)
    return out


@dataclass
class Observation:
    """One learner decision's inputs, all public and causal (the current block and earlier ones only)."""
    turn: int
    turns: int
    teams: int
    me: int
    size: int
    alive: int                 # bit k: own unit k alive (or respawning this turn)
    valid: int                 # feasible joint moves
    feats: np.ndarray          # [3, 5, 112] float32
    unit_legal: np.ndarray     # [3, 5] bool
    joint_mask: np.ndarray     # [125] bool
    value_feats: np.ndarray    # [416] float32
    ref_prob: np.ndarray       # [3, 5] float32, the reference actor (parity only)


class BridgeError(RuntimeError):
    pass


def _ptr(a: np.ndarray, kind):
    return a.ctypes.data_as(ctypes.POINTER(kind))


class Session:
    """One episode of one seat. Not thread-safe (the reference parser reads std::cin): one live session per process."""

    def __init__(self, lib: ctypes.CDLL | None = None):
        self.lib = lib or load()
        self.handle = self.lib.mr_create()
        if not self.handle:
            raise BridgeError("mr_create failed")

    def close(self):
        if self.handle:
            self.lib.mr_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def _check(self, code: int):
        if code != 0:
            raise BridgeError(self.lib.mr_error(self.handle).decode(errors="replace"))

    def observe(self, text: str) -> Observation:
        feats = np.zeros((UNITS, MOVES, PF), np.float32)
        legal = np.zeros((UNITS, MOVES), np.uint8)
        mask = np.zeros(JOINT, np.uint8)
        value = np.zeros(VF, np.float32)
        ref = np.zeros((UNITS, MOVES), np.float32)
        info = np.zeros(8, np.int32)
        self._check(self.lib.mr_observe(self.handle, text.encode(), _ptr(feats, ctypes.c_float), _ptr(legal, ctypes.c_uint8),
                                        _ptr(mask, ctypes.c_uint8), _ptr(value, ctypes.c_float),
                                        _ptr(ref, ctypes.c_float), _ptr(info, ctypes.c_int)))
        turn, turns, teams, me, size, alive, valid, _units = (int(x) for x in info)
        return Observation(turn, turns, teams, me, size, alive, valid, feats, legal.astype(bool), mask.astype(bool),
                           value, ref)

    def resolve(self, joint: int) -> tuple[str, list[int]]:
        """The reply line for the observed turn and the resolved actions (target id, -1 pickup/wait, -3 drop)."""
        buf = ctypes.create_string_buffer(256)
        actions = np.zeros(3, np.int32)
        self._check(self.lib.mr_resolve(self.handle, int(joint), buf, len(buf), _ptr(actions, ctypes.c_int)))
        return buf.value.decode(), [int(a) for a in actions]

    def scalar_features(self) -> np.ndarray:
        out = np.zeros((UNITS, MOVES, PF), np.float32)
        self._check(self.lib.mr_scalar_features(self.handle, _ptr(out, ctypes.c_float)))
        return out


def split_stream(text: str) -> list[str]:
    """A recorded seat input stream (INIT + TURN blocks, each ending in END) as one string per turn."""
    blocks, cur = [], []
    for line in text.splitlines(keepends=True):
        cur.append(line)
        if line.strip() == "END":
            blocks.append("".join(cur))
            cur = []
    if "".join(cur).strip():
        raise ValueError("stream ends without END")
    return blocks


def parse_reply(line: str) -> tuple[int, list[int], list[str]]:
    """(turn, the three moves as indices into SUDLR, the three action tokens); raises ValueError on malformed text."""
    tok = line.split("#", 1)[0].split()
    if len(tok) != 7 or not tok[0].isdigit():
        raise ValueError(f"malformed reply {line!r}")
    moves, acts = [], []
    for k in range(3):
        mv, act = tok[1 + 2 * k], tok[2 + 2 * k]
        if len(mv) != 1 or mv not in "SUDLR":
            raise ValueError(f"malformed move {mv!r} in {line!r}")
        if act not in ("-", "P", "X") and not act.isdigit():
            raise ValueError(f"malformed action {act!r} in {line!r}")
        moves.append("SUDLR".index(mv))
        acts.append(act)
    return int(tok[0]), moves, acts


def joint_index(moves) -> int:
    a, b, c = moves
    return 25 * a + 5 * b + c


def joint_moves(j: int) -> tuple[int, int, int]:
    return j // 25, j // 5 % 5, j % 5
