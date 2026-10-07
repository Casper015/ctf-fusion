"""Compile and run the C++ tools. Builds are cached by a hash of their sources."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import BUILD, NEXTGEN_REL, REPO
from .runner import CXX_FLAGS, EXE, find_compiler

FLAGS = tuple(CXX_FLAGS)  # -std=c++20 -O2, plus -static on Windows


def compiler() -> str:
    cxx = find_compiler()
    if not cxx:
        raise SystemExit("no C++ compiler found (install g++/clang++ or w64devkit, or set CXX)")
    return cxx


def _digest(paths, extra) -> str:
    h = hashlib.sha256(repr(extra).encode())
    for p in sorted(map(str, paths)):
        h.update(p.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def build(source: Path, name: str, deps=(), defines=(), includes=()) -> Path:
    """Compile `source` into build/<name>-<hash>; deps are extra files that invalidate the cache."""
    source = Path(source)
    deps = [source, *deps, *sorted((REPO / "inference").glob("*.hpp")), *sorted((REPO / "tools").glob("*.hpp"))]
    key = _digest(deps, (FLAGS, tuple(defines), tuple(map(str, includes)), compiler()))
    out = BUILD / f"{name}-{key}{EXE}"
    if out.is_file():
        return out
    BUILD.mkdir(parents=True, exist_ok=True)
    tmp = BUILD / f"{name}-{key}.tmp{os.getpid()}{EXE}"
    cmd = [compiler(), *FLAGS, *[f"-D{d}" for d in defines], *[f"-I{i}" for i in includes], str(source), "-o", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"compile failed: {' '.join(cmd)}\n{r.stderr[-4000:]}")
    os.replace(tmp, out)
    return out


@lru_cache(maxsize=1)
def extractor() -> Path:
    return build(REPO / "tools" / "extract_features.cpp", "extract_features")


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(subprocess.run([str(extractor()), "--schema"], capture_output=True, check=True).stdout.decode())


def extract(stream_text: str) -> np.ndarray:
    """Feature records (rows = turn-major, team-minor) for one match stream."""
    r = subprocess.run([str(extractor())], input=stream_text.encode(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"extractor failed: {r.stderr.decode()[-2000:]}")
    width = schema()["record_width"]
    data = np.frombuffer(r.stdout, dtype="<i2")
    if data.size % width:
        raise RuntimeError("extractor output is not a whole number of records")
    return data.reshape(-1, width)


def model_tool(tool: str, model_header: Path, namespace: str) -> Path:
    """Build a tool that takes one exported model header (infer_check, runtime_check)."""
    model_header = Path(model_header).resolve()
    return build(REPO / "tools" / f"{tool}.cpp", f"{tool}_{namespace.split('::')[-1]}", deps=[model_header],
                 defines=[f'FUSION_MODEL_HEADER="{model_header.as_posix()}"', f"FUSION_MODEL_NS={namespace}"],
                 includes=[REPO / "inference"])


def infer_check(model_header: Path, namespace: str) -> Path:
    """Parity driver for one exported model header (see tools/infer_check.cpp)."""
    return model_tool("infer_check", model_header, namespace)


def run_tool(binary: Path, stdin: bytes = b"", args=()) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *map(str, args)], input=stdin, capture_output=True)


@lru_cache(maxsize=1)
def sim_check() -> Path:
    return build(REPO / "tools" / "sim_check.cpp", "sim_check")


def sim_check_match(match: dict, workdir: Path) -> tuple[bool, str]:
    """Replay a recorded match through inference/sim.hpp; (exact, summary line)."""
    from . import matchlog  # noqa: PLC0415
    stream, replies = Path(workdir) / "sim_stream.txt", Path(workdir) / "sim_replies.txt"
    stream.write_bytes(matchlog.seat_stream(match, 0, final=True).encode())
    replies.write_bytes(("\n".join("-" if l is None else l for row in match["replies"] for l in row) + "\n").encode())
    r = run_tool(sim_check(), args=(stream, replies))
    return r.returncode == 0, (r.stdout.decode() + r.stderr.decode()).strip()


def run_infer_check(binary: Path, stream_text: str) -> np.ndarray:
    r = subprocess.run([str(binary)], input=stream_text.encode(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"infer_check failed: {r.stderr.decode()[-2000:]}")
    return np.frombuffer(r.stdout, dtype="<i4")


def forecast_dump(ai_test: Path) -> Path:
    """Legacy multi-team forecast printer from the AI_Test nextgen tests (heuristic baseline q_H)."""
    nextgen = ai_test / NEXTGEN_REL
    src = nextgen / "tests" / "forecast_dump.cpp"
    deps = [nextgen / "master.cpp", *sorted((nextgen / "modules").glob("*.inc")), *sorted((nextgen / "search").glob("*.hpp"))]
    return build(src, "forecast_dump", deps=[d for d in deps if d.is_file()])


def registry_check(models_dir: Path) -> Path:
    """Dispatch printer compiled against <models_dir>/registry.hpp (tools/registry_check.cpp)."""
    models_dir = Path(models_dir).resolve()
    deps = sorted(models_dir.glob("*.hpp"))
    return build(REPO / "tools" / "registry_check.cpp", "registry_check", deps=deps, includes=[models_dir, REPO / "inference"])
