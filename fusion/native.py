"""Compile and run the C++ tools. Builds are cached by a hash of their sources."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import BUILD, NEXTGEN_REL, REPO

FLAGS = ("-std=c++20", "-O2")


def compiler() -> str:
    override = os.environ.get("CXX")
    if override and shutil.which(override):
        return override
    for name in ("g++-16", "g++", "clang++"):
        if shutil.which(name):
            return name
    raise SystemExit("no C++ compiler found (set CXX)")


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
    out = BUILD / f"{name}-{key}"
    if out.is_file():
        return out
    BUILD.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    cmd = [compiler(), *FLAGS, *[f"-D{d}" for d in defines], *[f"-I{i}" for i in includes], str(source), "-o", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed: {' '.join(cmd)}\n{r.stderr[-4000:]}")
    tmp.replace(out)
    return out


@lru_cache(maxsize=1)
def extractor() -> Path:
    return build(REPO / "tools" / "extract_features.cpp", "extract_features")


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(subprocess.run([str(extractor()), "--schema"], capture_output=True, text=True, check=True).stdout)


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


def infer_check(model_header: Path, namespace: str) -> Path:
    """Parity driver for one exported model header (see tools/infer_check.cpp)."""
    model_header = Path(model_header).resolve()
    return build(REPO / "tools" / "infer_check.cpp", f"infer_check_{namespace.split('::')[-1]}", deps=[model_header],
                 defines=[f'FUSION_MODEL_HEADER="{model_header}"', f"FUSION_MODEL_NS={namespace}"],
                 includes=[REPO / "inference"])


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
