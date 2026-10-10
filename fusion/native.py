"""Compile and run the C++ tools. Builds are cached by a hash of their sources."""
from __future__ import annotations

import hashlib
import json
import os
import re
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
    deps = [source, *deps, *sorted((REPO / "inference").glob("*.hpp")), *sorted((REPO / "tools" / "cpp").glob("*.hpp"))]
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
    return build(REPO / "tools" / "cpp" / "extract_features.cpp", "extract_features")


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(subprocess.run([str(extractor()), "--schema"], capture_output=True, check=True).stdout.decode())


def check_records(rec: np.ndarray, teams: int) -> None:
    """Every record must start with (turn, team) in turn-major, team-minor order.

    Catches a misaligned byte stream, e.g. a Windows tool writing in text mode
    (each 0x0A byte becomes 0x0D 0x0A, and turn 10 already contains one).
    """
    turns = len(rec) // teams if teams else 0
    if teams <= 0 or len(rec) != turns * teams or not (
            np.array_equal(rec[:, 0], np.repeat(np.arange(1, turns + 1), teams)) and
            np.array_equal(rec[:, 1], np.tile(np.arange(teams), turns))):
        raise RuntimeError("extractor records are misaligned (raw output must be binary)")


def extract(stream_text: str) -> np.ndarray:
    """Feature records (rows = turn-major, team-minor) for one match stream, alignment-checked."""
    r = subprocess.run([str(extractor())], input=stream_text.encode(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"extractor failed: {r.stderr.decode()[-2000:]}")
    width = schema()["record_width"]
    if len(r.stdout) % (2 * width):
        raise RuntimeError("extractor output is not a whole number of records (raw output must be binary)")
    rec = np.frombuffer(r.stdout, dtype="<i2").reshape(-1, width)
    check_records(rec, int(stream_text.split(None, 2)[1]))
    return rec


def model_tool(tool: str, model_header: Path, namespace: str) -> Path:
    """Build a tool that takes one exported model header (infer_check, runtime_check), for its feature families."""
    model_header = Path(model_header).resolve()
    pfam, vfam = model_families(model_header)
    return build(REPO / "tools" / "cpp" / f"{tool}.cpp", f"{tool}_{namespace.split('::')[-1]}_{pfam}{vfam}", deps=[model_header],
                 defines=[f'FUSION_MODEL_HEADER="{model_header.as_posix()}"', f"FUSION_MODEL_NS={namespace}",
                          f"FUSION_MAX_WIDTH={model_buffer_width(model_header)}",
                          f"FUSION_POLICY_FAMILY={pfam[1:]}", f"FUSION_VALUE_FAMILY={vfam[1:]}"],
                 includes=[REPO / "inference"])


def model_families(header: Path) -> tuple[str, str]:
    """(policy, value) feature families from an exported header's preamble; headers without the line are p2/v2."""
    with Path(header).open(encoding="utf-8") as f:
        for _, line in zip(range(64), f):
            match = re.match(r"// Feature families: policy (p\d+), value (v\d+)", line)
            if match:
                return match.group(1), match.group(2)
    return "p2", "v2"


def model_buffer_width(header: Path) -> int:
    """Read the small generated preamble; do not scan millions of weight literals.

    Support older generated headers' 'Widest layer' marker too. Native static
    assertions remain authoritative if metadata understates the actual shapes.
    """
    from .capacity import buffer_width  # noqa: PLC0415
    from .quant import MAX_WIDTH  # noqa: PLC0415
    with Path(header).open(encoding="utf-8") as f:
        for _, line in zip(range(64), f):
            match = re.match(r"// FUSION_MAX_WIDTH_REQUIRED=(\d+)\s*$", line)
            if not match:
                match = re.match(r"// Widest layer (\d+)(?:[:.]|\s)", line)
            if match:
                return buffer_width([int(match.group(1))])
    return MAX_WIDTH


def infer_check(model_header: Path, namespace: str) -> Path:
    """Parity driver for one exported model header (see tools/cpp/infer_check.cpp)."""
    return model_tool("infer_check", model_header, namespace)


def run_tool(binary: Path, stdin: bytes = b"", args=()) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *map(str, args)], input=stdin, capture_output=True)


@lru_cache(maxsize=1)
def sim_check() -> Path:
    return build(REPO / "tools" / "cpp" / "sim_check.cpp", "sim_check")


def sim_check_match(match: dict, workdir: Path) -> tuple[bool, str]:
    """Replay a recorded match through inference/sim.hpp; (exact, summary line)."""
    from . import matchlog  # noqa: PLC0415
    stream, replies = Path(workdir) / "sim_stream.txt", Path(workdir) / "sim_replies.txt"
    stream.write_bytes(matchlog.seat_stream(match, 0, final=True).encode())
    replies.write_bytes(("\n".join("-" if l is None else l for row in match["replies"] for l in row) + "\n").encode())
    r = run_tool(sim_check(), args=(stream, replies))
    # Exit 0 only when every compared turn matched and at least one turn was compared;
    # the summary line reports compared and skipped turns (sim_check.cpp).
    return r.returncode == 0, (r.stdout.decode() + r.stderr.decode()).strip()


INFER_RECORD = 3 * 5 + 125 + 3 * 5 * 4 + 3 * 8 * 5 + 1  # int32 per (turn, team), tools/cpp/infer_check.cpp


def run_infer_check(binary: Path, stream_text: str, rows: int) -> np.ndarray:
    """infer_check accumulators, (rows, INFER_RECORD) int32; the byte count must match exactly."""
    r = subprocess.run([str(binary)], input=stream_text.encode(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"infer_check failed: {r.stderr.decode()[-2000:]}")
    if len(r.stdout) != 4 * rows * INFER_RECORD:
        raise RuntimeError(f"infer_check wrote {len(r.stdout)} bytes, expected {4 * rows * INFER_RECORD} (raw output must be binary)")
    return np.frombuffer(r.stdout, dtype="<i4").reshape(rows, INFER_RECORD)


def forecast_sources(ai_test: Path) -> list[Path]:
    """The legacy planner sources behind q_H: forecast_dump.cpp and the nextgen files it includes."""
    nextgen = ai_test / NEXTGEN_REL
    deps = [nextgen / "tests" / "forecast_dump.cpp", nextgen / "master.cpp",
            *sorted((nextgen / "modules").glob("*.inc")), *sorted((nextgen / "search").glob("*.hpp"))]
    return [d for d in deps if d.is_file()]


def forecast_digest(ai_test: Path) -> str:
    """Identity of the legacy forecast: a digest of its sources (independent of host and compiler)."""
    h = hashlib.sha256()
    for p in forecast_sources(ai_test):
        h.update(p.relative_to(ai_test).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def forecast_dump(ai_test: Path) -> Path:
    """Legacy multi-team forecast printer from the AI_Test nextgen tests (heuristic baseline q_H)."""
    src = ai_test / NEXTGEN_REL / "tests" / "forecast_dump.cpp"
    if not src.is_file():
        raise SystemExit(f"{src} not found: the legacy forecast needs AI_Test nextgen")
    return build(src, "forecast_dump", deps=[p for p in forecast_sources(ai_test) if p != src])


def kernel_check(simd: bool = True, width: int = 1024) -> Path:
    """Bit-exact parity of the SIMD multiply-add kernels against the original scalar loop (tools/cpp/kernel_check.cpp)."""
    return build(REPO / "tools" / "cpp" / "kernel_check.cpp", f"kernel_check_w{width}{'' if simd else '_scalar'}",
                 defines=[f"FUSION_MAX_WIDTH={width}", f"FUSION_SIMD={int(simd)}"])


def run_kernel_check(cases: int = 300) -> tuple[bool, str]:
    """(every available kernel path matches, summary line naming the active and tested paths)."""
    r = run_tool(kernel_check(), args=(cases,))
    return r.returncode == 0, (r.stdout.decode() + r.stderr.decode()).strip()


def size_bench(width: int, simd: bool = True) -> Path:
    """The capacity workload driver (tools/cpp/size_bench.cpp), with or without the SIMD kernels."""
    return build(REPO / "tools" / "cpp" / "size_bench.cpp", f"size_bench_w{width}{'' if simd else '_scalar'}",
                 defines=[f"FUSION_MAX_WIDTH={width}", f"FUSION_SIMD={int(simd)}"])


@lru_cache(maxsize=1)
def adapter_check() -> Path:
    """Exact probabilities and empirical draws of inference/adapter.hpp (tools/cpp/adapter_check.cpp)."""
    return build(REPO / "tools" / "cpp" / "adapter_check.cpp", "adapter_check")


def registry_check(models_dir: Path) -> Path:
    """Dispatch printer compiled against <models_dir>/registry.hpp (tools/cpp/registry_check.cpp)."""
    models_dir = Path(models_dir).resolve()
    deps = sorted(models_dir.glob("*.hpp"))
    width = max((model_buffer_width(p) for p in deps), default=256)
    return build(REPO / "tools" / "cpp" / "registry_check.cpp", "registry_check", deps=deps,
                 defines=[f"FUSION_MAX_WIDTH={width}"], includes=[models_dir, REPO / "inference"])
