"""Deterministic initializer fragments for dense weight headers, below the tracked-file limit.

Fragments are included inside a single constexpr array: inference keeps one contiguous
matrix and the exact integer format. Small arrays retain the legacy header text.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from fusion.paths import sha256

FILE_LIMIT = 50_000_000
CHUNK_ELEMENTS = 500_000


class HeaderWriter:
    def __init__(self, path: Path, chunk_elements: int = CHUNK_ELEMENTS):
        if chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        self.path = Path(path)
        self.chunk_elements = chunk_elements
        self.chunks = []

    def array(self, name: str, ctype: str, values: np.ndarray) -> str:
        from training.export_weights import _array
        flat = np.asarray(values).reshape(-1)
        if len(flat) <= self.chunk_elements:
            return _array(name, ctype, flat)
        directory = self.path.parent / (self.path.stem + "_parts")
        directory.mkdir(parents=True, exist_ok=True)
        includes = []
        for i, start in enumerate(range(0, len(flat), self.chunk_elements)):
            part = directory / f"{name}_{i:04d}.hpp"
            numbers = [str(int(v)) for v in flat[start:start + self.chunk_elements]]
            text = "// Generated initializer fragment. Do not edit.\n" + "\n".join(
                "    " + ", ".join(numbers[j:j + 24]) + "," for j in range(0, len(numbers), 24)) + "\n"
            if len(text.encode("utf-8")) >= FILE_LIMIT:
                raise ValueError(f"{part.name}: initializer fragment exceeds the tracked-file limit")
            part.write_bytes(text.encode("utf-8"))
            relative = part.relative_to(self.path.parent).as_posix()
            includes.append(f'#include "{relative}"')
            self.chunks.append({"path": relative, "sha256": sha256(part), "bytes": part.stat().st_size,
                                "array": name, "offset": start, "elements": len(numbers)})
        return f"inline constexpr {ctype} {name}[] = {{\n" + "\n".join(includes) + "\n};\n"

    def write(self, text: str) -> list[dict]:
        if len(text.encode("utf-8")) >= FILE_LIMIT:
            raise ValueError(f"{self.path.name}: header exceeds the tracked-file limit")
        self.path.write_bytes(text.encode("utf-8"))
        # Regeneration may shrink a matrix. Remove only this header's generated fragments.
        directory = self.path.parent / (self.path.stem + "_parts")
        wanted = {c["path"] for c in self.chunks}
        if directory.is_dir():
            for part in directory.glob("*.hpp"):
                if re.fullmatch(r"[a-z0-9_]+_\d{4}\.hpp", part.name) and part.relative_to(self.path.parent).as_posix() not in wanted:
                    part.unlink()
        return self.chunks


def verify_chunks(directory: Path, chunks: list[dict]) -> None:
    """Fail closed when a manifest-bound fragment is missing, moved or edited."""
    root = Path(directory).resolve()
    for item in chunks:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or invalid header chunk: {item['path']}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"header chunk differs from manifest: {item['path']}")
