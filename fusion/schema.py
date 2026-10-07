"""Extractor record layout and feature scales, as printed by `extract_features --schema`."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import native


class Layout:
    def __init__(self, schema: dict):
        self.schema = schema
        self.width = schema["record_width"]
        self.blocks = {name: (offset, tuple(shape)) for name, offset, shape in schema["layout"]}
        self.specs = {k: schema[k] for k in ("global", "unit", "move", "cand_f", "value_unit", "value_agg")}

    def get(self, rec: np.ndarray, name: str) -> np.ndarray:
        offset, shape = self.blocks[name]
        size = int(np.prod(shape))
        return rec[..., offset:offset + size].reshape(*rec.shape[:-1], *shape)

    def scale(self, group: str) -> np.ndarray:
        return np.array([s[3] for s in self.specs[group]], dtype=np.float64)

    def bounds(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        spec = self.specs[group]
        return np.array([s[1] for s in spec], dtype=np.int64), np.array([s[2] for s in spec], dtype=np.int64)

    def names(self, group: str) -> list[str]:
        return [s[0] for s in self.specs[group]]

    def value_groups(self) -> list[str]:
        """Value input = global, three own units, aggregates (inference/features.hpp encodeValue)."""
        return ["global", "value_unit", "value_unit", "value_unit", "value_agg"]

    def value_scale(self) -> np.ndarray:
        return np.concatenate([self.scale(g) for g in self.value_groups()])

    def value_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        parts = [self.bounds(g) for g in self.value_groups()]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


@lru_cache(maxsize=1)
def layout() -> Layout:
    return Layout(native.schema())
