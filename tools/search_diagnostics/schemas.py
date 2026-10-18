"""Data schemas and contract models for agent_v2 branch diagnostics (T0085).

All records are immutable, serializable to JSON, and input-bound with cryptographic digests.
Fail closed: missing fields, non-finite values, or corrupted signatures raise ValidationErrors.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_ROOT = "ctf-diag-root-1"
SCHEMA_BRANCH = "ctf-diag-branch-1"
SCHEMA_ANALYSIS = "ctf-diag-analysis-1"
SCHEMA_PARITY = "ctf-diag-parity-1"
SCHEMA_P_FIDELITY = "ctf-diag-p-fidelity-1"
SCHEMA_ESTIMATE = "ctf-diag-estimate-1"


class ValidationError(ValueError):
    """Raised when diagnostic input or record fails integrity, schema or contract checks."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnv64(text: str) -> str:
    h = 1469598103934665603
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


@dataclass(frozen=True)
class CandidateOrder:
    """A candidate joint action and its completed full order line."""
    index: int                          # candidate index 0..K-1
    joint: int                          # joint move index 0..124
    moves: tuple[int, int, int]         # (m0, m1, m2) where m in 0..4 (S, U, D, L, R)
    order: str                          # full order line (e.g. "turn m0 act0 m1 act1 m2 act2")
    prior_score: float                  # actor score / logit
    prior_rank: int                     # rank in actor's prior distribution (0-indexed)
    is_baseline: bool                   # True iff this is the actually sampled direct action
    leaf_score: float | None = None     # optional critic score from search tree
    leaf_feats_sha256: str | None = None # optional hash of 416 float32 features

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "joint": self.joint,
            "moves": list(self.moves),
            "order": self.order,
            "prior_score": self.prior_score,
            "prior_rank": self.prior_rank,
            "is_baseline": self.is_baseline,
            "leaf_score": self.leaf_score,
            "leaf_feats_sha256": self.leaf_feats_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateOrder:
        moves = tuple(d["moves"])
        if len(moves) != 3 or not all(0 <= m < 5 for m in moves):
            raise ValidationError(f"invalid moves tuple: {moves}")
        if not (0 <= d["joint"] < 125):
            raise ValidationError(f"invalid joint: {d['joint']}")
        if not math.isfinite(d["prior_score"]):
            raise ValidationError(f"non-finite prior score: {d['prior_score']}")
        return cls(
            index=int(d["index"]),
            joint=int(d["joint"]),
            moves=moves,  # type: ignore
            order=str(d["order"]).strip(),
            prior_score=float(d["prior_score"]),
            prior_rank=int(d["prior_rank"]),
            is_baseline=bool(d["is_baseline"]),
            leaf_score=float(d["leaf_score"]) if d.get("leaf_score") is not None else None,
            leaf_feats_sha256=d.get("leaf_feats_sha256"),
        )


@dataclass
class RootRecord:
    """A causal root extracted from a match or live stream."""
    schema: str                         # SCHEMA_ROOT
    root_id: str                        # unique digest of (match_id, seat, turn)
    match_id: str
    map_seed: int
    flag_seed: int
    teams: int
    seat: int
    turn: int
    block_text: str                     # raw public TURN block at turn t
    prefix_sha256: str                  # sha256 of causal history (INIT + turns 1..t-1)
    baseline_joint: int                 # actually sampled direct joint
    baseline_order: str                 # resolver-completed order for baseline joint
    candidates: list[CandidateOrder]     # candidate set (nested size 8 or 24, includes baseline)
    model_identities: dict[str, str]    # actor, critic, p, simulator checksums
    random_keys: dict[str, str] = field(default_factory=dict)
    stratum: str = "general"            # tactical stratum tag (e.g. carrier_duel, calm)
    stratum_weight: float = 1.0         # population weight for strata sampling

    def validate(self) -> None:
        if self.schema != SCHEMA_ROOT:
            raise ValidationError(f"unknown root schema: {self.schema}")
        if not (2 <= self.teams <= 15):
            raise ValidationError(f"invalid teams count: {self.teams}")
        if not (0 <= self.seat < self.teams):
            raise ValidationError(f"invalid seat: {self.seat}")
        if self.turn < 1:
            raise ValidationError(f"turn must be positive: {self.turn}")
        if not self.candidates:
            raise ValidationError("root has empty candidates list")
        # Exact self-comparison check: exactly one candidate must be is_baseline=True
        baselines = [c for c in self.candidates if c.is_baseline]
        if len(baselines) != 1:
            raise ValidationError(f"expected exactly 1 baseline candidate, got {len(baselines)}")
        if baselines[0].joint != self.baseline_joint:
            raise ValidationError(
                f"baseline candidate joint {baselines[0].joint} != root baseline_joint {self.baseline_joint}"
            )
        # Ensure Candidate 0 is the baseline for exact self-comparison
        if self.candidates[0].index != 0 or not self.candidates[0].is_baseline:
            raise ValidationError("candidate 0 must be the direct baseline order")
        # Check uniqueness of candidates
        joints = [c.joint for c in self.candidates]
        if len(set(joints)) != len(joints):
            raise ValidationError(f"duplicate joint action in candidates: {joints}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "root_id": self.root_id,
            "match_id": self.match_id,
            "map_seed": self.map_seed,
            "flag_seed": self.flag_seed,
            "teams": self.teams,
            "seat": self.seat,
            "turn": self.turn,
            "block_text": self.block_text,
            "prefix_sha256": self.prefix_sha256,
            "baseline_joint": self.baseline_joint,
            "baseline_order": self.baseline_order,
            "candidates": [c.to_dict() for c in self.candidates],
            "model_identities": self.model_identities,
            "random_keys": self.random_keys,
            "stratum": self.stratum,
            "stratum_weight": self.stratum_weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RootRecord:
        record = cls(
            schema=str(d.get("schema", SCHEMA_ROOT)),
            root_id=str(d["root_id"]),
            match_id=str(d["match_id"]),
            map_seed=int(d["map_seed"]),
            flag_seed=int(d["flag_seed"]),
            teams=int(d["teams"]),
            seat=int(d["seat"]),
            turn=int(d["turn"]),
            block_text=str(d["block_text"]),
            prefix_sha256=str(d["prefix_sha256"]),
            baseline_joint=int(d["baseline_joint"]),
            baseline_order=str(d["baseline_order"]),
            candidates=[CandidateOrder.from_dict(c) for c in d["candidates"]],
            model_identities=dict(d["model_identities"]),
            random_keys=dict(d.get("random_keys", {})),
            stratum=str(d.get("stratum", "general")),
            stratum_weight=float(d.get("stratum_weight", 1.0)),
        )
        record.validate()
        return record


@dataclass
class BranchResult:
    """Execution receipt of one branch continuation rollout to terminal."""
    schema: str                         # SCHEMA_BRANCH
    root_id: str
    candidate_index: int
    repeat_id: int
    repeat_key: str
    final_score: list[int]              # raw terminal score vector from engine
    pairwise_utility: float             # pairwise rank outcome U_T for learner's seat in [0, 1]
    faults_prefix: int                  # faults during prefix replay
    faults_root_discarded: int          # faults from computed root-turn replies
    faults_continuation: int            # faults during live continuation
    latency_ms: float                   # wall-clock elapsed ms
    prefix_mismatch_count: int = 0      # order divergence during prefix replay

    def validate(self) -> None:
        if self.schema != SCHEMA_BRANCH:
            raise ValidationError(f"unknown branch schema: {self.schema}")
        if not math.isfinite(self.pairwise_utility) or not (0.0 <= self.pairwise_utility <= 1.0):
            raise ValidationError(f"invalid pairwise utility: {self.pairwise_utility}")
        if self.candidate_index < 0:
            raise ValidationError(f"invalid candidate index: {self.candidate_index}")
        if self.repeat_id < 0:
            raise ValidationError(f"invalid repeat id: {self.repeat_id}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "root_id": self.root_id,
            "candidate_index": self.candidate_index,
            "repeat_id": self.repeat_id,
            "repeat_key": self.repeat_key,
            "final_score": self.final_score,
            "pairwise_utility": self.pairwise_utility,
            "faults_prefix": self.faults_prefix,
            "faults_root_discarded": self.faults_root_discarded,
            "faults_continuation": self.faults_continuation,
            "latency_ms": self.latency_ms,
            "prefix_mismatch_count": self.prefix_mismatch_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BranchResult:
        res = cls(
            schema=str(d.get("schema", SCHEMA_BRANCH)),
            root_id=str(d["root_id"]),
            candidate_index=int(d["candidate_index"]),
            repeat_id=int(d["repeat_id"]),
            repeat_key=str(d["repeat_key"]),
            final_score=[int(s) for s in d["final_score"]],
            pairwise_utility=float(d["pairwise_utility"]),
            faults_prefix=int(d.get("faults_prefix", 0)),
            faults_root_discarded=int(d.get("faults_root_discarded", 0)),
            faults_continuation=int(d.get("faults_continuation", 0)),
            latency_ms=float(d.get("latency_ms", 0.0)),
            prefix_mismatch_count=int(d.get("prefix_mismatch_count", 0)),
        )
        res.validate()
        return res
