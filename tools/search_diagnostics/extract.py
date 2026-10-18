"""Root selection and extraction through the real agent_v2 runtime (task T0085).

Source games are match records (fusion/matchlog.py, schema ctf-fusion-match-1) with one designated seat played by
the package's production variant. For each selected root turn t of that seat:

  1. the seat's exact causal stream (its INIT and the recorded TURN blocks 1 .. t) is replayed through the package's
     native root tool (native/diag_tool.cpp), which runs the direct bot's code path;
  2. the tool's replies for turns 1 .. t - 1 must equal the seat's recorded replies token for token (turn number
     included, comments aside), and its complete default order at t must equal the recorded reply at t. Only then is the baseline the order the deployed bot actually sent
     (the sampled joint, not the prior's top joint), and only then is the root kept; otherwise it is refused with
     the reason;
  3. the candidates are the tool's tree roots: the sampled joint first, then the feasible joints by prior
     (inference/agent_v2/search.hpp roots()), each with its complete assign_v1 order, the critic's per-scenario
     values and the rows of its leaves' 416 value features in the root's sidecar file (<root_id>.leaves.f32, its
     SHA-256 in the record). Candidate sets are nested prefixes: "primary" (8) and "extended" (24).

Root selection is public and outcome-blind: it reads the recorded TURN blocks' unit positions, HP and carried flags
(the strata below) and whether the seat's reply exists, never a score, an event or the result. Eligibility is
decided per turn, so the set of eligible turns spans the whole game. Rules (strata):
  unconditioned  every turn in [MARGIN + 2, 400 - MARGIN]
  contact        ... where a living unit of ours and a living enemy are within Manhattan distance 3
  carrier        ... where one of our units carries a flag
  override       ... where the frozen search gate, run offline with fixed work (diag_tool gate: the first 8 roots x
                 the tree's samples, tau as registered), would replace the actor's joint. This is the gate's
                 decision on the recorded states, not what a timed search bot did there
Within a stratum the turns with the smallest SHA-256 of (salt, match, stratum, turn) are taken; strata are disjoint
(earlier strata first). Each root records its population weight: eligible turns / selected turns of its stratum.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from fusion import matchlog
from fusion.protocol import parse_block
from tools.search_diagnostics.schemas import (
    NATIVE_ROOT, SCHEMA_ROOT, ValidationError, digest, sha256_bytes, sha256_file, sha256_text, validate_root,
)

MARGIN = 5
RULES = ("unconditioned", "contact", "carrier", "override")
SETS = {"primary": 8, "extended": 24}


class RootRefused(Exception):
    """A root that cannot be extracted faithfully; the message is the reason."""


def tokens(line: str | None) -> list[str] | None:
    """A reply's tokens with its turn number, without a comment."""
    return None if line is None else line.partition("#")[0].split()


def order_of(line: str | None) -> str | None:
    """A reply's complete order without its turn number and comment."""
    if line is None:
        return None
    return " ".join(line.partition("#")[0].split()[1:])


def designated_seat(match: dict, name: str) -> int:
    seats = [i for i, s in enumerate(match["seats"]) if s["name"] == name]
    if len(seats) != 1:
        raise ValidationError(f"{match['match_id']}: needs exactly one seat named {name}, found {seats}")
    return seats[0]


def in_stratum(block_text: str, seat: int, stratum: str, turn: int = 0, overrides: set[int] | None = None) -> bool:
    if stratum == "unconditioned":
        return True
    if stratum == "override":
        if overrides is None:
            raise ValidationError("the override stratum needs the gate's offline decisions (run_gate)")
        return turn in overrides
    b = parse_block(block_text)
    ours = [u for u in b.units if u[1] == seat and u[2] >= 0 and u[4] > 0]
    if stratum == "carrier":
        return any(u[5] >= 0 for u in ours)
    if stratum == "contact":
        foes = [u for u in b.units if u[1] != seat and u[2] >= 0 and u[4] > 0]
        return any(abs(a[2] - e[2]) + abs(a[3] - e[3]) <= 3 for a in ours for e in foes)
    raise ValidationError(f"unknown stratum {stratum}; known: {RULES}")


def select_roots(match: dict, seat: int, plan: list[tuple[str, int]], salt: str,
                 overrides: set[int] | None = None) -> list[dict]:
    """The planned roots of one match: [{"turn", "stratum", "weight", "eligible"}], outcome-blind."""
    taken: set[int] = set()
    out = []
    last = len(match["blocks"])
    for stratum, count in plan:
        if count < 1:
            raise ValidationError(f"stratum {stratum}: count {count}")
        eligible = [t for t in range(MARGIN + 2, min(last, 400 - MARGIN) + 1)
                    if t not in taken and match["replies"][t - 1][seat] is not None
                    and in_stratum(match["blocks"][t - 1], seat, stratum, t, overrides)]
        ranked = sorted(eligible, key=lambda t: hashlib.sha256(f"{salt}/{match['match_id']}/{stratum}/{t}".encode()).digest())
        chosen = sorted(ranked[:count])
        for t in chosen:
            taken.add(t)
            out.append({"turn": t, "stratum": stratum, "eligible": len(eligible),
                        "weight": len(eligible) / len(chosen)})
    return out


def run_gate(tool: Path, match: dict, seat: int, tau: float) -> set[int]:
    """The turns where the frozen gate would override the actor on the seat's recorded states (fixed work)."""
    stream = matchlog.init_for_seat(match["init"], seat) + "".join(match["blocks"])
    with tempfile.TemporaryDirectory(prefix="diag_gate_") as d:
        p = Path(d) / "stream.txt"
        p.write_text(stream, encoding="utf-8", newline="\n")
        r = subprocess.run([str(tool), "gate", str(p), repr(float(tau))], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise ValidationError(f"gate failed: {(r.stdout or r.stderr)[:300]}")
    rows = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    if not rows:
        raise ValidationError("gate printed nothing")
    return {row["turn"] for row in rows if row["override"]}


def context(block_text: str, seat: int) -> dict:
    """Public context of the root state: our score gap to the best other team, carriers, contact."""
    b = parse_block(block_text)
    others = [s for i, s in enumerate(b.score) if i != seat]
    return {"score_gap": b.score[seat] - max(others), "our_carriers": sum(1 for u in b.units if u[1] == seat and u[5] >= 0),
            "enemy_carriers": sum(1 for u in b.units if u[1] != seat and u[5] >= 0),
            "contact": in_stratum(block_text, seat, "contact")}


def prefix_digest(match: dict, turn: int) -> str:
    """Every seat's recorded inputs and replies before the root, and the root block."""
    return digest({"init": match["init"], "blocks": match["blocks"][:turn], "replies": match["replies"][:turn - 1]})


def run_root_tool(tool: Path, stream: str, turn: int) -> tuple[dict, bytes]:
    with tempfile.TemporaryDirectory(prefix="diag_root_") as d:
        p = Path(d) / "stream.txt"
        p.write_text(stream, encoding="utf-8", newline="\n")
        leaves = Path(d) / "leaves.f32"
        r = subprocess.run([str(tool), "root", str(p), str(turn), str(leaves)], capture_output=True, text=True,
                           timeout=600)
        data = leaves.read_bytes() if leaves.exists() else b""
    if r.returncode != 0:
        raise RootRefused(f"root tool: {(r.stdout or r.stderr).strip()[:300]}")
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise ValidationError(f"root tool printed no JSON: {e}") from None
    if out.get("schema") != NATIVE_ROOT:
        raise ValidationError(f"root tool schema {out.get('schema')!r}")
    if len(data) != out["leaf_rows"] * out["leaf_dim"] * 4:
        raise ValidationError(f"root tool wrote {len(data)} leaf bytes for {out['leaf_rows']} rows")
    return out, data


def extract_root(match_path: Path, match: dict, seat: int, plan_row: dict, tool_identity: dict, tool: Path,
                 rule: str) -> tuple[dict, bytes]:
    """One root record and its leaf-feature sidecar bytes, or RootRefused with the reason."""
    turn = plan_row["turn"]
    if not 2 <= turn <= len(match["blocks"]):
        raise RootRefused(f"turn {turn} is not in the match")
    stream = matchlog.init_for_seat(match["init"], seat) + "".join(match["blocks"][:turn])
    native, leaf_bytes = run_root_tool(tool, stream, turn)
    if native["turn"] != turn or native["teams"] != match["teams"] or native["me"] != seat:
        raise ValidationError(f"root tool answered turn {native['turn']} seat {native['me']} of {native['teams']} teams")
    recorded = [r[seat] for r in match["replies"][:turn]]
    if any(r is None for r in recorded):
        raise RootRefused("the seat's recorded prefix has a missing reply (a recorded fault)")
    if len(native["replies"]) != turn - 1:
        raise ValidationError(f"root tool replayed {len(native['replies'])} turns, expected {turn - 1}")
    for t, (got, want) in enumerate(zip(native["replies"], recorded), start=1):
        if tokens(got) != tokens(want):
            raise RootRefused(f"the package does not reproduce the seat's recorded reply at turn {t}: "
                              f"{want!r} recorded, {got!r} replayed")
    if tokens(native["default_reply"]) != tokens(recorded[-1]):
        raise RootRefused(f"the package's default order {native['default_reply']!r} is not the recorded reply "
                          f"{recorded[-1]!r} at the root")
    cands = []
    for i, c in enumerate(native["candidates"]):
        cands.append({"index": i, "joint": c["joint"], "prior": c["prior"], "order": c["reply"], "v_values": c["values"],
                      "v_terminal": c["terminal"], "leaf_feature_fnv": c["leaf_feature_fnv"], "leaf_rows": c["leaf_rows"]})
    if cands[0]["joint"] != native["sampled_joint"] or cands[0]["order"] != native["default_reply"]:
        raise ValidationError("the tool's first root is not the sampled joint's default order")
    s = match["seats"][seat]
    root_id = sha256_text(f"{match['match_id']}/{seat}/{turn}")[:16]
    record = {
        "schema": SCHEMA_ROOT, "root_id": root_id, "turn": turn,
        "source": {"match_file": Path(match_path).name, "match_sha256": sha256_file(match_path),
                   "match_id": match["match_id"], "teams": match["teams"], "map_seed": match["map_seed"],
                   "flag_seed": match["flag_seed"], "seat": seat, "seat_name": s["name"], "seat_sha256": s["sha256"],
                   "seats": [{"name": x["name"], "sha256": x["sha256"]} for x in match["seats"]],
                   "engine_sha256": match["engine_sha256"]},
        "selection": {"rule": rule, "stratum": plan_row["stratum"], "eligible": plan_row["eligible"],
                      "weight": plan_row["weight"]},
        "context": context(match["blocks"][turn - 1], seat),
        "history": {"seat_stream": stream, "seat_stream_sha256": sha256_text(stream),
                    "prefix_digest": prefix_digest(match, turn), "root_block_sha256": sha256_text(match["blocks"][turn - 1])},
        "tool": {"package": tool_identity["package"], "binary_sha256": tool_identity["binary_sha256"],
                 "bundle_sha256": tool_identity["bundle_sha256"], "tree": tool_identity["tree"]},
        "model": {k: native[k] for k in ("carrier", "actor", "actor_digest", "predictor_kind", "predictor", "critic",
                                          "resolver", "resolver_scenarios", "tree_candidates", "tree_samples")},
        "policy": {k: native[k] for k in ("seed", "uniform", "state_key", "parent_key", "policy_feature_fnv", "mask",
                                           "scores", "sampled_joint", "tree_T", "refused_turns")},
        "baseline": {"order": native["default_reply"], "joint": native["sampled_joint"], "verified": True,
                     "check": "native replies of turns 1..t-1 and the default order at t equal the recorded replies"},
        "leaf_features": {"file": f"{root_id}.leaves.f32", "sha256": sha256_bytes(leaf_bytes), "rows": native["leaf_rows"],
                          "dim": native["leaf_dim"], "dtype": "float32 little-endian"},
        "candidates": cands,
        "candidate_sets": {k: list(range(min(n, len(cands)))) for k, n in SETS.items()},
    }
    return validate_root(record), leaf_bytes
