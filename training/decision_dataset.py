"""Versioned root/candidate dataset for own-action decisions (task T0034, docs/RL_PLAN.md section 3).

    # carrier configuration from the collection's receipts (checked again by every build)
    python -m training.decision_dataset carrier-config --manifest bots/p_only/report/v_data/manifest.json \\
        --identities bots/p_only/report/v_data/identities.json --variant v12_ponly_xxl --p-family p2 --status admitted \\
        --out build/decision/t0010_carrier.json
    # the dataset
    python -m training.decision_dataset --branches data/branches/v12_ponly_xxl --matches data/p_only/v_data/matches \\
        --audit bots/p_only/report/v_data/DATA_AUDIT.json --manifest-receipt bots/p_only/report/v_data/MANIFEST_METADATA_ADAPTER.json \\
        --carrier-config build/decision/t0010_carrier.json --out data/decision/t0010_p2
    # source digests an auditor records in a receipt (reads the same bytes the build reads)
    python -m training.decision_dataset source-digests --branches data/branches/v12_ponly_xxl --matches data/p_only/v_data/matches

Input: common-prefix branch roots (training.branch_collect or bots/p_only/tools/branch_portfolio.py, schema
ctf-fusion-branches-1) and the recorded matches they were cut from. Every root is linked to its source match
and checked; any inconsistency stops the build with the root's file name (nothing is skipped silently):
match id, team count, map, designated seat, carrier binary hash and INIT must agree; every branch's root block
must be the recorded TURN block of that turn; candidate 0 must be the recorded reply and the only on-policy
candidate (and source "recorded" where sources are stored); requested candidate orders must be distinct; every
repeat's stored outcome must equal the pairwise utility of its stored final score vector.

Carrier scores. A root written by the carrier's shortlist hook carries
    "carrier_scores": {"version": "ctf-fusion-shortlist-scores-1", "k": K, "orders": [...], "scores": [...]}
with the shortlist's complete orders (normalized reply lines) and the carrier's own score of each. Every order must
be one of the root's candidates, appear once, include the incumbent, and have a finite score; K must equal the
number of orders. The scores land in cand_carrier_score aligned by order (NaN for a candidate the carrier did not
score, e.g. an added deviation), and K in root_shortlist_k (-1 without a hook, as in T0010).

Root features come from the recorded prefix INIT + TURN 1..t of our seat, the exact stream the bot had read when
it answered turn t (the extractor's history features see every earlier turn). A root is never re-encoded as the
first turn of a one-turn match, and no block after t is read for features. Successor records (the realized
t + 1 state of each candidate) are written only with --successors, to <split>/oracle.npz, for the diagnostic
oracle of training.evaluate_decisions; the scorer's loader never opens that file.

Output (one file per split, splits by map exactly as training.dataset.split_of):
  <out>/<split>/decision.npz   roots and candidates (arrays below); <out>/index.json   identity and counts
Roots: root_match, root_map, root_teams, root_seat, root_turn, root_carrier, root_rec (extractor row), root_first,
  root_count, root_incumbent (candidate index), root_original_outcome (the recorded game's utility, a replay
  diagnostic, not a repeat), root_prefix_mismatch, root_prefix_faults, root_shortlist_k, root_file.
Candidates: cand_root, cand_line (requested order), cand_key (canonical identity), cand_source, cand_on_policy,
  cand_move/cand_act/cand_target/cand_slot (3,), cand_flags (fusion.decision_features), cand_dup_of,
  cand_support, cand_n (repeats with an outcome), cand_repeat_state, cand_score (C, R, 15; -1 pad), cand_outcome
  (C, R; NaN pad), cand_faults, cand_faults_prefix, cand_faults_root, cand_y (mean over the repeats with an outcome;
  NaN without one), cand_var (sample variance; NaN below two outcomes), cand_carrier_score (the carrier's logged
  score; NaN where none was logged).
Missing outcomes and scores stay missing; nothing is imputed. Dropped roots are listed with their reasons.

Repeats keep their identity (schema ctf-fusion-decision-2). Column r of every (C, R) array is the branch file's
repeat r for every candidate of the dataset, R = the largest repeat id + 1: a candidate whose repeat 0 failed has
NaN in column 0 and its repeat 1 still in column 1. Nothing is packed or renumbered, so repeat r of a candidate and
repeat r of its incumbent stay in the same column (paired comparisons) and even/odd repeat-id partitions
(training.evaluate_decisions.headroom) are the same whichever repeats are missing. The same repeat id does not make
two repeats correlated or independent: the collector has no controlled seed (decision_branch_collect.RNG_NOTE).
cand_repeat_state per column: 0 no branch record, 1 an outcome, 2 a branch record without a usable final score
(a failed continuation, kept and listed in index.json "failed_repeats", never imputed).

Faults per repeat, by phase (training.decision_branch_collect fault_counts; -1 without a branch record, -2 when the
collector did not record that phase):
  cand_faults         live continuation faults after the root turn ("faults"); it is not an all-phase total;
  cand_faults_prefix  faults while the recorded prefix was replayed ("prefix_faults");
  cand_faults_root    faults of the discarded root-turn replies the bots computed while the engine played the
                      forced orders ("discarded_root_computed_faults").

Authority. Every source file is hashed from the exact bytes the build parses. --authoritative refuses unless
  - the --audit receipt is internally consistent (its status and per-split flags agree) and says "passed";
  - it carries "decision_source": {"version": "ctf-fusion-decision-source-1", "branches_sha256", "matches_sha256"}
    equal to the digests of these branch files and of the match files their retained roots link to
    (`source-digests` prints them), and a T0010-style root_content_manifest_sha256, where present, also matches;
  - the manifest receipt's scope is a full baseline (a selected-identity scope needs --accept-selected-scope);
  - a --carrier-config validates (below).
A given --carrier-config is always validated, authoritative or not: schema ctf-fusion-carrier-config-1; the
roots' carrier binary; and, field by field, the collection receipts it names (binary and bundle hashes, every
opponent-P model file, whose exported headers carry the folded temperatures, the search source and the digest of
all build inputs, fixed samples, V, and the reply limits), with the receipts' own bytes pinned by hash. A
conflicting or missing field refuses the build. Without --authoritative the dataset is exploratory, and every
report made from it says so.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from fusion import matchlog, native
from fusion.labels import pairwise_outcome
from fusion.paths import REPO
from fusion.protocol import parse_block

SCHEMA = "ctf-fusion-decision-2"
MAX_REPEATS = 64
SOURCE_VERSION = "ctf-fusion-decision-source-1"
CARRIER_CONFIG_SCHEMA = "ctf-fusion-carrier-config-1"
SCORES_VERSION = "ctf-fusion-shortlist-scores-1"
CARRIER_STATUS = ("admitted", "experimental")
SPLITS = ("train", "val", "test")
MAX_TEAMS = 15
OUTCOME_TOL = 1e-9
FULL_SCOPES = ("full_original_manifest_bytes", "normalized_digest_recorded_at_freeze")

ROOT_KEYS = ("root_match", "root_map", "root_teams", "root_seat", "root_turn", "root_carrier", "root_rec", "root_first",
             "root_count", "root_incumbent", "root_original_outcome", "root_prefix_mismatch", "root_prefix_faults",
             "root_shortlist_k", "root_file")
CAND_KEYS = ("cand_root", "cand_line", "cand_key", "cand_source", "cand_on_policy", "cand_move", "cand_act", "cand_target",
             "cand_slot", "cand_flags", "cand_dup_of", "cand_support", "cand_n", "cand_repeat_state", "cand_score",
             "cand_outcome", "cand_faults", "cand_faults_prefix", "cand_faults_root", "cand_y", "cand_var",
             "cand_carrier_score")
REPEAT_NONE, REPEAT_OUTCOME, REPEAT_FAILED = 0, 1, 2
FAULT_NO_BRANCH, FAULT_UNRECORDED = -1, -2


class LinkError(ValueError):
    """A root that does not belong to the match, seat, carrier or turn it claims."""


def layout():
    from fusion.schema import layout as _l  # noqa: PLC0415
    return _l()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split_of_map(map_seed: int) -> str:
    from training.dataset import split_of  # noqa: PLC0415
    return split_of(f"map-{map_seed}")


def designated_seat(match: dict) -> int:
    from training.branch_collect import designated_seat as d  # noqa: PLC0415
    return d(match)


def find_match(matches: Path, match_id: str, teams: int) -> Path:
    p = Path(matches) / f"n{teams:02d}" / f"{match_id}.json.gz"
    if p.is_file():
        return p
    hits = sorted(Path(matches).rglob(f"{match_id}.json.gz"))
    if len(hits) != 1:
        raise LinkError(f"match {match_id}: {len(hits)} files under {matches}")
    return hits[0]


def read_gz_json(path: Path) -> tuple[dict, str]:
    """(object, sha256 of the file's bytes): the hash is of exactly the bytes that are parsed."""
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw).decode("utf-8")), hashlib.sha256(raw).hexdigest()


def read_match(path: Path) -> tuple[dict, str]:
    match, sha = read_gz_json(path)
    if match.get("schema") != matchlog.SCHEMA:
        raise LinkError(f"{path}: unknown match schema {match.get('schema')}")
    return match, sha


def prefix_stream(match: dict, seat: int, turn: int) -> str:
    """INIT + TURN 1..turn for our seat: what the bot had read when it answered `turn`."""
    blocks = match.get("blocks") or []
    if turn < 1 or len(blocks) < turn or any(not isinstance(b, str) or not b.startswith("TURN ") for b in blocks[:turn]):
        raise LinkError(f"{match.get('match_id')}: recorded history up to turn {turn} is missing or malformed")
    return matchlog.init_for_seat(match["init"], seat) + "".join(blocks[:turn])


def root_record(match: dict, seat: int, turn: int) -> np.ndarray:
    rec = native.extract(prefix_stream(match, seat, turn))
    row = rec[(turn - 1) * match["teams"] + seat].copy()
    F.check_root_record(row, turn, seat)
    return row


def successor_record(match: dict, seat: int, turn: int, next_block: str) -> np.ndarray:
    """Realized t + 1 record of our seat after one branch (oracle only: it contains the opponents' actual replies)."""
    rec = native.extract(prefix_stream(match, seat, turn) + next_block)
    row = rec[turn * match["teams"] + seat].copy()
    F.check_root_record(row, turn + 1, seat)
    return row


def check_link(row: dict, match: dict, name: str) -> int:
    """Validate a branch root against its source match; returns the seat."""
    def need(cond, msg):
        if not cond:
            raise LinkError(f"{name}: {msg}")
    need(row.get("schema") == "ctf-fusion-branches-1", f"unknown branch schema {row.get('schema')}")
    need(row["match_id"] == match["match_id"], f"match id {row['match_id']} != source {match['match_id']}")
    need(row["teams"] == match["teams"], "team count differs from the source match")
    need(row["map_seed"] == match["map_seed"], "map seed differs from the source match")
    need(row.get("init") == match["init"], "INIT differs from the source match")
    seat = designated_seat(match)
    need(row["seat"] == seat, f"seat {row['seat']} is not the match's designated seat {seat}")
    need(row["carrier_sha256"] == match["seats"][seat]["sha256"], "carrier hash differs from the seat that played the match")
    turn = row["turn"]
    need(1 <= turn <= len(match["blocks"]), f"turn {turn} outside the recorded match")
    need(len(match["replies"]) >= turn, "recorded replies end before the root")
    cands = row["candidates"]
    need(len(cands) >= 1, "no candidates")
    need(len(set(cands)) == len(cands), "duplicate requested candidate orders")
    need(F.normalize_line(match["replies"][turn - 1][seat]) == cands[0],
         "candidate 0 is not the recorded reply of the designated seat")
    sources = row.get("sources")
    if sources is not None:
        need(len(sources) == len(cands), "sources and candidates differ in length")
        need(sources[0] == "recorded", "candidate 0's source is not 'recorded'")
    root_block = match["blocks"][turn - 1]
    seen = set()
    for b in row["branches"]:
        c = b["candidate"]
        need(0 <= c < len(cands), f"branch candidate {c} out of range")
        need(bool(b["on_policy"]) == (c == 0), f"on_policy flag of candidate {c} is wrong")
        need(b["line"] == cands[c], f"branch line of candidate {c} differs from the candidate")
        rep = b.get("repeat")
        need(isinstance(rep, int) and not isinstance(rep, bool) and 0 <= rep < MAX_REPEATS,
             f"candidate {c}: repeat id {rep!r} is not an integer in [0, {MAX_REPEATS})")
        need((c, rep) not in seen, f"candidate {c} repeat {rep} appears twice")
        seen.add((c, rep))
        need(b.get("root_block") == root_block, f"candidate {c}: root block is not the recorded TURN {turn} block")
        score = b.get("score")
        if score is None:          # a failed continuation: kept as such (cand_repeat_state 2), never given an outcome
            need(b.get("outcome") is None, f"candidate {c} repeat {rep}: an outcome without a final score")
            continue
        need(len(score) == row["teams"], f"candidate {c}: score vector has {len(score)} teams")
        u = float(pairwise_outcome(score)[seat])
        need(abs(u - float(b["outcome"])) <= OUTCOME_TOL, f"candidate {c}: stored outcome {b['outcome']} != utility {u} of its score")
    return seat


def phase_faults(b: dict) -> dict:
    """Faults of one branch by phase; FAULT_UNRECORDED where the collector did not record that phase."""
    counts = b.get("fault_counts") or {}
    cont = len(b["faults"]) if isinstance(b.get("faults"), list) else counts.get("continuation", FAULT_UNRECORDED)
    prefix = b.get("prefix_faults", counts.get("prefix", FAULT_UNRECORDED))
    root = (len(b["discarded_root_computed_faults"]) if isinstance(b.get("discarded_root_computed_faults"), list)
            else counts.get("root_discarded", FAULT_UNRECORDED))
    return {"continuation": int(cont), "prefix": int(prefix), "root": int(root)}


def carrier_scores(row: dict, name: str) -> tuple[np.ndarray, int]:
    """(score per candidate, aligned by order; NaN where the carrier logged none), and the shortlist's K (-1: no hook)."""
    cands = row["candidates"]
    cs = row.get("carrier_scores")
    out = np.full(len(cands), np.nan)
    if cs is None:
        return out, -1

    def need(cond, msg):
        if not cond:
            raise LinkError(f"{name}: carrier_scores: {msg}")
    need(isinstance(cs, dict) and cs.get("version") == SCORES_VERSION, f"version must be {SCORES_VERSION}")
    orders, scores = cs.get("orders"), cs.get("scores")
    need(isinstance(orders, list) and isinstance(scores, list) and len(orders) == len(scores) and orders,
         "orders and scores must be equal-length, non-empty lists")
    need(cs.get("k") == len(orders), f"k {cs.get('k')} differs from the {len(orders)} logged orders")
    need(len(set(orders)) == len(orders), "an order is logged twice")
    need(cands[0] in orders, "the incumbent has no logged score")
    for o, s in zip(orders, scores):
        need(o in cands, f"logged order {o!r} is not a candidate of this root")
        need(isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s), f"score of {o!r} is not a finite number")
        out[cands.index(o)] = float(s)
    return out, int(cs["k"])


def candidate_source(row: dict, c: int) -> str:
    if row.get("sources") is not None:
        return row["sources"][c]
    if c == 0:
        return "recorded"
    return "move1" if row.get("candidate_mode", "single") == "single" else f"{row['candidate_mode']}:unrecorded"


# ---------------------------------------------------------------- sources and their digests

def scan(branch_files, matches: Path) -> dict:
    """Read every branch file once (hashing its bytes), and locate the match of every retained root."""
    entries, rows, drops, match_paths = [], [], [], {}
    for f in branch_files:
        row, sha = read_gz_json(Path(f))
        entries.append((Path(f), sha))
        if "dropped" in row:
            drops.append({"file": Path(f).name, "match_id": row.get("match_id"), "turn": row.get("turn"), "reason": row["dropped"]})
            continue
        rows.append((Path(f), row))
        key = (row["match_id"], row["teams"])
        if key not in match_paths:
            match_paths[key] = find_match(matches, *key)
    return {"branch_entries": entries, "rows": rows, "drops": drops, "match_paths": match_paths}


def source_digest(entries, base: Path) -> str:
    """ctf-fusion-decision-source-1: sha256 of sorted '<path relative to base>\\t<sha256 of bytes>' lines."""
    base = Path(base).resolve()
    lines = sorted(f"{Path(p).resolve().relative_to(base).as_posix()}\t{sha}\n" for p, sha in entries)
    return sha256_text("".join(lines))


def legacy_root_manifest(entries) -> str:
    """T0010's root_content_manifest_sha256 (REPRODUCE.md): '<path as listed> <sha256>' lines in sorted path order;
    it equals the receipt only when --branches is given exactly as the audit listed it (data/branches/...)."""
    return sha256_text("".join(f"{Path(p).as_posix()} {sha}\n" for p, sha in sorted(entries, key=lambda e: Path(e[0]))))


def digests(branches: Path, matches: Path, branch_entries, match_entries) -> dict:
    return {"version": SOURCE_VERSION, "branches_sha256": source_digest(branch_entries, branches),
            "matches_sha256": source_digest(match_entries, matches),
            "branch_files": len(branch_entries), "match_files": len(match_entries),
            "root_content_manifest_sha256": legacy_root_manifest(branch_entries)}


def source_digests(branches: Path, matches: Path, teams=None) -> dict:
    """The digests a build of these sources computes, for an auditor's receipt (no extraction)."""
    files = branch_listing(branches, teams)
    s = scan(files, Path(matches))
    match_entries = [(p, read_match(p)[1]) for p in sorted(s["match_paths"].values())]
    return digests(Path(branches), Path(matches), s["branch_entries"], match_entries)


def branch_listing(branches: Path, teams=None) -> list[Path]:
    files = sorted(Path(branches).rglob("*.json.gz"))
    if teams:
        files = [f for f in files if f.parent.name in {f"n{t:02d}" for t in teams}]
    return files


# ---------------------------------------------------------------- build

def build(branch_files, matches: Path, successors: bool = False, log=print) -> tuple[dict, dict]:
    """(per-split arrays, stats); raises LinkError on the first inconsistent root."""
    L = layout()
    F.require_layout(L)
    s = scan(branch_files, Path(matches))
    rows = s["rows"]
    by_match = defaultdict(list)
    for f, row in rows:
        by_match[(row["match_id"], row["teams"])].append((f, row))
    reps = max([1] + [int(b["repeat"]) + 1 for _, row in rows for b in row["branches"]
                      if isinstance(b.get("repeat"), int) and 0 <= b["repeat"] < MAX_REPEATS])
    out = {sp: defaultdict(list) for sp in SPLITS}
    oracle = {sp: defaultdict(list) for sp in SPLITS}
    carriers, match_entries, scored, failed = set(), [], 0, []
    for key, items in sorted(by_match.items()):
        match_id, teams = key
        mpath = s["match_paths"][key]
        match, msha = read_match(mpath)
        match_entries.append((mpath, msha))
        size = int(match["init"].split("\n", 1)[0].split()[3])
        for f, row in sorted(items, key=lambda x: x[1]["turn"]):
            seat = check_link(row, match, f.name)
            cscore, k = carrier_scores(row, f.name)
            scored += k >= 0
            turn = row["turn"]
            carriers.add(row["carrier_sha256"])
            split = split_of_map(row["map_seed"])
            d = out[split]
            rec = root_record(match, seat, turn)
            units = parse_block(match["blocks"][turn - 1]).units
            cands = row["candidates"]
            canon = [F.canonicalize(rec, L, line, turn, seat, units, size) for line in cands]
            flags = [c["flags"] for c in canon]
            dup = F.mark_duplicates([c["key"] for c in canon], flags, incumbent=0)
            by_cand = defaultdict(list)
            for b in row["branches"]:
                by_cand[b["candidate"]].append(b)
            r = len(d["root_match"])
            base = len(d["cand_root"])
            for name, val in (("root_match", match_id), ("root_map", row["map_seed"]), ("root_teams", teams), ("root_seat", seat),
                              ("root_turn", turn), ("root_carrier", row["carrier_sha256"]), ("root_rec", rec),
                              ("root_first", base), ("root_count", len(cands)), ("root_incumbent", base),
                              ("root_original_outcome", float(row["root_outcome"]) if "root_outcome" in row else np.nan),
                              ("root_prefix_mismatch", sum(int(b.get("prefix_mismatch", 0)) for b in row["branches"])),
                              ("root_prefix_faults", sum(int(b.get("prefix_faults", 0)) for b in row["branches"])),
                              ("root_shortlist_k", k), ("root_file", f.name)):
                d[name].append(val)
            for c, (line, cn) in enumerate(zip(cands, canon)):
                bs = sorted(by_cand.get(c, []), key=lambda b: b["repeat"])
                score = np.full((reps, MAX_TEAMS), -1, np.int32)
                outc = np.full(reps, np.nan)
                state = np.full(reps, REPEAT_NONE, np.int8)
                faults = {ph: np.full(reps, FAULT_NO_BRANCH, np.int16) for ph in ("continuation", "prefix", "root")}
                for b in bs:
                    i = int(b["repeat"])          # the column is the repeat id: nothing is packed or renumbered
                    for ph, n_ph in phase_faults(b).items():
                        faults[ph][i] = n_ph
                    if b.get("score") is None:
                        state[i] = REPEAT_FAILED
                        failed.append({"file": f.name, "candidate": c, "repeat": i,
                                       "reason": str(b.get("error", "no final score recorded"))})
                        continue
                    state[i] = REPEAT_OUTCOME
                    score[i, :teams] = b["score"]
                    outc[i] = float(pairwise_outcome(b["score"])[seat])
                have = np.isfinite(outc)
                n = int(have.sum())
                for name, val in (("cand_root", r), ("cand_line", line), ("cand_key", cn["key"]),
                                  ("cand_source", candidate_source(row, c)), ("cand_on_policy", c == 0),
                                  ("cand_move", cn["move"]), ("cand_act", cn["act"]), ("cand_target", cn["target"]),
                                  ("cand_slot", cn["slot"]), ("cand_flags", flags[c]),
                                  ("cand_dup_of", dup[c] + base if dup[c] >= 0 else -1),
                                  ("cand_support", bool(F.supported(flags[c]))), ("cand_n", n),
                                  ("cand_repeat_state", state), ("cand_score", score), ("cand_outcome", outc),
                                  ("cand_faults", faults["continuation"]), ("cand_faults_prefix", faults["prefix"]),
                                  ("cand_faults_root", faults["root"]),
                                  ("cand_y", float(np.mean(outc[have])) if n else np.nan),
                                  ("cand_var", float(np.var(outc[have], ddof=1)) if n >= 2 else np.nan),
                                  ("cand_carrier_score", cscore[c])):
                    d[name].append(val)
                if successors:
                    nb = {b["next_block"] for b in bs if b.get("score") is not None}
                    o = oracle[split]
                    if len(nb) == 1 and None not in nb:
                        o["succ_rec"].append(successor_record(match, seat, turn, nb.pop()))
                        o["succ_status"].append("ok")
                    else:
                        o["succ_rec"].append(np.zeros(L.width, np.int16))
                        o["succ_status"].append("no outcome" if not bs else "match ended" if None in nb else "repeats disagree")
        log(f"[decision_dataset] {match_id}: {len(items)} roots")
    if len(carriers) > 1:
        raise LinkError(f"roots of {len(carriers)} different carrier binaries: one dataset binds one carrier")
    stats = {"drops": s["drops"], "carriers": sorted(carriers), "repeats_max": reps, "roots": len(rows), "failed": failed,
             "roots_with_carrier_scores": int(scored), "branch_entries": s["branch_entries"], "match_entries": match_entries}
    return {"decision": out, "oracle": oracle if successors else None}, stats


_DTYPES = {"root_map": np.int64, "root_teams": np.int16, "root_seat": np.int16, "root_turn": np.int16, "root_rec": np.int16,
           "root_first": np.int32, "root_count": np.int16, "root_incumbent": np.int32, "root_original_outcome": np.float64,
           "root_prefix_mismatch": np.int32, "root_prefix_faults": np.int32, "root_shortlist_k": np.int16,
           "cand_root": np.int32, "cand_on_policy": bool,
           "cand_move": np.int16, "cand_act": np.int16, "cand_target": np.int16, "cand_slot": np.int16, "cand_flags": np.int16,
           "cand_dup_of": np.int32, "cand_support": bool, "cand_n": np.int16, "cand_repeat_state": np.int8,
           "cand_score": np.int32, "cand_outcome": np.float64, "cand_faults": np.int16, "cand_faults_prefix": np.int16,
           "cand_faults_root": np.int16, "cand_y": np.float64, "cand_var": np.float64,
           "cand_carrier_score": np.float64}


def to_arrays(d: dict) -> dict:
    out = {}
    for k in ROOT_KEYS + CAND_KEYS:
        v = d.get(k, [])
        out[k] = np.asarray(v, dtype=_DTYPES[k]) if k in _DTYPES else np.asarray(v, dtype=str)
    return out


# ---------------------------------------------------------------- receipts

def load_receipt(path: str | None) -> tuple[dict | None, str | None]:
    if not path:
        return None, None
    return json.loads(Path(path).read_text(encoding="utf-8")), sha256_file(Path(path))


def receipt_claims(audit: dict | None) -> list[str]:
    """Problems visible in the receipt alone: missing, contradictory or not passed."""
    if audit is None:
        return ["no audit receipt given"]
    status = audit.get("source_alignment_status", "unrecorded")
    flags = [v.get("source_row_alignment_verified") for v in (audit.get("rank_npz") or {}).values()]
    out = []
    if status == "passed" and any(f is not True for f in flags):
        out.append("receipt contradicts itself: status passed but a split is not source-aligned")
    if status != "passed" and flags and all(f is True for f in flags):
        out.append(f"receipt contradicts itself: every split aligned but status {status}")
    if status != "passed":
        out.append(f"strict source alignment not passed ({status})")
    src = audit.get("decision_source")
    if not isinstance(src, dict) or src.get("version") != SOURCE_VERSION:
        out.append(f"receipt has no {SOURCE_VERSION} digests of the branches and matches")
    return out


def receipt_matches(audit: dict | None, dig: dict) -> list[str]:
    """Problems between the receipt and the bytes this build read."""
    if audit is None:
        return []
    out = []
    src = audit.get("decision_source")
    if isinstance(src, dict) and src.get("version") == SOURCE_VERSION:
        for k in ("branches_sha256", "matches_sha256"):
            if src.get(k) != dig[k]:
                out.append(f"receipt's {k} certifies different bytes than this build read")
    legacy = audit.get("root_content_manifest_sha256")
    if legacy is not None and legacy != dig["root_content_manifest_sha256"]:
        out.append("receipt's root_content_manifest_sha256 describes a different root listing or content")
    return out


def scope_reasons(manifest_receipt: dict | None, accept_selected_scope: bool) -> list[str]:
    scope = (manifest_receipt or {}).get("verification_scope", "no manifest receipt given")
    if scope in FULL_SCOPES or (accept_selected_scope and scope == "selected_frozen_identities_only"):
        return []
    return [f"manifest verification scope: {scope}"]


# ---------------------------------------------------------------- carrier configuration

def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO / p


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def derive_carrier_config(manifest: str, identities: str, variant: str, family: str, status: str) -> dict:
    """A carrier configuration copied from a collection's receipts (build manifest and identities)."""
    man = json.loads(_resolve(manifest).read_text(encoding="utf-8"))
    ids = json.loads(_resolve(identities).read_text(encoding="utf-8"))
    v, comp = man["variants"][variant], ids[variant]["components"]
    inputs = {_posix(e["path"]): e["sha256"] for e in v["inputs"]}
    weights = [p for p in inputs if p.startswith("inference/models") and p.endswith(".hpp")]
    models = sorted({p.rsplit("/", 1)[0] for p in weights})
    if len(models) != 1:
        raise SystemExit(f"{variant}: model headers from {models or 'no directory'}; name the opponent P explicitly")
    source = _posix(v["source"])
    return {"schema": CARRIER_CONFIG_SCHEMA, "status": status, "variant": variant,
            "binary_sha256": v["binary_sha256"], "bundle_sha256": v["bundle_sha256"],
            "opponent_policy": {"family": family, "models": models[0],
                                "files": {p: s for p, s in sorted(inputs.items()) if p.startswith(models[0] + "/")},
                                "calibration": "temperatures folded into the exported headers listed in files"},
            "value": {"enabled": v.get("value_tier") is not None, "tier": v.get("value_tier")},
            "search": {"source": source, "source_sha256": inputs[source], "inputs_sha256": comp["inputs_sha256"],
                       "fixed_samples": v.get("fixed_samples", 0)},
            "budget": {"first_ms": comp["limits_ms"][0], "turn_ms": comp["limits_ms"][1]},
            "collection": {"manifest": manifest, "manifest_sha256": sha256_file(_resolve(manifest)),
                           "identities": identities, "identities_sha256": sha256_file(_resolve(identities))}}


def check_carrier_config(cfg: dict, carriers: list[str]) -> dict:
    """Refuse a configuration with a missing or conflicting field; returns what was verified."""
    def need(cond, msg):
        if not cond:
            raise SystemExit(f"carrier config: {msg}")
    need(cfg.get("schema") == CARRIER_CONFIG_SCHEMA, f"schema must be {CARRIER_CONFIG_SCHEMA}")
    need(cfg.get("status") in CARRIER_STATUS, f"status must be one of {CARRIER_STATUS}")
    for k in ("variant", "binary_sha256", "bundle_sha256", "opponent_policy", "value", "search", "budget", "collection"):
        need(cfg.get(k) not in (None, "", {}), f"missing {k}")
    op, se, bu, co, va = cfg["opponent_policy"], cfg["search"], cfg["budget"], cfg["collection"], cfg["value"]
    for k in ("family", "models", "files", "calibration"):
        need(op.get(k), f"missing opponent_policy.{k}")
    need(op["family"] in ("p2", "p3"), f"unknown opponent P family {op['family']}")
    for k in ("source", "source_sha256", "inputs_sha256"):
        need(se.get(k), f"missing search.{k}")
    need(isinstance(se.get("fixed_samples"), int), "missing search.fixed_samples")
    need(isinstance(bu.get("first_ms"), (int, float)) and isinstance(bu.get("turn_ms"), (int, float)), "missing budget limits")
    need(isinstance(va.get("enabled"), bool), "missing value.enabled")
    for k in ("manifest", "manifest_sha256", "identities", "identities_sha256"):
        need(co.get(k), f"missing collection.{k}")
    need(carriers == [cfg["binary_sha256"]],
         f"binary {cfg['binary_sha256'][:12]} is not the carrier that played these roots ({[c[:12] for c in carriers]})")
    mpath, ipath = _resolve(co["manifest"]), _resolve(co["identities"])
    need(mpath.is_file() and ipath.is_file(), "collection receipts not found")
    need(sha256_file(mpath) == co["manifest_sha256"], "collection manifest bytes differ from the pinned hash")
    need(sha256_file(ipath) == co["identities_sha256"], "collection identities bytes differ from the pinned hash")
    man = json.loads(mpath.read_text(encoding="utf-8")).get("variants", {})
    ids = json.loads(ipath.read_text(encoding="utf-8"))
    need(cfg["variant"] in man and cfg["variant"] in ids, f"variant {cfg['variant']} is not in the collection receipts")
    v, comp = man[cfg["variant"]], ids[cfg["variant"]]["components"]
    need(v["binary_sha256"] == cfg["binary_sha256"] == comp["binary_sha256"], "binary differs from the collection receipts")
    need(v["bundle_sha256"] == cfg["bundle_sha256"] == comp["bundle_sha256"], "bundle differs from the collection receipts")
    inputs = {_posix(e["path"]): e["sha256"] for e in v["inputs"]}
    need(_posix(v["source"]) == se["source"] and inputs.get(se["source"]) == se["source_sha256"],
         "search source differs from the collection manifest")
    need(comp["inputs_sha256"] == se["inputs_sha256"], "search/build inputs digest differs from the collection identities")
    need(v.get("fixed_samples", 0) == se["fixed_samples"], "fixed samples differ from the collection manifest")
    need(list(comp["limits_ms"]) == [bu["first_ms"], bu["turn_ms"]], "reply limits differ from the collection identities")
    need((v.get("value_tier") is not None) == va["enabled"] and v.get("value_tier") == va.get("tier"),
         "V setting differs from the collection manifest")
    prefix = op["models"].rstrip("/") + "/"
    in_dir = {p: s for p, s in inputs.items() if p.startswith(prefix)}
    need(in_dir, f"no build input under {op['models']}")
    need(op["files"] == in_dir, "opponent P files differ from (or omit some of) the collection manifest's model inputs")
    family_verified = None
    for p, s in sorted(in_dir.items()):
        local = REPO / p
        if p.endswith(".hpp") and "weights_" in p and local.is_file() and sha256_file(local) == s:
            fam = native.model_families(local)[0]
            need(fam == op["family"], f"{p} is a {fam} model, the config says {op['family']}")
            family_verified = p
            break
    return {"verified": True, "status": cfg["status"],
            "family_verified_from": family_verified or "no local header with the receipt's hash: family not verified"}


# ---------------------------------------------------------------- write and load

def write(out: Path, built: dict, stats: dict, meta: dict) -> dict:
    out = Path(out)
    index = {"schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta, "splits": {}}
    for split in SPLITS:
        arr = to_arrays(built["decision"][split])
        (out / split).mkdir(parents=True, exist_ok=True)
        path = out / split / "decision.npz"
        np.savez_compressed(path, **arr)
        entry = {"sha256": sha256_file(path), "roots": int(len(arr["root_match"])), "candidates": int(len(arr["cand_root"])),
                 "maps": int(len(np.unique(arr["root_map"]))) if len(arr["root_map"]) else 0}
        if built["oracle"] is not None:
            o = built["oracle"][split]
            opath = out / split / "oracle.npz"
            np.savez_compressed(opath, succ_rec=np.asarray(o["succ_rec"], np.int16).reshape(-1, layout().width),
                                succ_status=np.asarray(o["succ_status"], dtype=str))
            entry["oracle_sha256"] = sha256_file(opath)
        index["splits"][split] = entry
    maps = {s: set(np.load(out / s / "decision.npz")["root_map"].tolist()) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and maps[a] & maps[b]:
                raise LinkError(f"maps in both {a} and {b}: {sorted(maps[a] & maps[b])[:5]}")
    index["drops"] = {"count": len(stats["drops"]), "by_reason": dict(Counter(d["reason"] for d in stats["drops"])),
                      "roots": stats["drops"]}
    index["failed_repeats"] = {"count": len(stats.get("failed", [])), "repeats": stats.get("failed", [])}
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def index_of(data: Path) -> dict:
    index = json.loads((Path(data) / "index.json").read_text(encoding="utf-8"))
    if index.get("schema") != SCHEMA:
        raise SystemExit(f"{data}: not a {SCHEMA} dataset")
    return index


def split_maps(data: Path, split: str) -> np.ndarray:
    """Map ids of one split (no outcomes are read)."""
    with np.load(Path(data) / split / "decision.npz") as z:
        return z["root_map"].copy()


def check_isolation(data: Path) -> None:
    """Every map in exactly one split, as the builder wrote it (refuses a contaminated dataset)."""
    seen = {}
    for s in SPLITS:
        p = Path(data) / s / "decision.npz"
        if not p.exists():
            continue
        for m in np.unique(split_maps(data, s)).tolist():
            if m in seen and seen[m] != s:
                raise SystemExit(f"{data}: map {m} appears in {seen[m]} and {s}: split contamination")
            seen[m] = s


def load(data: Path, split: str, *, test_unlocked: bool = False, verify: bool = True) -> dict:
    """One split's arrays. The test split opens only after training.evaluate_decisions verified a frozen selection."""
    if split not in SPLITS:
        raise SystemExit(f"unknown split {split}")
    if split == "test" and not test_unlocked:
        raise SystemExit("the test split opens only through training.evaluate_decisions with a frozen selection")
    index = index_of(data)
    path = Path(data) / split / "decision.npz"
    if verify and sha256_file(path) != index["splits"][split]["sha256"]:
        raise SystemExit(f"{path} differs from the identity recorded in index.json")
    check_isolation(data)
    with np.load(path) as z:
        d = {k: z[k].copy() for k in z.files}
    expected = {split}
    got = {split_of_map(int(m)) for m in np.unique(d["root_map"]).tolist()}
    if got - expected:
        raise SystemExit(f"{path}: roots of maps that belong to {sorted(got - expected)}: split contamination")
    d["_index"] = index
    d["_split"] = split
    return d


def scorer_inputs(d: dict, layout_=None) -> tuple[np.ndarray, np.ndarray]:
    """(root features per candidate, candidate features): only the root record and the canonical order are read."""
    L = layout_ or layout()
    roots = root_features_cached(d, L)
    cand = np.stack([F.candidate_features(d["root_rec"][r], L, mv, a, s) for r, mv, a, s in
                     zip(d["cand_root"], d["cand_move"], d["cand_act"], d["cand_slot"])]) if len(d["cand_root"]) else \
        np.zeros((0, F.CAND_DIM), np.float32)
    return roots[d["cand_root"]], cand


def root_features_cached(d: dict, L) -> np.ndarray:
    if "_root_x" not in d:
        d["_root_x"] = F.root_features(d["root_rec"], L) if len(d["root_rec"]) else np.zeros((0, F.root_dim(L)), np.float32)
    return d["_root_x"]


# ---------------------------------------------------------------- commands

def carrier_config_main(argv):
    ap = argparse.ArgumentParser(prog="decision_dataset carrier-config",
                                 description="Write a carrier configuration copied from a collection's receipts.")
    ap.add_argument("--manifest", required=True, help="the collection's build manifest (report copy)")
    ap.add_argument("--identities", required=True, help="the collection's identities.json")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--p-family", required=True, choices=("p2", "p3"))
    ap.add_argument("--status", required=True, choices=CARRIER_STATUS,
                    help="admitted: the frozen deployment carrier; experimental: a deliberately experimental one")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = derive_carrier_config(a.manifest, a.identities, a.variant, a.p_family, a.status)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"[decision_dataset] carrier config {a.out}: {a.variant} {cfg['binary_sha256'][:12]} ({a.status})")


def source_digests_main(argv):
    ap = argparse.ArgumentParser(prog="decision_dataset source-digests")
    ap.add_argument("--branches", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--teams", type=int, nargs="*")
    a = ap.parse_args(argv)
    print(json.dumps(source_digests(Path(a.branches), Path(a.matches), a.teams), indent=1))


def main(argv=None):
    import sys  # noqa: PLC0415
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "carrier-config":
        return carrier_config_main(argv[1:])
    if argv and argv[0] == "source-digests":
        return source_digests_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches", required=True, help="branch root files (ctf-fusion-branches-1)")
    ap.add_argument("--matches", required=True, help="the recorded matches the roots were cut from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", help="DATA_AUDIT.json-style receipt of the branch data")
    ap.add_argument("--manifest-receipt", help="MANIFEST_METADATA_ADAPTER.json-style receipt (verification scope)")
    ap.add_argument("--authoritative", action="store_true", help="refuse unless the receipts certify these exact sources")
    ap.add_argument("--accept-selected-scope", action="store_true",
                    help="accept a manifest receipt that verified selected frozen identities only (recorded)")
    ap.add_argument("--carrier-config", help="ctf-fusion-carrier-config-1 JSON (see the carrier-config command)")
    ap.add_argument("--successors", action="store_true", help="also write realized successor records (diagnostic oracle only)")
    ap.add_argument("--teams", type=int, nargs="*")
    a = ap.parse_args(argv)
    audit, audit_sha = load_receipt(a.audit)
    mreceipt, mreceipt_sha = load_receipt(a.manifest_receipt)
    cfg_raw = Path(a.carrier_config).read_bytes() if a.carrier_config else None
    early = receipt_claims(audit) + scope_reasons(mreceipt, a.accept_selected_scope)
    if a.authoritative and not a.carrier_config:
        early.append("no --carrier-config: final data binds to the actual carrier, P, search and budget")
    if a.authoritative and early:          # nothing is extracted for a build that cannot be authoritative
        raise SystemExit("refusing authoritative ingestion: " + "; ".join(early))
    built, stats = build(branch_listing(Path(a.branches), a.teams), Path(a.matches), a.successors)
    dig = digests(Path(a.branches), Path(a.matches), stats["branch_entries"], stats["match_entries"])
    carrier = {"binary_sha256": stats["carriers"], "config": None}
    if cfg_raw is not None:
        cfg = json.loads(cfg_raw)
        carrier["config"] = {"path": a.carrier_config, "sha256": hashlib.sha256(cfg_raw).hexdigest(), "config": cfg,
                             "verification": check_carrier_config(cfg, stats["carriers"])}
    reasons = early + receipt_matches(audit, dig)
    if a.authoritative and reasons:        # checked against the bytes read, before anything is written
        raise SystemExit("refusing authoritative ingestion: " + "; ".join(reasons))
    receipts = {"audit": a.audit, "audit_sha256": audit_sha, "manifest_receipt": a.manifest_receipt,
                "manifest_receipt_sha256": mreceipt_sha,
                "source_alignment_status": (audit or {}).get("source_alignment_status", "no audit receipt given"),
                "manifest_scope": (mreceipt or {}).get("verification_scope", "no manifest receipt given"),
                "accepted_selected_manifest_scope": bool(a.accept_selected_scope and (mreceipt or {}).get("verification_scope")
                                                         == "selected_frozen_identities_only")}
    meta = {"features": F.feature_identity(layout()), "carrier": carrier,
            "source": {"branches": str(a.branches), "matches": str(a.matches), "teams": a.teams, "digests": dig},
            "receipts": receipts, "authoritative": bool(a.authoritative and not reasons), "exploratory_reasons": reasons,
            "carrier_scores": {"roots": stats["roots"], "roots_with_scores": stats["roots_with_carrier_scores"]},
            "repeats_max": stats["repeats_max"], "successors": bool(a.successors)}
    index = write(Path(a.out), built, stats, meta)
    for s, e in index["splits"].items():
        print(f"[decision_dataset] {s}: {e['roots']} roots, {e['candidates']} candidates, {e['maps']} maps")
    print(f"[decision_dataset] dropped roots: {index['drops']['by_reason']}; "
          f"{'authoritative' if index['authoritative'] else 'EXPLORATORY: ' + '; '.join(index['exploratory_reasons'])}")


if __name__ == "__main__":
    main()
