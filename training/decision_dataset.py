"""Versioned root/candidate dataset for own-action decisions (task T0034, docs/RL_PLAN.md section 3).

    python -m training.decision_dataset --branches data/branches/v12_ponly_xxl --matches data/p_only/v_data/matches \\
        --audit bots/p_only/report/v_data/DATA_AUDIT.json --manifest-receipt bots/p_only/report/v_data/MANIFEST_METADATA_ADAPTER.json \\
        --carrier-config build/decision/t0010_carrier.json --out data/decision/t0010_p2

Input: common-prefix branch roots (training.branch_collect or bots/p_only/tools/branch_portfolio.py, schema
ctf-fusion-branches-1) and the recorded matches they were cut from. Every root is linked to its source match
and checked; any inconsistency stops the build with the root's file name (nothing is skipped silently):
match id, team count, map, designated seat, carrier binary hash and INIT must agree; every branch's root block
must be the recorded TURN block of that turn; candidate 0 must be the recorded reply and the only on-policy
candidate (and source "recorded" where sources are stored); requested candidate orders must be distinct; every
repeat's stored outcome must equal the pairwise utility of its stored final score vector.

Root features come from the recorded prefix INIT + TURN 1..t of our seat, the exact stream the bot had read when
it answered turn t (the extractor's history features see every earlier turn). A root is never re-encoded as the
first turn of a one-turn match, and no block after t is read for features. Successor records (the realized
t + 1 state of each candidate) are written only with --successors, to <split>/oracle.npz, for the diagnostic
oracle of training.evaluate_decisions; the scorer's loader never opens that file.

Output (one file per split, splits by map exactly as training.dataset.split_of):
  <out>/<split>/decision.npz   roots and candidates (arrays below); <out>/index.json   identity and counts
Roots: root_match, root_map, root_teams, root_seat, root_turn, root_carrier, root_rec (extractor row), root_first,
  root_count, root_incumbent (candidate index), root_original_outcome (the recorded game's utility, a replay
  diagnostic, not a repeat), root_prefix_mismatch, root_prefix_faults, root_file.
Candidates: cand_root, cand_line (requested order), cand_key (canonical identity), cand_source, cand_on_policy,
  cand_move/cand_act/cand_target/cand_slot (3,), cand_flags (fusion.decision_features), cand_dup_of,
  cand_support, cand_n (repeats), cand_score (C, R, 15; -1 pad), cand_outcome (C, R; NaN pad), cand_faults
  (C, R; -1 pad), cand_y (mean; NaN without repeats), cand_var (sample variance; NaN below two repeats),
  cand_carrier_score (the carrier's own score where a hook logged it; NaN otherwise).
Missing outcomes stay missing (n = 0, NaN); nothing is imputed. Dropped roots are listed with their reasons.

Authority. --authoritative refuses unless the --audit receipt records a passed strict source alignment, and a
manifest receipt whose scope is only selected frozen identities needs --accept-selected-scope (recorded). Without
--authoritative the dataset is marked exploratory, and every report made from it says so.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from fusion import matchlog, native
from fusion.labels import pairwise_outcome
from fusion.protocol import parse_block

SCHEMA = "ctf-fusion-decision-1"
SPLITS = ("train", "val", "test")
MAX_TEAMS = 15
OUTCOME_TOL = 1e-9

ROOT_KEYS = ("root_match", "root_map", "root_teams", "root_seat", "root_turn", "root_carrier", "root_rec", "root_first",
             "root_count", "root_incumbent", "root_original_outcome", "root_prefix_mismatch", "root_prefix_faults",
             "root_file")
CAND_KEYS = ("cand_root", "cand_line", "cand_key", "cand_source", "cand_on_policy", "cand_move", "cand_act", "cand_target",
             "cand_slot", "cand_flags", "cand_dup_of", "cand_support", "cand_n", "cand_score", "cand_outcome",
             "cand_faults", "cand_y", "cand_var", "cand_carrier_score")


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
        need((c, b["repeat"]) not in seen, f"candidate {c} repeat {b['repeat']} appears twice")
        seen.add((c, b["repeat"]))
        need(b.get("root_block") == root_block, f"candidate {c}: root block is not the recorded TURN {turn} block")
        score = b["score"]
        need(len(score) == row["teams"], f"candidate {c}: score vector has {len(score)} teams")
        u = float(pairwise_outcome(score)[seat])
        need(abs(u - float(b["outcome"])) <= OUTCOME_TOL, f"candidate {c}: stored outcome {b['outcome']} != utility {u} of its score")
    return seat


def candidate_source(row: dict, c: int) -> str:
    if row.get("sources") is not None:
        return row["sources"][c]
    if c == 0:
        return "recorded"
    return "move1" if row.get("candidate_mode", "single") == "single" else f"{row['candidate_mode']}:unrecorded"


def build(branch_files, matches: Path, successors: bool = False, log=print) -> tuple[dict, dict]:
    """(per-split arrays, stats); raises LinkError on the first inconsistent root."""
    L = layout()
    F.require_layout(L)
    rows = []
    drops = []
    for f in branch_files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            row = json.load(fh)
        if "dropped" in row:
            drops.append({"file": Path(f).name, "match_id": row.get("match_id"), "turn": row.get("turn"), "reason": row["dropped"]})
            continue
        rows.append((Path(f), row))
    by_match = defaultdict(list)
    for f, row in rows:
        by_match[(row["match_id"], row["teams"])].append((f, row))
    reps = max([1] + [n for _, row in rows for n in Counter(b["candidate"] for b in row["branches"]).values()])
    out = {s: defaultdict(list) for s in SPLITS}
    oracle = {s: defaultdict(list) for s in SPLITS}
    carriers, sources_used, match_files = set(), [], []
    for (match_id, teams), items in sorted(by_match.items()):
        mpath = find_match(matches, match_id, teams)
        match = matchlog.load(mpath)
        match_files.append(mpath)
        size = int(match["init"].split("\n", 1)[0].split()[3])
        for f, row in sorted(items, key=lambda x: x[1]["turn"]):
            seat = check_link(row, match, f.name)
            turn = row["turn"]
            carriers.add(row["carrier_sha256"])
            sources_used.append(f)
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
                              ("root_file", f.name)):
                d[name].append(val)
            for c, (line, cn) in enumerate(zip(cands, canon)):
                bs = sorted(by_cand.get(c, []), key=lambda b: b["repeat"])
                score = np.full((reps, MAX_TEAMS), -1, np.int32)
                outc = np.full(reps, np.nan)
                faults = np.full(reps, -1, np.int16)
                for i, b in enumerate(bs):
                    score[i, :teams] = b["score"]
                    outc[i] = float(pairwise_outcome(b["score"])[seat])
                    faults[i] = len(b.get("faults", []))
                n = len(bs)
                for name, val in (("cand_root", r), ("cand_line", line), ("cand_key", cn["key"]),
                                  ("cand_source", candidate_source(row, c)), ("cand_on_policy", c == 0),
                                  ("cand_move", cn["move"]), ("cand_act", cn["act"]), ("cand_target", cn["target"]),
                                  ("cand_slot", cn["slot"]), ("cand_flags", flags[c]),
                                  ("cand_dup_of", dup[c] + base if dup[c] >= 0 else -1),
                                  ("cand_support", bool(F.supported(flags[c]))), ("cand_n", n), ("cand_score", score),
                                  ("cand_outcome", outc), ("cand_faults", faults),
                                  ("cand_y", float(np.mean(outc[:n])) if n else np.nan),
                                  ("cand_var", float(np.var(outc[:n], ddof=1)) if n >= 2 else np.nan),
                                  ("cand_carrier_score", np.nan)):
                    d[name].append(val)
                if successors:
                    nb = {b["next_block"] for b in bs}
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
    stats = {"drops": drops, "carriers": sorted(carriers), "repeats_max": reps,
             "sources": [str(p) for p in sources_used], "match_files": [str(p) for p in match_files]}
    return {"decision": out, "oracle": oracle if successors else None}, stats


_DTYPES = {"root_map": np.int64, "root_teams": np.int16, "root_seat": np.int16, "root_turn": np.int16, "root_rec": np.int16,
           "root_first": np.int32, "root_count": np.int16, "root_incumbent": np.int32, "root_original_outcome": np.float64,
           "root_prefix_mismatch": np.int32, "root_prefix_faults": np.int32, "cand_root": np.int32, "cand_on_policy": bool,
           "cand_move": np.int16, "cand_act": np.int16, "cand_target": np.int16, "cand_slot": np.int16, "cand_flags": np.int16,
           "cand_dup_of": np.int32, "cand_support": bool, "cand_n": np.int16, "cand_score": np.int32,
           "cand_outcome": np.float64, "cand_faults": np.int16, "cand_y": np.float64, "cand_var": np.float64,
           "cand_carrier_score": np.float64}


def to_arrays(d: dict) -> dict:
    out = {}
    for k in ROOT_KEYS + CAND_KEYS:
        v = d.get(k, [])
        out[k] = np.asarray(v, dtype=_DTYPES[k]) if k in _DTYPES else np.asarray(v, dtype=str)
    return out


def audit_status(audit_path: str | None, receipt_path: str | None) -> dict:
    """What the T0010-style receipts certify; nothing is inferred from current bytes."""
    st = {"audit": audit_path, "manifest_receipt": receipt_path, "source_alignment_verified": False,
          "source_alignment_status": "no audit receipt given", "manifest_scope": "no manifest receipt given"}
    if audit_path:
        a = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        per_split = [v.get("source_row_alignment_verified") for v in (a.get("rank_npz") or {}).values()]
        st["source_alignment_status"] = a.get("source_alignment_status", "unrecorded")
        st["source_alignment_verified"] = bool(per_split) and all(x is True for x in per_split) and \
            st["source_alignment_status"] not in ("pending_host_reaudit", "unrecorded")
        st["audit_sha256"] = sha256_file(Path(audit_path))
    if receipt_path:
        m = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        st["manifest_scope"] = m.get("verification_scope", "unrecorded")
        st["all_other_frozen_fields_exact"] = m.get("all_other_frozen_fields_exact")
        st["manifest_receipt_sha256"] = sha256_file(Path(receipt_path))
    return st


def authority(st: dict, authoritative: bool, accept_selected_scope: bool) -> dict:
    reasons = []
    if not st["source_alignment_verified"]:
        reasons.append(f"strict source alignment not passed ({st['source_alignment_status']})")
    if st["manifest_scope"] != "full_original_manifest_bytes" and st["manifest_scope"] != "normalized_digest_recorded_at_freeze":
        if not (accept_selected_scope and st["manifest_scope"] == "selected_frozen_identities_only"):
            reasons.append(f"manifest verification scope: {st['manifest_scope']}")
    if authoritative and reasons:
        raise SystemExit("refusing authoritative ingestion: " + "; ".join(reasons))
    return {"authoritative": authoritative and not reasons, "exploratory_reasons": reasons,
            "accepted_selected_manifest_scope": bool(accept_selected_scope and st["manifest_scope"] == "selected_frozen_identities_only")}


def manifest_digest(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branches", required=True, help="branch root files (ctf-fusion-branches-1)")
    ap.add_argument("--matches", required=True, help="the recorded matches the roots were cut from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", help="DATA_AUDIT.json-style receipt of the branch data")
    ap.add_argument("--manifest-receipt", help="MANIFEST_METADATA_ADAPTER.json-style receipt (verification scope)")
    ap.add_argument("--authoritative", action="store_true", help="refuse unless the receipts certify strict source alignment")
    ap.add_argument("--accept-selected-scope", action="store_true",
                    help="accept a manifest receipt that verified selected frozen identities only (recorded)")
    ap.add_argument("--carrier-config", help="JSON naming the carrier, P (family, models, identity, temperatures), search "
                                             "settings and budget the roots were played with")
    ap.add_argument("--successors", action="store_true", help="also write realized successor records (diagnostic oracle only)")
    ap.add_argument("--teams", type=int, nargs="*")
    a = ap.parse_args(argv)
    files = sorted(Path(a.branches).rglob("*.json.gz"))
    if a.teams:
        files = [f for f in files if f.parent.name in {f"n{t:02d}" for t in a.teams}]
    st = audit_status(a.audit, a.manifest_receipt)
    auth = authority(st, a.authoritative, a.accept_selected_scope)
    carrier_config = None
    if a.carrier_config:
        raw = Path(a.carrier_config).read_bytes()
        carrier_config = {"path": a.carrier_config, "sha256": hashlib.sha256(raw).hexdigest(), "config": json.loads(raw)}
    elif a.authoritative:
        raise SystemExit("--authoritative needs --carrier-config: final data binds to the actual carrier, P, search and budget")
    built, stats = build(files, Path(a.matches), a.successors)
    meta = {"features": F.feature_identity(layout()), "carrier": {"binary_sha256": stats["carriers"], "config": carrier_config},
            "source": {"branches": str(a.branches), "matches": str(a.matches), "root_files": len(stats["sources"]),
                       "root_files_digest": manifest_digest(stats["sources"]),
                       "match_files_digest": manifest_digest(stats["match_files"])},
            "receipts": st, **auth, "repeats_max": stats["repeats_max"], "successors": bool(a.successors)}
    index = write(Path(a.out), built, stats, meta)
    for s, e in index["splits"].items():
        print(f"[decision_dataset] {s}: {e['roots']} roots, {e['candidates']} candidates, {e['maps']} maps")
    print(f"[decision_dataset] dropped roots: {index['drops']['by_reason']}; "
          f"{'authoritative' if index['authoritative'] else 'EXPLORATORY: ' + '; '.join(index['exploratory_reasons'])}")


if __name__ == "__main__":
    main()
