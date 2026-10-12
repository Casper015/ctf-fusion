"""Branch continuations over the carrier's logged shortlists (task T0036, docs/RL_PLAN.md sections 3, 5 and 6).

    python tools/repo/task.py lock cpu --task T0037 -- python -m training.decision_branch_collect \\
        --frozen build/decision/run1/FROZEN.json --matches data/decision/run1/matches --out data/decision/run1/branches \\
        --repeats 4 --alternatives 3 --workers 6 --ai-test $AI_TEST_ROOT

Input: whole matches collected by training.decision_collect with a log-only carrier (PONLY_DECISION 1,
bots/p_only/profiles/variants_decision.json), whose replies carry the shortlist the carrier actually scored
(bots/p_only/include/decision_hook.hpp, ctf-fusion-shortlist-log-1) after '#'.

Roots. One representative root per match (so per map): among turns away from the first and last five whose log
(a) parses, (b) is bound to the recorded TURN block by its root digest, (c) names the reply's own order first and
(d) has deciding scores for the reply and at least one alternative, the turn with the smallest hash of
(match, turn). Every turn that fails is counted by reason in summary.json (no log, malformed log, digest mismatch,
a stage without deciding scores, an unscored reply, K = 1); nothing is hidden.

Deciding scores by stage (the hook's stage field): refine -> the refined scores of the candidates the rollouts
finished; beam / choose -> the beam scores; search -> the scores of the C08 search survivors only (eliminated plans
were valued on fewer samples). Stages cut, v_cut, duel, none and any "+v" stage have no deciding scores. Missing
scores are never filled in.

Candidates: the reply's own order (on-policy, candidate 0) plus up to --alternatives of the best-scored others, each
a complete legal order exactly as the carrier would have written it. Each is played --repeats times (four to start):
the engine is rebuilt from the match's seeds and stepped with the recorded orders up to turn t - 1 (every reproduced
TURN block must equal the recorded one, or the root is dropped with that reason); every bot is a fresh process fed
the recorded prefix; at turn t the opponents' recorded replies are common to every candidate and our seat plays the
candidate; from t + 1 everything is live under the official 2000/50 ms limits to the end.

Bot state. The engine applies the forced order; the carrier's own memory at turn t reflects its own choice there,
not the forced order (bot_state_mechanism records this approximation: no restore is attempted, and none is
claimed). Opponents are live and reactive after t and may keep adaptive memory. Repeats differ only through the
bots' own wall-clock-dependent search: there is no controlled seed (each repeat records its key, not a seed), so
repeat noise is real continuation noise, never zero by construction.

Per branch: the raw final score vector and its derived utility, the root and successor blocks, faults, and the
prefix replay's mismatches against the recorded replies, both by order (comments stripped) and raw, with the first
examples. Faults are kept by phase (fault_counts): the prefix, the continuation, and the root turn. At the root turn
every bot is still queried under the official limit but its reply is replaced by the recorded or forced order; the
computed replies (root_computed_replies) and their timeouts, missing replies and malformed text
(discarded_root_computed_faults) are retained without changing the forced orders, the candidates or the strikes.
Rows follow ctf-fusion-branches-1 (training.decision_dataset ingests them) and add carrier_scores
(ctf-fusion-shortlist-scores-1, aligned to the played candidates), the parsed shortlist, sources, repeat keys, the
continuation policy tag, both identities and input_binding: digests of the whole input match, the root block and
the recorded prefix (blocks and every seat's replies), plus the selected complete orders and their aligned scores.
Resume reuses a stored row (complete or dropped) only with the same collector identity and an equal input_binding;
a row whose input changed, or one left at a root turn the input no longer selects, is set aside under
<out>/stale/ (counted in summary.json by reason) and replayed; a row of another identity stops the run. The T0035
frozen identities must still verify.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fusion import matchlog
from fusion.labels import pairwise_outcome
from fusion.paths import REPO
from fusion.protocol import parse_block, parse_reply

LOG_VERSION = "ctf-fusion-shortlist-log-1"
SCORES_VERSION = "ctf-fusion-shortlist-scores-1"   # training.decision_dataset.SCORES_VERSION
SCHEMA = "ctf-fusion-branches-1"
MARK = "dsl1"
LIMITS = (2.0, 0.05)
PREFIX_LIMIT = 5.0
MARGIN = 5
CONTINUATIONS = ("official_50ms", "fixed_work_diagnostic")
COLLECTOR_FILES = ("training/decision_branch_collect.py", "training/branch_collect.py", "fusion/runner.py",
                   "bots/p_only/include/decision_hook.hpp")
BOT_STATE = ("engine_forced: the engine applies the forced order at turn t; the carrier's internal memory reflects its "
             "own turn-t choice (no restore attempted; an approximation)")
RNG_NOTE = "no controlled seed: repeats differ through the bots' wall-clock-dependent search; repeat_key identifies each"
NO_SCORES = {"cut": "a deadline cut the refinement: the reply is the beam choice, the refined scores are partial",
             "v_cut": "V was cut: the reply is the choice without V",
             "duel": "C08's two-team planner has no shortlist",
             "none": "no shortlist was recorded at this turn"}


# ---------------------------------------------------------------- logs and roots

def fnv64(text: str) -> str:
    h = 1469598103934665603
    for b in text.encode():
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def root_digest(block_text: str) -> str:
    """The hook's root digest of a recorded TURN block: turn, scores and every unit as read (decision_hook.hpp)."""
    b = parse_block(block_text)
    s = f"{b.turn}|" + ",".join(map(str, b.score)) + "|" + "".join(",".join(map(str, u)) + ";" for u in b.units)
    return fnv64(s)


def order_of(line: str | None) -> str | None:
    """The complete order of a reply line without its turn number and comment."""
    if line is None:
        return None
    return " ".join(line.partition("#")[0].split()[1:])


def _num(x: str) -> float | None:
    if x == "na":
        return None
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"non-finite score {x}")
    return v


def parse_log(line: str | None, turn: int) -> dict | None:
    """The shortlist logged in a reply, or None when the reply has none; ValueError when it is malformed."""
    if line is None or "#" not in line:
        return None
    comment = line.partition("#")[2].strip()
    if not comment.startswith(MARK + " "):
        return None
    parts = comment.split(" | ")
    head = dict(x.split("=", 1) for x in parts[0].split()[1:])
    entries = []
    for p in parts[1:]:
        f = p.split(";")
        if len(f) != 4:
            raise ValueError(f"turn {turn}: malformed shortlist entry {p!r}")
        order, beam, refined, source = f
        if parse_reply(f"{turn} {order}", turn) is None:
            raise ValueError(f"turn {turn}: logged order {order!r} is not a valid reply")
        entries.append({"order": order, "line": f"{turn} {order}", "beam": _num(beam), "refined": _num(refined),
                        "source": source})
    if int(head["k"]) != len(entries) or not entries:
        raise ValueError(f"turn {turn}: k={head['k']} but {len(entries)} entries")
    if len({e["order"] for e in entries}) != len(entries):
        raise ValueError(f"turn {turn}: an order is logged twice")
    if int(head.get("inc", 0)) != 0 or entries[0]["order"] != order_of(line):
        raise ValueError(f"turn {turn}: the first logged order is not the reply's own order")
    return {"version": LOG_VERSION, "stage": head["stage"], "k": len(entries), "root": head["root"],
            "survivors": int(head.get("surv", -1)), "checked": head.get("checked") == "1", "entries": entries}


def deciding_scores(log: dict) -> tuple[dict | None, str | None]:
    """(the entries with deciding scores and their scale, None) or (None, the reason there are none)."""
    stage = log["stage"]
    if stage in NO_SCORES or "+v" in stage:
        return None, NO_SCORES.get(stage, "V re-ranked the logged candidates: the logged scores did not decide")
    if stage == "refine":
        scale, keep = "refined", [e for e in log["entries"] if e["refined"] is not None]
        value = lambda e: e["refined"]  # noqa: E731
    elif stage in ("beam", "choose"):
        scale, keep = "beam", [e for e in log["entries"] if e["beam"] is not None]
        value = lambda e: e["beam"]  # noqa: E731
    elif stage == "search":
        scale, keep = "search_survivors", [e for e in log["entries"] if e["source"] == "survivor" and e["beam"] is not None]
        value = lambda e: e["beam"]  # noqa: E731
    else:
        return None, f"stage {stage}: no deciding scores"
    if not keep or keep[0] is not log["entries"][0]:
        return None, "the reply's own order has no deciding score"
    if len(keep) < 2:
        return None, "fewer than two scored orders (K = 1)"
    return {"scale": scale, "stage": stage, "checked": log["checked"],
            "entries": [dict(e, score=value(e)) for e in keep]}, None


def select_candidates(scored: dict, alternatives: int) -> list[dict]:
    """The reply's order, then the best-scored others (ties by order text): at most 1 + alternatives."""
    inc, rest = scored["entries"][0], scored["entries"][1:]
    rest = sorted(rest, key=lambda e: (-e["score"], e["order"]))[:alternatives]
    return [inc, *rest]


def designated_seat(match: dict) -> int:
    from training.branch_collect import designated_seat as d  # noqa: PLC0415
    return d(match)


def unit_hash(*parts) -> int:
    from training.branch_collect import unit_hash as h  # noqa: PLC0415
    return h(*parts)


def pick_root(match: dict, seat: int) -> tuple[dict | None, dict]:
    """(the chosen root, counts of turns by outcome)."""
    reasons: dict[str, int] = {}
    ok = []
    turns = len(match["blocks"])
    for t in range(1 + MARGIN, max(turns - MARGIN, 1 + MARGIN)):
        line = match["replies"][t - 1][seat]
        try:
            log = parse_log(line, t)
        except (ValueError, KeyError) as exc:
            reasons[f"malformed log: {str(exc)[:60]}"] = reasons.get(f"malformed log: {str(exc)[:60]}", 0) + 1
            continue
        if log is None:
            reasons["no log"] = reasons.get("no log", 0) + 1
            continue
        if log["root"] != root_digest(match["blocks"][t - 1]):
            reasons["root digest differs from the recorded block"] = reasons.get("root digest differs from the recorded block", 0) + 1
            continue
        scored, why = deciding_scores(log)
        if scored is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        reasons["usable"] = reasons.get("usable", 0) + 1
        ok.append((unit_hash(match["match_id"], "decision-root", t), t, log, scored))
    if not ok:
        return None, reasons
    _, t, log, scored = min(ok)
    return {"turn": t, "log": log, "scored": scored}, reasons


# ---------------------------------------------------------------- branches

def play_branch(engine, match: dict, seat: int, turn: int, line: str, commands: list[list[str]],
                limits: tuple[float, float]) -> dict:
    """Replay the recorded prefix into fresh bots, force `line` for our seat at `turn`, play live to the end."""
    from fusion import runner  # noqa: PLC0415
    from training.branch_collect import recorded_orders  # noqa: PLC0415
    game = engine.Game(match["teams"], match["map_seed"], flag_seed=match["flag_seed"])
    names = [s["name"] for s in match["seats"]]
    bots = [runner.Bot(cmd, name, (limits[0], PREFIX_LIMIT)) for cmd, name in zip(commands, names)]
    by_order = raw = prefix_faults = 0
    examples, faults, root_faults = [], [], []
    root_block = next_block = root_computed = None
    try:
        while not game.done:
            t = game.turn + 1
            inputs = [game.encode(i) for i in range(match["teams"])]
            if t == turn:
                root_block = inputs[0] if t > 1 else matchlog.split_init(inputs[0])[1]
                for b in bots:
                    b.turn_limit = limits[1]
                    b.strikes = 0
            if t == turn + 1:
                next_block = inputs[0]
            starts = [b.send(text) for b, text in zip(bots, inputs)]
            lines = [b.receive(t0, t) for b, t0 in zip(bots, starts)]
            if t < turn:
                for i, (got, rec) in enumerate(zip(lines, match["replies"][t - 1])):
                    raw += got != rec
                    if order_of(got) != order_of(rec):
                        by_order += 1
                        if len(examples) < 5:
                            examples.append({"turn": t, "seat": i, "recorded": rec, "replayed": got})
                prefix_faults += sum(b.faults[-1][0] == t for b in bots if b.faults)
                orders = recorded_orders(game, match, t)
            else:
                if t == turn:
                    # Every bot still answers the root turn under the official limit, but the engine plays the
                    # recorded orders and our forced candidate. Keep what the bots computed, and every fault of
                    # those discarded replies (timeouts, missing replies, malformed text), apart from the prefix and
                    # continuation faults; nothing here changes the forced orders, the candidates or the strikes.
                    root_computed = list(lines)
                    for i, (b, got) in enumerate(zip(bots, lines)):
                        reasons = [r for tt, r in b.faults if tt == t]
                        if got is None and not reasons:
                            reasons.append(f"no reply ({b.dead})" if b.dead else "no reply")
                        if got is not None:
                            try:
                                game.decode(i, got)
                            except ValueError as exc:
                                reasons.append(f"malformed output: {exc}")
                        root_faults += [[t, i, r] for r in reasons]
                    lines = list(match["replies"][t - 1])
                    lines[seat] = line
                orders = {}
                for i, (b, ln) in enumerate(zip(bots, lines)):
                    if ln is None:
                        orders[i] = {}
                        continue
                    try:
                        orders[i] = game.decode(i, ln)
                        b.ok()
                    except ValueError as exc:
                        orders[i] = {}
                        if t > turn:
                            b.strike(t, f"malformed output: {exc}")
                faults += [[t, i, reason] for i, b in enumerate(bots) for tt, reason in b.faults if tt == t and t > turn]
            game.step(orders)
        score = game.result()["score"]
    finally:
        for b in bots:
            b.close()
    return {"line": line, "score": score, "outcome": float(pairwise_outcome(score)[seat]), "prefix_mismatch": by_order,
            "prefix_mismatch_raw": raw, "prefix_examples": examples, "prefix_faults": prefix_faults, "faults": faults,
            "discarded_root_computed_faults": root_faults, "root_computed_replies": root_computed,
            "fault_counts": {"prefix": prefix_faults, "root_discarded": len(root_faults), "continuation": len(faults)},
            "root_block": root_block, "next_block": next_block}


def collector_identity(frozen: dict, repeats: int, alternatives: int, continuation: str) -> str:
    from training.decision_collect import runner_identity  # noqa: PLC0415
    from training.decision_opponents import digest, sha256_file  # noqa: PLC0415
    return digest({"frozen": frozen["identity"], "files": {f: sha256_file(REPO / f) for f in COLLECTOR_FILES},
                   "t0035_runner": runner_identity(), "repeats": repeats, "alternatives": alternatives,
                   "continuation": continuation, "limits_ms": [2000, 50], "log": LOG_VERSION, "margin": MARGIN})


def check_sources(match: dict, seat: int, frozen: dict) -> None:
    """Every seat of the match must be the frozen bot it names (the designated binary, the opponents' sources)."""
    for i, s in enumerate(match["seats"]):
        if i == seat:
            want = frozen["designated"]["binary_sha256"]
        else:
            o = frozen["opponents"].get(s["name"])
            if o is None:
                raise SystemExit(f"{match['match_id']}: seat {i} ({s['name']}) is not a frozen opponent")
            want = o.get("source_sha256", o["binary_sha256"])
        if s["sha256"] != want:
            raise SystemExit(f"{match['match_id']}: seat {i} ({s['name']}) was played by {s['sha256'][:12]}, "
                             f"the frozen identity is {want[:12]}")


def row_path(out: Path, match: dict, turn: int) -> Path:
    return Path(out) / f"n{match['teams']:02d}" / f"{match['match_id']}-t{turn:03d}.json.gz"


def write_gz(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def input_binding(match: dict, seat: int, turn: int, chosen: dict, cands: list[dict]) -> dict:
    """What a row was played from: the whole input match, the root block, the recorded prefix (blocks and every
    seat's replies before the root), and the selected complete orders with their aligned carrier scores. A stored
    row (completed or dropped) is reused only when every field is equal."""
    return {"match_sha256": _sha(match), "root_block_sha256": _sha(match["blocks"][turn - 1]),
            "prefix_sha256": _sha({"init": match.get("init"), "blocks": match["blocks"][:turn - 1],
                                   "replies": match["replies"][:turn - 1]}),
            "seat": seat, "turn": turn, "root_digest": chosen["log"]["root"], "stage": chosen["scored"]["stage"],
            "scale": chosen["scored"]["scale"], "candidates": [e["line"] for e in cands],
            "scores": [e["score"] for e in cands]}


def binding_differences(row: dict, binding: dict) -> list[str]:
    """The binding fields a stored row does not share with the current input (all of them for an unbound row)."""
    old = row.get("input_binding")
    if not isinstance(old, dict):
        return ["unbound"]
    return sorted(k for k in set(old) | set(binding) if old.get(k) != binding.get(k))


def set_aside(path: Path, out: Path, kind: str) -> Path:
    """Move a stored row out of the dataset's glob (``<out>/<kind>/<name>.<time>``): never deleted, never ingested."""
    dest = out / kind / f"{path.name}.{time.time_ns()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), dest)
    return dest


def is_complete(row: dict, repeats: int) -> bool:
    if "dropped" in row:
        return True
    have = {(b["candidate"], b["repeat"]) for b in row["branches"]}
    return have == {(c, r) for c in range(len(row["candidates"])) for r in range(repeats)}


def root_job(job) -> dict:
    """Play every candidate and repeat of one root (a worker)."""
    (mpath, turn, cands, sources, repeats, commands, root, play_fn, reproduce_fn) = job
    match = matchlog.load(mpath)
    seat = designated_seat(match)
    engine = None
    if play_fn is None or reproduce_fn is None:
        from fusion.paths import load_engine  # noqa: PLC0415
        engine = load_engine(Path(root))
    from training.branch_collect import reproduce  # noqa: PLC0415
    if not (reproduce_fn or (lambda e, m, t: reproduce(e, m, t) is not None))(engine, match, turn):
        return {"dropped": "recorded blocks not reproduced"}
    branches = []
    for c, line in enumerate(cands):
        for r in range(repeats):
            b = (play_fn or play_branch)(engine, match, seat, turn, line, commands, LIMITS)
            branches.append({"candidate": c, "repeat": r, "on_policy": c == 0, "source": sources[c],
                             "repeat_key": f"{match['match_id']}/t{turn}/c{c}/r{r}"} | b)
    return {"branches": branches}


def collect(frozen_path: Path, matches: Path, out: Path, root: Path | None, repeats: int = 4, alternatives: int = 3,
            workers: int = 1, continuation: str = "official_50ms", play_fn=None, reproduce_fn=None, verify_fn=None,
            dry_run: bool = False) -> dict:
    from training.decision_collect import commands_and_info, verify  # noqa: PLC0415
    from training.decision_dataset import SCORES_VERSION as DATASET_SCORES  # noqa: PLC0415
    if DATASET_SCORES != SCORES_VERSION:
        raise SystemExit("shortlist score versions differ between the collector and the dataset builder")
    if continuation not in CONTINUATIONS:
        raise SystemExit(f"--continuation must be one of {CONTINUATIONS}")
    if repeats < 1 or alternatives < 1:
        raise SystemExit("--repeats and --alternatives must be positive")
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    bad = (verify_fn or verify)(frozen, root)
    if bad:
        raise SystemExit("refusing to collect branches: " + "; ".join(bad))
    ident = collector_identity(frozen, repeats, alternatives, continuation)
    commands_by_name, _ = commands_and_info(frozen)
    out = Path(out)
    summary = {"collector_identity": ident, "frozen_identity": frozen["identity"], "repeats": repeats,
               "alternatives": alternatives, "continuation_policy": continuation, "matches": 0, "roots": 0,
               "no_root": {}, "turn_reasons": {}, "complete_before": 0, "played": 0, "dropped": {}, "stale": {}}

    def stored(path: Path) -> dict:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            row = json.load(fh)
        if row.get("collector_identity") != ident:
            raise SystemExit(f"{path}: a row of another collector identity; collect into a new --out")
        return row

    def stale(path: Path, why: str) -> None:
        set_aside(path, out, "stale")
        summary["stale"][why] = summary["stale"].get(why, 0) + 1

    jobs, meta = [], []
    for mp in sorted(Path(matches).rglob("*.json.gz")):
        match = matchlog.load(mp)
        summary["matches"] += 1
        seat = designated_seat(match)
        check_sources(match, seat, frozen)
        chosen, reasons = pick_root(match, seat)
        for k, v in reasons.items():
            summary["turn_reasons"][k] = summary["turn_reasons"].get(k, 0) + v
        rp = row_path(out, match, chosen["turn"]) if chosen else None
        # A stored row of this match at another root turn was played from an input that no longer selects it.
        for other in sorted(row_path(out, match, 0).parent.glob(f"{match['match_id']}-t*.json.gz")):
            if other != rp:
                stored(other)
                stale(other, "root turn changed")
        if chosen is None:
            why = "no usable shortlist in any turn"
            summary["no_root"][why] = summary["no_root"].get(why, 0) + 1
            continue
        summary["roots"] += 1
        t = chosen["turn"]
        cands = select_candidates(chosen["scored"], alternatives)
        binding = input_binding(match, seat, t, chosen, cands)
        if rp.exists():
            old = stored(rp)
            diff = binding_differences(old, binding)
            if diff:
                stale(rp, "input changed: " + ",".join(diff))
            elif is_complete(old, repeats):
                summary["complete_before"] += 1
                continue
            else:
                set_aside(rp, out, "partial")
        lines = [e["line"] for e in cands]
        sources = ["recorded"] + [f"shortlist:{e['source']}" for e in cands[1:]]
        commands = [commands_by_name["@candidate" if i == seat else s["name"]] for i, s in enumerate(match["seats"])]
        jobs.append((str(mp), t, lines, sources, repeats, commands, str(root) if root else None, play_fn, reproduce_fn))
        meta.append((match, seat, t, chosen, cands, rp, binding))
    print(f"[branches] {summary['matches']} matches, {summary['roots']} roots, {summary['complete_before']} complete, "
          f"{sum(summary['stale'].values())} stale set aside, "
          f"{len(jobs)} to play ({len(jobs) * repeats * (1 + alternatives)} continuations at most)", flush=True)
    if not dry_run and jobs:
        pool = ProcessPoolExecutor(workers) if workers > 1 and play_fn is None else None
        results = pool.map(root_job, jobs, chunksize=1) if pool else map(root_job, jobs)
        for (match, seat, t, chosen, cands, rp, binding), res in zip(meta, results):
            base = {"schema": SCHEMA, "match_id": match["match_id"], "teams": match["teams"], "map_seed": match["map_seed"],
                    "seat": seat, "turn": t, "collector_identity": ident, "frozen_identity": frozen["identity"],
                    "input_binding": binding}
            if "dropped" in res:
                write_gz(rp, base | {"dropped": res["dropped"]})
                summary["dropped"][res["dropped"]] = summary["dropped"].get(res["dropped"], 0) + 1
                continue
            row = base | {
                "init": match["init"], "carrier_sha256": match["seats"][seat]["sha256"],
                "candidates": [e["line"] for e in cands],
                "sources": ["recorded"] + [f"shortlist:{e['source']}" for e in cands[1:]],
                "candidate_mode": "shortlist", "branches": res["branches"],
                "root_outcome": float(pairwise_outcome(match["result"]["score"])[seat]),
                "carrier_scores": {"version": SCORES_VERSION, "k": len(cands), "orders": [e["line"] for e in cands],
                                   "scores": [e["score"] for e in cands], "scale": chosen["scored"]["scale"],
                                   "stage": chosen["scored"]["stage"], "checked": chosen["scored"]["checked"],
                                   "root_digest": chosen["log"]["root"]},
                "shortlist": chosen["log"], "repeats": len({b["repeat"] for b in res["branches"]}),
                "bot_state_mechanism": BOT_STATE, "rng": RNG_NOTE, "continuation_policy": continuation}
            write_gz(rp, row)
            summary["played"] += 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True, help="training.decision_collect FROZEN.json of the collection")
    ap.add_argument("--matches", required=True, help="that collection's matches directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--alternatives", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--continuation", choices=CONTINUATIONS, default="official_50ms",
                    help="fixed_work_diagnostic only for a separately frozen fixed-work carrier, never pooled with 50 ms")
    ap.add_argument("--ai-test")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.ai_test) if a.ai_test else (Path(os.environ["AI_TEST_ROOT"]) if os.environ.get("AI_TEST_ROOT") else None)
    if root is None:
        raise SystemExit("pass --ai-test or set AI_TEST_ROOT (the engine is replayed)")
    collect(Path(a.frozen), Path(a.matches), Path(a.out), root, a.repeats, a.alternatives, a.workers, a.continuation,
            dry_run=a.dry_run)


if __name__ == "__main__":
    main()
