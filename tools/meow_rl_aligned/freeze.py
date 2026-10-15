"""Freeze schedules A and B of the aligned Meowmind RL screens (task T0061, docs/MEOWMIND_RL_ALIGNED.md; host only).

    python tools/meow_rl_aligned/freeze.py --seed-a <ledgered> --seed-b <ledgered> --out reports/host/meow_rl_aligned/FREEZE.json

Run after the eight-worker preflight passed and before any screen game. It refuses unless:
  * the preflight receipt passed at 8 workers and measured exactly the binaries now built for every arm;
  * the two seeds are distinct, listed in docs/MEOWMIND_RL_SCREEN_SEEDS.md, and neither is the preflight's seed;
  * the held-out roster passes the audit below.
Then it writes each seed into its profile (profiles/meow_rl_aligned_screen_{a,b}.json, the only change it makes), and
FREEZE.json: the arms and their bundle/binary sha256, the comparisons and families, margins, multiplicity, workers,
fixture and game budgets, the preflight receipt's sha256 and the roster audit. The screens then run with
bots/p_only/tools/run_screen.py --config profiles/meow_rl_aligned_screen_a.json (and b); nothing frozen here may
change after the first game.

Roster audit (the five held-out opponents of profiles/meow_rl_screen.json): for each, its family in the roster
registry and its source sha256 in the AI_Test checkout; it must not appear in either training pool
(profiles/meow_rl_training_pool.json, profiles/meow_rl_training_pool_v2.json) by name, path or source sha256, and its
source must not carry the Meowmind reference's distinctive identifiers (a copy or descendant of the actor under
test). Families shared with a training pool are listed (descriptive: a family label is not an ancestry). A marker scan
is a screen, not a proof of independence; a hit blocks the freeze until a human decides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROFILES = {"a": REPO / "profiles" / "meow_rl_aligned_screen_a.json", "b": REPO / "profiles" / "meow_rl_aligned_screen_b.json"}
POOLS = [REPO / "profiles" / "meow_rl_training_pool.json", REPO / "profiles" / "meow_rl_training_pool_v2.json"]
LEDGER = REPO / "docs" / "MEOWMIND_RL_SCREEN_SEEDS.md"
MARKERS = ("NN_JOINT_BELIEF", "OPPONENT_HEADS", "NN_ACTION_EVIDENCE", "workingExperts", "headPosterior", "meowmind",
           "Meowmind", "meow_rl", "policyMoves")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool_bots(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    bots = doc.get("bots", [])
    if isinstance(bots, dict):
        bots = [dict(v, name=k) for k, v in bots.items()]
    return bots


def audit(roster: list[str], root: Path) -> dict:
    from fusion.paths import bot_path, load_pool  # noqa: PLC0415
    registry = {e["name"]: e for e in load_pool()["bots"]}
    training = [(p.name, b) for p in POOLS for b in pool_bots(p)]
    t_names = {b.get("name") for _, b in training}
    t_paths = {b.get("path") for _, b in training}
    t_shas = {b.get("sha256") for _, b in training if b.get("sha256")}
    t_families = {b.get("family") for _, b in training}
    rows, blocked = [], []
    for name in roster:
        e = registry[name]
        path = bot_path(e, root)
        text = path.read_text(encoding="utf-8", errors="replace")
        sha = sha256(path)
        hits = sorted({m for m in MARKERS if re.search(re.escape(m), text)})
        row = {"name": name, "family": e.get("family"), "path": e.get("path"), "source_sha256": sha,
               "in_training_pool": {"name": name in t_names, "path": e.get("path") in t_paths, "sha256": sha in t_shas},
               "family_in_training_pool": e.get("family") in t_families, "reference_markers": hits}
        if any(row["in_training_pool"].values()) or hits:
            blocked.append(name)
        rows.append(row)
    return {"roster": rows, "blocked": blocked, "pools": {p.name: sha256(p) for p in POOLS}, "markers": MARKERS}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-a", type=int, required=True)
    ap.add_argument("--seed-b", type=int, required=True)
    ap.add_argument("--preflight", default=str(REPO / "reports" / "host" / "meow_rl_aligned" / "PREFLIGHT.json"))
    ap.add_argument("--out", default=str(REPO / "reports" / "host" / "meow_rl_aligned" / "FREEZE.json"))
    ap.add_argument("--ai-test")
    a = ap.parse_args(argv)
    from fusion.paths import ai_test_root  # noqa: PLC0415
    sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))
    from run_screen import check_preflight  # noqa: PLC0415
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists: the screens are frozen once")
    if a.seed_a == a.seed_b:
        raise SystemExit("schedules A and B need distinct seeds")
    ledger = LEDGER.read_text(encoding="utf-8")
    receipt = json.loads(Path(a.preflight).read_text(encoding="utf-8"))
    for s in (a.seed_a, a.seed_b):
        if str(s) not in ledger:
            raise SystemExit(f"seed {s} is not reserved in {LEDGER.relative_to(REPO)}")
        if s == receipt.get("seed"):
            raise SystemExit(f"seed {s} is the preflight's seed")
    manifest = json.loads((REPO / "build" / "p_only" / "manifest.json").read_text(encoding="utf-8"))["variants"]
    cfgs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in PROFILES.items()}
    root = ai_test_root(a.ai_test)
    roster = cfgs["a"]["reference"]
    if cfgs["b"]["reference"] != roster:
        raise SystemExit("schedules A and B must use the same held-out roster")
    report = audit(roster, root)
    if report["blocked"]:
        raise SystemExit(f"roster audit blocks the freeze: {report['blocked']} (see the audit; a human decides)")
    frozen = {}
    for k, cfg in cfgs.items():
        if cfg.get("seed") is not None:
            raise SystemExit(f"{PROFILES[k].name} already has seed {cfg['seed']}: it is frozen")
        arms = cfg["arms"] + cfg.get("aa", [])
        check_preflight(Path(a.preflight), manifest, arms, cfg["workers"])
        cfg["seed"] = a.seed_a if k == "a" else a.seed_b
        frozen[k] = {"profile": PROFILES[k].relative_to(REPO).as_posix(), "seed": cfg["seed"],
                     "arms": {x: {"bundle_sha256": manifest[x]["bundle_sha256"], "binary_sha256": manifest[x]["binary_sha256"]}
                              for x in arms},
                     "aa": cfg.get("aa", []), "comparisons": cfg["comparisons"], "families": cfg["families"],
                     "multiplicity": cfg.get("multiplicity"), "margin": cfg.get("margin", 0.0), "workers": cfg["workers"],
                     "fixtures": len(cfg["teams"]) * cfg["maps_per_count"],
                     "games": len(cfg["teams"]) * cfg["maps_per_count"] * len(arms)}
        if frozen[k]["fixtures"] != cfg["fixtures"] or frozen[k]["games"] != cfg["games"]:
            raise SystemExit(f"{PROFILES[k].name}: fixtures/games differ from its stated budget")
    for k, cfg in cfgs.items():
        PROFILES[k].write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
        frozen[k]["profile_sha256"] = sha256(PROFILES[k])
    out = {"schema": "ctf-meow-rl-aligned-freeze-1", "generator": "tools/meow_rl_aligned/freeze.py",
           "preflight": {"path": Path(a.preflight).relative_to(REPO).as_posix() if Path(a.preflight).is_relative_to(REPO) else a.preflight,
                         "sha256": sha256(Path(a.preflight)), "seed": receipt["seed"], "workers": receipt["workers"]},
           "schedules": frozen, "roster_audit": report, "total_games": sum(f["games"] for f in frozen.values())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in main().items() if k != "roster_audit"}, indent=1))
