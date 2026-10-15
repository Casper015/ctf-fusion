"""Record calibration observations for the integer export from fresh games of a frozen actor (host; T0056).

    python bots/meow_rl/calibrate.py --checkpoint runs/meow_rl/large_v_s101/checkpoints/final.pt \\
        --pool profiles/meow_rl_training_pool.json --seed-base <reserved> --seed-ledger docs/MEOWMIND_RL_SEEDS.md \\
        --episodes 40 --workers 6 --out data/meow_rl_calibration/large_v_s101
    python bots/meow_rl/export.py --checkpoint .../final.pt --name large_v_s101 --calibration-batches data/meow_rl_calibration/large_v_s101

For runs trained without --keep-features (T0055): the frozen actor plays --episodes complete games against the training
pool through training.meow_rl's own collector (same bridge, sampling, opponents and limits), and every decision's actor
and value features are saved as out/update_<episode>.npz, the layout export.py --calibration-batches reads. The games
only describe the states the deployed actor reaches; they train nothing and are never evidence. The seed base must be
reserved in the ledger for calibration (never a training, screen or consumed seed). out/RECEIPT.json records the
actor, pool, seed and every file's sha256. Loading a .pt checkpoint needs PyTorch; --actor-npz takes NumPy arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "meow_rl"))

from fusion import paths  # noqa: E402
from training.meow_rl import bridge as B  # noqa: E402
from training.meow_rl import identity as I  # noqa: E402
from training.meow_rl import policy as P  # noqa: E402
from training.meow_rl.pool import Pool  # noqa: E402
from training.meow_rl.store import atomic_json, atomic_npz  # noqa: E402
from training.meow_rl.train import ProcessCollector, check_ledger  # noqa: E402


def actor_from(args) -> tuple[dict, dict]:
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("meow_rl_export", REPO / "bots" / "meow_rl" / "export.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    if args.checkpoint:
        actor, _, identity = E.load_checkpoint(Path(args.checkpoint))
        src = {"checkpoint": E.portable(args.checkpoint), "sha256": E.sha256(Path(args.checkpoint))}
    else:
        actor = dict(np.load(args.actor_npz))
        identity = json.loads(Path(args.identity_json).read_text(encoding="utf-8"))
        src = {"actor_npz": E.portable(args.actor_npz), "sha256": E.sha256(Path(args.actor_npz))}
    E.check_identity(identity)
    return P.check_actor(actor), src


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--actor-npz")
    ap.add_argument("--identity-json", help="with --actor-npz: the run identity")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--seed-base", type=int, required=True, help="reserved in the ledger for this calibration")
    ap.add_argument("--seed-ledger", help="required unless --dev")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--counts", type=int, nargs="+", default=list(range(2, 16)))
    ap.add_argument("--limits", type=float, nargs=2, default=[2.0, 0.05])
    ap.add_argument("--ai-test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", action="store_true", help="development plumbing: unledgered seed allowed")
    a = ap.parse_args(argv)
    if a.actor_npz and not a.identity_json:
        ap.error("--actor-npz needs --identity-json")
    ledger = check_ledger(a.seed_base, a.seed_ledger, a.dev)
    actor, source = actor_from(a)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: calibration games are never mixed or resumed")
    out.mkdir(parents=True, exist_ok=True)
    pool = Pool.load(Path(a.pool))
    root = paths.ai_test_root(a.ai_test)
    commands, info = pool.commands(root, I.REPO / "build" / "meow_rl" / "bots")
    lib = B.build()
    collector = ProcessCollector(a.workers, commands, info, a.limits, root, lib)
    index = iter(range(a.episodes))
    started: list[int] = []
    try:
        results = collector.run(lambda: pool.episode(a.seed_base, next(index), a.counts), actor, 0,
                                a.episodes * 400, started.extend) if a.episodes else []
    finally:
        collector.close()
    bad = [r for r in results if r["status"] != "ok"]
    if bad:
        raise SystemExit(f"{bad[0]['status']}: {bad[0].get('error')} (nothing written)")
    files = {}
    for r in results:
        path = out / f"update_{r['index']:06d}.npz"
        atomic_npz(path, {"feats": r["arrays"]["feats"], "value_feats": r["arrays"]["value_feats"],
                          "turn": r["arrays"]["turn"], "action": r["arrays"]["action"]})
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"schema": "ctf-meow-rl-calibration-1", "actor": source, "pool_sha256": I.sha256_file(Path(a.pool)),
               "seed_base": a.seed_base, "seed_ledger": ledger, "episodes": len(results),
               "decisions": int(sum(len(r["arrays"]["action"]) for r in results)), "counts": a.counts, "limits": a.limits,
               "code_digest": I.sources()["code_digest"], "files": files, "dev": a.dev,
               "faults": int(sum(len(r["record"]["faults"]) for r in results))}
    atomic_json(out / "RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in main().items() if k != "files"}, indent=1))
