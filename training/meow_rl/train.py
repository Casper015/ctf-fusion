"""Host entry point of the Meowmind PPO pilot: CPU rollout workers, CUDA learning (docs/MEOWMIND_RL_IMPLEMENTATION.md).

    # 1. correctness smoke: 20,000 learner decisions, then checks (smoke_report.json)
    python -m training.meow_rl.train --mode smoke --arm large_v --pool profiles/meow_rl_training_pool.json \\
        --seed-base <reserved seed> --seed-ledger docs/MEOWMIND_RL_SEEDS.md --device cuda --workers 8 \\
        --out runs/meow_rl/large_v_s101
    # 2. continue the same run to the full budget (1,000,000 decisions in total, smoke included)
    python -m training.meow_rl.train --mode full --resume <same arguments>

The arena is --ai-test or AI_TEST_ROOT. Each update freezes one actor snapshot, collects complete new episodes with
it in `--workers` spawned CPU processes (episode indices 0, 1, ... of the seed base, never reused), computes old
values on the device, checks that the learner recomputes the stored log probabilities, runs GAE and one PPO update,
and commits (training.meow_rl.store). Checkpoints at every checkpoint_interval decisions and at the budget; the
final checkpoint is the predeclared candidate. Budgets count learner decisions (one per turn), never all-seat
replies. --dev allows a seed outside the ledger and marks the run as development plumbing (never evidence).

Run it inside the task's leases: python tools/repo/task.py lock gpu --task T0055 -- python -m training.meow_rl.train ...
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from fusion import paths, runner
from . import bridge as B
from . import identity as I
from .env import collect
from .gae import gae
from .pool import Pool

LOGP_TOLERANCE = 1e-4          # stored (NumPy snapshot) vs recomputed (PyTorch learner) old log probability


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--arm", choices=sorted(I.CRITIC_ARCH), required=True)
    ap.add_argument("--out", required=True, help="run directory (under runs/)")
    ap.add_argument("--pool", required=True, help="frozen opponent pool (schema ctf-meow-rl-pool-1)")
    ap.add_argument("--seed-base", type=int, required=True, help="game-seed base reserved in the ledger")
    ap.add_argument("--seed-ledger", help="ledger file that must list --seed-base (required unless --dev)")
    ap.add_argument("--ai-test", help="arena root (default AI_TEST_ROOT)")
    ap.add_argument("--profile", default=str(I.PROFILE))
    ap.add_argument("--device", default="cuda", help="cuda, cpu or auto")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls (a recorded variant; the pilot is FP32)")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--decisions", type=int, help="override the mode's total budget (tests and dry runs only; recorded)")
    ap.add_argument("--rollout-decisions", type=int, help="override the profile's decisions per update (recorded)")
    ap.add_argument("--minibatch", type=int, help="override the minibatch after a measured memory failure (recorded)")
    ap.add_argument("--limits", type=float, nargs=2, default=[2.0, 0.05], metavar=("FIRST_S", "TURN_S"),
                    help="opponent reply limits during collection")
    ap.add_argument("--counts", type=int, nargs="+", help="team counts (default: the profile's 2-15)")
    ap.add_argument("--keep-features", action="store_true", help="also store actor/value features per batch (large)")
    ap.add_argument("--resume", action="store_true", help="continue the run in --out")
    ap.add_argument("--dev", action="store_true", help="development plumbing: unledgered seed allowed, never evidence")
    return ap.parse_args(argv)


def check_ledger(seed: int, ledger: str | None, dev: bool) -> dict | None:
    if ledger is None:
        if not dev:
            raise SystemExit("--seed-ledger is required: reserve the seed base in the ledger before consuming it")
        return None
    text = Path(ledger).read_text(encoding="utf-8")
    if not re.search(rf"(?<!\d){seed}(?!\d)", text):
        raise SystemExit(f"seed base {seed} is not in {ledger}: reserve it before the run")
    return {"path": str(ledger), "sha256": I.sha256_file(Path(ledger))}


class ProcessCollector:
    """Complete episodes from spawned CPU workers (no CUDA there); results ordered by episode index."""

    def __init__(self, workers: int, commands: dict, info: dict, limits, root: Path, lib_path: Path):
        self.workers, self.commands, self.info = workers, commands, info
        self.limits, self.root, self.lib_path = tuple(limits), str(root), str(lib_path)
        self.pool = mp.get_context("spawn").Pool(workers)
        self.mean_len = 400.0

    def close(self):
        self.pool.terminate()
        self.pool.join()

    def run(self, next_spec, actor: dict, version: int, target: int, on_start) -> list[dict]:
        inflight, results, collected = {}, [], 0
        while True:
            while len(inflight) < self.workers and collected + len(inflight) * self.mean_len < target:
                spec = next_spec()
                on_start([spec.index])
                job = (spec, actor, version, self.commands, self.info, self.limits, self.root, self.lib_path)
                inflight[spec.index] = self.pool.apply_async(collect, (job,))
            if not inflight:
                break
            done = [i for i, r in inflight.items() if r.ready()]
            if not done:
                time.sleep(0.02)
                continue
            for i in done:
                res = inflight.pop(i).get()
                results.append(res)
                if res["status"] == "ok":
                    n = len(res["arrays"]["action"])
                    collected += n
                    self.mean_len += 0.2 * (n - self.mean_len)
        return sorted(results, key=lambda r: r["index"])


def make_batch(results: list[dict], learner, cfg, keep_features: bool) -> tuple[dict, dict, dict]:
    """Concatenate episodes, compute old values and GAE per episode, verify the stored log probabilities."""
    arr = {k: np.concatenate([r["arrays"][k] for r in results]) for k in results[0]["arrays"]}
    lengths = [len(r["arrays"]["action"]) for r in results]
    episode = np.repeat([r["index"] for r in results], lengths).astype(np.int64)
    values = learner.values(arr["value_feats"])
    recomputed = learner.log_probs(arr["feats"], arr["mask"])
    stored = arr["logp"].astype(np.float32)
    again = recomputed[np.arange(len(stored)), arr["action"].astype(np.int64)]
    gap = float(np.max(np.abs(again - stored)))
    if not np.isfinite(again).all() or gap > LOGP_TOLERANCE:
        raise RuntimeError(f"behavior/learner log-probability mismatch {gap:.3g} (> {LOGP_TOLERANCE}): batch invalid")
    adv, ret = np.zeros_like(values), np.zeros_like(values)
    start = 0
    for n in lengths:
        sl = slice(start, start + n)
        adv[sl], ret[sl] = gae(arr["reward"][sl], values[sl], terminated=True, bootstrap=0.0, gamma=cfg.gamma,
                               lam=cfg.gae_lambda)
        start += n
    batch = {"feats": arr["feats"], "mask": arr["mask"], "action": arr["action"].astype(np.int64),
             "old_logp": stored, "value_feats": arr["value_feats"], "adv": adv, "ret": ret}
    keep = {"episode": episode, "turn": arr["turn"], "action": arr["action"], "old_logp": stored, "old_value": values,
            "adv": adv, "ret": ret, "reward": arr["reward"], "mask": np.packbits(arr["mask"], axis=1), "valid": arr["valid"]}
    if keep_features:
        keep |= {"feats": arr["feats"], "value_feats": arr["value_feats"]}
    return batch, keep, {"logp_recompute_max_abs": gap, "old_value_mean": float(values.mean())}


def run_config(args, profile: dict, pool: Pool, ledger, counts, cfg, rollout: int, budget: int, engine: dict, lib: Path,
               device_info: dict) -> dict:
    return {"schema": "ctf-meow-rl-run-1", "task": "T0054/T0055", "arm": args.arm, "identity": I.identity(args.arm),
            "critic_hidden": profile["critic" if args.arm == "large_v" else "small_critic"]["hidden"],
            "profile": profile, "profile_sha256": I.sha256_file(Path(args.profile)), "ppo": cfg.record(),
            "sources": I.sources(), "pool": pool.identity(), "pool_sha256": I.sha256_file(Path(args.pool)),
            "seed_base": args.seed_base, "seed_ledger": ledger, "counts": counts, "limits": list(args.limits),
            "rollout_decisions": rollout, "checkpoint_interval": profile["pilot"]["checkpoint_interval"],
            "optimizer_seed": profile["pilot"]["optimizer_seed"], "budget": budget, "engine": engine,
            "bridge_library": {"name": lib.name, "sha256": I.sha256_file(lib), "flags": B.build_flags(),
                               "compiler": runner.find_compiler()},
            "device": device_info, "dev": args.dev, "workers": args.workers, "mode": args.mode,
            "python": sys.version.split()[0], "platform": sys.platform}


RESUME_FREE = {"budget", "device", "workers", "mode", "python"}      # may differ between a smoke and its continuation


def train(argv=None, collector=None, engine: dict | None = None) -> dict:
    args = parse_args(argv)
    import torch  # noqa: PLC0415  (rollout workers never import it)

    from .nets import Actor, Critic, configure, pick_device  # noqa: PLC0415
    from .ppo import Learner, PPOConfig  # noqa: PLC0415
    from .store import RunStore  # noqa: PLC0415

    profile = I.load_profile(Path(args.profile))
    pilot = profile["pilot"]
    ledger = check_ledger(args.seed_base, args.seed_ledger, args.dev)
    counts = args.counts or pilot["team_counts"]
    budget = args.decisions or (pilot["smoke_decisions_included_in_total"] if args.mode == "smoke" else pilot["decisions_per_arm"])
    cfg = PPOConfig.from_profile(profile["ppo"])
    if args.minibatch:
        cfg.minibatch = args.minibatch
    rollout = args.rollout_decisions or profile["ppo"]["rollout_decisions"]
    pool = Pool.load(Path(args.pool))
    lib = B.build()
    dev = pick_device(args.device)
    device_info = configure(dev, args.tf32)
    store = RunStore(Path(args.out))
    own_collector = collector is None
    if collector is None:
        root = paths.ai_test_root(args.ai_test)
        engine = {"sha256": paths.sha256(paths.engine_path(root)), "commit": paths.git_commit(root)}
        commands, info = pool.commands(root, I.REPO / "build" / "meow_rl" / "bots")
        collector = ProcessCollector(args.workers, commands, info, args.limits, root, lib)
    config = run_config(args, profile, pool, ledger, counts, cfg, rollout, budget, engine or {}, lib, device_info)

    torch.manual_seed(pilot["optimizer_seed"])
    critic_hidden = config["critic_hidden"]
    actor = Actor().load_numpy(B.reference_weights(B.load(lib), "policy"))     # warm start: the reference actor
    critic = Critic(critic_hidden)
    learner = Learner(actor, critic, cfg, dev)
    rng = np.random.default_rng(pilot["optimizer_seed"])
    st = {"decisions": 0, "update": 0, "next_index": 0, "episodes": 0, "checkpoints": [], "families": Counter(),
          "counts": Counter(), "faults": 0, "history": []}
    old = store.config()
    if args.resume:
        if old is None:
            raise SystemExit(f"--resume: no run in {args.out}")
        diff = sorted(k for k in set(old) | set(config) if k not in RESUME_FREE and old.get(k) != config.get(k))
        if diff:
            raise SystemExit(f"--resume: the run in {args.out} differs in {diff}")
        saved = store.load_state()
        if saved is not None:
            learner.load_state_dict(saved["learner"])
            rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"])
            st = saved["counters"]
            st["families"], st["counts"] = Counter(st["families"]), Counter(st["counts"])
        rec = store.recover(st["next_index"], st["update"])
        st["next_index"] = rec["next_index"]
        store.log("resumed", mode=args.mode, budget=budget, decisions=st["decisions"], recover=rec)
    elif old is not None:
        raise SystemExit(f"{args.out} already holds a run: --resume or a new --out")
    store.write_config(config)
    store.log("config", mode=args.mode, budget=budget)

    def next_spec():
        spec = pool.episode(args.seed_base, st["next_index"], counts)
        st["next_index"] += 1
        return spec

    checks = {"updates": [], "nonzero_actor_grad": False, "nonzero_critic_grad": False}
    interval = config["checkpoint_interval"]
    try:
        while st["decisions"] < budget:
            target = min(rollout, budget - st["decisions"])
            t0 = time.time()
            results = collector.run(next_spec, learner.actor.numpy(), st["update"], target,
                                    lambda idx: store.log("started", indices=idx, update=st["update"]))
            bad = [r for r in results if r["status"] != "ok"]
            if bad:
                store.log("discarded", indices=[r["index"] for r in results], reason=f"{bad[0]['status']} in batch",
                          errors=[r.get("error") for r in bad])
                raise SystemExit(f"{bad[0]['status']}: batch discarded (fix the cause before continuing): {bad[0].get('error')}")
            t1 = time.time()
            batch, keep, diag = make_batch(results, learner, cfg, args.keep_features)
            stats = learner.update(batch, rng)
            t2 = time.time()
            n = len(batch["action"])
            st["decisions"] += n
            st["update"] += 1
            st["episodes"] += len(results)
            records = [r["record"] for r in results]
            for r in records:
                st["counts"][str(r["teams"])] += 1
                st["faults"] += len(r["faults"])
                for fam in r["families"].values():
                    st["families"][fam] += 1
            summary = {"update": st["update"], "decisions": st["decisions"], "batch": n, "episodes": len(results),
                       "utility_mean": float(np.mean([r["utility"] for r in records])),
                       "collect_s": round(t1 - t0, 2), "learn_s": round(t2 - t1, 2), **diag, **stats}
            st["history"].append(summary)
            checks["updates"].append(summary)
            checks["nonzero_actor_grad"] |= bool(stats["max_actor_grad_norm"])
            checks["nonzero_critic_grad"] |= bool(stats["max_critic_grad_norm"])
            store.write_batch(st["update"], keep, {"summary": summary, "episodes": records})
            name = None
            mark = st["decisions"] // interval * interval
            if mark and mark not in st["checkpoints"] and st["decisions"] < budget:
                st["checkpoints"].append(mark)
                name = f"decisions_{mark:010d}.pt"
            if st["decisions"] >= budget:
                name = "final.pt" if args.mode == "full" else "smoke.pt"
            state = {"learner": learner.state_dict(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                     "counters": st | {"families": dict(st["families"]), "counts": dict(st["counts"])},
                     "config_digest": I.digest({k: v for k, v in config.items() if k not in RESUME_FREE}),
                     "identity": config["identity"], "actor_numpy": learner.actor.numpy(),
                     "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
            store.save_state(state, name)
            store.log("committed", update=st["update"], decisions=st["decisions"],
                      indices=[r["index"] for r in results], checkpoint=name)
            print(f"[meow_rl] update {st['update']} decisions {st['decisions']}/{budget} utility "
                  f"{summary['utility_mean']:.3f} kl {stats['approx_kl']} v_loss {stats['v_loss']:.4f} "
                  f"collect {summary['collect_s']}s learn {summary['learn_s']}s", flush=True)
    finally:
        if own_collector:
            collector.close()
    out = {"decisions": st["decisions"], "updates": st["update"], "episodes": st["episodes"],
           "families": dict(st["families"]), "counts": dict(st["counts"]), "faults": st["faults"]}
    if args.mode == "smoke":
        out |= {"checks": smoke_checks(store, learner, checks), "dev": args.dev}
        from .store import atomic_json  # noqa: PLC0415
        atomic_json(store.out / "smoke_report.json", out)
    return out


def smoke_checks(store, learner, checks: dict) -> dict:
    """Plumbing evidence only: finite non-zero gradients, log-probability parity, and an exact state round trip."""
    import torch  # noqa: PLC0415
    saved = store.load_state()
    same = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(learner.actor.state_dict().values(),
                                                           saved["learner"]["actor"].values()))
    same &= all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(learner.critic.state_dict().values(),
                                                            saved["learner"]["critic"].values()))
    ups = checks["updates"]
    return {"nonzero_actor_grad": checks["nonzero_actor_grad"], "nonzero_critic_grad": checks["nonzero_critic_grad"],
            "max_logp_recompute_gap": max(u["logp_recompute_max_abs"] for u in ups) if ups else None,
            "all_finite": all(np.isfinite([u["v_loss"]]).all() for u in ups),
            "state_round_trip": bool(same), "updates": len(ups)}


if __name__ == "__main__":
    mp.freeze_support()
    train()
