"""PPO on the agent_v2 carrier: training.meow_rl's trainer with the agent_v2 bridge and run identity (task T0065).

    python -m training.agent_v2.train --init-actor <actor arrays .npz> [training.meow_rl.train arguments ...]

Nothing of the trainer is copied: training.meow_rl.train.train (collection, GAE, PPO, storage, checkpoints, resume)
runs as it is, with exactly three substitutions, all visible in the run config:
  * the native library: the agent_v2 bridge (training.agent_v2.bridge.build) instead of the Meowmind reference
    bridge, so every rollout decision is resolved by the carrier's P + resolver, the code the deployed bot runs;
  * the warm-start actor: arrays given with --init-actor (an exported actor, e.g. a checkpoint's actor_numpy saved as
    .npz: w0..b2), instead of the reference's weights read through the old bridge;
  * the run config: identity = training.agent_v2.identity.identity(arm) (carrier, predictor, resolver), the
    agent_v2 sources, task T0065. --resume of a run made on another carrier is therefore refused by the trainer's own
    config comparison; a changed P or resolver is a new run.
Starting a run is a host task with its own seeds, leases and budget; this module only provides the entry point.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np

from training.meow_rl import policy as P
from training.meow_rl import train as T

from . import bridge as AB
from . import identity as AI


def init_actor(path: Path) -> dict:
    z = dict(np.load(path))
    return P.check_actor({k: z[k] for k in ("w0", "b0", "w1", "b1", "w2", "b2")})


def agent_v2_config(base_config, init_path: Path):
    """Wraps training.meow_rl.train.run_config: the same record with the agent_v2 identity."""
    def run_config(args, *rest, **kw):
        cfg = base_config(args, *rest, **kw)
        cfg["task"] = "T0065"
        cfg["identity"] = AI.identity(args.arm)
        cfg["agent_v2_sources"] = AI.sources()
        cfg["init_actor"] = {"path": str(init_path), "sha256": AI.sha256_file(init_path)}
        if isinstance(cfg.get("bridge_library"), dict):
            cfg["bridge_library"]["flags"] = AB.build_flags()            # the agent_v2 build, not the reference bridge's
        return cfg
    return run_config


@contextlib.contextmanager
def substituted(init_path: Path, lib: Path | None = None):
    """The three substitutions, undone afterwards (tests call train() more than once in one process)."""
    saved = (T.B.build, T.B.reference_weights, T.run_config)
    actor = init_actor(init_path)
    built = lib or AB.build()
    try:
        T.B.build = lambda *a, **k: built
        T.B.reference_weights = lambda *a, **k: {k2: v.copy() for k2, v in actor.items()}
        T.run_config = agent_v2_config(saved[2], init_path)
        yield built
    finally:
        T.B.build, T.B.reference_weights, T.run_config = saved


def train(argv=None, collector=None, engine=None, lib: Path | None = None) -> dict:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--init-actor", required=True)
    known, rest = ap.parse_known_args(argv)
    with substituted(Path(known.init_actor), lib):
        return T.train(rest, collector=collector, engine=engine)


if __name__ == "__main__":
    train(sys.argv[1:])
