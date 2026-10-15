"""The live episode adapter: the official engine, opponent processes and the learner seat in-process.

Per turn (all seats move simultaneously):
  1. every seat's input is encoded once (game.encode);
  2. the inputs are sent to the opponent processes, which keep running for the whole episode (their memory persists);
  3. the learner observes only its own input through the bridge, samples a joint movement from the frozen actor
     snapshot under the bridge's mask, and the reference resolver completes the order; the reply is checked to carry
     exactly the sampled movement (no repair);
  4. only then are the opponents' replies collected (fusion.runner.Bot limits and fault rules), so they can never
     reach the learner's observation;
  5. every reply is decoded and the engine steps once.
The terminal reward 2 U - 1 goes to the last decision. Opponent faults follow the engine's rules (an empty order)
and are recorded for every seat; an opponent that cannot even start raises EnvironmentFault (the batch is discarded,
the episode never becomes data). A learner-side failure (bridge error, a malformed or altered reply, a turn mismatch)
raises LearnerFault: the batch holding that episode is invalid and the run stops for a fix.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fusion import runner
from fusion.paths import load_engine
from . import bridge as B
from . import policy as P
from .gae import terminal_reward, utility
from .identity import JOINT, MOVES, PF, UNITS, VF
from .pool import EpisodeSpec


class LearnerFault(RuntimeError):
    pass


class EnvironmentFault(RuntimeError):
    """The episode could not be played as specified (an opponent process failed to start): never training data."""


@dataclass
class Episode:
    spec: EpisodeSpec
    actor_version: int
    feats: np.ndarray            # [T, 3, 5, 112] float32
    mask: np.ndarray             # [T, 125] bool
    value_feats: np.ndarray      # [T, 416] float32
    action: np.ndarray           # [T] int16, the sampled joint
    resolved: np.ndarray         # [T, 3] int16, the resolver's actions (target id, -1 pickup/wait, -3 drop)
    logp: np.ndarray             # [T] float32, its log probability under the snapshot
    turn: np.ndarray             # [T] int16
    valid: np.ndarray            # [T] int16, feasible joints
    reward: np.ndarray           # [T] float32, 0 except the terminal decision
    record: dict = field(default_factory=dict)

    @property
    def decisions(self) -> int:
        return len(self.action)


def run_episode(spec: EpisodeSpec, actor: dict, actor_version: int, commands: dict, info: dict,
                limits: tuple[float, float], root: Path | None = None, lib_path: Path | None = None,
                engine_factory=None, bot_factory=None) -> Episode:
    """Play one full episode. engine_factory(teams, map_seed, flag_seed) and bot_factory(command, name, limits)
    default to the official engine under `root` and fusion.runner.Bot (tests pass fakes)."""
    actor = P.check_actor(actor)
    if engine_factory is None:
        engine = load_engine(Path(root))
        engine_factory = lambda n, m, f: engine.Game(n, m, flag_seed=f)  # noqa: E731
    bot_factory = bot_factory or runner.Bot
    rng = np.random.default_rng(spec.act_seed)
    game = engine_factory(spec.teams, spec.map_seed, spec.flag_seed)
    seats = spec.seats()
    bots = {i: bot_factory(commands[name], name, tuple(limits)) for i, name in enumerate(seats) if i != spec.seat}
    dead = {i: b.dead for i, b in bots.items() if getattr(b, "dead", None)}
    if dead:
        for b in bots.values():
            b.close()
        raise EnvironmentFault(f"opponents failed to start: {dead}")
    rows = {k: [] for k in ("feats", "mask", "value_feats", "action", "resolved", "logp", "turn", "valid")}
    faults, learner_ms = [], []
    session = B.Session(B.load(lib_path))
    try:
        while not game.done:
            turn = game.turn + 1
            inputs = [game.encode(i) for i in range(spec.teams)]
            starts = {i: b.send(inputs[i]) for i, b in bots.items()}
            t0 = time.perf_counter()
            try:
                obs = session.observe(inputs[spec.seat])
            except B.BridgeError as exc:
                raise LearnerFault(f"turn {turn}: observe: {exc}") from exc
            if obs.turn != turn or obs.me != spec.seat or obs.teams != spec.teams:
                raise LearnerFault(f"turn {turn}: bridge saw turn {obs.turn}, seat {obs.me}, teams {obs.teams}")
            logp = P.joint_log_probs(actor, obs.feats, obs.joint_mask)
            j = P.sample(logp, rng.random())
            try:
                line, actions = session.resolve(j)
            except B.BridgeError as exc:
                raise LearnerFault(f"turn {turn}: resolve {j}: {exc}") from exc
            t_reply, moves, _ = B.parse_reply(line)
            if t_reply != turn or B.joint_index(moves) != j:
                raise LearnerFault(f"turn {turn}: reply {line!r} does not carry the sampled joint {B.joint_moves(j)}")
            learner_ms.append(1000 * (time.perf_counter() - t0))
            lines = {i: b.receive(starts[i], turn) for i, b in bots.items()}
            orders = {}
            try:
                orders[spec.seat] = game.decode(spec.seat, line)
            except ValueError as exc:
                raise LearnerFault(f"turn {turn}: the engine refused the learner reply {line!r}: {exc}") from exc
            for i, b in bots.items():
                if lines[i] is None:
                    orders[i] = {}
                    continue
                try:
                    orders[i] = game.decode(i, lines[i])
                    b.ok()
                except ValueError as exc:
                    orders[i] = {}
                    b.strike(turn, f"malformed output: {exc}")
            game.step(orders)
            faults += [[turn, i, reason] for i, b in bots.items() for t, reason in b.faults if t == turn]
            rows["feats"].append(obs.feats)
            rows["mask"].append(obs.joint_mask)
            rows["value_feats"].append(obs.value_feats)
            rows["action"].append(j)
            rows["resolved"].append(actions)
            rows["logp"].append(logp[j])
            rows["turn"].append(turn)
            rows["valid"].append(obs.valid)
    finally:
        session.close()
        reports = {i: b.report() for i, b in bots.items()}
        for b in bots.values():
            b.close()
    score = list(game.result()["score"])
    T = len(rows["action"])
    if T == 0:
        raise LearnerFault("an episode without decisions")
    reward = np.zeros(T, np.float32)
    reward[-1] = terminal_reward(score, spec.seat)
    ep = Episode(spec, actor_version,
                 np.asarray(rows["feats"], np.float32).reshape(T, UNITS, MOVES, PF),
                 np.asarray(rows["mask"], bool).reshape(T, JOINT),
                 np.asarray(rows["value_feats"], np.float32).reshape(T, VF),
                 np.asarray(rows["action"], np.int16), np.asarray(rows["resolved"], np.int16).reshape(T, 3),
                 np.asarray(rows["logp"], np.float32),
                 np.asarray(rows["turn"], np.int16), np.asarray(rows["valid"], np.int16), reward)
    ep.record = {"schema": "ctf-meow-rl-episode-1", **spec.record(), "actor_version": actor_version,
                 "seats": seats, "families": [info[s]["family"] for s in spec.opponents],   # one per opponent seat
                 "score": score, "utility": utility(score, spec.seat), "reward": float(reward[-1]),
                 "decisions": T, "terminated": True, "faults": faults,
                 "opponents": {str(i): {k: r[k] for k in ("name", "dead", "faults", "late", "first_ms", "p99_ms", "max_ms")}
                               for i, r in reports.items()},
                 "learner_ms": {"mean": float(np.mean(learner_ms)), "max": float(np.max(learner_ms))},
                 "forced_decisions": int((ep.valid == 1).sum()), "status": "ok"}
    return ep


def collect(job) -> dict:
    """multiprocessing entry point (spawn-safe): one episode, returned as plain arrays and its record, or the failure."""
    spec, actor, version, commands, info, limits, root, lib_path = job
    try:
        ep = run_episode(spec, actor, version, commands, info, limits, root=root, lib_path=lib_path)
    except LearnerFault as exc:
        return {"status": "learner_fault", "index": spec.index, "error": str(exc)}
    except EnvironmentFault as exc:
        return {"status": "environment_fault", "index": spec.index, "error": str(exc)}
    return {"status": "ok", "index": spec.index, "record": ep.record,
            "arrays": {k: getattr(ep, k) for k in ("feats", "mask", "value_feats", "action", "resolved", "logp", "turn",
                                                   "valid", "reward")}}
