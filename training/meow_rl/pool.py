"""Opponent pool, family shares and the deterministic episode schedule (docs/MEOWMIND_RL_PLAN.md section 5).

Pool file (frozen per run; T0055 commits profiles/meow_rl_training_pool.json):

    {"schema": "ctf-meow-rl-pool-1", "family_cap": 0.15,
     "family_shares": {"astra": 0.1, ...},            # optional; default equal shares over the families present
     "bots": [{"name": "Astra", "family": "astra", "path": "<arena-relative or @repo/ path>", "sha256": "..."}]}

Each opponent seat draws a family by its share, then a member uniformly. The cap bounds every family's share of all
opponent seats in expectation (a duel has one opponent seat, so it cannot hold per game); a pool whose shares cannot
meet the cap is refused. Realized shares are counted per run (training.meow_rl.train). No learner self-play.

Episode i of a run with seed base S draws, from sha256 derivations of (S, purpose, i) only: the team count (uniform
over the configured counts), the learner seat (uniform), the map and flag seeds, the opponents and the actor's
sampling seed. A schedule never depends on worker timing or completion order.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fusion import paths, runner
from training.collect import derive

SCHEMA = "ctf-meow-rl-pool-1"


@dataclass
class EpisodeSpec:
    index: int
    teams: int
    seat: int
    map_seed: int
    flag_seed: int
    act_seed: int
    opponents: list[str]           # seat order without the learner

    def seats(self, learner: str = "@learner") -> list[str]:
        s = list(self.opponents)
        s.insert(self.seat, learner)
        return s

    def record(self) -> dict:
        return asdict(self)


class Pool:
    def __init__(self, spec: dict):
        if spec.get("schema") != SCHEMA:
            raise ValueError(f"pool schema must be {SCHEMA}")
        self.spec = spec
        self.cap = float(spec.get("family_cap", 0.15))
        self.bots = {b["name"]: b for b in spec["bots"]}
        if len(self.bots) != len(spec["bots"]):
            raise ValueError("duplicate bot names in the pool")
        self.families: dict[str, list[str]] = {}
        for b in spec["bots"]:
            self.families.setdefault(b["family"], []).append(b["name"])
        for members in self.families.values():
            members.sort()
        shares = spec.get("family_shares") or {f: 1.0 for f in self.families}
        if set(shares) != set(self.families):
            raise ValueError("family_shares must name exactly the pool's families")
        total = sum(shares.values())
        self.shares = {f: shares[f] / total for f in sorted(self.families)}
        worst = max(self.shares.values())
        if worst > self.cap + 1e-12:
            raise ValueError(f"family share {worst:.3f} exceeds the cap {self.cap}: add families or set family_shares "
                             f"({len(self.families)} families)")

    @classmethod
    def load(cls, path: Path) -> "Pool":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def family_of(self, name: str) -> str:
        return self.bots[name]["family"]

    def _draw(self, u: float) -> str:
        acc = 0.0
        for f, s in self.shares.items():
            acc += s
            if u < acc:
                return f
        return next(reversed(self.shares))

    def episode(self, seed_base: int, index: int, counts: list[int]) -> EpisodeSpec:
        teams = counts[derive(seed_base, "teams", index) % len(counts)]
        seat = derive(seed_base, "seat", index) % teams
        opponents = []
        for slot in range(teams - 1):
            fam = self._draw(derive(seed_base, "family", index, slot) / float(1 << 31))
            members = self.families[fam]
            opponents.append(members[derive(seed_base, "member", index, slot) % len(members)])
        return EpisodeSpec(index, teams, seat, derive(seed_base, "map", index), derive(seed_base, "flag", index, bits=63),
                           derive(seed_base, "act", index, bits=63), opponents)

    def commands(self, root: Path, build_dir: Path) -> tuple[dict, dict]:
        """name -> command line, name -> {family, sha256}; compiles C++ opponents once. A pinned sha256 must match."""
        cmds, info = {}, {}
        for name, b in sorted(self.bots.items()):
            path = paths.bot_path(b, Path(root))
            if not path.is_file():
                raise FileNotFoundError(f"pool bot {name}: {path}")
            sha = paths.sha256(path)
            if b.get("sha256") and b["sha256"] != sha:
                raise ValueError(f"pool bot {name}: sha256 {sha}, the pool pins {b['sha256']}")
            cmds[name] = runner.command_for(path, build_dir)
            info[name] = {"family": b["family"], "sha256": sha}
        return cmds, info

    def identity(self) -> dict:
        return {"schema": SCHEMA, "cap": self.cap, "shares": self.shares,
                "bots": [{k: self.bots[n].get(k) for k in ("name", "family", "path", "sha256")} for n in sorted(self.bots)]}
