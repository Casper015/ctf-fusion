"""Hierarchical opponent sampling for the open training pool (roster/registry.py builds the pool).

Per team count N the bot distribution is a mixture of three components
(handoff BOT_GROUP_AND_SAMPLING_REVIEW.md):

    rho_N = l_field * rho_field,N + l_style * rho_style,N + l_hard * rho_hard,N

Each component is group-balanced over the members it covers at N: every
behavior group at N (complete-linkage, roster/distance.py) gets equal mass,
split equally among its versions, so a lineage cannot dominate by its number
of files. A lineage cap then limits any one source lineage's share of the
component (water-filling; relaxed to 1/lineages when infeasible). The default
0.5 is a provisional regularizer: it binds only when one lineage would hold
most of a component, since distinct groups inside one lineage are legitimate
mass. A cap at or below 1/lineages makes every lineage equal, overriding the
group balance, so set it with the lineage count in mind.

  field  members reviewed as the intended strong competition field;
  style  every admitted policy (behavior coverage);
  hard   members reviewed as hard cases at the counts they were verified at.

A component with no members at N is dropped and the others renormalized; the
effective weights are recorded. Augmentation members (random/noise policies,
never P labels) are outside the mixture: with probability `augmentation` one
seat of a game is theirs.

Seats are filled one by one: draw a component by its weight, then a bot from
that component among the bots that still have copies left (max_copies, default
1). Regimes: "mixed" (the representative game), "focus" (with probability
`focus`, one style-drawn bot takes `focus_fraction` of the seats: a controlled
same-policy concentration test) and the augmentation seat. Inclusion
probabilities come from the same sampler (inclusion()), never from rho alone:
without replacement they differ, and a pool no larger than N gives every game
the same composition, which the report flags.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

COMPONENTS = ("field", "style", "hard")
DEFAULTS = {"lambdas": {"field": 0.6, "style": 0.3, "hard": 0.1}, "augmentation": 0.0, "focus": 0.0,
            "focus_fraction": 0.5, "lineage_cap": 0.5}


def settings(sampling: dict | None) -> dict:
    s = {**DEFAULTS, **(sampling or {})}
    s["lambdas"] = {c: float(s["lambdas"].get(c, 0.0)) for c in COMPONENTS}
    if any(v < 0 for v in s["lambdas"].values()) or sum(s["lambdas"].values()) <= 0:
        raise ValueError(f"component weights must be non-negative with a positive sum: {s['lambdas']}")
    for k in ("augmentation", "focus", "focus_fraction"):
        if not 0.0 <= float(s[k]) <= 1.0:
            raise ValueError(f"{k} must be in [0, 1]")
    if not 0.0 < float(s["lineage_cap"]) <= 1.0:
        raise ValueError("lineage_cap must be in (0, 1]")
    return s


def cap_lineages(rho: dict[str, float], lineage_of: dict[str, str], cap: float) -> dict[str, float]:
    """Scale lineages above `cap` down to it and the rest up in proportion, until none exceeds it."""
    if not rho:
        return {}
    lineages = {lineage_of[n] for n in rho}
    cap = max(cap, 1.0 / len(lineages))
    out, fixed = dict(rho), set()
    for _ in range(len(lineages)):
        mass = defaultdict(float)
        for n, p in out.items():
            mass[lineage_of[n]] += p
        over = [l for l, m in mass.items() if l not in fixed and m > cap + 1e-12]
        if not over:
            break
        for l in over:
            for n in out:
                if lineage_of[n] == l:
                    out[n] *= cap / mass[l]
            fixed.add(l)
        free = sum(p for n, p in out.items() if lineage_of[n] not in fixed)
        left = 1.0 - cap * len(fixed)
        if free > 0:
            for n in out:
                if lineage_of[n] not in fixed:
                    out[n] *= left / free
    total = sum(out.values())
    return {n: p / total for n, p in out.items()}


def group_balanced(members: list[dict], teams: int, lineage_cap: float) -> dict[str, float]:
    """rho_style-like distribution over `members` at `teams`: 1 / (K_N |C_N(b)|), then the lineage cap."""
    groups = defaultdict(list)
    for m in members:
        groups[m["groups"][str(teams)]].append(m["name"])
    rho = {n: 1.0 / (len(groups) * len(g)) for g in groups.values() for n in g}
    return cap_lineages(rho, {m["name"]: m["lineage"] for m in members}, lineage_cap)


def mixture(members: list[dict], teams: int, sampling: dict | None = None) -> dict:
    """The sampling distribution at `teams`: components, effective weights, rho and the augmentation members."""
    s = settings(sampling)
    here = [m for m in members if teams in m["counts"]]
    policies = [m for m in here if "augmentation" not in m["roles"]]
    comps = {
        "field": group_balanced([m for m in policies if "field" in m["roles"]], teams, s["lineage_cap"]),
        "style": group_balanced(policies, teams, s["lineage_cap"]),
        "hard": group_balanced([m for m in policies if "hard" in m["roles"] and teams in m.get("hard_counts", m["counts"])],
                               teams, s["lineage_cap"]),
    }
    lam = {c: s["lambdas"][c] for c in COMPONENTS if comps[c] and s["lambdas"][c] > 0}
    total = sum(lam.values())
    lam = {c: v / total for c, v in lam.items()} if total > 0 else {}
    rho = defaultdict(float)
    for c, w in lam.items():
        for n, p in comps[c].items():
            rho[n] += w * p
    return {"teams": teams, "lambdas": lam, "components": {c: comps[c] for c in lam}, "rho": dict(rho),
            "augmentation": sorted(m["name"] for m in here if "augmentation" in m["roles"]),
            "augmentation_share": float(s["augmentation"]) if any("augmentation" in m["roles"] for m in here) else 0.0,
            "focus": float(s["focus"]), "focus_fraction": float(s["focus_fraction"]),
            "max_copies": {m["name"]: int(m.get("max_copies", 1)) for m in here}}


def _pick(rng: random.Random, dist: dict[str, float], left: Counter) -> str | None:
    names = [n for n in sorted(dist) if left[n] > 0 and dist[n] > 0]
    return rng.choices(names, [dist[n] for n in names])[0] if names else None


def draw(rng: random.Random, mix: dict) -> dict:
    """One game's seats at mix["teams"]: {"seats", "components", "regime", "forced_repeats"}."""
    teams = mix["teams"]
    left = Counter(mix["max_copies"])
    seats, comps, forced = [], [], 0
    regime = "mixed"
    if mix["augmentation"] and rng.random() < mix["augmentation_share"]:
        regime = "augmented"
        seats.append(rng.choice(mix["augmentation"]))
        comps.append("augmentation")
    if mix["focus"] > 0 and rng.random() < mix["focus"] and "style" in mix["components"]:
        regime = "focus" if regime == "mixed" else regime + "+focus"
        bot = _pick(rng, mix["components"]["style"], Counter({n: 1 for n in mix["rho"]}))
        k = min(teams - len(seats), max(2, round(mix["focus_fraction"] * teams)))
        seats += [bot] * k
        comps += ["focus"] * k
        left[bot] = 0
    while len(seats) < teams:
        open_comps = {c: w for c, w in mix["lambdas"].items() if any(left[n] > 0 for n in mix["components"][c])}
        if not open_comps:                     # every policy used up its copies: repeat from rho
            bot = _pick(rng, mix["rho"], Counter({n: 1 for n in mix["rho"]}))
            seats.append(bot)
            comps.append("repeat")
            forced += 1
            continue
        c = rng.choices(sorted(open_comps), [open_comps[k] for k in sorted(open_comps)])[0]
        bot = _pick(rng, mix["components"][c], left)
        left[bot] -= 1
        seats.append(bot)
        comps.append(c)
    order = list(range(teams))
    rng.shuffle(order)
    return {"seats": [seats[i] for i in order], "components": [comps[i] for i in order], "regime": regime,
            "forced_repeats": forced}


def inclusion(mix: dict, games: int = 20000, seed: int = 0) -> dict:
    """What the sampler actually produces at mix["teams"], by simulation: per bot the probability of being in a
    game and its expected seats; seat shares per component and regime; distinct compositions."""
    rng = random.Random(seed)
    present, seats_of, comp_seats, regimes, compositions, forced = Counter(), Counter(), Counter(), Counter(), Counter(), 0
    for _ in range(games):
        g = draw(rng, mix)
        for n in set(g["seats"]):
            present[n] += 1
        seats_of.update(g["seats"])
        comp_seats.update(g["components"])
        regimes[g["regime"]] += 1
        compositions[tuple(sorted(g["seats"]))] += 1
        forced += g["forced_repeats"]
    total = games * mix["teams"]
    probs = [c / games for c in compositions.values()]
    names = sorted(set(mix["rho"]) | set(mix["augmentation"]))
    return {"games": games,
            "in_game": {n: round(present[n] / games, 4) for n in names},
            "seats_per_game": {n: round(seats_of[n] / games, 4) for n in names},
            "component_seat_share": {c: round(k / total, 4) for c, k in sorted(comp_seats.items())},
            "regimes": {r: round(k / games, 4) for r, k in sorted(regimes.items())},
            "distinct_compositions": len(compositions),
            "composition_entropy_bits": round(-sum(p * math.log2(p) for p in probs), 3),
            "fixed_composition": len(compositions) == 1,
            "forced_repeat_seats_per_game": round(forced / games, 4)}
