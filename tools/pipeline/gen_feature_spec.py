"""Regenerate training/specs/FEATURES.md from the C++ encoder's schema.

    python tools/pipeline/gen_feature_spec.py           # write
    python tools/pipeline/gen_feature_spec.py --check   # exit 1 if the file is stale (tests run this)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fusion import families  # noqa: E402
from fusion.native import schema  # noqa: E402

TARGET = REPO / "training" / "specs" / "FEATURES.md"

DESC = {
    "global": {"teams": "team count of the match", "turn": "turn about to settle (1-400)", "remaining": "turns left including this one",
               "own_score": "captures so far", "best_enemy_score": "highest other score", "own_rank": "1 + teams with a strictly higher score",
               "margin": "own score minus best other score", "flags_home": "flags at home",
               "flags_dropped": "dropped flags that do not return in this settlement", "flags_carried_own": "flags our units carry",
               "flags_carried_enemy": "flags other teams carry", "flags_unavailable": "cooldown flags and dropped flags returning now",
               "map_size": "board width", "own_active": "own units on the board this turn (incl. respawning)",
               "enemy_active": "other units on the board this turn"},
    "unit": {"active": "on the board this turn (alive, or respawning now)", "immune": "respawns in this settlement: cannot shoot or be shot",
             "dead_wait": "turns until respawn when off the board", "hp": "hit points (100 when respawning)", "carrying": "holds a flag",
             "home_dist": "BFS distance to own base", "flag_dist": "BFS distance to nearest available flag",
             "nearest_enemy": "Manhattan distance to nearest hittable enemy", "enemies_r3": "enemies within Manhattan 3",
             "allies_r3": "teammates within Manhattan 3", "threatened_by": "enemies able to shoot this cell now",
             "last_dx": "x step since the previous block", "last_dy": "y step since the previous block",
             "hits_taken": "attack events on this unit last settlement"},
    "move": {"legal": "destination is open", "d_home": "change of own-base BFS distance", "d_flag": "change of nearest-flag BFS distance",
             "shooters": "enemies able to shoot the destination from where they stand", "enemies_r2": "enemies within Manhattan 2 of the destination",
             "flag_cell": "available flag on the destination", "own_base": "destination is an own base cell",
             "occupied": "another unit stands on the destination now", "continues": "repeats the last step",
             "carrier_targets": "enemy carriers shootable from the destination", "targets": "enemies shootable from the destination"},
    "cand_f": {"dist": "Manhattan distance from the unit (candidates are within 4)", "hp": "target hit points", "carrying": "target holds a flag",
               "support": "teammates that can shoot the target now", "hit_S": "shootable from the destination of S", "hit_U": "... of U",
               "hit_D": "... of D", "hit_L": "... of L", "hit_R": "... of R",
               "attacked_by_team": "our attack events on the target last settlement",
               "target_home": "carrier's BFS distance to its base (99 if not carrying)"},
    "value_unit": {},
    "value_agg": {"enemy_carriers": "other teams' carriers", "enemy_carrier_home": "nearest enemy carrier's BFS distance to its base",
                  "enemies_near_base": "enemies within Manhattan 5 of our base center", "escort_threat": "enemies within 3 of our carriers (sum)",
                  "own_carrier_home": "our nearest carrier's BFS distance to base", "dropped_near": "available dropped flags within 5 of our units",
                  "enemies_r6": "enemies within 6 of any of our units", "carrier_shots": "our units able to shoot an enemy carrier now",
                  "rivals_tied": "other teams with exactly our score (v2)",
                  "rivals_1_2_ahead": "other teams 1 or 2 captures ahead of us (v2)",
                  "rivals_1_2_behind": "other teams 1 or 2 captures behind us (v2)",
                  "pairwise_now": "current pairwise outcome x 100: (lower + tied / 2) / (teams - 1), rounded half up (v2)"},
    "combat": {"kill1_stay": "enemies the unit can shoot from the destination that one hit kills (they stay)",
               "kill1_carrier_stay": "... of them carrying a flag",
               "volley_stay": "enemies shootable from the destination that this unit plus teammates able to reach a firing cell can kill",
               "volley_carrier_stay": "... of them carrying a flag",
               "focus_present": "an enemy carrier is shootable from the destination (they stay)",
               "focus_hits": "that carrier's hits to kill (fewest hits, then nearest delivery, then id; 0 = none)",
               "focus_home": "the same carrier's BFS distance to its base (0 = none)",
               "focus_support": "teammates able to reach a firing cell on the same carrier",
               "chase_present": "an enemy carrier within Manhattan 6 of the destination",
               "chase_dist": "Manhattan distance to it (nearest, then fewest hits, then nearest delivery)",
               "chase_hits": "the same carrier's hits to kill", "chase_home": "the same carrier's BFS distance to its base",
               "exposure_reach": "enemy shooters able to fire at the destination after their move (upper bound on incoming hits)",
               "lethal_stay": "enemy shooters able to fire at the destination now, times damage, reach the unit's hp",
               "lethal_reach": "exposure_reach times damage reaches the unit's hp"},
    "value_ext": {"u0_hits": "own unit 0: hits that kill it (0 off the board or respawning)",
                  "u0_exposure": "own unit 0: enemy shooters able to fire at its cell after their move",
                  "u0_lethal": "own unit 0: u0_exposure times damage reaches its hp",
                  "ec0_present": "enemy carrier slot 0 (nearest delivery, then fewest hits, then id)",
                  "ec0_hits": "its hits to kill", "ec0_home": "its BFS distance to its base (delivery distance)",
                  "ec0_shots_stay": "own units able to shoot it now", "ec0_shots_reach": "own units able to reach a firing cell on it",
                  "ec0_killable": "ec0_shots_reach >= ec0_hits (a coordinated volley kills it if it stays)",
                  "ec0_escort": "its teammates within Manhattan 3", "ec0_gap": "its team's score minus ours",
                  "own_carrier_escort": "teammates within 3 of our carrier nearest to delivery",
                  "dropped_return_min": "turns until the soonest available dropped flag returns (16 = none)",
                  "cooldown_ready_min": "turns until the soonest cooldown flag is home (9 = none)",
                  "flag_own_dist": "our nearest BFS distance to an available flag (99 = none)",
                  "flag_enemy_dist": "the other teams' nearest BFS distance to an available flag",
                  "flag_contest": "flag_enemy_dist minus flag_own_dist",
                  "dropped_own_closer": "available dropped flags where one of our units is strictly closest (Manhattan)",
                  "dropped_enemy_closer": "available dropped flags where another team's unit is strictly closest",
                  "r0_present": "rival slot 0 (highest score, then team id)", "r0_gap": "its score minus ours",
                  "r0_carriers": "its units carrying a flag", "r0_active": "its units on the board",
                  "r0_carrier_home": "its nearest carrier's BFS distance to its base (99 = none)",
                  "own_respawning": "own units off the board", "enemy_respawning": "other teams' units off the board",
                  "rivals_ahead_carrying": "rivals strictly ahead of us that hold a flag",
                  "rivals_close_carrying": "rivals tied or one behind that hold a flag (a capture ties or passes us)",
                  "enemy_carriers_killable": "enemy carriers our units can kill this turn if they stay",
                  "own_carriers_lethal": "own carriers whose exposure can kill them this turn"},
}
for _n in ("u1", "u2"):
    for _f in ("hits", "exposure", "lethal"):
        DESC["value_ext"][f"{_n}_{_f}"] = f"own unit {_n[1]}: as u0"
for _f in ("present", "hits", "home", "shots_stay", "shots_reach", "killable", "escort", "gap"):
    DESC["value_ext"][f"ec1_{_f}"] = "enemy carrier slot 1: as slot 0"
for _r in ("r1", "r2"):
    for _f in ("present", "gap", "carriers", "active", "carrier_home"):
        DESC["value_ext"][f"{_r}_{_f}"] = f"rival slot {_r[1]}: as slot 0"
TITLE = {"global": "Global (per team)", "unit": "Policy: unit (per own unit, 3 rows by id)",
         "move": "Policy: move (per own unit and move S U D L R)",
         "cand_f": "Policy: attack candidate (per own unit, up to 8 hittable enemies within 4, nearest first)",
         "value_unit": "Value: own unit (same meaning as the policy unit features)", "value_agg": "Value: aggregates",
         "combat": "Policy p3: combat context (per own unit and move, `inference/combat.hpp`)",
         "value_ext": "Value v3: extension (`inference/combat.hpp`)"}


def render() -> str:
    s = schema()
    out = ["# Feature spec v%d" % s["spec_version"], "",
           "Generated by `tools/pipeline/gen_feature_spec.py` from `extract_features --schema`; do not edit by hand.", "",
           "`inference/features.hpp` (families p2, v2) and `inference/combat.hpp` (the p3 and v3 additions) are the only",
           "implementation. The training extractor (`tools/cpp/extract_features.cpp`) and the bot call the same functions.",
           "Every feature is a clipped integer in [lo, hi]. Training divides by `scale`; the exporter folds that division",
           "into the first layer, so inference is integer and exact.", "",
           "Facts shared by all features of a turn (`makeView` in `inference/state.hpp`):", "",
           "- **origin**: the cell a unit moves from this turn. A dead unit whose respawn time has come takes the first free",
           "  cell of its team's spawn order (engine phase 1, ascending unit id) and is `immune` this turn. Tests check every",
           "  prediction against the engine's respawn events.",
           "- **available flag**: home, or dropped with `turn < return_at` (a dropped flag at its return time moves to a",
           "  random spot in phase 1, so it cannot be picked up where it lies).",
           "- **shooter**: on the board, not respawning, not carrying. **hittable**: on the board, not respawning.",
           "- Distances: BFS over open cells for base and flag distances, Manhattan for ranges; `canAttack` is engine",
           "  `Game.can_attack`.",
           "- Combat context (p3, v3): damage and hit points from INIT; hits to kill = ceil(hp / damage); `*_stay` features",
           "  assume the named enemies stay where they stand, `*_reach` features bound over every legal move they could",
           "  make (movement blocking is not modelled). Nothing uses future actions or outcomes.", "",
           "## Feature families", "",
           "A model reads one policy or one value family (`fusion/families.py`). The family digest covers exactly the",
           "groups, block shapes and head inputs it reads, so changing one family never invalidates another, and the record",
           "layout (below) is versioned apart from the families.", "",
           "| Family | Net | Input | Width | Digest |", "|---|---|---|---:|---|"]
    G, U, M = len(s["global"]), len(s["unit"]), len(s["move"])
    C = len(s["combat"])
    V = G + 3 * len(s["value_unit"]) + len(s["value_agg"])
    dg = families.available(s)
    out += [f"| p2 | P mover (per unit, per legal move) | global + unit k + move (k, m) | {G + U + M} | `{dg['p2']}` |",
            f"| p2 | P squad (per team) | global + 3 units + 15 moves | {G + 3 * U + 15 * M} | |",
            f"| p2 | P act (per unit, per move) | same as mover; output wait/pickup/drop/attack masked by `act_legal` | {G + U + M} | |",
            "| p2 | P target (per unit, candidate, move) | dist, hp, carrying, support, hit bit of the move, attacked_by_team, target_home | 7 | |",
            f"| p3 | P mover, act | p2 input + combat (k, m) | {G + U + M + C} | `{dg['p3']}` |",
            f"| p3 | P squad | p2 input + 15 combat blocks | {G + 3 * U + 15 * M + 15 * C} | |",
            "| p3 | P target | unchanged from p2 | 7 | |",
            f"| v2 | V (per team) | global + 3 value units + value aggregates | {V} | `{dg['v2']}` |",
            f"| v3 | V (per team) | v2 input + value extension | {V + len(s['value_ext'])} | `{dg['v3']}` |", "",
            "V's first layer is the accumulator: `inference/value.hpp` (`value_v3.hpp` for v3) keeps its int32",
            "pre-activations and updates them per changed feature, bit-identical to the full pass.", ""]
    for g in ("global", "unit", "move", "cand_f", "value_unit", "value_agg", "combat", "value_ext"):
        out += [f"## {TITLE[g]}", "", "| # | name | lo | hi | scale | meaning |", "|---:|---|---:|---:|---:|---|"]
        for i, (n, lo, hi, sc) in enumerate(s[g]):
            out.append(f"| {i} | `{n}` | {lo} | {hi} | {sc} | {DESC[g].get(n, '')} |")
        out.append("")
    out += [f"## Extractor record, layout {s['layout_version']} (int16, one per turn and team)", "",
            "| block | offset | shape |", "|---|---:|---|"]
    for name, off, shape in s["layout"]:
        out.append(f"| `{name}` | {off} | {' x '.join(map(str, shape))} |")
    out += ["", f"Record width {s['record_width']}. `legal`, `act_legal` are 0/1 masks; `origin` is a cell index",
            "(y*size+x, -1 off the board); `cand` holds enemy unit ids (-1 = empty slot). Layout 3 appended `pc` (p3) and",
            "`vx` (v3) after the layout-2 blocks, which kept their offsets.", "",
            "Versioning: changing a feature, range or order changes its family's digest. A legacy family (p2, v2) must not",
            "change: its digest is frozen (`fusion/families.py` LEGACY_DIGESTS, checked by the tests). A new family gets a",
            "new name; a new combat semantics bumps `kCombatVersion`. Runs record the family they trained on; datasets,",
            "warm starts, evaluation and export refuse a family whose digest the extractor no longer reproduces. Exported",
            "headers `static_assert` the spec version and, for p3/v3, the family and the input widths.", "",
            "**v2** (from v1) adds the score-distribution value aggregates `rivals_tied`, `rivals_1_2_ahead`,",
            "`rivals_1_2_behind` and `pairwise_now`. Under v1, team 0 with scores [5,6,4] and [5,6,5] had identical",
            "value inputs although the outcome target differs (0.5 and 0.25 if nobody scores again).", ""]
    return "\n".join(out)


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} is stale: run python tools/pipeline/gen_feature_spec.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
