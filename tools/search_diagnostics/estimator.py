"""Dry-run cost of a branch-diagnostic collection (task T0085). Counts are exact; time is measured or unavailable.

The default design is T0086's registration: team counts 2-15, 8 source maps per count, 3 roots per map (336 roots),
the primary set of 8 candidates (the actual direct order included) x 4 repeats = 10,752 branch episodes, plus a
predeclared subset of 2 maps per count with 1 root each expanded to 24 candidates with the same 4 repeats
(16 more candidates per root: 1,792 episodes), 112 source games, and a separately ledgered smoke of at most 16
branches. Every branch is a whole game: the faithful protocol replays the causal prefix through every bot process
before the root, so a branch costs about one full game whatever the root turn.

Wall time comes only from measured branches: --measured reads a collection's rows (wall_s of complete branches, per
team count). A count without measured branches has no time estimate, and then neither has the total; no step time
is assumed. With --model linear-teams, counts without measurements get a least-squares line of branch wall time
against team count, fitted on the measured branches (two or more measured counts required); those counts are
labelled "modelled, not measured".
"""
from __future__ import annotations

from pathlib import Path

from tools.search_diagnostics.schemas import SCHEMA_ESTIMATE, ValidationError, read_gz

T0086 = {"counts": list(range(2, 16)), "maps_per_count": 8, "roots_per_map": 3, "candidates": 8, "repeats": 4,
         "subset_maps_per_count": 2, "subset_roots_per_map": 1, "subset_candidates": 24, "smoke_branches": 16,
         "workers": 8}


def measured_seconds(rows_dir: Path) -> dict[int, dict]:
    by: dict[int, list[float]] = {}
    for p in sorted(Path(rows_dir).glob("n[0-9][0-9]/*.json.gz")):
        row = read_gz(p)
        n = int(p.parent.name[1:])
        by.setdefault(n, []).extend(b["wall_s"] for b in row.get("branches", []) if b.get("status") == "complete")
    return {n: {"branches": len(v), "mean_wall_s": sum(v) / len(v)} for n, v in by.items() if v}


def linear_teams(measured: dict[int, dict]) -> tuple[float, float]:
    """Least squares wall_s = a + b * teams over measured counts, weighted by their branch counts."""
    if len(measured) < 2:
        raise ValidationError("the linear-teams model needs measured branches at two or more team counts")
    w = {n: m["branches"] for n, m in measured.items()}
    sw = sum(w.values())
    mx = sum(w[n] * n for n in w) / sw
    my = sum(w[n] * measured[n]["mean_wall_s"] for n in w) / sw
    b = sum(w[n] * (n - mx) * (measured[n]["mean_wall_s"] - my) for n in w) / sum(w[n] * (n - mx) ** 2 for n in w)
    return my - b * mx, b


def estimate(design: dict | None = None, measured: dict[int, dict] | None = None, model: str | None = None) -> dict:
    d = {**T0086, **(design or {})}
    fit = linear_teams(measured or {}) if model == "linear-teams" else None
    if model not in (None, "linear-teams"):
        raise ValidationError(f"unknown model {model}")
    for k in ("maps_per_count", "roots_per_map", "candidates", "repeats", "workers"):
        if d[k] < 1:
            raise ValidationError(f"{k} must be >= 1")
    if d["subset_candidates"] and d["subset_candidates"] < d["candidates"]:
        raise ValidationError("the subset must extend the primary set")
    per_count = {}
    for n in d["counts"]:
        roots = d["maps_per_count"] * d["roots_per_map"]
        primary = roots * d["candidates"] * d["repeats"]
        extra = d["subset_maps_per_count"] * d["subset_roots_per_map"] * (d["subset_candidates"] - d["candidates"]) * d["repeats"]
        games = d["maps_per_count"]
        m = (measured or {}).get(n)
        per = m["mean_wall_s"] if m else (fit[0] + fit[1] * n if fit else None)
        cpu = (primary + extra + games) * per if per is not None and per > 0 else None
        status = "measured" if m else "modelled, not measured (linear-teams)" if cpu is not None else \
            "unavailable: no measured branches"
        per_count[str(n)] = {"roots": roots, "primary_episodes": primary, "subset_episodes": extra, "source_games": games,
                             "measured_branches": m["branches"] if m else 0, "branch_wall_s": per,
                             "cpu_seconds": cpu, "time_status": status}
    tot = lambda k: sum(v[k] for v in per_count.values())  # noqa: E731
    cpus = [v["cpu_seconds"] for v in per_count.values()]
    cpu = None if any(c is None for c in cpus) else sum(cpus)
    return {
        "schema": SCHEMA_ESTIMATE, "design": d, "roots": tot("roots"), "primary_episodes": tot("primary_episodes"),
        "subset_episodes": tot("subset_episodes"), "episodes": tot("primary_episodes") + tot("subset_episodes"),
        "source_games": tot("source_games"), "smoke_branches_max": d["smoke_branches"],
        "cpu_hours": cpu / 3600 if cpu is not None else None,
        "wall_hours": cpu / 3600 / d["workers"] if cpu is not None else None,
        "time_status": ("unavailable: measure every team count, or pass --model linear-teams" if cpu is None else
                        "measured on the given rows" if not fit else
                        f"partly modelled: wall_s = {fit[0]:.3f} + {fit[1]:.3f} x teams on unmeasured counts"),
        "per_count": per_count,
        "notes": "an estimate until the full run is timed; parallel overheads and opponent start-up are inside the "
                 "measured wall_s only if the smoke ran under the same worker count",
    }


def markdown(e: dict) -> str:
    h = "n/a" if e["wall_hours"] is None else f"{e['wall_hours']:.2f}"
    lines = ["| Quantity | Value |", "|---|---:|",
             f"| Roots | {e['roots']:,} |", f"| Primary branch episodes | {e['primary_episodes']:,} |",
             f"| Subset episodes | {e['subset_episodes']:,} |", f"| Total episodes | {e['episodes']:,} |",
             f"| Source games | {e['source_games']:,} |", f"| Smoke branches (max) | {e['smoke_branches_max']} |",
             f"| Wall hours on {e['design']['workers']} workers | {h} |", "", f"Time: {e['time_status']}."]
    return "\n".join(lines) + "\n"

