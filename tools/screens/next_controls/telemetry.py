"""Parse the telemetry reported by the frozen next-controls arms."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

FORMATS = ("agent_v2", "meowmind")
BASE = ("turns", "refused", "inconsistent", "search_used", "search_fallback", "leaves")
SEARCH = ("trajectories", "refuse_setup", "refuse_trajectory", "refuse_expand", "refuse_critic", "refuse_late",
          "max_setup_us", "max_traj_us", "max_expand_us", "max_critic_us", "max_search_us", "branches", "critic")
FIELDS = BASE + SEARCH
COUNTER_RE = re.compile(
    r"\bAGENTV2\s+"
    + r"\s+".join(rf"{k}\s+(\d+)" for k in BASE)
    + r"(?:\s+"
    + r"\s+".join(k + (r"\s+(\S+)" if k == "critic" else r"\s+(\d+)") for k in SEARCH)
    + r")?\s*$"
)
IDENTITY_RE = re.compile(r"\bAGENTV2\s+(carrier\s+.+?)\s*$")


def format_for(variant: str | Mapping) -> str:
    """Explicit entry telemetry wins; otherwise infer the format from its name."""
    if isinstance(variant, str):
        name, fmt = variant, None
    else:
        name, fmt = variant["name"], variant.get("telemetry")
        if "telemetry" in variant and fmt not in FORMATS:
            raise ValueError(f"unknown telemetry format: {fmt!r}")
    fmt = fmt if fmt is not None else "meowmind" if name.startswith("meowmind") else "agent_v2"
    if fmt not in FORMATS:
        raise ValueError(f"unknown telemetry format: {fmt!r}")
    return fmt


def _check_format(fmt: str) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"unknown telemetry format: {fmt!r}")


def parse(fmt: str, stderr_lines: Iterable[str]) -> dict:
    """Return identity, last cumulative counters and absent field names, without defaults."""
    _check_format(fmt)
    identity, counters = None, None
    if fmt == "agent_v2":
        if isinstance(stderr_lines, str):
            stderr_lines = stderr_lines.splitlines()
        for line in stderr_lines:
            match = IDENTITY_RE.search(line)
            if match and identity is None:
                tokens = match[1].split()
                candidate = dict(zip(tokens[::2], tokens[1::2])) if len(tokens) % 2 == 0 else {}
                required = ("actor", "digest", "predictor", "model", "search", "greedy")
                if all(k in candidate for k in required):
                    try:
                        for k in ("scenarios", "search", "greedy"):
                            if k in candidate:
                                candidate[k] = int(candidate[k])
                    except ValueError:
                        continue
                    identity = candidate
            match = COUNTER_RE.search(line)
            if match:
                counters = {k: v if k == "critic" else int(v)
                            for k, v in zip(FIELDS, match.groups()) if v is not None}
    return {"format": fmt, "identity": identity, "counters": counters,
            "absent": [k for k in FIELDS if counters is None or k not in counters]}


def complete(fmt: str, parsed: dict, turns_played: int) -> tuple[bool, list[str]]:
    """Check against the game's turn count; Meowmind needs parsed['runner_turns']."""
    _check_format(fmt)
    why = []
    if fmt == "agent_v2":
        if not parsed.get("identity"):
            why.append("identity missing")
        counters = parsed.get("counters")
        if counters is None:
            why.append("counters missing")
        elif counters.get("turns") != turns_played:
            why.append(f"counter turns {counters.get('turns')} != game turns {turns_played}")
    else:
        observed = parsed.get("runner_turns")
        if observed is None:
            why.append("runner turn count missing")
        elif observed != turns_played:
            why.append(f"runner turns {observed} != game turns {turns_played}")
    if turns_played is None or turns_played <= 0:
        why.append("game turn count missing or empty")
    return not why, why


def work_totals(fmt: str, parsed_list: Iterable[dict]) -> dict | None:
    """Sum numeric counters present in every game; Meowmind reports no work counters."""
    _check_format(fmt)
    if fmt == "meowmind":
        return None
    counters = [p.get("counters") or {} for p in parsed_list]
    if not counters:
        return {}
    return {k: sum(c[k] for c in counters) for k in FIELDS
            if k != "critic" and all(isinstance(c.get(k), int) for c in counters)}
