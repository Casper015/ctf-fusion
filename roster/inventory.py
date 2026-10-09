"""Inventory every candidate bot: sources, builds, aliases, code lineage and explicit exclusions.

    python -m roster.inventory --ai-test ~/code/AI_Test --out data/roster/bot_inventory.json
    python -m roster.inventory --ai-test ... --scan-dir some/new/exports --no-build

Discovery is not limited to profiles/pool.json: every .cpp under the scan
directories (the handoff's list, extendable with --scan-dir) plus every pool
entry. For each candidate it records:

  * identity: repository-relative path, source sha256, the sha256 of its local
    #include closure (dependency hash), size, language;
  * build: compiled with the runner's flags (fusion.runner, the same as data
    collection), compile time, binary sha256, or the compiler's error;
  * aliases: byte-identical sources with their includes (byte_alias_of), and
    sources whose code is identical apart from comments and formatting
    (code_alias_of: the same token stream, so the same compiled policy).
    Binary hashes are recorded but never used for aliasing: toolchains stamp
    each build (macOS UUIDs, PE timestamps), so one file compiled twice
    already gives two binaries;
  * lineage: code similarity (Jaccard over 8-token shingles of comment-free
    source) to every other distinct source, the nearest relative, and groups
    of sources above --lineage-threshold (default 0.7; connected components,
    so chains merge), plus the groups at 0.5, 0.7 and 0.9 similarity. At 0.5
    forks chain official bots into our own family (a Player2 experiment
    shares 56% of Sol's code), so the default is 0.7. Shared code is lineage
    evidence, not behavior equivalence: roster.distance measures behavior per
    team count;
  * exclusion: a reason for every candidate that cannot be an opponent (no
    entry point, test or tool driver, build failure). Excluded candidates stay
    in the inventory.

Only one representative of each code identity is built and evaluated (pool
entries first, then the judge bots, then exports and benchmark suites, then
the rest; ties by path); an alias never counts as an independent bot.
Nothing here measures strength.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fusion import runner
from fusion.paths import REPO, ai_test_root, bot_path, git_commit, load_pool
from roster.common import SCHEMA_INVENTORY, save_json, sha256_bytes, sha256_file

# The handoff's directories (handoffs/diverse15_distance_tiered_policy/tools/inventory_bots.py) plus New_AI_Test.
DEFAULT_DIRS = [
    "ctf-arena/Player7/judge/ctf/bots", "ctf-arena/Player7/workspace/experiments/round3/export",
    "ctf-stress-pack/bots", "ctf-stress-pack/bots_v2", "ctf-stress-pack/bots_v3",
    "ctf-stress-pack/bots_c_pending", "ctf-stress-pack/bots_10k_suite",
    "ctf-benchmark-v3/track1_10k_suite/bots", "ctf-benchmark-v3/track2_2k_c_series/bots",
    "ctf-s1-controls/exports", "New_AI_Test",
]
DRIVER_PARTS = {"tests", "test", "judge_tests", "tools", "bench", "benchmarks", "kernel"}
DRIVER_NAMES = re.compile(r"(^|_)(test|tests|driver|bench|benchmark|check|dump|probe|harness)(_|\.|$)", re.I)
MAIN = re.compile(r"\b(?:int|signed|auto)\s+main\s*\(")
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|\S")
LEX = re.compile(r'''(?P<comment>//[^\n]*|/\*.*?\*/)
                   |(?P<string>R"[^(\s]*\(.*?\)[^"\s]*"|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
                   |(?P<token>[A-Za-z_]\w*|\d[\w.]*|\S)''', re.S | re.X)
SHINGLE = 8


def strip_code(text: str) -> str:
    """Remove comments and string/char literals so that similarity reflects code."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '""', text)


def normalized_code(text: str) -> str:
    """The token stream without comments or layout; string literals kept verbatim."""
    return " ".join(m.group() for m in LEX.finditer(text) if m.group("comment") is None)


def shingles(text: str) -> set[int]:
    tok = TOKEN.findall(strip_code(text))
    return {zlib.crc32(" ".join(tok[i:i + SHINGLE]).encode()) for i in range(max(1, len(tok) - SHINGLE + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def include_closure(path: Path, root: Path) -> tuple[list[Path], list[str]]:
    """Local #include "..." files reachable from `path` (inside root), and includes that could not be resolved."""
    seen, missing, stack = [], [], [path]
    while stack:
        p = stack.pop()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in INCLUDE.findall(text):
            q = (p.parent / name).resolve()
            if not q.is_file() or not q.is_relative_to(root):
                missing.append(name)
            elif q not in seen and q != path:
                seen.append(q)
                stack.append(q)
    return sorted(seen), sorted(set(missing))


def exclusion_reason(rel: str, text: str, language: str) -> str | None:
    if language != "cpp":
        return None
    parts = set(Path(rel).parts[:-1])
    name = Path(rel).name
    if parts & DRIVER_PARTS or DRIVER_NAMES.search(name):
        return "test or tool driver (path or name), not an opponent"
    if not MAIN.search(strip_code(text)):
        return "no main(): a library or fragment, not a standalone bot"
    return None


PRIORITY = ("ctf-arena/Player7/judge/", "ctf-arena/Player7/workspace/experiments/round3/export/",
            "ctf-benchmark-v3/", "ctf-s1-controls/", "ctf-stress-pack/")


def priority(bot: dict) -> tuple:
    """Representatives and lineage names prefer pool entries, then official and frozen locations."""
    if bot.get("pool_name"):
        return (0, bot["id"])
    for i, prefix in enumerate(PRIORITY, 1):
        if bot["id"].startswith(prefix):
            return (i, bot["id"])
    return (len(PRIORITY) + 1, bot["id"])


def discover(root: Path, dirs: list[str], pool: dict) -> tuple[list[dict], list[str]]:
    """Candidates from the scan directories plus every pool entry; missing directories are reported."""
    found, missing = {}, []
    for d in dirs:
        top = REPO.resolve() if d.startswith("@repo/") else root   # @repo/...: a folder inside this repository
        base = (top / d.removeprefix("@repo/")).resolve()
        if not base.is_relative_to(top):
            raise SystemExit(f"--scan-dir {d} leaves {'this repository' if top != root else 'the AI_Test root'}")
        if not base.is_dir():
            missing.append(d)
            continue
        for p in base.rglob("*.cpp"):
            if p.is_file() and "build" not in p.relative_to(top).parts:
                found[p.resolve()] = None
    names = {}
    for entry in pool["bots"]:
        p = bot_path(entry, root).resolve()
        if p.is_file():
            found[p] = None
            names[p] = entry
    out = []
    for p in sorted(found):
        inside_repo = p.is_relative_to(REPO.resolve())
        rel = ("@repo/" + p.relative_to(REPO.resolve()).as_posix()) if inside_repo else p.relative_to(root).as_posix()
        entry = names.get(p)
        out.append({"id": rel, "path": rel, "abs": str(p), "language": "cpp" if p.suffix == ".cpp" else p.suffix.lstrip("."),
                    "pool_name": entry["name"] if entry else None, "pool_family": entry["family"] if entry else None,
                    "policy_labels": entry.get("policy_labels", True) if entry else True})
    return sorted(out, key=priority), missing


def build_one(bot: dict, build_dir: Path) -> dict:
    t0 = time.time()
    try:
        exe = runner.compile_cpp(Path(bot["abs"]), build_dir)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        lines = [l for l in str(exc).splitlines() if "error" in l.lower()] or str(exc).splitlines()
        return {"status": "failed", "error": (lines[0] if lines else "unknown")[:400], "seconds": round(time.time() - t0, 1)}
    return {"status": "ok", "binary": str(exe), "binary_sha256": sha256_file(exe), "seconds": round(time.time() - t0, 1),
            "compiler": runner.find_compiler(), "flags": runner.CXX_FLAGS}


def components(ids: list[str], pairs: dict, threshold: float, rank: dict) -> dict[str, str]:
    """Connected components of the similarity graph at `threshold`, named after their highest-priority member."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), s in pairs.items():
        if s >= threshold:
            parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    label = {}
    for members in groups.values():
        name = "lineage:" + min(members, key=rank.get).rsplit("/", 1)[-1].removesuffix(".cpp")
        for m in members:
            label[m] = name
    return label


def lineage(bots: list[dict], texts: dict[str, str], threshold: float) -> None:
    """Code-similarity nearest relative and groups (union-find over pairs above the threshold), over distinct code."""
    reps = [b for b in bots if not b["byte_alias_of"] and not b["code_alias_of"] and b["language"] == "cpp"]
    rank = {b["id"]: priority(b) for b in bots}
    sh = {b["id"]: shingles(texts[b["id"]]) for b in reps}
    pairs = {}
    best = {b["id"]: (None, 0.0) for b in reps}
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            s = jaccard(sh[a["id"]], sh[b["id"]])
            pairs[(a["id"], b["id"])] = s
            for x, y in ((a["id"], b["id"]), (b["id"], a["id"])):
                if s > best[x][1]:
                    best[x] = (y, s)
    ids = [b["id"] for b in reps]
    label = components(ids, pairs, threshold, rank)
    levels = {t: components(ids, pairs, t, rank) for t in (0.5, 0.7, 0.9)}
    by_id = {b["id"]: b for b in bots}
    for b in bots:
        rep = b["byte_alias_of"] or b["code_alias_of"] or b["id"]
        b["lineage_group"] = label.get(rep, f"lineage:{b['language']}:{Path(rep).stem}")
        b["lineage_levels"] = {str(t): f.get(rep, b["lineage_group"]) for t, f in levels.items()}
        rel, sim = best.get(rep, (None, 0.0))
        b["nearest_relative"] = {"id": rel, "similarity": round(sim, 4)} if rel else None
        b["lineage_group_size"] = sum(1 for x in label.values() if x == b["lineage_group"]) or 1
    for b in bots:  # aliases inherit their representative's build
        rep = b["byte_alias_of"] or b["code_alias_of"]
        if rep:
            b["build"] = dict(by_id[rep].get("build") or {}) | {"inherited_from": rep}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai-test", help="AI_Test checkout (or AI_TEST_ROOT)")
    ap.add_argument("--scan-dir", action="append", default=[],
                    help="additional AI_Test-relative directory to scan; @repo/<dir> scans a folder of this repository")
    ap.add_argument("--no-build", action="store_true", help="skip compilation (no build or binary-alias evidence)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel compilations")
    ap.add_argument("--lineage-threshold", type=float, default=0.7, help="code Jaccard for one lineage group")
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "bot_inventory.json"))
    a = ap.parse_args(argv)
    root = ai_test_root(a.ai_test).resolve()
    bots, missing = discover(root, DEFAULT_DIRS + a.scan_dir, load_pool())
    texts, first, first_code = {}, {}, {}
    for b in bots:
        p = Path(b["abs"])
        raw = p.read_bytes()
        texts[b["id"]] = raw.decode("utf-8", errors="replace")
        deps, unresolved = include_closure(p, root) if b["language"] == "cpp" else ([], [])
        dep_blob = b"".join(d.relative_to(root).as_posix().encode() + b"\0" + d.read_bytes() for d in deps)
        code = "\0".join(normalized_code(t) for t in [texts[b["id"]], *(d.read_text(encoding="utf-8", errors="replace") for d in deps)])
        b |= {"source_sha256": sha256_bytes(raw), "source_bytes": len(raw), "local_includes": [d.relative_to(root).as_posix() for d in deps],
              "unresolved_includes": unresolved, "dependency_sha256": sha256_bytes(raw + b"\0" + dep_blob),
              "code_sha256": sha256_bytes(code.encode())}
        b["byte_alias_of"] = first.get(b["dependency_sha256"])
        if b["byte_alias_of"] is None:
            first[b["dependency_sha256"]] = b["id"]
        b["code_alias_of"] = None if b["byte_alias_of"] else first_code.get(b["code_sha256"])
        first_code.setdefault(b["code_sha256"], b["id"])
        b["excluded"] = exclusion_reason(b["path"], texts[b["id"]], b["language"])
        if unresolved and not b["excluded"]:
            b["excluded"] = f"unresolved local includes: {', '.join(unresolved)}"
    to_build = [b for b in bots if b["language"] == "cpp" and not b["excluded"] and not b["byte_alias_of"] and not b["code_alias_of"]]
    if not a.no_build:
        build_dir = REPO / "build" / "roster"
        with ThreadPoolExecutor(max(1, a.jobs)) as pool:
            for b, result in zip(to_build, pool.map(lambda x: build_one(x, build_dir), to_build)):
                b["build"] = result
                if result["status"] != "ok":
                    b["excluded"] = f"build failed: {result['error']}"
                print(f"[inventory] {b['id']}: {result['status']} ({result['seconds']} s)", flush=True)
    for b in bots:
        b.setdefault("build", {"status": "not built" if b["language"] == "cpp" else "interpreted"})
    lineage(bots, texts, a.lineage_threshold)
    for b in bots:
        if b["byte_alias_of"] and not b["excluded"]:
            b["excluded"] = f"byte-identical to {b['byte_alias_of']} (its evidence applies; not an independent bot)"
        elif b["code_alias_of"] and not b["excluded"]:
            b["excluded"] = (f"same code as {b['code_alias_of']} apart from comments and formatting "
                             "(its evidence applies; not an independent bot)")
        rep = b["byte_alias_of"] or b["code_alias_of"]
        b["representative"] = rep or b["id"]
        b["assessment_status"] = "alias" if rep else "excluded" if b["excluded"] else "inventoried"
        b.pop("abs")
    eligible = [b for b in bots if not b["excluded"]]
    out = {"schema": SCHEMA_INVENTORY, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "ai_test_root": str(root),
           "ai_test_commit": git_commit(root), "scan_dirs": DEFAULT_DIRS + a.scan_dir, "missing_dirs": missing,
           "lineage_threshold": a.lineage_threshold, "built": not a.no_build,
           "counts": {"candidates": len(bots), "distinct_sources": len(first), "distinct_code": len(first_code),
                      "excluded": len(bots) - len(eligible), "eligible": len(eligible),
                      "lineage_groups": len({b["lineage_group"] for b in eligible})},
           "bots": bots,
           "limits": "Inventory and builds only. Code similarity is lineage evidence; behavior equivalence comes from "
                     "roster.distance, strength from roster.strength. The working tree may hold uncommitted files."}
    save_json(a.out, out)
    print(f"[inventory] {out['counts']} -> {a.out}")
    return out


if __name__ == "__main__":
    main()
