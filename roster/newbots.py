"""Collect CTF bot sources found anywhere in an AI_Test folder that the archive does not hold yet.

    python -m roster.newbots scan --ai-test AI_TEST [--out roster/new_bots]
    python -m roster.inventory --scan-dir @repo/roster/new_bots --out data/roster/bot_inventory_new.json
    python -m roster.newbots inventory --inventory data/roster/bot_inventory_new.json   # -> roster/new_bots/inventory.json

`scan` walks the whole AI_Test folder (including untracked files; .git,
node_modules and build folders skipped; OneDrive placeholders that are not
downloaded are listed, never forced). A file is a CTF bot candidate when it
has a main, reads INIT and TURN blocks and parses the CTF map (BASES, SPOTS
or FLAGS: Bomberman bots also read INIT/TURN), and roster.inventory does not
classify it as a test, tool or driver. A candidate is new when no byte
variant (as is, CRLF->LF, LF->CRLF) of it is archived and its code identity
(comments and layout ignored) is not archived either.

New candidates are deduplicated by code identity (one file per distinct
program, every copy's path recorded) and compared with every archived source
by 8-token-shingle Jaccard similarity:

  independent/   closest archived similarity below 0.5: new code;
  forks/         0.5 or more: a version or variant of an archived bot;
  skipped        similarity >= 0.999 (only strings or comments changed: the
                 same program for our purposes), or local #include files
                 (cannot compile alone; e.g. parameterized master.cpp
                 sources whose builds are archived).

Files keep their exact bytes (roster/new_bots/**/*.cpp is -text in
.gitattributes), under readable names derived from their origin path.
manifest.json and README.md are written from the scan. `inventory` turns a
roster.inventory run that included `--scan-dir @repo/roster/new_bots` into
roster/new_bots/inventory.json (machine-local build paths removed): the full
roster, archived bots plus these, for the existing fingerprint, distance and
strength commands.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import time
from pathlib import Path

from fusion.paths import REPO, ai_test_root
from roster.archive import ARCHIVE
from roster.common import load_json, save_json, sha256_bytes
from roster.inventory import exclusion_reason, jaccard, normalized_code, shingles, strip_code

OUT = REPO / "roster" / "new_bots"
SKIP_PARTS = {".git", "node_modules", "build", "__pycache__"}
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
GENERIC = {"workspace", "submission", "experiments", "judge", "ctf", "bots", "source", "versions", "analysis", "lab"}
INDEPENDENT_BELOW, COSMETIC_FROM = 0.5, 0.999


def dataless(p: Path) -> bool:
    """A OneDrive/iCloud placeholder whose content is not on disk (reading it would trigger a download)."""
    return bool(getattr(os.stat(p, follow_symlinks=False), "st_flags", 0) & getattr(stat, "SF_DATALESS", 0x40000000))


def is_ctf_bot(text: str) -> bool:
    return (re.search(r"\b(?:int|signed|auto)\s+main\s*\(", text) is not None and "INIT" in text and "TURN" in text
            and any(k in text for k in ("BASES", "SPOTS", "FLAGS")))


def variants(data: bytes) -> set[str]:
    lf = data.replace(b"\r\n", b"\n")
    return {sha256_bytes(data), sha256_bytes(lf), sha256_bytes(lf.replace(b"\n", b"\r\n"))}


def readable_name(rel: str, taken: set[str]) -> str:
    parts = Path(rel).parts
    keep = [p for p in parts[:-1] if p.lower() not in GENERIC and not p.startswith(".")][-3:]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(keep + [Path(rel).stem])) or "bot"
    name, k = base, 2
    while name in taken:
        name, k = f"{base}_{k}", k + 1
    taken.add(name)
    return name


def scan(root: Path, archive: Path = ARCHIVE) -> dict:
    manifest = load_json(archive / "manifest.json")["bots"]
    archived_shas = {e["source_sha256"] for e in manifest.values()}
    archived_code = {e["code_sha256"] for e in manifest.values() if e.get("code_sha256")}
    reps = {}
    for e in manifest.values():
        if e.get("archived") and e["source_sha256"] not in reps:
            text = (archive / e["archived"]).read_text(encoding="utf-8", errors="replace")
            reps[e["source_sha256"]] = (e["id"], shingles(strip_code(text)))
    counts = {"cpp": 0, "archived": 0, "same_code": 0, "not_ctf_bot": 0, "driver": 0, "new_files": 0}
    placeholders, groups = [], {}
    for p in sorted(root.rglob("*.cpp")):
        rel = p.relative_to(root).as_posix()
        if SKIP_PARTS & set(p.relative_to(root).parts) or not p.is_file() or p.is_symlink():
            continue
        counts["cpp"] += 1
        if dataless(p):
            placeholders.append(rel)
            continue
        data = p.read_bytes()
        if variants(data) & archived_shas:
            counts["archived"] += 1
            continue
        text = data.decode("utf-8", errors="replace")
        if not is_ctf_bot(text):
            counts["not_ctf_bot"] += 1
            continue
        if exclusion_reason(rel, text, "cpp"):
            counts["driver"] += 1
            continue
        code = sha256_bytes(normalized_code(text).encode())
        if code in archived_code:
            counts["same_code"] += 1
            continue
        counts["new_files"] += 1
        g = groups.setdefault(code, {"code_sha256": code, "paths": [], "source_sha256": sha256_bytes(data), "data": data,
                                     "text": text, "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))})
        g["paths"].append(rel)
    bots, skipped = [], []
    for g in groups.values():
        sh = shingles(strip_code(g["text"]))
        sim, near = max(((jaccard(sh, s), bid) for bid, s in reps.values()), default=(0.0, None))
        includes = sorted(set(INCLUDE.findall(g["text"])))
        entry = {"origin": g["paths"][0], "copies": g["paths"][1:], "source_sha256": g["source_sha256"],
                 "code_sha256": g["code_sha256"], "bytes": len(g["data"]), "mtime": g["mtime"],
                 "closest_archived": near, "similarity": round(sim, 3)}
        if includes:
            skipped.append(entry | {"reason": f"local includes {includes}: cannot compile alone"})
        elif sim >= COSMETIC_FROM:
            skipped.append(entry | {"reason": f"same program as archived {near} apart from strings/comments"})
        else:
            bots.append(entry | {"tier": "independent" if sim < INDEPENDENT_BELOW else "forks", "_data": g["data"]})
    return {"bots": sorted(bots, key=lambda b: (b["tier"] != "independent", b["similarity"], b["origin"])),
            "skipped": sorted(skipped, key=lambda b: b["origin"]), "counts": counts, "placeholders": placeholders}


def write(result: dict, root: Path, out: Path = OUT) -> dict:
    for tier in ("independent", "forks"):
        if (out / tier).is_dir():
            shutil.rmtree(out / tier)
        (out / tier).mkdir(parents=True)
    taken: set[str] = set()
    for b in result["bots"]:
        name = readable_name(b["origin"], taken)
        b["file"] = f"{b['tier']}/{name}.cpp"
        (out / b["file"]).write_bytes(b.pop("_data"))
    manifest = {"schema": "ctf-new-bots-1", "scanned": time.strftime("%Y-%m-%dT%H:%M:%S"), "ai_test_root_name": root.name,
                "thresholds": {"independent_below": INDEPENDENT_BELOW, "cosmetic_from": COSMETIC_FROM},
                "counts": result["counts"] | {"independent": sum(b["tier"] == "independent" for b in result["bots"]),
                                              "forks": sum(b["tier"] == "forks" for b in result["bots"]),
                                              "skipped": len(result["skipped"]), "placeholders": len(result["placeholders"])},
                "bots": result["bots"], "skipped": result["skipped"], "placeholders": result["placeholders"]}
    save_json(out / "manifest.json", manifest)
    (out / "README.md").write_text(readme(manifest), encoding="utf-8")
    return manifest


def readme(m: dict) -> str:
    c = m["counts"]
    out = [
        "# New bots: CTF sources found in AI_Test that the archive does not hold",
        "",
        "<!-- Generated by `python -m roster.newbots scan` from manifest.json. Do not edit by hand. -->",
        "",
        f"Scan of the local AI_Test folder on {m['scanned'][:10]}: {c['cpp']} .cpp files read; {c['archived']} already archived, "
        f"{c['same_code']} the same code as an archived bot, {c['not_ctf_bot']} not CTF bot candidates (Bomberman, tools, binaries, stubs that never read the map), "
        f"{c['driver']} tests or drivers. {c['new_files']} new files are {c['independent'] + c['forks'] + c['skipped']} distinct programs: "
        f"**{c['independent']} independent**, **{c['forks']} forks**, {c['skipped']} skipped. "
        f"{c['placeholders']} OneDrive placeholder(s) were not downloaded and not read.",
        "",
        "| Folder | What it holds |",
        "|---|---|",
        f"| `independent/` | Closest archived similarity below {m['thresholds']['independent_below']} (8-token-shingle Jaccard): new code. |",
        "| `forks/` | Versions and variants of archived bots (tuning runs, experiment lines, earlier versions). |",
        "| `manifest.json` | Every bot: origin path and all copies, hashes, size, mtime, closest archived bot and similarity; skipped programs with reasons. |",
        "| `inventory.json` | After `roster.newbots inventory`: the full roster (archived bots plus these) for the roster commands. |",
        "",
        "Files keep their exact bytes; never edit them (the hashes are their identity). Names come from their origin path.",
        "",
        "## Run them",
        "",
        "Build the combined inventory once (compiles every candidate; failures are excluded with the compiler's error):",
        "",
        "```bash",
        "python -m roster.inventory --scan-dir @repo/roster/new_bots --out data/roster/bot_inventory_new.json",
        "```",
        "",
        "```bash",
        "python -m roster.newbots inventory --inventory data/roster/bot_inventory_new.json",
        "```",
        "",
        "Then use `--inventory roster/new_bots/inventory.json` in place of `roster/bots/inventory.json` in the commands of",
        "[docs/FIRST_RUN.md](../../docs/FIRST_RUN.md); restrict to these bots with `--only` (ids start with `@repo/roster/new_bots/`).",
        "Fingerprints of the archived bots stay valid (same identities); only the new bots need new runs, on the same frozen suite.",
        "To play one by hand: `g++ -std=c++20 -O2 -o bot roster/new_bots/independent/<name>.cpp`, or add it to a pool",
        "(`{\"name\": ..., \"path\": \"@repo/roster/new_bots/independent/<name>.cpp\", \"family\": ...}`).",
        "",
        "Similarity is shared code, not shared behavior: a fork can play very differently and independent code can play alike.",
        "Behavior comes from `roster.distance`, outcomes from `roster.trajectory`, strength from `roster.strength`.",
        "",
    ]
    for tier, title in (("independent", "Independent bots"), ("forks", "Forks")):
        rows = [b for b in m["bots"] if b["tier"] == tier]
        out += [f"## {title} ({len(rows)})", "", "| # | File | Origin (AI_Test) | Copies | Bytes | Modified | Closest archived | Similarity |",
                "|---:|---|---|---:|---:|---|---|---:|"]
        out += [f"| {i} | [`{b['file']}`]({b['file']}) | `{b['origin']}` | {len(b['copies'])} | {b['bytes']} | {b['mtime']} | "
                f"`{b['closest_archived']}` | {b['similarity']} |" for i, b in enumerate(rows, 1)]
        out.append("")
    out += ["## Skipped", "", "| Origin | Reason |", "|---|---|"]
    out += [f"| `{b['origin']}` | {b['reason']} |" for b in m["skipped"]]
    if m["placeholders"]:
        out += ["", "## Not read (OneDrive placeholders)", ""] + [f"- `{p}`" for p in m["placeholders"]]
    return "\n".join(out) + "\n"


def inventory(inv: dict, out: Path = OUT) -> Path:
    local = ("binary", "path", "command")
    bots = [b | {"build": {k: v for k, v in b.get("build", {}).items() if k not in local}} for b in inv["bots"]]
    new = [b for b in bots if b["id"].startswith("@repo/roster/new_bots/")]
    if not new:
        raise SystemExit("this inventory has no @repo/roster/new_bots bots: run roster.inventory with --scan-dir @repo/roster/new_bots")
    snap = inv | {"bots": bots, "ai_test_root": None,
                  "_comment": "Full roster: archived bots (run from roster/bots/src) plus roster/new_bots (run in place). "
                              "Generated by roster.newbots inventory; build fields describe the machine that inventoried."}
    save_json(out / "inventory.json", snap, indent=None)
    return out / "inventory.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--ai-test", help="AI_Test folder to scan (or AI_TEST_ROOT)")
    s.add_argument("--out", default=str(OUT))
    i = sub.add_parser("inventory")
    i.add_argument("--inventory", required=True, help="a roster.inventory output that scanned @repo/roster/new_bots")
    i.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        root = ai_test_root(a.ai_test).resolve()
        m = write(scan(root), root, Path(a.out))
        print(f"[newbots] {m['counts']} -> {a.out}")
    else:
        inv = load_json(a.inventory)
        path = inventory(inv, Path(a.out))
        print(f"[newbots] full inventory, {sum(b['id'].startswith('@repo/roster/new_bots/') for b in inv['bots'])} new bots -> {path}")


if __name__ == "__main__":
    main()
