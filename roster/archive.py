"""Archive every inventoried bot source in this repository, so later checks never depend on AI_Test's working tree.

    python -m roster.archive save  --inventory data/roster/bot_inventory.json      # writes roster/bots/
    python -m roster.archive check [--pool profiles/open_training_pool.json ...] [--ai-test AI_TEST]
    python -m roster.archive pool  [--all] --out profiles/pool_archive.json   # a collection pool of archived bots
    python -m roster.archive readme                                          # rewrite roster/bots/README.md

Sources are stored once per content, named by their sha256
(roster/bots/src/<sha256>.<ext>), with their local #include files the same
way. .gitattributes marks them -text: git must never convert their line
endings (Windows checkouts), or every hash would break. roster/bots/manifest.json maps every candidate of the inventory
(aliases and excluded ones included) to its archived file, its AI_Test path
and commit, lineage, alias and exclusion. Bots already in this repository
(@repo/ paths) are recorded but not copied. roster/bots/inventory.json is the
inventory snapshot of the latest save (its AI_Test root removed), so the
host can fingerprint and screen the archived sources with
--inventory roster/bots/inventory.json without its own inventory.

check verifies that:
  - every archived file still hashes to its name (no edits);
  - every manifest entry has its source archived;
  - every pool, registry or frozen roster given with --pool names only
    sources that are archived (by source_sha256), so its evidence stays
    traceable;
  - with --ai-test (or AI_TEST_ROOT), which AI_Test files changed or
    disappeared since they were archived (a report, not a failure: the
    archived copy is the evaluated one).
Exit 1 on any integrity or coverage failure.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from fusion.paths import REPO, ai_test_root
from roster.common import load_json, save_json, sha256_bytes, sha256_file

ARCHIVE = REPO / "roster" / "bots"
SCHEMA_ARCHIVE = "ctf-bot-archive-1"
KEEP = ("id", "path", "language", "pool_name", "source_sha256", "source_bytes", "dependency_sha256", "code_sha256",
        "byte_alias_of", "code_alias_of", "representative", "excluded", "lineage_group", "lineage_levels")


def stored_name(sha: str, path: str) -> str:
    return f"src/{sha}{Path(path).suffix or '.txt'}"


def save(inventory: dict, root: Path, out: Path = ARCHIVE) -> dict:
    """Copy every inventoried source (and its local includes) into `out`; refuse a source whose hash moved."""
    (out / "src").mkdir(parents=True, exist_ok=True)
    manifest = load_json(out / "manifest.json") if (out / "manifest.json").exists() else {"bots": {}}
    copied = 0
    for b in inventory["bots"]:
        entry = {k: b.get(k) for k in KEEP}
        if b["path"].startswith("@repo/"):
            entry["archived"] = None
            entry["in_repo"] = b["path"][len("@repo/"):]
        else:
            src = root / b["path"]
            data = src.read_bytes()
            if sha256_bytes(data) != b["source_sha256"]:
                raise SystemExit(f"{b['id']}: {src} changed since the inventory was taken; re-run roster.inventory first")
            name = stored_name(b["source_sha256"], b["path"])
            if not (out / name).exists():
                (out / name).write_bytes(data)
                copied += 1
            entry["archived"] = name
            entry["includes"] = {}
            for inc in b.get("local_includes", []):
                blob = (root / inc).read_bytes()
                iname = stored_name(sha256_bytes(blob), inc)
                if not (out / iname).exists():
                    (out / iname).write_bytes(blob)
                    copied += 1
                entry["includes"][inc] = iname
        entry["ai_test_commit"] = inventory.get("ai_test_commit")
        entry["inventory_created"] = inventory.get("created")
        manifest["bots"][b["id"]] = entry
    manifest |= {"schema": SCHEMA_ARCHIVE, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "_comment": "Every inventoried bot source, stored once by sha256 (python -m roster.archive). Entries "
                             "accumulate across inventories; ai_test_commit is the AI_Test commit of the inventory that "
                             "archived the entry (the working tree may hold uncommitted files)."}
    manifest["bots"] = dict(sorted(manifest["bots"].items()))
    manifest["counts"] = {"candidates": len(manifest["bots"]),
                          "stored_files": sum(1 for _ in (out / "src").iterdir()),
                          "stored_bytes": sum(p.stat().st_size for p in (out / "src").iterdir())}
    save_json(out / "manifest.json", manifest)
    local = ("binary", "path", "command")                          # machine-local build paths stay out of the repo
    bots = [b | {"build": {k: v for k, v in b.get("build", {}).items() if k not in local}} for b in inventory["bots"]]
    snapshot = inventory | {"bots": bots, "ai_test_root": None, "archived": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "_comment": "Inventory snapshot of the archived sources (roster.archive save). Bots run from "
                                        "roster/bots/src by default; build fields describe the machine that inventoried."}
    save_json(out / "inventory.json", snapshot, indent=None)
    write_readme(out)
    return {"copied": copied, **manifest["counts"]}


def check(out: Path = ARCHIVE, pools: list[str] = (), root: Path | None = None) -> dict:
    manifest = load_json(out / "manifest.json")
    errors, changed, missing, untraced = [], [], [], []
    for f in sorted((out / "src").iterdir()):
        if f.stem != sha256_file(f):
            errors.append(f"{f.name}: content no longer matches its sha256 name")
    stored = {f.stem for f in (out / "src").iterdir()}
    for bid, e in manifest["bots"].items():
        if e["archived"] and Path(e["archived"]).stem not in stored:
            errors.append(f"{bid}: archived file {e['archived']} is missing")
        for inc, name in (e.get("includes") or {}).items():
            if Path(name).stem not in stored:
                errors.append(f"{bid}: include {inc} ({name}) is missing")
        if root is not None and e["archived"]:
            p = root / e["path"]
            if not p.exists():
                missing.append(bid)
            elif sha256_file(p) != e["source_sha256"]:
                changed.append(bid)
    in_repo = {e["source_sha256"] for e in manifest["bots"].values() if e.get("in_repo")}
    for pool in pools:
        for m in load_json(pool).get("bots", []):
            sha = m.get("source_sha256")
            if sha is None:
                untraced.append(f"{pool}: {m.get('name')}")             # a legacy pool entry: not hash-locked
            elif sha not in stored and sha not in in_repo:
                errors.append(f"{pool}: {m.get('name')} source {sha[:12]} is not archived")
    return {"errors": errors, "untraced": untraced, "changed_in_ai_test": changed, "missing_in_ai_test": missing,
            "entries": len(manifest["bots"]), "stored_files": len(stored)}


ORIGINS = {   # what each AI_Test source directory holds (for the README); unknown directories are listed as-is
    "ctf-arena/Player7/judge/ctf/bots": "the judge bots of the CTF arena (official reference opponents)",
    "ctf-arena/Player7/workspace/experiments/round3/export": "Player7 round-3 exported candidates",
    "ctf-arena/Player7/workspace/experiments/round3/export/archive_r3_single_mode": "Player7 round-3 single-mode archive",
    "ctf-stress-pack/bots": "stress-pack opponents (includes copies of the judge bots)",
    "ctf-stress-pack/bots_v2": "stress-pack opponents, second batch",
    "ctf-benchmark-v3/track1_10k_suite/bots": "benchmark v3, track 1 (10k-game suite)",
    "ctf-benchmark-v3/track2_2k_c_series/bots": "benchmark v3, track 2 (2k-game C series)",
    "ctf-s1-controls/exports": "S1/S2 heuristic control variants",
}


def files(manifest: dict) -> list[dict]:
    """One row per stored file: every candidate with that content, its status, name and origins."""
    by = {}
    for e in manifest["bots"].values():
        if e.get("archived"):
            by.setdefault(e["archived"], []).append(e)
    rows = []
    for name, es in by.items():
        rep = next((e for e in es if e["representative"] == e["id"] and not e["excluded"]), None)
        alias = next((e for e in es if e.get("code_alias_of")), None)
        if rep:
            status, main = "eligible", rep
        elif alias:
            status, main = f"code alias of `{alias['code_alias_of']}`", alias
        else:
            main = sorted(es, key=lambda e: e["id"])[0]
            status = f"excluded: {main['excluded']}"
        rows.append({"file": name, "sha": es[0]["source_sha256"], "main": main, "status": status,
                     "paths": sorted(e["path"] for e in es), "lineage": main.get("lineage_group")})
    stems = {}
    for r in rows:
        stems.setdefault(Path(r["main"]["path"]).stem, []).append(r)
    for stem, rs in stems.items():
        for r in rs:
            parent = Path(r["main"]["path"]).parent.name
            r["name"] = r["main"].get("pool_name") or (stem if len(rs) == 1 else f"{parent}/{stem}")
    order = {"eligible": 0}
    return sorted(rows, key=lambda r: (order.get(r["status"], 1), str(Path(r["main"]["path"]).parent), r["name"]))


def origin_dir(path: str) -> str:
    return str(Path(path).parent.as_posix())


def readme(manifest: dict) -> str:
    rows = files(manifest)
    eligible = [r for r in rows if r["status"] == "eligible"]
    commit = sorted({e.get("ai_test_commit") for e in manifest["bots"].values() if e.get("ai_test_commit")})
    dirs = {}
    for r in rows:
        dirs.setdefault(origin_dir(r["main"]["path"]), []).append(r)
    in_repo = [e for e in manifest["bots"].values() if e.get("in_repo")]
    out = [
        "# Archived opponent bots",
        "",
        "<!-- Generated by `python -m roster.archive readme` (also run by `save`) from manifest.json. Do not edit by hand:",
        "     change the template in roster/archive.py and regenerate. -->",
        "",
        f"This folder freezes every candidate opponent bot that `roster.inventory` found in AI_Test (commit {', '.join(f'`{c[:7]}`' for c in commit)}),",
        f"so later evidence never depends on AI_Test's working tree: **{len(rows)} stored source files** "
        f"({len(eligible)} eligible representatives, {manifest['counts']['stored_bytes'] / 2 ** 20:.1f} MiB) for "
        f"{manifest['counts']['candidates']} inventoried candidates, plus {len(in_repo)} bots that already live in this repository "
        "(" + ", ".join(f"`{e['in_repo']}`" for e in in_repo) + ").",
        "",
        "## Layout",
        "",
        "| Path | What it is |",
        "|---|---|",
        "| `src/<sha256>.cpp` | One bot source, named by the sha256 of its bytes. Identical copies share one file. Never edit: the name is the checksum. |",
        "| `manifest.json` | Every inventoried candidate (aliases and excluded ones too): AI_Test path, source hash, archived file, alias, exclusion, lineage, AI_Test commit. |",
        "| `inventory.json` | The inventory snapshot the tools read (`--inventory roster/bots/inventory.json`). Machine-local build paths removed. |",
        "| `README.md` | This file, generated. |",
        "",
        "`.gitattributes` marks `src/` as `-text`: git never rewrites line endings (some sources use CRLF), so every hash survives a Windows checkout.",
        "",
        "## Quick start for the next agent",
        "",
        "Everything runs from the ctf-fusion root. Set `AI_TEST_ROOT` to an AI_Test checkout: the engine and the recorded histories come from there, the bot sources from here.",
        "",
        "1. **Check the archive** (hashes, and optionally which AI_Test files changed since):",
        "",
        "   ```bash",
        "   python -m roster.archive check --ai-test $AI_TEST_ROOT",
        "   ```",
        "",
        "2. **Run one bot by hand.** Every bot is a single C++20 file with no local includes. It reads the engine's INIT and TURN blocks on stdin and answers one line per turn",
        "   (`<turn> <move> <action>` for its three units). Build it with the collection flags:",
        "",
        "   ```bash",
        "   g++ -std=c++20 -O2 -o astra roster/bots/src/<sha256>.cpp",
        "   ```",
        "",
        "3. **Play your own bot against them.** Write a collection pool of the archived bots (eligible representatives; `--all` for every stored file), add an entry for your bot, and play:",
        "",
        "   ```bash",
        "   python -m roster.archive pool --out profiles/pool_archive.json",
        "   ```",
        "",
        '   Add your bot to the `bots` list of that file, e.g. `{"name": "MyBot", "path": "path/inside/AI_Test/my_bot.cpp", "family": "mine"}`',
        "   (paths are relative to AI_Test; `@repo/...` means inside ctf-fusion). Then name the seats (cycled over the team count):",
        "",
        "   ```bash",
        "   python -m training.collect --pool profiles/pool_archive.json --teams 4 --matches 20 --bots MyBot,Astra,Opus,Sol --out data/raw_mybot",
        "   ```",
        "",
        "   Without `--bots`, seats are drawn from the pool by weight. Every archived entry carries its `source_sha256`, and collection refuses a file whose bytes changed.",
        "   Live outcomes of those games (kills, deaths, flag episodes, survival):",
        "",
        "   ```bash",
        "   python -m roster.trajectory --raw data/raw_mybot --inventory roster/bots/inventory.json --out data/roster/live_mybot.json",
        "   ```",
        "",
        "4. **Fingerprint and screen the whole roster** (host work, hours): follow `docs/ROSTER.md`, *Host command sequence*. The roster tools run inventory bots from this archive by default",
        "   (`--source ai-test` uses the AI_Test files instead) and refuse any file that no longer matches its hash.",
        "",
        "5. **Add new bots** after AI_Test gains some: `python -m roster.inventory`, then `python -m roster.archive save --inventory data/roster/bot_inventory.json`.",
        "   New files are added, existing entries are kept, and this README is regenerated. Evidence of changed sources is reported as stale, never reused.",
        "",
        "## Rules",
        "",
        "- Never edit, rename or delete files in `src/`; `check` fails on any change. A changed bot is a new file with a new hash.",
        "- A bot's identity is its source hash, not its file name or AI_Test path. Results (fingerprints, strength rows, pools) record that hash.",
        "- Status *eligible* means an evaluable representative. A code alias (same code apart from comments and layout) shares its representative's evidence;",
        "  excluded files keep their reason. Bot names and paths are offline metadata, never runtime features of a trained model.",
        "- Lineage (`lineage:...`) is shared code (8-token-shingle Jaccard >= 0.7), not a playing style; behavior comes from `roster.distance`.",
        "",
        "## Where they come from",
        "",
        "| AI_Test directory | Files | Eligible | Contents |",
        "|---|---:|---:|---|",
    ]
    for d, rs in sorted(dirs.items(), key=lambda kv: -len(kv[1])):
        desc = ORIGINS.get(d) or ("another agent's workspace in New_AI_Test" if d.startswith("New_AI_Test/") else "")
        out.append(f"| `{d}` | {len(rs)} | {sum(r['status'] == 'eligible' for r in rs)} | {desc} |")
    out += ["", "A file's *origin* below is the AI_Test path of its representative; *also at* lists byte-identical copies elsewhere.", "",
            f"## The {len(rows)} files", "",
            "| # | Name | Status | Lineage | Origin (AI_Test) | Also at | File |", "|---:|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        others = [x for x in r["paths"] if x != r["main"]["path"]]
        out.append(f"| {i} | {r['name']} | {r['status']} | `{r['lineage']}` | `{r['main']['path']}` | "
                   f"{'<br>'.join(f'`{x}`' for x in others) or ''} | [`{r['sha'][:12]}`]({r['file']}) |")
    return "\n".join(out) + "\n"


def write_readme(out: Path = ARCHIVE) -> Path:
    path = out / "README.md"
    path.write_text(readme(load_json(out / "manifest.json")), encoding="utf-8")
    return path


def pool(manifest: dict, every: bool = False) -> dict:
    """A training.collect pool of archived bots (@repo paths, source hashes enforced by the collector)."""
    bots, taken = [], set()
    for r in files(manifest):
        if r["status"] != "eligible" and not every:
            continue
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", r["name"].replace("/", "-"))
        name, k = base, 2
        while name in taken:
            name, k = f"{base}_{k}", k + 1
        taken.add(name)
        bots.append({"name": name, "path": f"@repo/roster/bots/{r['file']}", "family": r["lineage"], "weight": 1,
                     "source_sha256": r["sha"], "origin": r["main"]["path"], "status": r["status"]})
    return {"_comment": "Generated by python -m roster.archive pool from roster/bots/manifest.json. Add your own bot entries; "
                        "paths are relative to AI_Test, @repo/ means inside ctf-fusion.", "bots": bots}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--inventory", default=str(REPO / "data" / "roster" / "bot_inventory.json"))
    s.add_argument("--ai-test", help="AI_Test checkout (default: the inventory's root, else AI_TEST_ROOT)")
    s.add_argument("--out", default=str(ARCHIVE))
    c = sub.add_parser("check")
    c.add_argument("--pool", action="append", default=[], help="pool / registry / frozen roster whose sources must be archived")
    c.add_argument("--ai-test", help="also report AI_Test files that changed since archiving")
    c.add_argument("--out", default=str(ARCHIVE))
    pl = sub.add_parser("pool")
    pl.add_argument("--all", action="store_true", help="every stored file, not only eligible representatives")
    pl.add_argument("--archive", default=str(ARCHIVE))
    pl.add_argument("--out", default=str(REPO / "profiles" / "pool_archive.json"))
    rd = sub.add_parser("readme")
    rd.add_argument("--archive", default=str(ARCHIVE))
    a = ap.parse_args(argv)
    if a.cmd == "pool":
        p = pool(load_json(Path(a.archive) / "manifest.json"), a.all)
        save_json(a.out, p)
        print(f"[archive] pool of {len(p['bots'])} archived bots -> {a.out}")
        return
    if a.cmd == "readme":
        print(f"[archive] -> {write_readme(Path(a.archive))}")
        return
    if a.cmd == "save":
        inv = load_json(a.inventory)
        root = Path(a.ai_test) if a.ai_test else Path(inv["ai_test_root"]) if Path(inv["ai_test_root"]).is_dir() else ai_test_root(None)
        r = save(inv, root.resolve(), Path(a.out))
        print(f"[archive] {r['copied']} new files; {r['candidates']} candidates, {r['stored_files']} stored files, "
              f"{r['stored_bytes'] / 2 ** 20:.1f} MiB -> {a.out}")
        return
    root = ai_test_root(a.ai_test).resolve() if (a.ai_test or os.environ.get("AI_TEST_ROOT")) else None
    r = check(Path(a.out), a.pool, root)
    print(f"[archive] {r['entries']} entries, {r['stored_files']} stored files, {len(r['errors'])} errors")
    for e in r["errors"]:
        print(f"[archive] error: {e}")
    for e in r["untraced"]:
        print(f"[archive] no source_sha256 (not hash-locked): {e}")
    if root is not None:
        print(f"[archive] AI_Test: {len(r['changed_in_ai_test'])} changed, {len(r['missing_in_ai_test'])} missing since archiving")
        for bid in r["changed_in_ai_test"] + r["missing_in_ai_test"]:
            print(f"[archive]   {bid}")
    sys.exit(1 if r["errors"] else 0)


if __name__ == "__main__":
    main()
