"""Archive every inventoried bot source in this repository, so later checks never depend on AI_Test's working tree.

    python -m roster.archive save  --inventory data/roster/bot_inventory.json      # writes roster/bots/
    python -m roster.archive check [--pool profiles/open_training_pool.json ...] [--ai-test AI_TEST]

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
    a = ap.parse_args(argv)
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
