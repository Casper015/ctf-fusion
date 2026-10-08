"""Inventory C++ bot source candidates; no compilation or strength claims."""
import argparse
import hashlib
import json
from pathlib import Path
import re


DEFAULT_DIRS = [
    "ctf-arena/Player7/judge/ctf/bots", "ctf-arena/Player7/workspace/experiments/round3/export",
    "ctf-stress-pack/bots", "ctf-stress-pack/bots_v2", "ctf-stress-pack/bots_v3",
    "ctf-stress-pack/bots_c_pending", "ctf-stress-pack/bots_10k_suite",
    "ctf-benchmark-v3/track1_10k_suite/bots", "ctf-benchmark-v3/track2_2k_c_series/bots",
    "ctf-s1-controls/exports", "New_AI_Test/Player0/workspace/bots",
]


def inventory(root, directories):
    root = Path(root).resolve(strict=True)
    paths, missing = set(), []
    for name in directories:
        directory = (root / name).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("scan directories must remain inside the declared repository")
        if not directory.is_dir():
            missing.append(name)
            continue
        for path in directory.rglob("*.cpp"):
            if path.resolve().is_relative_to(root):
                paths.add(path)
    first_hash, bots = {}, []
    for path in sorted(paths):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        relative = path.relative_to(root).as_posix()
        text = content.decode("utf-8", errors="replace")
        alias = first_hash.get(digest)
        first_hash.setdefault(digest, relative)
        bots.append({"id": relative, "path": relative, "source_sha256": digest,
                     "source_bytes": len(content), "byte_identical_source_of": alias,
                     "entrypoint_text_detected": bool(re.search(r"\b(?:int|signed)\s+main\s*\(", text)),
                     "assessment_status": "unmeasured", "lineage_status": "needs_source_and_behavior_review"})
    return {"schema": "ctf-bot-inventory-1", "scan_dirs": directories, "missing_dirs": missing,
            "bots": bots, "source_candidates": len(bots), "distinct_source_hashes": len(first_hash),
            "limits": "Text entrypoint detection and source aliases are provisional; inspect dependencies/build options and evaluate behavior."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scan-dir", action="append", default=[], help="additional repository-relative candidate directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = inventory(a.root, DEFAULT_DIRS + a.scan_dir)
    Path(a.out).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("source_candidates", "distinct_source_hashes", "missing_dirs")}))


if __name__ == "__main__":
    main()
