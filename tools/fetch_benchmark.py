"""Download the raw benchmark match records (ctf-benchmark-v3, ctf-elite-duels-v3) from AI_Test into data/benchmark_raw/.

    python tools/fetch_benchmark.py [--ref main]

Uses the GitHub CLI (`gh`, already authenticated for the private repository).
Writes each file under data/benchmark_raw/<path in AI_Test> and a
MANIFEST.json with the source commit, sizes and SHA-256 hashes, so analyses
can name exactly which data they read. data/ is not committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "benchmark_raw"
SOURCE = "Casper015/AI_Test"
T1 = "ctf-benchmark-v3/track1_10k_suite/"
T2 = "ctf-benchmark-v3/track2_2k_c_series/"
FILES = [
    T1 + "results/games.jsonl", T1 + "results/games.jsonl.gz", T1 + "results/final_90k_analysis.json",
    T1 + "results/report_90k_final.md", T1 + "results/progress.json", T1 + "BUILD_MANIFEST.json",
    T2 + "results/games_c_2000.jsonl", T2 + "results/games_c_2000.jsonl.gz", T2 + "results/report_c_2000.md",
    T2 + "results/progress_c_2000.json", T2 + "MANIFEST_C.json", "ctf-benchmark-v3/BOTS_MAPPING.md",
    "ctf-elite-duels-v3/README.md", "ctf-elite-duels-v3/fixtures_duels_2000.json", "ctf-elite-duels-v3/run_elite_duels.py",
    "ctf-elite-duels-v3/results/games_elite_duels.jsonl", "ctf-elite-duels-v3/results/games_elite_duels.jsonl.gz",
    "ctf-elite-duels-v3/results/progress_elite_duels.json", "ctf-elite-duels-v3/results/report_elite_duels.md",
]


def gh(*args) -> bytes:
    return subprocess.run(["gh", "api", *args], capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="main")
    a = ap.parse_args()
    commit = json.loads(gh(f"repos/{SOURCE}/commits/{a.ref}"))["sha"]
    manifest = {"source": SOURCE, "ref": a.ref, "commit": commit, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {}}
    for path in FILES:
        data = gh("-H", "Accept: application/vnd.github.raw", f"repos/{SOURCE}/contents/{path}?ref={commit}")
        dest = OUT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest["files"][path] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        print(f"{len(data):>10}  {path}")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"commit {commit} -> {OUT}")


if __name__ == "__main__":
    main()
