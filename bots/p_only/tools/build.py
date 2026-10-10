"""Bundle and compile the bots of bots/p_only/profiles/variants.json, with a manifest of exact identities.

    python bots/p_only/tools/build.py                       # every variant
    python bots/p_only/tools/build.py c08_ponly_m v12_ponly_m
    python bots/p_only/tools/build.py --stats ...           # -DPONLY_STATS=1: network call counters on stderr (screens)
    python bots/p_only/tools/build.py --variants bots/p_only/profiles/variants_families.json v12_ponly_pxxl_vpD_v

For each variant: bots/p_only/tools/bundle.py inlines its local includes into one
.cpp (build/p_only/src/<name>.cpp, the file a judge would compile), which is
compiled with the repository's flags (fusion/runner.py: -std=c++20 -O2, -static
on Windows) into build/p_only/bin/<name>-<bundle sha>.exe. An executable whose
bundle is unchanged is reused. build/p_only/manifest.json records, per variant,
the bundle's and the binary's SHA-256, every inlined file with its SHA-256, the
compiler and the flags; screens identify candidates by these hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from bundle import bundle, sha256_text  # noqa: E402
from fusion.runner import EXE, WINDOWS, find_compiler  # noqa: E402

VARIANTS = REPO / "bots" / "p_only" / "profiles" / "variants.json"
OUT = REPO / "build" / "p_only"
FLAGS = ["-std=c++20", "-O2"] + (["-static"] if WINDOWS else [])


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_variants(path: Path = VARIANTS) -> list[dict]:
    """A variant list; its `include` lists (paths from the repository root) come first, names must be unique."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = [v for inc in doc.get("include", []) for v in load_variants(REPO / inc)]
    for v in doc["variants"]:
        if v["name"] in {x["name"] for x in out}:
            raise SystemExit(f"{path}: variant {v['name']} is defined twice")
        out.append(v)
    return out


def compiler_version(cxx: str) -> str:
    r = subprocess.run([cxx, "--version"], capture_output=True, text=True)
    return (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else cxx


def build_one(v: dict, cxx: str, out: Path, stats: bool) -> dict:
    text, files = bundle(REPO / v["source"])
    if stats:
        text = "#define PONLY_STATS 1\n" + text
    digest = sha256_text(text)
    src = out / "src" / f"{v['name']}.cpp"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(text, encoding="utf-8")
    exe = out / "bin" / f"{v['name']}-{digest[:12]}{EXE}"
    exe.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not exe.is_file():
        tmp = exe.with_name(exe.stem + f".tmp{os.getpid()}{EXE}")
        r = subprocess.run([cxx, *FLAGS, str(src), "-o", str(tmp)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900)
        if r.returncode != 0:
            raise SystemExit(f"compile failed for {v['name']}:\n{(r.stderr or r.stdout)[-3000:]}")
        os.replace(tmp, exe)
    rel = lambda p: str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)  # noqa: E731
    return {**v, "stats": stats, "bundle": rel(src), "bundle_sha256": digest, "bundle_bytes": len(text.encode("utf-8")),
            "binary": rel(exe), "binary_sha256": sha256_file(exe), "compile_seconds": round(time.time() - t0, 1),
            "inputs": [{"path": rel(f), "sha256": sha256_file(f)} for f in files]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="variants to build (default: all)")
    ap.add_argument("--stats", action="store_true", help="define PONLY_STATS=1 (P-only bots print call counters)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variants", default=str(VARIANTS), help="variant list (default: bots/p_only/profiles/variants.json)")
    a = ap.parse_args(argv)
    cxx = find_compiler()
    if not cxx:
        raise SystemExit("no C++ compiler found (install g++/clang++ or w64devkit, or set CXX)")
    variants = load_variants(Path(a.variants))
    unknown = set(a.names) - {v["name"] for v in variants}
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")
    out = Path(a.out)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"variants": {}}
    for v in variants:
        if a.names and v["name"] not in a.names:
            continue
        print(f"[build] {v['name']} <- {v['source']}", flush=True)
        manifest["variants"][v["name"]] = build_one(v, cxx, out, a.stats)
    manifest.update({"compiler": compiler_version(cxx), "flags": FLAGS, "host": platform.node(),
                     "platform": platform.platform(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for name, m in manifest["variants"].items():
        if not a.names or name in a.names:
            print(f"  {name:16s} bundle {m['bundle_sha256'][:12]}  binary {m['binary_sha256'][:12]}  {m['binary']}")
    print(f"[build] manifest: {path}")


if __name__ == "__main__":
    main()
