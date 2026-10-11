"""Package the P-only bots as the single files a judge compiles, with their SHA-256 hashes (the CD half of CI/CD).

    python tools/repo/release_bots.py --out dist                      # every variant of bots/p_only/profiles/variants.json
    python tools/repo/release_bots.py --out dist --only c08_ponly_xxl v12_ponly_xl
    python tools/repo/release_bots.py --out dist --check              # also compile each bundle with -fsyntax-only
    python tools/repo/release_bots.py --out dist --no-gzip            # write plain .cpp files

Writes, into --out:
    <variant>.cpp.gz   the bundled bot (bots/p_only/tools/bundle.py: every local include inlined, model weights included),
                       gzip with a zero timestamp so the same bundle always gives the same file
    MANIFEST.json      commit, per variant the SHA-256 of the *uncompressed* bundle, its size and what it is
    SHA256SUMS         `sha256sum -c` format, over the files written here

The bundle is built exactly as bots/p_only/tools/build.py builds it, except that no `#define PONLY_STATS` is prepended:
a release bundle is the deployed form, not the instrumented form that screens use. Its hash therefore differs from a
screen manifest's `bundle_sha256` by exactly that line (screens: sha256 of "#define PONLY_STATS 1\\n" + bundle).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from build import load_variants  # noqa: E402
from bundle import bundle, sha256_text  # noqa: E402
from fusion.runner import find_compiler  # noqa: E402

VARIANTS = REPO / "bots" / "p_only" / "profiles" / "variants.json"


def git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def gzip_bytes(data: bytes) -> bytes:
    """Deterministic gzip: no file name, mtime 0, fixed level."""
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as f:
        f.write(data)
    return buf.getvalue()


def syntax_check(cxx: str, path: Path) -> str | None:
    """None when the bundle compiles (syntax and semantic analysis, no code generation), else the compiler's output."""
    r = subprocess.run([cxx, "-std=c++20", "-fsyntax-only", str(path)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1200)
    return None if r.returncode == 0 else (r.stderr or r.stdout)[-2000:]


def package(variants: list[dict], out: Path, compress: bool = True, check: bool = False, jobs: int | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "note": "sha256 is of the uncompressed bundle; screens hash '#define PONLY_STATS 1\\n' + bundle",
                "variants": {}}
    plain: dict[str, Path] = {}
    for v in variants:
        text, files = bundle(REPO / v["source"])
        data = text.encode("utf-8")
        entry = {"sha256": sha256_text(text), "bytes": len(data), "source": v["source"],
                 **{k: v.get(k) for k in ("carrier", "role", "tier", "value_tier", "fixed_samples") if k in v},
                 "inlined_files": len(files)}
        if compress:
            gz = gzip_bytes(data)
            name = f"{v['name']}.cpp.gz"
            (out / name).write_bytes(gz)
            entry.update({"file": name, "gzip_bytes": len(gz)})
        else:
            name = f"{v['name']}.cpp"
            (out / name).write_bytes(data)
            entry["file"] = name
        manifest["variants"][v["name"]] = entry
        plain[v["name"]] = out / f"{v['name']}.cpp"
        print(f"[bundle] {v['name']:28s} {len(data) / 1e6:7.2f} MB  {entry['sha256'][:12]}", flush=True)

    if check:
        cxx = find_compiler()
        if not cxx:
            raise SystemExit("no C++ compiler found for --check (set CXX)")
        scratch = out / ".check"
        scratch.mkdir(exist_ok=True)
        failures = {}

        def one(name: str):
            src = scratch / f"{name}.cpp"
            text, _ = bundle(REPO / next(v["source"] for v in variants if v["name"] == name))
            src.write_text(text, encoding="utf-8")
            try:
                return name, syntax_check(cxx, src)
            finally:
                src.unlink(missing_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs or min(4, os.cpu_count() or 1)) as pool:
            for name, err in pool.map(one, [v["name"] for v in variants]):
                manifest["variants"][name]["syntax_check"] = "ok" if err is None else "failed"
                print(f"[check]  {name:28s} {'ok' if err is None else 'FAILED'}", flush=True)
                if err:
                    failures[name] = err
        scratch.rmdir()
        if failures:
            for name, err in failures.items():
                print(f"\n--- {name} ---\n{err}", file=sys.stderr)
            raise SystemExit(f"{len(failures)} bundle(s) do not compile: {', '.join(failures)}")

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    sums = [f"{hashlib.sha256((out / e['file']).read_bytes()).hexdigest()}  {e['file']}" for e in manifest["variants"].values()]
    sums.append(f"{hashlib.sha256((out / 'MANIFEST.json').read_bytes()).hexdigest()}  MANIFEST.json")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=str(VARIANTS), help="variant list (default: bots/p_only/profiles/variants.json)")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="package only these variants")
    ap.add_argument("--check", action="store_true", help="compile every bundle with -fsyntax-only")
    ap.add_argument("--no-gzip", action="store_true")
    ap.add_argument("--jobs", type=int, help="parallel compiler runs for --check (default: min(4, cpus))")
    a = ap.parse_args(argv)
    variants = load_variants(Path(a.variants))
    if a.only:
        unknown = set(a.only) - {v["name"] for v in variants}
        if unknown:
            raise SystemExit(f"unknown variants: {sorted(unknown)}")
        variants = [v for v in variants if v["name"] in set(a.only)]
    manifest = package(variants, Path(a.out), compress=not a.no_gzip, check=a.check, jobs=a.jobs)
    total = sum(e.get("gzip_bytes", e["bytes"]) for e in manifest["variants"].values())
    print(f"[release] {len(manifest['variants'])} bundles, {total / 1e6:.1f} MB at {a.out} (commit {manifest['commit'][:10]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
