"""Pin every T0075 arm: source, model and bundle hashes (tools/screens/next_controls/MANIFEST.json).

    python tools/screens/next_controls/manifest.py            # rewrite MANIFEST.json
    python tools/screens/next_controls/manifest.py --check    # exit 1 when MANIFEST.json is not current

The arms are the two variant lists bots/agent_v2/controls/variants.json (G, I; T0077) and
bots/meowmind_p/attribution/variants.json (ON, OFF; T0078). A bundle is bots/p_only/tools/bundle.py's text, the file
bots/p_only/tools/build.py compiles (without PONLY_STATS), with its begin/end markers in POSIX form (canonical()): on
Windows bundle.py prints them with backslashes, so build.py's bundle_sha256 there differs from this one by exactly the
separators, and the preflight compares canonical forms. On other platforms the two hashes are equal. The
Meowmind section records the line diff between the ON and OFF bundles, which must be the NN_SEARCH line alone, and the
original source's sha256. The init-only arm stays "awaiting_init_export" until export_init_actor.py has written
inference/agent_v2_controls/init_actor.hpp on the host; nothing about it is filled in before that. No binary hashes:
they depend on the compiler, and build.py records them on the machine that plays.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from bundle import bundle, sha256_text  # noqa: E402

LISTS = ("bots/agent_v2/controls/variants.json", "bots/meowmind_p/attribution/variants.json")
OUT = REPO / "tools" / "screens" / "next_controls" / "MANIFEST.json"
MEOWMIND = "bots/meowmind_p/meowmind-ctf-374-20261011.cpp"
MEOWMIND_SHA256 = "cc7b730cf77c087c8c7f415b3b7d07101c3f003924d3a845a2ea3b81588794bc"   # bots/meowmind_p/README.md
RL_MANIFEST = "inference/count_specialists_rl/MANIFEST.json"
INIT_MANIFEST = "inference/agent_v2_controls/MANIFEST.json"


MARKER = re.compile(r"^// \[bundle\] (begin|end) (.*)$", re.M)


def canonical(text: str) -> str:
    """A bundle with its begin/end markers in POSIX form, whatever platform bundled it."""
    return MARKER.sub(lambda m: f"// [bundle] {m[1]} {m[2].replace(chr(92), '/')}", text)


def canonical_sha256(text: str) -> str:
    return sha256_text(canonical(text))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO).as_posix()


def model_of(v: dict) -> dict | None:
    """The actor an agent_v2 control plays, from its producer's manifest (None for Meowmind)."""
    actor = v.get("actor")
    if not actor:
        return None
    header = REPO / actor["header"]
    if actor["namespace"] == "fusion::models::rl_general":
        m = json.loads((REPO / RL_MANIFEST).read_text(encoding="utf-8"))["models"]["general"]
        return {"name": m["name"], "namespace": actor["namespace"], "header": actor["header"],
                "header_sha256": sha256_file(header), "npz": m["npz_path"], "npz_sha256": sha256_file(REPO / m["npz_path"]),
                "digest": m["digest"], "producer": RL_MANIFEST, "producer_sha256": sha256_file(REPO / RL_MANIFEST)}
    if not header.is_file():
        return None
    m = json.loads((REPO / INIT_MANIFEST).read_text(encoding="utf-8"))
    return {"name": m["model"]["name"], "namespace": actor["namespace"], "header": actor["header"],
            "header_sha256": sha256_file(header), "npz": m["model"]["npz_path"],
            "npz_sha256": sha256_file(REPO / m["model"]["npz_path"]), "digest": m["model"]["digest"],
            "source": m["source"], "producer": INIT_MANIFEST, "producer_sha256": sha256_file(REPO / INIT_MANIFEST)}


def arm(v: dict, listed_in: str) -> tuple[dict, str | None]:
    src = REPO / v["source"]
    entry = {"list": listed_in, "role": v["role"], "telemetry": v["telemetry"], "source": v["source"],
             "source_sha256": sha256_file(src)}
    if v.get("actor"):
        entry["model"] = model_of(v)
        if entry["model"] is None:
            return {**entry, "status": "awaiting_init_export"}, None
    text, files = bundle(src)
    text = canonical(text)
    entry.update({"status": "ready", "bundle_sha256": sha256_text(text), "bundle_bytes": len(text.encode("utf-8")),
                  "inlined": [{"path": rel(f), "sha256": sha256_file(f)} for f in files]})
    return entry, text


def build() -> dict:
    arms, texts = {}, {}
    for listed_in in LISTS:
        for v in json.loads((REPO / listed_in).read_text(encoding="utf-8"))["variants"]:
            arms[v["name"]], texts[v["name"]] = arm(v, listed_in)
    on, off = texts["meowmind_search_on"].splitlines(), texts["meowmind_search_off"].splitlines()
    diff = [d for d in difflib.unified_diff(on, off, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    return {
        "schema": "ctf-next-controls-manifest-1",
        "task": "T0075",
        "generator": "tools/screens/next_controls/manifest.py",
        "lists": {p: sha256_file(REPO / p) for p in LISTS},
        "arms": arms,
        "meowmind": {
            "original": MEOWMIND, "original_sha256": sha256_file(REPO / MEOWMIND),
            "original_unmodified": sha256_file(REPO / MEOWMIND) == MEOWMIND_SHA256,
            "on_off_bundle_diff": diff,
            "differ_only_in_nn_search": diff == ["-#define NN_SEARCH 1", "+#define NN_SEARCH 0"],
        },
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed MANIFEST.json, write nothing")
    a = ap.parse_args(argv)
    text = render(build())
    if a.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != text:
            print(f"{rel(OUT)} is not current: run python tools/screens/next_controls/manifest.py", file=sys.stderr)
            return 1
        return 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"[next-controls manifest] {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
