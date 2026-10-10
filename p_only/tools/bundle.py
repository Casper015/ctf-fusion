"""Bundle a bot into one self-contained .cpp: every local #include "..." is inlined once.

    python p_only/tools/bundle.py p_only/bots/variants/c08_ponly_m.cpp -o build/p_only/src/c08_ponly_m.cpp

A judge compiles a single source file, and the bundle is exactly what is compiled
and tested, so its SHA-256 is the bot's identity. Local includes are searched in
the including file's directory, then p_only/include, p_only/bots and inference/.
System includes (<...>) stay as they are.

Conditional includes (the model-size selection) are followed the way the compiler
will take them: the bundler tracks #define/#undef and evaluates #if, #ifdef,
#ifndef, #elif and #else with defined(), integer macros, !, && and ||. The
directives themselves are kept, so the compiler sees the same structure; an
include in a branch the compiler skips is not inlined. An include under a
condition the bundler cannot evaluate is an error, never a guess.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEARCH = [REPO / "p_only" / "include", REPO / "p_only" / "bots", REPO / "inference"]

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$")


class Unknown(Exception):
    pass


def compiler_macro(name: str) -> bool:
    """Macros the compiler predefines (__AVX2__, _WIN32, _M_X64, ...): the bundler cannot know them."""
    return name.startswith("__") or name.startswith("_M_") or name.startswith("_WIN")


def evaluate(expr: str, macros: dict[str, str]) -> bool:
    """A preprocessor condition with defined(), integer macros, !, &&, || and parentheses."""
    e = re.sub(r"//.*$|/\*.*?\*/", "", expr).strip()

    def defined(m):
        name = m.group(1) or m.group(2)
        if compiler_macro(name) and name not in macros:
            raise Unknown(f"compiler macro {name}")
        return "1" if name in macros else "0"
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)", defined, e)

    def ident(m):
        name = m.group(0)
        if compiler_macro(name) and name not in macros:
            raise Unknown(f"compiler macro {name}")
        value = macros.get(name, "0").strip() or "1"
        if not re.fullmatch(r"-?\d+", value):
            raise Unknown(f"macro {name} = {value!r} is not an integer")
        return value
    e = re.sub(r"\b[A-Za-z_]\w*\b", ident, e)
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    if set(re.findall(r"[A-Za-z_]+", e)) - {"and", "or", "not"} or re.search(r"[^\w\s()<>=!+\-*/%]", e):
        raise Unknown(f"cannot evaluate #if {expr.strip()!r}")
    try:
        return bool(eval(e, {"__builtins__": {}}, {}))  # noqa: S307 - integers, comparisons and boolean words only
    except Exception as exc:  # noqa: BLE001
        raise Unknown(f"cannot evaluate #if {expr.strip()!r}: {exc}") from exc


def find(name: str, here: Path) -> Path | None:
    for d in [here, *SEARCH]:
        p = d / name
        if p.is_file():
            return p.resolve()
    return None


def bundle(source: Path) -> tuple[str, list[Path]]:
    """The bundled text and the files it contains (the source first)."""
    macros: dict[str, str] = {}
    seen: list[Path] = []
    out: list[str] = []
    # Each frame: [this branch active, some branch already taken, enclosing active, evaluable]
    stack: list[list[bool]] = []

    def active() -> bool:
        return all(f[0] for f in stack)

    def known() -> bool:
        return all(f[3] for f in stack)

    def emit(path: Path) -> None:
        seen.append(path)
        for line in path.read_text(encoding="utf-8").split("\n"):
            m = INCLUDE.match(line)
            if m:
                if not active():
                    out.append(f"// [bundle] skipped (inactive branch): {line.strip()}")
                    continue
                if not known():
                    raise SystemExit(f"{path}: include under a condition the bundler cannot evaluate: {line.strip()}")
                target = find(m.group(1), path.parent)
                if target is None:
                    raise SystemExit(f"{path}: include not found: {m.group(1)}")
                if target in seen:
                    out.append(f"// [bundle] already inlined: {line.strip()}")
                    continue
                rel = target.relative_to(REPO) if target.is_relative_to(REPO) else target
                out.append(f"// [bundle] begin {rel}")
                emit(target)
                out.append(f"// [bundle] end {rel}")
                continue
            d = DIRECTIVE.match(line)
            if d:
                kind, rest = d.group(1), d.group(2).strip()
                if kind in ("if", "ifdef", "ifndef"):
                    outer = active()
                    try:
                        if kind == "if":
                            cond = evaluate(rest, macros)
                        else:
                            name = rest.split()[0]
                            if compiler_macro(name) and name not in macros:
                                raise Unknown(name)
                            cond = (name in macros) == (kind == "ifdef")
                        stack.append([cond, cond, outer, True])
                    except Unknown:
                        stack.append([True, True, outer, False])
                elif kind == "elif":
                    f = stack[-1]
                    if f[3]:
                        try:
                            cond = not f[1] and evaluate(rest, macros)
                            f[0] = cond
                            f[1] = f[1] or cond
                        except Unknown:
                            f[0], f[3] = True, False
                elif kind == "else":
                    f = stack[-1]
                    if f[3]:
                        f[0] = not f[1]
                        f[1] = True
                elif kind == "endif":
                    stack.pop()
                elif active() and known():
                    parts = rest.split(None, 1)
                    if parts and kind == "define" and "(" not in parts[0]:
                        macros[parts[0]] = parts[1] if len(parts) > 1 else ""
                    elif parts and kind == "undef":
                        macros.pop(parts[0], None)
            out.append(line)

    emit(source.resolve())
    if stack:
        raise SystemExit(f"{source}: unbalanced #if/#endif")
    return "\n".join(out) + "\n", seen


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    text, files = bundle(Path(a.source))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"{out}: {len(files)} files, {len(text)} bytes, sha256 {sha256_text(text)[:16]}", file=sys.stderr)


if __name__ == "__main__":
    main()
