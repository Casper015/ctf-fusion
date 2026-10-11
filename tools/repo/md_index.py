"""docs/MARKDOWN_INDEX.md: every tracked Markdown file, grouped by folder, with its title and the docs/DIGEST.md
section that summarises it.

    python tools/repo/md_index.py            # print the index
    python tools/repo/md_index.py --write    # write docs/MARKDOWN_INDEX.md
    python tools/repo/md_index.py --check    # exit 1 when the committed index is stale; list files DIGEST.md misses

It reads `git ls-files`, so `git add` a new file before you regenerate. The index is a pure function of the tracked
file names, their titles and DIGEST.md's source lines (no sizes, dates or commit hashes), so it only changes when a
Markdown file is added, moved, removed or retitled. Each DIGEST.md section opens with a `**Sources:**` line linking
the files and folders it summarises: a file maps to the section that links it exactly, else to the one whose folder
link is the longest prefix of its path.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
INDEX = "docs/MARKDOWN_INDEX.md"
DIGEST = "docs/DIGEST.md"
SOURCES = re.compile(r"^\*\*Sources:\*\*")
LINK = re.compile(r"\]\(([^)\s]+)\)")

# (path prefix, heading, what is there). The first matching prefix wins, so a subfolder comes before its parent.
# The index lists the root first, then the groups by heading, so a subfolder follows its parent.
GROUPS = [
    ("docs/benchmark/", "docs/benchmark", "Analysis of the AI_Test heuristic benchmark and of elite duels."),
    ("docs/handoffs/", "docs/handoffs", "Briefs and prompts from before the task board: an archive, superseded by tasks."),
    ("docs/", "docs", "Design notes, plans, decisions, and the two overviews of all the others (DIGEST, this index)."),
    ("training/", "training", "Contracts of the training pipeline: actions, data, features, models, simulator."),
    ("inference/", "inference", "The C++ that ships inside a bot, and its exported models."),
    ("bots/p_only/report/", "bots/p_only/report", "P-only screens and their analyses: records of runs."),
    ("bots/", "bots", "Carriers with the mixture integration, and the P-only bots with their experiment notes."),
    ("roster/", "roster", "The opponent roster: inventory, fingerprints, strength screens, training selection."),
    ("reports/eval/", "reports/eval", "Match results: screens, tournament, ablations."),
    ("reports/host/", "reports/host", "Measurements on the competition host: environment, capacity, training speed."),
    ("reports/models/", "reports/models", "Offline gate reports per size tier, and the controlled model studies."),
    ("reports/reviews/", "reports/reviews", "Written reviews and audits, each pinned to a commit."),
    ("reports/", "reports", "Results committed as Markdown, JSON or JSONL."),
    ("tasks/", "tasks", "The task board: one file per task, and its generated summary."),
    ("tools/", "tools", "Scripts that are not the importable pipeline."),
]
ROOT = ("", "Repository root", "Entry points: what the project is and how to work in it.")


def tracked(repo: Path = REPO) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--", "*.md"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(f for f in out.split("\0") if f)


def group_of(path: str) -> tuple[str, str, str]:
    if "/" not in path:
        return ROOT
    for prefix, heading, note in GROUPS:
        if path.startswith(prefix):
            return prefix, heading, note
    top = path.split("/", 1)[0] + "/"
    return top, top.rstrip("/"), ""


def title_of(text: str, path: str) -> str:
    """A task's front-matter title, else the first '# ' heading, else the first line of text (an '@file' import line
    becomes 'Imports file'), else the file name."""
    if text.startswith("---\n") and "\n---" in text[3:]:
        end = text.index("\n---", 3)
        m = re.search(r"^title:[ \t]*(.*?)[ \t]*$", text[4:end], re.M)
        if m and m.group(1):
            value = m.group(1)
            if value.startswith('"'):
                try:
                    return json.loads(value)            # task.py writes titles as JSON strings
                except ValueError:
                    return value.strip('"')
            return value
        text = text[end + 4:]
    first = ""
    for line in text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
        if not first and line.strip():
            first = line.strip()
    if first.startswith("@") and " " not in first:
        return f"Imports {first[1:]}"
    return first or PurePosixPath(path).name


def anchor(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation other than '-' and '_' removed, each space a hyphen."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def digest_sources(text: str, base: str = "docs") -> list[tuple[str, str]]:
    """(repository path or folder prefix ending in '/', section heading) for each link on a section's Sources line."""
    out, heading = [], ""
    for line in text.split("\n"):
        if line.startswith("## "):
            heading = line[3:].strip()
        elif heading and SOURCES.match(line):
            for m in LINK.finditer(line):
                target = m.group(1).split("#")[0]
                if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target):
                    continue
                path = PurePosixPath(base, target)
                parts: list[str] = []
                for p in path.parts:
                    if p == "..":
                        if parts:
                            parts.pop()
                    elif p != ".":
                        parts.append(p)
                out.append(("/".join(parts) + ("/" if target.endswith("/") else ""), heading))
    return out


def section_for(path: str, sources: list[tuple[str, str]]) -> str | None:
    exact = [h for p, h in sources if p == path]
    if exact:
        return exact[0]
    folders = sorted(((p, h) for p, h in sources if p.endswith("/") and path.startswith(p)), key=lambda x: -len(x[0]))
    return folders[0][1] if folders else None


def render(files: list[str], titles: dict[str, str], digest: str) -> str:
    sources = digest_sources(digest)
    cell = lambda s: s.replace("|", "\\|")                                            # noqa: E731
    groups: dict[tuple[str, str, str], list[str]] = {}
    for f in files:
        groups.setdefault(group_of(f), []).append(f)
    ordered = sorted(groups, key=lambda g: (g != ROOT, g[1]))
    missing = [f for f in files if section_for(f, sources) is None]
    out = ["# Markdown index", "",
           f"All {len(files)} tracked Markdown files, by folder. Generated by `tools/repo/md_index.py` from `git ls-files`; "
           "do not edit it by hand. Run `python tools/repo/md_index.py --write` after adding, moving, removing or "
           "retitling a Markdown file. **Digest** is the section of [DIGEST.md](DIGEST.md) that summarises the file "
           "together with the others on its topic.", ""]
    if missing:
        out.extend([f"{len(missing)} file(s) are not summarised in DIGEST.md yet (Digest column: *not yet*).", ""])
    out.extend(["| Folder | Files | What is there |", "|---|---:|---|"])
    for g in ordered:
        out.append(f"| [{g[1]}](#{anchor(g[1])}) | {len(groups[g])} | {g[2]} |")
    out.append("")
    for g in ordered:
        out.extend([f"## {g[1]}", "", "| File | Title | Digest |", "|---|---|---|"])
        for f in groups[g]:
            section = section_for(f, sources)
            where = f"[{section}](DIGEST.md#{anchor(section)})" if section else "*not yet*"
            out.append(f"| [{f}](../{f}) | {cell(titles.get(f, ''))} | {where} |")
        out.append("")
    return "\n".join(out)


def build(repo: Path = REPO) -> tuple[str, list[str], list[str]]:
    """The index for the files tracked in `repo` (working-copy content), those files, and the ones DIGEST.md misses.
    The index always lists itself, so writing it for the first time does not make it stale."""
    files = sorted(set(tracked(repo)) | {INDEX})
    titles = {f: title_of((repo / f).read_text(encoding="utf-8", errors="replace"), f)
              for f in files if (repo / f).is_file()}
    digest = (repo / DIGEST).read_text(encoding="utf-8") if (repo / DIGEST).is_file() else ""
    titles.setdefault(INDEX, "Markdown index")
    sources = digest_sources(digest)
    return render(files, titles, digest), files, [f for f in files if section_for(f, sources) is None]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help=f"write {INDEX}")
    mode.add_argument("--check", action="store_true", help=f"exit 1 when {INDEX} differs from a fresh render")
    args = ap.parse_args(argv)
    text, files, missing = build()
    for f in missing:
        print(f"md_index: not summarised in {DIGEST}: {f}", file=sys.stderr)
    if args.write:
        (REPO / INDEX).write_text(text, encoding="utf-8", newline="\n")
        print(f"md_index: wrote {INDEX} ({len(files)} files)", file=sys.stderr)
        return 0
    if args.check:
        current = (REPO / INDEX).read_text(encoding="utf-8") if (REPO / INDEX).is_file() else ""
        if current.strip() != text.strip():
            print(f"md_index: {INDEX} is stale: run python tools/repo/md_index.py --write", file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
