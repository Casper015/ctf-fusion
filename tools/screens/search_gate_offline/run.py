"""Fixed-tree receipts for T0076; no games, deadlines, timing or strength measurements."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from fusion import runner  # noqa: E402

GAMES = ("n02", "n04", "n15")
FIXTURES = REPO / "bots/p_only/tests/fixtures"
OUT = REPO / "bots/agent_v2/search_gate/diagnostics"
ACTORS = "inference/meow_rl/aligned/p2_large"
SOURCE = "tools/cpp/agent_v2/check.cpp"
FLAGS = ["-std=c++20", "-O2", "-w", "-I", "inference", "-I", "tools/cpp/meow_rl",
         "-DFUSION_TIER_PXXL_VM=1", '-DACTOR_HEADER="meow_rl/aligned/p2_large/actor.hpp"',
         '-DCRITIC_HEADER="meow_rl/aligned/p2_large/critic_fp32.hpp"', "-DAGENT_V2_CHECK_F64=1"]


def splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


SALTS = [splitmix64(20262120 + r) for r in range(4)]
REF = splitmix64(20262120 + 1000)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command: list[str], kind: str, output: Path | None = None) -> dict:
    """Keep the literal output, including numeric lexemes used by the parity checks."""
    record = {"kind": kind, "command": command, "cwd": "."}
    try:
        if output is None:
            p = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=3600)
            stdout = p.stdout
        else:
            with output.open("w", encoding="utf-8", newline="\n") as f:
                p = subprocess.run(command, cwd=REPO, stdout=f, stderr=subprocess.PIPE, text=True, timeout=3600)
            stdout = output.read_text(encoding="utf-8")
            record.update(output_file=str(output), output_sha256=sha256(output))
        record.update(returncode=p.returncode, stdout_lines=stdout.splitlines(), stderr_lines=p.stderr.splitlines(),
                      passed=p.returncode == 0)
    except (OSError, subprocess.SubprocessError) as exc:
        record.update(returncode=None, stdout_lines=[], stderr_lines=[str(exc)], passed=False)
    return record


def reseat(text: str, seat: int) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("INIT "):
            words = line.split()
            words[2] = str(seat)
            lines[i] = " ".join(words) + "\n"
            return "".join(lines)
    raise ValueError("stream has no INIT")


def through_turn(text: str, last: int) -> str:
    """Retain the entire replay history through the last parity turn."""
    lines, turn = [], None
    for line in text.splitlines(keepends=True):
        lines.append(line)
        if line.startswith("TURN "):
            turn = int(line.split()[1])
        if line.startswith("END") and turn == last:
            return "".join(lines)
    raise ValueError(f"missing parity turn {last}")


def sampled_turns(text: str, stride: int) -> list[int]:
    return [int(line.split()[1]) for line in text.splitlines()
            if line.startswith("TURN ") and int(line.split()[1]) % stride == 0]


def check_rows(record: dict, turns: list[int], teams: int, seat: int) -> None:
    try:
        rows = [json.loads(line) for line in record["stdout_lines"]]
        assert [row["turn"] for row in rows] == turns, "missing/extra sampled turns"
        assert all(row["teams"] == teams and row["me"] == seat and row["roots"][0] == row["actor"]
                   for row in rows), "state/actor identity"
        assert turns, "no sampled turns"
    except (ValueError, KeyError, AssertionError, TypeError, IndexError) as exc:
        record.update(passed=False, failure=str(exc))


def parity(record: dict, diag: dict, turns: list[int], roots: int, samples: int) -> None:
    # Both integer-looking and fractional values must retain their original C++ spelling.
    try:
        native = [json.loads(line, parse_float=str, parse_int=str) for line in record["stdout_lines"]]
        reference = {str(row["turn"]): row for row in
                     (json.loads(line, parse_float=str, parse_int=str) for line in diag["stdout_lines"])}
        assert [row["turn"] for row in native] == list(map(str, turns)), "parity turn coverage"
        for row in native:
            ref = reference[row["turn"]]
            assert all(row[k] == ref[k] for k in ("actor", "teams", "me")), f"state at {row['turn']}"
            assert row["roots"] == ref["roots"][:roots], f"roots at {row['turn']}"
            a, b = row["trees"][0], ref["trees"][0]
            assert a["salt"] == b["salt"] == str(SALTS[0]), "parity salt"
            assert a["S"] == str(samples), "parity sample count"
            for field in ("v", "terminal"):
                assert a[field] == [v[:samples] for v in b[field][:roots]], f"{field} at {row['turn']}"
        record["comparison"] = f"exact roots/value strings/terminal prefixes: {len(turns)} states, {roots}x{samples}"
    except (ValueError, KeyError, AssertionError, TypeError, IndexError) as exc:
        record.update(passed=False, failure=str(exc))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--work", type=Path, default=REPO / "build/search_gate_offline")
    ap.add_argument("--smoke", action="store_true", help="seat 0, stride 100; all outputs stay in --work")
    args = ap.parse_args()
    if args.stride < 1 or args.jobs < 1:
        ap.error("--stride and --jobs must be positive")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    out = work if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    stride = 100 if args.smoke else args.stride
    games = sorted(set(args.games))
    inputs = [f"bots/p_only/tests/fixtures/stream_{game}.txt.gz" for game in games]
    inputs += [f"{ACTORS}/actor.hpp", f"{ACTORS}/critic_fp32.hpp", "inference/agent_v2/search.hpp", SOURCE]
    receipts = {"schema": 1, "smoke": args.smoke, "games": games, "stride": stride,
                "salts": [str(s) for s in SALTS], "salt_ref": str(REF), "flags": FLAGS,
                "input_sha256": {p: sha256(REPO / p) for p in inputs}, "commands": [], "passed": False}
    records = receipts["commands"]
    try:
        cxx = runner.find_compiler()
        if cxx is None:
            raise RuntimeError("no C++ compiler found (set CXX)")
        version = execute([cxx, "--version"], "compiler")
        head = execute(["git", "rev-parse", "HEAD"], "git")
        records.extend([version, head])
        receipts.update(compiler=cxx, compiler_version=version["stdout_lines"], git_head="\n".join(head["stdout_lines"]))
        binaries = {}
        for name, candidates, samples in (("diag", 8, 16), ("wide", 8, 4), ("narrow", 4, 2)):
            exe = work / (name + runner.EXE)
            flags = [*FLAGS, f"-DAGENT_V2_SEARCH_CANDIDATES={candidates}", f"-DAGENT_V2_SEARCH_SAMPLES={samples}"]
            rec = execute([cxx, *flags, SOURCE, "-o", str(exe)], f"build_{name}")
            records.append(rec)
            if not rec["passed"]:
                raise RuntimeError(f"{name} build failed: {' '.join(rec['stderr_lines'][-3:])}")
            binaries[name] = exe
        receipts["binary_sha256"] = {name: sha256(path) for name, path in binaries.items()}
        gate = execute([str(binaries["wide"]), "gate"], "gate")
        gate["passed"] &= len(gate["stdout_lines"]) == 1 and gate["stdout_lines"][0].startswith("ok gate cases ")
        records.append(gate)
        jobs, texts, originals = [], {}, {}
        for game in games:
            text = gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode("utf-8")
            originals[game] = work / f"{game}-original.txt"
            originals[game].write_text(text, encoding="utf-8", newline="\n")
            teams = int(text.split()[1])
            for seat in range(1 if args.smoke else teams):
                text_seat = reseat(text, seat)
                stream = work / f"{game}-seat{seat:02d}.txt"
                stream.write_text(text_seat, encoding="utf-8", newline="\n")
                texts[game, seat] = text_seat
                command = [str(binaries["diag"]), "gatetree", str(stream), str(stride), "8",
                           *[f"{salt}:4" for salt in SALTS], f"{REF}:16"]
                jobs.append((game, seat, teams, command, work / f"{game}-seat{seat:02d}.jsonl"))
        print(f"Replaying {len(jobs)} seat streams (stride {stride}, workers {args.jobs})", flush=True)
        def replay(job):
            game, seat, teams, command, path = job
            rec = execute(command, "diagnostic", path)
            check_rows(rec, sampled_turns(texts[game, seat], stride), teams, seat)
            return rec
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            diagnostics = list(pool.map(replay, jobs))
        records.extend(diagnostics)
        outputs = {id(rec): rec["stdout_lines"] for rec in diagnostics}   # the trees go to TREES.jsonl.gz, not the receipts
        for rec in diagnostics:
            rec["output_line_count"] = len(rec.pop("stdout_lines"))
        if not all(rec["passed"] for rec in diagnostics):
            raise RuntimeError("diagnostic replay/coverage failed")
        diag_by_seat = {(job[0], job[1]): rec for job, rec in zip(jobs, diagnostics)}
        for game in games:
            turns = sampled_turns(texts[game, 0], stride)[:10]
            stream = work / f"{game}-parity.txt"
            stream.write_text(through_turn(texts[game, 0], turns[-1]), encoding="utf-8", newline="\n")
            for name, roots, samples in (("wide", 8, 4), ("narrow", 4, 2)):
                rec = execute([str(binaries[name]), "gatetree", str(stream), str(stride), str(roots),
                               f"{SALTS[0]}:{samples}"], f"parity_{name}")
                parity(rec, {"stdout_lines": outputs[id(diag_by_seat[game, 0])]}, turns, roots, samples)
                records.append(rec)
            rec = execute([str(binaries["wide"]), "isolation", str(originals[game])], "isolation")
            rec["passed"] &= len(rec["stdout_lines"]) == 1 and rec["stdout_lines"][0].startswith("ok isolation ")
            records.append(rec)
            print(f"{game}: parity and isolation {'passed' if all(r['passed'] for r in records) else 'FAILED'}", flush=True)
        if not all(rec["passed"] for rec in records):
            raise RuntimeError("integrity stop: gate/parity/isolation or provenance command failed")
        if any(sha256(REPO / p) != digest for p, digest in receipts["input_sha256"].items()):
            raise RuntimeError("integrity stop: an input changed during the run; rebuild and replay")
        # Preserve every C++ numeric lexeme, add metadata, sort independently of worker completion order.
        lines = []
        for (game, seat, _, _, _), rec in zip(jobs, diagnostics):
            for line in outputs[id(rec)]:
                turn = json.loads(line)["turn"]
                meta = json.dumps({"game": game, "seat": seat, "split": "validation" if game == "n04" else "selection"},
                                  sort_keys=True, separators=(",", ":"))
                lines.append(((game, seat, turn), meta[:-1] + "," + line[1:]))
        path = out / "TREES.jsonl.gz"
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                for _, line in sorted(lines):
                    gz.write((line + "\n").encode("utf-8"))
        receipts.update(passed=True, states=len(lines), trees_sha256=sha256(path),
                        preregistered_states=not args.smoke and games == list(GAMES) and stride == 4 and len(lines) == 2100)
        print(f"Wrote {len(lines)} states to {path}", flush=True)
    except (RuntimeError, OSError, ValueError, KeyError, IndexError) as exc:
        receipts["failure"] = str(exc)
        print(str(exc), file=sys.stderr)
    write_json(out / "INTEGRITY.json", receipts)
    return 0 if receipts["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
