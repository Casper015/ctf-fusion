"""Own-action features: C++ (inference/own_action.hpp) against Python (fusion/decision_features.py), task T0043.

Roots are recorded matches (roster/report/results/matches) at turns chosen by observable state only: the first turn,
respawn, flag carrying, low hit points, contact (filled target slots) and units close enough to collide. Candidates
are requested replies built from each root's recorded reply: every move and action of each unit (waits, pickups,
drops, attacks on every slot, on an own unit and on unknown ids), malformed lines and duplicate effective orders.
One root is *constructed* from a recorded 15-team root by moving nine enemies next to our unit, because no recorded
root fills all eight target slots; it covers targets that are hittable but unavailable to the scorer (out of cap).

Everything must agree exactly: canonical key, flags, duplicate pointer, support, move/action/target/slot, and every
float32 of the root and candidate features. Nothing here plays a game or scores anything.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from fusion import matchlog, native
from fusion.paths import REPO
from fusion.protocol import MOVES, parse_block
from fusion.schema import layout
from training import decision_dataset as D

ROSTER = REPO / "roster" / "report" / "results" / "matches"
TEAM_COUNTS = (2, 4, 8, 15)


def tool() -> Path:
    return native.build(REPO / "tools" / "cpp" / "own_action_check.cpp", "own_action_check")


def run_tool(stream: str, seat: int, turn: int, lines: list[str]) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        path = fh.name
    try:
        r = subprocess.run([str(tool()), str(seat), str(turn), path], input=stream.encode(), capture_output=True)
    finally:
        Path(path).unlink()
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode())
    return json.loads(r.stdout)


def categories(match: dict, seat: int) -> dict[str, int]:
    """The first turn of each observable situation of our units (no outcome or later block is read)."""
    found = {"first": 1}
    size = int(match["init"].split("\n", 1)[0].split()[3])
    for t in range(2, len(match["blocks"]) - 1):
        units = parse_block(match["blocks"][t - 1]).units
        ours = [units[3 * seat + k] for k in range(3)]
        enemies = [u for u in units if u[1] != seat and u[2] >= 0]
        checks = {
            "respawn": any(u[2] < 0 for u in ours),
            "carrying": any(u[5] >= 0 for u in ours),
            "low_hp": any(0 < u[4] <= 34 for u in ours if u[2] >= 0),
            "contact": any(abs(u[2] - e[2]) + abs(u[3] - e[3]) <= 2 for u in ours if u[2] >= 0 for e in enemies),
            "collision": any(abs(a[2] - b[2]) + abs(a[3] - b[3]) == 2 for i, a in enumerate(ours) for b in ours[i + 1:]
                             if a[2] >= 0 and b[2] >= 0),
        }
        for name, hit in checks.items():
            if hit and name not in found and match["replies"][t - 1][seat]:
                found[name] = t
        if len(found) == 6:
            break
    assert size > 0
    return found


def requests(rec: np.ndarray, line: str, turn: int, seat: int, teams: int) -> list[str]:
    """The recorded reply first, then single-unit variations, malformed lines and duplicates."""
    L = layout()
    cand = L.get(rec, "cand")
    tok = F.normalize_line(line).split()
    out = [line]
    for k in range(3):
        acts = ["-", "P", "X", str(3 * seat + (k + 1) % 3), "999", "4" * 25]
        acts += [str(int(c)) for c in cand[k] if c >= 0]
        acts += ["0" + str(int(c)) for c in cand[k][:1] if c >= 0]
        acts += [str(u) for u in range(3 * teams) if u // 3 != seat][:6]      # enemies, hittable or not
        for m in MOVES:
            for a in acts:
                alt = list(tok)
                alt[1 + 2 * k], alt[2 + 2 * k] = m, a
                out.append(" ".join(alt))
    out += [" ".join(tok) + "  # comment", "\t".join(tok), " ".join([str(turn + 1)] + tok[1:]), " ".join(tok[:-1]),
            " ".join(tok[:1] + ["s"] + tok[2:]), " ".join(tok[:2] + ["Q"] + tok[3:]), "", "garbage"]
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def python_side(match: dict, stream_rec: np.ndarray, units, seat: int, turn: int, lines: list[str]) -> dict:
    L = layout()
    size = int(match["init"].split("\n", 1)[0].split()[3])
    canon = [F.canonicalize(stream_rec, L, ln, turn, seat, units, size) for ln in lines]
    flags = [c["flags"] for c in canon]
    dup = F.mark_duplicates([c["key"] for c in canon], flags, incumbent=0)
    x = [F.candidate_features(stream_rec, L, c["move"], c["act"], c["slot"]) for c in canon]
    return {"root": F.root_features(stream_rec, L), "canon": canon, "flags": flags, "dup": dup, "x": x}


class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = layout()
        cls.matches = []
        for t in TEAM_COUNTS:
            for p in sorted(ROSTER.rglob("*.json.gz")):
                m = matchlog.load(p)
                if m["teams"] == t:
                    cls.matches.append(m)
                    break

    def compare(self, stream: str, match: dict, units, seat: int, turn: int, lines: list[str], where: str) -> dict:
        rec = native.extract(stream)
        rec = rec[(rec[:, 0] == turn) & (rec[:, 1] == seat)][0]
        py = python_side(match, rec, units, seat, turn, lines)
        cc = run_tool(stream, seat, turn, lines)
        np.testing.assert_array_equal(np.asarray(cc["root"], np.float32), py["root"], err_msg=f"{where}: root features")
        self.assertEqual(len(cc["cands"]), len(lines))
        for i, (c, p) in enumerate(zip(cc["cands"], py["canon"])):
            msg = f"{where}: candidate {i} {lines[i]!r}"
            self.assertEqual(c["key"], p["key"], msg)
            self.assertEqual(c["flags"], py["flags"][i], msg)
            self.assertEqual(c["dup"], py["dup"][i], msg)
            self.assertEqual(c["support"], bool(F.supported(py["flags"][i])), msg)
            for name in ("move", "act", "target", "slot"):
                self.assertEqual(c[name], [int(v) for v in p[name]], f"{msg}: {name}")
            np.testing.assert_array_equal(np.asarray(c["x"], np.float32), py["x"][i], err_msg=msg)
        return {"flags": [c["flags"] for c in cc["cands"]]}

    def test_identity_is_pinned(self):
        got = json.loads(subprocess.run([str(tool()), "--identity"], capture_output=True, check=True).stdout)
        ident = F.feature_identity(self.L)
        self.assertEqual(got["version"], F.FEATURE_VERSION)
        self.assertEqual(got["digest"], ident["digest"], "inference/own_action.hpp kIdentityDigest is stale")
        self.assertEqual(got["layout_version"], self.L.layout_version)
        self.assertEqual((got["per_unit"], got["cand_dim"], got["root_dim"]), (F.PER_UNIT, F.CAND_DIM, F.root_dim(self.L)))
        self.assertEqual(got["slots"], self.L.blocks["cand"][1][1])

    def test_recorded_roots(self):
        seen = {f: 0 for f in F.FLAG_NAMES}
        covered = set()
        for m in self.matches:
            seat = 0
            for name, turn in categories(m, seat).items():
                covered.add(name)
                stream = D.prefix_stream(m, seat, turn)
                units = parse_block(m["blocks"][turn - 1]).units
                rec = native.extract(stream)
                rec = rec[(rec[:, 0] == turn) & (rec[:, 1] == seat)][0]
                lines = requests(rec, m["replies"][turn - 1][seat], turn, seat, m["teams"])
                res = self.compare(stream, m, units, seat, turn, lines, f"{m['match_id']} t{turn} {name}")
                for fl in res["flags"]:
                    for bit in seen:
                        seen[bit] += bool(fl & bit)
        self.assertEqual(covered, {"first", "respawn", "carrying", "low_hp", "contact", "collision"})
        for bit, name in F.FLAG_NAMES.items():
            if bit != F.TARGET_OUT_OF_CAP:            # covered by the constructed root below
                self.assertGreater(seen[bit], 0, f"no recorded candidate exercised {name}")

    def test_future_blocks_are_not_read(self):
        m = self.matches[1]
        turn = categories(m, 0)["contact"]
        lines = requests(D.root_record(m, 0, turn), m["replies"][turn - 1][0], turn, 0, m["teams"])[:40]
        prefix = run_tool(D.prefix_stream(m, 0, turn), 0, turn, lines)
        whole = run_tool(matchlog.init_for_seat(m["init"], 0) + "".join(m["blocks"]), 0, turn, lines)
        self.assertEqual(prefix, whole)

    def test_constructed_full_slots(self):
        """Nine enemies around our unit 0 (constructed from a recorded 15-team root): out-of-cap targets agree."""
        m = self.matches[-1]
        self.assertEqual(m["teams"], 15)
        seat, turn = 0, None
        for t in range(5, len(m["blocks"])):
            units = parse_block(m["blocks"][t - 1]).units
            if units[0][2] >= 0 and units[0][5] < 0 and m["replies"][t - 1][seat]:
                turn = t
                break
        size = int(m["init"].split("\n", 1)[0].split()[3])
        walls = matchlog.init_for_seat(m["init"], seat)
        grid = [ln for ln in walls.split("\n") if ln and set(ln) <= set(".#")][:size]
        block = m["blocks"][turn - 1]
        units = parse_block(block).units
        ux, uy = units[0][2], units[0][3]
        taken = {(u[2], u[3]) for u in units if u[2] >= 0}
        cells = sorted(((x, y) for y in range(size) for x in range(size)
                        if grid[y][x] == "." and (x, y) not in taken and 1 <= abs(x - ux) + abs(y - uy) <= 3),
                       key=lambda c: (abs(c[0] - ux) + abs(c[1] - uy), c))
        movers = [u for u in units if u[1] != seat and u[2] >= 0 and u[5] < 0][:10]
        self.assertGreaterEqual(min(len(cells), len(movers)), 9)
        lines = block.split("\n")
        for u, (x, y) in zip(movers, cells):
            for i, ln in enumerate(lines):
                parts = ln.split()
                if parts[:2] == ["U", str(u[0])]:
                    parts[3], parts[4] = str(x), str(y)
                    lines[i] = " ".join(parts)
        block2 = "\n".join(lines)
        m2 = dict(m, blocks=m["blocks"][:turn - 1] + [block2])
        stream = D.prefix_stream(m2, seat, turn)
        units2 = parse_block(block2).units
        rec = native.extract(stream)
        rec = rec[(rec[:, 0] == turn) & (rec[:, 1] == seat)][0]
        cand = self.L.get(rec, "cand")[0]
        self.assertTrue((cand >= 0).all(), "the constructed root does not fill unit 0's slots")
        beyond = [u[0] for u in movers if u[0] not in cand]
        tok = F.normalize_line(m["replies"][turn - 1][seat]).split()
        reqs = [m["replies"][turn - 1][seat]] + [" ".join([tok[0], mv, str(t)] + tok[3:]) for t in beyond for mv in "SU"]
        res = self.compare(stream, m2, units2, seat, turn, reqs, "constructed full slots")
        self.assertTrue(any(fl & F.TARGET_OUT_OF_CAP for fl in res["flags"]), "no out-of-cap candidate was produced")


if __name__ == "__main__":
    unittest.main()
