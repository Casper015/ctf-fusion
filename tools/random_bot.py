"""Seeded random bot for pipeline tests: random open moves, picks up flags, shoots in range.

It exercises every label path (moves, blocked moves, pickups, drops, attacks)
in a few seconds per match without loading the machine. Not a real opponent.
"""
import random
import sys

DELTAS = {"S": (0, 0), "U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def main():
    rng = random.Random(7)
    lines = iter(sys.stdin.readline, "")
    walls, me, size = set(), 0, 0
    for line in lines:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "INIT":
            me, size = int(tok[2]), int(tok[3])
            rng.seed(1000 + me)
        elif tok[0] == "MAP":
            for y in range(size):
                row = next(lines)
                walls |= {(x, y) for x, c in enumerate(row) if c == "#"}
        elif tok[0] == "TURN":
            turn = int(tok[1])
            units, flags = {}, []
            for line in lines:
                tok = line.split()
                if tok[0] == "U":
                    units[int(tok[1])] = [int(v) for v in tok[1:8]]
                elif tok[0] == "F":
                    flags.append((tok[2], int(tok[3]), int(tok[4])))
                elif tok[0] == "END":
                    break
            out = [str(turn)]
            for uid in range(3 * me, 3 * me + 3):
                _, team, x, y, hp, flag, _ = units[uid]
                moves = [m for m, (dx, dy) in DELTAS.items() if (x + dx, y + dy) not in walls]
                move = rng.choice(moves) if x >= 0 else "S"
                act = "-"
                if flag >= 0 and rng.random() < 0.02:
                    act = "X"
                elif any(s in ("home", "dropped") and (fx, fy) == (x, y) for s, fx, fy in flags) or rng.random() < 0.05:
                    act = "P"
                else:
                    near = [v for v in units.values() if v[1] != team and v[2] >= 0 and abs(v[2] - x) + abs(v[3] - y) <= 3]
                    if near and rng.random() < 0.7:
                        act = str(rng.choice(near)[0])
                out += [move, act]
            print(" ".join(out), flush=True)


if __name__ == "__main__":
    main()
