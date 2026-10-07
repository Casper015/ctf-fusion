"""Seeded test bot: mostly greedy toward flags and home, sometimes random.

It plays fast, legal and varied games (moves, collisions, pickups, captures,
drops, attacks), so tests exercise every label and simulator path in about a
second per match without loading the machine. It is not a real opponent.
"""
import random
import sys

DELTAS = {"S": (0, 0), "U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def main():
    rng = random.Random(7)
    lines = iter(sys.stdin.readline, "")
    walls, me, size, bases = set(), 0, 0, []
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
        elif tok[0] == "BASES":
            bases = [tuple(int(v) for v in next(lines).split()) for _ in range(int(tok[1]))]
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
            ground = {(x, y) for s, x, y in flags if s in ("home", "dropped")}
            out = [str(turn)]
            for uid in range(3 * me, 3 * me + 3):
                _, team, x, y, hp, flag, _ = units[uid]
                if x < 0:
                    out += ["S", "-"]
                    continue
                moves = [m for m, (dx, dy) in DELTAS.items() if (x + dx, y + dy) not in walls]
                goal = bases[team] if flag >= 0 else min(ground, key=lambda g: abs(g[0] - x) + abs(g[1] - y), default=None)
                if goal is not None and rng.random() < 0.7:
                    move = min(moves, key=lambda m: (abs(x + DELTAS[m][0] - goal[0]) + abs(y + DELTAS[m][1] - goal[1]), rng.random()))
                else:
                    move = rng.choice(moves)
                dest = (x + DELTAS[move][0], y + DELTAS[move][1])
                act = "-"
                if flag >= 0 and rng.random() < 0.01:
                    act = "X"
                elif flag < 0 and dest in ground:
                    act = "P"
                else:
                    near = [v for v in units.values() if v[1] != team and v[2] >= 0 and abs(v[2] - x) + abs(v[3] - y) <= 3]
                    if near and rng.random() < 0.7:
                        act = str(rng.choice(near)[0])
                out += [move, act]
            print(" ".join(out), flush=True)


if __name__ == "__main__":
    main()
