"""Test-only bot whose replies come from os.urandom: a process that can never replay its record (task T0085 tests)."""
import os
import sys

turn = 0
for line in sys.stdin:
    if line.startswith("TURN "):
        turn = int(line.split()[1])
    elif line.strip() == "END":
        b = os.urandom(3)
        print(f"{turn} " + " ".join(f"{'SUDLR'[x % 5]} -" for x in b), flush=True)
