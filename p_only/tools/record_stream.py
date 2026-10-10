"""Record the exact input one seat receives during a game (fixtures for p_only/tests/contract_test.cpp).

    python p_only/tools/record_stream.py --teams 4 --seed 31 --bot build/p_only/bin/c08_control-<hash> \\
        --out p_only/tests/fixtures/stream_n04.txt.gz

Plays one game with the same bot in every seat under relaxed limits (5 s / 1 s; a
fixture, not a timing or strength measurement) and writes, gzip-compressed, every
turn block the engine sends seat 0, in order: what a bot reads on stdin.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fusion import runner  # noqa: E402
from fusion.paths import ai_test_root, load_engine  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teams", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bot", required=True, help="executable played in every seat")
    ap.add_argument("--ai-test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    engine = load_engine(ai_test_root(a.ai_test))
    game = engine.Game(a.teams, a.seed, flag_seed=a.seed * 7919 + 1)
    exe = str(Path(a.bot).resolve())   # bots start in their own temporary directory
    bots = [runner.Bot([exe], f"seat{i}", (5.0, 1.0)) for i in range(a.teams)]
    blocks = []
    try:
        while not game.done:
            turn = game.turn + 1
            texts = [game.encode(i) for i in range(a.teams)]
            blocks.append(texts[0])
            starts = [b.send(t) for b, t in zip(bots, texts)]
            lines = [b.receive(t0, turn) for b, t0 in zip(bots, starts)]
            orders = {}
            for i, (b, line) in enumerate(zip(bots, lines)):
                try:
                    orders[i] = game.decode(i, line) if line is not None else {}
                    b.ok()
                except ValueError:
                    orders[i] = {}
            game.step(orders)
    finally:
        for b in bots:
            b.close()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(t if t.endswith("\n") else t + "\n" for t in blocks)
    out.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    print(f"{out}: {len(blocks)} turns, score {game.result()['score']}, {len(text)} bytes")


if __name__ == "__main__":
    main()
