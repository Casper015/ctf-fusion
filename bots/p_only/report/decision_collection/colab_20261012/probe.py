
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, "/content/ctf-fusion")
from fusion import matchlog, runner
from training import decision_opponents as O
from training.decision_collect import commands_and_info

def probe(frozen, match, composition, limits=(2.0, 0.05)):
    """Every seat runs concurrently through the recorded game (replies are not applied)."""
    n = match["teams"]
    cmds, _ = commands_and_info(frozen)
    bots = [runner.Bot(cmds[name], name, limits) for name in composition]
    replies_of_seat0 = []
    try:
        for t in range(1, len(match["blocks"]) + 1):
            inputs = [(matchlog.init_for_seat(match["init"], i) if t == 1 else "") + match["blocks"][t - 1] for i in range(n)]
            starts = [b.send(x) for b, x in zip(bots, inputs)]
            got = [b.receive(s, t) for b, s in zip(bots, starts)]
            replies_of_seat0.append((t, got[0]))
    finally:
        for b in bots: b.close()
    return [dict(b.report(), seat=i, name=name) for i, (b, name) in enumerate(zip(bots, composition))], replies_of_seat0

def compositions(frozen, n, k, salt):
    reg = frozen["registry"]
    mix = O.mixture(reg)
    out = []
    for j in range(k):
        rng = random.Random(f"t0037-probe/{salt}/{n}/{j}")
        out.append(["@candidate"] + [mix["units"][u]["name"] for u in O.draw_opponents(rng, mix, n - 1)])
    return out
