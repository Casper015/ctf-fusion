# ctf-fusion

[![CI](https://github.com/Casper015/ctf-fusion/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Casper015/ctf-fusion/actions/workflows/ci.yml)

**Teaching our Capture the Flag bots to predict what their opponents will do next.**

Our bots compete in a Capture the Flag game ([AI_Test](https://github.com/Casper015/AI_Test), `ctf-arena`). Between 2 and 15 teams share one map. Each team has three units that move, shoot, and carry flags home for points. A bot gets 2 seconds for its first move and 50 milliseconds for every move after that. It runs on an ordinary CPU, with no GPU and no internet.

To plan a good move, a bot has to guess what every other team is about to do. Our older bots make that guess with rules people wrote by hand. This repository replaces the hand-written guess with small neural networks that learned it from thousands of recorded games, and checks at every step that the replacement is really better.

## The idea in five steps

1. **Play and record games.** Our bots play many games against a wide range of opponents (the *roster*: 130+ archived bots, grouped by how they behave). Every turn is saved.
2. **Learn from them.** Two networks train on those records:
   - **P**, the *policy*, predicts what each opponent team does next;
   - **V**, the *value*, estimates how good a position is for us.
3. **Check them against the old guess.** On games the networks never saw, P must predict the opponents better than the hand-written rules, by a clear margin, for each number of teams separately. Where it doesn't, the bot falls back to the old rules automatically. We call this check the *gate*.
4. **Turn them into C++.** The networks are exported as plain C++ with integer arithmetic, which matches the Python version bit for bit. They fit easily inside the 50 ms limit.
5. **Put them in a bot and play.** The networks go into two of our strongest existing bots, called *carriers*. We compare the result with the original bots in thousands of paired games on the official test machine.

## Where things stand (October 2026)

| | |
|---|---|
| **Networks** | P and V are trained in five sizes, from S (about 30,000 multiply-adds per opponent) to XXL (about 680,000). Each size has 11 models: one for each of 2–8, 10, 12 and 15 teams, plus a general model for 9, 11, 13 and 14. **Every size passes the gate for all 14 team counts.** |
| **Speed** | Even the largest size fits. In the timing run on the test machine (8, 12 and 15 teams), the slowest reply of the heaviest bot was 31.4 ms, against the 50 ms limit. |
| **Bots** | Two carriers, `C08` and `v12`, each in two versions. *Hybrid*: the network's guess mixed with the old one. *P-only*: every guess about opponents comes from P. |
| **Match results** | Promising, not yet confirmed. In the first 280-map screen, the C08 P-only bot beat the original C08: +4.0 points of "share of other teams beaten", 95% interval +0.8 to +7.1. On a second set of fresh maps the gain was +2.9, but its interval touched zero. The v12 P-only bots showed the reverse pattern: not shown on the first maps, clearly ahead on the second (+5 to +7). |
| **Next** | Controlled experiments to separate what helps: a bigger P, more search, or V. Then a deeper lookahead, then a large confirmation run. All of it is on the [task board](tasks/README.md). |

The numbers come from [bots/p_only/report/](bots/p_only/report/screen/SCREEN.md) and [reports/models/](reports/README.md).

## Where things are

| Folder | What's inside |
|---|---|
| [`bots/`](bots/hybrid/README.md) | The bots we build: `hybrid/` (network mixed with the old guess) and [`p_only/`](bots/p_only/README.md) (network only), with their tools, tests and match reports |
| `fusion/`, `training/` | The Python side: recording games, building datasets, training, the gate, exporting to C++ |
| `inference/` | The C++ that goes inside a bot: game state, features, the network code, the simulator, and the exported models (`models*/`) |
| `roster/`, `profiles/` | The opponents: every archived bot, how they behave, which ones we train against |
| `reports/` | Results: gate reports per model, measurements from the test machine, match results, written reviews |
| `docs/` | Design, decisions and deeper explanations; start with the [technical overview](docs/TECHNICAL_OVERVIEW.md) |
| `tasks/` | The [task board](tasks/README.md): what is being worked on, by whom, and what is next |
| `tools/`, `tests/` | Scripts for the test machine, experiments and reports, plus the automatic checks |

## Who works here, and how

People and several AI agents work in this repository at the same time, from different machines: the Windows test machine, a Mac, cloud sandboxes. To keep them from getting in each other's way:

- **All work is a task** on the [board](tasks/README.md). `python tools/repo/task.py list` shows what is open, who is doing what, and what is blocked waiting for a person.
- **Agents follow [AGENTS.md](AGENTS.md).** Each one claims a task, works in its own copy of the repository, and may only change the folders that task names. Two agents can't take the same task or edit the same folders at once.
- **Where files go, and what may be committed,** is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Getting started

You need Python 3.10+ with NumPy, a C++20 compiler (g++, clang, or w64devkit on Windows), and git. To run games you also need a checkout of AI_Test.

```bash
git clone https://github.com/Casper015/ctf-fusion.git && cd ctf-fusion
pip install -r requirements-dev.txt              # NumPy, plus matplotlib for the report generators
python -m unittest discover -s tests -t .        # about 20 seconds
```

To run games, point to AI_Test first: `export AI_TEST_ROOT=~/code/AI_Test`. The tests then also play short test games. A full first run, step by step, is in [docs/FIRST_RUN.md](docs/FIRST_RUN.md).

**Two rules worth knowing:**
- Speed and strength are measured only on the Windows test machine. Numbers from a laptop don't count.
- Keep the repository outside OneDrive.

## Words used in this project

| Word | Meaning |
|---|---|
| **P / policy network** | Predicts each opponent team's next moves: where its units go, whether they shoot, whom they target. |
| **V / value network** | Scores a position: how well is this going for us? |
| **q_H, the legacy forecast** | The old hand-written guess of opponent moves, which P has to beat. |
| **Gate** | The check that P beats q_H on unseen games, per team count. If it fails, that team count keeps the old guess. |
| **Size / tier** | Network size: S, M, L, XL, XXL. Bigger is more accurate but slower. |
| **Carrier** | An existing strong bot that we add the networks to (`C08`, `v12`). |
| **Hybrid / P-only** | Hybrid mixes the network's guess with the old one. P-only uses the network's guess alone. |
| **Screen** | A batch of paired games: every bot version plays the same maps, so the differences are fair. |
| **Host / test machine** | The Windows machine (`DESKTOP-G60OJUD`) that matches the competition's conditions. |
| **"Helps" / "not shown"** | Helps means the whole 95% interval is above zero. Not shown means we can't yet tell the change from luck. |

## Read more

- [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md): the pipeline in detail, the guarantees each test enforces, all commands.
- [docs/DESIGN.md](docs/DESIGN.md) and [docs/REVIEW.md](docs/REVIEW.md): why it is built this way, and the decisions so far.
- [docs/ROSTER.md](docs/ROSTER.md): how opponents are chosen.
- [bots/p_only/README.md](bots/p_only/README.md): the P-only bots and their experiments.
- [docs/PATH_MAP.md](docs/PATH_MAP.md): where files moved in the October 2026 reorganization.
