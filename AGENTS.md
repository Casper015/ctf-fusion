# Working in ctf-fusion as an agent

Several agents work in this repository at the same time, on different machines: the Windows host, a Mac, cloud sandboxes. This file is the protocol that keeps them from overwriting each other's work. Read it before you change anything. Humans start at [README.md](README.md). Where files go and what may be committed is in [CONTRIBUTING.md](CONTRIBUTING.md).

The idea in one paragraph: **all work is a task on the board in [`tasks/`](tasks/README.md).** You claim a task before you touch the repository. A claim is atomic, so two agents can never hold the same task. Each task lists the paths it may write (its *lanes*), and two active tasks never share a lane. You work in your own git worktree on your own branch. You land on `main` only through `task.py land`, which refuses files outside your lanes. When you stop, the task file records what you did and what comes next. A separate "prompt for the next agent" file is never needed.

## 1. Say who you are

```bash
export CTF_AGENT=codex@desktop-g60ojud     # <tool>@<machine>, lowercase; UNIQUE per running agent (see below)
export CTF_CAN=host,cuda,ai-test           # what this machine has (table below); empty on a laptop or in the cloud
```

| Machine | `CTF_CAN` | What runs there |
|---|---|---|
| `desktop-g60ojud` (Windows) | `host,cuda,ai-test` | The only machine for timing, matches, screens and roster runs; GPU training |
| `caspers-macbook` | *(empty)* | Code, docs, offline analysis, unit tests (the AI_Test pipeline tests skip) |
| cloud sandboxes | *(empty)* | Code and docs only |

- `host`: AI_Test plus the official limits. This is the only place where timing and strength numbers are evidence.
- `cuda`: an NVIDIA GPU for training.
- `ai-test`: an AI_Test checkout, enough for pipeline tests and smoke games.

When a machine changes, update this table.

**Your identity must be unique among running agents.** The board treats two agents with the same `CTF_AGENT` as one agent: the second one's `claim` of the first one's task silently succeeds, and both then work on it. If two instances of the same tool run on one machine, give each a suffix after the tool name: `claude-ci@caspers-macbook`, `claude-2@caspers-macbook`. Always use the id that `task.py new` or `task.py list` printed, never a number you remember.

## 2. The loop

```bash
T=tools/repo/task.py
python $T list --ready                 # open tasks this machine can take now
python $T show T0002                   # goal, spec links, done-when, last handoff
python $T claim T0002                  # refused if someone holds it, its lanes are busy, or your machine lacks a need
python $T start T0002                  # -> ../ctf-fusion.worktrees/T0002 on branch <tool>/t0002-...; work only there
#   ... work, commit small commits inside the task's writes ...
python $T note T0002 "stage 1 running: 900 of 2,520 games, no faults"     # at milestones, so others see progress
python $T amend T0002 --writes bots/p_only/tools/    # you need another path: refused while another active task writes there
python $T land T0002                   # rebase on main, lane check, core tests, fast-forward main
python $T done T0002 --text "result: bots/p_only/report/capacity/SCREEN.md; XXL helps at 11-15, not shown elsewhere"
git worktree remove ../ctf-fusion.worktrees/T0002
```

Stopping before the task is finished? Hand it off. Never leave a task `active` when you stop:

```bash
python $T handoff T0002 --status open --file handoff.md      # someone else may continue
python $T handoff T0002 --status blocked --text "Question for Casper: reuse seed 20261107 after the crash, or a new seed?"
```

Write the handoff as if the next agent knows nothing:
- what is done, with commit hashes and output paths;
- what is next, as exact commands;
- what is uncertain or was not checked.

## 3. Rules

**Isolation**
- One task, one agent, one worktree. Never work in another agent's worktree or in the shared main checkout.
- Keep the main checkout on `main`, clean, and update it only with `git pull --ff-only`.
- The board changes only through `task.py`, which pushes each change to `main` atomically. Never edit `tasks/*.md` on your branch: `land` would refuse it.
- Stay in your lanes. To change something outside them, create a task for it (`task.py new`) or ask the holder of that lane in their task's log. Never widen your own `writes` silently.

**Shared machine**
- Wrap every run that produces timing or strength evidence (screens, timing, collection, fingerprints) in the CPU lock: `python $T lock cpu --task T0002 -- python bots/p_only/tools/run_screen.py ...`.
- Wrap every GPU training run in `lock gpu`. A run that shared the CPU with another run is not evidence.
- Don't pull, rebuild or clean `build/` while your own screen runs. Results are bound to binary hashes and to the runner's code, and a pull can change both.

**Landing**
- `land` is the only way onto `main`, except through a pull request.
- A task marked `land: pr` (shared core code, models, bots going to the host) goes through a pull request: `git push -u origin HEAD && gh pr create --fill`. Link the PR in a `note`.
- Never force-push `main` or rewrite pushed history. Never use a bare `git stash`. Never commit `data/`, `build/` or `runs/`.
- **`main` stays green.** CI ([CONTRIBUTING.md](CONTRIBUTING.md#automatic-checks-ci)) runs on every push to `main`. If something you landed turns it red, fixing or reverting it comes before any other work. `gh run list --branch main --limit 3` shows the state; `land` runs the same core tests, but not the P-only, torch or Windows jobs.

**Evidence**
- Report what the data shows. An interval that includes zero means "not shown". Missing timing stays missing, never zero.
- Development-machine timings are never evidence.
- Never reuse a used seed (see the task's spec). Never edit a result file by hand: regenerate it.

**Blocked**
- If you need a human decision, hand off with `--status blocked` and a one-line question. Humans watch `task.py list` for blocked tasks.
- If you find a task `active` whose owner is gone (no log line for a day and no lock held), ask in its log first. Take it with `claim --force` only after a human agrees, and record that in the log.

## 4. Lanes: who usually writes where

| Paths | Who writes them | How they land |
|---|---|---|
| `fusion/`, `training/`, `inference/*.hpp`, `tests/`, `tools/cpp/` | any machine (shared core) | pull request |
| `inference/models*/`, `reports/models/` | the CUDA machine, through `training.export_weights` | pull request |
| `reports/host/`, `reports/eval/`, `bots/p_only/report/`, `roster/report/`, `profiles/*pool*.json` | the host only | `land` |
| `bots/hybrid/`, `bots/p_only/` (code, profiles) | any machine; evidence comes from the host | `land`, or a PR when the task says so |
| `roster/*.py`, `tools/` | any machine | `land` |
| `docs/`, `README.md`, `CONTRIBUTING.md`, `AGENTS.md` | any machine | `land` |
| `tasks/` | only `task.py` | automatic |

A task's `writes` field is the binding version of this table for that task.

## 5. When something goes wrong

| Symptom | Do this |
|---|---|
| `claim` refused | Read the reason. Pick another task, or wait for the lane to free up. |
| `land`: files outside the lane | Move those commits to another branch or task, or open a PR. Don't widen the lane yourself. |
| `land`: rebase conflict | Resolve it in your worktree, `git rebase --continue`, then `land` again. |
| `land`: tests fail | Fix them on your branch. Nothing was pushed. |
| `lock` held | `python $T locks` shows who. Wait (`--wait 3600`). `unlock` only when that process is gone. |
| The board push keeps failing | Someone is changing the board right now. Run the same command again. |
| CI is red on your pull request | `gh run view --log-failed`. Reproduce it with the matching command in [CONTRIBUTING.md](CONTRIBUTING.md#automatic-checks-ci). The Windows leg is informational; every other job must be green to merge. |

## 6. Quick reference

- Task file format and statuses: [tasks/README.md](tasks/README.md).
- Every path that moved, and old handoff briefs: [docs/PATH_MAP.md](docs/PATH_MAP.md), [docs/handoffs/](docs/handoffs/README.md).
- Tests before landing (`land` runs the first one for you):
  - `python -m unittest discover -s tests -t .`
  - `python -m unittest discover -s bots/p_only/tests` when you touched `bots/`.
