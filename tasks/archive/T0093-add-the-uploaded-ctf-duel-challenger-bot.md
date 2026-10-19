---
id: T0093
title: "Add the uploaded ctf_duel_challenger bot as a buildable variant"
status: done
owner: claude-fervent@cloud
needs: []
writes: [bots/challengers/]
depends: []
land: direct
branch: claude-fervent/t0093-add-the-uploaded-ctf-duel-challenger-bot
updated: 2026-10-19
---

## Goal

Casper uploaded ctf_duel_challenger.zip (2026-10-19, zip sha256 31f3c872..., one file ctf_duel_challenger.cpp, sha256 79bb5106..., MIT license, meow-ai-arena contributors; a MeowMind-family learned policy/value search bot). Commit it unmodified under bots/challengers/ with a README recording provenance, license and hashes, and a variants.json that builds it with bots/p_only/tools/build.py beside the current contenders (original Meowmind with and without search, Meowmind with our P3, agent_v2_p2xxl_direct) for the host ranking task. A development-machine replay may check that it answers every turn; no timing or strength claim.

## Done when

bots/challengers/ holds the unmodified file, README and variants.json; build.py builds the challenger; the layout tests pass; landed on main.

## Handoff

*2026-10-19 18:42Z claude-fervent@cloud:*

Landed edd7c1a: bots/challengers/ctf_duel_challenger.cpp (unmodified, sha256 79bb5106...), README (provenance, MIT license, hashes) and variants.json (challenger + original Meowmind, Meowmind+P3, agent_v2_p2xxl_direct). build.py builds it; dev replay of n02/n04/n15 answered 400/400 turns (not evidence). Core tests 475 OK. Next: host task T0094 ranks the contenders.

## Log

- 2026-10-19 17:08Z claude-fervent@cloud: created
- 2026-10-19 17:09Z claude-fervent@cloud: claimed
- 2026-10-19 18:41Z claude-fervent@cloud: landed: edd7c1a178 on main, 3 file(s)
- 2026-10-19 18:42Z claude-fervent@cloud: done
