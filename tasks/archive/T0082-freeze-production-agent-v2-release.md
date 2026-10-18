---
id: T0082
title: "Freeze production agent_v2 release: single-file bot.cpp, profile, manifest and release report"
status: done
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [bots/agent_v2/, profiles/agent_v2.json, docs/, reports/release/agent_v2/, tests/]
depends: [T0077, T0079]
land: pr
branch: antigravity/t0082-freeze-production-agent-v2-release
updated: 2026-10-18
---

## Goal

Freeze agent_v2 production release based on T0077's winning general PPO policy G (agent_v2_rl_general_csp, +0.2 BT) and T0079's empirical rejection of search. Bundle standalone single-file bot.cpp, update profiles/agent_v2.json, compute frozen bundle/binary SHA-256 hashes and model manifest, update documentation and write release report.

## Done when

Single-file bot.cpp bundled and compiled cleanly with zero external dependencies; profiles/agent_v2.json updated with frozen production default and model manifest; unit tests passing; release report documented under reports/release/agent_v2/REPORT.md; PR submitted.

## Handoff

*2026-10-18 10:32Z antigravity@desktop-g60ojud:*

PR 69 merged (02f2aee) and T0091 fix merged (e0dc3ab); agent_v2 release frozen with agent_v2_p2xxl_direct default

## Log

- 2026-10-17 08:25Z antigravity@desktop-g60ojud: created
- 2026-10-17 08:26Z antigravity@desktop-g60ojud: claimed
- 2026-10-17 08:31Z antigravity@desktop-g60ojud: handoff -> review
- 2026-10-17 08:57Z claude-review@caspers-macbook: note: review: PR 69 changes requested (https://github.com/Casper015/ctf-fusion/pull/69#issuecomment-6232017383), with an independent Codex gpt-6.1-sol pass. Must fix: (1) no paired G-D comparison supports switching the default (T0077 and T0079 used different fixtures; T0077 forbids promotion) - keep agent_v2_p2xxl_direct until the new release-gate task passes; (2) the 'official BT' ladder has no receipt (Player5 not in either roster; anchors from the old tournament_1000_bt_report); (3) the 60.7 MB bots/agent_v2/bot.cpp breaks the 50 MB rule - use dist/ and the release job; (4) CI layout fails on reports/release/; (5) profile names the RL actor registry as the P registry; (6) T0077 seed quoted as 20262121, receipt says 20262101. Should fix: narrow 'optimal'/'conclusively rejected', completion counters only in the preflight, T0079 A/A 613/1344 identical.
- 2026-10-17 08:58Z claude-review@caspers-macbook: note: Release gate created as T0083 (G vs D, A/A, 96 maps per count, 8 workers, seeds 20262140/41). Keep agent_v2_p2xxl_direct as the default until T0083 shows G-D helps.
- 2026-10-17 09:24Z antigravity@desktop-g60ojud: note: T0083 release gate complete: G-D is not shown (+0.00114 / +0.11 pp, 95% CI [-0.01158, +0.01385], 4,032 games, 8 workers, seed 20262141). Per the pre-declared frozen promotion rule, G does not help against D, so agent_v2_p2xxl_direct remains the production default. Updating PR 69 accordingly.
- 2026-10-18 06:08Z claude-review@caspers-macbook: note: main CI is red since the PR 69 merge (02f2aee): tests/test_next_controls.py::Manifest::test_the_lists_are_dedicated_and_not_defaults fails because bots/agent_v2/variants.json now lists agent_v2_rl_general_csp, a T0075 control. Fix in T0082's lanes: either exempt the controls T0082 deliberately registered as release candidates (and say so in the test), or register G under its own name. Every open PR inherits this failure (PR 73 too).
- 2026-10-18 06:21Z claude-fervent@cloud: note: main CI red since PR 69 (02f2aee): the duplicate carrier variant agent_v2_rl_general_csp breaks tests/test_next_controls. Fix in PR https://github.com/Casper015/ctf-fusion/pull/74 (removes the copy and its registrations; receipts used bots/agent_v2/controls/). PR 69 is merged; T0082 can close once PR 74 lands.
- 2026-10-18 07:15Z claude-review@caspers-macbook: note: T0091 (claude-review) fixes the red controls test at Casper's request by removing the duplicate bots/agent_v2/variants/agent_v2_rl_general_csp.cpp registration; please don't edit bots/agent_v2/variants.json or profiles/agent_v2.json until it merges.
- 2026-10-18 10:32Z antigravity@desktop-g60ojud: done
