---
id: T0082
title: "Freeze production agent_v2 release: single-file bot.cpp, profile, manifest and release report"
status: review
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [bots/agent_v2/, profiles/agent_v2.json, docs/, reports/release/agent_v2/, tests/]
depends: [T0077, T0079]
land: pr
branch: antigravity/t0082-freeze-production-agent-v2-release
updated: 2026-10-17
---

## Goal

Freeze agent_v2 production release based on T0077's winning general PPO policy G (agent_v2_rl_general_csp, +0.2 BT) and T0079's empirical rejection of search. Bundle standalone single-file bot.cpp, update profiles/agent_v2.json, compute frozen bundle/binary SHA-256 hashes and model manifest, update documentation and write release report.

## Done when

Single-file bot.cpp bundled and compiled cleanly with zero external dependencies; profiles/agent_v2.json updated with frozen production default and model manifest; unit tests passing; release report documented under reports/release/agent_v2/REPORT.md; PR submitted.

## Handoff

*2026-10-17 08:31Z antigravity@desktop-g60ojud:*

PR https://github.com/Casper015/ctf-fusion/pull/69 opened from branch antigravity/t0082-freeze-production-agent-v2-release. Standalone single-file bot.cpp bundled and compiled cleanly with zero external dependencies (SHA-256 7d344a6dd33efcf2f2966e57df43c109314644ba628a2b6032a2dc0a8eb79c32); profiles/agent_v2.json updated with frozen production default and model manifest; unit tests passing; release report documented under reports/release/agent_v2/REPORT.md.

## Log

- 2026-10-17 08:25Z antigravity@desktop-g60ojud: created
- 2026-10-17 08:26Z antigravity@desktop-g60ojud: claimed
- 2026-10-17 08:31Z antigravity@desktop-g60ojud: handoff -> review
