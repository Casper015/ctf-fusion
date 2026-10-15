---
id: T0048
title: "Add the original meowmind source next to its P3 hybrids"
status: open
owner: ""
needs: []
writes: [bots/meowmind_p/]
depends: []
land: pr
branch: ""
updated: 2026-10-15
---

## Goal

Casper's request (2026-10-14): commit the unmodified meowmind bot (meowmind-ctf-374-20261011.cpp, sha256 cc7b730c...) in bots/meowmind_p/, so the two hybrid bundles can be rebuilt from the repository alone. Casper chose to publish it; the source carries no license notice.

## Done when

PR merged; the committed file's sha256 equals the README's, and make.py rebuilds meowmind_p_t0011_xxl_both.cpp byte for byte from it.

## Handoff

Not started.

## Log

- 2026-10-15 00:59Z claude@caspers-macbook: created
