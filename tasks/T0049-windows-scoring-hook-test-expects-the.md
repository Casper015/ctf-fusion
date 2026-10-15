---
id: T0049
title: "Windows: scoring-hook test expects the .exe the compiler writes"
status: review
owner: claude@caspers-macbook
needs: []
writes: [bots/p_only/tests/test_decision_scoring.py]
depends: []
land: pr
branch: claude/t0049-windows-scoring-hook-test-expects-the
updated: 2026-10-15
---

## Goal

T0038's handoff found that bots/p_only/tests/test_decision_scoring.py (T0045) fails on Windows: g++ writes <name>.exe when -o has no suffix, and the test checks the unsuffixed path. Name the test binaries with the platform's executable suffix.

## Done when

PR merged; the test passes on macOS and names .exe binaries on Windows.

## Handoff

*2026-10-15 02:05Z claude@caspers-macbook:*

PR #48: test binaries named with .exe on Windows (g++ appends it); 6 scoring tests pass on macOS; not run on Windows here.

## Log

- 2026-10-15 02:04Z claude@caspers-macbook: created
- 2026-10-15 02:04Z claude@caspers-macbook: claimed
- 2026-10-15 02:05Z claude@caspers-macbook: handoff -> review
