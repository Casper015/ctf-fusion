---
id: T0043
title: "Own-action features in C++: runtime encoder and Python parity on recorded inputs"
status: review
owner: claude@caspers-macbook
needs: []
writes: [inference/own_action.hpp, tools/cpp/own_action_check.cpp, tests/test_own_action_parity.py]
depends: []
land: pr
branch: claude/t0043-own-action-features-in-c-runtime-encoder
updated: 2026-10-14
---

## Goal

Stage 2 of Casper's own-action scorer preparation (2026-10-14). A C++ runtime encoder of fusion/decision_features.py (own-action-1): canonical complete three-unit orders (move, wait, pickup, drop, attack target slot; illegal moves, canonicalized actions, out-of-cap and invalid targets, off-board units, parse failures), effective duplicates and support, root features and candidate features with the same normalization. Inputs are the decision root and its recorded history only. A check tool and tests compare C++ and Python on recorded matches (respawn, collisions, flag carrying, low HP, unavailable targets, duplicate orders) and pin version, dimensions, normalization, slot mapping and the feature identity digest. Code and tests only.

## Done when

PR merged; parity tests pass bit for bit on recorded inputs; the C++ header pins the feature identity digest checked against Python.

## Handoff

*2026-10-14 19:53Z claude@caspers-macbook:*

PR #40 (branch claude/t0043-own-action-features-in-c-runtime-encoder). Done: inference/own_action.hpp (runtime encoder, canonicalization, duplicates, support, root/candidate features; pins own-action-1 and digest 9ed719f238cd5d70), tools/cpp/own_action_check.cpp, tests/test_own_action_parity.py (24 recorded roots, 4,534 replies, one constructed full-slot root; bit-exact). Verified: core suite 303 OK. Next: T0044 (scorer export + C++ inference) builds on this header. Not checked: Windows compilers.

## Log

- 2026-10-14 19:48Z claude@caspers-macbook: created
- 2026-10-14 19:48Z claude@caspers-macbook: claimed
- 2026-10-14 19:53Z claude@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/40 opened.
- 2026-10-14 19:53Z claude@caspers-macbook: handoff -> review
