---
id: T0043
title: "Own-action features in C++: runtime encoder and Python parity on recorded inputs"
status: open
owner: ""
needs: []
writes: [inference/own_action.hpp, tools/cpp/own_action_check.cpp, tests/test_own_action_parity.py]
depends: []
land: pr
branch: ""
updated: 2026-10-14
---

## Goal

Stage 2 of Casper's own-action scorer preparation (2026-10-14). A C++ runtime encoder of fusion/decision_features.py (own-action-1): canonical complete three-unit orders (move, wait, pickup, drop, attack target slot; illegal moves, canonicalized actions, out-of-cap and invalid targets, off-board units, parse failures), effective duplicates and support, root features and candidate features with the same normalization. Inputs are the decision root and its recorded history only. A check tool and tests compare C++ and Python on recorded matches (respawn, collisions, flag carrying, low HP, unavailable targets, duplicate orders) and pin version, dimensions, normalization, slot mapping and the feature identity digest. Code and tests only.

## Done when

PR merged; parity tests pass bit for bit on recorded inputs; the C++ header pins the feature identity digest checked against Python.

## Handoff

Not started.

## Log

- 2026-10-14 19:48Z claude@caspers-macbook: created
