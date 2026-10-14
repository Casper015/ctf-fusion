---
id: T0044
title: "Own-action scorer export and C++ float inference (linear, MLP)"
status: active
owner: claude@caspers-macbook
needs: []
writes: [inference/own_action_scorer.hpp, training/export_decision_scorer.py, tools/cpp/own_action_scorer_check.cpp, tests/test_own_action_scorer.py]
depends: [T0043]
land: pr
branch: claude/t0044-own-action-scorer-export-and-c-float
updated: 2026-10-14
---

## Goal

Stage 3 of Casper's own-action scorer preparation (2026-10-14). A dedicated export and load route for model kind own_action_scorer (linear and the existing small MLP of training/train_decision_ranker.py), never written over or loadable as opponent P or legacy V. The export binds the feature identity, run config and model hashes, normalization, the frozen selection (lambda, norm) and the carrier configuration of the training data. A float reference C++ inference (inference/own_action_scorer.hpp) is validated against Python with fixed test weights; quantization is not part of this task. No trained model is needed or committed.

## Done when

PR merged; exported headers of fixed-weight linear and MLP scorers reproduce Python's Q on recorded roots within a stated tolerance; kind, identity and width contracts are refused when violated.

## Handoff

Not started.

## Log

- 2026-10-14 19:54Z claude@caspers-macbook: created
- 2026-10-14 19:55Z claude@caspers-macbook: claimed (forced)
- 2026-10-14 19:55Z claude@caspers-macbook: note: Claimed with --force before T0043 (same owner) merged: the branch is stacked on T0043's branch (PR #40) because it includes inference/own_action.hpp; its PR shows T0043's commit until #40 merges.
