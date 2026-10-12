# T0013 delivery checks

The required core suite ran 224 tests: 223 passed, one platform skip, zero failures or errors, exit 0. It ran on frozen source 64576e0d6d428e5f45f9ad0a89216d6c4808bfb4 after report generation; log hash and elapsed time are in CHECKS.json. Runtime is not a timing/strength measurement.

The independent audit checked all 1,960 original games, raw/source/binary/engine/seed/map/flag/seat/limits identities, 784,000 full turn blocks and 6,664,000 all-seat replies, with zero all-seat faults and missing/invalid replies. It independently recomputed all paired/interaction/A/A confidence intervals and PONLY totals. SEARCH sample ratios remain the original analyzer output, and individual timing samples are not retained for independent percentile recomputation. See INDEPENDENT_AUDIT.json.

The four actual rebuilt feature-family carrier binaries match the T0011 binaries whose real weights, registry dispatch and runtime parity were already verified; bundle hash differences come from carrier-header commentary. Both model sets match their original exports. No game, cost probe, training, prediction or successful audit was repeated. No production code, model, export or default change. Public path scan, embedded-source hashes and report layout are checked during final packaging; PR exact-head CI is separate.
