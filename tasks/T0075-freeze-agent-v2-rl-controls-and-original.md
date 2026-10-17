---
id: T0075
title: "Freeze agent_v2 RL controls and original Meowmind search ablation"
status: blocked
owner: claude-t75@caspers-macbook
needs: []
writes: [bots/agent_v2/controls/, inference/agent_v2_controls/, bots/meowmind_p/attribution/, tools/screens/next_controls/, tests/test_next_controls.py]
depends: [T0068]
land: pr
branch: claude-t75/t0075-freeze-agent-v2-rl-controls-and-original
updated: 2026-10-17
---

## Goal

Build the controls that the T0068 evaluation and the Meowmind gap attribution need. (1) agent_v2 variants with the CSP P bank (FUSION_TIER_CSP) that play only T0068's general control (rl_general) on every count, and only the common initialization actor (data/count_specialists_rl/init_actor.npz, exported as INT16 with the same scheme, temperature, resolver and precision as agent_v2_rl_csp). Recover and record the init actor's sha256 and origin; do not assume another warm actor is equivalent, and stop with a blocked handoff if its provenance cannot be established. Use a dedicated variant manifest under bots/agent_v2/controls/ (T0068 reserves bots/agent_v2/variants.json). (2) Original Meowmind wrappers that differ only in NN_SEARCH=1 versus NN_SEARCH=0 (bots/meowmind_p/meowmind-ctf-374-20261011.cpp line ~680), source and predictor untouched. (3) A preflight adapter for both telemetry formats that never fabricates absent counters. No games, no seed consumption, no timing or strength claims: tests replay the recorded streams only (bots/p_only/tests/fixtures/stream_n02/n04/n15).

## Done when

PR merged: every control compiles in CI; tests build the real variant files and replay n02/n04/n15 with valid orders, check that the general-only variant routes every count to rl_general and the init-only variant to the init actor with matching outExp, and check INT16 agreement of the init export; a manifest pins every arm's source, model and bundle hashes and shows that the two Meowmind wrappers differ only in NN_SEARCH.

## Handoff

*2026-10-17 05:13Z claude-t75@caspers-macbook:*

Blocked: the init-only arm I needs data/count_specialists_rl/init_actor.npz, which exists only on desktop-g60ojud. T0068's committed records (profiles/count_specialists_rl.json, REPORT.md, TRAINING_SUMMARY.json, MANIFEST.json) name the path but give no sha256 or origin. Question for the host/Casper: from T0068's run configs (runs/count_specialists_rl/*/config.json, record "init_actor": {"path", "sha256"}, written by training/agent_v2/train.py), what are the sha256 of init_actor.npz and of data/count_specialists_rl/calibration.npz, and where did init_actor.npz come from (producing task/commit)?

Done (PR https://github.com/Casper015/ctf-fusion/pull/66, branch claude-t75/t0075-freeze-agent-v2-rl-controls-and-original, commits 10b3e1c and f0be891):
- G = bots/agent_v2/controls/agent_v2_rl_general_csp.cpp (rl_general on every count), I = agent_v2_rl_init_csp.cpp (rl_init), list bots/agent_v2/controls/variants.json; routing in inference/agent_v2_controls/.
- ON/OFF = bots/meowmind_p/attribution/meowmind_search_{on,off}.cpp, list variants.json; bundles differ only in the NN_SEARCH line, ON preprocesses to the original.
- tools/screens/next_controls/: MANIFEST.json + manifest.py (pins), telemetry.py + preflight.py (adapter and eight-worker gate for T0077/T0078; written by Codex gpt-6.1-sol, reviewed), export_init_actor.py, probe.hpp.
- tests/test_next_controls.py: G routing 0-16 -> rl_general outExp 16; G replays n02/n04/n15 (valid orders, refused 0, identity) and INT16 agreement on 1,200 stream observations vs weights_general.npz; ON/OFF replays; manifest --check; telemetry. I tests skip until the export exists. The I path was dry-run on the Mac with a stand-in actor (never committed): routing, replay and agreement passed.

Next, on the host (needs the answer above), in a checkout of the PR branch:
  python tools/screens/next_controls/export_init_actor.py --expect-sha256 <init_actor.npz sha256> --expect-calibration-sha256 <calibration.npz sha256> --origin "<task, run config path, commit>"
  python tools/screens/next_controls/manifest.py
  python -m unittest tests.test_next_controls -v
  git add inference/agent_v2_controls/ tools/screens/next_controls/MANIFEST.json && git commit && git push
The exporter refuses a sha mismatch, and refuses unless re-exporting the general control reproduces its committed digest and agreement figures. Then the PR is reviewable and mergeable (land: pr).

Not checked: CI on the PR was still pending at handoff; a full local unittest discover was killed before finishing, so CI is the full-suite check. Windows paths (binary dump, canonical bundle markers, text bundle hash in preflight) were fixed from the Codex review but only exercised by CI's Windows leg. preflight.py has not run against the engine (no games on the Mac).

## Log

- 2026-10-17 04:06Z claude-review@caspers-macbook: created
- 2026-10-17 04:41Z claude-t75@caspers-macbook: claimed
- 2026-10-17 05:13Z claude-t75@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/66: G, ON/OFF, manifest, preflight adapter and tests done; I awaits the host export of init_actor.npz (sha256 not in T0068's committed records)
- 2026-10-17 05:13Z claude-t75@caspers-macbook: handoff -> blocked
