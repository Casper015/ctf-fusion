# bots/agent_v2/controls/

The agent_v2 controls for T0077 (task T0075). T0077 compares three arms, all actor-direct agent_v2 with T0067's per-count opponent P bank (`FUSION_TIER_CSP`):

| Arm | Variant | Actor on every count |
|---|---|---|
| S | [`agent_v2_rl_csp`](../variants/agent_v2_rl_csp.cpp) (T0068) | the exact-count specialist `rl_nXX`, `rl_general` outside 2–15 |
| G | [`agent_v2_rl_general_csp`](agent_v2_rl_general_csp.cpp) | T0068's compute-matched general control `rl_general` ([`weights_general.hpp`](../../../inference/count_specialists_rl/weights_general.hpp)) |
| I | [`agent_v2_rl_init_csp`](agent_v2_rl_init_csp.cpp) | T0068's common initialization actor `rl_init` (`inference/agent_v2_controls/init_actor.hpp`, exported on the host) |

G and I are `agent_v2_rl_csp` with only the routing changed: each defines the same `meow_rl_model` constants (scheme `meow_rl_reference16_v1`, 12 input bits, temperature 1.05, `outExp` 16) and checks them against its actor header with `static_assert`. The P bank, the 2048 width, the resolver `assign_v1` and the carrier (`bots/agent_v2/bot.hpp`) are the same. The routing lives in [`inference/agent_v2_controls/`](../../../inference/agent_v2_controls/general_only.hpp). The stderr identity line names the actor (`count_specialists_rl_general_only`, `count_specialists_rl_init_only`) and its INT16 digest.

The list is [`variants.json`](variants.json), separate from T0068's [`bots/agent_v2/variants.json`](../variants.json). Neither control is a production default.

```bash
python bots/p_only/tools/build.py --variants bots/agent_v2/controls/variants.json
```

## The init actor

I plays `data/count_specialists_rl/init_actor.npz`, the actor that T0068's specialized and general arms both started from. `data/` is not committed, and T0068's committed records do not give this file's sha256. So I stays `awaiting_init_export` in [`MANIFEST.json`](../../../tools/screens/next_controls/MANIFEST.json), and its tests skip, until the host runs:

```bash
python tools/screens/next_controls/export_init_actor.py --expect-sha256 <sha256 from T0068's run configs> --origin "<task, run config, commit>"
python tools/screens/next_controls/manifest.py
```

The exporter refuses a file whose sha256 differs from `--expect-sha256`. It then repeats T0068's export of the general control (`training/train_count_specialists_rl.py`, `calibrate_and_export_actor`: the whole calibration corpus `data/count_specialists_rl/calibration.npz`, seed 20262016, subsampled to 50,000 rows, margin 1.1) and refuses unless it reproduces the committed digest. Only after both checks does it export the init actor the same way. It writes `init_actor.hpp`, `init_actor.npz` and `MANIFEST.json` (source sha256 and origin, calibration sha256, agreement) into `inference/agent_v2_controls/`.

## Tests

[`tests/test_next_controls.py`](../../../tests/test_next_controls.py) bundles each control with `bots/p_only/tools/bundle.py` and compiles it together with [`tools/screens/next_controls/probe.hpp`](../../../tools/screens/next_controls/probe.hpp). The probe checks that every team count 0–16 routes to the expected actor with `outExp` 16. It then plays the bundle's own `agent_v2_bot::run()` on the recorded streams n02, n04 and n15: valid orders, nothing refused or inconsistent, the identity line. Finally it measures INT16 agreement: the integer joint the bundle computes on those 1,200 observations against the FP32 weights it was exported from. These tests play no games and measure nothing about timing or strength.
