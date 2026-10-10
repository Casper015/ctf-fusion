# bots/hybrid/

The two carriers, C08 `Player5_Omega_LIT` and `Player5_v12`, with the mixture integration: the legacy forecast q_H blended with the trained policy P (and optionally value V) through `inference/`.

- `<carrier>_Fusion.hpp`: the integrated carrier. It includes `models*/registry.hpp` from `inference/`, so build with `-Iinference`.
- `<carrier>_p_{s,m,l,xl,xxl}.cpp`: a thin wrapper per policy size; `_control.cpp` is the unmodified carrier; `_pm_no_v`, `_pm_with_v`, `_pvl` are the mixture and value variants.
- `bot_baseline.cpp`, `bot_p_s.cpp`: minimal reference bots.

The P-only descendants, with every hand-written opponent prediction removed, are built from these headers by `p_only/tools/derive/`; a test re-derives them. Editing a `*_Fusion.hpp` therefore means re-running the derive step (`p_only/README.md`).
