# Phase 1 size rule

Rule: C08 gets the largest P-only size with worst-game p99 <= 40 ms, no fault and no late reply; v12 gets XL and XXL unless one faults or replies late.

- c08_ponly_xxl: worst-game p99 31.198 ms, max 31.442 ms, faults 0, late 0 -> admitted
- c08_ponly_xl: worst-game p99 30.597 ms, max 31.017 ms, faults 0, late 0 -> admitted
- c08_ponly_l: worst-game p99 30.211 ms, max 30.737 ms, faults 0, late 0 -> admitted
- c08_ponly_m: worst-game p99 30.199 ms, max 30.255 ms, faults 0, late 0 -> admitted
- v12_ponly_xl: faults 0, late 0, worst-game p99 8.053 ms -> admitted
- v12_ponly_xxl: faults 0, late 0, worst-game p99 10.55 ms -> admitted

Decision: C08 c08_ponly_xxl; v12 v12_ponly_xl, v12_ponly_xxl.
