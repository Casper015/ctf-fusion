# Search gate diagnostic seed ledger (T0076)

| Seed | Use | Status |
|---:|---|---|
| 20262120 | Offline scenario randomness of the T0076 fixed-work trees only ([AGENT_V2_SEARCH_GATE.md](AGENT_V2_SEARCH_GATE.md)) | Reserved 2026-10-17 |

- **Checked before reserving (2026-10-17):** `git grep` over `main` and `git log --all -S20262120` find the value only in the T0076 task text. No seed ledger in `docs/`, `reports/` or `profiles/` uses it.
- **Derivations:**
  - repeat r = 0..3: `salt_r = splitmix64(20262120 + r)`;
  - the reference draw: `splitmix64(20262120 + 1000)`;
  - the bootstrap: numpy PCG64 seeded with the first 8 bytes (little-endian) of SHA-256(`"20262120|bootstrap"`).

  Each salt is mixed into the deployed per-turn tree seed: `splitmix64(turn_seed(game seed, turn) ^ salt)`.
- **Never for games:** no map, flag, seat or schedule seed may be derived from 20262120. The derived integers 20262121–20262123 and 20263120 are salts inside this derivation, not seeds, and are not reserved for anything else.
