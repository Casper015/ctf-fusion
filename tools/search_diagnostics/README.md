# agent_v2 branch diagnostics (task T0085)

This folder measures what agent_v2's root candidates are worth when every candidate is played to the end of the game. It measures that the same way for every candidate, under one frozen continuation policy. It also checks the simulator's mechanics against the engine and P's predictions against what opponents actually did.

Every number these tools write is computed from rows. When evidence is absent, the output says "unavailable" and gives the reason.

Nothing here is a strength screen or training data for a deployed model. The host collection that uses this harness is task T0086.

## What each part does

| File | Role |
|---|---|
| `native/diag_tool.cpp` | Replays a seat's recorded stream through the real agent_v2 runtime (the direct bot's code path). `root`: at turn t it prints the replies of turns before t; the actor's scores, feasible mask and sampled joint; the complete `assign_v1` default order; and the search tree's roots, each with its complete order and the critic's per-scenario values. `pmarg`: P's movement, action and target distributions at every factual state. `gate`: the frozen search gate's offline decision at every turn. |
| `native/continuation_bot.cpp` | The frozen direct continuation policy π_c: the production direct bot plus one switch, an actor key applied from turn t + 1. With key 0, and before that turn, it answers exactly like the production binary (tested reply by reply). |
| `packages.json`, `build.py` | The packages that can be built: the production default D (`p2xxl_direct`), general P2 (`p2general`) and general P3 (`p3general`). Both programs are bundled into one source each, whose SHA-256 is their identity. |
| `extract.py` | Selects roots outcome-blind and extracts them. A root is kept only when the package reproduces the seat's recorded replies before t and its default order equals the recorded reply at t. The baseline is therefore the order actually sent, not the prior's top joint. Strata: `unconditioned`, `contact`, `carrier`, and `override`, where the frozen gate (offline, fixed work, `diag_tool gate`) would replace the actor. Candidate sets are nested prefixes: `primary` (8) and `extended` (24), both starting with the baseline. Each root records its public context: score gap, carriers and contact. |
| `branch.py` | The faithful protocol `faithful-replay-v1`. The engine is rebuilt from the match's seeds, and every bot is a fresh process fed its exact causal prefix. Any reproduced block or reply that differs from the record refuses the branch. At t, the opponents' recorded commands are common to every candidate; the environment key reseeds the engine's refresh RNG, and our seat plays the candidate. From t + 1 everything is live to turn 400. |
| `seeds.py` | Registered scenario seed and common random keys. The seed must be reserved in the run's ledger and absent from every other `docs/*SEEDS*.md`. Repeat k of a root uses the same `env_key` and `actor_key` for every candidate; repeats use independent keys. The opponents' own randomness is not controlled, and each row says so. |
| `collect.py` | Writes an immutable `MANIFEST.json` (collector identity), then one row per root, bound to the root record's hash. Resuming keeps only equal rows. Exactly 8 workers, unless `--dev` (the manifest then says not evidence). |
| `p_eval.py` | Factual P fidelity per team count and event (move, act, target): count, NLL, top-1 accuracy, 10-bin ECE, and every exclusion and relabelling by reason. Probability vectors are validated; a label with probability 0 makes that NLL unavailable. |
| `parity.py` | The native simulator step (`tools/cpp/agent_v2/check.cpp step`) against the engine on identical full commands with controlled refresh outcomes. It passes only with zero differences and the event coverage minimums met. |
| `analyze.py` | Per root: signed paired gains over the baseline; the in-sample best gain (labelled biased); shortlist regret against the best observed mean; average ranks with ties; a disjoint-fold cross-fit gain (the baseline may be selected); the critic's rank correlation with outcomes; and the frozen gate's pick and its gain. Aggregates per stratum and exact count, by source-map cluster, with counts weighted equally. A standard error exists only when every count has two or more maps. |
| `estimator.py` | Exact episode counts for T0086's registered design. Wall time only from measured branches. |
| `tests/` | `test_unit.py` (no compiler) and `test_integration.py`. The integration tests check the native programs against the production binary, the 8×4 prefix of the tree, faithful and keyed branches, refusal of a diverging process (`chaos_bot.py`), of a changed record and of a changed source file, and parity. |

What the protocol does not do:
- It does not restore a process from memory. It replays the record, and refuses when a replay diverges.
- It never reuses a factual future reply on another branch.
- It never feeds hidden state to a model.

Our continuation process keeps no state of the order it computed at t: `History` holds observed states only (`inference/agent_v2/core.hpp`), and the encoder is `const`. So forcing candidate c at t leaves its memory consistent with the branch.

## Runbook (host, task T0086)

Run from the repository root, with `AI_TEST_ROOT` set and every evidence step under the CPU lease. The seeds and the selection salt are T0086's: reserve them in `docs/SEARCH_BRANCH_DIAGNOSTIC_SEEDS.md` before any game.

1. **Build** the package and record its identity:

   ```bash
   python -m tools.search_diagnostics.cli build p2general --out build/search_diagnostics
   ```

2. **Play the source games** with the package's production variant in the designated seat. Use a screen profile with `keep_matches` (`bots/p_only/tools/run_screen.py`), whose records name the seat `@candidate`. Copy the kept `nXX/` folders to a sources directory.

3. **Extract roots.** Refused roots are listed with their reasons in `ROOTS/SELECTION.json` and are not replaced:

   ```bash
   python -m tools.search_diagnostics.cli extract --matches SOURCES --build build/search_diagnostics/p2general.json \
       --plan unconditioned:1,contact:1,override:1 --salt <reserved salt> --out ROOTS
   ```

4. **Smoke** with at most 16 branches (for example 2 roots of different team counts, `--repeats 1`), on its own ledgered seed and its own output directory. Then estimate from it; with fewer measured counts than registered, `--model linear-teams` fills the rest, labelled as modelled:

   ```bash
   python -m tools.search_diagnostics.cli estimate --measured SMOKE_ROWS --model linear-teams --markdown
   ```

5. **Collect**, once per candidate set:

   ```bash
   python tools/repo/task.py lock cpu --task T0086 -- python -m tools.search_diagnostics.cli collect --roots ROOTS \
       --matches SOURCES --build build/search_diagnostics/p2general.json --set primary --repeats 4 \
       --scenario-seed <reserved> --seed-ledger docs/SEARCH_BRANCH_DIAGNOSTIC_SEEDS.md --workers 8 --out ROWS
   ```

   For the 24-candidate subset, run it again on the subset's roots with `--set extended` and its own output directory.

6. **Parity** and **P** on the same sources:

   ```bash
   python -m tools.search_diagnostics.cli parity --seed <reserved> --out PARITY.json
   ```

   ```bash
   python -m tools.search_diagnostics.cli peval --matches SOURCES --build build/search_diagnostics/p3general.json --out P3.json
   ```

   Run `peval` once for each package.

7. **Analyze** with T0086's map splits (`{"<match_id>": "train" | "validation" | "holdout"}`):

   ```bash
   python -m tools.search_diagnostics.cli analyze --roots ROOTS --rows ROWS --splits SPLITS.json --out ANALYSIS.json
   ```

Outputs are never overwritten, and a collection with another identity is refused.

## Tests

```bash
python -m unittest discover -s tools/search_diagnostics/tests -t .
```

The integration tests need a C++ compiler; their engine part also needs `AI_TEST_ROOT`. Without these they skip, which is reported. The tests live in this folder because `tests/` belongs to other tasks' lanes, so CI does not run them.
