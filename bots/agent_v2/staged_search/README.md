# agent_v2 staged search (T0087)

An opt-in variant of the agent_v2 value search that keeps usable completed work. It is not screened and not a production default; `agent_v2_p2xxl_direct` stays the default.

**Why.** In the T0081 screen the gated 8x4 search played its tree on under half of its turns from 6 or 7 teams up, and on about 2% at 15 teams (T0084, [EVIDENCE_AUDIT.md](../../../reports/host/agent_v2_p3_search/EVIDENCE_AUDIT.md)). The deployed search (`agent_v2::Search`) works root by root, so any refusal before the last trajectory throws the whole tree away. T0086's Gate D, recomputed from the receipts by T0092, permits this experiment. Gate A (a learned scorer) is not permitted, so nothing here changes how candidates are scored.

## What changes, and what does not

`agent_v2::StagedSearch` ([inference/agent_v2/search.hpp](../../../inference/agent_v2/search.hpp)) runs the same tree as `Search`. It keeps the roots, the scenarios and streams, the trajectories, the critic, the gate and tau, the admission estimates (`max(prior, worst seen) x 1.5`) and the reply reserve. Only the order of the work and what an interruption keeps are new:

- **Frozen schedule.** Checkpoints `{k, s}` are fixed at compile time: compare the first k roots (the actor's joint first) over scenarios `0..s-1`. Both k and s never decrease, each step raises one of them, and the last checkpoint is the whole `kCandidates x kSamples` tree (a `static_assert`).
- **Scenario-major work.** Each round gives every root of the checkpoint its next scenario, so every compared root has the same completed scenarios. The search never compares partially evaluated roots or roots with unequal sample counts.
- **Immutable decisions.** A completed checkpoint's decision is fixed: `gated_choice` over its means, summed in scenario order. Later partial work never revises it.
- **Interruption.** The search can stop on a refused admission (setup, trajectory, ply-2 expansion, critic), on the deadline, or at the fixed-work cap. It then plays the last completed checkpoint's decision, or the default order (resolved before the search) if no checkpoint completed. Trajectories finished after that checkpoint are discarded and counted. A trajectory that ends after the deadline is discarded too; returning an earlier result does not undo time already spent.
- **Matched control.** A run that completes the schedule decides exactly as `Search`'s whole tree, with the same means bit for bit.
- **Out of scope, on purpose:**
  - no cache, because each trajectory depends only on (tree, root, scenario);
  - no change to the admission cost estimator;
  - no one-ply screening or two-ply refinement split;
  - no change to P, the actor, the critic, the resolver, the features, tau or the objective.

`Search`, every existing variant and the production default are unchanged; the change to `search.hpp` is additive.

## Variants and controls

Build all four with `python bots/p_only/tools/build.py --variants bots/agent_v2/staged_search/variants.json agent_v2_staged_p2_r8x4_fp32 agent_v2_staged_p2_w4to8x4_fp32 agent_v2_p3exp_p2_gw8x4_fp32 agent_v2_p3exp_p2_direct`.

| Variant | Role | Schedule |
|---|---|---|
| `agent_v2_staged_p2_r8x4_fp32` | staged, full-width rounds | 8:1, 8:2, 8:3, 8:4 |
| `agent_v2_staged_p2_w4to8x4_fp32` | staged, widening | 4:1, 4:2, 8:2, 8:3, 8:4 |
| `agent_v2_p3exp_p2_gw8x4_fp32` | control: the same tree, all or nothing (T0081 arm) | - |
| `agent_v2_p3exp_p2_direct` | control: the same carrier without search (T0081 arm) | - |

All four use general P2 XXL, the aligned `p2_large` actor and the FP32 critic, with 8 candidates, 4 scenarios and tau 0.002. Rounds against gw8x4 isolates "keep completed rounds". Widening against rounds isolates the schedule.

## Counters

The first stderr line is the identity: `... staged schedule <k>:<s>,... gate tau .. candidates .. samples .. fixed_work <n>`. The `-1` value of `fixed_work` is the timed mode.

Every later line starts like `bots/agent_v2/bot.hpp`'s line, so existing parsers (such as `tools/screens/p3_search_host/audit_host.py`) read it:
- `search_used` counts turns decided by a completed checkpoint;
- `search_fallback` counts turns that played the default order because no checkpoint completed.

The staged-specific counters follow:

| Counter | Meaning |
|---|---|
| `checkpoints`, `complete` | Completed checkpoints, and turns that finished the whole schedule |
| `eff_candidates`, `eff_scenarios` | The decision checkpoint's k and s, summed over decided turns. Divide by `search_used` |
| `overrides`, `baseline_kept`, `gate_kept` | Decisions that replaced the actor, decisions that kept it, and kept decisions where a root beat it by at most tau |
| `discarded` | Trajectories finished after the last completed checkpoint |
| `refuse_setup`, `refuse_trajectory`, `refuse_expand`, `refuse_critic`, `refuse_late`, `stopped_work_cap`, `complete` | Why each turn's work stopped. These seven add up to `turns` |

## Offline checks (any machine)

```
python -m unittest discover -s tools/search_diagnostics/staged/tests -t .
```

The harness [tools/search_diagnostics/staged/staged_check.cpp](../../../tools/search_diagnostics/staged/staged_check.cpp) runs in fixed-work mode, with no clock, on both schedules and on the 2-, 4- and 15-team fixtures. It checks:
- **Full schedule equals `Search`:** same means bit for bit, same decision.
- **Branch isolation:** every cell has the same value when the cells run in reverse order on another simulator, and the runtime's prediction, history and default order are left unchanged.
- **Every interruption point:** every work cap from 0 to all, and a forced refusal of each kind at every trajectory, gives exactly the last completed checkpoint's decision, or the actor's joint before the first. The discarded work and stop causes add up.
- **NaN:** a NaN at any trajectory keeps the actor in every checkpoint containing it.
- **Nondefault rules:** a 125-turn game with hp 150 and damage 40, which reaches terminal leaves.
- **Determinism:** fixed-work runs repeat exactly.
- **Bundled bots:** the variants built as the host builds them play 400 valid replies, and their counters add up.

Nothing here is timing or strength evidence. On the development machine, the gw8x4 control saw one 41 ms critic call in a replay of the n04 fixture and refused setup on 315 of its 400 turns; its admission estimate keeps the worst time seen in the game. Whether the sticky estimate drives fallbacks on the host is a hypothesis for the preflight to measure, not a finding. Changing the estimator would be a separate arm.

## Host preflight runbook (desktop-g60ojud)

This is a separate host task: create it with `task.py new`, `needs: [host, ai-test]`. It is a design, not results.

1. **Freeze.** Build the four variants above on the host and record the bundle, binary and source hashes from `build/p_only/manifest.json`, the ctf-fusion and AI_Test commits, and the engine hash.
2. **Fixed-work identity on the host.** Run `python -m unittest discover -s tools/search_diagnostics/staged/tests -t .`. It must pass before any timed run.
3. **Seeds.** Reserve fresh map, flag and scenario seeds in a new ledger under `docs/` before any game. Check derived-seed collisions against every ledger (`docs/*SEEDS*.md`), and never reuse T0081's 20262151 or T0086's 20262160.
4. **Timed preflight, no strength claim.** Run under `python tools/repo/task.py lock cpu --task <id> -- ...` with exactly 8 game workers, GPU training stopped, and 2,000/50 ms complete replies. Use a few maps per team count 2-15 and all four variants. Report per count:
   - completed checkpoints per decided turn, mean `eff_candidates` and `eff_scenarios`;
   - the fallback share and the stop causes;
   - discarded work;
   - p99 and max reply time, faults and late replies.
   
   Treat a missing counter as unavailable, never as zero. Set the stop rules before the run: any fault or late reply attributable to the staged arms, or a p99 above the gw8x4 control's, stops the task with a report.
5. **Screen (only if the preflight passes).** Run a paired screen of the four arms plus an A/A copy of `agent_v2_staged_p2_r8x4_fp32`:
   - map clusters, equal weight per count;
   - primary contrast: rounds minus gw8x4; secondary: widening minus rounds, and rounds minus direct;
   - a predeclared multiplicity rule and fixed budgets.
   
   No default promotion follows from it. Any promotion needs its own release gate on fresh seeds and an untouched roster.

## Decision record

T0087 implements the opt-in search, its frozen variants and controls, the offline checks and this runbook. It does not run games. The final screening waits on the host task in step 5. If that task is not created, screening goes ahead without the staged arms; T0087 does not hold it back.
