# Own-action scorer at run time

How a trained own-action scorer reaches a carrier, and what the host must check before any claim. Written for tasks T0042–T0045 (2026-10-14), which added code and synthetic tests only. **No scorer has been trained, exported for deployment or played in a game.** The plan and the offline evaluation are in [RL_PLAN.md](RL_PLAN.md).

## The pieces

| Stage | Files | What it guarantees |
|---|---|---|
| Diagnostics (T0042) | `training/decision_dataset.py`, `training/decision_noise.py`, `training/evaluate_decisions.py` | Dataset schema `ctf-fusion-decision-2`: repeat ids kept, failed repeats kept, faults by phase. Noise statistics over one population. |
| Features (T0043) | `inference/own_action.hpp`, `tools/cpp/own_action_check.cpp` | The C++ encoder equals `fusion/decision_features.py` bit for bit on recorded roots: canonical orders, flags, duplicates, support, every float. Version `own-action-1`, digest `9ed719f238cd5d70`. |
| Export and inference (T0044) | `training/export_decision_scorer.py`, `inference/own_action_scorer.hpp`, `tools/cpp/own_action_scorer_check.cpp` | Kind `own_action_scorer` only, never P or V. The export binds features, run hashes, the frozen selection (λ, normalization) and the carrier configuration. C++ Q equals numpy float64 within 1e-9. Float weights; no quantization. |
| Hook (T0045) | `bots/p_only/include/decision_score.hpp`, `bots/p_only/tools/derive/make_scoring.py` → `bots/p_only/bots/v12_ponly_fs.hpp` | `PONLY_DECISION 2` in the v12 scoring carrier: the offline rule on the carrier's own shortlist (below). Off and λ = 0 give the family carrier's replies. |

`decision_hook.hpp`, `v12_ponly_f.hpp`, the log-only carriers and the collectors are unchanged: T0037's frozen identities hash them. The scoring carrier is a separate generated file, so `PONLY_DECISION 2` on `v12_ponly_f.hpp` still refuses to build. C08 has no scoring carrier yet.

## What the hook does each turn

1. **Shortlist:** the orders the carrier scored this turn, in the order the log-only carrier would log them (the reply first, then by the deciding stage's score, at most `PONLY_DECISION_K` = 8).
2. **Deciding scores,** as `decision_branch_collect.deciding_scores` reads them from the log, at the log's 9 significant digits: `refine` uses the refined scores, `beam`/`choose` the beam scores. Every other stage (`cut`, `v_cut`, none), an unscored reply, or K = 1 keeps the reply.
3. **Candidates,** as `decision_branch_collect.select_candidates`: the reply, then the `PONLY_DSCORE_ALTERNATIVES` (3) best others by (−score, order text).
4. **Canonical orders, duplicates and support** from the root and its recorded history (`own_action.hpp`).
5. **The rule** of `training/evaluate_decisions.apply_rule` for one root:
   - g = Q(c) − Q(reply), undefined outside the support and for every alternative of an unsupported reply;
   - S is `zscore` (population z-scores of the carrier's scores over the rule's candidates, with the reply raised to their top) or `incumbent_override`;
   - the hook picks argmax S + λ·g; ties go to the reply, then to the smallest canonical key.
   The only difference offline is that a candidate there also needs an observed outcome.

**Fallbacks keep the reply:** an invalid model (`validate`), a stage without deciding scores, K = 1, or the deadline. The deadline is `PONLY_DSCORE_DEADLINE_MS` after the turn began, default 40. That is not a free budget: the carrier's own budgets are unchanged, and only the host can say whether 40 is safe.

**Build-time identity:** the scorer must be bound to `PONLY_DSCORE_DATA_CARRIER`, the collection carrier whose shortlist this build reproduces, with the same opponent-P family and V setting. Otherwise the build fails. An unbound test export builds only with `PONLY_DSCORE_ALLOW_UNBOUND`.

**Counters:** one stderr line at the last turn: `DSCORE roots scored switched no_scores k1 deadline invalid candidates unsupported`.

## Host handoff

Nothing below has run. Take it as a separate host task, after T0038's dataset audit and a separately authorized training task. That task produces a run and a frozen `SELECTION.json`, with the primary model and the negligible margin set before test.

1. **Toolchain parity on Windows** (the Mac ran these with clang and g++-16 only):
   ```bash
   python -m unittest tests.test_own_action_parity tests.test_own_action_scorer
   python -m unittest bots.p_only.tests.test_decision_scoring bots.p_only.tests.test_decision_hook
   ```
2. **Export** the selected run:
   ```bash
   python -m training.export_decision_scorer --run runs/<run> --selection <SELECTION.json> --model-name <name> \
       --out inference/own_action_scorers/<name>
   ```
   Then check the exported header against Python on the dataset's own roots with `tools/cpp/own_action_scorer_check.cpp`, in the way `tests/test_own_action_scorer.py` does.
3. **A deployment variant,** next to `v12_ponly_xxl_dlog.cpp`, with the same tier and V setting as the carrier that collected the data:
   ```cpp
   #define FUSION_TIER_XXL 1
   #define PONLY_DECISION 2
   #define PONLY_DSCORE_DATA_CARRIER "v12_ponly_xxl_dlog"
   #include "own_action_scorers/<name>/scorer_model.hpp"
   #include "v12_ponly_fs.hpp"
   ```
   An A/A check on recorded inputs: with a λ = 0 export the variant must reply exactly as `v12_ponly_xxl`.
4. **The 50 ms validation,** under `lock cpu`, at representative concurrency including 15 teams:
   - the per-turn reply time distribution (p99, max) and late replies of the scoring variant against the unchanged control, with `PONLY_STATS` for the `DSCORE` counters;
   - a predeclared rule for the deadline setting.
   Development-machine timings are not evidence.
5. **The equal-full-turn-time paired strength screen** on fresh maps, with a newly reserved unused seed:
   - the scoring variant against the **unchanged production control** `v12_ponly_xxl`, plus an **A/A** copy of the control;
   - paired fixtures, every fault and timeout kept, the actual valid denominators, and a 95% interval clustered by map;
   - only this screen can establish a deployed improvement. Lower offline loss, more switches or a positive offline Δ are not strength claims.
