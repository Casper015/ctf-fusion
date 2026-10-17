# P3 opponent prediction in agent_v2 search

This is an experiment replacing the opponent predictor, not a new search algorithm or a trained own policy.

The existing `models_pf_p3xxl_vm` bank serves the general P3 XXL model from T0011. Its 125-way joint movement, P3 combat features, conditional action and attack-target heads run through the generated `ponly_p3.hpp` implementation. Both real-root and successor views receive the current INIT rules before combat encoding. There is no original Meowmind prediction mixture. The bank's paired V is never read: all search variants explicitly include the same aligned FP32 critic as the P2 control.

The main comparison uses **general P2 XXL versus general P3 XXL**, both exported from the T0011 seed-1 feature-family study (`models_pf_p2xxl_vm`, `models_pf_p3xxl_vm`). Both registries serve one general policy at all counts. The older count-routed P2 package remains an external deployment reference through T0079, not the primary P2 control. The family comparison includes different feature inputs, learned weights and calibrated heads; it is not a comparison with the newer CSP specialists. P3 is not the separately embedded distilled P3 model used by other Meowmind wrappers.

## Variants

`variants.json` contains four new variants. Paired P2/P3 source files differ only in the bank-selection define:

| Suffix | Predictor | Own actor / critic | Search |
|---|---|---|---|
| `p2_direct` | general P2 XXL | frozen T0059 aligned actor; no critic use | off |
| `p3_direct` | general P3 XXL | same actor | off |
| `p2_gw8x4_fp32` | general P2 XXL | same actor and aligned FP32 critic | 8 roots, 4 scenarios, tau 0.002 |
| `p3_gw8x4_fp32` | general P3 XXL | same actor and aligned FP32 critic | same |

The full prefix is `agent_v2_p3exp_`. Resolver scores, rule semantics, scenario salts, actor sampling, search admission priors and deadlines are identical. P3 affects predictions used by BOTH the resolver and search: it is never replaced by P2 at deeper plies. This does not yet test the newer G actor or its critic. No existing default or T0079 arm is changed.

```
python bots/p_only/tools/build.py --variants bots/agent_v2/p3_search/variants.json
python -m unittest tests.test_agent_v2_p3 -v
```

The general banks are selected by `FUSION_TIER_P2GENERAL_VM` and `FUSION_TIER_P3XXL_VM`. Multiple model tiers are rejected. `PredictionKey` uses the actual model type and pointer. Identity output distinguishes `predictor P model p2xxl_general` from `predictor P3 model p3xxl_general`. The existing `training.agent_v2` CLI's bank allowlist is unchanged; P3 PPO training is not part of this experiment.

## Fixed work before equal time

```
python bots/agent_v2/p3_search/fixed_work.py --stream bots/p_only/tests/fixtures/stream_n04.txt.gz --turns 20 80 --out build/agent_v2_p3/fixed_n04
```

The helper uses `check tree` without a deadline. Both predictors see the same full causal stream, actor, root moves, scenario random keys, breadth and critic header. Output roots and simulator successors differ legitimately as forecasts and resolved actions differ. It records the complete tree metadata and native float32 leaf features, with output hashes. Score both sets using the same critic if comparing decisions. It is NOT appropriate to force identical successors when the purpose is to test changed opponent forecasts.

Model self-consistency or a larger predicted value is not proof of game improvement. Do not tune P3 settings on formal test outcomes.

## Host follow-up

An independent host task follows the code PR and the completed T0079 run. It must freeze build/model identities and run exactly eight game workers with GPU training stopped. Preflight all four variants at counts 2/5/8/15, 16 fixtures per count (256 games). Check all-seat faults, complete counters, full reply limits (2 seconds first, 50 ms thereafter), and search completion/override/fallback counts by team count. If P3 rarely/never finishes a search, report the compute limitation rather than attributing the result only to prediction quality. Do not silently narrow the tree or change the deadline; changed budgets require a new named experiment and fresh seeds.

Proposed bounded screen: 56 maps per count 2-15, four arms plus a P3-search A/A copy, 784 paired fixtures / 3,920 games. Prespecified contrasts: P3-search minus P2-search, P3-direct minus P2-direct, P3-search minus P3-direct. Use one Bonferroni family over the three overall comparisons, equal count weight and map-cluster intervals; per-count rows are descriptive. Keep fixed-work and equal-time conclusions separate. A result is not known until host validation finishes.
