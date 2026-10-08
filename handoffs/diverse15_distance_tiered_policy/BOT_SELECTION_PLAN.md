# Open training opponents and frozen evaluation panels

**Updated requirement:** the training pool has no fixed 15-bot cap. Reliable policies with stable behavioral differences, new state/event coverage or useful matchup responses can all contribute. Keep a separately frozen evaluation roster; a 15-member core panel is an optional comparison budget, not a training admission rule. No final pool or panel is asserted in this package.

## Inventory and evidence

Use the current `roster.inventory` tools and extend discovery to every new export. The handoff's `tools/inventory_bots.py` remains a lightweight source scanner. Neither inventory source grouping nor successful compilation proves style equivalence or strength. Inventory output is not a measured selector input.

For every source candidate, record repository-relative path, commit/source/dependency hashes, build options, binary identity, inherited strategy lineage, entrypoint/build/protocol status, and assessment status. Source aliases with different dependencies or macros are not automatically interchangeable. Freeze the original official competitors and trustworthy controls as the strength reference.

Compile/legality and limited behavior screening can precede expensive confirmation. Each genuinely different bot's final confirmed strength assessment still needs at least 10,000 independent map groups, a common paired schedule, all 14 team counts, and latency/fault reports. Provisional exploratory collection may precede that confirmation, with its evidence level and contribution clearly marked; it must not be reported as confirmed strength. Failed/inconclusive candidates stay in the catalog with reasons. Evidence from an equivalent representative must be explicitly attributed, not counted as a new experiment.

Strength fields should include official frozen BT and its interval, plus transparent per-count pairwise utility and paired changes against controls. A candidate with specialized value may be weaker overall; describe the specialist behavior and validate that it adds coverage. P predicts what an opponent does, so a weak but reliable policy can be useful. Randomness without learnable structure, protocol errors and cosmetic variants do not satisfy uniqueness.

## Fingerprints and distances

Use the same public-state/history probe set across candidates. Measure requested joint moves, pickup/drop/attack decisions, target identity, role assignment and policy changes near flags, carriers, respawns, bottlenecks and endgame. Include all counts; identical duel behavior does not imply identical multi-team behavior. Treat probe clock noise as A/A variation.

A possible normalized distance is a pre-registered weighted average of joint-move disagreement, conditional-action disagreement, target disagreement when eligible, and event-conditioned role disagreement. First average within an independent map/history group, then across team counts/event strata. Document denominators and target coverage. Missing target opportunities are missing evidence, not automatic agreement. Static mechanism tags support the explanation but do not replace measured fingerprints.

Group byte aliases and empirically equivalent policies conservatively. A behavior cluster describes the actual probe evidence and range of counts, not a universal equivalence proof. Thresholds should exceed observed A/A noise. Manually investigate close pairs selected for rare specialized behavior.

Follow [BOT_GROUP_AND_SAMPLING_REVIEW.md](BOT_GROUP_AND_SAMPLING_REVIEW.md) when constructing profiles: preserve count/event specialists; bound group diameter rather than rely only on connected chains; distinguish stochasticity from equivalence; retain joint-team correlations; and treat token-only aliases as provisional until compiler-aware identity and behavior are checked.

## Admission to the open training registry

Record a source/dependency/build identity, stable protocol behavior, meaningful contribution, evidence level, allowed counts/contexts, lineage, behavior/response profiles and sampling rules for every admitted policy. A weaker predictable specialist can qualify. Source changes trigger re-review. Cosmetic copies share the original's mass; a new parameter or module name alone does not earn equal sampling weight.

Admission is independent of a fixed-size optimization. Do not reject an otherwise useful specialist because a global distance is below 0.05 or a source family already contains three entries. Lineage quotas are redundancy regularizers, not universal limits. Explain exceptions with count/context evidence. Conversely, high noise or prediction loss from random actions is not automatically useful novelty.

Maintain separate, versioned artifacts:

- `profiles/open_training_pool.json`: extensible membership, per-count/context weights, evidence and source identities.
- `profiles/frozen_eval_roster.json`: fixed opponents, weights, source/build identities and fixture policy for a comparison.
- reserved OOD lineages and maps: outside all training/tuning data for a strict unseen-family claim.
- admission/sampling manifests: changes, reasons, target distributions and dataset revision hashes.

These filenames describe the target implementation; the present fixed-panel pool writer has not yet produced this open-registry workflow. Freeze every training run's membership and sampling identity even though later dataset revisions can admit additional policies.

## Optional fixed-size reference panel

`tools/select_roster.py` consumes a completed evaluation catalog, never raw inventory. It remains useful for a budget-limited reference panel, defaulting to 15 candidates with distinct behavior clusters, minimum pair distance, family quotas, independent-map confirmation and contribution explanations. It is not the open training-pool admission gate. The current global-distance/cluster inputs need the safeguards described above before the panel is trusted.

The illustrative objective combines mean normalized strength lower bound, facility coverage of the eligible behavior population, and coverage of verified skill tags. Behavior population weights are balanced by cluster so repeated files cannot multiply coverage importance. Default objective weights and distance/family limits are starting choices; tune them before the final confirmation, then freeze them. The greedy multi-start search is a heuristic and does not prove optimality or infeasibility.

Input format:

```json
{
  "schema": "ctf-roster-evidence-1",
  "evidence_manifest_sha256": "64 hexadecimal characters",
  "bots": [
    {
      "id": "repository-relative unique bot identity",
      "source_sha256": "64 hexadecimal characters",
      "lineage": "verified strategy lineage",
      "behavior_cluster": "verified probe cluster",
      "eligible": true,
      "independent_maps": 10000,
      "maps_by_count": {"2": 718, "3": 714, "4": 714, "5": 714, "6": 714, "7": 714, "8": 714, "9": 714, "10": 714, "11": 714, "12": 714, "13": 714, "14": 714, "15": 714},
      "quality_lcb": 0.61,
      "skills": ["verified carrier interception"],
      "unique_skill_evidence": "Exact probe/report references and distinctive contribution"
    }
  ],
  "distance_ids": ["same bot ids in distance-matrix order"],
  "distances": [[0.0]]
}
```

`quality_lcb` is a normalized [0,1] lower confidence bound for utility against the same frozen reference schedule; it is not a raw BT number. `maps_by_count` must cover 2–15, with at least 500 groups each by default and totals matching `independent_maps`. All catalog values must derive from real confirmation data. The shown numbers/schema are illustrative only. Keep underlying per-count reports and proof of common fixtures in the evidence manifest.

For an optional panel, run `python3 tools/select_roster.py --input evidence.json --size 15 --out frozen_eval15.json`. Defaults: 15 slots, at least 10,000 independent map groups per candidate, minimum pair distance 0.05, at most three bots from one lineage. Those are configurable panel choices, not training-pool constraints or learned facts. A failed panel search does not mean no useful training data can be collected. Keep its output pending manual review and freeze evaluation identities separately.

Review the panel manually, inspecting close pairs, specialists and lineage. Compare open behavior-weighted collection, file-uniform collection, lineage-balanced collection and a fixed-15 baseline at equal data/training budgets on held-out maps/lineages. That downstream ablation determines whether admission and sampling improve P/V; panel diversity alone is not evidence of better training.

## Collection integration

Implement the open registry and explicit hierarchical schedule. Sample team count, matchup regime, behavior profile and version, with source lineage controlling redundant exposure. A mixture of representative strong-field, style-balanced and verified hard-case sampling can be tuned; proposed 0.6/0.3/0.1 weights are an initial hypothesis, not proven optimality. Main confirmation uses a frozen representative roster; persistent random opponents remain separately labeled augmentation/stress.

Keep group mass independent of the number of copied versions. Preserve per-count/context specialists and explicit rare-event coverage. Track actual inclusion probabilities and chosen continuing policies. Equal games per count do not imply equal training mass when every team contributes rows.

At 15 seats, a 15-bot reservoir sampled without replacement always gives the same members. Use the wider open reservoir or deliberate repeated-policy instances in designated training regimes to vary composition. Source roles, roster sizes and replayed seat variants remain explicit in the manifest. Adding Noise automatically changes the distribution and must be an explicit collection choice.

Do not feed bot names/source ids as runtime-observable features. If a wider pool makes prediction ambiguous, test public-history descriptors or measured adaptation with the same offline/runtime encoder. The shared P models remain organized by supported team count/capacity; pool size does not require a separate deployed model per opponent.

Retain requested actions, public observations, resolved events, seeds, map identities, source identities, fault status and sampling probabilities. P labels from malformed/missing replies remain excluded and exclusion rates are reported. V's label meaning must retain its continuing-policy and opponent-distribution contract. Freeze the training registry, data and evaluation roster while comparing architectures/tiers. Admit/reweight/retire candidates between versioned runs using tuning evidence, then confirm on a fresh independent set.
