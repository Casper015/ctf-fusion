# Selecting exactly 15 useful, distinct opponents

No final roster is asserted in this package. The implementation agent must discover and evaluate the newly written bots; selecting old filenames now would fail the request.

## Inventory and evidence

Run `tools/inventory_bots.py --root /path/to/AI_Test --out bot_inventory.json`. Add `--scan-dir` for new export directories. The utility hashes source candidates and flags byte-identical sources; it does not compile them, infer behavior equivalence or produce strength scores. Inventory output is not selector input.

For every source candidate, record repository-relative path, commit/source/dependency hashes, build options, binary identity, inherited strategy lineage, entrypoint/build/protocol status, and assessment status. Source aliases with different dependencies or macros are not automatically interchangeable. Freeze the original official competitors and trustworthy controls as the strength reference.

Compile/legality screening can precede expensive confirmation. Each genuinely different bot's final assessment needs at least 10,000 independent map groups, a common paired schedule, all 14 team counts, and latency/fault reports. Failed/inconclusive candidates stay in the catalog with reasons. Evidence from an equivalent representative must be explicitly attributed, not counted as a new experiment.

Strength fields should include official frozen BT and its interval, plus transparent per-count pairwise utility and paired changes against controls. A candidate with specialized value may be weaker overall; describe the specialist behavior and validate that it adds coverage. P predicts what an opponent does, so a weak but reliable policy can be useful. Randomness without learnable structure, protocol errors and cosmetic variants do not satisfy uniqueness.

## Fingerprints and distances

Use the same public-state/history probe set across candidates. Measure requested joint moves, pickup/drop/attack decisions, target identity, role assignment and policy changes near flags, carriers, respawns, bottlenecks and endgame. Include all counts; identical duel behavior does not imply identical multi-team behavior. Treat probe clock noise as A/A variation.

A possible normalized distance is a pre-registered weighted average of joint-move disagreement, conditional-action disagreement, target disagreement when eligible, and event-conditioned role disagreement. First average within an independent map/history group, then across team counts/event strata. Document denominators and target coverage. Missing target opportunities are missing evidence, not automatic agreement. Static mechanism tags support the explanation but do not replace measured fingerprints.

Group byte aliases and empirically equivalent policies conservatively. A behavior cluster describes the actual probe evidence and range of counts, not a universal equivalence proof. Thresholds should exceed observed A/A noise. Manually investigate close pairs selected for rare specialized behavior.

## Transparent selection objective

`tools/select_roster.py` consumes a completed evaluation catalog, never raw inventory. It searches for 15 candidates with distinct behavior clusters, minimum pair distance, family quotas, sufficient independent-map confirmation and an explicit useful-skill explanation.

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

Run `python3 tools/select_roster.py --input evidence.json --out selected15.json`. Defaults: 15 slots, at least 10,000 independent map groups per candidate, minimum pair distance 0.05, at most three bots from one lineage. Those are configurable pre-registration choices, not learned facts. The tool reports failure if it cannot find a qualified 15; it must not pad with duplicate behavior.

Review the selected roster manually. Require a distinct contribution sentence for each member, inspect the minimum-distance pairs, record weaker specialist exceptions, and cross-check family lineage. Finally compare data from this pool against the 15-strongest and broader pools under equal training budget on strictly held-out lineages. That downstream ablation determines whether the selection was genuinely useful to P/V.

## Collection integration

Write a hash-locked `profiles/pool_selected15.json` and an explicit collection schedule. Sample strategy family first and version second. Initial training quotas may be 60% strong-field, 30% mixed-style and 10% exploratory-state starts; these are match quotas, not raw file weights. Main confirmation should use the official representative frozen roster where available, and should separate persistent random opponents into stress evaluation.

Retain requested actions, public observations, resolved events, seeds, map identities, source identities, fault status and sampling probabilities. P labels from malformed/missing replies remain excluded and exclusion rates are reported. Noise augmentation affects V's continuing-policy interpretation and must be labeled. Keep the selected roster fixed while comparing model architectures and inference tiers.
