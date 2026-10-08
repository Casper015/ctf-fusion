"""Phase A: evaluate every candidate opponent bot, admit an open training pool, freeze evaluation rosters.

    python -m roster.inventory   --ai-test AI_TEST --out data/roster/bot_inventory.json
    python -m roster.probes      --raw data/raw --out data/roster/probe_suite.json
    python -m roster.fingerprint --suite data/roster/probe_suite.json --inventory data/roster/bot_inventory.json
    python -m roster.distance    --suite ... --fingerprints data/roster/fingerprints --out data/roster/behavior_distance.json
    python -m roster.strength    schedule | run | aggregate         (matches: target host only)
    python -m roster.assess      --inventory ... --distance ... [--strength ...]
    python -m roster.registry    template | build | freeze-eval    (open training pool, frozen evaluation roster)
    python -m roster.catalog     --assessments ... --distance ... --strength ... --review roster_review.json   (optional panel)
    python handoffs/diverse15_distance_tiered_policy/tools/select_roster.py --input catalog.json --out selected15.json
    python -m roster.pool        --selected selected15.json --inventory ... --out profiles/pool_selected15.json

See docs/ROSTER.md. Static inspection and fingerprints are mechanism evidence;
only the strength screen and confirmation matches are strength evidence.
"""
