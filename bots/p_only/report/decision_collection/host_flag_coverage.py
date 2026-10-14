"""Correct protocol-ground-flag diagnostics on the immutable collected sample.

The original frozen collector used the nonexistent status ``ground``. This
module fixes that predicate for audit/future implementations. It never selects
replacement roots, runs bots, changes the freeze or expands the sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from fusion.protocol import parse_block

CATEGORIES = ('carrier', 'flag', 'contact', 'interception', 'defense', 'quiet')
STATES = {'home', 'dropped', 'carried', 'cooldown'}


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def ground_flags(block):
    """Return protocol ground positions; home and dropped can be picked up."""
    result = []
    for line in block.splitlines():
        fields = line.split()
        if not fields or fields[0] != 'F':
            continue
        if len(fields) != 7 or fields[2] not in STATES:
            raise ValueError('unsupported flag protocol status/shape: ' + line)
        if fields[2] in ('home', 'dropped'):
            xy = tuple(map(int, fields[3:5]))
            if min(xy) < 0:
                raise ValueError('ground flag has unavailable coordinates')
            result.append(xy)
    return result


def categories(init, block, seat):
    units = parse_block(block).units
    own = [u for u in units if u[1] == seat and u[4] > 0 and u[2] >= 0]
    enemies = [u for u in units if u[1] != seat and u[4] > 0 and u[2] >= 0]
    distance = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])
    lines = init.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('BASES '))
    base = tuple(map(int, lines[index+1+seat].split()))
    active = []
    if any(u[5] >= 0 for u in own):
        active.append('carrier')
    if any(distance(u[2:4], xy) <= 4 for u in own for xy in ground_flags(block)):
        active.append('flag')
    if any(distance(u[2:4], v[2:4]) <= 4 for u in own for v in enemies):
        active.append('contact')
    if any(v[5] >= 0 and distance(u[2:4], v[2:4]) <= 8 for u in own for v in enemies):
        active.append('interception')
    if any(distance(v[2:4], base) <= 6 for v in enemies):
        active.append('defense')
    return active or ['quiet']


def audit(repo):
    data = repo/'data/decision/host_20261014'
    build = repo/'build/decision_host_20261014'
    plan = read(build/'PLAN.json')
    schedule = read(build/'SCHEDULE.json')
    original = read(data/'SELECTION_LEDGER.json')
    root_paths = {read(p)['match_id']: p for p in data.glob('roots/n*/*.json.gz')}
    rows = []
    total = {key: Counter() for key in ('opportunity_turns', 'maps_with_opportunities', 'selected_root_contexts', 'requested_target_hits')}
    per_count = {}
    for fx, old in zip(schedule['fixtures'], original):
        match = read(data/'matches'/f"n{fx['teams']:02d}"/f"{fx['fixture_id']}.json.gz")
        root = read(root_paths[fx['fixture_id']])
        assert root['plan_identity'] == plan['identity']
        opportunities = Counter()
        for turn, block in enumerate(match['blocks'], 1):
            if turn >= 6:
                opportunities.update(categories(match['init'], block, fx['seat']))
        contexts = categories(match['init'], match['blocks'][root['turn']-1], fx['seat'])
        requested = old['requested_stratum']
        row = {'match_id': match['match_id'], 'teams': fx['teams'], 'map_seed': fx['map_seed'], 'seat': fx['seat'], 'turn': root['turn'], 'original_requested_stratum': requested, 'original_selected_stratum': old['selected_stratum'], 'original_fallback': old['fallback'], 'corrected_opportunity_turns': dict(opportunities), 'actual_selected_root_contexts': contexts, 'requested_target_actually_present': requested in contexts, 'root_changed': False}
        rows.append(row)
        metrics = per_count.setdefault(str(fx['teams']), {key: Counter() for key in total})
        for target in (total, metrics):
            target['opportunity_turns'].update(opportunities)
            target['maps_with_opportunities'].update(opportunities.keys())
            target['selected_root_contexts'].update(contexts)
            if requested in contexts:
                target['requested_target_hits'][requested] += 1
    requested_counts = Counter(row['original_requested_stratum'] for row in rows)
    summary = {key: dict(value) for key, value in total.items()}
    summary['requested_target_hit_rates'] = {key: total['requested_target_hits'][key]/value for key, value in requested_counts.items()}
    summary['selected_context_map_rates'] = {key: total['selected_root_contexts'][key]/len(rows) for key in CATEGORIES}
    result = {'schema': 'ctf-host-flag-protocol-coverage-correction-1', 'plan_identity': plan['identity'], 'source_commit': plan['source_commit'], 'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'defect': 'frozen selector looked for F status ground, but engine protocol encodes ground flags as home/dropped; all56 requested flag selections fell back and the original flag/quiet opportunity labels were wrong', 'method': 'reclassify observable blocks only with corrected protocol predicate; preserve every original root, candidate, repeat, outcome and freeze; contexts overlap; no bots executed and no roots reselected', 'summary': summary, 'per_count': {n: {key: dict(value) for key, value in group.items()} for n, group in metrics_items(per_count)}, 'rows': rows, 'interpretation': 'actual flag-context coverage may exist through fallback/other requested categories, but the predeclared flag-targeted quota was not achieved by the implemented rule; original quiet labels require corrected reclassification. No claim of balanced intended strata or unbiased flag-targeted selection.'}
    (build/'FLAG_PROTOCOL_COVERAGE.json').write_text(json.dumps(result, sort_keys=True, indent=1)+'\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True))
    return result


def metrics_items(per_count):
    return sorted(per_count.items(), key=lambda item: int(item[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=REPO)
    audit(parser.parse_args().repo)
