# T0025 reproduction

Do not replay a completed fixture or rerun a successful score stage. The retained master `20261250` is consumed. Any independent future collection needs a newly checked seed and new identity.

Use an isolated checkout of measurement source `de5dece1b90206684e06d82617d772e82d6658a3`. Set `AI_TEST_ROOT`, `T0025_TRAIN_INDEX` and `T0025_T28_RESULTS` only at runtime; never write the retained original data. The last input is the 50,000-game `games.jsonl` behind T0028. Use GCC 15.2, Python 3.14, OMP/MKL/OpenBLAS 8 and a scoped Git safe.directory for this checkout. All official host computation uses an exclusive T0025 CPU lease, with no formal GPU job at the same time. Long phases use the passive helper/observer recorded in the launch receipts.

The exact commands were the following, each inside that CPU lease. The collection helper added a passive 15-second observer. Preparation reads a fresh committed seed ledger at `build/t0025/REMOTE_DIGEST.md`.

```powershell
python reports/models/p_ood/inventory_inputs.py
python reports/models/p_ood/audit_training_lineage.py
python reports/models/p_ood/prepare_collection.py
python reports/models/p_ood/collect_ood.py
python reports/models/p_ood/score_ood.py
python reports/models/p_ood/aggregate_ood.py
python reports/models/p_ood/audit_metrics.py
python reports/models/p_ood/package_ood.py
```

The canonical extractor, labeler, legacy forecast parser and metric routines are unchanged. Native scoring takes actual committed registries in separate executables, with scratch width 2048 and unchanged P2 temperatures. The reference observer is the admitted R3_EXPECTIMAX seat. Only valid requests from the two reserved families enter the primary cohort. Conditional act/target heads are not a primary metric in this study; their weights are included in raw parity.

ID-context figures are the existing artifact-matched pure-quantized fields in `reports/models/xl` and `reports/models/xxl`, not their fitted mixture fields. Only original ID validation fits inertia. q_H missing rows would have their own comparison denominator and would never be replaced by a claimed legacy forecast; this run had complete coverage.

All runnable companions below have LF bytes and the shown SHA256. Extract each fenced block to its heading path as an untracked runtime file, including its final newline. Their public Markdown transport does not change measurement inputs.

## reports/models/p_ood/aggregate_ood.py

SHA256: `1542f4c85bc1ff3093e935fc92890a2ebd5c2b2e16132374fc4c33e08d7a6d03`; 7156 bytes.

```python
"""Aggregate frozen integer OOD predictions with independent-map uncertainty."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

P = Path.cwd()
sys.path.insert(0, str(P))
from fusion import metrics
R = P / 'reports/models/p_ood'
B = P / 'build/t0025'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def interval(values, maps):
    mean, se, k = metrics.clustered_mean(values, None, maps)
    return {'mean': mean, 'se': se, 'ci95': [mean - 1.96 * se, mean + 1.96 * se], 'maps': k,
            'rows': len(values), 'method': 'row mean, SE clustered by independent map'}

def calibration(conf, correct, maps, key):
    unique, inv = np.unique(maps, return_inverse=True)
    bins = np.minimum((conf * 10).astype(int), 9)
    shape = (len(unique), 10)
    count = np.zeros(shape)
    csum = np.zeros(shape)
    hsum = np.zeros(shape)
    np.add.at(count, (inv, bins), 1)
    np.add.at(csum, (inv, bins), conf)
    np.add.at(hsum, (inv, bins), correct)
    seed = int.from_bytes(hashlib.sha256(('T0025/ece/v1/' + key).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=2000)
    e = np.abs(draws @ (hsum - csum)).sum(1) / (draws @ count).sum(1)
    return {'ece': metrics.ece(conf, correct), 'ci95': np.quantile(e, [.025, .975]).tolist(),
            'bins': 10, 'method': '2000 independent-map percentile bootstrap replicates', 'statistical_rng_seed': seed}

def main():
    assert read(Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/cpu.json')['task'] == 'T0025'
    assert not (Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/gpu.json').exists()
    assert read(B / 'score_state.json')['stage'] == 'complete'
    receipts = read(R / 'SCORE_RECEIPTS.json')
    plan = read(R / 'COLLECTION_PLAN.json')
    parts, clusters = [], []
    for fx in plan['fixtures']:
        p = B / 'ood_scores' / (fx['match_id'] + '.npz')
        assert sha(p) == receipts['matches'][fx['match_id']]['npz_sha256']
        with np.load(p) as a:
            parts.append(dict(a))
        clusters.append(np.full(len(parts[-1]['nll']), fx['map_seed'], np.int64))
    arrays = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
    maps = np.concatenate(clusters)
    rows = []
    id_reports = read(R / 'BASELINE_FIT.json')['id_test_reports']
    families = sorted(set(arrays['family']))
    for n in range(2, 16):
        for family in [*families, 'both reserved lineages']:
            for active in [False, True]:
                sel = arrays['teams'] == n
                if family in families:
                    sel &= arrays['family'] == family
                if active:
                    sel &= arrays['active_units'] > 0
                assert sel.any()
                cmaps = maps[sel]
                assert len(np.unique(cmaps)) >= 20
                common_qh = np.isfinite(arrays['qh_nll'][sel])
                out = {'teams': n, 'lineage': family, 'active_only': active, 'rows': int(sel.sum()),
                       'maps': len(np.unique(cmaps)), 'qh_common_rows': int(common_qh.sum()), 'sizes': {}}
                for j, size in enumerate(('XL', 'XXL')):
                    loss = arrays['nll'][sel, j]
                    row = {'joint_nll': interval(loss, cmaps),
                        'top1': interval(arrays['correct'][sel, j], cmaps),
                        'calibration': calibration(arrays['confidence'][sel, j], arrays['correct'][sel, j], cmaps,
                                                   f'{n}/{family}/{active}/{size}'),
                        'baselines': {}, 'reported_id_test': id_reports[size + '/' + str(n)]}
                    for baseline in ('uniform', 'inertia', 'qh'):
                        available = common_qh if baseline == 'qh' else np.ones(len(loss), bool)
                        if not available.any():
                            row['baselines'][baseline] = {'rows': 0, 'status': 'unavailable; no uniform substitution'}
                            continue
                        baseline_loss = arrays[baseline + '_nll'][sel][available]
                        delta = interval(loss[available] - baseline_loss, cmaps[available])
                        delta['interpretation'] = 'lower P NLL shown' if delta['ci95'][1] < 0 else (
                            'higher P NLL shown' if delta['ci95'][0] > 0 else 'not shown')
                        row['baselines'][baseline] = {'joint_nll': interval(baseline_loss, cmaps[available]),
                            'P_minus_baseline': delta, 'rows': len(baseline_loss)}
                    out['sizes'][size] = row
                out['XXL_minus_XL'] = interval(arrays['nll'][sel, 1] - arrays['nll'][sel, 0], cmaps)
                rows.append(out)
    result = {'task': 'T0025', 'identity': receipts['identity'], 'seed': plan['seed'],
        'independent_maps': len(np.unique(maps)), 'valid_reserved_rows': len(maps), 'rows': rows,
        'weighting': 'Every canonical valid OOD reserved-lineage turn has weight 1; 400 turns retained per game.',
        'uncertainty': 'Map-clustered normal 95% intervals for row means and paired NLL differences; map bootstrap for ECE. Within-map lineage rows share one cluster.',
        'ID_context': 'Matched deployed artifacts, existing published ID test numbers with original stride/importance-weighted sampling; not a matched-trajectory causal OOD penalty.',
        'limitation': 'Prediction metrics only; no game-strength, deadline latency, training, OOD fit, mixture or export conclusion.'}
    (R / 'OOD_RESULTS.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    lines = ['# T0025 pure deployed P2 on reserved lineages', '',
        'Native integer scores use the unchanged XL/XXL artifacts. Negative paired NLL differences favor P.', '',
        'All raw games and faults are retained. These prediction results do not establish game strength.', '',
        '| Teams | Lineage | Rows | Maps | XL NLL | XXL NLL | XXL − XL (95% map CI) |',
        '|---:|---|---:|---:|---:|---:|---|']
    for row in rows:
        if row['active_only']:
            continue
        d = row['XXL_minus_XL']
        lines.append(f"| {row['teams']} | {row['lineage']} | {row['rows']} | {row['maps']} | "
            f"{row['sizes']['XL']['joint_nll']['mean']:.6f} | {row['sizes']['XXL']['joint_nll']['mean']:.6f} | "
            f"{d['mean']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}] |")
    lines += ['', 'Full baseline differences, calibration intervals, active-only cohorts and matched-artifact ID test sources are in OOD_RESULTS.json.',
              '', result['weighting'], '', result['uncertainty'], '', result['ID_context'], '', result['limitation'], '']
    (R / 'OOD_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps({'rows': len(maps), 'maps': len(np.unique(maps)), 'cohorts': len(rows)}), flush=True)

if __name__ == '__main__':
    main()
```

## reports/models/p_ood/audit_collection.py

SHA256: `03ab2e283aaa1b63a9a75063ff57f0cf397af2ea419b4f16bc7195167756e7bd`; 4066 bytes.

```python
"""Independent complete retained-game audit, without replay or scoring."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from fusion import matchlog
from fusion.protocol import parse_block, parse_reply

def audit(P):
    R = P / 'reports/models/p_ood'
    def read(p):
        return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan = read(R / 'COLLECTION_PLAN.json')
    frozen = read(R / 'FROZEN_COLLECTION.json')
    receipts = read(R / 'COLLECTION_RECEIPTS.json')['matches']
    assert sha(R / 'COLLECTION_PLAN.json') == frozen['collection_plan_sha256']
    assert len(plan['fixtures']) == len(receipts) == 300
    for name, source in frozen['programs'].items():
        assert sha(P / source['binary']) == source['binary_sha256']
    for path, digest in frozen['measurement_sources'].items():
        assert sha(P / path) == digest
    seen, lineage_maps = set(), {}
    totals = Counter()
    allseat = []
    for fx in plan['fixtures']:
        path = P / 'data/t0025_ood/matches' / fx['match_id'][:3] / (fx['match_id'] + '.json.gz')
        assert sha(path) == receipts[fx['match_id']]['raw_sha256']
        m = matchlog.load(path)
        assert m['match_id'] == fx['match_id'] and m['teams'] == fx['teams']
        assert m['map_seed'] == fx['map_seed'] and m['flag_seed'] == fx['flag_seed']
        assert m['map_seed'] not in seen
        seen.add(m['map_seed'])
        assert m['limits_ms'] == [2000, 50] and m['engine_sha256'] == frozen['engine_sha256']
        assert m['sampling']['registry'] == frozen['identity']
        assert [x['name'] for x in m['seats']] == fx['seats']
        assert len(m['blocks']) == len(m['replies']) == 400
        assert parse_block(m['final_block']).score == m['result']['score']
        for s in m['seats']:
            declared = plan['programs'][s['name']]
            assert s['family'] == declared['family'] and s['sha256'] == declared['source_sha256']
            if s['family'] in plan['reserved_lineages']:
                lineage_maps.setdefault((m['teams'], s['family']), set()).add(m['map_seed'])
        for t, (block, replies) in enumerate(zip(m['blocks'], m['replies']), 1):
            assert parse_block(block).turn == t and len(replies) == m['teams']
            for seat, line in enumerate(replies):
                valid = parse_reply(line, t) is not None
                totals['allseat_turn_rows'] += 1
                totals['valid_reply_rows' if valid else 'invalid_or_missing_reply_rows'] += 1
                if m['seats'][seat]['family'] in plan['reserved_lineages']:
                    totals['reserved_turn_rows'] += 1
                    totals['reserved_valid_reply_rows' if valid else 'reserved_invalid_or_missing_reply_rows'] += 1
        totals['allseat_faults'] += len(m['faults'])
        assert sum(b['faults'] for b in m['bots']) == len(m['faults']) == receipts[fx['match_id']]['faults']
        for seat, b in enumerate(m['bots']):
            allseat.append({'match_id': fx['match_id'], 'seat': seat, 'name': m['seats'][seat]['name'],
                'family': m['seats'][seat]['family'], 'report': b})
    assert len(lineage_maps) == 28 and all(len(v) == 20 for v in lineage_maps.values())
    assert totals['reserved_turn_rows'] == 224000
    result = {'identity': frozen['identity'], 'raw_receipts_sha256': sha(R / 'COLLECTION_RECEIPTS.json'),
        'independent_matches': 300, 'independent_maps': len(seen), 'counts': dict(totals),
        'lineage_maps': [{'teams': n, 'lineage': f, 'maps': len(v)} for (n, f), v in sorted(lineage_maps.items())],
        'all_seat_reports': allseat,
        'checks': 'Raw hashes, binary and measurement sources, seed/seat/family/engine/limits/fixture,400 blocks,final scores,canonical reply validity and allseat faults independently checked. No replay or exclusions.'}
    (R / 'RAW_COLLECTION_AUDIT.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8', newline='\n')
    return result
```

## reports/models/p_ood/audit_metrics.py

SHA256: `372910f03559fcc18afe3a1728924e3189d3bcf31475f0043924ea1959cacae7`; 4655 bytes.

```python
"""Independently recompute published per-map means and clustered uncertainty."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

P = Path.cwd()
R = P / 'reports/models/p_ood'
B = P / 'build/t0025'
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check(values, clusters, declared):
    mean = float(np.mean(values))
    sums = []
    for c in sorted(set(clusters)):
        sums.append(float(np.sum(values[clusters == c] - mean)))
    k = len(sums)
    se = float(np.sqrt(sum(v * v for v in sums) * k / (k - 1)) / len(values))
    np.testing.assert_allclose([mean, se, mean - 1.96 * se, mean + 1.96 * se],
        [declared['mean'], declared['se'], *declared['ci95']], rtol=0, atol=1e-12)
    assert declared['rows'] == len(values) and declared['maps'] == k

def main():
    assert read(Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/cpu.json')['task'] == 'T0025'
    assert not (Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/gpu.json').exists()
    receipts = read(R / 'SCORE_RECEIPTS.json')
    result = read(R / 'OOD_RESULTS.json')
    chunks, maps = [], []
    for mid, receipt in receipts['matches'].items():
        p = B / 'ood_scores' / (mid + '.npz')
        assert sha(p) == receipt['npz_sha256']
        with np.load(p) as a:
            d = dict(a)
        chunks.append(d)
        maps.append(np.full(len(d['nll']), receipt['map_seed']))
        assert len(d['nll']) == receipt['rows']
        assert np.isfinite(d['nll']).all() and np.isfinite(d['confidence']).all()
        assert ((d['confidence'] > 0) & (d['confidence'] <= 1)).all()
        assert (d['nll'] >= -1e-12).all()
        assert d['qh_nll'].shape == (len(d['nll']),)
        assert np.isfinite(d['qh_nll']).sum() == receipt['qh_coverage']['complete_rows']
    a = {k: np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
    maps = np.concatenate(maps)
    checked = 0
    for row in result['rows']:
        sel = a['teams'] == row['teams']
        if row['lineage'] != 'both reserved lineages':
            sel &= a['family'] == row['lineage']
        if row['active_only']:
            sel &= a['active_units'] > 0
        assert sel.sum() == row['rows'] and len(set(maps[sel])) == row['maps']
        check(a['nll'][sel, 1] - a['nll'][sel, 0], maps[sel], row['XXL_minus_XL'])
        for j, size in enumerate(('XL', 'XXL')):
            r = row['sizes'][size]
            check(a['nll'][sel, j], maps[sel], r['joint_nll'])
            check(a['correct'][sel, j], maps[sel], r['top1'])
            conf, hit = a['confidence'][sel, j], a['correct'][sel, j]
            b = np.minimum((conf * 10).astype(int), 9)
            ece = sum(abs(float(np.sum(hit[b == i] - conf[b == i]))) for i in range(10)) / len(conf)
            np.testing.assert_allclose(ece, r['calibration']['ece'], rtol=0, atol=1e-12)
            for name, baseline in r['baselines'].items():
                values = a[name + '_nll'][sel]
                valid = np.isfinite(values)
                assert baseline['rows'] == valid.sum()
                if valid.any():
                    check(values[valid], maps[sel][valid], baseline['joint_nll'])
                    check(a['nll'][sel, j][valid] - values[valid], maps[sel][valid], baseline['P_minus_baseline'])
                    lo, hi = baseline['P_minus_baseline']['ci95']
                    expected = 'lower P NLL shown' if hi < 0 else 'higher P NLL shown' if lo > 0 else 'not shown'
                    assert baseline['P_minus_baseline']['interpretation'] == expected
            checked += 1
    assert len(maps) == result['valid_reserved_rows'] == 224000 and len(set(maps)) == 300
    receipt = {'task': 'T0025', 'identity': receipts['identity'], 'score_receipts_sha256': sha(R / 'SCORE_RECEIPTS.json'),
        'result_sha256': sha(R / 'OOD_RESULTS.json'), 'auditor_sha256': sha(Path(__file__)),
        'cohorts_checked': checked, 'all_cached_array_hashes_verified': True,
        'independent_NLL_top1_baseline_mean_cluster_SE_and_CI_recomputed': True,
        'independent_ECE_points_recomputed': True,
        'limitation': 'ECE bootstrap endpoints use the declared aggregator and deterministic map draws; this independent audit recomputes ECE points, not percentile endpoints. No model predictions or games rerun.'}
    (R / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(receipt, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'cohorts_checked': checked, 'rows': len(maps), 'maps': len(set(maps))}), flush=True)

if __name__ == '__main__':
    main()
```

## reports/models/p_ood/audit_training_lineage.py

SHA256: `edb0dd9b9d02a17662e8d53e7d6df47e6af41ddeebc3920888566aa635e65efe`; 5703 bytes.

```python
"""Audit original training headers and checkpoint provenance without scoring.

Run under a T0025 CPU lease. Private inputs are runtime environment variables.
Only gzip JSON metadata before the blocks array is read; game bodies are not
replayed or scored. This proves exclusion relative to the recorded inventory,
not unknowable historical authorship.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

P = Path.cwd()
R = P / 'reports/models/p_ood'
B = P / 'build/t0025'
B.mkdir(parents=True, exist_ok=True)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

lease = read(Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/cpu.json')
assert lease['task'] == 'T0025'
assert not (Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/gpu.json').exists()
index_path = Path(os.environ['T0025_TRAIN_INDEX'])
original = index_path.parents[2]
index = read(index_path)
inventory_path = P / 'roster/new_bots/inventory.json'
inventory = read(inventory_path)['bots']
reserved = {'lineage:AI_Arena_CTF_Player02_bot', 'lineage:ctf-arena_Player4_v2'}
ids = {x['id'] for x in inventory if x.get('lineage_group') in reserved}
changed = True
while changed:
    before = len(ids)
    for x in inventory:
        if any(x.get(k) in ids for k in ('byte_alias_of', 'code_alias_of', 'representative')):
            ids.add(x['id'])
    changed = before != len(ids)
excluded = [x for x in inventory if x['id'] in ids]
bad_sha = {x['source_sha256'] for x in excluded}
by_sha = {}
for x in inventory:
    by_sha.setdefault(x['source_sha256'], set()).add(x.get('lineage_group'))
assert not reserved.intersection(index['families'])
pool_path = P / 'profiles/open_training_pool_recommended.json'
pool = read(pool_path)
assert set(pool['reserved_ood_lineages']) == reserved
assert not reserved.intersection(x['family'] for x in pool['bots'])
headers = []
unknown = {}
source_families = {}
for j, m in enumerate(index['matches']):
    path = original / m['file'].replace('\\', '/')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        prefix = ''
        while '"blocks":' not in prefix:
            part = f.read(4096)
            assert part, (m['match_id'], 'missing blocks delimiter')
            prefix += part
            assert len(prefix) < 100000, 'metadata unexpectedly large'
    h = json.loads(prefix.split('"blocks":', 1)[0].rstrip(',') + '}')
    assert h['match_id'] == m['match_id'] and h['map_seed'] == m['map_seed'] and h['teams'] == m['teams']
    assert [x['name'] for x in h['seats']] == m['seats']
    assert [x['family'] for x in h['seats']] == m['families']
    for s in h['seats']:
        assert s['sha256'] not in bad_sha and s['family'] not in reserved
        assert not reserved.intersection(by_sha.get(s['sha256'], set()))
        source_families.setdefault(s['sha256'], set()).add(s['family'])
        if s['sha256'] not in by_sha:
            unknown.setdefault(s['sha256'], set()).add(s['name'])
    headers.append({'match_id': h['match_id'], 'map_seed': h['map_seed'], 'seats': h['seats'],
                    'header_sha256': hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()})
    if (j + 1) % 1000 == 0:
        print(json.dumps({'headers_checked': j + 1, 'total': len(index['matches'])}), flush=True)
configs = {}
for size, folder in [('XL', 'models_xl'), ('XXL', 'models_xxl')]:
    manifest = read(P / 'inference' / folder / 'MANIFEST.json')
    for model_id, v in manifest.items():
        run = original / ('runs_xl' if size == 'XL' else 'runs_xxl') / ('policy_' + model_id)
        config = read(run / 'config.json')
        assert sha(run / 'model.npz') == v['identity']['policy_sha256']
        counts = config['data']['team_counts']
        digest = hashlib.sha256('\n'.join(sorted(m['match_id'] for m in index['matches'] if m['teams'] in counts)).encode()).hexdigest()[:16]
        assert digest == config['data']['digest']
        configs[size + '/' + model_id] = {'config_sha256': sha(run / 'config.json'),
            'policy_checkpoint_sha256': sha(run / 'model.npz'), 'data_match_digest': digest,
            'counts': counts, 'seed': config['seed'], 'capacity': config['capacity'],
            'data_options': config['data']['options']}
result = {'task': 'T0025', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'inventory_sha256': sha(inventory_path), 'training_index_sha256': sha(index_path),
    'pool_sha256': sha(pool_path), 'recorded_reserved_lineage_members':
        [{k: x.get(k) for k in ('id', 'source_sha256', 'lineage_group', 'byte_alias_of', 'code_alias_of', 'representative')} for x in excluded],
    'training_headers_checked': len(headers), 'training_header_digest': hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
    'reserved_name_or_source_hits': 0, 'unknown_inventory_sources': {k: sorted(v) for k, v in unknown.items()},
    'training_source_families': {k: sorted(v) for k, v in source_families.items()}, 'model_training_configs': configs,
    'limitation': 'Exclusion is against the recorded source-lineage and alias inventory. Header metadata is independently checked; game bodies and training are not rerun. Sources missing from that inventory remain explicitly unresolved.'}
(B / 'TRAINING_HEADERS.json').write_text(json.dumps(headers, indent=1) + '\n', encoding='utf-8', newline='\n')
(R / 'TRAINING_LINEAGE_AUDIT.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'headers': len(headers), 'checkpoint_sources': len(configs), 'unresolved_source_count': len(unknown)}), flush=True)
```

## reports/models/p_ood/collect_ood.py

SHA256: `88cf300d93254dce351016a7a4a93f2a4bbe676fc74648d10803dd390737e4b5`; 7155 bytes.

```python
"""Hash-frozen official OOD collection, using the unchanged collect.play.

No training or scoring is performed. Complete fixtures are retained and
validated before reuse. All source paths in public receipts are portable.
"""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path.cwd()
sys.path.insert(0, str(P))
from fusion import matchlog, runner
from fusion.paths import bot_path, engine_path, git_commit
from training.collect import Fixture, _work

R = P / 'reports/models/p_ood'
B = P / 'build/t0025'
RAW = P / 'data/t0025_ood/matches'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8', newline='\n')

def portable(p, arena):
    path = Path(p).resolve()
    for root, tag in ((P.resolve(), '@repo'), (arena.resolve(), '@arena')):
        try:
            return tag + '/' + path.relative_to(root).as_posix()
        except ValueError:
            pass
    raise AssertionError('Unexpected compiler dependency outside declared roots')

def main():
    start = time.time()
    lease = read(Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/cpu.json')
    assert lease['task'] == 'T0025'
    assert not (Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/gpu.json').exists()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == 'de5dece1b90206684e06d82617d772e82d6658a3'
    plan = read(R / 'COLLECTION_PLAN.json')
    assert plan['seed'] == 20261250 and plan['independent_maps'] == 300
    assert plan['old_map_intersections'] == 0
    arena = Path(os.environ['AI_TEST_ROOT'])
    cxx = runner.find_compiler()
    assert cxx
    version = subprocess.check_output([cxx, '--version'], text=True).splitlines()[0]
    sources, commands, info = {}, {}, {}
    for name, member in plan['programs'].items():
        source = bot_path(member, arena)
        assert sha(source) == member['source_sha256'], name
        dep = subprocess.run([cxx, *runner.CXX_FLAGS, '-MM', '-MT', 'artifact', str(source)],
                             capture_output=True, text=True, check=True)
        tokens = dep.stdout.replace('\\\n', ' ').split(': ', 1)[1].split()
        dependencies = {}
        for token in tokens:
            path = Path(token)
            if not path.is_absolute():
                path = P / path
            dependencies[portable(path, arena)] = sha(path)
        assert portable(source, arena) in dependencies
        binary = runner.compile_cpp(source, B / 'bots')
        commands[name] = [str(binary.resolve())]
        info[name] = {'name': name, 'family': member['family'], 'sha256': sha(source),
                      'policy_labels': member['policy_labels']}
        sources[name] = {'source': portable(source, arena), 'source_sha256': sha(source),
                         'dependencies': dependencies, 'binary': binary.relative_to(P).as_posix(),
                         'binary_sha256': sha(binary)}
        print(json.dumps({'stage': 'compile', 'program': name, 'done': len(sources), 'total': len(plan['programs'])}), flush=True)
    inputs = {p: sha(P / p) for p in ('training/collect.py', 'fusion/runner.py', 'fusion/matchlog.py',
                                     'fusion/paths.py', 'reports/models/p_ood/collect_ood.py')}
    frozen = {'task': 'T0025', 'seed': plan['seed'], 'source_commit': plan['source_commit'],
              'collection_plan_sha256': sha(R / 'COLLECTION_PLAN.json'), 'limits_ms': [2000, 50],
              'workers': 6, 'engine_sha256': sha(engine_path(arena)),
              'compiler_version': version, 'compiler_sha256': sha(cxx),
              'flags': [x.replace(str(P / 'inference'), '${REPO}/inference') for x in runner.CXX_FLAGS],
              'measurement_sources': inputs, 'programs': sources,
              'model_input_preparation_sha256': sha(R / 'INPUT_PREPARATION.json'),
              'training_lineage_audit_sha256': sha(R / 'TRAINING_LINEAGE_AUDIT.json')}
    frozen['identity'] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    info['__engine__'] = frozen['engine_sha256']
    info['__commit__'] = git_commit(arena)
    frozen_path = R / 'FROZEN_COLLECTION.json'
    if frozen_path.exists():
        assert read(frozen_path) == frozen, 'Existing collection identity changed; no compatible resume'
    else:
        write(frozen_path, frozen)
    jobs, completed, receipts = [], [], {}
    for f in plan['fixtures']:
        fx = Fixture(**f, registry=frozen['identity'])
        out = RAW / f'n{fx.teams:02d}' / f'{fx.match_id}.json.gz'
        if out.exists():
            m = matchlog.load(out)
            assert m['match_id'] == fx.match_id and m['map_seed'] == fx.map_seed and m['flag_seed'] == fx.flag_seed
            assert m['engine_sha256'] == frozen['engine_sha256'] and m['limits_ms'] == [2000, 50]
            assert [s['name'] for s in m['seats']] == fx.seats
            assert all(s['sha256'] == info[s['name']]['sha256'] and s['family'] == info[s['name']]['family'] for s in m['seats'])
            assert m['sampling']['registry'] == frozen['identity'] and len(m['blocks']) == 400
            completed.append(fx.match_id)
            receipts[fx.match_id] = {'raw_sha256': sha(out), 'bytes': out.stat().st_size}
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((fx, commands, info, str(arena), [2.0, .05], str(out)))
    write(B / 'collection_state.json', {'stage': 'games', 'completed': len(completed), 'total': 300,
                                       'pending': len(jobs), 'frozen_identity': frozen['identity']})
    with mp.get_context('spawn').Pool(6) as pool:
        for mid, score, faults in pool.imap_unordered(_work, jobs):
            completed.append(mid)
            out = RAW / mid[:3] / (mid + '.json.gz')
            assert out.is_file()
            receipts[mid] = {'raw_sha256': sha(out), 'bytes': out.stat().st_size, 'faults': faults}
            write(B / 'COLLECTION_RECEIPTS.partial.json', receipts)
            write(B / 'collection_state.json', {'stage': 'games', 'completed': len(completed), 'total': 300,
                                               'last_match': mid, 'last_faults': faults,
                                               'frozen_identity': frozen['identity']})
            print(json.dumps({'stage': 'games', 'completed': len(completed), 'total': 300, 'match': mid, 'faults': faults}), flush=True)
    assert len(completed) == 300 and len(set(completed)) == 300
    write(R / 'COLLECTION_RECEIPTS.json', {'identity': frozen['identity'], 'matches': receipts})
    write(B / 'collection_state.json', {'stage': 'complete', 'completed': 300, 'total': 300,
                                       'elapsed_seconds': time.time() - start, 'exit_code': 0,
                                       'frozen_identity': frozen['identity']})

if __name__ == '__main__':
    main()
```

## reports/models/p_ood/inventory_inputs.py

SHA256: `1c824b9a132df6586eeba5d2fdebcf1d880d88e1035bb3dc7c9b970cf85a8711`; 3977 bytes.

```python
"""Read-only T0025 source/model/dataset inventory, under a short CPU lease.

No games, inference, fits or exports. External inputs are runtime env paths.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

P=Path.cwd();R=P/'reports/models/p_ood'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

assert read(Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json')['task']=='T0025'
assert not (Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='de5dece1b90206684e06d82617d772e82d6658a3'
reserved={'AI_Arena_CTF_Player02_bot':'ef8719390bcdb0ff95c757bf56c73d1138064068b02b318eb1adf6623bf268ab',
          'ctf-arena_Player4_v2':'90038f9ab9b6120938e852cbb13b3a1bd9c074e568a5f648a55a0ec49681b779'}
sources={}
for name,expected in reserved.items():
    p=P/'roster/new_bots/independent'/f'{name}.cpp'
    assert p.is_file() and sha(p)==expected
    sources[name]=dict(path=p.relative_to(P).as_posix(),sha256=sha(p),bytes=p.stat().st_size,lineage='lineage:'+name)
registry=read(P/'profiles/open_training_pool_recommended.json')
index_path=Path(os.environ['T0025_TRAIN_INDEX']);index=read(index_path)
families=index['families']
assert isinstance(families,list)
exact=[f for f in families if f in ('lineage:'+n for n in reserved)]
substring=[f for f in families if any(n.lower() in f.lower() for n in reserved)]
models={}
for size,folder in [('XL','models_xl'),('XXL','models_xxl')]:
    directory=P/'inference'/folder;manifest=read(directory/'MANIFEST.json');entries={}
    for model_id,v in manifest.items():
        header=directory/v['header'];npz=directory/f'weights_{model_id}.npz';raw=header.read_bytes()
        restored=hashlib.sha256(raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
        assert v['header_sha256'] in (sha(header),restored) and v['npz_sha256']==sha(npz)
        assert v['identity']['spec_version']==2
        entries[model_id]=dict(npz_sha256=sha(npz),committed_header_sha256=sha(header),declared_export_header_sha256=v['header_sha256'],
                               policy_sha256=v['identity']['policy_sha256'],weights_digest=v['identity']['weights_digest'],
                               joint_policy_macs=v['macs']['joint_policy_per_team'])
    models[size]=dict(directory=directory.relative_to(P).as_posix(),manifest_sha256=sha(directory/'MANIFEST.json'),registry_sha256=sha(directory/'registry.hpp'),entries=entries)
arena=Path(os.environ['AI_TEST_ROOT'])
forecast=[p.relative_to(arena).as_posix() for p in arena.rglob('forecast_dump.cpp')]
result=dict(task='T0025',source_commit='de5dece1b90206684e06d82617d772e82d6658a3',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            reserved_sources=sources,recommended_pool_sha256=sha(P/'profiles/open_training_pool_recommended.json'),
            dataset_index_sha256=sha(index_path),dataset_matches=len(index['matches']),dataset_maps=len({m['map_seed'] for m in index['matches']}),
            exact_reserved_family_hits=exact,substring_family_hits=substring,dataset_family_names=families,models=models,
            legacy_forecast_sources=forecast,seed_status='unchosen; no formal collection or seed consumption',
            limitation='Family-name inventory only; complete alias/source-lineage exclusion and model training-source identity still need verification before collection. No OOD loss or strength result.')
(R/'INPUT_PREPARATION.json').write_text(json.dumps(result,indent=1)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(reserved_sources=len(sources),models={k:len(v['entries']) for k,v in models.items()},exact_family_hits=exact,
                      substring_family_hits=substring,forecast_sources=forecast,dataset_matches=result['dataset_matches'],dataset_maps=result['dataset_maps'])),flush=True)
```

## reports/models/p_ood/joint_score.cpp

SHA256: `1e06e109bb63cc3a91fda6ac3303cbe163542db1bc2becaa68b800c5ffe400fc`; 5632 bytes.

```cpp
// T0025 read-only batch scorer for one unchanged committed P2 registry.
// Compile three separate executables: T0025_TIER=0(S),1(XL),2(XXL).
// Stdin: uint32 record width, then int16 rows [extractor record, move3,
// act3, target3, joint1, team_count1]. Stdout: ten float64 values per row.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "policy.hpp"
#ifndef T0025_TIER
#error T0025_TIER must explicitly identify the frozen registry
#endif
#if T0025_TIER == 0
#include "models/registry.hpp"
#elif T0025_TIER == 1
#include "models_xl/registry.hpp"
#elif T0025_TIER == 2
#include "models_xxl/registry.hpp"
#else
#error Invalid T0025_TIER
#endif
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

double nll(const double* z, int n, int y) {
    if (y<0 || y>=n || z[y]<=fusion::kNegInf) throw std::runtime_error("invalid label or mask");
    double mx=*std::max_element(z,z+n),sum=0;
    for(int i=0;i<n;i++) if(z[i]>fusion::kNegInf) sum+=std::exp(z[i]-mx);
    return mx+std::log(sum)-z[y];
}
int main(int argc, char** argv) try {
    const bool raw=argc==2 && std::string(argv[1])=="--raw";
    if(argc>1 && !raw) throw std::runtime_error("unknown mode");
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY); _setmode(_fileno(stdout),_O_BINARY);
#endif
    uint32_t width=0;
    if(std::fread(&width,4,1,stdin)!=1 || width<644 || width>10000) throw std::runtime_error("record framing");
    std::vector<int16_t> row(width+11);int32_t x[fusion::kMaxWidth],acc[fusion::kMaxWidth];
    while(true) {
        size_t got=std::fread(row.data(),2,row.size(),stdin);
        if(!got) break;
        if(got!=row.size()) throw std::runtime_error("partial record");
        const auto* p=row.data()+2;fusion::PolicyFeatures f;
        for(auto& a:f.g) a=*p++;
        for(auto& u:f.u) for(auto& a:u) a=*p++;
        for(auto& u:f.m) for(auto& m:u) for(auto& a:m) a=*p++;
        for(auto& u:f.legal) for(auto& a:u) a=*p++;
        for(auto& u:f.actLegal) for(auto& m:u) for(auto& a:m) a=*p++;
        p+=3; // origin is a record field, not part of PolicyFeatures.
        for(auto& u:f.cand) for(auto& a:u) a=*p++;
        for(auto& u:f.candF) for(auto& c:u) for(auto& a:c) a=*p++;
        if(p>row.data()+width) throw std::runtime_error("feature prefix exceeds record");
        const int16_t* y=row.data()+width;
        const auto* set=fusion::models::forTeams(y[10]);
        if(!set || !set->policy) throw std::runtime_error("missing registry model");
        const auto& model=*set->policy;
        if(raw) {
            std::vector<int32_t> values;
            for(int k=0;k<3;k++) for(int m=0;m<fusion::kMoves;m++) {
                acc[0]=0;
                if(f.legal[k][m]) {fusion::moverInput(f,k,m,x);fusion::forwardRaw(model.mover,x,acc);}
                values.push_back(acc[0]);
            }
            fusion::squadInput(f,x);fusion::forwardRaw(model.squad,x,acc);
            values.insert(values.end(),acc,acc+fusion::kJoint);
            for(int k=0;k<3;k++) for(int m=0;m<fusion::kMoves;m++) {
                std::fill(acc,acc+fusion::kActs,0);
                if(f.legal[k][m]) {fusion::moverInput(f,k,m,x);fusion::forwardRaw(model.act,x,acc);}
                values.insert(values.end(),acc,acc+fusion::kActs);
            }
            for(int k=0;k<3;k++) for(int c=0;c<fusion::kCand;c++) for(int m=0;m<fusion::kMoves;m++) {
                acc[0]=0;
                if(f.cand[k][c]>=0) {fusion::targetInput(f,k,c,m,x);fusion::forwardRaw(model.target,x,acc);}
                values.push_back(acc[0]);
            }
            if(values.size()!=320 || std::fwrite(values.data(),4,320,stdout)!=320) throw std::runtime_error("raw output failure");
            continue;
        }
        fusion::PolicyOutput joint;fusion::jointPolicy(model,f,joint);
        double out[10]={nll(joint.logit.data(),fusion::kJoint,y[9]),0,0,0,0,0,0,0};
        out[5]=std::max_element(joint.logit.begin(),joint.logit.end())-joint.logit.begin()==y[9];
        bool complete=true;
        for(int k=0;k<3;k++) {
            int move=y[k],act=y[3+k],target=y[6+k];
            if(move<0 || move>=fusion::kMoves) throw std::runtime_error("invalid move");
            if(act>=0) {
                double z[fusion::kActs];fusion::moverInput(f,k,move,x);fusion::forwardRaw(model.act,x,acc);
                for(int a=0;a<fusion::kActs;a++) z[a]=f.actLegal[k][move][a]?fusion::toReal(model.act,acc[a]):fusion::kNegInf;
                out[1]+=nll(z,fusion::kActs,act);out[2]++;
                if(act==fusion::kAttack && target<0) complete=false;
            }
            if(target>=0) {
                double z[fusion::kCand];
                for(int c=0;c<fusion::kCand;c++) {
                    z[c]=fusion::kNegInf;
                    if(f.cand[k][c]>=0) {fusion::targetInput(f,k,c,move,x);fusion::forwardRaw(model.target,x,acc);z[c]=fusion::toReal(model.target,acc[0]);}
                }
                out[3]+=nll(z,fusion::kCand,target);out[4]++;
            }
        }
        out[6]=complete?out[0]+out[1]+out[3]:std::nan("");
        out[7]=set->id[0]=='n'?std::atoi(set->id+1):0; // actual serving ID;0 is general.
        double mx=*std::max_element(joint.logit.begin(),joint.logit.end()),sum=0;
        for(double z:joint.logit) if(z>fusion::kNegInf) {sum+=std::exp(z-mx);out[9]++;}
        out[8]=1.0/sum;
        if(std::fwrite(out,sizeof(double),10,stdout)!=10) throw std::runtime_error("output failure");
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
```

## reports/models/p_ood/package_ood.py

SHA256: `140d3381b776855d565783493c59fd9cc5fcd2533a5e16ea60762a314148e7cf`; 12196 bytes.

```python
"""Generate narrative and byte-exact runnable companions inside Markdown.

The repository report contract tracks only Markdown/JSON, so experimental
sources are embedded in REPRODUCE.md and remain untracked at runtime.
"""
import hashlib
import json
import math
from pathlib import Path

P = Path.cwd()
R = P / 'reports/models/p_ood'
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    result = read(R / 'OOD_RESULTS.json')
    validation = read(R / 'NATIVE_VALIDATION.json')
    independent = read(R / 'INDEPENDENT_AUDIT.json')
    raw = read(R / 'RAW_COLLECTION_AUDIT.json')
    assert result['valid_reserved_rows'] == 224000 and len(validation['cases']) == 28
    assert independent['cohorts_checked'] == 168 and raw['counts']['allseat_faults'] == 0
    overview = {}
    for family in ('lineage:AI_Arena_CTF_Player02_bot', 'lineage:ctf-arena_Player4_v2', 'both reserved lineages'):
        rows = [r for r in result['rows'] if r['lineage'] == family and not r['active_only']]
        assert len(rows) == 14
        values = {}
        for size in ('XL', 'XXL'):
            delta = [r['sizes'][size]['baselines']['qh']['P_minus_baseline'] for r in rows]
            mean = sum(d['mean'] for d in delta) / 14
            se = math.sqrt(sum(d['se'] ** 2 for d in delta)) / 14
            values[size] = {'joint_nll': sum(r['sizes'][size]['joint_nll']['mean'] for r in rows) / 14,
                           'P_minus_qH': mean, 'ci95': [mean - 1.96 * se, mean + 1.96 * se]}
        delta = [r['XXL_minus_XL'] for r in rows]
        mean = sum(d['mean'] for d in delta) / 14
        se = math.sqrt(sum(d['se'] ** 2 for d in delta)) / 14
        values['XXL_minus_XL'] = {'mean': mean, 'ci95': [mean - 1.96 * se, mean + 1.96 * se]}
        overview[family] = values
    summary = {'source_result_sha256': sha(R / 'OOD_RESULTS.json'), 'overview': overview,
        'weighting': 'Equal weight across 14 team counts, using the separately map-clustered count estimates; maps do not overlap between counts.'}
    (R / 'OVERVIEW.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8', newline='\n')
    lines = ['# T0025: deployed P2 on two reserved lineages', '',
        'Both deployed integer P sizes predict the two recorded reserved lineages better than the unchanged legacy q_H on this fresh-map collection. '
        'XXL has a small lower joint NLL than XL. These are prediction results; production defaults and model exports remain unchanged.', '',
        '| Reserved lineage | XL joint NLL | XXL joint NLL | XL − q_H, 95% map CI | XXL − q_H, 95% map CI |',
        '|---|---:|---:|---|---|']
    for family, row in overview.items():
        x, y = row['XL'], row['XXL']
        lines.append(f"| {family} | {x['joint_nll']:.6f} | {y['joint_nll']:.6f} | "
            f"{x['P_minus_qH']:+.6f} [{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}] | "
            f"{y['P_minus_qH']:+.6f} [{y['ci95'][0]:+.6f}, {y['ci95'][1]:+.6f}] |")
    d = overview['both reserved lineages']['XXL_minus_XL']
    lines += ['', f"Equal-count XXL − XL: {d['mean']:+.6f} nats [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}].",
        '', 'All figures above give each count 2–15 equal weight. Lower joint NLL is better. '
        'Per-count/per-lineage comparisons with uniform, ID-validation-fitted inertia and q_H, '
        '10-bin calibration with map-bootstrap intervals, top-1 and active-only cohorts are in [OOD_RESULTS.json](OOD_RESULTS.json) '
        'and the compact [per-count table](OOD_RESULTS.md). A confidence interval containing zero is labelled `not shown`.',
        '', 'The collection used fresh master seed `20261250`, 300 distinct maps, six workers and official 2000/50 ms limits. '
        'At two teams, 40 separate maps balance the lineages (20 each); at counts 3–15, 20 maps per count contain one of each. '
        'R3_EXPECTIMAX_sampled is the fixed admitted reference and fillers have distinct recorded lineages within each fixture. '
        'Both lineages have 20 maps per count, and share map clusters at counts 3–15.',
        '', 'All 300 games and all 968,000 seat-turn replies are retained. There were zero all-seat faults and zero missing/invalid protocol replies. '
        'The primary cohort contains 224,000 canonical valid reserved-lineage rows, with one row per observed team turn and weight 1. '
        'q_H covers all 224,000 joint rows and 514,151 active units, with zero missing or off-grid active forecasts. '
        'Success latency summaries are collection diagnostics; this report does not measure P deployment latency or show a 50 ms strength gain.',
        '', 'The independent training provenance audit checked 21,000 original match headers, every recorded source hash against the lineage/alias inventory, '
        'and 22 original XL/XXL checkpoint/config identities. No reserved-source/name hit or unresolved inventory source was found. '
        'Exclusion is relative to that frozen lineage assignment, not a claim about unknowable historical authorship. '
        'Fresh-map checks include the original train/val/test index, inspected prior schedules and all 50,000 actual T0028 records; '
        'the formerly unrecorded tournament master was located as `20261010`. See [seed provenance](SEED_PROVENANCE.json).',
        '', 'The P2 weights and folded temperatures are unchanged. Native versus actual NPZ parity checked 320 raw P-head accumulators '
        'and joint NLL/confidence at tolerance 1e-12 in 28 count/size cases before full scoring. '
        'No mixture or OOD-fitted parameter enters primary scores. Inertia was recomputed on original ID validation rows '
        'and matched the original report parameters. ID test NLL/top-1/calibration numbers are existing reports for exactly matching artifacts, '
        'shown alongside each OOD cohort; their stride/importance-weighted trajectories differ from these complete OOD trajectories. '
        'This is not a causal estimate of an OOD penalty.',
        '', 'The independent [metric audit](INDEPENDENT_AUDIT.json) rechecked all score-array hashes and 168 size/cohort entries, '
        'including paired means, map-clustered SE/CI, coverage and ECE points. ECE bootstrap endpoints use the declared deterministic aggregator; '
        'the separate auditor does not independently repeat those percentile endpoints. '
        'No game or model prediction was repeated for that audit.',
        '', 'The first scoring entry stopped before audit/inference because a local variable shadowed the raw-audit function. '
        'The failure log is retained; the name was fixed before any affected computation. Official games were never replayed. '
        'No training, OOD temperature fit, artifact export, main/task-board change or production routing change occurred.',
        '', 'Reproduction sources and commands are in [REPRODUCE.md](REPRODUCE.md). Local checks and PR CI are separate delivery steps.', '']
    (R / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    status = ['# T0025 branch status', '', 'Branch: `codex/t0025-unseen-lineage-evidence`.',
        'Frozen measurement source: `de5dece1b90206684e06d82617d772e82d6658a3`.', '',
        'Training/source audit, fresh fixture preparation, all 300 official games, raw audit, 28 native parity cases, '
        '224,000-row pure-P/q_H scoring, original-ID-validation baseline verification and aggregation are complete. '
        'Completed stages and hash receipts are retained. No replay, retraining, model export or production default change is authorized by this evidence.', '',
        'The user requires an isolated PR; the shared main checkout remains untouched. Task status bookkeeping was authorized after measurement completion. '
        'The report lane is `reports/models/p_ood/`. Reproduction sources are embedded in Markdown because reports track only Markdown/JSON.', '',
        'The fresh remote task board marks T0030 done; its broad tests lane is released. '
        'The isolated branch adds only p_ood to MODEL_STUDIES in tests/test_repo_layout.py. '
        'All measurement and independent audit stages are complete. Delivery checks are recorded in CHECKS.md; '
        'PR exact-head CI and review status are tracked separately in the user delivery receipts.', '']
    (R / 'BRANCH_STATUS.md').write_text('\n'.join(status), encoding='utf-8', newline='\n')
    sources = {}
    for p in sorted(R.iterdir()):
        if p.suffix in ('.py', '.cpp'):
            sources['reports/models/p_ood/' + p.name] = p
    text = ['# T0025 reproduction', '', 'Do not replay a completed fixture or rerun a successful score stage. '
        'The retained master `20261250` is consumed. Any independent future collection needs a newly checked seed and new identity.', '',
        'Use an isolated checkout of measurement source `de5dece1b90206684e06d82617d772e82d6658a3`. '
        'Set `AI_TEST_ROOT`, `T0025_TRAIN_INDEX` and `T0025_T28_RESULTS` only at runtime; never write the retained original data. '
        'The last input is the 50,000-game `games.jsonl` behind T0028. Use GCC 15.2, Python 3.14, '
        'OMP/MKL/OpenBLAS 8 and a scoped Git safe.directory for this checkout. All official host computation uses an exclusive T0025 CPU lease, '
        'with no formal GPU job at the same time. Long phases use the passive helper/observer recorded in the launch receipts.', '',
        'The exact commands were the following, each inside that CPU lease. The collection helper added a passive 15-second observer. '
        'Preparation reads a fresh committed seed ledger at `build/t0025/REMOTE_DIGEST.md`.', '',
        '```powershell', 'python reports/models/p_ood/inventory_inputs.py',
        'python reports/models/p_ood/audit_training_lineage.py',
        'python reports/models/p_ood/prepare_collection.py', 'python reports/models/p_ood/collect_ood.py',
        'python reports/models/p_ood/score_ood.py', 'python reports/models/p_ood/aggregate_ood.py',
        'python reports/models/p_ood/audit_metrics.py', 'python reports/models/p_ood/package_ood.py', '```', '',
        'The canonical extractor, labeler, legacy forecast parser and metric routines are unchanged. '
        'Native scoring takes actual committed registries in separate executables, with scratch width 2048 and unchanged P2 temperatures. '
        'The reference observer is the admitted R3_EXPECTIMAX seat. Only valid requests from the two reserved families enter the primary cohort. '
        'Conditional act/target heads are not a primary metric in this study; their weights are included in raw parity.', '',
        'ID-context figures are the existing artifact-matched pure-quantized fields in `reports/models/xl` and `reports/models/xxl`, '
        'not their fitted mixture fields. Only original ID validation fits inertia. q_H missing rows would have their own comparison denominator '
        'and would never be replaced by a claimed legacy forecast; this run had complete coverage.', '',
        'All runnable companions below have LF bytes and the shown SHA256. Extract each fenced block to its heading path '
        'as an untracked runtime file, including its final newline. Their public Markdown transport does not change measurement inputs.', '']
    embedded = {}
    for name, p in sources.items():
        raw_bytes = p.read_bytes()
        assert b'\r' not in raw_bytes and raw_bytes.endswith(b'\n')
        embedded[name] = {'sha256': sha(p), 'bytes': len(raw_bytes)}
        language = 'cpp' if p.suffix == '.cpp' else 'python'
        text += [f'## {name}', '', f'SHA256: `{sha(p)}`; {len(raw_bytes)} bytes.', '',
                 '```' + language, raw_bytes.decode('utf-8').rstrip('\n'), '```', '']
    (R / 'REPRODUCE.md').write_text('\n'.join(text), encoding='utf-8', newline='\n')
    (R / 'EMBEDDED_SOURCES.json').write_text(json.dumps(embedded, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'reproduction_sources': len(embedded), 'overview_lineages': len(overview)}), flush=True)

if __name__ == '__main__':
    main()
```

## reports/models/p_ood/prepare_collection.py

SHA256: `aa4bd16a7d8615d850af6432eb0e6e76e0929d8bac3b7058686a19c3a747d8aa`; 5929 bytes.

```python
"""Freeze fresh OOD fixtures; no compiler, bot, engine game, fit or export runs."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

P = Path.cwd()
sys.path.insert(0, str(P))
from training.collect import derive
from fusion.paths import bot_path

R = P / 'reports/models/p_ood'
B = P / 'build/t0025'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert read(Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/cpu.json')['task'] == 'T0025'
assert not (Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/gpu.json').exists()
audit = read(R / 'TRAINING_LINEAGE_AUDIT.json')
assert audit['training_headers_checked'] == 21000 and not audit['unknown_inventory_sources']
seed = 20261250
ledger = B / 'REMOTE_DIGEST.md'
assert ledger.is_file() and str(seed) not in ledger.read_text(encoding='utf-8')
index_path = Path(os.environ['T0025_TRAIN_INDEX'])
index = read(index_path)
assert sha(index_path) == audit['training_index_sha256']
seen_maps = {m['map_seed'] for m in index['matches']}
seen_sources = set()
records = Path(os.environ['T0025_T28_RESULTS'])
gids = set()
t28_maps = set()
with records.open(encoding='utf-8') as f:
    for line in f:
        m = json.loads(line)
        assert m['gid'] not in gids
        gids.add(m['gid'])
        t28_maps.add(m['map_seed'])
assert len(gids) == 50000
seen_maps.update(t28_maps)
old_schedules = {}
for tree in P.parent.iterdir():
    if not tree.is_dir():
        continue
    for base in ('data', 'build'):
        folder = tree / base
        if not folder.is_dir():
            continue
        for path in folder.rglob('schedule.json'):
            try:
                d = read(path)
            except (ValueError, OSError):
                continue
            if isinstance(d, dict) and d.get('seed') is not None:
                assert d['seed'] != seed, str(path)
            fixtures = d if isinstance(d, list) else d.get('fixtures', d.get('schedule', []))
            found = {int(x['map_seed']) for x in fixtures if isinstance(x, dict) and 'map_seed' in x}
            seen_maps.update(found)
            if found:
                old_schedules[f'{tree.name}/{path.relative_to(tree).as_posix()}'] = {'sha256': sha(path), 'maps': len(found)}
pool_path = P / 'profiles/open_training_pool_recommended.json'
pool = read(pool_path)
assert sha(pool_path) == audit['pool_sha256']
arena = Path(os.environ['AI_TEST_ROOT'])
reference = next(x for x in pool['bots'] if x['name'] == 'R3_EXPECTIMAX_sampled')
assert set(range(2, 16)).issubset(reference['counts'])
reserved = ['AI_Arena_CTF_Player02_bot', 'ctf-arena_Player4_v2']
members = {x['name']: x for x in pool['bots']}
preparation = read(R / 'INPUT_PREPARATION.json')
for name in reserved:
    v = preparation['reserved_sources'][name]
    members[name] = {'name': name, 'path': '@repo/' + v['path'], 'family': v['lineage'],
                     'source_sha256': v['sha256'], 'policy_labels': True, 'counts': list(range(2, 16))}
fixtures = []
for n in range(2, 16):
    per_count = 40 if n == 2 else 20
    for i in range(per_count):
        rng = random.Random(derive(seed, n, i, 'seats'))
        seats = [reference['name']] + ([reserved[i % 2]] if n == 2 else reserved.copy())
        families = {members[x]['family'] for x in seats}
        candidates = [x for x in pool['bots'] if n in x['counts'] and x.get('policy_labels', True)
                      and x['family'] not in families]
        rng.shuffle(candidates)
        for candidate in candidates:
            if len(seats) >= n:
                break
            if candidate['family'] not in families:
                seats.append(candidate['name'])
                families.add(candidate['family'])
        assert len(seats) == n and len(families) == n
        rng.shuffle(seats)
        map_seed = derive(seed, n, i, 'map')
        assert map_seed not in seen_maps, (n, i, map_seed)
        seen_maps.add(map_seed)
        fixtures.append({'match_id': f'n{n:02d}-s{seed}-{i:06d}', 'teams': n, 'map_seed': map_seed,
                         'flag_seed': derive(seed, n, i, 'flag', bits=63), 'seats': seats,
                         'regime': 'reserved-lineage-ood', 'components': ['reference' if x == reference['name'] else
                             'reserved-ood' if x in reserved else 'admitted-id' for x in seats]})
needed = sorted({name for f in fixtures for name in f['seats']})
sources = {}
for name in needed:
    member = members[name]
    path = bot_path(member, arena)
    assert sha(path) == member['source_sha256'], name
    sources[name] = {k: member[k] for k in ('name', 'path', 'family', 'source_sha256', 'policy_labels')}
result = {'task': 'T0025', 'seed': seed, 'seed_status': 'frozen plan; no games started or seed consumed',
    'source_commit': 'de5dece1b90206684e06d82617d772e82d6658a3',
    'official_limits_ms': [2000, 50], 'workers': 6,
    'reference': reference['name'], 'reserved_lineages': ['lineage:' + x for x in reserved],
    'independent_maps': len(fixtures), 'maps_per_lineage_per_count': 20,
    'training_lineage_audit_sha256': sha(R / 'TRAINING_LINEAGE_AUDIT.json'),
    'remote_seed_ledger_sha256': sha(ledger), 'old_schedules': old_schedules,
    'original_index_sha256': sha(index_path), 'T0028_records_sha256': sha(records),
    'T0028_games': len(gids), 'T0028_unique_maps': len(t28_maps), 'old_map_intersections': 0,
    'programs': sources, 'fixtures': fixtures,
    'next_gate': 'Freeze compiled programs, transitive includes, engine, compiler and collection wrapper before games. No scoring, qH fits or export performed.'}
(R / 'COLLECTION_PLAN.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'fixtures': len(fixtures), 'programs': len(sources), 'seed': seed, 'previous_tournament_games': len(gids)}), flush=True)
```

## reports/models/p_ood/score_ood.py

SHA256: `4128ab335db71b32f0ee3693e4153ebdc44fc07b335a08dbb5d8c1def46140b0`; 14177 bytes.

```python
"""Actual deployed pure-P joint scoring; no OOD fit, mixture or export.

Canonical labels and the legacy forecast are read from the recorded admitted
reference seat. Integer parity precedes full scoring. Scores are cached only
with raw, source, binary and model identities that still match.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np

P = Path.cwd()
sys.path.insert(0, str(P))
from fusion import matchlog, native, metrics
from fusion.labels import label_match
from fusion.models import QuantizedTier
from fusion.nn import masked_log_softmax
from fusion.paths import NEXTGEN_REL
from fusion.protocol import DELTAS
from fusion.schema import Layout
from training import dataset
from training.common import policy_rows
from training.evaluate_offline import fit_inertia, unit_inertia, joint_of_units, unit_heuristic
from training.heuristic_baseline import make_forecaster
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_collection import audit as audit_raw_collection

R = P / 'reports/models/p_ood'
B = P / 'build/t0025'
SIZES = {'XL': ('models_xl', 1), 'XXL': ('models_xxl', 2)}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8', newline='\n')

def serving(n):
    return f'n{n:02d}' if n in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15) else 'general'

def subset(d, ix):
    return {k: v[ix] for k, v in d.items()}

def pack(rec, y, n):
    # This study measures joint moves. Conditional act/target labels are not
    # scored and cannot accidentally narrow the joint-move denominator.
    missing = np.full((len(rec), 3), -1)
    rows = np.column_stack([rec, y['y_move'], missing, missing, y['y_joint'], np.full(len(rec), n)]).astype('<i2')
    return struct.pack('<I', rec.shape[1]) + rows.tobytes()

def run(binary, data, raw=False):
    r = subprocess.run([str(binary), *(['--raw'] if raw else [])], input=data, capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='replace'))
    return np.frombuffer(r.stdout, dtype='<i4' if raw else '<f8').reshape(-1, 320 if raw else 10)

def records(fx, raw_receipts, collection, layout):
    path = P / 'data/t0025_ood/matches' / fx['match_id'][:3] / (fx['match_id'] + '.json.gz')
    assert sha(path) == raw_receipts[fx['match_id']]['raw_sha256']
    m = matchlog.load(path)
    assert len(m['blocks']) == len(m['replies']) == 400
    assert m['map_seed'] == fx['map_seed'] and m['flag_seed'] == fx['flag_seed'] and m['teams'] == fx['teams']
    assert m['engine_sha256'] == collection['engine_sha256'] and m['limits_ms'] == [2000, 50]
    assert m['sampling']['registry'] == collection['identity']
    assert [s['name'] for s in m['seats']] == fx['seats']
    for seat in m['seats']:
        declared = collection['programs'][seat['name']]
        assert seat['sha256'] == declared['source_sha256']
        assert seat['family'] == read(R / 'COLLECTION_PLAN.json')['programs'][seat['name']]['family']
    reference = fx['seats'].index('R3_EXPECTIMAX_sampled')
    rec = native.extract(matchlog.seat_stream(m, reference))
    y = label_match(m, rec, layout)
    assert len(rec) == 400 * fx['teams']
    team = rec[:, 1].astype(int)
    families = np.array([s['family'] for s in m['seats']])
    reserved = np.isin(families[team], ['lineage:AI_Arena_CTF_Player02_bot', 'lineage:ctf-arena_Player4_v2'])
    assert not (reserved & (team == reference)).any()
    selected = reserved & y['valid']
    return path, m, reference, rec[selected], subset(y, selected), families[team[selected]], {
        'reserved_rows': int(reserved.sum()), 'valid_reserved_rows': int(selected.sum()),
        'invalid_reserved_rows': int((reserved & ~y['valid']).sum()),
        'allseat_faults': len(m['faults']), 'fault_records': m['faults']}

def qh_rows(m, reference, path, rec, layout, forecast):
    table = forecast(str(path), reference)
    origin = layout.get(rec, 'origin').astype(int)
    size = int(m['init'].splitlines()[0].split()[3])
    q = np.full((len(rec), 3, 5), np.nan)
    missing = off_grid = active = 0
    for i, row in enumerate(rec):
        for k, o in enumerate(origin[i]):
            if o < 0:
                q[i, k] = [1, 0, 0, 0, 0]
                continue
            active += 1
            entries = table.get((int(row[0]), 3 * int(row[1]) + k))
            if entries is None:
                missing += 1
                continue
            z = np.zeros(5)
            for x, yy, p in entries:
                delta = (x - o % size, yy - o // size)
                if delta not in DELTAS:
                    off_grid += 1
                    break
                z[DELTAS.index(delta)] += p
            else:
                if z.sum() > 0:
                    q[i, k] = z / z.sum()
    return q, {'active_units': active, 'missing_units': missing, 'off_grid_units': off_grid,
               'complete_rows': int(np.isfinite(q).all((1, 2)).sum())}

def main():
    start = time.time()
    assert read(Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/cpu.json')['task'] == 'T0025'
    assert not (Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/gpu.json').exists()
    assert read(B / 'collection_state.json')['stage'] == 'complete'
    print(json.dumps({'stage': 'raw-audit', 'counts': audit_raw_collection(P)['counts']}), flush=True)
    plan = read(R / 'COLLECTION_PLAN.json')
    collection = read(R / 'FROZEN_COLLECTION.json')
    receipts = read(R / 'COLLECTION_RECEIPTS.json')
    assert len(receipts['matches']) == 300 and receipts['identity'] == collection['identity']
    arena = Path(os.environ['AI_TEST_ROOT'])
    layout = Layout(native.schema())
    bins = {}
    for size, (folder, tier) in SIZES.items():
        for model_id, entry in read(P / 'inference' / folder / 'MANIFEST.json').items():
            assert sha(P / 'inference' / folder / ('weights_' + model_id + '.npz')) == entry['npz_sha256']
        bins[size] = native.build(R / 'joint_score.cpp', 't0025_joint_' + size.lower(),
            deps=list((P / 'inference' / folder).glob('*.hpp')),
            defines=['FUSION_MAX_WIDTH=2048', f'T0025_TIER={tier}'])
    qh_binary = native.forecast_dump(arena)
    forecast = make_forecaster(qh_binary)
    frozen = {'task': 'T0025', 'collection_identity': collection['identity'],
        'source_commit': plan['source_commit'], 'raw_receipts_sha256': sha(R / 'COLLECTION_RECEIPTS.json'),
        'score_script_sha256': sha(Path(__file__)), 'native_source_sha256': sha(R / 'joint_score.cpp'),
        'schema': layout.schema, 'native_binaries': {k: sha(v) for k, v in bins.items()},
        'extractor_binary_sha256': sha(native.extractor()), 'qh_binary_sha256': sha(qh_binary),
        'qh_source_digest': native.forecast_digest(arena),
        'qh_sources': {p.relative_to(arena).as_posix(): sha(p) for p in native.forecast_sources(arena)},
        'model_manifests': {s: sha(P / 'inference' / folder / 'MANIFEST.json') for s, (folder, _) in SIZES.items()},
        'measurement_sources': {p: sha(P / p) for p in ['fusion/native.py', 'fusion/models.py', 'fusion/quant.py',
            'fusion/labels.py', 'training/evaluate_offline.py', 'training/heuristic_baseline.py', 'fusion/metrics.py',
            'training/dataset.py', 'training/common.py', 'reports/models/p_ood/audit_collection.py']},
        'scope': 'All canonical valid reserved-lineage rows; reference observer; pure deployed P2 joint. No OOD fit or mixture.'}
    frozen['identity'] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    if (R / 'FROZEN_SCORING.json').exists():
        assert read(R / 'FROZEN_SCORING.json') == frozen
    else:
        write(R / 'FROZEN_SCORING.json', frozen)
    parity = []
    for n in range(2, 16):
        fx = next(f for f in plan['fixtures'] if f['teams'] == n)
        _, _, _, rec, y, _, _ = records(fx, receipts['matches'], collection, layout)
        ix = np.linspace(0, len(rec) - 1, 16, dtype=int)
        rr, yy = rec[ix], subset(y, ix)
        for size, (folder, _) in SIZES.items():
            with np.load(P / 'inference' / folder / ('weights_' + serving(n) + '.npz')) as a:
                tier = QuantizedTier(dict(a), layout=layout)
            np.testing.assert_array_equal(run(bins[size], pack(rr, yy, n), True), tier.raw(rr, layout)[:, :320])
            got = run(bins[size], pack(rr, yy, n))
            z, mask = tier.joint_logits(rr, layout)
            lp = masked_log_softmax(z.astype(np.float64), mask)
            np.testing.assert_allclose(got[:, 0], -lp[np.arange(len(rr)), yy['y_joint']], rtol=0, atol=1e-12)
            np.testing.assert_allclose(got[:, 8], np.exp(lp).max(1), rtol=0, atol=1e-12)
            np.testing.assert_array_equal(got[:, 9], mask.sum(1))
            np.testing.assert_array_equal(got[:, 7], 0 if serving(n) == 'general' else n)
            parity.append({'teams': n, 'size': size, 'model': serving(n), 'rows': len(rr), 'raw320_exact': True,
                           'joint_nll_confidence_tolerance': 1e-12})
    write(R / 'NATIVE_VALIDATION.json', {'identity': frozen['identity'], 'cases': parity})
    original_shards = Path(os.environ['T0025_TRAIN_INDEX']).parent
    id_reports, rho, fit_receipts = {}, {}, {}
    for n in range(2, 16):
        fit = dataset.load(original_shards, [n], 'val')
        take = policy_rows(fit, original_shards)
        fit = subset(fit, take)
        rho[n] = fit_inertia(fit, layout)[n]
        fit_receipts[str(n)] = {'rows': len(fit['rec']), 'inertia_rate': rho[n],
            'validation_shards': {p.relative_to(original_shards).as_posix(): sha(p)
                                  for p in sorted((original_shards / f'n{n:02d}/val').glob('shard_*.npz'))}}
        for size, (folder, _) in SIZES.items():
            report = P / 'reports/models' / size.lower() / (serving(n) + '.json')
            d = read(report)
            manifest = read(P / 'inference' / folder / 'MANIFEST.json')[serving(n)]
            for key in ('policy_sha256', 'value_sha256', 'weights_digest'):
                assert d['identity'][key] == manifest['identity'][key]
            assert abs(d['policy']['inertia_rate'][str(n)] - rho[n]) < 1e-7
            row = next(x for x in d['policy']['by_teams'] if x['teams'] == n)
            id_reports[size + '/' + str(n)] = {'source': report.relative_to(P).as_posix(), 'report_sha256': sha(report),
                'reported_id_test': {k: row[k] for k in ('teams', 'rows', 'maps', 'quantized', 'uniform', 'inertia', 'heuristic', 'top1', 'ece')},
                'note': 'Existing matched-artifact ID test report; not recomputed or interpreted as OOD/strength.'}
        del fit
        print(json.dumps({'stage': 'ID-validation-inertia', 'count': n}), flush=True)
    write(R / 'BASELINE_FIT.json', {'identity': frozen['identity'], 'original_index_sha256': sha(original_shards / 'index.json'),
        'fit_split': 'original ID val only', 'counts': fit_receipts, 'id_test_reports': id_reports})
    score_receipts, audit = {}, []
    for i, fx in enumerate(plan['fixtures']):
        out = B / 'ood_scores' / (fx['match_id'] + '.npz')
        side = out.with_suffix('.json')
        if out.exists() and side.exists():
            old = read(side)
            assert old['identity'] == frozen['identity'] and old['npz_sha256'] == sha(out)
            assert old['raw_sha256'] == receipts['matches'][fx['match_id']]['raw_sha256']
            score_receipts[fx['match_id']] = old
            audit.append(old['raw_audit'])
            continue
        path, m, ref, rec, y, family, raw_audit = records(fx, receipts['matches'], collection, layout)
        qh, coverage = qh_rows(m, ref, path, rec, layout, forecast)
        selected = np.isfinite(qh).all((1, 2))
        h = np.full(len(rec), np.nan)
        h[selected] = -np.log(np.maximum(joint_of_units(qh[selected], y['y_move'][selected]), 1e-300))
        pi = joint_of_units(unit_inertia(rec, layout, np.full(len(rec), fx['teams']), rho), y['y_move'])
        got = np.stack([run(bins[size], pack(rec, y, fx['teams'])) for size in SIZES], axis=1)
        assert np.isfinite(got[:, :, [0, 5, 7, 8, 9]]).all()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, nll=got[:, :, 0], correct=got[:, :, 5], confidence=got[:, :, 8],
            serving_id=got[:, :, 7], uniform_nll=np.log(got[:, 0, 9]),
            inertia_nll=-np.log(np.maximum(pi, 1e-300)), qh_nll=h,
            family=family, turn=rec[:, 0], team=rec[:, 1], teams=np.full(len(rec), fx['teams']),
            active_units=(layout.get(rec, 'origin') >= 0).sum(1), stratum=y['stratum'])
        old = {'identity': frozen['identity'], 'raw_sha256': sha(path), 'npz_sha256': sha(out),
               'map_seed': fx['map_seed'], 'teams': fx['teams'], 'rows': len(rec),
               'qh_coverage': coverage, 'raw_audit': raw_audit}
        write(side, old)
        score_receipts[fx['match_id']] = old
        audit.append(raw_audit)
        write(B / 'score_state.json', {'stage': 'OOD-scoring', 'completed': i + 1, 'total': 300})
        print(json.dumps({'stage': 'OOD-scoring', 'completed': i + 1, 'total': 300}), flush=True)
    write(R / 'SCORE_RECEIPTS.json', {'identity': frozen['identity'], 'matches': score_receipts})
    write(R / 'COLLECTION_AUDIT.json', {'matches': len(audit),
        'allseat_faults': sum(a['allseat_faults'] for a in audit),
        'reserved_rows': sum(a['reserved_rows'] for a in audit),
        'valid_reserved_rows': sum(a['valid_reserved_rows'] for a in audit),
        'invalid_reserved_rows': sum(a['invalid_reserved_rows'] for a in audit),
        'checks': 'All 300 raw hashes, source/engine, seed/seat/limits,400 blocks,canonical labels and frozen fixture identities verified.'})
    write(B / 'score_state.json', {'stage': 'complete', 'completed': 300, 'elapsed_seconds': time.time() - start, 'exit_code': 0})

if __name__ == '__main__':
    main()
```
