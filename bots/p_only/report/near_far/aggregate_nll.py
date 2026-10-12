"""Paired size differences, equally weighted counts and map-cluster intervals.

Consumes verified completed native score receipts under a T0004 CPU lease.
No raw game or inference computation is repeated.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

P = Path.cwd()
R = P / 'bots/p_only/report/near_far'
B = P / 'build/t0004'
INF = 1 << 20


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def interval(value, variance):
    half = 1.96 * np.sqrt(variance)
    return [float(value-half), float(value+half)]


def ratio(rows, metric, size, difference):
    by_count = {}
    for e, mask, metrics in rows:
        if metric in ('joint', 'top1'):
            col = 0 if metric == 'joint' else 5
            numerator = metrics[:, size, col] - (metrics[:, 0, col] if difference else 0)
            denominator = np.ones(len(mask))
        elif metric in ('act', 'target'):
            col = 1 if metric == 'act' else 3
            numerator = metrics[:, size, col] - (metrics[:, 0, col] if difference else 0)
            denominator = metrics[:, size, col+1]
        else:
            col = 6
            mask = mask & np.isfinite(metrics[:, size, col]) & np.isfinite(metrics[:, 0, col])
            numerator = metrics[:, size, col] - (metrics[:, 0, col] if difference else 0)
            denominator = np.ones(len(mask))
        a = float(numerator[mask].sum())
        d = float(denominator[mask].sum())
        by_count.setdefault(e['teams'], []).append((e['map_seed'], a, d, int(mask.sum())))
    estimates = []
    result = {}
    for n, points in sorted(by_count.items()):
        active = [(a, d) for _, a, d, _ in points if d > 0]
        if not active:
            result[str(n)] = dict(value=None, denominator=0, maps=0)
            continue
        numer, denom = np.array(active).T
        value = numer.sum()/denom.sum()
        # Ratio-of-sums influence by independent map; finite-cluster correction.
        g = len(active)
        variance = g/(g-1) * np.square(numer-value*denom).sum()/denom.sum()**2 if g > 1 else None
        result[str(n)] = dict(value=float(value), ci95=interval(value, variance) if variance is not None else None,
                             denominator=float(denom.sum()), maps=g, rows=sum(x[3] for x in points))
        estimates.append((value, variance))
    if not estimates:
        return dict(overall=None, counts=result)
    mean = np.mean([x[0] for x in estimates])
    variance = sum(x[1] for x in estimates)/len(estimates)**2 if all(x[1] is not None for x in estimates) else None
    return dict(overall=dict(value=float(mean), ci95=interval(mean, variance) if variance is not None else None,
                            equally_weighted_counts=len(estimates), counts_with_no_labels=[n for n, x in result.items() if x['value'] is None]), counts=result)


def masks(g, primary):
    out = {'all_primary': primary}
    for name, col in [('reference', 2), ('flags', 3), ('carriers', 4)]:
        out[name+'_0_3'] = primary & (g[:, col] <= 3)
        out[name+'_4_6'] = primary & (g[:, col] >= 4) & (g[:, col] <= 6)
        out[name+'_7_plus'] = primary & (g[:, col] >= 7) & (g[:, col] < INF)
        out[name+'_unreachable_or_absent'] = primary & (g[:, col] >= INF)
    for name, col in [('geometric_contact_2', 5), ('opponent_carrier', 6), ('reference_carrier', 7)]:
        out[name+'_yes'] = primary & (g[:, col] > 0)
        out[name+'_no'] = primary & (g[:, col] == 0)
    out['inactive_opponent'] = g[:, 8] == 0
    out['inactive_reference'] = g[:, 9] == 0
    out['unreachable_active_reference'] = (g[:, 8] > 0) & (g[:, 9] > 0) & (g[:, 2] >= INF)
    return out


def main():
    lease = read(Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json')
    assert lease['task'] == 'T0004'
    assert not (Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
    receipts = read(R/'SCORE_RECEIPTS.json')
    frozen = read(R/'FROZEN_NATIVE.json')
    assert receipts['identity'] == frozen['identity'] and len(receipts['matches']) == 260
    groups = {}
    concentration = {'L-S': [], 'XXL-S': []}
    totals = {}
    for e in receipts['matches']:
        path = B/'scores'/(e['match_id']+'.npz')
        assert sha(path) == e['npz_sha256'] and e['identity'] == receipts['identity']
        for key in ('total_rows', 'reference_rows', 'excluded_family_rows', 'invalid_reply_rows', 'valid_rows', 'primary_rows',
                    'inactive_opponent_rows', 'inactive_reference_rows', 'unreachable_active_rows'):
            totals[key] = totals.get(key, 0) + e[key]
        with np.load(path) as arr:
            g, m, primary = arr['geometry'], arr['metrics'], arr['primary']
            assert len(m) == e['valid_rows'] and primary.sum() == e['primary_rows']
            np.testing.assert_array_equal(m[:, 0, 2:5:2], m[:, 1, 2:5:2])
            np.testing.assert_array_equal(m[:, 0, 2:5:2], m[:, 2, 2:5:2])
            for name, mask in masks(g, primary).items():
                groups.setdefault(name, []).append((e, mask, m))
            near = primary & (g[:, 2] <= 3)
            far = primary & (g[:, 2] >= 7) & (g[:, 2] < INF)
            if near.any() and far.any():
                for size, name in [(1, 'L-S'), (2, 'XXL-S')]:
                    delta = m[:, size, 0]-m[:, 0, 0]
                    concentration[name].append(dict(teams=e['teams'], map_seed=e['map_seed'], near_rows=int(near.sum()),
                                                     far_rows=int(far.sum()), near_minus_far=float(delta[near].mean()-delta[far].mean())))
    assert totals['total_rows'] == 936000
    assert totals['total_rows'] == totals['reference_rows'] + totals['excluded_family_rows'] + totals['invalid_reply_rows'] + totals['valid_rows']
    output = dict(identity=receipts['identity'], totals=totals, groups={}, concentration={},
                  uncertainty='95% normal intervals, independent map clusters within team count; equal team-count weighting. Exploratory grouped evidence; no strength inference.',
                  sign='Negative size-minus-S NLL is better. Negative near-minus-far paired-map concentration means greater improvement near.',
                  metric_denominators='Joint/top1 per valid team-turn, action/target per observed unit label, complete per fully modeled order. Primary requires both teams active and reachable. Missing conditional labels never become zero loss.')
    for name, rows in groups.items():
        output['groups'][name] = dict(rows=sum(int(mask.sum()) for _, mask, _ in rows), metrics={})
        for metric in ('joint', 'act', 'target', 'top1', 'complete'):
            output['groups'][name]['metrics'][metric] = {s: ratio(rows, metric, i, False) for i, s in enumerate(('S', 'L', 'XXL'))}
            output['groups'][name]['metrics'][metric].update({s: ratio(rows, metric, i, True) for i, s in [(1, 'L-S'), (2, 'XXL-S')]})
    for name, points in concentration.items():
        counts, means, variances = {}, [], []
        for n in range(3, 16):
            pp = [p for p in points if p['teams'] == n]
            if not pp:
                counts[str(n)] = dict(maps=0, value=None)
                continue
            values = np.array([p['near_minus_far'] for p in pp])
            value = values.mean()
            variance = values.var(ddof=1)/len(values) if len(values) > 1 else None
            counts[str(n)] = dict(maps=len(pp), value=float(value), ci95=interval(value, variance) if variance is not None else None)
            means.append(value)
            variances.append(variance)
        value = np.mean(means) if means else None
        variance = sum(variances)/len(means)**2 if means and all(v is not None for v in variances) else None
        output['concentration'][name] = dict(counts=counts, maps=len(points), points=points,
                                            overall=dict(value=float(value), ci95=interval(value, variance) if variance is not None else None,
                                                         equally_weighted_counts=len(means)) if means else None)
    (R/'NLL_SUMMARY.json').write_text(json.dumps(output, indent=1, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(totals=totals, primary_joint={k:v['overall'] for k,v in output['groups']['all_primary']['metrics']['joint'].items()},
                         concentration={k:v['overall'] for k,v in output['concentration'].items()})), flush=True)


if __name__ == '__main__':
    main()
