"""Identity-bound read-only native NLL scan. No training or game simulation.

Run from the frozen T0004 checkout under its exclusive host CPU lease.
Outputs stay in ignored build/t0004; completed match receipts are retained.
"""
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

import numpy as np

P = Path.cwd()
sys.path.insert(0, str(P))
from fusion import matchlog, native, quant
from fusion.labels import label_match
from fusion.models import QuantizedTier, policy_batch
from fusion.paths import is_unlabeled, unlabeled_families
from fusion.protocol import parse_block
from fusion.schema import Layout

R = P / 'bots/p_only/report/near_far'
B = P / 'build/t0004'
SOURCE = '8a4eab4bef5bf65e971ec6712eebb89821738130'
SIZES = ('S', 'L', 'XXL')
INF = 1 << 20


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(p)


def log(stage, **kwargs):
    obj = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), stage=stage, **kwargs)
    write(B / 'score_state.json', obj)
    with (B / 'score_events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj) + '\n')
    print(json.dumps(obj), flush=True)


def run(binary, data, args=()):
    cp = subprocess.run([str(binary), *args], input=data, capture_output=True)
    if cp.returncode:
        raise RuntimeError(cp.stderr.decode('utf-8', errors='replace'))
    return cp.stdout


def pack(rec, labels, n):
    rows = np.column_stack([rec, labels['y_move'], labels['y_act'], labels['y_tgt'],
                            labels['y_joint'], np.full(len(rec), n)]).astype('<i2')
    return struct.pack('<I', rec.shape[1]) + rows.tobytes()


def subset(labels, ix):
    return {k: v[ix] for k, v in labels.items()}


def serving(n):
    return f'n{n:02d}' if n in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15) else 'general'


def load_match(entry, files):
    path = files[entry['name']]
    assert sha(path) == entry['sha256']
    m = matchlog.load(path)
    assert len(m['blocks']) == 400 and m['map_seed'] == entry['map_seed']
    assert m['teams'] == entry['teams'] and m['engine_sha256'] == INPUT['engine_sha256']
    assert m['seats'][entry['reference_seat']]['sha256'] == entry['reference_binary_sha256']
    return m


def records(entry, files, geometry_binary, layout):
    m = load_match(entry, files)
    stream = matchlog.seat_stream(m, entry['reference_seat'])
    rec = native.extract(stream)
    geo = np.frombuffer(run(geometry_binary, stream.encode()), dtype='<i4').reshape(-1, 10)
    assert len(rec) == len(geo) == 400 * m['teams']
    np.testing.assert_array_equal(rec[:, :2], geo[:, :2])
    labels = label_match(m, rec, layout)
    return m, rec, geo, labels


def geometry_reference(m, reference, turns):
    lines = m['init'].splitlines()
    n = int(lines[0].split()[3])
    board = lines[2:2+n]
    opened = {(x, y) for y in range(n) for x in range(n) if board[y][x] != '#'}
    distances = {}

    def bfs(a):
        if a not in distances:
            d = {a: 0}
            q = collections.deque([a])
            while q:
                x, y = q.popleft()
                for b in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if b in opened and b not in d:
                        d[b] = d[(x, y)] + 1
                        q.append(b)
            distances[a] = d
        return distances[a]

    def closest(a, b):
        return min((bfs(x).get(y, INF) for x in a for y in b), default=INF)

    def attack(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        if abs(dx)+abs(dy) > 2:
            return False
        if abs(dx) == 2:
            return (a[0]+dx//2, a[1]) in opened
        if abs(dy) == 2:
            return (a[0], a[1]+dy//2) in opened
        if abs(dx) == abs(dy) == 1:
            return (b[0], a[1]) in opened or (a[0], b[1]) in opened
        return True

    result = []
    for t in turns:
        block = parse_block(m['blocks'][t-1])
        teams = [[] for _ in range(m['teams'])]
        carried = [0] * m['teams']
        carriers, flags = [], []
        for u in block.units:
            if u[2] >= 0:
                pos = u[2:4]
                teams[u[1]].append(pos)
                if u[5] >= 0:
                    carried[u[1]] = 1
                    carriers.append(pos)
        ls = m['blocks'][t-1].splitlines()
        start = next(i for i, x in enumerate(ls) if x.startswith('FLAGS '))
        for line in ls[start+1:start+1+int(ls[start].split()[1])]:
            v = line.split()
            if v[2] == 'cooldown':
                continue
            if v[2] == 'carried':
                u = block.units[int(v[5])]
                if u[2] >= 0:
                    flags.append(u[2:4])
            elif int(v[3]) >= 0:
                flags.append((int(v[3]), int(v[4])))
        me = teams[reference]
        for team, opp in enumerate(teams):
            d = closest(opp, me)
            contact = False
            if team != reference and d <= 6:
                contact = any(attack(x, y) for a in opp for b in me if bfs(a).get(b, INF) <= 6
                              for x, dd in bfs(a).items() if dd <= 2
                              for y, de in bfs(b).items() if de <= 2)
            result.append([t, team, d, closest(opp, flags), closest(opp, carriers), int(contact),
                           carried[team], carried[reference], len(opp), len(me)])
    return np.array(result)


def losses(z, mask, y):
    z = np.where(mask, z, -np.inf)
    mx = z.max(-1)
    return mx + np.log(np.exp(z-mx[..., None]).sum(-1)) - np.take_along_axis(z, y[..., None], -1)[..., 0]


def validate(files, bins, geometry_binary, layout):
    receipt = []
    for n in range(3, 16):
        e = next(e for e in INPUT['matches'] if e['teams'] == n)
        m, rec, geo, y = records(e, files, geometry_binary, layout)
        # Spaced valid rows plus first contact and carrier records, without selecting on loss.
        valid = np.flatnonzero(y['valid'] & (rec[:, 1] != e['reference_seat']))
        ix = list(valid[np.linspace(0, len(valid)-1, 12, dtype=int)])
        for cond in (geo[:, 5] > 0, geo[:, 6] > 0, geo[:, 7] > 0):
            ix.extend(np.flatnonzero(cond & y['valid'] & (rec[:, 1] != e['reference_seat']))[:4])
        ix = np.unique(ix)
        rr, yy = rec[ix], subset(y, ix)
        payload = pack(rr, yy, n)
        turns = sorted(set([1, 80, 160, 240, 320, 400, *geo[ix, 0].tolist()]))
        check = geometry_reference(m, e['reference_seat'], turns)
        np.testing.assert_array_equal(check, geo[np.isin(geo[:, 0], turns)])
        for size in SIZES:
            model_id = serving(n)
            directory = P / INPUT['models'][size]['directory']
            with np.load(directory / ('weights_' + model_id + '.npz')) as arrays:
                tier = QuantizedTier(dict(arrays), layout=layout)
            raw = np.frombuffer(run(bins[size], payload, ('--raw',)), dtype='<i4').reshape(-1, 320)
            np.testing.assert_array_equal(raw, tier.raw(rr, layout)[:, :320])
            got = np.frombuffer(run(bins[size], payload), dtype='<f8').reshape(-1, 8)
            X = policy_batch(rr, layout, normalize=False)
            z, mask = tier.joint_logits(rr, layout)
            expected = np.zeros_like(got)
            expected[:, 0] = losses(z, mask, yy['y_joint'].astype(int))
            expected[:, 5] = np.argmax(np.where(mask, z, -np.inf), axis=1) == yy['y_joint']
            expected[:, 7] = 0 if model_id == 'general' else n
            za = quant.to_real(tier.q['act'], quant.forward_raw(tier.q['act'], X.act_in(yy['y_move'])))
            zt = quant.to_real(tier.q['target'], quant.forward_raw(tier.q['target'], X.target_in(yy['y_move'])))[..., 0]
            for k in range(3):
                va, vt = yy['y_act'][:, k] >= 0, yy['y_tgt'][:, k] >= 0
                expected[va, 1] += losses(za[va, k], X.act_mask(yy['y_move'])[va, k], yy['y_act'][va, k].astype(int))
                expected[va, 2] += 1
                expected[vt, 3] += losses(zt[vt, k], X.cand_valid[vt, k], yy['y_tgt'][vt, k].astype(int))
                expected[vt, 4] += 1
            expected[:, 6] = expected[:, 0] + expected[:, 1] + expected[:, 3]
            incomplete = ((yy['y_act'] == 3) & (yy['y_tgt'] < 0)).any(1)
            expected[incomplete, 6] = np.nan
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
            receipt.append(dict(teams=n, size=size, serving_id=model_id, rows=len(ix), geometry_turns=len(turns),
                                raw_head_values=len(ix)*320, exact_raw=True, nll_atol=1e-12,
                                contact_rows=int((geo[ix, 5] > 0).sum()), carrier_rows=int((geo[ix, 6:8] > 0).any(1).sum())))
        log('validation', teams=n, rows=len(ix))
    write(R / 'NATIVE_VALIDATION.json', dict(passed=True, cases=receipt))


def score_one(entry, files, bins, geometry_binary, layout, identity):
    target = B / 'scores' / (entry['match_id'] + '.npz')
    receipt = target.with_suffix('.json')
    if target.exists() and receipt.exists():
        old = read(receipt)
        assert old['identity'] == identity and old['input_sha256'] == entry['sha256'] and old['npz_sha256'] == sha(target)
        return old
    m, rec, geo, y = records(entry, files, geometry_binary, layout)
    reference = rec[:, 1] == entry['reference_seat']
    excluded_families = unlabeled_families()
    permitted = np.array([s['family'] != 'candidate' and not is_unlabeled(s['family'], excluded_families) for s in m['seats']])
    allowed = ~reference & permitted[rec[:, 1]]
    valid = allowed & y['valid']
    primary = valid & (geo[:, 8] > 0) & (geo[:, 9] > 0) & (geo[:, 2] < INF)
    ix = np.flatnonzero(valid)
    payload = pack(rec[ix], subset(y, ix), m['teams'])
    metrics = np.stack([np.frombuffer(run(bins[s], payload), dtype='<f8').reshape(-1, 8) for s in SIZES], axis=1)
    sid = 0 if serving(m['teams']) == 'general' else m['teams']
    assert (metrics[:, :, 7] == sid).all() and np.isfinite(metrics[:, :, :6]).all()
    np.testing.assert_array_equal(metrics[:, 0, [2, 4]], metrics[:, 1, [2, 4]])
    np.testing.assert_array_equal(metrics[:, 0, [2, 4]], metrics[:, 2, [2, 4]])
    tmp = target.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, geometry=geo[ix], metrics=metrics, primary=primary[ix], y_move=y['y_move'][ix],
                        y_act=y['y_act'][ix], y_tgt=y['y_tgt'][ix], blocked=y['blocked'][ix])
    tmp.replace(target)
    out = dict(match_id=entry['match_id'], teams=m['teams'], map_seed=m['map_seed'], identity=identity,
               input_sha256=entry['sha256'], npz_sha256=sha(target), total_rows=len(rec), reference_rows=int(reference.sum()),
               excluded_family_rows=int((~reference & ~permitted[rec[:, 1]]).sum()),
               invalid_reply_rows=int((allowed & ~y['valid']).sum()), valid_rows=len(ix), primary_rows=int(primary.sum()),
               inactive_opponent_rows=int((valid & (geo[:, 8] == 0)).sum()),
               inactive_reference_rows=int((valid & (geo[:, 9] == 0)).sum()),
               unreachable_active_rows=int((valid & (geo[:, 8] > 0) & (geo[:, 9] > 0) & (geo[:, 2] >= INF)).sum()))
    write(receipt, out)
    return out


def main():
    global INPUT
    B.mkdir(parents=True, exist_ok=True)
    (B / 'scores').mkdir(exist_ok=True)
    lease = read(Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/cpu.json')
    assert lease['task'] == 'T0004'
    assert not (Path(os.environ['USERPROFILE']) / '.ctf-fusion/locks/gpu.json').exists()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    INPUT = read(R / 'FROZEN_INPUTS.json')
    assert INPUT['source_commit'] == SOURCE
    files = {p.name: p for p in Path(os.environ['T0004_RETAINED_MATCHES']).rglob('*.json.gz')}
    assert sha(os.environ['T0004_TRAIN_INDEX']) == INPUT['existing_dataset_index_sha256']
    log('compile')
    bins = {}
    required = []
    deps = []
    for size in SIZES:
        d = P / INPUT['models'][size]['directory']
        assert sha(d/'registry.hpp') == INPUT['models'][size]['registry_sha256']
        for model_id, e in INPUT['models'][size]['entries'].items():
            h = d / ('weights_' + model_id + '.hpp')
            assert sha(h) == e['committed_header_sha256'] and sha(d/('weights_'+model_id+'.npz')) == e['npz_sha256']
            required.append(native.model_buffer_width(h))
        deps.extend(native.model_dependencies(d/'registry.hpp'))
    scratch_width = max(required)
    assert scratch_width == 2048
    for i, size in enumerate(SIZES):
        log('compile', size=size)
        bins[size] = native.build(R/'policy_nll.cpp', 't0004_policy_'+size, deps=deps,
                                  defines=[f'T0004_TIER={i}', f'FUSION_MAX_WIDTH={scratch_width}'], includes=[P/'inference'])
    geometry_binary = native.build(R/'geometry.cpp', 't0004_geometry', includes=[P/'tools/cpp'])
    layout = Layout(native.schema())
    source_files = sorted(set(deps + list((P/'inference').glob('*.hpp')) + list((P/'tools/cpp').glob('*.hpp')) +
                              [P/'tools/cpp/extract_features.cpp', R/'score_records.py', R/'policy_nll.cpp', R/'geometry.cpp',
                               P/'fusion/models.py', P/'fusion/quant.py', P/'fusion/labels.py', P/'fusion/native.py', P/'fusion/schema.py',
                               P/'fusion/paths.py', P/'profiles/pool.json']))
    frozen = dict(source_commit=SOURCE, inputs_sha256=sha(R/'FROZEN_INPUTS.json'), workers=4,
                  sources={p.relative_to(P).as_posix(): sha(p) for p in source_files},
                  binaries={s: dict(name=p.name, sha256=sha(p)) for s, p in bins.items()},
                  geometry=dict(name=geometry_binary.name, sha256=sha(geometry_binary)),
                  extractor=dict(name=native.extractor().name, sha256=sha(native.extractor())),
                  compiler_version=subprocess.check_output([native.compiler(), '--version'], text=True).splitlines()[0],
                  flags=list(native.FLAGS), defines=['T0004_TIER=0/1/2', f'FUSION_MAX_WIDTH={scratch_width}'],
                  required_width_max=max(required), schema=layout.schema)
    identity = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    frozen['identity'] = identity
    if (R/'FROZEN_NATIVE.json').exists():
        assert read(R/'FROZEN_NATIVE.json') == frozen
    else:
        write(R/'FROZEN_NATIVE.json', frozen)
    if (R/'NATIVE_VALIDATION.json').exists():
        assert read(R/'NATIVE_VALIDATION.json')['passed']
    else:
        validate(files, bins, geometry_binary, layout)
    log('full_scan', completed=0, matches=len(INPUT['matches']))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(score_one, e, files, bins, geometry_binary, layout, identity) for e in INPUT['matches']]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
            log('full_scan', completed=len(results), matches=len(futs), match_id=results[-1]['match_id'])
    assert len(results) == 260
    write(R/'SCORE_RECEIPTS.json', dict(identity=identity, matches=sorted(results, key=lambda x:x['match_id']),
                                      scope='Native offline NLL only; chosen-order swaps and aggregation remain separate.'))
    log('complete', matches=len(results), valid_rows=sum(x['valid_rows'] for x in results), primary_rows=sum(x['primary_rows'] for x in results))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        B.mkdir(parents=True, exist_ok=True)
        with (B/'failures.log').open('a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        raise
