# T0011 reproduction

This records the exact transient drivers used with frozen repository source. No shared training code was edited. Data, caches, checkpoints and binaries remain local and ignored.

The initial unbounded NPZ concatenate/subset path exhausted host RAM before any epoch or checkpoint. RESTART.json records the failed attempt. POLICY_CACHE.json records every eligible source row in the original count/shard order and its bit-exact cache verification. The read-only NPY file mappings replace input storage only; the original trainer, initialization, global permutation, objectives, FP32/TF32 settings, GPU windows, validation, stopping, calibration and export remain unchanged. Labels are checked using the original checker in bounded slices. No training rows were subsampled.

Extract each code block below into the named file in a scratch directory. Run from the T0011 repository worktree with CUDA PyTorch installed. Set CTF_AGENT=codex@desktop-g60ojud, CTF_CAN=host,cuda,ai-test, AI_TEST_ROOT to the arena checkout, CXX to the C++ compiler, and OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=8. data/raw is a read-only local link to the original 21000-match corpus. The original shards are used only for parity.

Preparation used t0011_prepare.py under the CPU lock followed by t0011_data_parity.py. On a 64 GB host, reuse the completed audit receipt from that attempt instead of repeating its high-memory audit. Existing caches are reused only when their completed receipt and index identity agree. The training command after cache preparation is:

```text
python tools/repo/task.py lock --task T0011 cpu -- python <scratch>/t0011_resume.py
```

The resume driver builds the verified cache, then acquires the GPU lock and executes the frozen experiment driver. See FROZEN_PLAN.json for the validation-only finalist rule, unchanged arguments and seed1 export rule. Failed inputs are never treated as zero-valued measurements. The resumed pipeline writes build/t0011/pipeline_mmap.log and pipeline_mmap_state.json; do not rerun it over a completed model without reviewing its receipts.


Before any resumed GPU epoch, t0011_cache_contract.py compares every eligible n02 train/validation row against the original preparation for both families, including unchanged RNG state. Cache writing separately verifies all counts and every copied field.

## t0011_prepare.py

```python
"""Rebuild T0011 layout-3 data per count, preserving all original sampling.

Run from the T0011 worktree under its CPU lock; original index is read only.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--original-index',required=True)
args=ap.parse_args()
source=Path(args.original_index)
old=json.loads(source.read_text())
options=old['options']
assert options == {'stride':2,'keep':{'flag':1.0,'contact':1.0,'respawn':0.6,'quiet':0.35},'holdout_family':[],'holdout_mode':'fixture','split_by':'map'}
assert len(old['matches']) == 21000
expected={m['match_id']:m for m in old['matches']}
files=list(Path('data/raw').rglob('*.json.gz'))
actual={p.name.split('.')[0] for p in files}
assert len(files)==len(actual)==len(expected) and actual==set(expected)
out=Path('reports/models/p_features')
out.mkdir(parents=True,exist_ok=True)
common=['-m','training.dataset','--raw','data/raw','--out','data/shards_l3','--workers','8','--stride','2','--keep','flag=1,contact=1,respawn=0.6,quiet=0.35','--holdout-mode','fixture','--split-by','map']
commands=[]
index=Path('data/shards_l3/index.json')
for n in range(2,16):
    if index.exists():
        current=json.loads(index.read_text())
        found={m['match_id'] for m in current['matches'] if m['teams']==n}
        want={k for k,m in expected.items() if m['teams']==n}
        if found==want:
            print(f'[T0011] count {n} already complete; preserved',flush=True)
            continue
    command=[sys.executable,*common,'--teams',str(n)]
    if index.exists(): command.append('--append')
    commands.append(['python',*command[1:]])
    print(f'[T0011] rebuild count {n}',flush=True)
    subprocess.run(command,check=True)
new=json.loads(index.read_text())
assert new['options']==options
assert new['counts']==old['counts'], ('row counts differ',new['counts'],old['counts'])
assert {m['match_id'] for m in new['matches']}==set(expected)
for m in new['matches']:
    ref=expected[m['match_id']]
    for k in ('teams','split','map_seed','families','seats','faults','ai_test_commit','sampling'):
        assert m[k]==ref[k],(m['match_id'],k)
assert new['schema']['record_width']==923
receipt={'original_index_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rebuilt_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
         'original_record_width':old['schema']['record_width'],'rebuilt_record_width':923,'matches':21000,'options':options,'counts':new['counts'],
         'same_match_identity_split_and_metadata':True,'same_row_counts':True,'commands':commands,
         'note':'Rebuilt per team count to bound host memory; all 21000 original matches and every recorded sampling option retained.'}
(out/'DATA_REBUILD.json').write_text(json.dumps(receipt,indent=1)+'\n')
subprocess.run([sys.executable,'-m','training.feature_audit','--shards','data/shards_l3','--teams',*[str(n) for n in range(2,16)],'--split','train','--out','reports/models/p_features/audit'],check=True)
print('[T0011] rebuild and feature audit complete',flush=True)
```

## t0011_data_parity.py

```python
"""Compare every rebuilt legacy feature and label with the original dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--original-shards',required=True)
args=ap.parse_args()
old=Path(args.original_shards)
new=Path('data/shards_l3')
oi=json.loads((old/'index.json').read_text())
ni=json.loads((new/'index.json').read_text())
assert oi['options']==ni['options'] and oi['counts']==ni['counts']
old_matches=[m['match_id'] for m in oi['matches']]
new_matches=[m['match_id'] for m in ni['matches']]
assert old_matches==new_matches, 'raw-match order differs; compare canonical keys before training'
assert oi['families']==ni['families']
fields=('y_move','y_joint','y_act','y_tgt','blocked','valid','v_y','stratum','weight','match','teams','family','turn','team')
receipt={'all_equal':True,'legacy_record_width':oi['schema']['record_width'],'rows':{},'fields':['rec:first644',*fields],'shards':0}
for group,count in sorted(oi['counts'].items()):
    of=sorted((old/group).glob('shard_*.npz'))
    nf=sorted((new/group).glob('shard_*.npz'))
    assert len(of)==len(nf)
    total=0
    digest=hashlib.sha256()
    for op,npath in zip(of,nf):
        with np.load(op) as ov,np.load(npath) as nv:
            a,b=ov['rec'],nv['rec'][:,:oi['schema']['record_width']]
            assert a.shape==b.shape and np.array_equal(a,b), (group,op.name,'rec')
            total+=len(a)
            digest.update(a.tobytes())
            for key in fields:
                a,b=ov[key],nv[key]
                assert a.shape==b.shape and np.array_equal(a,b), (group,op.name,key)
                digest.update(a.tobytes())
        receipt['shards']+=1
    assert total==count
    receipt['rows'][group]={'rows':total,'legacy_feature_and_label_digest':digest.hexdigest()}
    print(f'[T0011 parity] {group}: {total} rows, all legacy features and labels equal',flush=True)
assert sum(v['rows'] for v in receipt['rows'].values())==30049121
Path('reports/models/p_features/DATA_PARITY.json').write_text(json.dumps(receipt,indent=1)+'\n')
print('[T0011 parity] all 30049121 rows passed',flush=True)
```

## t0011_policy_cache.py

```python
"""Build bounded, bit-exact policy-row mmap caches for T0011; hold the CPU lock."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from training.common import policy_rows
from training.dataset import check_schema, load_index

SHARDS = Path('data/shards_l3')
OUT = Path('data/pf_policy_cache')
FIELDS = ('rec', 'y_move', 'y_joint', 'y_act', 'y_tgt', 'weight', 'teams')
REPORT = Path('reports/models/p_features/POLICY_CACHE.json')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=1) + '\n')

def main():
    check_schema(SHARDS, ('p2', 'p3'))
    index = load_index(SHARDS)
    identity = {'index_sha256': sha(SHARDS / 'index.json'), 'counts': list(range(2, 16)),
                'keep_unlabeled': False, 'max_rows': 0, 'fields': list(FIELDS),
                'selection': 'training.common.policy_rows; original sorted count/shard order'}
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split in ('train', 'val'):
        dest = OUT / split
        meta = dest / 'CACHE.json'
        if meta.exists():
            prior = json.loads(meta.read_text())
            assert prior['identity'] == identity and prior['complete']
            reports[split] = prior
            print(f'[policy cache] reuse verified {split}: {prior["rows"]} rows', flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        paths = [p for t in range(2, 16) for p in sorted((SHARDS / f'n{t:02d}' / split).glob('shard_*.npz'))]
        parts, total, specs = [], 0, None
        for path in paths:
            with np.load(path) as z:
                # First pass loads only tiny selection columns, never full records.
                eligible = policy_rows({'valid': z['valid'], 'family': z['family']}, SHARDS)
                rows = int(eligible.sum())
                if specs is None:
                    # One source shard bounds the temporary record allocation.
                    specs = {k: (z[k].dtype, z[k].shape[1:]) for k in FIELDS}
            parts.append({'path': path.as_posix(), 'rows': rows, 'sha256': sha(path)})
            total += rows
        arrays = {k: np.lib.format.open_memmap(dest / f'{k}.npy', mode='w+', dtype=dtype,
                                             shape=(total, *tail)) for k, (dtype, tail) in specs.items()}
        source_hash = {k: hashlib.sha256() for k in FIELDS}
        cache_hash = {k: hashlib.sha256() for k in FIELDS}
        start = 0
        for part in parts:
            with np.load(part['path']) as z:
                selected = np.flatnonzero(policy_rows({'valid': z['valid'], 'family': z['family']}, SHARDS))
                assert len(selected) == part['rows']
                for key in FIELDS:
                    src = z[key]
                    assert src.dtype == arrays[key].dtype and src.shape[1:] == arrays[key].shape[1:]
                    for offset in range(0, len(selected), 8192):
                        take = selected[offset:offset + 8192]
                        value = src[take]
                        target = arrays[key][start + offset:start + offset + len(take)]
                        target[...] = value
                        assert np.array_equal(target, value), (part['path'], key, offset)
                        source_hash[key].update(value.tobytes())
                        cache_hash[key].update(target.tobytes())
                    arrays[key].flush()
                    del src
            start += part['rows']
            print(f'[policy cache] {split}: {start}/{total} rows', flush=True)
        assert start == total
        assert all(source_hash[k].digest() == cache_hash[k].digest() for k in FIELDS)
        receipt = {'identity': identity, 'complete': True, 'rows': total,
                   'source_shards': parts, 'row_order_and_values_equal': True,
                   'arrays': {k: {'shape': list(a.shape), 'dtype': a.dtype.str,
                                  'data_sha256': cache_hash[k].hexdigest(), 'bytes': int(a.nbytes)}
                              for k, a in arrays.items()}}
        write(meta, receipt)
        reports[split] = receipt
        del arrays
    write(REPORT, {'identity': identity, 'splits': reports,
                  'reason': 'Avoid full NPZ concatenation and advanced-index subset copies after host RAM exhaustion.',
                  'no_rows_dropped': True, 'no_training_argument_changes': True})
    print('[policy cache] complete, all eligible rows preserved and verified', flush=True)

if __name__ == '__main__':
    main()
```

## t0011_train_mmap.py

```python
"""Run the unchanged policy trainer with verified policy-row mmap input, T0011."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from training import train_policy
from training.dataset import check_schema

original_check = train_policy.check_labels

def prepare(shards, counts, split, max_rows, rng, keep_unlabeled=False, family='p2'):
    assert Path(shards) == Path('data/shards_l3')
    assert list(counts) == list(range(2, 16)) and not max_rows and not keep_unlabeled
    check_schema(shards, (family,))
    meta = json.loads(Path(f'data/pf_policy_cache/{split}/CACHE.json').read_text())
    assert meta['complete'] and meta['row_order_and_values_equal']
    digest = hashlib.sha256((Path(shards) / 'index.json').read_bytes()).hexdigest()
    assert meta['identity']['index_sha256'] == digest
    out = {}
    for key, spec in meta['arrays'].items():
        # Read-only file mappings avoid Windows' whole-view commit charge for copy-on-write mappings.
        # The unchanged trainer reads these arrays; count weighting creates a new small weight array.
        # Torch transfers read-only NumPy views to CUDA; it never writes into the source views.
        out[key] = np.load(f'data/pf_policy_cache/{split}/{key}.npy', mmap_mode='r')
        assert list(out[key].shape) == spec['shape'] and out[key].dtype.str == spec['dtype']
    print(f'[T0011 mmap] {split}: {meta["rows"]} identical eligible rows, no RAM concatenation', flush=True)
    return out

def check_labels(data, layout):
    for start in range(0, len(data['rec']), 8192):
        original_check({k: v[start:start + 8192] for k, v in data.items()}, layout)

def main():
    train_policy.prepare = prepare
    train_policy.check_labels = check_labels
    train_policy.main()

if __name__ == '__main__':
    main()
```

## t0011_cache_contract.py

```python
"""Check bounded prepare against the original loader on every n02 eligible row."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from training.train_policy import prepare as original_prepare

spec = importlib.util.spec_from_file_location('t0011_mmap', Path(__file__).with_name('t0011_train_mmap.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
receipt = {'complete': True, 'original_loader_counts': [2], 'checked': [],
           'all_cache_rows_verified_separately': 'POLICY_CACHE.json'}
for split in ('train', 'val'):
    for family in ('p2', 'p3'):
        original_rng = np.random.default_rng(1)
        cache_rng = np.random.default_rng(1)
        old = original_prepare('data/shards_l3', [2], split, 0, original_rng, False, family)
        new = module.prepare('data/shards_l3', list(range(2, 16)), split, 0, cache_rng, False, family)
        n = len(old['rec'])
        assert (new['teams'][:n] == 2).all() and new['teams'][n] == 3
        for key in new:
            assert old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key][:n]), (split, family, key)
        assert original_rng.bit_generator.state == cache_rng.bit_generator.state
        receipt['checked'].append({'split': split, 'family': family, 'rows': n,
                                   'all_fields_exact': True, 'rng_state_exact': True})
        del old, new
Path('reports/models/p_features/CACHE_CONTRACT.json').write_text(json.dumps(receipt, indent=1) + '\n')
print('[T0011 cache contract] original preparation and mmap rows/RNG are exact for both families', flush=True)
```

## t0011_train_pipeline_mmap.py

```python
"""Frozen T0011 full-data p2/p3 XXL experiment driver; hold CPU and GPU locks."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path.cwd()
REPORT=Path('reports/models/p_features')
BUILD=Path('build/t0011')
BUILD.mkdir(parents=True,exist_ok=True)
REPORT.mkdir(parents=True,exist_ok=True)
LOG=BUILD/'pipeline_mmap.log'
STATE=BUILD/'pipeline_mmap_state.json'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=1)+'\n')
baseline=read('inference/models_xxl/MANIFEST.json')['general']['sources']['policy']
ref=baseline['config']['args']
assert ref['epochs']==8 and ref['batch']==1024 and ref['seed']==1 and ref['count_weighting']=='inv_teams'
assert read(REPORT/'DATA_PARITY.json')['all_equal']
assert read(REPORT/'DATA_REBUILD.json')['same_row_counts']
assert (REPORT/'audit.json').is_file()
subprocess.run([sys.executable,str(Path(__file__).with_name('t0011_cache_contract.py'))],check=True)
fields=('epochs','patience','plateau_decay','lr_schedule','lr','min_lr','batch','weight_decay','max_rows','val_split','max_width','count_weighting','backend','device','gpu_resident_gb','gpu_window_rows','smoothing','act_weight','target_weight')
cli=[]
for key in fields: cli += ['--'+key.replace('_','-'),str(ref[key])]
assert not ref['tf32'] and not ref['keep_unlabeled']
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
plan={'source_commit':commit,'baseline_manifest_sha256':sha('inference/models_xxl/MANIFEST.json'),
      'baseline_policy_sha256':baseline['model_sha256'],'shards':'data/shards_l3','capacity':'XXL','policy_families':['p2','p3'],
      'arguments':{k:ref[k] for k in fields},'tf32':False,'keep_unlabeled':False,'seed1':1,
      'finalist_rule':'Before test evaluation: p3 calibrated validation overall joint-NLL paired map-clustered 95% interval wholly below zero. If true, train both families at seeds2 and3 with identical arguments. Otherwise retain seed1 only and report inconclusive/harm as measured.',
      'export_rule':'Export the prespecified seed1 models, never select a seed or a checkpoint from test results.',
      'host':platform.node(),'python':platform.python_version(),'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
      'data_parity_receipt_sha256':sha(REPORT/'DATA_PARITY.json'),'audit_sha256':sha(REPORT/'audit.json'), 'input_loading':'verified policy-only NPY mmap cache; all original eligible rows and row order preserved', 'policy_cache_receipt_sha256':sha(REPORT/'POLICY_CACHE.json'), 'cache_contract_receipt_sha256':sha(REPORT/'CACHE_CONTRACT.json'), 'mmap_launcher_sha256':sha(Path(__file__).with_name('t0011_train_mmap.py'))}
write(REPORT/'FROZEN_PLAN.json',plan)
steps=[]
def step(stage,cmd):
    start=time.monotonic()
    entry={'stage':stage,'command':['python',*[Path(x).name if x.endswith('.py') and Path(x).is_absolute() else x for x in cmd[1:]]],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(STATE,{'stage':stage,'steps':steps,'pid':os.getpid(),'source_commit':commit})
    print('[T0011] '+stage,flush=True)
    with LOG.open('a',encoding='utf-8') as log:
        log.write(json.dumps(entry)+'\n'); log.flush()
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
        code=proc.wait()
    entry.update(exit_code=code,wall_seconds=round(time.monotonic()-start,3))
    steps.append(entry)
    write(REPORT/'PIPELINE.json',{'plan':plan,'steps':steps})
    if code: raise SystemExit(f'{stage} failed: exit{code}; see build/t0011/pipeline.log')

def train(seed):
    for family in ('p2','p3'):
        run=Path(f'runs/pf_{family}_xxl_s{seed}')
        assert not (run/'model.npz').exists(), f'{run} already trained; verify receipt before any resume'
        step(f'train_{family}_s{seed}',[sys.executable,str(Path(__file__).with_name('t0011_train_mmap.py')),'--model','general','--shards','data/shards_l3','--capacity','XXL','--policy-family',family,'--seed',str(seed),'--out',str(run),*cli])
        dest=REPORT/'runs'/f'{family}_s{seed}'
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','log.jsonl'):
            shutil.copyfile(run/name,dest/name)

def compare(seed,split):
    step(f'compare_{split}_s{seed}',[sys.executable,'-m','training.compare_policy','--shards','data/shards_l3','--split',split,'--run',f'p2=runs/pf_p2_xxl_s{seed}','--run',f'p3=runs/pf_p3_xxl_s{seed}','--compare','p3:p2','--out',str(REPORT/f'{split}_s{seed}')])

train(1)
compare(1,'val')
validation=read(REPORT/'val_s1.json')['compare'][0]['strata']['all']['joint_nll']
finalist=validation['hi']<0
write(REPORT/'FINALIST_DECISION.json',{'split':'val','rule':plan['finalist_rule'],'seed1_validation':validation,'p3_finalist':finalist,'test_not_used':True})
seeds=[1]
if finalist:
    for seed in (2,3):
        train(seed); compare(seed,'val'); seeds.append(seed)
for seed in seeds: compare(seed,'test')
for family in ('p2','p3'):
    step(f'export_{family}_seed1',[sys.executable,'-m','training.export_mixed','--policy-run',f'runs/pf_{family}_xxl_s1','--value-tier','m','--out',f'inference/models_pf_{family}xxl_vm'])
write(STATE,{'stage':'complete','steps':steps,'pid':os.getpid(),'source_commit':commit,'seeds':seeds})
print('[T0011] training, comparisons and exports complete; review reports before PR',flush=True)
```

## t0011_resume.py

```python
"""Build bounded input then resume T0011 under GPU lock; call under CPU lock."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

scripts = Path(__file__).parent
report = Path('reports/models/p_features')
prior = report / 'FROZEN_PLAN.json'
failed = report / 'FROZEN_PLAN_before_restart.json'
if prior.exists() and not failed.exists():
    shutil.copyfile(prior, failed)
assert not Path('runs/pf_p2_xxl_s1/model.npz').exists(), 'Review completed run before a restart'
(report / 'RESTART.json').write_text(json.dumps({
    'reason': 'User reported host RAM exhaustion and reboot during the original P2 seed1 preparation.',
    'original_loader_observed_private_gb': 84.55, 'physical_ram_gb': 64,
    'completed_epochs_before_restart': 0, 'checkpoint_before_restart': False,
    'reused': ['21000-match layout3 rebuild', 'full train feature audit', '30049121-row exact legacy parity'],
    'input_change': 'Shard-at-a-time policy filtering into verified NPY mmaps; no row/order/dtype changes.',
    'training_parameters_unchanged': True,
}, indent=1) + '\n')
subprocess.run([sys.executable, str(scripts / 't0011_policy_cache.py')], check=True)
subprocess.run([sys.executable, 'tools/repo/task.py', 'lock', '--task', 'T0011', '--wait', '600', 'gpu', '--',
                sys.executable, str(scripts / 't0011_train_pipeline_mmap.py')], check=True)
```

## Live memory guard after user requested continuous monitoring

During P2 seed1 after epoch2, resident working set was about53 GiB and physical availability about2 GiB, despite private commit about17 GiB. The following external Windows guard applies a40 GiB hard maximum resident working set to verified direct Python children of the running pipeline. Samples are recorded every15 seconds in build/t0011/memory_guard.jsonl. This changes only resident-page policy; no model/data/CLI arguments changed. It may increase file/page IO and slow training. Training times across this change are not a controlled throughput comparison. A thread heartbeat also checks task health every5 minutes.

Windows API: [SetProcessWorkingSetSizeEx](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex).

```powershell
param([int]$PipelinePid = 34248, [int]$MaxWorkingGiB = 40)
$ErrorActionPreference = 'Stop'
$taskRoot = Join-Path $PSScriptRoot 'ctf-fusion.worktrees/T0011'
$guardDir = Join-Path $taskRoot 'build/t0011'
$guardLog = Join-Path $guardDir 'memory_guard.jsonl'
$guardState = Join-Path $guardDir 'memory_guard_state.json'
$guardOwner = Join-Path $guardDir 'memory_guard_pid.json'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class T0011WorkingSet {
 [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint rights, bool inherit, uint pid);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool CloseHandle(IntPtr handle);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetProcessWorkingSetSizeEx(IntPtr handle, UIntPtr min, UIntPtr max, uint flags);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GetProcessWorkingSetSizeEx(IntPtr handle, out UIntPtr min, out UIntPtr max, out uint flags);
}
'@

function Save-GuardEvent($event) {
    $event | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath $guardLog -Encoding utf8
    $event | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $guardState -Encoding utf8
}

if (Test-Path -LiteralPath $guardOwner) {
    $existing = Get-Content -LiteralPath $guardOwner -Raw | ConvertFrom-Json
    $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($existing.pid)"
    if ($existingProcess -and $existingProcess.CommandLine -match 't0011_memory_guard.ps1') {
        throw "T0011 memory guard already running, pid $($existing.pid)"
    }
}
$pipeline = Get-CimInstance Win32_Process -Filter "ProcessId=$PipelinePid"
if (-not $pipeline -or $pipeline.CommandLine -notmatch 't0011_train_pipeline_mmap.py') {
    throw 'The expected T0011 pipeline is not alive; inspect checkpoints before any restart.'
}
$pipelineCreated = $pipeline.CreationDate
@{pid=$PID; pipeline_pid=$PipelinePid; max_working_gib=$MaxWorkingGiB; interval_seconds=15; started_utc=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath $guardOwner -Encoding utf8
$limited = @{}
$lastCpu = @{}
$lastProgress = [DateTime]::UtcNow

try {
    while ($true) {
        $root = Get-CimInstance Win32_Process -Filter "ProcessId=$PipelinePid"
        if (-not $root -or $root.CreationDate -ne $pipelineCreated -or $root.CommandLine -notmatch 't0011_train_pipeline_mmap.py') {
            Save-GuardEvent @{utc=[DateTime]::UtcNow.ToString('o'); event='pipeline_exited'; pipeline_pid=$PipelinePid; inspect_results=$true}
            break
        }
        $os = Get-CimInstance Win32_OperatingSystem
        $samples = @()
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$PipelinePid" | Where-Object {
            $_.Name -match '^python' -and $_.CommandLine -match 't0011_train_mmap.py|training.compare_policy|training.export_mixed|t0011_cache_contract.py'
        })
        foreach ($child in $children) {
            $process = Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
            if (-not $process) { continue }
            $identity = "$($child.ProcessId):$($child.CreationDate.Ticks)"
            if (-not $limited.ContainsKey($identity)) {
                # Verify this exact child again; never act on a recycled/unrelated PID.
                $verified = Get-CimInstance Win32_Process -Filter "ProcessId=$($child.ProcessId)"
                if (-not $verified -or $verified.ParentProcessId -ne $PipelinePid -or $verified.CreationDate -ne $child.CreationDate) { continue }
                $handle = [T0011WorkingSet]::OpenProcess(0x500, $false, [uint32]$child.ProcessId)
                if ($handle -eq [IntPtr]::Zero) { throw "OpenProcess failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())" }
                try {
                    # Hard maximum, soft minimum. Only the process's resident page policy changes.
                    $ok = [T0011WorkingSet]::SetProcessWorkingSetSizeEx($handle, [UIntPtr][uint64](64MB), [UIntPtr][uint64]($MaxWorkingGiB * 1GB), 6)
                    if (-not $ok) { throw "SetWorkingSet failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())" }
                    $minimum=[UIntPtr]::Zero; $maximum=[UIntPtr]::Zero; [uint32]$flags=0
                    $verifiedLimit = [T0011WorkingSet]::GetProcessWorkingSetSizeEx($handle, [ref]$minimum, [ref]$maximum, [ref]$flags)
                    if (-not $verifiedLimit -or $maximum.ToUInt64() -ne ($MaxWorkingGiB * 1GB) -or ($flags -band 4) -eq 0) { throw 'Working-set limit verification failed' }
                    Save-GuardEvent @{utc=[DateTime]::UtcNow.ToString('o'); event='resident_limit_applied'; process_id=$child.ProcessId; maximum_working_gib=$MaxWorkingGiB; flags=$flags; weights_and_arguments_unchanged=$true}
                    $limited[$identity]=$true
                } finally { [void][T0011WorkingSet]::CloseHandle($handle) }
            }
            if (-not $lastCpu.ContainsKey($identity) -or $process.CPU -gt $lastCpu[$identity]) { $lastProgress=[DateTime]::UtcNow }
            $lastCpu[$identity]=$process.CPU
            $samples += @{pid=$child.ProcessId; cpu_seconds=$process.CPU; working_gib=[math]::Round($process.WorkingSet64/1GB,3); private_commit_gib=[math]::Round($process.PrivateMemorySize64/1GB,3)}
        }
        $event=@{utc=[DateTime]::UtcNow.ToString('o'); event='sample'; pipeline_pid=$PipelinePid; available_physical_gib=[math]::Round($os.FreePhysicalMemory/1MB,3); available_commit_gib=[math]::Round($os.FreeVirtualMemory/1MB,3); children=$samples; cpu_stalled_seconds=[int]([DateTime]::UtcNow-$lastProgress).TotalSeconds}
        if ($event.available_commit_gib -lt 8 -or $event.available_physical_gib -lt 1 -or $event.cpu_stalled_seconds -gt 900) { $event.needs_attention=$true }
        Save-GuardEvent $event
        Start-Sleep -Seconds 15
    }
} catch {
    Save-GuardEvent @{utc=[DateTime]::UtcNow.ToString('o'); event='guard_error'; error=$_.Exception.Message; needs_attention=$true}
    throw
} finally {
    if (Test-Path -LiteralPath $guardOwner) {
        $owner=Get-Content -LiteralPath $guardOwner -Raw | ConvertFrom-Json
        if ($owner.pid -eq $PID) { Remove-Item -LiteralPath $guardOwner }
    }
}
```

## User-authorized resume after temporary pause

The original source and experiment plan remain frozen. Completed seeds1/2 and
validation stages are verified and retained; interrupted seed3 has no checkpoint.
Resume under the CPU lock using the following wrapper, which acquires the GPU
lock, and start the unchanged 40 GiB memory guard for the verified new pipeline.

### t0011_resume_after_pause.py
```python
"""Acquire GPU lock inside the CPU-locked user-authorized resume."""
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, 'tools/repo/task.py', 'lock', '--task', 'T0011',
                '--wait', '600', 'gpu', '--', sys.executable,
                str(Path(__file__).with_name('t0011_resumed_t0011_train_pipeline_mmap.py'))], check=True)
```
### t0011_resumed_t0011_train_pipeline_mmap.py
```python
"""Resume the user-paused frozen T0011 experiment, retaining completed stages."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
SCRIPTS = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/models/p_features'
BUILD = ROOT / 'build/t0011'
STATE = BUILD / 'pipeline_mmap_state.json'
LOG = BUILD / 'pipeline_mmap.log'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): Path(p).write_text(json.dumps(value, indent=1)+'\n', encoding='utf-8')

plan = read(REPORT/'FROZEN_PLAN.json')
commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
assert commit == plan['source_commit'] == '9a78345172010a207fe89c54af12704b8cc3fddf'
assert subprocess.check_output(['git','branch','--show-current'], text=True).strip() == 'codex/t0011-p-feature-ablation-train-p2-and-p3-at'
for name, key in [('DATA_PARITY.json','data_parity_receipt_sha256'), ('audit.json','audit_sha256'),
                  ('POLICY_CACHE.json','policy_cache_receipt_sha256'), ('CACHE_CONTRACT.json','cache_contract_receipt_sha256')]:
    assert sha(REPORT/name) == plan[key], name
assert sha(ROOT/'inference/models_xxl/MANIFEST.json') == plan['baseline_manifest_sha256']
assert sha(SCRIPTS/'t0011_train_mmap.py') == plan['mmap_launcher_sha256']
assert read(REPORT/'FINALIST_DECISION.json')['p3_finalist'] is True
assert read(REPORT/'FINALIST_DECISION.json')['test_not_used'] is True

expected = {
 ('p2',1): '5c99d94be5efe45b30a1b8c0f0b79e4414d0cb7b77bccabd46dfdeb127ddcbba',
 ('p3',1): 'b3f863a0e1810a02bdaa87543d5adf1185b2269d99bce99eaa92c816becd9def',
 ('p2',2): '7026126306d765078e36b74fd2727297c58567a86da459f585b1a49406c38ac9',
 ('p3',2): 'af1d28b5b11c508e9181c653654e05c1d3d57e2c996c979861e6237253a6f1db',
}
reference = read(ROOT/'runs/pf_p2_xxl_s1/config.json')
verified = []
for (family,seed), digest in expected.items():
    run = ROOT/f'runs/pf_{family}_xxl_s{seed}'
    cfg = read(run/'config.json')
    assert sha(run/'model.npz') == digest, str(run)
    assert cfg['data'] == reference['data'] and cfg['hidden'] == reference['hidden']
    for key, value in reference['args'].items():
        wanted = seed if key == 'seed' else family if key == 'policy_family' else value
        if key != 'out': assert cfg['args'][key] == wanted, (family,seed,key)
    assert cfg['seed'] == seed
    verified.append({'family':family,'seed':seed,'model_sha256':digest,'config_sha256':sha(run/'config.json')})
for seed in (1,2):
    assert read(REPORT/f'val_s{seed}.json')['compare'][0]['strata']['all']['joint_nll']['hi'] < 0

prior = read(REPORT/'PIPELINE.json')
assert prior['plan'] == plan
assert [s['stage'] for s in prior['steps']] == ['train_p2_s1','train_p3_s1','compare_val_s1',
                                              'train_p2_s2','train_p3_s2','compare_val_s2','train_p2_s3']
assert all(s['exit_code'] == 0 for s in prior['steps'][:-1])
assert prior['steps'][-1]['exit_code'] != 0
archive = BUILD/'paused_20261011T0755'
assert not archive.exists(), 'Resume already initiated; inspect state before retrying'
assert not (ROOT/'runs/pf_p2_xxl_s3/model.npz').exists()
assert not (ROOT/'runs/pf_p2_xxl_s3/config.json').exists()
assert not (ROOT/'runs/pf_p3_xxl_s3').exists()
archive.mkdir()
shutil.copyfile(REPORT/'PIPELINE.json', archive/'PIPELINE.json')
shutil.copyfile(STATE, archive/STATE.name)
shutil.copyfile(LOG, archive/LOG.name)
partial = (ROOT/'runs/pf_p2_xxl_s3').resolve()
assert partial.parent == (ROOT/'runs').resolve()
shutil.move(str(partial), str(archive/'pf_p2_xxl_s3'))
shutil.copyfile(archive/'pf_p2_xxl_s3/log.jsonl', REPORT/'P2_S3_INTERRUPTED_LOG.jsonl')
write(REPORT/'RESUME_AFTER_PAUSE.json', {
    'requested_by_user':True, 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'source_commit':commit, 'preserved_completed_runs':verified,
    'preserved_validation':[{'seed':s,'sha256':sha(REPORT/f'val_s{s}.json')} for s in (1,2)],
    'interrupted_stage':'P2 seed3 calibration after epoch8; no checkpoint saved',
    'recovery':'Rerun P2 seed3 from the beginning, then P3 seed3 and remaining comparisons/exports.',
    'data_arguments_and_selection_unchanged':True, 'max_working_gib':40,
    'previous_failed_step':prior['steps'][-1], 'resume_driver_sha256':sha(__file__)
})
steps = prior['steps'][:-1]
cli = []
for key,value in plan['arguments'].items(): cli += ['--'+key.replace('_','-'),str(value)]

def step(stage,cmd):
    start = time.monotonic()
    entry = {'stage':stage,'command':['python',*[Path(x).name if x.endswith('.py') and Path(x).is_absolute() else x for x in cmd[1:]]],
             'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(STATE,{'stage':stage,'steps':steps,'pid':os.getpid(),'source_commit':commit,'resumed_after_user_pause':True})
    print('[T0011] '+stage,flush=True)
    with LOG.open('a',encoding='utf-8') as log:
        log.write(json.dumps(entry)+'\n'); log.flush()
        proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
        code = proc.wait()
    entry.update(exit_code=code,wall_seconds=round(time.monotonic()-start,3))
    steps.append(entry)
    write(REPORT/'PIPELINE.json',{'plan':plan,'steps':steps})
    if code: raise SystemExit(f'{stage} failed: exit {code}; inspect preserved state before recovery')

for family in ('p2','p3'):
    run = Path(f'runs/pf_{family}_xxl_s3')
    assert not run.exists()
    step(f'train_{family}_s3',[sys.executable,str(SCRIPTS/'t0011_train_mmap.py'),'--model','general',
         '--shards','data/shards_l3','--capacity','XXL','--policy-family',family,'--seed','3','--out',str(run),*cli])
    dest = REPORT/'runs'/f'{family}_s3'
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','log.jsonl'): shutil.copyfile(run/name,dest/name)

def compare(seed,split):
    assert not (REPORT/f'{split}_s{seed}.json').exists()
    step(f'compare_{split}_s{seed}',[sys.executable,'-m','training.compare_policy','--shards','data/shards_l3',
         '--split',split,'--run',f'p2=runs/pf_p2_xxl_s{seed}','--run',f'p3=runs/pf_p3_xxl_s{seed}',
         '--compare','p3:p2','--out',str(Path('reports/models/p_features')/f'{split}_s{seed}')])

compare(3,'val')
for seed in (1,2,3): compare(seed,'test')
for family in ('p2','p3'):
    assert not Path(f'inference/models_pf_{family}xxl_vm').exists()
    step(f'export_{family}_seed1',[sys.executable,'-m','training.export_mixed','--policy-run',f'runs/pf_{family}_xxl_s1',
         '--value-tier','m','--out',f'inference/models_pf_{family}xxl_vm'])
write(STATE,{'stage':'complete','steps':steps,'pid':os.getpid(),'source_commit':commit,'seeds':[1,2,3]})
print('[T0011] training, comparisons and exports complete; review reports before PR',flush=True)
```

## Completed-run receipt verification

Run the following script with family and seed (for example, python verify_complete_run.py p3 3). It compares optional metadata presence with the same-family seed1 config: legacy spec_version is absent in both p3 configs. Only completed artifacts are read and receipts written.

```python
"""Verify a completed T0011 run against its frozen same-family seed1 contract."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

family, seed = sys.argv[1], int(sys.argv[2])
assert family in ('p2','p3') and seed in (2,3)
report = Path('reports/models/p_features')
run = Path(f'runs/pf_{family}_xxl_s{seed}')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg = read(run/'config.json')
ref = read(f'runs/pf_{family}_xxl_s1/config.json')
assert cfg['seed'] == seed
assert cfg['args']['out'] == str(run)
assert {k:v for k,v in cfg['args'].items() if k not in ('seed','out')} == {k:v for k,v in ref['args'].items() if k not in ('seed','out')}
for key in ('data','hidden','rows','features','repo_commit','capacity','model','kind','init'):
    assert cfg[key] == ref[key], key
assert ('spec_version' in cfg) == ('spec_version' in ref)
assert cfg.get('spec_version') == ref.get('spec_version')
assert cfg['repo_commit'] == subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
logs = [json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
assert [x['epoch'] for x in logs if 'epoch' in x] == list(range(1,9))
assert logs[-1]['stage'] == 'calibrated'
for head in ('joint','act','target'): assert cfg['beta'][head] == logs[-1]['beta_'+head]
pipeline = read(report/'PIPELINE.json')
stage = f'train_{family}_s{seed}'
step = [x for x in pipeline['steps'] if x['stage']==stage]
assert len(step)==1 and step[0]['exit_code']==0
for name in ('config.json','log.jsonl'):
    assert sha(run/name)==sha(report/'runs'/f'{family}_s{seed}'/name)
receipt = {'stage':stage,'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'arguments_identical_except_seed_output':True,'data_hidden_rows_source_identical':True,
           'model_sha256':sha(run/'model.npz'),'config_sha256':sha(run/'config.json'),
           'beta':cfg['beta'],'calibrated':logs[-1],'step':step[0]}
(report/f'{family.upper()}_S{seed}_VERIFIED.json').write_text(json.dumps(receipt,indent=1)+'\n')
print(json.dumps({'stage':stage,'model_sha256':receipt['model_sha256'],'verified':True}))
```

## Export regeneration and native parity

Run the following script from the repository root after the frozen pipeline completes.

```python
"""Verify frozen T0011 exports, regeneration and real-weight native parity."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
# Match the repository's required LF delivery bytes on Windows too.
original_write = Path.write_text
def write_lf(self, data, encoding=None, errors=None, newline=None):
    return original_write(self, data, encoding=encoding, errors=errors, newline='\n')
Path.write_text = write_lf
import numpy as np
from fusion import native, quant
from fusion.models import QuantizedTier
from fusion.paths import served_counts
from fusion.schema import layout
from tests.test_feature_families import fixture_stream
from training import export_mixed

R = Path('reports/models/p_features')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
state = read('build/t0011/pipeline_mmap_state.json')
assert state['stage']=='complete' and all(x['exit_code']==0 for x in state['steps'])
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==state['source_commit']
L = layout()
checks=[]
for family in ('p2','p3'):
    out = Path(f'inference/models_pf_{family}xxl_vm')
    m = read(out/'MANIFEST.json')['general']
    assert m['features']['policy']['family']==family and m['features']['value']['family']=='v2'
    assert not m['value_approved'] and m['alpha']=={}
    # Profile metadata lists explicit routes; the general-only registry also
    # supplies the fallback, whose complete dispatch is checked below.
    assert m['served_team_counts']==served_counts('general')
    assert m['sources']['policy']['model_sha256']==sha(f'runs/pf_{family}_xxl_s1/model.npz')
    arrays = dict(np.load(out/'weights_general.npz'))
    source_v = dict(np.load('inference/models_m/weights_general.npz'))
    for k,v in arrays.items():
        if k.startswith('value.'):
            np.testing.assert_array_equal(v,source_v[k]); assert v.dtype==source_v[k].dtype
    assert sha(out/m['header'])==m['header_sha256']
    assert sha(out/'weights_general.npz')==m['npz_sha256']
    rerun = Path('build/t0011/export_regeneration')/family
    export_mixed.export_runs(rerun,policy_run=Path(f'runs/pf_{family}_xxl_s1'),value_tier='m')
    regenerated = dict(np.load(rerun/'weights_general.npz'))
    assert arrays.keys()==regenerated.keys()
    for k,v in arrays.items():
        np.testing.assert_array_equal(v,regenerated[k]); assert v.dtype==regenerated[k].dtype
    assert (out/'registry.hpp').read_bytes()==(rerun/'registry.hpp').read_bytes()
    # The exporter dates only the first header line.
    assert (out/m['header']).read_text().split('\n',1)[1]==(rerun/m['header']).read_text().split('\n',1)[1]
    registered = native.run_tool(native.registry_check(out))
    assert registered.returncode==0, registered.stderr.decode()
    registry_lines=registered.stdout.decode().strip().splitlines()
    for team in range(2,16): assert any(x.startswith(f'{team} general ') for x in registry_lines)
    hdr=out/m['header']
    assert native.model_families(hdr)==(family,'v2')
    infer=native.infer_check(hdr,'fusion::models::general')
    runtime=native.model_tool('runtime_check',hdr,'fusion::models::general')
    parity=[]
    py=QuantizedTier(arrays,family,'v2',layout=L)
    for name in ('n02','n04','n15'):
        stream=fixture_stream(name)
        rec=native.extract(stream)
        cpp=native.run_infer_check(infer,stream,len(rec))
        expected=py.raw(rec,L)
        np.testing.assert_array_equal(cpp,expected)
        runtime_result=native.run_tool(runtime,stream.encode())
        assert runtime_result.returncode==0, (runtime_result.stdout+runtime_result.stderr).decode()
        parity.append({'fixture':name,'stream_sha256':hashlib.sha256(stream.encode()).hexdigest(),
                       'rows':len(rec),'all_integer_accumulators_equal':True,
                       'runtime_contract_exit_code':0,'runtime_output':runtime_result.stdout.decode().strip()})
    if family=='p2':
        old=dict(np.load('inference/models_xxl/weights_general.npz'))
        for k,v in arrays.items():
            if not k.startswith('value.'): np.testing.assert_array_equal(v,old[k]); assert v.dtype==old[k].dtype
    checks.append({'family':family,'source_model_sha256':m['sources']['policy']['model_sha256'],
                   'header_sha256':m['header_sha256'],'npz_sha256':m['npz_sha256'],
                   'policy_digest':quant.digest({k:v for k,v in arrays.items() if not k.startswith('value.')}),
                   'regeneration_arrays_and_registry_exact':True,'header_equal_except_date':True,
                   'committed_value_arrays_exact':True,'profile_explicit_team_counts':m['served_team_counts'],
                   'effective_registry_team_counts':list(range(2,16)),
                   'registry_lines':registry_lines,'parity':parity,
                   'p2_legacy_policy_arrays_exact':family=='p2',
                   'infer_binary_sha256':sha(infer),'runtime_binary_sha256':sha(runtime)})
version=subprocess.check_output([native.compiler(),'--version'],text=True).splitlines()[0]
receipt={'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'source_commit':state['source_commit'],'compiler_version':version,
         'compiler_flags':[x.replace(str(Path.cwd()), '${REPO}') for x in native.FLAGS],
         'checks':checks,'scope':'Real seed1 INT16 exports; recorded fixture parity/registry/runtime correctness. No timing or playing-strength claim.'}
(R/'EXPORT_VERIFICATION.json').write_text(json.dumps(receipt,indent=1)+'\n',encoding='utf-8')
print('Both real-weight exports passed regeneration, value-copy, registry and native/Python parity checks.')
```

## Evidence summary generator

Run the following script from the repository root after the frozen pipeline completes.

```python
"""Generate the T0011 evidence summary from completed, frozen receipts."""
import hashlib
import json
import time
from pathlib import Path
original_write=Path.write_text
def write_lf(self,data,encoding=None,errors=None,newline=None):
    return original_write(self,data,encoding=encoding,errors=errors,newline='\n')
Path.write_text=write_lf

R = Path('reports/models/p_features')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = read(R/'FROZEN_PLAN.json')
state = read('build/t0011/pipeline_mmap_state.json')
assert state['stage'] == 'complete', 'Wait for the frozen pipeline to complete'
assert all(s['exit_code']==0 for s in state['steps'])
decision = read(R/'FINALIST_DECISION.json')
assert decision['p3_finalist'] and decision['test_not_used']
comparisons = []
for split in ('val','test'):
    for seed in (1,2,3):
        d = read(R/f'{split}_s{seed}.json')
        assert d['split']==split and d['runs']['p2']['seed']==d['runs']['p3']['seed']==seed
        c = d['compare'][0]
        assert (c['x'],c['y'])==('p3','p2')
        comparisons.append({'split':split,'seed':seed,'rows':d['rows'],'maps':d['maps'],
                            'p2':d['runs']['p2']['strata']['all'],'p3':d['runs']['p3']['strata']['all'],
                            'difference':c['strata']['all'],'strata':c['strata'],
                            'report_sha256':sha(R/f'{split}_s{seed}.json')})
runs = []
for family in ('p2','p3'):
    ref = read(f'runs/pf_{family}_xxl_s1/config.json')
    for seed in (1,2,3):
        run = Path(f'runs/pf_{family}_xxl_s{seed}')
        cfg = read(run/'config.json')
        for k in ('hidden','data','rows','features','repo_commit'):
            assert cfg[k]==ref[k], (family,seed,k)
        assert cfg['repo_commit']==plan['source_commit']
        assert {k:v for k,v in cfg['args'].items() if k not in ('seed','out')} == {k:v for k,v in ref['args'].items() if k not in ('seed','out')}
        log = [json.loads(x) for x in (run/'log.jsonl').read_text().splitlines()]
        epochs = [x for x in log if 'epoch' in x]
        assert len(epochs)==8 and epochs[-1]['optimizer_steps']==188704 and log[-1]['stage']=='calibrated'
        step = [x for x in state['steps'] if x['stage']==f'train_{family}_s{seed}']
        assert len(step)==1
        runs.append({'family':family,'seed':seed,'model_sha256':sha(run/'model.npz'),
                     'config_sha256':sha(run/'config.json'),'beta':cfg['beta'],'rows':cfg['rows'],
                     'epochs':epochs,'calibrated':log[-1],'wall_seconds':step[0]['wall_seconds'],
                     'peak_allocated_gb':cfg['backend']['peak_allocated_gb'],
                     'peak_reserved_gb':cfg['backend']['peak_reserved_gb']})
assert runs[0]['model_sha256']==plan['baseline_policy_sha256']
exports = []
for family in ('p2','p3'):
    out = Path(f'inference/models_pf_{family}xxl_vm')
    man = read(out/'MANIFEST.json')
    assert list(man)==['general']
    m = man['general']
    for name,field in [(m['header'],'header_sha256'),('weights_general.npz','npz_sha256')]:
        assert sha(out/name)==m[field]
    exports.append({'family':family,'path':str(out).replace('\\','/'),
                    'manifest_sha256':sha(out/'MANIFEST.json'),'entry':m})
summary = {'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'source_commit':plan['source_commit'],'finalist_decision':decision,
           'comparisons':comparisons,'runs':runs,'exports':exports,
           'interpretation':'Offline prediction evidence only; equal-full-turn-time playing strength remains T0013.'}
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=1)+'\n',encoding='utf-8')
cell = lambda v: '-' if v is None else f"{v['mean']:+.6f} [{v['lo']:+.6f}, {v['hi']:+.6f}] {v['verdict']}"
lines = ['# T0011: P2/P3 XXL feature ablation', '',
         'Generated by `t0011_summarize.py` (source embedded in REPRODUCE.md) from the frozen run and comparison receipts.', '',
         'P3 passes the prespecified validation finalist rule. The three held-out test comparisons below measure opponent prediction; they do not certify stronger play or a 50 ms deployment budget. T0013 supplies that equal-full-turn-time baseline.', '',
         '## Controlled experiment', '',
         f"Measured source: `{plan['source_commit']}`. Same 21,000 raw matches, layout-3 rebuild, 30,049,121 extracted rows, and exact legacy-column/label parity. Each run uses all 24,153,825 eligible training rows and 2,981,919 validation rows, with no row reduction or test-based fitting.", '',
         'The XXL hidden widths are mover256, squad1024/256, act256 and target32. P3 adds the combat input block; the target feature schema remains unchanged. Arguments follow the committed XXL general checkpoint: eight epochs, batch1024, Adam coupled matrix L2=1e-5, inverse-team-count training weights, cosine LR0.001 to0.0001, label smoothing0.01, auxiliary act/target weights0.5, FP32 and TF32 off. See FROZEN_PLAN.json and the copied configs for all arguments.', '',
         'Seed1 validation selected whether to run both families at seeds2/3 before any test comparison. Exports always use seed1. P2 seed1 reproduces the prior committed XXL general checkpoint SHA256 exactly. Validation-only calibrated temperatures are fixed before test scoring.', '',
         '## Overall paired comparisons', '',
         'Differences are P3 minus P2, with 95% intervals clustered by map. Negative NLL is better. Rows retain the dataset sample weights; action and target losses average the available unit labels within each row. These aggregate weights/conditional denominators differ from the trainer calibration log, so its raw numbers are not interchangeable with this comparison.', '',
         '| Split | Seed | Rows / maps | P2 joint NLL | P3 joint NLL | Joint difference / 95% interval | Act difference / 95% interval | Target difference / 95% interval |',
         '|---|---:|---:|---:|---:|---|---|---|']
for c in comparisons:
    lines.append(f"| {c['split']} | {c['seed']} | {c['rows']} / {c['maps']} | {c['p2']['joint_nll']:.6f} | {c['p3']['joint_nll']:.6f} | {cell(c['difference']['joint_nll'])} | {cell(c['difference']['act_nll'])} | {cell(c['difference']['target_nll'])} |")
lines += ['', '## Held-out seed1 strata', '',
          'These contextual and per-count views are descriptive; many strata overlap. Distance is the extractor\'s nearest-enemy Manhattan distance, not wall-aware path distance. This feature ablation is separate from T0004\'s small/large capacity and decision-change experiment.', '',
          '| Stratum | Joint difference / 95% interval | Act difference / 95% interval | Target difference / 95% interval |',
          '|---|---|---|---|']
test1 = next(c for c in comparisons if c['split']=='test' and c['seed']==1)
for g,c in test1['strata'].items(): lines.append(f"| {g} | {cell(c['joint_nll'])} | {cell(c['act_nll'])} | {cell(c['target_nll'])} |")
lines += ['', 'All three seed-specific full tables are in test_s1.md, test_s2.md and test_s3.md. The seed results reuse the same held-out maps: their intervals must not be pooled as independent observations. No additional temperature, checkpoint or feature choice is made from these test results.', '',
          '## Learning curves and run resources', '',
          '| Family / seed | Final train loss | Best smoothed combined val | Calibrated joint / act / target (trainer weights) | Run wall seconds | CUDA allocated / reserved peak GiB |',
          '|---|---:|---:|---|---:|---:|']
for r in runs:
    e,c=r['epochs'][-1],r['calibrated']
    lines.append(f"| {r['family']} / {r['seed']} | {e['train']:.6f} | {e['val']:.6f} | {c['val_joint']:.6f} / {c['val_act']:.6f} / {c['val_target']:.6f} | {r['wall_seconds']:.3f} | {r['peak_allocated_gb']:.3f} / {r['peak_reserved_gb']:.3f} |")
lines += ['', 'Per-epoch learning curves and exact calibrated temperatures are recorded in runs/*/log.jsonl and config.json, plus SUMMARY.json. CPU preprocessing, calibration and evaluation are included in the whole-run wall times.', '',
          'The original RAM-concatenating loader exhausted host commit before a completed checkpoint; the cache restart preserved samples/order/dtypes and the original trainer. The verified input change uses read-only NPY mappings. The user later paused P2 seed3 after epoch8 before calibration saved a checkpoint; its failed step/log were retained and the entire seed3 run was reproduced after resumption. All eight epoch losses, learning rates and optimizer counts match the interrupted run exactly.', '',
          'The guard applies a 40 GiB resident working-set maximum to verified pipeline children every15 seconds. This cap bounds reclaimable mapped residency; it is not a private-commit cap. Cache residency and resource policy changed across stages, so these run times are not equal-resource architecture speed comparisons. Do not infer capacity throughput superiority from them. See MEMORY_GUARD.json, RESUME_AFTER_PAUSE.json, P2_S3_RESUME_PREFIX_VERIFIED.json and BRANCH_STATUS.md.', '',
          '## Export and delivery', '',
          '| Set | Seed | Policy parameters | Value parameters | Buffer width |',
          '|---|---:|---:|---:|---:|']
for x in exports:
    m=x['entry']; p=sum(m['params'][h] for h in ('mover','squad','act','target'))
    lines.append(f"| {x['path']} | 1 | {p} | {m['params']['value']} | {m['required_buffer_width']} |")
lines += ['', 'Both generated model sets contain the prespecified general seed1 policy serving counts2-15 with the committed V2/M arrays. V remains unapproved and T0013 disables it. These are the existing INT16 export path; T0030 low-bit experiments remain separate. Exact checkpoint, header, array, manifest, feature-family and calibration identities are retained in the manifests and SUMMARY.json.', '',
          'Export integrity, regeneration, native parity and required test results are recorded separately in EXPORT_VERIFICATION.json and CHECKS.md once those checks finish. T0011 results remain on the independent branch and are delivered via PR, with no direct main or automatic task-board push. Follow-on work is listed in FOLLOW_ON_QUEUE.md.']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Generated SUMMARY.json and RESULTS.md from all completed stages.')
```

## Canonical LF delivery

Script used locally from the repository root. Canonical LF transport runs once before verification; historical hashes remain archived.

```python
"""Regenerate Windows exports with LF and preserve transport digests of receipts."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
original_write=Path.write_text
def write_lf(self,data,encoding=None,errors=None,newline=None):
    return original_write(self,data,encoding=encoding,errors=errors,newline='\n')
Path.write_text=write_lf
from training.export_mixed import export_runs
report=Path('reports/models/p_features')
changed=[]
for root in (report,Path('inference/models_pf_p2xxl_vm'),Path('inference/models_pf_p3xxl_vm')):
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.md','.hpp'):
            raw=p.read_bytes(); canonical=raw.replace(b'\r\n',b'\n')
            if raw!=canonical:
                if root==report and p.name not in ('REPRODUCE.md','EXPORT_VERIFICATION.json','SUMMARY.json','RESULTS.md','ARTIFACT_CHECKS.json','BRANCH_STATUS.md'):
                    changed.append({'path':p.as_posix(),'captured_sha256':hashlib.sha256(raw).hexdigest(),
                                    'lf_sha256':hashlib.sha256(canonical).hexdigest()})
                p.write_bytes(canonical)
for family in ('p2','p3'):
    export_runs(Path(f'inference/models_pf_{family}xxl_vm'),
                policy_run=Path(f'runs/pf_{family}_xxl_s1'),value_tier='m')
receipt={'operation':'Lossless CRLF-to-LF text transport required by CONTRIBUTING.md; INT16 weights unchanged.',
         'historical_receipts':'Captured digests describe original measurement bytes. This map records equivalent LF delivery copies; run/config/model digests still refer to uncommitted original runs. Final manifests and verification are regenerated after transport.',
         'files':changed}
(report/'TEXT_TRANSPORT.json').write_text(json.dumps(receipt,indent=1)+'\n',encoding='utf-8')
with (report/'REPRODUCE.md').open('a',encoding='utf-8',newline='\n') as out:
    out.write('\n## Canonical LF delivery\n\nRepository exports were regenerated with LF writes. Historical receipt text is transported losslessly to LF; TEXT_TRANSPORT.json maps original captured hashes to equivalent LF copies. Binary weights and measurement inputs are unchanged. Before final artifact verification, run:\n\n```python\n')
    out.write(Path(__file__).read_text());out.write('```\n')
print('Regenerated LF exports and recorded historical receipt transport hashes.')
```

## Final integrated delivery checks

Script used locally from the repository root. Canonical LF transport runs once before verification; historical hashes remain archived.

```python
"""Run required T0011 delivery checks on the integrated branch under CPU lock."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
original_write=Path.write_text
def write_lf(self,data,encoding=None,errors=None,newline=None):
    return original_write(self,data,encoding=encoding,errors=errors,newline='\n')
Path.write_text=write_lf

report=Path('reports/models/p_features')
build=Path('build/t0011/final_checks'); build.mkdir(parents=True,exist_ok=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
source=json.loads((report/'FROZEN_PLAN.json').read_text())['source_commit']
commands=[
    ['-m','unittest','discover','-s','tests','-t','.'],
    ['-m','unittest','discover','-s','bots/p_only/tests'],
    ['tools/repo/release_bots.py','--out','dist','--check'],
    ['tools/repo/task.py','check'],
]
rows=[]
for i,cmd in enumerate(commands,1):
    start=time.monotonic()
    stdout=build/f'check{i}.stdout'; stderr=build/f'check{i}.stderr'
    print('Starting: python '+' '.join(cmd),flush=True)
    with stdout.open('wb') as out,stderr.open('wb') as err:
        completed=subprocess.run([sys.executable,*cmd],stdout=out,stderr=err)
    row={'command':'python '+' '.join(cmd),'exit_code':completed.returncode,
         'wall_seconds':round(time.monotonic()-start,3),
         'stdout_sha256':hashlib.sha256(stdout.read_bytes()).hexdigest(),
         'stderr_sha256':hashlib.sha256(stderr.read_bytes()).hexdigest(),
         'summary':[(x.replace(str(Path.cwd()),'${REPO}').replace(str(Path.home()).replace('\\','\\\\'),'${USER_HOME}').replace(str(Path.home()),'${USER_HOME}')) for x in (stdout.read_text(errors='replace')+'\n'+stderr.read_text(errors='replace')).strip().splitlines()[-8:]]}
    rows.append(row)
    receipt={'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'delivery_check_commit':head,'measurement_source_commit':source,'checks':rows}
    (report/'CHECKS.json').write_text(json.dumps(receipt,indent=1)+'\n')
    print(f"Finished: exit={completed.returncode}, seconds={row['wall_seconds']}",flush=True)
    if completed.returncode: raise SystemExit(completed.returncode)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==head
lines=['# T0011 delivery checks','',f'Checks ran on integrated branch source `{head}`. Frozen measurement and real-weight native verification use `{source}`. Report-only commits after this check do not change the tested code or models.','',
       'Both exported seed1 policies passed exact regeneration, copied V2/M array equality, actual registry dispatch for every count2-15, integer-accumulator parity with Python, and native runtime contracts on n02/n04/n15 recorded fixtures. See EXPORT_VERIFICATION.json. The manifest served_team_counts field lists explicit legacy profile routes9/11/13/14; the general-only registry also provides the fallback for all counts2-15. Compiler paths use ${REPO} for portability.','',
       'Two preliminary verification-only attempts failed on script import setup and an incorrect assumption about that profile metadata. They did not alter or invalidate training, checkpoints, comparisons or exports; their logs remain in build/t0011/artifact_attempt1 and artifact_attempt2. The corrected verifier is embedded in REPRODUCE.md.','',
       '| Command | Exit | Seconds |','|---|---:|---:|']
for row in rows: lines.append(f"| `{row['command']}` | {row['exit_code']} | {row['wall_seconds']} |")
lines+=['','Exact check summaries and log digests are in CHECKS.json. Full local check logs remain in build/t0011/final_checks and are not committed. Tests supply correctness evidence; their runtimes are not host inference-cost measurements.','']
(report/'CHECKS.md').write_text('\n'.join(lines))
print('All required T0011 delivery checks passed.',flush=True)
```

## Real-weight T0013 carrier compilation

Script used locally from the repository root. Canonical LF transport runs once before verification; historical hashes remain archived.

```python
"""Build the four T0013 carriers with the actual T0011 seed1 exports."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
R=Path('reports/models/p_features');B=Path('build/t0011/final_checks')
names=['c08_pf_p2xxl','c08_pf_p3xxl','v12_pf_p2xxl','v12_pf_p3xxl']
args=['bots/p_only/tools/build.py','--variants','bots/p_only/profiles/variants_families.json','--stats',*names]
start=time.monotonic()
with (B/'carriers.stdout').open('wb') as out,(B/'carriers.stderr').open('wb') as err:
    result=subprocess.run([sys.executable,*args],stdout=out,stderr=err)
assert result.returncode==0,(B/'carriers.stderr').read_text(errors='replace')
m=json.loads(Path('build/p_only/manifest.json').read_text())['variants']
checks=[]
for name in names:
    entry=m[name]
    assert hashlib.sha256(Path(entry['binary']).read_bytes()).hexdigest()==entry['binary_sha256']
    checks.append({'variant':name,'binary_sha256':entry['binary_sha256'],
                   'bundle_sha256':entry['bundle_sha256'],'inputs':entry['inputs']})
receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'command':['python',*args],'exit_code':result.returncode,
         'wall_seconds':round(time.monotonic()-start,3),'variants':checks,
         'scope':'Real-weight compilation only; no timing or strength measurement.'}
(R/'CARRIER_BUILD.json').write_text(json.dumps(receipt,indent=1)+'\n',encoding='utf-8',newline='\n')
with (R/'CHECKS.md').open('a',encoding='utf-8',newline='\n') as out:
    out.write('\nThe four T0013 carriers c08/v12 x p2/p3 also compile with the actual exported seed1 models. Binary, bundle and inlined-input identities are in CARRIER_BUILD.json. This compilation uses the integrated source above and spends no match seed.\n')
print('All four real-weight T0013 carriers compiled successfully.',flush=True)
```

## Staged delivery integrity

Script used locally from the repository root. Canonical LF transport runs once before verification; historical hashes remain archived.

```python
"""Check Git's staged delivery bytes against final export and summary digests."""
import hashlib
import json
import subprocess
from pathlib import Path
R=Path('reports/models/p_features')
read=lambda p:json.loads(p.read_text())
blob=lambda p:subprocess.check_output(['git','show',':'+str(p).replace('\\','/')])
sha=lambda b:hashlib.sha256(b).hexdigest()
for family in ('p2','p3'):
    root=Path(f'inference/models_pf_{family}xxl_vm')
    m=read(root/'MANIFEST.json')['general']
    assert sha(blob(root/m['header']))==m['header_sha256']
    assert sha(blob(root/'weights_general.npz'))==m['npz_sha256']
    check=next(x for x in read(R/'EXPORT_VERIFICATION.json')['checks'] if x['family']==family)
    assert check['header_sha256']==m['header_sha256'] and check['npz_sha256']==m['npz_sha256']
for c in read(R/'SUMMARY.json')['comparisons']:
    assert sha(blob(R/f"{c['split']}_s{c['seed']}.json"))==c['report_sha256']
for x in read(R/'TEXT_TRANSPORT.json')['files']:
    assert sha(blob(x['path']))==x['lf_sha256'],x['path']
print('Staged Git bytes match both final manifests, native verification, six comparisons and historical LF transport receipts.')
```

## Portable check summaries

Script used locally from the repository root. Canonical LF transport runs once before verification; historical hashes remain archived.

```python
"""Portable final check summaries and a checked map of immutable receipt transport."""
import hashlib
import json
import os
from pathlib import Path
R=Path('reports/models/p_features')
scripts=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
write=lambda p,d:p.write_text(json.dumps(d,indent=1)+'\n',encoding='utf-8',newline='\n')
transport=read(R/'TEXT_TRANSPORT.json')
excluded={'REPRODUCE.md','EXPORT_VERIFICATION.json','SUMMARY.json','RESULTS.md','ARTIFACT_CHECKS.json','BRANCH_STATUS.md'}
transport['files']=[x for x in transport['files'] if x['path'].startswith(R.as_posix()+'/') and Path(x['path']).name not in excluded]
for x in transport['files']:
    assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['lf_sha256'],x['path']
write(R/'TEXT_TRANSPORT.json',transport)
checks=read(R/'CHECKS.json')
assert len(checks['checks'])==4 and all(x['exit_code']==0 for x in checks['checks'])
def portable(text):
    for path,label in ((str(Path.cwd()),'${REPO}'),(str(Path.home()),'${USER_HOME}')):
        for variant in (path.replace('\\','\\\\'),path,path.replace('\\','/')):
            text=text.replace(variant,label)
    return text
for row in checks['checks']:
    row['summary']=[portable(x) for x in row['summary']]
write(R/'CHECKS.json',checks)
reproduction=R/'REPRODUCE.md'
text=reproduction.read_text(encoding='utf-8').split('\n## Canonical LF delivery\n')[0]
for title,name in [('Canonical LF delivery','t0011_canonical_delivery.py'),('Final integrated delivery checks','t0011_final_checks.py'),('Real-weight T0013 carrier compilation','t0011_carrier_build.py'),('Staged delivery integrity','t0011_git_integrity.py'),('Portable check summaries','t0011_finalize_receipts.py')]:
    text+=f'\n## {title}\n\nScript used locally from the repository root. Canonical LF transport runs once before verification; historical hashes remain archived.\n\n```python\n'+(scripts/name).read_text(encoding='utf-8')+'```\n'
reproduction.write_text(text,encoding='utf-8',newline='\n')
print('Preserved and verified historical receipt transport; finalized portable check records.')
```
