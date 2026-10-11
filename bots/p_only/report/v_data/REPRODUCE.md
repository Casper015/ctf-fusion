# T0010 pipeline reproduction

Run the saved script below from the frozen repository root under tools/repo/task.py lock --task T0010 cpu. Set AI_TEST_ROOT and CXX for the host; collection uses6 workers. Resume only the same identity-bound data and receipts, retaining completed games and roots. Never rebuild between collection and continuations. No raw data or build output is committed.

```python
"""Frozen T0010 host collection, continuations and ranking; run under CPU lock."""
import collections
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path.cwd()))
from fusion import matchlog
from fusion.paths import engine_path
from training.branch_collect import root_turns

SOURCE='ebda2c4867fb6503f8e28f3bd0a8b434e6affb96'
BRANCH='codex/t0010-frozen-carrier-v-data'
REPORT=Path('bots/p_only/report/v_data')
BUILD=Path('build/t0010')
CONFIG=Path('bots/p_only/profiles/collect_v_data.json')
MANIFEST=Path('build/p_only/manifest.json')
IDENTITIES=Path('data/p_only/v_data/identities.json')
SCHEDULE=Path('data/p_only/v_data/schedule.json')
STATE=BUILD/'pipeline_state.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
utc=lambda:time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
git=lambda *a:subprocess.check_output(['git',*a],text=True).strip()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=1)+'\n',encoding='utf-8',newline='\n')

def freeze_identity():
    cfg=read(CONFIG)
    assert cfg['seed']==20261201 and cfg['teams']==list(range(2,16))
    assert cfg['maps_per_count']==500 and cfg['workers']==6
    assert cfg['first_ms']==2000 and cfg['turn_ms']==50 and cfg['keep_matches']
    assert cfg['arms']==['v12_ponly_xxl'] and cfg['aa']==[] and cfg['comparisons']==[]
    sched=read(SCHEDULE)
    assert len(sched['fixtures'])==7000
    counts=collections.Counter(x['teams'] for x in sched['fixtures'])
    assert dict(counts)=={n:500 for n in range(2,16)},counts
    m=read(MANIFEST)['variants']['v12_ponly_xxl']
    assert sha(m['binary'])==m['binary_sha256']
    return {'source_commit':git('rev-parse','HEAD'),'config_sha256':sha(CONFIG),
            'manifest_sha256':sha(MANIFEST),'schedule_sha256':sha(SCHEDULE),
            'identities_sha256':sha(IDENTITIES),'identity':read(IDENTITIES),
            'carrier_binary_sha256':m['binary_sha256'],'carrier_bundle_sha256':m['bundle_sha256'],
            'compiler':os.environ.get('CXX',''),
            'engine_commit':git('-C',os.environ['AI_TEST_ROOT'],'rev-parse','HEAD'),
            'engine_sha256':sha(engine_path(Path(os.environ['AI_TEST_ROOT']))),
            'policy_manifest_sha256':sha('inference/models_xxl/MANIFEST.json'),
            'config':cfg,'branches':{'roots_per_match':2,'candidates':4,'repeats':1,'workers':6},
            'collection_is_training_data_only':True,'V_enabled':False,'policy_family':'p2'}

def main():
    REPORT.mkdir(parents=True,exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True)
    assert git('branch','--show-current')==BRANCH
    assert git('rev-parse','HEAD')==git('rev-parse',SOURCE)
    state=read(STATE) if STATE.exists() else {'task':'T0010','source_commit':git('rev-parse','HEAD'),
            'started_utc':utc(),'pid':os.getpid(),'steps':[],'stage':'preflight'}
    state['pid']=os.getpid(); write(STATE,state)
    def step(name,args):
        if any(x['stage']==name and x['exit_code']==0 for x in state['steps']):
            print('Verified completed stage retained:',name,flush=True);return
        state['stage']=name;write(STATE,state)
        print(utc(),name,'python',' '.join(args),flush=True)
        start=time.monotonic()
        code=subprocess.call([sys.executable,*args])
        state['steps'].append({'stage':name,'command':['python',*args],'exit_code':code,
                               'finished_utc':utc(),'wall_seconds':round(time.monotonic()-start,3)})
        write(STATE,state)
        if code: state['stage']='failed';write(STATE,state);raise SystemExit(code)
    step('prepare',['bots/p_only/tools/run_screen.py','--config',CONFIG.as_posix(),'--dry-run'])
    frozen=freeze_identity()
    # Do not store the private host compiler path in a committed report.
    frozen['compiler']='g++ (CXX override); version recorded below'
    frozen['compiler_version']=subprocess.check_output([os.environ['CXX'],'--version'],text=True).splitlines()[0]
    receipt=REPORT/'FROZEN_COLLECTION.json'
    if receipt.exists(): assert read(receipt)==frozen,'Identity changed; do not resume across new inputs'
    else: write(receipt,frozen)
    step('collect',['bots/p_only/tools/run_screen.py','--config',CONFIG.as_posix()])
    assert sha(MANIFEST)==frozen['manifest_sha256'] and sha(SCHEDULE)==frozen['schedule_sha256']
    assert sha(IDENTITIES)==frozen['identities_sha256']
    assert sha(engine_path(Path(os.environ['AI_TEST_ROOT'])))==frozen['engine_sha256']
    m=read(MANIFEST)['variants']['v12_ponly_xxl']; assert sha(m['binary'])==frozen['carrier_binary_sha256']
    screen=read(REPORT/'screen.json')
    assert screen['fixtures']==7000 and not screen['incomplete'] and not screen['stale_rows']
    assert screen['arms']['v12_ponly_xxl']['games']==7000
    matches=list(matchlog.iter_matches('data/p_only/v_data/matches'))
    assert len(matches)==7000,len(matches)
    step('branches',['-m','training.branch_collect','--matches','data/p_only/v_data/matches',
         '--carrier','v12_ponly_xxl','--out','data/branches/v12_ponly_xxl',
         '--roots-per-match','2','--candidates','4','--repeats','1','--workers','6'])
    assert sha(MANIFEST)==frozen['manifest_sha256'] and sha(m['binary'])==frozen['carrier_binary_sha256']
    step('rank',['-m','training.branch_dataset','--branches','data/branches/v12_ponly_xxl',
         '--out','data/rank/v12_ponly_xxl'])
    expected=sum(len(root_turns(matchlog.load(p),2)) for p in matches)
    roots=list(Path('data/branches/v12_ponly_xxl').rglob('*.json.gz'))
    assert len(roots)==expected,(len(roots),expected)
    dropped=collections.Counter();prefix=collections.Counter();faults=0;prefix_faults=0;continuations=0
    for p in roots:
        with gzip.open(p,'rt',encoding='utf-8') as f: row=json.load(f)
        if 'dropped' in row: dropped[row['dropped']]+=1;continue
        for b in row['branches']:
            prefix[b['prefix_mismatch']]+=1;prefix_faults+=b['prefix_faults']
            faults+=len(b['faults']);continuations+=1
    index=read('data/rank/v12_ponly_xxl/index.json')
    pairs={k:{'pairs':v['pairs'],'resolved_pairs':v['resolved_pairs'],
              'uncertain_pairs':v['pairs']-v['resolved_pairs']} for k,v in index['stats'].items()}
    result={'completed_utc':utc(),'source_commit':state['source_commit'],'collection':screen['arms']['v12_ponly_xxl'],
            'kept_matches':len(matches),'expected_roots':expected,'root_files':len(roots),
            'dropped_roots':dict(dropped),'continuations':continuations,'prefix_mismatch_distribution':dict(prefix),
            'prefix_faults':prefix_faults,'continuation_faults':faults,'rank':index,'pair_counts':pairs,
            'rank_index_sha256':sha('data/rank/v12_ponly_xxl/index.json'),'steps':state['steps']}
    write(REPORT/'COLLECTION_RESULT.json',result)
    state['stage']='complete';state['completed_utc']=utc();write(STATE,state)
    lines=['# T0010 frozen-carrier V training data','',
           f"Source `{state['source_commit']}`. Training-only seed20261201; old p2 v12_ponly_xxl, models_xxl, V off, frozen reference roster, official2000/50 ms limits and6 workers.",'',
           f"Collected{len(matches)} matches, faults{result['collection']['faults']}. Root files{len(roots)}/{expected}, dropped{sum(dropped.values())}; continuations{continuations}. Prefix mismatch distribution and reasons are in COLLECTION_RESULT.json, alongside resolved/uncertain ranking pairs by map split.",'',
           'This single-arm data collection is not a paired strength comparison. Match and branch data remain local under data/. No carrier binary was rebuilt between collection and continuation. All source, binary, roster, runner, config and schedule identities are retained in FROZEN_COLLECTION.json, identities.json and manifest.json.','',
           'The user requires independent branches and PRs, so automatic main/task-board updates are disabled. Required delivery checks and PR follow completion. T0012 may use this frozen data after it is verified.','']
    (REPORT/'COLLECTION.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('T0010 collection, continuations and ranking completed.',flush=True)

if __name__=='__main__': main()
```

When the verified worktree and host process use different accounts, launch with process-scoped GIT_CONFIG_COUNT=1, GIT_CONFIG_KEY_0=safe.directory and GIT_CONFIG_VALUE_0 set to that exact worktree. No global/wildcard trust is required.

## Verified post-collection handoff

The original controller checks the full build manifest, whose updated/compile_seconds metadata is rewritten by the repeated preparation build. A queued continuation waits for the CPU lock, requires completed collection, verifies every other frozen field plus actual inlined sources/runner, and ignores only those volatile metadata fields. It preserves the original receipt and skips the completed preparation/collection stages; no game is replayed. Set T0010_PYTHON to the same verified interpreter.

```python
"""Resume after collection, adapting only volatile build-manifest metadata."""
import importlib.util
import hashlib
import json
import subprocess
import time
from pathlib import Path

def main():
    spec=importlib.util.spec_from_file_location('t0010_frozen_driver',Path(__file__).with_name('t0010_pipeline.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    state=m.read(m.STATE)
    assert any(x['stage']=='collect' and x['exit_code']==0 for x in state['steps']), 'Collection is not verified complete'
    captured=m.read(m.REPORT/'FROZEN_COLLECTION.json')
    actual=m.freeze_identity()
    actual['compiler']='g++ (CXX override); version recorded below'
    actual['compiler_version']=subprocess.check_output([m.os.environ['CXX'],'--version'],text=True).splitlines()[0]
    assert {k:v for k,v in actual.items() if k!='manifest_sha256'}=={k:v for k,v in captured.items() if k!='manifest_sha256'}, 'Frozen experimental identity changed'
    manifest=m.read(m.MANIFEST)
    from bots.p_only.tools.screen_runner import runner_digest
    assert runner_digest()==captured['identity']['v12_ponly_xxl']['components']['runner_sha256']
    for entry in manifest['variants'].values():
        for source in entry['inputs']: assert m.sha(source['path'])==source['sha256'],source['path']
    # build.py rewrites updated and compile_seconds even when the existing binary
    # is reused. All measurement-bearing fields remain part of this digest.
    manifest.pop('updated',None)
    for entry in manifest['variants'].values(): entry.pop('compile_seconds',None)
    stable=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    m.write(m.REPORT/'MANIFEST_METADATA_ADAPTER.json',{'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'captured_raw_manifest_sha256':captured['manifest_sha256'],'collection_raw_manifest_sha256':actual['manifest_sha256'],
        'stable_manifest_sha256':stable,'ignored_metadata_fields':['updated','variants.*.compile_seconds'],
        'all_other_frozen_fields_exact':True,'runner_and_inlined_sources_verified':True,
        'reason':'Preparation and collection reuse the same binary; run_screen rebuild writes volatile metadata. Preserve the original receipt and skip both completed stages.'})
    original_sha=m.sha; original_read=m.read
    def stable_sha(path):
        if Path(path).resolve()==m.MANIFEST.resolve():
            value=original_read(path);value.pop('updated',None)
            for entry in value['variants'].values():entry.pop('compile_seconds',None)
            return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        return original_sha(path)
    def adapted_read(path):
        value=original_read(path)
        if Path(path).resolve()==(m.REPORT/'FROZEN_COLLECTION.json').resolve():
            value['manifest_sha256']=stable
        return value
    m.sha=stable_sha;m.read=adapted_read
    m.main()

if __name__=='__main__': main()
```

```powershell
$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$self=Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent=Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($parent.CommandLine -notmatch 'task.py lock.*T0010.*t0010_after_collection.ps1') { throw 'Unverified lock parent' }
$lease=Get-Content -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/cpu.json') -Raw | ConvertFrom-Json
if ($lease.pid -ne $parent.ProcessId -or $lease.task -ne 'T0010') { throw 'Wrong CPU lock lease' }
$created=$parent.CreationDate.ToUniversalTime().ToString('o')
@{pipeline_pid=$parent.ProcessId;created_utc=$created;source_commit='ebda2c4867fb6503f8e28f3bd0a8b434e6affb96';phase='post_collection'} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0010/pipeline_pid.json' -Encoding utf8
$watch=Start-Process -FilePath 'pwsh.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','../../t0010_resource_monitor.ps1',
    '-RepoPath',$root,'-PipelinePid',[string]$parent.ProcessId,'-CreationUtc',$created,'-ExpectedCommandTag','t0010_after_collection.ps1') `
    -WindowStyle Hidden -RedirectStandardOutput 'build/t0010/resource_monitor_successor.stdout' `
    -RedirectStandardError 'build/t0010/resource_monitor_successor.stderr' -PassThru
@{monitor_pid=$watch.Id;created_utc=$watch.StartTime.ToUniversalTime().ToString('o');pipeline_pid=$parent.ProcessId} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0010/resource_monitor_pid.json' -Encoding utf8
if (-not $env:T0010_PYTHON) { throw 'T0010_PYTHON must name the verified Python executable' }
& $env:T0010_PYTHON ../../t0010_after_collection_pipeline.py
exit $LASTEXITCODE
```

## Passive resource observer

Run this PowerShell observer outside the repository with the verified lock-wrapper PID and creation timestamp. It only writes build/t0010 state/log records; it never changes resource settings or terminates processes.

```powershell
param([Parameter(Mandatory=$true)][string]$RepoPath,
      [Parameter(Mandatory=$true)][int]$PipelinePid,
      [Parameter(Mandatory=$true)][string]$CreationUtc,
      [ValidateSet('t0010_pipeline.py','t0010_after_collection.ps1')][string]$ExpectedCommandTag='t0010_pipeline.py')
$ErrorActionPreference='Stop'
$root = (Resolve-Path -LiteralPath $RepoPath).Path
$output = Join-Path $root 'build/t0010'
$expected = [datetime]::Parse($CreationUtc).ToUniversalTime()
$encoding = [System.Text.UTF8Encoding]::new($false)
function Write-Sample($row) {
    $json = $row | ConvertTo-Json -Depth 5 -Compress
    [System.IO.File]::WriteAllText((Join-Path $output 'resource_state.json'),$json+"`n",$encoding)
    [System.IO.File]::AppendAllText((Join-Path $output 'resource_monitor.jsonl'),$json+"`n",$encoding)
}
while ($true) {
    try {
        $all = @(Get-CimInstance Win32_Process)
        $pipeline = $all | Where-Object ProcessId -eq $PipelinePid
        $utc = (Get-Date).ToUniversalTime().ToString('o')
        if (-not $pipeline) { Write-Sample @{utc=$utc;event='pipeline_exited';pipeline_pid=$PipelinePid}; break }
        if ([math]::Abs(($pipeline.CreationDate.ToUniversalTime()-$expected).TotalSeconds) -gt 1 -or
            $pipeline.CommandLine -notmatch ('task.py lock.*T0010.*'+[regex]::Escape($ExpectedCommandTag))) {
            Write-Sample @{utc=$utc;event='identity_mismatch';pipeline_pid=$PipelinePid;attention=$true}; break
        }
        $ids = [System.Collections.Generic.HashSet[int]]::new()
        [void]$ids.Add($PipelinePid)
        do {
            $before=$ids.Count
            foreach ($process in $all) {
                if ($ids.Contains([int]$process.ParentProcessId)) { [void]$ids.Add([int]$process.ProcessId) }
            }
        } while ($ids.Count -gt $before)
        $processes = @()
        foreach ($process in $all) {
            if ($ids.Contains([int]$process.ProcessId)) {
                $live=Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
                if ($live) { $processes+=@{pid=$live.Id;parent_pid=$process.ParentProcessId;
                    private_gib=[math]::Round($live.PrivateMemorySize64/1GB,3);
                    working_gib=[math]::Round($live.WorkingSet64/1GB,3);cpu_seconds=$live.CPU} }
            }
        }
        $os=Get-CimInstance Win32_OperatingSystem
        $memory=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
        $free=[math]::Round($os.FreePhysicalMemory/1MB,3)
        $commit=[math]::Round(($memory.CommitLimit-$memory.CommittedBytes)/1GB,3)
        Write-Sample @{utc=$utc;event='sample';pipeline_pid=$PipelinePid;monitor_pid=$PID;
                      physical_available_gib=$free;commit_available_gib=$commit;
                      attention=($free -lt 4 -or $commit -lt 8);processes=$processes;
                      policy='passive_only_no_kill_no_restart_no_resource_change'}
    } catch {
        Write-Sample @{utc=(Get-Date).ToUniversalTime().ToString('o');event='monitor_error';attention=$true;error=$_.Exception.Message}
    }
    Start-Sleep -Seconds 15
}
```

## Completed-artifact audit and report repair

All four stages exited successfully. The original finalizer assumed the all/dropped_roots metadata row also had pair counts. Keep the original traceback and state, then run the following script under the T0010 CPU lock. It verifies complete schedule/root coverage, frozen inputs, map splits and ranking arrays; it repairs reports only and never launches a completed measurement stage. The first audit attempt used an incorrect fixture field name and exited before writing results; the saved version uses fixture_id plus the runner arm key.

```python
"""Audit completed frozen T0010 artifacts and repair report-only finalization.

Run from the T0010 worktree under its CPU lock. Never launches matches,
continuations, extraction, ranking, training, export or a carrier rebuild.
"""
import collections
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from fusion import matchlog
from training.branch_collect import root_turns
from training.dataset import split_of
import t0010_pipeline as original

REPORT = Path('bots/p_only/report/v_data')
BUILD = Path('build/t0010')
RANK = Path('data/rank/v12_ponly_xxl')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write(p, d):
    Path(p).write_text(json.dumps(d, indent=1) + '\n', encoding='utf-8', newline='\n')


def main():
    start = time.monotonic()
    state = read(BUILD / 'pipeline_state.json')
    assert {s['stage'] for s in state['steps'] if s['exit_code'] == 0} == {'prepare', 'collect', 'branches', 'rank'}
    if not (BUILD / 'pipeline_state_before_finalize.json').exists():
        write(BUILD / 'pipeline_state_before_finalize.json', state)
    frozen = read(REPORT / 'FROZEN_COLLECTION.json')
    actual = original.freeze_identity()
    actual['compiler'] = 'g++ (CXX override); version recorded below'
    actual['compiler_version'] = subprocess.check_output([os.environ['CXX'], '--version'], text=True).splitlines()[0]
    assert {k: v for k, v in actual.items() if k != 'manifest_sha256'} == {k: v for k, v in frozen.items() if k != 'manifest_sha256'}
    adapter = read(REPORT / 'MANIFEST_METADATA_ADAPTER.json')
    assert actual['manifest_sha256'] == adapter['collection_raw_manifest_sha256']
    screen = read(REPORT / 'screen.json')
    assert screen['fixtures'] == 7000 and not screen['incomplete'] and not screen['stale_rows']
    matches = list(matchlog.iter_matches('data/p_only/v_data/matches'))
    assert len(matches) == 7000
    schedule = read('data/p_only/v_data/schedule.json')
    arm_key = hashlib.sha256(b'v12_ponly_xxl').hexdigest()[:10]
    fixtures = {f"{x['fixture_id']}-{arm_key}": x for x in schedule['fixtures']}
    expected, match_maps, match_counts, seen, faults = set(), {}, collections.Counter(), set(), 0
    for i, p in enumerate(matches):
        m = matchlog.load(p)
        assert m['match_id'] not in seen
        fx = fixtures[m['match_id']]
        assert (m['map_seed'], m['flag_seed'], m['teams']) == (fx['map_seed'], fx['flag_seed'], fx['teams'])
        assert m['seats'][fx['seat']]['name'] == '@candidate'
        seen.add(m['match_id'])
        assert m['engine_sha256'] == frozen['engine_sha256'] and m['ai_test_commit'] == frozen['engine_commit']
        assert m['limits_ms'] == [2000, 50]
        candidate = [s for s in m['seats'] if s['name'] == '@candidate']
        assert len(candidate) == 1 and candidate[0]['sha256'] == frozen['carrier_binary_sha256']
        match_maps[m['match_id']] = m['map_seed']
        match_counts[m['teams']] += 1
        faults += len(m['faults'])
        expected.update((m['match_id'], t) for t in root_turns(m, 2))
        if i % 1000 == 0: print('audit matches', i, flush=True)
    assert dict(match_counts) == {n: 500 for n in range(2, 16)}
    assert len(expected) == 14000 and faults == 0
    assert set(fixtures) == seen
    roots = sorted(Path('data/branches/v12_ponly_xxl').rglob('*.json.gz'))
    assert len(roots) == len(expected)
    root_keys, dropped, prefix = set(), collections.Counter(), collections.Counter()
    prefix_faults = continuation_faults = continuations = terminal_candidates = 0
    retained_maps = collections.defaultdict(set)
    root_split = collections.Counter()
    root_manifest = hashlib.sha256()
    for i, p in enumerate(roots):
        raw = p.read_bytes()
        root_manifest.update((p.as_posix() + ' ' + hashlib.sha256(raw).hexdigest() + '\n').encode())
        row = json.loads(gzip.decompress(raw).decode('utf-8'))
        key = (row['match_id'], row['turn'])
        assert key in expected and key not in root_keys
        root_keys.add(key)
        if 'dropped' in row:
            dropped[row['dropped']] += 1
            continue
        assert row['map_seed'] == match_maps[row['match_id']]
        assert row['carrier_sha256'] == frozen['carrier_binary_sha256']
        assert len(row['candidates']) == 4 and len(row['branches']) == 4
        assert len({(b['candidate'], b['repeat']) for b in row['branches']}) == 4
        split = split_of(f"map-{row['map_seed']}")
        retained_maps[split].add(row['map_seed'])
        root_split[split] += 1
        for b in row['branches']:
            assert b['repeat'] == 0 and 0 <= b['candidate'] < 4
            assert np.isfinite(b['outcome']) and 0 <= b['outcome'] <= 1
            prefix[b['prefix_mismatch']] += 1
            prefix_faults += b['prefix_faults']
            continuation_faults += len(b['faults'])
            terminal_candidates += b['next_block'] is None
            continuations += 1
        if i % 2000 == 0: print('audit roots', i, flush=True)
    assert root_keys == expected
    index = read(RANK / 'index.json')
    assert index['files'] == 14000 and index['stats']['all']['dropped_roots'] == sum(dropped.values())
    assert index['options'] == {'tie_steps': 1.0, 'tie_z': 2.0}
    assert index['schema']['record_width'] == 923 and index['schema']['layout_version'] == 3
    pair_counts, npz_checks = {}, {}
    for split in ('train', 'val', 'test'):
        st = index['stats'][split]
        assert st['roots'] == root_split[split] and st['pairs'] == 6 * st['roots']
        p = RANK / split / 'rank.npz'
        with np.load(p, allow_pickle=False) as z:
            a, b, y, w = (z[k] for k in ('pair_a', 'pair_b', 'pair_y', 'pair_w'))
            cr, cy, rm = (z[k] for k in ('cand_root', 'cand_y', 'root_map'))
            assert len(a) == st['pairs'] and len(rm) == st['roots']
            assert z['cand_rec'].shape == (4 * st['roots'], 923) and z['cand_rec'].dtype == np.int16
            assert np.all(z['cand_n'] == 1) and np.all(np.isfinite(cy)) and np.all((0 <= cy) & (cy <= 1))
            assert np.all((a >= 0) & (a < len(cy)) & (b >= 0) & (b < len(cy)))
            assert np.all(cr[a] == cr[b]) and np.all(a < b)
            assert set(np.unique(y)) <= {-1, 0, 1}
            assert np.count_nonzero(y >= 0) == st['resolved_pairs']
            assert np.all(w[y < 0] == 0) and np.all(w[y >= 0] > 0)
            diff = cy[a] - cy[b]
            assert np.all(diff[y == 1] > 0) and np.all(diff[y == 0] < 0)
            margin = .5 / np.maximum(z['root_teams'][cr[a]] - 1, 1)
            assert np.all(np.abs(diff[y < 0]) <= margin[y < 0] + 1e-6)
            assert np.all(np.abs(diff[y >= 0]) >= margin[y >= 0] - 1e-6)
            per_root = np.bincount(cr[a], weights=w, minlength=st['roots'])
            resolved_roots = np.bincount(cr[a][y >= 0], minlength=st['roots']) > 0
            assert np.allclose(per_root[resolved_roots], 1, atol=1e-6)
            assert np.all(per_root[~resolved_roots] == 0)
            assert set(map(int, rm)) == retained_maps[split]
            assert all(split_of(f'map-{int(m)}') == split for m in rm)
            npz_checks[split] = {'sha256': sha(p), 'roots': len(rm), 'candidates': len(cy),
                                 'maps': len(set(map(int, rm))), 'resolved_roots': int(resolved_roots.sum()),
                                 'all_uncertain_roots': int((~resolved_roots).sum()), 'labels_weights_shapes_split_verified': True}
        pair_counts[split] = {'pairs': st['pairs'], 'resolved_pairs': st['resolved_pairs'],
                              'uncertain_pairs': st['pairs'] - st['resolved_pairs']}
    assert all(not retained_maps[a] & retained_maps[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    assert sum(root_split.values()) + sum(dropped.values()) == 14000
    result = {'completed_utc': utc(), 'source_commit': state['source_commit'],
              'collection': screen['arms']['v12_ponly_xxl'], 'kept_matches': len(matches),
              'expected_roots': len(expected), 'root_files': len(roots), 'dropped_roots': dict(dropped),
              'continuations': continuations, 'terminal_candidates': terminal_candidates,
              'prefix_mismatch_distribution': dict(prefix), 'prefix_faults': prefix_faults,
              'continuation_faults': continuation_faults, 'rank': index, 'pair_counts': pair_counts,
              'rank_index_sha256': sha(RANK / 'index.json'), 'steps': state['steps'],
              'audit': {'match_counts': dict(match_counts), 'match_faults': faults,
                        'exact_schedule_and_root_key_coverage': True, 'map_splits_disjoint': True,
                        'frozen_identity_verified': True, 'root_content_manifest_sha256': root_manifest.hexdigest(),
                        'rank_npz': npz_checks},
              'report_recovery': {'reason': "Original report finalizer indexed pairs in the all/dropped_roots metadata row.",
                                  'original_traceback_retained': 'build/t0010/post_collection.stderr.log',
                                  'completed_stages_replayed': [], 'audit_wall_seconds': round(time.monotonic()-start, 3)}}
    write(REPORT / 'COLLECTION_RESULT.json', result)
    write(REPORT / 'DATA_AUDIT.json', result['audit'])
    write(REPORT / 'FINALIZATION_RECOVERY.json', result['report_recovery'])
    lines = ['# T0010 frozen-carrier V training data', '',
             'Generated by the finalization script embedded in REPRODUCE.md; raw matches, roots and ranking arrays remain local.', '',
             f"Frozen source `{state['source_commit']}`. Old p2 `v12_ponly_xxl`, `models_xxl`, V off; training-only seed 20261201, 500 maps per count 2-15, 6 workers, official 2000/50 ms limits.", '',
             f"All {len(matches):,} scheduled matches are present, with {faults} match faults. All {len(roots):,} expected root keys are present. {sum(dropped.values())} roots were explicitly dropped ({sum(dropped.values())/len(roots):.2%}); reasons: {dict(dropped)}. Retained roots: {sum(root_split.values()):,}, continuations: {continuations:,}.", '',
             f"Continuation faults: {continuation_faults}; prefix faults: {prefix_faults}; terminal successors: {terminal_candidates}. Prefix mismatch distribution (recomputed replies, a nondeterminism diagnostic): {dict(prefix)}.", '',
             '| Map split | Retained roots | Pairs | Resolved | Uncertain |', '|---|---:|---:|---:|---:|']
    for split in ('train', 'val', 'test'):
        p = pair_counts[split]
        lines.append(f"| {split} | {root_split[split]:,} | {p['pairs']:,} | {p['resolved_pairs']:,} | {p['uncertain_pairs']:,} |")
    lines += ['', 'Resolved labels require an outcome gap strictly exceeding one pairwise step, 0.5/(teams-1). Each candidate has one continuation; tie-z=2 adds no repeat-noise estimate with repeats=1. Uncertain pairs retain label -1 and zero weight. Resolved weights sum to one per informative root. Entire maps are disjoint across training, validation and test; all branches inherit their source map split.', '',
              'This is single-arm training-data collection, not a bot leaderboard or a paired strength comparison. Prefix reply mismatches do not mean reproduced state blocks failed; roots whose state blocks could not be reproduced were dropped rather than forced into the dataset. Per-split counts, array hashes, identity checks and full diagnostics are in COLLECTION_RESULT.json and DATA_AUDIT.json.', '',
              '## Execution and recovery', '',
              'The original build-manifest receipt is immutable. MANIFEST_METADATA_ADAPTER.json documents the verified change only to updated/compile_seconds metadata; carrier binaries, inlined sources, runner and all other frozen inputs stayed the same. Preparation/collection were retained without replay.', '',
              'All four pipeline stages completed with exit code 0. The report finalizer subsequently failed because the all/dropped_roots metadata row has no pair counts. The original traceback and state are preserved locally. Finalization was repaired by reading only the three map-split rows and independently checking already completed artifacts; no collection, continuation, ranking, training or export was replayed.', '',
              '| Stage | Wall seconds |', '|---|---:|']
    lines += [f"| {s['stage']} | {s['wall_seconds']:.3f} |" for s in state['steps']]
    lines += ['', 'The passive observer sampled verified descendants every 15 seconds without process termination, paging changes or resource-setting changes. Required delivery checks follow this audit. User-required independent-branch PR delivery replaces atomic main/task-board updates.', '']
    (REPORT / 'COLLECTION.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    state['stage'] = 'complete'
    state['completed_utc'] = result['completed_utc']
    state['report_recovery'] = result['report_recovery']
    write(BUILD / 'pipeline_state.json', state)
    print('T0010 completed artifact audit:', json.dumps({'drops': dict(dropped), 'prefix': dict(prefix),
          'prefix_faults': prefix_faults, 'continuation_faults': continuation_faults,
          'pairs': pair_counts, 'rank_npz': npz_checks}), flush=True)


if __name__ == '__main__':
    main()
```

## Delivery checks

Run after all formal computation and artifact audits finish, under the T0010 CPU lock. These wall times are validation only.

```python
"""Required repository delivery checks after frozen T0010 compute completed."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

build=Path('build/t0010');report=Path('bots/p_only/report/v_data')
checks=[]
commands=[('core',[sys.executable,'-m','unittest','discover','-s','tests','-t','.']),
          ('p_only',[sys.executable,'-m','unittest','discover','-s','bots/p_only/tests']),
          ('single_file_bundles',[sys.executable,'tools/repo/release_bots.py','--out','dist','--check'])]
def save():
    value={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'python':sys.version.split()[0],'checks':checks,'validation_timings_are_not_model_capacity_measurements':True}
    (build/'delivery_checks_state.json').write_text(json.dumps(value,indent=1)+'\n',encoding='utf-8')
    return value
for name,args in commands:
    print('CHECK',name,flush=True);start=time.monotonic()
    log=build/f'check_{name}.log'
    with log.open('w',encoding='utf-8') as f:
        code=subprocess.call(args,stdout=f,stderr=subprocess.STDOUT)
    tail=log.read_text(encoding='utf-8',errors='replace').splitlines()[-10:]
    checks.append({'name':name,'command':['python',*args[1:]],'exit_code':code,
                   'wall_seconds':round(time.monotonic()-start,3),'log':log.as_posix(),'summary':tail})
    save();print(name,code,'\n'.join(tail),flush=True)
    if code:raise SystemExit(code)
value=save()
(report/'CHECKS.json').write_text(json.dumps(value,indent=1)+'\n',encoding='utf-8',newline='\n')
lines=['# T0010 delivery checks','',f"Measured source and validation checkout: `{value['source_commit']}`. Python {value['python']}.",'',
       'Generated by the checks script embedded in REPRODUCE.md. Logs remain local under build/t0010/. All checks ran after measurement stages ended. Check wall times are not timing/strength evidence.','']
for c in checks:
    lines += [f"## {c['name']}",'',f"Command: `{' '.join(c['command'])}`. Exit {c['exit_code']}.",'','```text',*c['summary'],'```','']
lines += ['The complete artifact audit in DATA_AUDIT.json independently checks the 7000 scheduled matches, 14000 root keys, frozen identities, NPZ shape/dtype/bounds, label directions and margins, uncertain zero weights, per-root resolved weights and disjoint map splits.',
          'The original manifest receipt and both original controller tracebacks remain preserved. No completed collection, continuation or ranking stage was replayed to repair report finalization.','']
(report/'CHECKS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
```

## Final report rendering

This renderer reads only audit/check receipts and logs. It reports test counts and concise nondeterminism diagnostics without replaying checks or measurement.

```python
"""Render T0010 delivery checks and clarify quality diagnostics from saved receipts."""
import json
import re
from pathlib import Path

r=Path('bots/p_only/report/v_data')
result=json.loads((r/'COLLECTION_RESULT.json').read_text())
c=json.loads(Path('build/t0010/delivery_checks_state.json').read_text())
assert len(c['checks'])==3 and all(x['exit_code']==0 for x in c['checks'])
for x in c['checks']:
    log=Path(x['log']).read_text(encoding='utf-8',errors='replace')
    lines=log.splitlines()
    if x['name'] in ('core','p_only'):
        summary=[v for v in lines if re.match(r'Ran \d+ tests|OK\b|FAILED\b',v)]
        assert len(summary)==2 and summary[-1].startswith('OK')
    else:
        good=[v for v in lines if v.startswith('[check]') and v.endswith('ok')]
        assert good and not any('FAILED' in v for v in lines)
        summary=[f'{len(good)} single-file bundles compiled successfully.',*[v for v in lines if v.startswith('[release]')]]
    x['summary']=summary
    x['full_log_sha256']=__import__('hashlib').sha256(Path(x['log']).read_bytes()).hexdigest()
(r/'CHECKS.json').write_text(json.dumps(c,indent=1)+'\n',encoding='utf-8',newline='\n')
lines=['# T0010 delivery checks','',f"Validation source: `{c['source_commit']}`; Python {c['python']}.",'',
       'Generated by the scripts embedded in REPRODUCE.md from saved check logs. Full logs stay local under build/t0010/. These wall times are validation only, not capacity or strength evidence.','']
for x in c['checks']:
    lines += [f"## {x['name']}",'',f"`{' '.join(x['command'])}`: exit 0.",'','```text',*x['summary'],'```','']
lines += ['DATA_AUDIT.json independently verifies complete scheduled-match/root-key coverage, frozen identity, NPZ shape/dtype/index bounds, label direction/margins, uncertain zero weights, resolved weight normalization and disjoint map splits.',
          'The report-only recovery retained original failure records and replayed no completed measurement stage.','']
(r/'CHECKS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
p=r/'COLLECTION.md'
text=p.read_text(encoding='utf-8')
prefix=result['prefix_mismatch_distribution'];nonzero=sum(v for k,v in prefix.items() if int(k)!=0)
diagnostic=(f"Continuation faults: {result['continuation_faults']}; prefix faults: {result['prefix_faults']}; terminal successors: {result['terminal_candidates']}. "
            f"Recomputed prefix replies differed in {nonzero:,}/{result['continuations']:,} continuations ({nonzero/result['continuations']:.3%}); maximum mismatch count {max(map(int,prefix))}. "
            'No root was dropped for failure to reproduce recorded state blocks. The complete nondeterminism distribution is preserved in COLLECTION_RESULT.json.')
text=re.sub(r'Continuation faults:.*?\n\n',diagnostic+'\n\n',text,count=1,flags=re.DOTALL)
p.write_text(text,encoding='utf-8',newline='\n')
print('Rendered final report and check summaries.')
```
