# T0013 reproduction

Frozen source 64576e0d6d428e5f45f9ad0a89216d6c4808bfb4, branch codex/t0013-p-features-host-evidence. Completed games, cost probes and audit stages must not be repeated. Seed 20261210 is consumed. Keep raw logs/binaries/checkpoints ignored. Original frozen metadata lives in build/t0013/FROZEN_SCREEN.json; public copies use ${WORK} and ${AI_TEST_ROOT} placeholders, with original/public hashes in METADATA_TRANSPORT.json.

Python 3.14, GCC 15.2 static/O2, OMP/MKL/OpenBLAS8. Set AI_TEST_ROOT, T0013_TRAIN_INDEX (original shard index), T0013_T28_RESULTS (actual 50,000-game JSONL), CXX and T0013_PYTHON at runtime. Prepare build/t0013/REMOTE_DIGEST.md from a fresh origin/main digest. The verified parent lease, hidden helper and passive 15-second observer are recorded in ignored launch/resource receipts. No formal GPU job overlaps the host CPU lease.

The original exclusive-lease command was `python tools/repo/task.py lock --task T0013 --wait 60 cpu -- pwsh -NoProfile -ExecutionPolicy Bypass -File ../../t0013_run.ps1`. It executes preparation, actual candidate build, dry-run, full binary/source freeze, cost, games and analysis in order. The separate audit used t0013_audit_run.ps1 with a new CPU lease after successful collection. Below are byte-exact original scripts; do not invoke completed stages.

## t0013_pipeline.py

SHA256 `42c197785b20c93d2793100d32cf871a4e7b231bb48a451940937b0162a99c29`, 9501 bytes.

```python
"""Bounded T0013 pipeline. Completed stage receipts are never repeated."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path.cwd()
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / 'bots/p_only/tools'))
from fusion import runner
from fusion.paths import ai_test_root, bot_path, engine_path, load_pool
from roster.strength import schedule, schedule_identity, REFERENCE
import screen_runner

B = P / 'build/t0013'
R = P / 'bots/p_only/report/p_features'
W = P / 'data/p_only/p_features'
CFG = P / 'bots/p_only/profiles/screen_p_features.json'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1) + '\n', encoding='utf-8', newline='\n')

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def stage(name, callback):
    receipt = B / (name + '.json')
    if receipt.exists():
        raise RuntimeError('Completed stage exists; do not rerun: ' + name)
    save(B / 'state.json', {'stage': name, 'status': 'running', 'utc': now()})
    print('[T0013] ' + name, flush=True)
    start = time.monotonic()
    result = callback()
    save(receipt, {'stage': name, 'exit_code': 0, 'elapsed_seconds': time.monotonic()-start, 'utc': now(), 'result': result})
    return result

def command(args):
    subprocess.run([sys.executable, *map(str, args)], cwd=P, check=True)

def prepare():
    cfg = read(CFG)
    assert cfg['seed'] == 20261210 and len(cfg['arms']) == 6 and len(cfg['aa']) == 1
    assert cfg['maps_per_count'] == 20 and cfg['teams'] == list(range(2,16))
    assert (cfg['first_ms'],cfg['turn_ms'],cfg['workers']) == (2000,50,6)
    assert not (W / 'results.jsonl').exists(), 'Existing official rows require separate verified resume'
    plan = schedule(20, cfg['teams'], REFERENCE, cfg['seed'], ai_test_root())
    new = {x['map_seed'] for x in plan['fixtures']}
    assert len(new) == 280
    index = Path(os.environ['T0013_TRAIN_INDEX'])
    historical = {x['map_seed'] for x in read(index)['matches']}
    t28 = Path(os.environ['T0013_T28_RESULTS'])
    records = 0
    with t28.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line); historical.add(row['map_seed']); records += 1
    assert records == 50000
    old = {}
    roots = [P.parent.parent/'ctf-fusion', *[t for t in P.parent.iterdir() if t.is_dir() and t != P]]
    for tree in roots:
        for directory in ('data','build'):
            base = tree / directory
            if not base.is_dir():
                continue
            for path in base.rglob('*schedule*.json'):
                try:
                    doc = read(path)
                except (OSError, ValueError):
                    continue
                if isinstance(doc, dict):
                    assert doc.get('seed') != cfg['seed'], str(path)
                    rows = doc.get('fixtures', doc.get('schedule', []))
                else:
                    rows = doc if isinstance(doc,list) else []
                maps = {x['map_seed'] for x in rows if isinstance(x,dict) and 'map_seed' in x}
                historical.update(maps)
                if maps: old[str(path)] = {'sha256':sha(path),'maps':len(maps)}
    assert not (new & historical), sorted(new & historical)
    prep = read(R / 'PREPARATION.json')
    for spec in prep['model_sets'].values():
        for path, v in spec['files'].items():
            assert sha(P/path) == v['sha256'], path
    save(W/'schedule.json', plan)
    result = {'seed':cfg['seed'],'games':1960,'fixtures':280,'schedule_identity':schedule_identity(plan),'schedule_sha256':sha(W/'schedule.json'),'original_index_sha256':sha(index),'T0028_records_sha256':sha(t28),'T0028_records':records,'inspected_previous_schedules':old,'old_map_intersections':0,'seed_state':'prepared; not yet consumed by a game','remote_digest_sha256':sha(B/'REMOTE_DIGEST.md')}
    save(B/'SEED_PROVENANCE.json',result)
    return result

def freeze():
    cfg=read(CFG); plan=read(W/'schedule.json'); manifest=read(P/'build/p_only/manifest.json')
    arms=cfg['arms']+[x+'~aa' for x in cfg['aa']]
    assert len(screen_runner.job_order(plan['fixtures'],arms))==1960
    ids=screen_runner.identities(arms,manifest['variants'],plan,(2,.05))
    assert ids[arms[-1]]==ids[cfg['aa'][0]]
    commands,info=screen_runner.reference_commands(plan,ai_test_root())
    deps={}; binaries={}
    pool={x['name']:x for x in load_pool()['bots']}
    for name,cmd in commands.items():
        source=bot_path(pool[name],ai_test_root())
        binaries[name]={'source':str(source),'source_sha256':sha(source),'command':cmd,'binary_sha256':sha(cmd[0])}
        if source.suffix in ('.cpp','.cc','.cxx'):
            result=subprocess.check_output([os.environ['CXX'],'-MM',str(source)],text=True)
            depfile=B/(name+'.dependencies.txt'); depfile.write_text(result,encoding='utf-8')
            # GCC on this host returns forward slash absolute paths; joining continuations preserves tokens.
            text=result.replace('\\\n',' ').split(': ',1)[-1]
            for token in text.split():
                path=Path(token)
                if path.is_file(): deps[str(path)] = sha(path)
    inputs={}
    for arm in cfg['arms']:
        m=manifest['variants'][arm]
        assert sha(P/m['binary'])==m['binary_sha256']
        for item in m['inputs']:
            assert sha(P/item['path'])==item['sha256'],item['path']
            inputs[item['path']]=item['sha256']
    helpers={str(p):sha(p) for p in (P.parents[1]/'t0013_pipeline.py',P.parents[1]/'t0013_run.ps1',P.parents[1]/'t0013_resource_monitor.ps1')}
    source={f:sha(P/f) for f in screen_runner.RUNNER_FILES+['bots/p_only/tools/build.py','bots/p_only/tools/analyze.py','tools/host/value_cost.py','profiles/pool.json'] if (P/f).is_file()}
    receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'config_sha256':sha(CFG),'schedule_sha256':sha(W/'schedule.json'),'manifest_sha256':sha(P/'build/p_only/manifest.json'),'engine_sha256':sha(engine_path(ai_test_root())),'candidate_inputs':inputs,'reference_binaries':binaries,'reference_transitive_includes':deps,'runner_sources':source,'auxiliary_helpers':helpers,'arm_identities':ids,'official_limits_ms':[2000,50],'workers':6,'games':1960}
    receipt['identity']=screen_runner.digest(receipt)
    save(B/'FROZEN_SCREEN.json',receipt)
    return {'identity':receipt['identity'],'games':1960,'reference_programs':len(commands)}

def screen():
    cfg=read(CFG)
    frozen=read(B/'FROZEN_SCREEN.json')
    assert sha(CFG)==frozen['config_sha256'] and sha(W/'schedule.json')==frozen['schedule_sha256']
    assert sha(P/'build/p_only/manifest.json')==frozen['manifest_sha256']
    for name,value in frozen['candidate_inputs'].items(): assert sha(P/name)==value,name
    for name,value in frozen['reference_transitive_includes'].items(): assert sha(name)==value,name
    for name,value in frozen['runner_sources'].items(): assert sha(P/name)==value,name
    assert sha(engine_path(ai_test_root()))==frozen['engine_sha256']
    command(['bots/p_only/tools/screen_runner.py','--schedule',W/'schedule.json','--manifest','build/p_only/manifest.json','--arms',*cfg['arms'],'--aa',*cfg['aa'],'--results',W/'results.jsonl','--workers','6','--first-ms','2000','--turn-ms','50','--keep-matches',W/'matches'])
    return {'results_sha256':sha(W/'results.jsonl'),'rows':sum(1 for x in (W/'results.jsonl').open(encoding='utf-8') if x.strip())}

def analyze():
    cfg=read(CFG)
    args=['bots/p_only/tools/analyze.py','--results',W/'results.jsonl','--schedule',W/'schedule.json','--manifest','build/p_only/manifest.json','--arms',*cfg['arms'],'--aa',*cfg['aa'],'--out',R,'--identities',W/'identities.json','--matches',W/'matches']
    for a,b in cfg['comparisons']: args+=['--compare',a+':'+b]
    for name,terms in cfg['contrasts'].items(): args+=['--contrast',name+'='+terms]
    command(args)
    return {'exit_code':0}

def main():
    B.mkdir(parents=True,exist_ok=True)
    lease=read(Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json')
    assert lease['task']=='T0013' and not (Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
    stage('prepare',prepare)
    cfg=read(CFG)
    stage('build',lambda:command(['bots/p_only/tools/build.py','--stats','--variants',cfg['variants'],*cfg['arms']]))
    stage('dry_run',lambda:command(['bots/p_only/tools/screen_runner.py','--schedule',W/'schedule.json','--manifest','build/p_only/manifest.json','--arms',*cfg['arms'],'--aa',*cfg['aa'],'--results',W/'results.jsonl','--workers','6','--first-ms','2000','--turn-ms','50','--dry-run']))
    stage('freeze',freeze)
    stage('cost',lambda:command(['tools/host/value_cost.py','--set','inference/models_pf_p2xxl_vm','--set','inference/models_pf_p3xxl_vm','--calls','3','--copies','1','6','15','--out','reports/host/feature_cost_DESKTOP-G60OJUD']))
    stage('games',screen)
    stage('analyze',analyze)
    save(B/'state.json',{'stage':'complete','status':'success','utc':now(),'next':'independent raw/source/fault audit and PR delivery; no game rerun'})

if __name__=='__main__':
    try: main()
    except Exception as e:
        save(B/'failure.json',{'utc':now(),'type':type(e).__name__,'error':str(e)})
        raise
```

## t0013_run.ps1

SHA256 `8da5662c4f82b912be1ca38a3c09520e8edcc38e378519221893ed645ba25618`, 1756 bytes.

```powershell
$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$self=Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent=Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($parent.CommandLine -notmatch 'task.py lock.*T0013.*t0013_run.ps1') { throw 'Unverified CPU lock parent' }
$lease=Get-Content -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/cpu.json') -Raw | ConvertFrom-Json
if ($lease.pid -ne $parent.ProcessId -or $lease.task -ne 'T0013') { throw 'Wrong CPU lease' }
if (Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/gpu.json')) { throw 'GPU lease present; no formal CPU/GPU concurrency' }
$created=$parent.CreationDate.ToUniversalTime().ToString('o')
@{pipeline_pid=$parent.ProcessId;created_utc=$created;source_commit='64576e0d6d428e5f45f9ad0a89216d6c4808bfb4';helper_pid=$PID} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0013/pipeline_pid.json' -Encoding utf8
if (-not $env:T0013_PYTHON) { throw 'T0013_PYTHON must name the verified interpreter' }
$watch=Start-Process -FilePath 'pwsh.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','../../t0013_resource_monitor.ps1',
    '-RepoPath',$root,'-PipelinePid',[string]$parent.ProcessId,'-CreationUtc',$created,'-ExpectedCommandTag','t0013_run.ps1') `
    -WindowStyle Hidden -RedirectStandardOutput 'build/t0013/resource_monitor.stdout' `
    -RedirectStandardError 'build/t0013/resource_monitor.stderr' -PassThru
@{monitor_pid=$watch.Id;created_utc=$watch.StartTime.ToUniversalTime().ToString('o');pipeline_pid=$parent.ProcessId} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0013/resource_monitor_pid.json' -Encoding utf8
& $env:T0013_PYTHON ../../t0013_pipeline.py
exit $LASTEXITCODE
```

## t0013_resource_monitor.ps1

SHA256 `de91501d0db8d99c6cf19d600ded33e01d7b1501a7d18483252a4f66525dc93e`, 3626 bytes.

```powershell
param([Parameter(Mandatory=$true)][string]$RepoPath,
      [Parameter(Mandatory=$true)][int]$PipelinePid,
      [Parameter(Mandatory=$true)][string]$CreationUtc,
      [string]$ExpectedCommandTag='t0013_run.ps1')
$ErrorActionPreference='Stop'
$root = (Resolve-Path -LiteralPath $RepoPath).Path
$output = Join-Path $root 'build/t0013'
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
            $pipeline.CommandLine -notmatch ('task.py lock.*T0013.*'+[regex]::Escape($ExpectedCommandTag))) {
            Write-Sample @{utc=$utc;event='identity_mismatch';pipeline_pid=$PipelinePid;attention=$true}; break
        }
        $ids = [System.Collections.Generic.HashSet[int]]::new()
        [void]$ids.Add($PipelinePid)
        do {
            $before=$ids.Count
            foreach ($process in $all) {
                if ($ids.Contains([int]$process.ParentProcessId)) {
                    $ancestor=$all | Where-Object ProcessId -eq $process.ParentProcessId
                    if ($ancestor -and $process.CreationDate -ge $ancestor.CreationDate) { [void]$ids.Add([int]$process.ProcessId) }
                }
            }
        } while ($ids.Count -gt $before)
        $processes = @()
        foreach ($process in $all) {
            if ($ids.Contains([int]$process.ProcessId)) {
                $live=Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
                if ($live) { $processes+=@{pid=$live.Id;parent_pid=$process.ParentProcessId;creation_utc=$process.CreationDate.ToUniversalTime().ToString('o');command=$process.CommandLine;
                    private_gib=[math]::Round($live.PrivateMemorySize64/1GB,3);
                    working_gib=[math]::Round($live.WorkingSet64/1GB,3);cpu_seconds=$live.CPU} }
            }
        }
        $os=Get-CimInstance Win32_OperatingSystem
        $memory=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
        $free=[math]::Round($os.FreePhysicalMemory/1MB,3)
        $commit=[math]::Round(($memory.CommitLimit-$memory.CommittedBytes)/1GB,3)
        $vram=(& nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null)
        $lease=Get-Content -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/cpu.json') -Raw | ConvertFrom-Json
        Write-Sample @{utc=$utc;event='sample';pipeline_pid=$PipelinePid;monitor_pid=$PID;
                      physical_available_gib=$free;commit_available_gib=$commit;gpu_readonly=$vram;cpu_lease_task=$lease.task;cpu_lease_pid=$lease.pid;
                      attention=($free -lt 4 -or $commit -lt 8);processes=$processes;
                      policy='passive_only_no_kill_no_restart_no_resource_change'}
    } catch {
        Write-Sample @{utc=(Get-Date).ToUniversalTime().ToString('o');event='monitor_error';attention=$true;error=$_.Exception.Message}
    }
    Start-Sleep -Seconds 15
}
```

## t0013_audit.py

SHA256 `994918c7e857b676134028ce7e70e1fa7e1e15967d30e5e6d5923ba615dfa28e`, 7879 bytes.

```python
"""Independent raw/identity/paired-CI audit; no games or predictions."""
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

P=Path.cwd(); sys.path.insert(0,str(P))
from fusion import matchlog
from fusion.protocol import parse_block, parse_reply
R=P/'bots/p_only/report/p_features'; B=P/'build/t0013'; W=P/'data/p_only/p_features'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,indent=1)+'\n',encoding='utf-8',newline='\n')

cfg=read(P/'bots/p_only/profiles/screen_p_features.json'); frozen=read(B/'FROZEN_SCREEN.json')
assert frozen['config_sha256']==sha(P/'bots/p_only/profiles/screen_p_features.json')
assert frozen['schedule_sha256']==sha(W/'schedule.json') and frozen['manifest_sha256']==sha(P/'build/p_only/manifest.json')
for k,v in frozen['candidate_inputs'].items(): assert sha(P/k)==v,k
for k,v in frozen['runner_sources'].items(): assert sha(P/k)==v,k
for k,v in frozen['reference_transitive_includes'].items(): assert sha(k)==v,k
for v in frozen['reference_binaries'].values(): assert sha(v['command'][0])==v['binary_sha256']
for k,v in frozen['auxiliary_helpers'].items(): assert sha(k)==v,k
manifest=read(P/'build/p_only/manifest.json')['variants']
for name,v in manifest.items(): assert sha(P/v['binary'])==v['binary_sha256'],name
plan=read(W/'schedule.json'); fixtures={x['fixture_id']:x for x in plan['fixtures']}
assert len(fixtures)==280 and len({x['map_seed'] for x in fixtures.values()})==280
arms=cfg['arms']+[x+'~aa' for x in cfg['aa']]
rows=[json.loads(s) for s in (W/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
assert len(rows)==1960 and len({(r['arm'],r['fixture_id']) for r in rows})==1960
assert sha(W/'results.jsonl')==read(B/'games.json')['result']['results_sha256']
totals=Counter(); receipts=[]; faults=[]; by=defaultdict(dict); counter_sums=defaultdict(Counter)
refs={x['name']:x for x in plan['reference']}
ponly=re.compile(r'PONLY builds (\d+) teams (\d+) act (\d+) target (\d+) fallback (\d+)(?: inconsistent (\d+) skipped (\d+))?')
for row in rows:
    arm=row['arm']; fx=fixtures[row['fixture_id']]; base=arm.removesuffix('~aa')
    assert row['candidate']==arm and row['identity']==frozen['arm_identities'][arm]['digest']
    assert row['source_sha256']==manifest[base]['binary_sha256']
    assert row['engine_sha256']==frozen['engine_sha256']
    assert row['schedule']==read(W/'identities.json')[arm]['components']['schedule']
    expected_order=list(fixtures).index(fx['fixture_id'])*7 + (arms[arms.index(arm):].index(arm) if False else (arms.index(arm)-list(fixtures).index(fx['fixture_id'])%7)%7)
    assert row['order']==expected_order
    key=hashlib.sha256(arm.encode()).hexdigest()[:10]
    path=W/'matches'/f"n{fx['teams']:02d}"/(fx['fixture_id']+'-'+key+'.json.gz')
    m=matchlog.load(path); n=fx['teams']; seat=fx['seat']
    assert m['match_id']==fx['fixture_id']+'-'+key
    assert m['teams']==row['teams']==n and m['map_seed']==row['map_seed']==fx['map_seed'] and m['flag_seed']==fx['flag_seed']
    assert m['limits_ms']==[2000,50] and m['engine_sha256']==frozen['engine_sha256']
    names=list(fx['opponents']); names.insert(seat,'@candidate')
    assert [s['name'] for s in m['seats']]==names
    for s in m['seats']:
        assert s['sha256']==(manifest[base]['binary_sha256'] if s['name']=='@candidate' else refs[s['name']]['source_sha256'])
    assert len(m['blocks'])==len(m['replies'])==400
    score=m['result']['score']; assert parse_block(m['final_block']).score==score==row['score']
    utility=sum(1 if score[seat]>x else .5 if score[seat]==x else 0 for i,x in enumerate(score) if i!=seat)/(n-1)
    assert abs(utility-row['utility'])<1e-12
    me=m['bots'][seat]
    for field in ('faults','dead','late','first_ms','p99_ms','max_ms'): assert me[field]==row[field],(arm,field)
    assert sum(x['faults'] for x in m['bots'])==len(m['faults'])
    totals['allseat_faults']+=len(m['faults']); totals['candidate_faults']+=me['faults']; totals['candidate_late']+=me['late']
    for t,(block,replies) in enumerate(zip(m['blocks'],m['replies']),1):
        assert parse_block(block).turn==t and len(replies)==n
        totals['allseat_turn_replies']+=n
        totals['invalid_or_missing_replies']+=sum(parse_reply(line,t) is None for line in replies)
    for f in m['faults']: faults.append({'arm':arm,'fixture_id':fx['fixture_id'],'fault':f})
    for line in me.get('stderr_tail',[]):
        q=ponly.search(line)
        if q:
            counter_sums[arm]['turns']+=400
            for i,k in enumerate(('views','encodes','act','target','fallbacks','inconsistent','skipped')): counter_sums[arm][k]+=int(q.group(i+1) or 0)
    receipts.append({'arm':arm,'fixture_id':fx['fixture_id'],'raw_sha256':sha(path),'candidate_binary_sha256':row['source_sha256'],'allseat_faults':len(m['faults'])})
    by[arm][fx['fixture_id']]=row
screen=read(R/'screen.json'); assert not screen['incomplete'] and not screen['stale_rows']
bands={'2':(2,2),'3-5':(3,5),'6-10':(6,10),'11-15':(11,15)}
comparisons={a+' - '+b:[(1,a),(-1,b)] for a,b in cfg['comparisons']}
comparisons['p3 effect, carrier interaction']=[(1,'c08_pf_p3xxl'),(-1,'c08_pf_p2xxl'),(-1,'v12_pf_p3xxl'),(1,'v12_pf_p2xxl')]
comparisons['v12_pf_p3xxl~aa - v12_pf_p3xxl']=[(1,'v12_pf_p3xxl~aa'),(-1,'v12_pf_p3xxl')]
for name,terms in comparisons.items():
    expected=(screen['comparisons']|screen['rerun_noise'])[name]; per=defaultdict(list)
    for fid,fx in fixtures.items(): per[fx['teams']].append(sum(weight*by[arm][fid]['utility'] for weight,arm in terms))
    for band,counts in [('overall',list(range(2,16)))]+[(k,list(range(lo,hi+1))) for k,(lo,hi) in bands.items()]:
        mean=statistics.mean(statistics.mean(per[c]) for c in counts)
        se=math.sqrt(sum(statistics.variance(per[c])/len(per[c]) for c in counts))/len(counts)
        lo,hi=mean-1.96*se,mean+1.96*se; target=expected['overall'] if band=='overall' else expected['bands'][band]
        for a,b in zip((mean,se,lo,hi),(target['mean'],target['se'],*target['ci95'])): assert abs(a-b)<=5.000001e-6,(name,band,a,b)
        assert target['verdict']==('helps' if lo>0 else 'hurts' if hi<0 else 'not shown')
for arm, rs in by.items():
    target=screen['arms'][arm]; assert target['games']==len(rs)==280
    assert target['faults']==sum(r['faults'] for r in rs.values()) and target['late_replies']==sum(r['late'] for r in rs.values())
    c=counter_sums[arm]; published=screen['network_calls'][arm]
    assert published['turns']==c['turns']
    for raw,field in (('views','views_per_turn'),('encodes','team_encodes_per_turn'),('act','act_heads_per_turn'),('target','target_heads_per_turn')): assert published[field]==round(c[raw]/c['turns'],2)
    for field in ('fallbacks','inconsistent','skipped'): assert published[field]==c[field]
save(R/'RAW_RECEIPTS.json',{'results_sha256':sha(W/'results.jsonl'),'matches':receipts})
save(R/'INDEPENDENT_AUDIT.json',{'frozen_identity':frozen['identity'],'raw_games_checked':1960,'map_clusters':280,'full_400_turn_blocks_checked':784000,'counts':dict(totals),'all_seat_faults_retained':faults,'paired_comparisons_and_AA_CI_recomputed':len(comparisons),'network_counters_recomputed':{k:dict(v) for k,v in counter_sums.items()},'auditor_sha256':sha(P.parents[1]/'t0013_audit.py'),'limits':'Successful reply timing fields match raw bot reports; individual timing samples are not retained for independent percentile recomputation. PONLY totals recomputed; SEARCH sample ratios are retained original analysis and not independently recomputed by this audit. No replay/prediction rerun.'})
print(json.dumps({'raw_games_checked':1960,'counts':dict(totals),'independent_paired_CI':'pass'}),flush=True)
```

## t0013_audit_run.ps1

SHA256 `2ca60aa42ceb266ae40c6d7d44991db91641aafe374c803fe80754844263dcca`, 1777 bytes.

```powershell
$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$self=Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent=Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($parent.CommandLine -notmatch 'task.py lock.*T0013.*t0013_audit_run.ps1') { throw 'Unverified CPU lock parent' }
$lease=Get-Content -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/cpu.json') -Raw | ConvertFrom-Json
if ($lease.pid -ne $parent.ProcessId -or $lease.task -ne 'T0013') { throw 'Wrong CPU lease' }
if (Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/gpu.json')) { throw 'GPU lease present; no formal CPU/GPU concurrency' }
$created=$parent.CreationDate.ToUniversalTime().ToString('o')
@{pipeline_pid=$parent.ProcessId;created_utc=$created;source_commit='64576e0d6d428e5f45f9ad0a89216d6c4808bfb4';helper_pid=$PID} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0013/audit_pipeline_pid.json' -Encoding utf8
if (-not $env:T0013_PYTHON) { throw 'T0013_PYTHON must name the verified interpreter' }
$watch=Start-Process -FilePath 'pwsh.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','../../t0013_resource_monitor.ps1',
    '-RepoPath',$root,'-PipelinePid',[string]$parent.ProcessId,'-CreationUtc',$created,'-ExpectedCommandTag','t0013_audit_run.ps1') `
    -WindowStyle Hidden -RedirectStandardOutput 'build/t0013/resource_monitor.stdout' `
    -RedirectStandardError 'build/t0013/resource_monitor.stderr' -PassThru
@{monitor_pid=$watch.Id;created_utc=$watch.StartTime.ToUniversalTime().ToString('o');pipeline_pid=$parent.ProcessId} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0013/audit_resource_monitor_pid.json' -Encoding utf8
& $env:T0013_PYTHON ../../t0013_audit.py
exit $LASTEXITCODE
```
