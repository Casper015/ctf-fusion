# T0010 pipeline reproduction

The initial collection controller below records the measurement procedure. Seed 20261201 is consumed: do not launch collection again. For review fixes, extract the strict helper, completed-artifact audit and final renderer below; run only those verification/rendering scripts from the frozen repository root under tools/repo/task.py lock --task T0010 cpu. Set AI_TEST_ROOT and CXX for the host; collection uses6 workers. Resume only the same identity-bound data and receipts, retaining completed games and roots. Never rebuild between collection and continuations. No raw data or build output is committed.

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
    else:
        baseline = BUILD/'manifest_at_freeze.json'
        baseline.write_bytes(MANIFEST.read_bytes())
        assert sha(baseline) == frozen['manifest_sha256']
        write(receipt,frozen)
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
              'uncertain_pairs':v['pairs']-v['resolved_pairs']} for k,v in index['stats'].items() if k in ('train','val','test')}
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

## Strict review contracts (PR #23 fixes)

Save this Python fence as `t0010_review_contracts.py` beside the extracted drivers. It is the current implementation used by the adapter and audit below. Full manifest verification requires original bytes whose digest matches the immutable receipt, or a normalized digest recorded at the original freeze point. It rejects a missing baseline by default. `--allow-selected-manifest` is an explicit legacy fallback which certifies only selected frozen identities and sets `all_other_frozen_fields_exact` to false; it never manufactures an original baseline from the current manifest.

The published corpus has not undergone the new raw-source alignment audit on this Mac: raw roots and NPZ files are host-local. Corrected published receipts therefore retain their historical counts/hashes and identify source alignment as pending. Once the host runs the strict audit below, it may publish the actual resulting receipts. No collection, continuation or ranking stage needs replay.

```python
"""Strict PR #23 audit contracts; saved verbatim in REPRODUCE.md."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_manifest(value):
    # Copy before removing the only allowed volatile fields.
    value = json.loads(json.dumps(value))
    value.pop('updated', None)
    for entry in value['variants'].values():
        entry.pop('compile_seconds', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_manifest_baseline(current, captured_sha, original_bytes=None,
                             captured_normalized_sha=None, allow_selected_only=False):
    current_digest = normalized_manifest(current)
    if original_bytes is not None:
        require(hashlib.sha256(original_bytes).hexdigest() == captured_sha,
                'Original manifest bytes do not match the immutable captured digest')
        baseline = normalized_manifest(json.loads(original_bytes.decode('utf-8-sig')))
        if captured_normalized_sha is not None:
            require(baseline == captured_normalized_sha, 'Captured manifest baselines disagree')
        scope = 'full_original_manifest_bytes'
    elif captured_normalized_sha is not None:
        baseline = captured_normalized_sha
        scope = 'normalized_digest_recorded_at_freeze'
    else:
        require(allow_selected_only, 'No captured original manifest baseline; full comparison unavailable')
        return {'stable_manifest_sha256': current_digest,
                'all_other_frozen_fields_exact': False,
                'verification_scope': 'selected_frozen_identities_only',
                'full_baseline_comparison': 'unavailable',
                'unverified_scope': 'original manifest nonvolatile content'}
    require(current_digest == baseline, 'A nonvolatile manifest field changed')
    return {'stable_manifest_sha256': current_digest,
            'captured_stable_manifest_sha256': baseline,
            'all_other_frozen_fields_exact': True,
            'verification_scope': scope, 'full_baseline_comparison': 'passed'}


def outcome_from_scores(scores, seat):
    scores = np.asarray(scores)
    require(scores.ndim == 1 and len(scores) >= 2 and 0 <= seat < len(scores), 'Invalid score vector')
    require(np.all(np.isfinite(scores)) and np.all(scores >= 0)
            and np.all(scores == np.floor(scores)), 'Invalid scores')
    rivals = np.delete(scores, seat)
    return float((np.count_nonzero(scores[seat] > rivals)
                  + 0.5 * np.count_nonzero(scores[seat] == rivals)) / len(rivals))


def canonical_order(line, turn=None):
    require(isinstance(line, str), 'Missing candidate order')
    parts = line.partition('#')[0].split()
    require(len(parts) == 7 and parts[0].isdigit()
            and all(parts[i] in ('S', 'U', 'D', 'L', 'R') for i in (1, 3, 5))
            and all(parts[i] in ('-', 'P', 'X') or parts[i].isdigit() for i in (2, 4, 6)),
            'Invalid candidate order')
    require(turn is None or int(parts[0]) == turn, 'Candidate order turn differs from root')
    return ' '.join(parts)


def verify_rank_sources(z, source_rows, record_from_source, match_info, width=923):
    """source_rows follows the producer's sorted raw-file order; never infer it from NPZ rows.

    Recompute labels from raw scores, then compare every array row and pair.
    record_from_source independently invokes the encoder on raw INIT/root/next blocks.
    """
    dtypes = {'cand_rec': np.int16, 'cand_y': np.float32, 'cand_n': np.int16,
              'cand_root': np.int32, 'cand_on_policy': np.bool_, 'pair_a': np.int32,
              'pair_b': np.int32, 'pair_y': np.int8, 'pair_w': np.float32,
              'root_teams': np.int8, 'root_turn': np.int16, 'root_map': np.int64}
    require(set(dtypes) <= set(z), 'Missing ranking arrays')
    # NpzFile decompresses on each lookup. Materialize each member once before
    # row-by-row checks, avoiding thousands of repeated whole-array inflations.
    z = {name: z[name] for name in dtypes}
    for name, dtype in dtypes.items():
        require(z[name].dtype == dtype, 'Unexpected dtype: ' + name)
        require(z[name].ndim == (2 if name == 'cand_rec' else 1), 'Unexpected shape: ' + name)
    require(z['cand_rec'].shape == (len(z['cand_y']), width), 'Wrong successor record shape')
    require(all(len(z[k]) == len(z['cand_y']) for k in ('cand_n', 'cand_root', 'cand_on_policy')),
            'Candidate arrays have different lengths')
    require(all(len(z[k]) == len(z['root_map']) for k in ('root_turn', 'root_teams')),
            'Root arrays have different lengths')
    require(all(len(z[k]) == len(z['pair_y']) for k in ('pair_a', 'pair_b', 'pair_w')),
            'Pair arrays have different lengths')
    root_cursor = candidate_cursor = pair_cursor = 0
    lineage = hashlib.sha256()
    for row in source_rows:
        require('dropped' not in row, 'Dropped root supplied to retained-row audit')
        mid, turn, teams, seat = (row[k] for k in ('match_id', 'turn', 'teams', 'seat'))
        require(mid in match_info, 'Unknown source match')
        info = match_info[mid]
        require(teams == info['teams'] and seat == info['seat'] and row['map_seed'] == info['map_seed'],
                'Root metadata differs from source match')
        require(isinstance(turn, int) and 1 <= turn < info['total_turns'] and turn in info['roots'],
                'Root turn differs from the scheduled raw root')
        require(row['init'] == info['init'], 'Root INIT differs from source match')
        original = info['roots'][turn]
        require(len(row['candidates']) == 4 and len(row['branches']) == 4,
                'T0010 requires four candidates and one continuation each')
        require(canonical_order(row['candidates'][0], turn) == canonical_order(original['reply'], turn),
                'Candidate zero is not the recorded on-policy order')
        require(root_cursor < len(z['root_map']), 'Too few array roots')
        require(int(z['root_map'][root_cursor]) == row['map_seed']
                and int(z['root_turn'][root_cursor]) == turn
                and int(z['root_teams'][root_cursor]) == teams, 'Array root metadata/order differs')
        branches = {b['candidate']: b for b in row['branches']}
        require(set(branches) == set(range(4)), 'Missing or duplicate raw candidate')
        means = []
        for cid in range(4):
            b = branches[cid]
            require(b['repeat'] == 0 and isinstance(b['on_policy'], bool) and b['on_policy'] == (cid == 0),
                    'Raw on-policy/repeat metadata differs')
            require(canonical_order(b['line'], turn) == canonical_order(row['candidates'][cid], turn),
                    'Branch order differs from its candidate')
            require(b['root_block'] == original['block'], 'Branch root block differs from source match')
            require(b['next_block'] is not None, 'Unexpected terminal candidate in this T0010 corpus')
            require(len(b['score']) == teams, 'Score vector team count differs')
            y = outcome_from_scores(b['score'], seat)
            require(np.isfinite(b['outcome']) and abs(float(b['outcome']) - y) <= 1e-12,
                    'Raw outcome differs from final scores')
            means.append(y)
            c = candidate_cursor + cid
            require(c < len(z['cand_y']), 'Too few candidate rows')
            require(int(z['cand_root'][c]) == root_cursor and int(z['cand_n'][c]) == 1
                    and bool(z['cand_on_policy'][c]) == (cid == 0), 'Array candidate metadata/order differs')
            require(z['cand_y'][c] == np.float32(y), 'Array outcome differs from raw scores')
            expected = np.asarray(record_from_source(row, b))
            require(expected.dtype == np.int16 and expected.shape == (width,), 'Invalid independently extracted record')
            require(int(expected[0]) == turn + 1 and int(expected[1]) == seat, 'Successor history/seat differs')
            require(np.array_equal(z['cand_rec'][c], expected), 'Array successor features differ from raw blocks')
            item = {'root_index': root_cursor, 'match_id': mid, 'turn': turn,
                    'map': row['map_seed'], 'seat': seat, 'candidate': cid,
                    'order': canonical_order(b['line']), 'scores': b['score'],
                    'record_sha256': hashlib.sha256(expected.tobytes()).hexdigest()}
            lineage.update((json.dumps(item, sort_keys=True) + '\n').encode())
        margin = 0.5 / (teams - 1)
        expected_pairs = [(i, j, -1 if abs(means[i] - means[j]) <= margin else int(means[i] > means[j]))
                          for i, j in itertools.combinations(range(4), 2)]
        resolved = sum(label >= 0 for _, _, label in expected_pairs)
        for i, j, label in expected_pairs:
            p = pair_cursor
            require(p < len(z['pair_y']), 'Too few pair rows')
            require(int(z['pair_a'][p]) == candidate_cursor + i
                    and int(z['pair_b'][p]) == candidate_cursor + j
                    and int(z['pair_y'][p]) == label, 'Array pair differs from raw candidate outcomes')
            weight = np.float32(1.0 / resolved if label >= 0 else 0.0)
            require(z['pair_w'][p] == weight, 'Pair weight differs from raw root')
            pair_cursor += 1
        root_cursor += 1
        candidate_cursor += 4
    require(root_cursor == len(z['root_map']) and candidate_cursor == len(z['cand_y'])
            and pair_cursor == len(z['pair_y']), 'Extra or missing array rows')
    return {'source_row_alignment_verified': True, 'root_metadata_and_order_verified': True,
            'on_policy_verified': True, 'outcomes_recomputed_from_scores': True,
            'successor_records_reextracted': candidate_cursor,
            'root_candidate_lineage_sha256': lineage.hexdigest()}
```

## Verified post-collection handoff

The original controller checks the full build manifest, whose updated/compile_seconds metadata is rewritten by the repeated preparation build. The current adapter below additionally compares against a hash-bound original baseline; it fails closed unless explicit selected-identity fallback is requested. A queued continuation waits for the CPU lock, requires completed collection, verifies every other frozen field plus actual inlined sources/runner, and ignores only those volatile metadata fields. It preserves the original receipt and skips the completed preparation/collection stages; no game is replayed. Set T0010_PYTHON to the same verified interpreter.

```python
"""Resume completed stages after explicit manifest-baseline verification."""
import argparse
from t0010_review_contracts import verify_manifest_baseline
import importlib.util
import hashlib
import json
import subprocess
import time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--allow-selected-manifest',action='store_true')
    ap.add_argument('--original-manifest',default='build/t0010/manifest_at_freeze.json')
    args=ap.parse_args()
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
    baseline=Path(args.original_manifest)
    verification=verify_manifest_baseline(manifest,captured['manifest_sha256'],
        original_bytes=baseline.read_bytes() if baseline.is_file() else None,
        captured_normalized_sha=captured.get('normalized_manifest_sha256'),
        allow_selected_only=args.allow_selected_manifest)
    stable=verification['stable_manifest_sha256']
    m.write(m.REPORT/'MANIFEST_METADATA_ADAPTER.json',{'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'captured_raw_manifest_sha256':captured['manifest_sha256'],'collection_raw_manifest_sha256':actual['manifest_sha256'],
        'stable_manifest_sha256':stable,'ignored_metadata_fields':['updated','variants.*.compile_seconds'],
        **verification,'runner_and_inlined_sources_verified':True,
        'reason':'Manifest verification scope is explicit. The original receipt stays immutable; completed measurement stages are retained.'})
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

All four stages exited successfully. The original finalizer assumed the all/dropped_roots metadata row also had pair counts. Keep the original traceback and state, then run the following script under the T0010 CPU lock. It verifies complete schedule/root coverage, frozen inputs, map splits and ranking arrays; it repairs reports only and never launches a completed measurement stage. The first audit attempt used an incorrect fixture field name and exited before writing results. The current version uses fixture_id plus the runner arm key and the strict source/manifest contracts above. The original audit receipt is reclassified below because it did not execute the new source-alignment checks.

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
from fusion import native
from t0010_review_contracts import verify_manifest_baseline, verify_rank_sources
import argparse
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
    ap=argparse.ArgumentParser()
    ap.add_argument('--allow-selected-manifest',action='store_true')
    ap.add_argument('--original-manifest',default='build/t0010/manifest_at_freeze.json')
    args=ap.parse_args()
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
    baseline=Path(args.original_manifest)
    manifest_verification=verify_manifest_baseline(read(original.MANIFEST),frozen['manifest_sha256'],
        original_bytes=baseline.read_bytes() if baseline.is_file() else None,
        captured_normalized_sha=frozen.get('normalized_manifest_sha256'),
        allow_selected_only=args.allow_selected_manifest)
    screen = read(REPORT / 'screen.json')
    assert screen['fixtures'] == 7000 and not screen['incomplete'] and not screen['stale_rows']
    matches = list(matchlog.iter_matches('data/p_only/v_data/matches'))
    assert len(matches) == 7000
    schedule = read('data/p_only/v_data/schedule.json')
    arm_key = hashlib.sha256(b'v12_ponly_xxl').hexdigest()[:10]
    fixtures = {f"{x['fixture_id']}-{arm_key}": x for x in schedule['fixtures']}
    expected, match_maps, match_counts, seen, faults = set(), {}, collections.Counter(), set(), 0
    match_info={}
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
        selected_turns=root_turns(m,2)
        seats=[j for j,entry in enumerate(m['seats']) if entry['name']=='@candidate']
        assert len(seats)==1
        seat=seats[0]
        match_info[m['match_id']]={'teams':m['teams'],'seat':seat,'map_seed':m['map_seed'],
            'init':m['init'],'total_turns':len(m['blocks']),
            'roots':{t:{'block':m['blocks'][t-1],'reply':m['replies'][t-1][seat]} for t in selected_turns}}
        expected.update((m['match_id'], t) for t in selected_turns)
        if i % 1000 == 0: print('audit matches', i, flush=True)
    assert dict(match_counts) == {n: 500 for n in range(2, 16)}
    assert len(expected) == 14000 and faults == 0
    assert set(fixtures) == seen
    roots = sorted(Path('data/branches/v12_ponly_xxl').rglob('*.json.gz'))
    assert len(roots) == len(expected)
    root_keys, dropped, prefix = set(), collections.Counter(), collections.Counter()
    prefix_faults = continuation_faults = continuations = terminal_candidates = 0
    retained_maps = collections.defaultdict(set)
    source_paths=collections.defaultdict(list)
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
        source_paths[split].append(p)
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
    historical=read(REPORT/'DATA_AUDIT.json') if (REPORT/'DATA_AUDIT.json').is_file() else None
    if historical is not None:
        assert historical['root_content_manifest_sha256']==root_manifest.hexdigest(),'Raw root bytes changed'
    def source_rows(paths):
        for path in paths:
            yield json.loads(gzip.decompress(path.read_bytes()).decode('utf-8'))
    def source_record(row,branch):
        stream=matchlog.init_for_seat(row['init'],row['seat'])+branch['root_block']+branch['next_block']
        records=native.extract(stream)
        assert records.shape==(2*row['teams'],923)
        return records[row['teams']+row['seat']]
    pair_counts, npz_checks = {}, {}
    for split in ('train', 'val', 'test'):
        st = index['stats'][split]
        assert st['roots'] == root_split[split] and st['pairs'] == 6 * st['roots']
        p = RANK / split / 'rank.npz'
        if historical is not None:
            assert sha(p)==historical['rank_npz'][split]['sha256'],'Raw ranking bytes changed'
        with np.load(p, allow_pickle=False) as z:
            source_check=verify_rank_sources(z,source_rows(source_paths[split]),source_record,match_info)
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
                                 'all_uncertain_roots': int((~resolved_roots).sum()),
                                 'internal_consistency_verified':True,**source_check}
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
                        'frozen_identity_verified': True,'source_alignment_status':'passed',
                        'manifest_verification':manifest_verification,'root_content_manifest_sha256': root_manifest.hexdigest(),
                        'rank_npz': npz_checks},
              'report_recovery': {'reason': "Original report finalizer indexed pairs in the all/dropped_roots metadata row.",
                                  'original_traceback_retained': 'build/t0010/post_collection.stderr.log',
                                  'completed_stages_replayed': [], 'audit_wall_seconds': round(time.monotonic()-start, 3)}}
    write(REPORT / 'COLLECTION_RESULT.json', result)
    write(REPORT / 'DATA_AUDIT.json', result['audit'])
    write(REPORT / 'FINALIZATION_RECOVERY.json', result['report_recovery'])
    from t0010_render_reports import render
    render(REPORT)
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

## Final report rendering and truthful review scope

The renderer below uses only published receipts, including their explicit verification scope. Save it as `t0010_render_reports.py` and run from the repository root. It recreates COLLECTION.md and CHECKS.md, including check completion, pending host source alignment and LF transport text. It does not replay checks, collection or continuations. The historical CHECKS.json remains a receipt for the measured source; REVIEW_FIX_CHECKS.json records the separate review-fix verification.

```python
"""Reproduce PR #23 report prose from published receipts and explicit audit scope."""
import json
from pathlib import Path


def render(report):
    report = Path(report)
    read = lambda name: json.loads((report / name).read_text())
    result, audit, frozen, checks, adapter = map(read, (
        'COLLECTION_RESULT.json', 'DATA_AUDIT.json', 'FROZEN_COLLECTION.json',
        'CHECKS.json', 'MANIFEST_METADATA_ADAPTER.json'))
    if result['audit'] != audit:
        raise ValueError('Audit receipt copies disagree')
    if len(checks['checks']) != 3 or any(c['exit_code'] != 0 for c in checks['checks']):
        raise ValueError('Incomplete historical delivery checks')
    source_passed = audit.get('source_alignment_status') == 'passed'
    if source_passed and any(not entry.get('source_row_alignment_verified')
                             for entry in audit['rank_npz'].values()):
        raise ValueError('Source alignment claimed without per-split evidence')
    manifest = audit.get('manifest_verification', adapter)
    manifest_passed = manifest.get('all_other_frozen_fields_exact') is True
    if manifest_passed and manifest.get('full_baseline_comparison') != 'passed':
        raise ValueError('Full manifest claim has no verified baseline')
    prefix = result['prefix_mismatch_distribution']
    nonzero = sum(v for k, v in prefix.items() if int(k) != 0)
    retained = sum(v['roots'] for v in audit['rank_npz'].values())
    dropped = sum(result['dropped_roots'].values())
    lines = ['# T0010 frozen-carrier V training data', '',
             'Generated by t0010_render_reports.py embedded in REPRODUCE.md; raw matches, roots and ranking arrays remain local.', '',
             f"Frozen source `{frozen['source_commit']}`. Old p2 `v12_ponly_xxl`, `models_xxl`, V off; training-only seed 20261201, 500 maps per count 2-15, 6 workers, official 2000/50 ms limits.", '',
             f"All {result['kept_matches']:,} scheduled matches are present, with {audit['match_faults']} match faults. All {result['expected_roots']:,} expected root keys are present. {dropped} roots were explicitly dropped ({dropped/result['expected_roots']:.2%}); reasons: {result['dropped_roots']}. Retained roots: {retained:,}, continuations: {result['continuations']:,}.", '',
             f"Continuation faults: {result['continuation_faults']}; prefix faults: {result['prefix_faults']}; terminal successors: {result['terminal_candidates']}. Recomputed prefix replies differed in {nonzero:,}/{result['continuations']:,} continuations ({nonzero/result['continuations']:.3%}); maximum mismatch count {max(map(int,prefix))}. No root was dropped for failure to reproduce recorded state blocks. The complete nondeterminism distribution is preserved in COLLECTION_RESULT.json.", '',
             '| Map split | Retained roots | Pairs | Resolved | Uncertain |',
             '|---|---:|---:|---:|---:|']
    for split in ('train', 'val', 'test'):
        counts = result['pair_counts'][split]
        lines.append(f"| {split} | {audit['rank_npz'][split]['roots']:,} | {counts['pairs']:,} | {counts['resolved_pairs']:,} | {counts['uncertain_pairs']:,} |")
    lines += ['', 'Resolved labels require an outcome gap strictly exceeding one pairwise step, 0.5/(teams-1). Each candidate has one continuation; tie-z=2 adds no repeat-noise estimate with repeats=1. Uncertain pairs retain label -1 and zero weight. Resolved weights sum to one per informative root. Entire maps are disjoint across training, validation and test.', '',
              'This is single-arm training-data collection. Historical audit results establish schedule/root coverage, frozen selected identities and internal array shape/label/weight consistency.', '',
              ('The strict raw-source alignment audit passed: outcomes were recomputed from branch scores, root/candidate metadata and order were checked, and every successor record was independently re-extracted.' if source_passed else
               'Raw-source alignment remains pending on the host. The original audit did not compare rank rows against raw branch outcomes/metadata/features. Corrected receipts preserve historical counts and hashes and do not certify that missing check. Run the strict audit in REPRODUCE.md before claiming source-aligned training data.'), '',
              ('Original and current normalized manifest baselines were compared successfully.' if manifest_passed else
               'Full original-manifest comparison was not performed by the historical adapter. Its selected frozen identity/source checks remain valid; all_other_frozen_fields_exact is false. Supply a hash-bound original baseline for full verification, or use the explicit limited-scope legacy mode.'), '',
              '## Execution and recovery', '',
              'The original build-manifest digest receipt is immutable. MANIFEST_METADATA_ADAPTER.json records its actual verification scope; a current normalized digest is not a captured original baseline. Completed preparation/collection stages were retained without replay.', '',
              'All four pipeline stages completed with exit code 0. The report finalizer failed on the all/dropped_roots metadata row. Its traceback/state remain local. Report-only recovery retained all measurement stages; no collection, continuation, ranking, training or export was replayed.', '',
              '| Stage | Wall seconds |', '|---|---:|']
    lines += [f"| {s['stage']} | {s['wall_seconds']:.3f} |" for s in result['steps']]
    lines += ['', 'The passive observer sampled verified descendants every 15 seconds without process termination, paging changes or resource-setting changes. Historical delivery checks passed: core189 (1 platform skip), P-only43 and29 single-file bundles; see CHECKS.md. These are checks of the measured source, not a host re-audit after this review fix. User-required independent-branch PR delivery replaces atomic main/task-board updates.', '',
              'Published screen by-product copies use Git-required LF line endings. RECEIPT_TRANSPORT.json maps their original raw digests to published digests; JSON values and measured data/build bytes remain unchanged. FROZEN_COLLECTION.json retains the original measurement digests.', '']
    (report / 'COLLECTION.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    lines = ['# T0010 delivery checks', '',
             f"Historical validation source: `{checks['source_commit']}`; Python {checks['python']}.", '',
             'Generated by t0010_render_reports.py embedded in REPRODUCE.md from CHECKS.json. Historical full logs remain under build/t0010/. Their recorded digests are preserved; these wall times are validation only.', '']
    for c in checks['checks']:
        lines += [f"## {c['name']}", '', f"`{' '.join(c['command'])}`: exit {c['exit_code']}.", '',
                  '```text', *c['summary'], '```', '', f"Saved log SHA-256: `{c['full_log_sha256']}`.", '']
    lines += ['DATA_AUDIT.json preserves the original internal-consistency audit and explicitly records whether raw-source alignment has subsequently passed. MANIFEST_METADATA_ADAPTER.json distinguishes full original-baseline verification from selected-identity checks.', '',
              'REVIEW_FIX_CHECKS.json records local regression/rendering checks for this review fix. Host raw files were not available for that local verification; it is not a receipt for the strict host re-audit.', '']
    (report / 'CHECKS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    render(Path('bots/p_only/report/v_data'))
```

## Review-scope correction and regressions

The historical host audit predates these fixes. The correction script below derives the scoped receipts from the original PR commit, preserves every measured count and data hash, and refuses to overwrite an already passed strict audit. Save it as `t0010_correct_scope.py`; run it only to reproduce this review correction, then run the renderer. Fetching the PR's Git objects may be needed to read that historical commit; no working-tree pull, collection, or ranking is performed.

```python
"""Reclassify historical receipts without inventing a host re-audit."""
import json
import subprocess
from pathlib import Path

BASE = 'de69645a7dc2652c710d2000fcf29f2df0b9619b'
REPORT = Path('bots/p_only/report/v_data')


def correct_scope():
    current = json.loads((REPORT / 'DATA_AUDIT.json').read_text())
    if current.get('source_alignment_status') == 'passed':
        raise ValueError('Refuse to replace a completed strict host audit with historical scope')
    def original(name):
        raw = subprocess.check_output(['git', 'show', BASE + ':' + (REPORT / name).as_posix()])
        return json.loads(raw)
    audit, result, adapter = map(original, (
        'DATA_AUDIT.json', 'COLLECTION_RESULT.json', 'MANIFEST_METADATA_ADAPTER.json'))
    for entry in audit['rank_npz'].values():
        entry.pop('labels_weights_shapes_split_verified', None)
        entry.update(internal_consistency_verified=True, source_row_alignment_verified=False,
                     source_alignment_status='pending_host_reaudit')
    audit.update(source_alignment_status='pending_host_reaudit',
        source_alignment_note='Historical audit checked internal array consistency; the stricter raw-score/metadata/feature audit has not been executed on retained host artifacts by this revision.')
    result['audit'] = audit
    adapter.update(all_other_frozen_fields_exact=False,
        verification_scope='selected_frozen_identities_only',
        full_baseline_comparison='not_performed_by_original_controller',
        unverified_scope='original manifest nonvolatile content',
        reason='Original controller compared selected frozen identity receipts and current source hashes, then hashed the current normalized manifest. No captured normalized manifest comparison was performed. The immutable original raw digest remains preserved.')
    for name, value in [('DATA_AUDIT.json', audit), ('COLLECTION_RESULT.json', result),
                        ('MANIFEST_METADATA_ADAPTER.json', adapter)]:
        (REPORT / name).write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    correct_scope()
```

Save the next fence as `test_t0010_review_contracts.py` beside `t0010_review_contracts.py`. Run from the repository root with `python <script-directory>/test_t0010_review_contracts.py -v`. It exercises the review counterexamples and a native encoder on recorded input, without playing matches.

```python
"""Regression checks for the embedded PR #23 source and baseline audits."""
import copy
import hashlib
import itertools
import json
import gzip
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from t0010_review_contracts import verify_manifest_baseline, verify_rank_sources


def fixture():
    orders = ['10 S - S - S -', '10 U - S - S -',
              '10 S - U - S -', '10 S - S - U -']
    scores = [[4, 0, 2], [2, 4, 0], [0, 4, 2], [2, 2, 4]]
    outcomes = [1., .5, 0., .25]
    row = {'match_id': 'fixture', 'turn': 10, 'teams': 3, 'seat': 0,
           'map_seed': 1, 'init': 'test-init', 'candidates': orders,
           'branches': [{'candidate': i, 'repeat': 0, 'on_policy': i == 0,
                         'line': orders[i], 'score': scores[i], 'outcome': outcomes[i],
                         'root_block': 'test-root', 'next_block': 'successor-' + str(i)} for i in range(4)]}
    info = {'fixture': {'teams': 3, 'seat': 0, 'map_seed': 1, 'init': 'test-init',
                        'total_turns': 400, 'roots': {10: {'block': 'test-root', 'reply': orders[0]}}}}
    def record(row, branch):
        a = np.zeros(923, np.int16)
        a[:3] = [row['turn'] + 1, row['seat'], branch['candidate'] + 7]
        return a
    pairs = [(i, j, -1 if abs(outcomes[i] - outcomes[j]) <= .25 else int(outcomes[i] > outcomes[j]))
             for i, j in itertools.combinations(range(4), 2)]
    resolved = sum(y >= 0 for _, _, y in pairs)
    z = {'root_map': np.array([1], np.int64), 'root_turn': np.array([10], np.int16),
         'root_teams': np.array([3], np.int8), 'cand_root': np.zeros(4, np.int32),
         'cand_n': np.ones(4, np.int16), 'cand_on_policy': np.array([True, False, False, False]),
         'cand_y': np.array(outcomes, np.float32), 'cand_rec': np.array([record(row, b) for b in row['branches']]),
         'pair_a': np.array([i for i, _, _ in pairs], np.int32),
         'pair_b': np.array([j for _, j, _ in pairs], np.int32),
         'pair_y': np.array([y for _, _, y in pairs], np.int8),
         'pair_w': np.array([1 / resolved if y >= 0 else 0 for _, _, y in pairs], np.float32)}
    return z, row, record, info


class RankSources(unittest.TestCase):
    def test_compressed_npz_source_alignment(self):
        z, row, record, info = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rank.npz'
            np.savez_compressed(path, **z)
            with np.load(path, allow_pickle=False) as archive:
                result = verify_rank_sources(archive, [row], record, info)
        self.assertEqual(result['successor_records_reextracted'], 4)

    def test_valid_source_alignment(self):
        z, row, record, info = fixture()
        result = verify_rank_sources(z, [row], record, info)
        self.assertTrue(result['source_row_alignment_verified'])
        self.assertEqual(result['successor_records_reextracted'], 4)

    def reject(self, alter, message):
        z, row, record, info = fixture()
        alter(z, row)
        with self.assertRaisesRegex(ValueError, message):
            verify_rank_sources(z, [row], record, info)

    def test_rejects_root_turn_999(self):
        self.reject(lambda z, r: z['root_turn'].fill(999), 'root metadata')

    def test_rejects_all_on_policy_false(self):
        self.reject(lambda z, r: z['cand_on_policy'].fill(False), 'candidate metadata')

    def test_rejects_forged_features_with_valid_shape_and_header(self):
        self.reject(lambda z, r: z['cand_rec'][:, 2:].fill(0), 'features differ')

    def test_rejects_internally_consistent_forged_outcomes(self):
        def alter(z, row):
            z['cand_y'].fill(0)
            z['pair_y'].fill(-1)
            z['pair_w'].fill(0)
        self.reject(alter, 'outcome differs')

    def test_rejects_raw_outcome_not_derived_from_scores(self):
        self.reject(lambda z, r: r['branches'][0].update(outcome=.7), 'final scores')

    def test_rejects_raw_wrong_on_policy(self):
        self.reject(lambda z, r: r['branches'][0].update(on_policy=False), 'on-policy')

    def test_rejects_pair_reordering(self):
        self.reject(lambda z, r: z['pair_b'].__setitem__(0, 3), 'Array pair')

    def test_rejects_wrong_candidate_turn(self):
        def alter(z, r):
            r['candidates'][1] = '999 U - S - S -'
            r['branches'][1]['line'] = r['candidates'][1]
        self.reject(alter, 'order turn differs')


class ManifestBaseline(unittest.TestCase):
    def setUp(self):
        self.original = {'flags': ['-std=c++20', '-O2'], 'updated': 'freeze',
                         'variants': {'bot': {'compile_seconds': 1, 'binary_sha256': 'frozen'}}}
        self.raw = (json.dumps(self.original) + '\n').encode()
        self.sha = hashlib.sha256(self.raw).hexdigest()

    def test_allows_only_volatile_metadata_changes(self):
        current = copy.deepcopy(self.original)
        current['updated'] = 'later'
        current['variants']['bot']['compile_seconds'] = 2
        result = verify_manifest_baseline(current, self.sha, self.raw)
        self.assertTrue(result['all_other_frozen_fields_exact'])

    def test_rejects_o2_to_o3(self):
        current = copy.deepcopy(self.original)
        current['flags'][1] = '-O3'
        with self.assertRaisesRegex(ValueError, 'nonvolatile'):
            verify_manifest_baseline(current, self.sha, self.raw)

    def test_rejects_missing_original_baseline_by_default(self):
        with self.assertRaisesRegex(ValueError, 'No captured'):
            verify_manifest_baseline(self.original, self.sha)

    def test_legacy_scope_cannot_certify_full_manifest(self):
        result = verify_manifest_baseline(self.original, self.sha, allow_selected_only=True)
        self.assertFalse(result['all_other_frozen_fields_exact'])
        self.assertEqual(result['verification_scope'], 'selected_frozen_identities_only')

    def test_rejects_wrong_original_digest(self):
        with self.assertRaisesRegex(ValueError, 'immutable captured digest'):
            verify_manifest_baseline(self.original, 'wrong', self.raw)


class NativeRecordBinding(unittest.TestCase):
    def test_recorded_two_team_successors(self):
        repo = Path.cwd()
        fixture_path = repo / 'bots/p_only/tests/fixtures/stream_n02.txt.gz'
        if not fixture_path.is_file():
            self.skipTest('Run from the frozen repository for native fixture coverage')
        sys.path.insert(0, str(repo))
        from fusion import native, matchlog
        stream = gzip.decompress(fixture_path.read_bytes()).decode()
        start = stream.index('\nTURN 1\n') + 1
        init = stream[:start]
        blocks = re.findall(r'TURN \d+\n.*?END\n', stream[start:], re.S)
        z, row, _, info = fixture()
        orders = ['2 S - S - S -', '2 U - S - S -', '2 S - U - S -', '2 S - S - U -']
        scores, outcomes = [[1, 0], [0, 1], [1, 1], [2, 0]], [1., 0., .5, 1.]
        row.update(turn=2, teams=2, init=init, candidates=orders)
        for i, b in enumerate(row['branches']):
            b.update(line=orders[i], score=scores[i], outcome=outcomes[i],
                     root_block=blocks[1], next_block=blocks[2])
        info['fixture'].update(teams=2, init=init, roots={2: {'block': blocks[1], 'reply': orders[0]}})
        def extract(row, branch):
            text = matchlog.init_for_seat(row['init'], row['seat']) + branch['root_block'] + branch['next_block']
            records = native.extract(text)
            self.assertEqual(records.shape, (4, 923))
            return records[2]
        record = extract(row, row['branches'][0])
        z['cand_rec'] = np.repeat(record[None], 4, axis=0)
        z['cand_y'] = np.array(outcomes, np.float32)
        z['root_turn'][0], z['root_teams'][0] = 2, 2
        pairs = [(i, j, -1 if abs(outcomes[i]-outcomes[j]) <= .5 else int(outcomes[i]>outcomes[j]))
                 for i, j in itertools.combinations(range(4), 2)]
        resolved = sum(y >= 0 for _, _, y in pairs)
        z['pair_y'] = np.array([y for _, _, y in pairs], np.int8)
        z['pair_w'] = np.array([1/resolved if y>=0 else 0 for _, _, y in pairs], np.float32)
        result = verify_rank_sources(z, [row], extract, info)
        self.assertEqual(result['successor_records_reextracted'], 4)
        z['cand_rec'][1, 17] ^= 1
        with self.assertRaisesRegex(ValueError, 'features differ'):
            verify_rank_sources(z, [row], extract, info)


if __name__ == '__main__':
    unittest.main()
```

## Local review-fix evidence

Save the next fence as `t0010_review_fix_checks.py` beside the extracted helper, renderer and regression test. After capturing the regression and repository-layout/task-board test logs, run `python <script-directory>/t0010_review_fix_checks.py <regression-log> <structure-log>` from the repository root. It creates REVIEW_FIX_CHECKS.json from actual successful logs and verifies script identity, rendering and transport hashes. This local evidence never replaces a host raw-source audit.

```python
"""Generate local PR #23 regression evidence; never certify host raw data."""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    report = Path('bots/p_only/report/v_data')
    regression_log, structure_log = map(Path, sys.argv[1:])
    def receipt(path, count):
        raw = path.read_bytes()
        text = raw.decode()
        if not re.search(rf'Ran {count} tests in .*\n\nOK\s*$', text):
            raise ValueError('Missing successful unskipped test log: ' + str(path))
        return {'tests': count, 'status': 'passed',
                'log_sha256': hashlib.sha256(raw).hexdigest(),
                'summary': text[text.rfind('Ran '):].strip()}
    checks = {'review_regressions': receipt(regression_log, 16),
              'repo_layout_and_task_board': receipt(structure_log, 22)}
    source = (report / 'REPRODUCE.md').read_text()
    fences = re.findall(r'```python\n(.*?)\n```', source, re.S)
    for fence in fences:
        ast.parse(fence)
    scripts = {}
    for name in ('t0010_review_contracts.py', 't0010_render_reports.py',
                 'test_t0010_review_contracts.py', 't0010_review_fix_checks.py'):
        raw = Path(__file__).with_name(name).read_bytes()
        if raw.decode().rstrip() not in fences:
            raise ValueError('Extracted source differs from embedded source: ' + name)
        scripts[name] = hashlib.sha256(raw).hexdigest()
    from t0010_render_reports import render
    names = ('COLLECTION.md', 'CHECKS.md')
    before = {n: (report / n).read_bytes() for n in names}
    render(report)
    if before != {n: (report / n).read_bytes() for n in names}:
        raise ValueError('Published reports differ from receipt-only rendering')
    read = lambda name: json.loads((report / name).read_text())
    audit = read('DATA_AUDIT.json')
    if read('COLLECTION_RESULT.json')['audit'] != audit:
        raise ValueError('Duplicate audit receipts disagree')
    transport = read('RECEIPT_TRANSPORT.json')['files']
    for entry in transport:
        raw = (report / entry['path']).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != entry['git_lf_sha256']
                or hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest() != entry['original_sha256']):
            raise ValueError('Transport digest mismatch: ' + entry['path'])
    subprocess.run(['git', 'diff', '--check'], check=True)
    result = {'scope': 'Local review-fix verification only; no matches or training',
              'python': sys.version, 'checks': checks, 'python_fences_parsed': len(fences),
              'embedded_script_sha256': scripts, 'report_rendering': 'byte_identical',
              'receipt_transport_hash_pairs_verified': len(transport), 'diff_check': 'passed',
              'strict_host_raw_reaudit_executed': False,
              'reason': 'Retained Windows match/branch/NPZ files and original manifest bytes were not available locally.'}
    (report / 'REVIEW_FIX_CHECKS.json').write_text(json.dumps(result, indent=1) + '\n', newline='\n')
    print('Local review-fix verification passed')


if __name__ == '__main__':
    main()
```

## Host re-audit

Extract the helper above as `t0010_review_contracts.py`, the first driver as `t0010_pipeline.py`, the completed-artifact audit as `t0010_audit_completed.py`, and the report renderer as `t0010_render_reports.py`, in one directory outside the tracked tree. From the frozen host checkout:

```powershell
python tools/repo/task.py lock --task T0010 cpu -- python <script-directory>/t0010_audit_completed.py
```

If the original manifest snapshot/digest was not retained, first verify that it cannot be recovered from the immutable receipts. Use the following explicit fallback for this historical run; the resulting receipt must retain `all_other_frozen_fields_exact=false` and its limited scope:

```powershell
python tools/repo/task.py lock --task T0010 cpu -- python <script-directory>/t0010_audit_completed.py --allow-selected-manifest
python <script-directory>/t0010_render_reports.py
```

This performs exhaustive raw-score, root/candidate metadata and independently re-extracted successor-feature comparisons, using the original file order and preserved source hashes. Publish the receipts actually generated by that run; do not change pending status by hand. Full baseline verification and selected-identity verification are distinct. An original snapshot recovered from another immutable location can be supplied with `--original-manifest <file>`; its raw digest must match FROZEN_COLLECTION.json. Never overwrite that receipt with a new baseline computed after collection.
