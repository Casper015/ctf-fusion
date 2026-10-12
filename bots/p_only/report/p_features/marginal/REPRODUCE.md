# Reproduce the new P2/P3 branch diagnostic

Measurement source 64576e0d6d428e5f45f9ad0a89216d6c4808bfb4; the initial report head c3b6e59cd87b4b6d68b3f1e4a265dd135e9d4bea did not change production code. Old T13 games/cost/analysis and old T4 swaps are completed and must not be repeated. This diagnostic is a new treatment on26 frozen T13 input streams and uses no new game or seed.

Set CXX to GCC15.2, T0013_PYTHON to Python3.14 and OMP/MKL/OpenBLAS8 at runtime. Compile and execute inside a single exclusive T0013 CPU lease, with no formal GPU lease. Use `python tools/repo/task.py lock --task T0013 --wait 60 cpu -- pwsh -NoProfile -ExecutionPolicy Bypass -File ../../t0013_swap_run.ps1`. Completed per-mode/map receipts are verified and skipped; successful binaries must not be rebuilt. The passive15-second observer does not change resources or user applications. Raw stdout/stderr, generated diagnostic carrier/View copies and DLL/executables remain ignored in build/t0013_marginal.

The files below are the exact original runtime sources; extract to the original relative work locations if reproducing in a separate independent study. The driver generates its typed hybrid View and modules without editing production source. Their hashes and original binaries/inputs are frozen in FROZEN_DIAGNOSTIC.json and DRIVER_BINDING.json.

## t0013_swap.py

SHA256 `a6ee7604b039b739db2688c7ac101944ab2937c92bafdf509afd35802b304619`; 13625 bytes.

```python
"""New P2->P3 branch intervention on retained T13 snapshots; no games."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
P=Path.cwd(); sys.path.insert(0,str(P))
from fusion import matchlog
from fusion.protocol import parse_reply
B=P/'build/t0013_marginal'; G=B/'src'; R=P/'bots/p_only/report/p_features/marginal'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(v,indent=1)+'\n',encoding='utf-8',newline='\n')
def note(stage,**kw): save(B/'state.json',{'stage':stage,**kw}); print(json.dumps({'stage':stage,**kw}),flush=True)
def rep(s,a,b,n=1): assert s.count(a)==n,(a,s.count(a),n); return s.replace(a,b)

HOOK=r'''#pragma once
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef near
#undef far
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "policy_p3.hpp"
namespace ponly {
inline bool t13_branch=false;
inline long long t13_calls=0,t13_close=0,t13_distant=0,t13_mid=0,t13_inactive=0,t13_selected=0;
inline int t13_mode(){static int v=[](){const char* p=std::getenv("T13_SWAP_MODE");return p?std::atoi(p):0;}();return v;}
inline const fusion::PolicyModelP3* t13_model(int teams){
 using Lookup=const fusion::PolicyModelP3*(*)(int);
 static Lookup lookup=[](){const char* p=std::getenv("T13_P3_DLL");if(!p)throw std::runtime_error("missing P3 module");HMODULE h=LoadLibraryA(p);if(!h)throw std::runtime_error("load P3 module failed");auto f=reinterpret_cast<Lookup>(GetProcAddress(h,"t13_p3_model"));if(!f)throw std::runtime_error("P3 ABI missing");return f;}();
 auto p=lookup(teams);if(!p)throw std::runtime_error("P3 registry missing");return p;
}
inline bool t13_select(const fusion::Board& b,const fusion::State& s,int me,int team){
 if(!t13_branch)return false;
 ++t13_calls;std::vector<int> sources,dist;
 for(const auto& u:s.units)if(u.team==me&&u.x>=0)sources.push_back(b.cell(u.x,u.y));
 b.bfs(sources.data(),int(sources.size()),dist);int d=fusion::kUnreachable;
 for(const auto& u:s.units)if(u.team==team&&u.x>=0)d=std::min(d,dist[b.cell(u.x,u.y)]);
 bool close=d<=3,distant=d>=7&&d<fusion::kUnreachable;
 t13_close+=close;t13_distant+=distant;t13_mid+=d>=4&&d<=6;t13_inactive+=sources.empty();
 int mode=t13_mode();if(mode<0||mode>4)throw std::runtime_error("invalid mode");
 bool use=mode==4||(mode==2&&distant)||(mode==3&&close);t13_selected+=use;return use;
}
inline void t13_report(){std::cerr<<"T13 mode "<<t13_mode()<<" branch_teams "<<t13_calls<<" close "<<t13_close<<" distant "<<t13_distant<<" middle "<<t13_mid<<" inactive "<<t13_inactive<<" selected_p3 "<<t13_selected<<'\n';}
}
'''

def generate():
    G.mkdir(parents=True,exist_ok=True)
    (G/'swap_hook.hpp').write_text(HOOK,encoding='utf-8',newline='\n')
    s=(P/'bots/p_only/include/ponly.hpp').read_text(encoding='utf-8')
    s=rep(s,'namespace ponly {','#include "swap_hook.hpp"\nnamespace ponly {')
    s=rep(s,'Counters* counters = nullptr;','Counters* counters = nullptr;\n    fusion::Rules rules{};')
    s=rep(s,'has_.assign(teams_, 0);','has_.assign(teams_, 0);\n        useP3_.assign(teams_,0);\n        bool csReady=false;')
    s=rep(s,'fusion::encodePolicy(b, s_, v_, t, feat_[t]);','fusion::encodePolicy(b, s_, v_, t, feat_[t]);\n            useP3_[t]=t13_select(b,s_,me,t);\n            if(useP3_[t]) { if(!csReady) {fusion::summarize(b,rules,s_,v_,cs_);csReady=true;} fusion::encodeCombatPolicy(b,s_,v_,cs_,t,feat_[t].c); }')
    s=rep(s,'fusion::jointPolicy(*model_, feat_[t], out_[t]);','if(useP3_[t]) fusion::jointPolicy(*t13_model(teams_),feat_[t],out_[t]);\n                else fusion::jointPolicy(*model_,feat_[t],out_[t]);')
    s=rep(s,'fusion::actPolicy(*model_, feat_[t], k, m, a.data());','if(useP3_[t]) fusion::actPolicy(*t13_model(teams_),feat_[t],k,m,a.data());\n            else fusion::actPolicy(*model_,feat_[t],k,m,a.data());')
    s=rep(s,'fusion::targetPolicy(*model_, feat_[t], k, m, p.data());','if(useP3_[t]) fusion::targetPolicy(*t13_model(teams_),feat_[t],k,m,p.data());\n            else fusion::targetPolicy(*model_,feat_[t],k,m,p.data());')
    s=rep(s,'std::vector<fusion::PolicyFeatures> feat_;','std::vector<fusion::PolicyFeaturesP3> feat_;\n    std::vector<char> useP3_;\n    fusion::CombatSummary cs_;')
    (G/'ponly_swap.hpp').write_text(s,encoding='utf-8',newline='\n')
    c=(P/'bots/p_only/bots/c08_ponly_f.hpp').read_text(encoding='utf-8')
    # Common fixed-work deadline change, identical in the original and routed copies.
    c=rep(c,'chrono::milliseconds(g.turn==1?1800:PONLY_TURN_MS)','chrono::milliseconds(1000000)')
    (G/'c08_original.hpp').write_text(c,encoding='utf-8',newline='\n')
    c=rep(c,'#include "ponly.hpp"','#include "ponly_swap.hpp"')
    c=rep(c,'#define PONLY_RULES_TO(v) ((void)0)','#define PONLY_RULES_TO(v) ((v).rules = rules)')
    c=rep(c,'branchView.build(board, makeState(next), &view.state(), next.me, policy());','ponly::t13_branch=true;\n            branchView.build(board, makeState(next), &view.state(), next.me, policy());\n            ponly::t13_branch=false;')
    c=rep(c,'out.build(board, s, &view.state(), me, policy(), teams);','ponly::t13_branch=true;\n            out.build(board, s, &view.state(), me, policy(), teams);\n            ponly::t13_branch=false;')
    (G/'c08_swap.hpp').write_text(c,encoding='utf-8',newline='\n')
    replay=(P/'bots/p_only/tests/replay_test.cpp').read_text(encoding='utf-8')
    for name in ('original','swap'):
        s=rep(replay,'#include "c08_ponly_f.hpp"',f'#include "c08_{name}.hpp"')
        if name=='swap':s=rep(s,'    return 0;\n}','    ponly::t13_report();\n    return 0;\n}')
        (G/f'replay_{name}.cpp').write_text(s,encoding='utf-8',newline='\n')
    module='#include "models_pf_p3xxl_vm/registry.hpp"\nextern "C" __declspec(dllexport) const fusion::PolicyModelP3* t13_p3_model(int n){auto p=fusion::models::forTeams(n);return p?p->policy:nullptr;}\n'
    (G/'module.cpp').write_text(module,encoding='utf-8',newline='\n')
    verifier=r'''#include "models_pf_p3xxl_vm/registry.hpp"
#include "swap_hook.hpp"
#include <cassert>
int main(){long long checked=0;for(int n=2;n<=15;++n){auto a=fusion::models::forTeams(n)->policy;auto b=ponly::t13_model(n);const fusion::Net* x[]={&a->mover,&a->squad,&a->act,&a->target};const fusion::Net* y[]={&b->mover,&b->squad,&b->act,&b->target};for(int h=0;h<4;++h){assert(x[h]->in()==y[h]->in()&&x[h]->out()==y[h]->out()&&x[h]->outExp==y[h]->outExp);for(int k=0;k<16;++k){int32_t input[fusion::kMaxWidth],left[fusion::kMaxWidth],right[fusion::kMaxWidth];for(int i=0;i<x[h]->in();++i)input[i]=((i+k)%7-3)*16;fusion::forwardRaw(*x[h],input,left);fusion::forwardRaw(*y[h],input,right);for(int j=0;j<x[h]->out();++j){assert(left[j]==right[j]);++checked;}}}}std::cout<<"exact_raw_outputs "<<checked<<'\n';}
'''
    (G/'verify.cpp').write_text(verifier,encoding='utf-8',newline='\n')

def build():
    generate(); exe={}
    flags=['-std=c++20','-O2','-static','-DFUSION_MAX_WIDTH=1024',f'-I{G}',f'-I{P/"inference"}',f'-I{P/"bots/p_only/include"}',f'-I{P/"bots/p_only/bots"}']
    for name in ('module','verify','original','swap'):
        note('compile',name=name)
        target=B/(name+('.dll' if name=='module' else '.exe'))
        if target.exists():raise RuntimeError('Existing successful binary; do not rebuild '+name)
        if name=='module':cmd=[os.environ['CXX'],*flags,'-shared',str(G/'module.cpp'),'-o',str(target)]
        elif name=='verify':cmd=[os.environ['CXX'],*flags,str(G/'verify.cpp'),'-o',str(target)]
        else:cmd=[os.environ['CXX'],*flags,'-DCARRIER_C08','-DPONLY_FAMILIES','-DPONLY_POLICY_FAMILY=2','-DPONLY_VALUE_FAMILY=2','-DMOD_FUSION_V=0','-DPONLY_FIXED_SAMPLES=128','-DPONLY_STATS=1','-include',str(P/'inference/models_pf_p2xxl_vm/registry.hpp'),str(G/f'replay_{name}.cpp'),'-o',str(target)]
        cp=subprocess.run(cmd,capture_output=True);(B/(name+'.compile.log')).write_bytes(cp.stdout+cp.stderr)
        assert cp.returncode==0,cp.stderr.decode(errors='replace')[-2000:]
        exe[name]=target
    env={**os.environ,'T13_P3_DLL':str(exe['module'])}
    v=subprocess.run([str(exe['verify'])],env=env,capture_output=True,check=True)
    save(B/'MODULE_PARITY.json',{'passed':True,'output':v.stdout.decode().strip(),'cases':'14 team counts x 4 heads x 16 deterministic input vectors, integer outputs and scaling exponents exact; synthetic inputs, not a re-evaluation of prediction losses'})
    return exe

def replay(entry,mode,exe,identity):
    target=B/'replays'/(entry['fixture_id']+'.'+mode+'.json')
    if target.exists():
        v=read(target);assert v['identity']==identity;return v
    m=matchlog.load(P/entry['path']);assert sha(P/entry['path'])==entry['sha256']
    env={**os.environ,'T13_P3_DLL':str(exe['module']),'T13_SWAP_MODE':str({'disabled':0,'branch_p2':1,'far_p3':2,'near_p3':3,'all_p3':4}.get(mode,0))}
    cp=subprocess.run([str(exe['original' if mode=='original' else 'swap'])],input=matchlog.seat_stream(m,entry['seat']).encode(),capture_output=True,env=env,timeout=7200)
    target.parent.mkdir(exist_ok=True);target.with_suffix('.stdout').write_bytes(cp.stdout);target.with_suffix('.stderr').write_bytes(cp.stderr)
    assert cp.returncode==0,cp.stderr.decode(errors='replace')[-1000:]
    rows=[];truncated=None
    for line in cp.stdout.decode().splitlines():
        if line.startswith('truncated '):truncated=int(line.split()[1])
        else:
            a,b,c=line.split(' ',2);assert parse_reply(c,int(a)) is not None;rows.append([int(a),b,c])
    assert len(rows)==400 and truncated==0,(mode,entry['fixture_id'],len(rows),truncated)
    v={'identity':identity,'fixture_id':entry['fixture_id'],'teams':entry['teams'],'input_sha256':entry['sha256'],'rows':rows,'truncated':truncated,'stdout_sha256':hashlib.sha256(cp.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(cp.stderr).hexdigest()}
    save(target,v);return v

def main():
    B.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    assert read(Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json')['task']=='T0013'
    assert not (Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
    plan=read(P/'data/p_only/p_features/schedule.json');selected=[];arm='c08_pf_p2xxl';key=hashlib.sha256(arm.encode()).hexdigest()[:10]
    for n in range(3,16):
        for fx in sorted([x for x in plan['fixtures'] if x['teams']==n],key=lambda x:x['fixture_id'])[:2]:
            path=Path('data/p_only/p_features/matches')/f'n{n:02d}'/(fx['fixture_id']+'-'+key+'.json.gz')
            selected.append({'fixture_id':fx['fixture_id'],'teams':n,'seat':fx['seat'],'path':path.as_posix(),'sha256':sha(P/path)})
    exe=build()
    frozen={'measurement_source':'64576e0d6d428e5f45f9ad0a89216d6c4808bfb4','current_report_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'states':selected,'generated_sources':{p.name:sha(p) for p in G.iterdir()},'repo_sources':{p.relative_to(P).as_posix():sha(p) for base in (P/'inference',P/'bots/p_only/search',P/'bots/p_only/include') for p in base.rglob('*.hpp')},'carrier_sha256':sha(P/'bots/p_only/bots/c08_ponly_f.hpp'),'replay_source_sha256':sha(P/'bots/p_only/tests/replay_test.cpp'),'binaries':{name:sha(path) for name,path in exe.items()},'fixed_samples':128,'deadline':'off','root_model':'unchanged T0011 P2 seed1 XXL general','route':'branch opponents only, team minimum BFS from active own units: close0..3/distant7..finite; desired-team mask unchanged','turns_per_mode':10400,'cache':'fresh View per state pins family route; no cross-mode cache reuse','no_new_games':True}
    identity=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest();frozen['identity']=identity
    save(R/'FROZEN_DIAGNOSTIC.json',frozen)
    per={};summary={'identity':identity,'modes':{},'module_parity':read(B/'MODULE_PARITY.json'),'causal_limits':'Decision sensitivity on 26 selected retained map streams, not strength, latency, tax amount, or proof of information blindness. All10400 turn denominator includes states with no eligible branch intervention. Different metric distance from T0011 Manhattan strata.'}
    for mode in ('original','disabled','branch_p2','far_p3','near_p3','all_p3'):
        results={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fs=[pool.submit(replay,e,mode,exe,identity) for e in selected]
            for f in concurrent.futures.as_completed(fs):
                v=f.result();results[v['fixture_id']]=v;note('replay',mode=mode,completed=len(results),total=26)
        per[mode]=results
        if mode in ('disabled','branch_p2'):
            assert all(v['rows']==per['original'][fid]['rows'] for fid,v in results.items()),'disabled/branch-P2 parity failure'
        changes=[]
        for fid,v in results.items():
            baseline=per['original'][fid];assert all(a[1]==b[1] for a,b in zip(v['rows'],baseline['rows'])),'root changed'
            changes.append({'fixture_id':fid,'teams':v['teams'],'changed_replies':sum(a[2]!=b[2] for a,b in zip(v['rows'],baseline['rows'])),'turns':400,'output_sha256':v['stdout_sha256']})
        summary['modes'][mode]={'changed_replies':sum(x['changed_replies'] for x in changes),'turns':10400,'root_digest_mismatches':0,'truncated':0,'maps':changes}
        save(R/'DECISION_SUMMARY.json',summary)
    note('complete',turns_per_mode=10400)

if __name__=='__main__':main()
```

## t0013_swap_run.ps1

SHA256 `c521d07adb37787c12f08f1e16f9762fccb8af30d833ad4e34cf5abc6dfb5f53`; 1794 bytes.

```powershell
$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$self=Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent=Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($parent.CommandLine -notmatch 'task.py lock.*T0013.*t0013_swap_run.ps1') { throw 'Unverified CPU lock parent' }
$lease=Get-Content -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/cpu.json') -Raw | ConvertFrom-Json
if ($lease.pid -ne $parent.ProcessId -or $lease.task -ne 'T0013') { throw 'Wrong CPU lease' }
if (Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.ctf-fusion/locks/gpu.json')) { throw 'GPU lease present; no formal CPU/GPU concurrency' }
$created=$parent.CreationDate.ToUniversalTime().ToString('o')
@{pipeline_pid=$parent.ProcessId;created_utc=$created;source_commit='64576e0d6d428e5f45f9ad0a89216d6c4808bfb4';helper_pid=$PID} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0013_marginal/pipeline_pid.json' -Encoding utf8
if (-not $env:T0013_PYTHON) { throw 'T0013_PYTHON must name the verified interpreter' }
$watch=Start-Process -FilePath 'pwsh.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','../../t0013_swap_monitor.ps1',
    '-RepoPath',$root,'-PipelinePid',[string]$parent.ProcessId,'-CreationUtc',$created,'-ExpectedCommandTag','t0013_swap_run.ps1') `
    -WindowStyle Hidden -RedirectStandardOutput 'build/t0013_marginal/resource_monitor.stdout' `
    -RedirectStandardError 'build/t0013_marginal/resource_monitor.stderr' -PassThru
@{monitor_pid=$watch.Id;created_utc=$watch.StartTime.ToUniversalTime().ToString('o');pipeline_pid=$parent.ProcessId} |
    ConvertTo-Json | Set-Content -LiteralPath 'build/t0013_marginal/resource_monitor_pid.json' -Encoding utf8
& $env:T0013_PYTHON ../../t0013_swap.py
exit $LASTEXITCODE
```

## t0013_swap_monitor.ps1

SHA256 `8d253c4c738bc9aaa9abe8deb258285c9d86c5208ae5fa397cb34ca7ebe925c4`; 3640 bytes.

```powershell
param([Parameter(Mandatory=$true)][string]$RepoPath,
      [Parameter(Mandatory=$true)][int]$PipelinePid,
      [Parameter(Mandatory=$true)][string]$CreationUtc,
      [string]$ExpectedCommandTag='t0013_swap_run.ps1')
$ErrorActionPreference='Stop'
$root = (Resolve-Path -LiteralPath $RepoPath).Path
$output = Join-Path $root 'build/t0013_marginal'
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
