"""Generate isolated report-only C08 diagnostic copies, then replay frozen states.

This never edits source lanes or model exports. Completed replays are retained.
26 maps, two lexicographically first retained matches per count, 400 turns each.
128 samples per search call, deadlines removed, V off, original root XXL P2.
Branch baseline S; intervention L at reference BFS0..3 or7+, plus all-L control.
Original and disabled routing controls must match every root digest and reply.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

P=Path.cwd();sys.path.insert(0,str(P))
from fusion import matchlog, native
R=P/'bots/p_only/report/near_far';B=P/'build/t0004';G=B/'swapsrc'
SOURCE='8a4eab4bef5bf65e971ec6712eebb89821738130'
MODES={'disabled':0,'branch_s':1,'far_l':2,'near_l':3,'all_l':4}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=1)+'\n',encoding='utf-8',newline='\n')
def note(stage,**kw):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stage=stage,**kw)
    write(B/'swap_state.json',row);print(json.dumps(row),flush=True)
def replacement(text,before,after,count):
    assert text.count(before)==count,(before,text.count(before),count)
    return text.replace(before,after)


def generate():
    G.mkdir(parents=True,exist_ok=True)
    view=(P/'bots/p_only/include/ponly.hpp').read_text(encoding='utf-8')
    view=replacement(view,'namespace ponly {','#include "swap_hook.hpp"\nnamespace ponly {',1)
    view=replacement(view,'has_.assign(teams_, 0);','has_.assign(teams_, 0);\n        teamModels_.assign(teams_,model);',1)
    view=replacement(view,'if (model_) {','if (teamModels_[t]) {',3)
    view=replacement(view,'*model_, feat_[t]','*teamModels_[t], feat_[t]',3)
    view=replacement(view,'fusion::encodePolicy(b, s_, v_, t, feat_[t]);',
                     'fusion::encodePolicy(b, s_, v_, t, feat_[t]);\n            teamModels_[t]=t4_route(b,s_,me,t,model);',1)
    view=replacement(view,'const fusion::PolicyModel* model_ = nullptr;',
                     'const fusion::PolicyModel* model_ = nullptr;\n    std::vector<const fusion::PolicyModel*> teamModels_;',1)
    (G/'ponly_swap.hpp').write_text(view,encoding='utf-8',newline='\n')
    carrier=(P/'bots/p_only/bots/c08_ponly.hpp').read_text(encoding='utf-8')
    # Common diagnostic deadline change is applied to BOTH original and routed builds.
    carrier=replacement(carrier,'chrono::milliseconds(g.turn==1?1800:35)','chrono::milliseconds(1000000)',1)
    (G/'c08_original.hpp').write_text(carrier,encoding='utf-8',newline='\n')
    carrier=replacement(carrier,'#include "ponly.hpp"','#include "ponly_swap.hpp"',1)
    carrier=replacement(carrier,'branchView.build(board, makeState(next), &view.state(), next.me, policy());',
                        'ponly::t4_in_branch=true;\n            branchView.build(board, makeState(next), &view.state(), next.me, policy());\n            ponly::t4_in_branch=false;',1)
    carrier=replacement(carrier,'out.build(board, s, &view.state(), me, policy(), teams);',
                        'ponly::t4_in_branch=true;\n            out.build(board, s, &view.state(), me, policy(), teams);\n            ponly::t4_in_branch=false;',1)
    (G/'c08_swap.hpp').write_text(carrier,encoding='utf-8',newline='\n')
    replay=(P/'bots/p_only/tests/replay_test.cpp').read_text(encoding='utf-8')
    for name in ('original','swap'):
        text=replacement(replay,'#include "c08_ponly.hpp"',f'#include "c08_{name}.hpp"',1)
        if name=='swap':text=replacement(text,'    return 0;\n}','    ponly::t4_report();\n    return 0;\n}',1)
        (G/f'replay_{name}.cpp').write_text(text,encoding='utf-8',newline='\n')
    module='#include "REGISTRY/registry.hpp"\nextern "C" __declspec(dllexport) const fusion::PolicyModel* t0004_model(int teams) {const auto* set=fusion::models::forTeams(teams);return set?set->policy:nullptr;}\n'
    for size,folder in [('S','models'),('L','models_l')]:
        (G/f'module_{size}.cpp').write_text(module.replace('REGISTRY',folder),encoding='utf-8',newline='\n')
        checker=(R/'policy_nll.cpp').read_text(encoding='utf-8')
        checker=replacement(checker,'#include "policy.hpp"','#include "policy.hpp"\n#include "swap_hook.hpp"',1)
        checker=replacement(checker,'const auto& model=*set->policy;',f'const auto& model=*ponly::t4_model({0 if size=="S" else 1},y[10]);',1)
        (G/f'check_{size}.cpp').write_text(checker,encoding='utf-8',newline='\n')


def build():
    generate();bins={}
    sources=list((P/'inference').rglob('*.hpp'))+list((P/'bots/p_only/search').rglob('*.hpp'))+list((P/'bots/p_only/include').glob('*.hpp'))
    root=P/'inference/models_pxxl_vm/registry.hpp'
    sources+=native.model_dependencies(root)
    for size in ('S','L'):
        target=B/f'model_{size}.dll';src=G/f'module_{size}.cpp'
        cmd=[native.compiler(),'-std=c++20','-O2','-shared','-static','-DFUSION_MAX_WIDTH=2048',f'-I{P/"inference"}',str(src),'-o',str(target)]
        note('compile_module',size=size)
        receipt=B/f'module_{size}.compile.json'
        if receipt.exists():
            old=read(receipt);assert old['source_sha256']==sha(src) and old['binary_sha256']==sha(target)
        else:
            cp=subprocess.run(cmd,capture_output=True)
            (B/f'module_{size}.compile.log').write_bytes(cp.stdout+cp.stderr)
            assert cp.returncode==0,cp.stderr.decode(errors='replace')[-3000:]
            write(receipt,dict(source_sha256=sha(src),binary_sha256=sha(target),exit_code=0))
        bins[size]=target
        bins['check_'+size]=native.build(G/f'check_{size}.cpp','t0004_module_check_'+size,deps=[*sources,R/'swap_hook.hpp'],
                                        defines=[f'T0004_TIER={0 if size=="S" else 1}','FUSION_MAX_WIDTH=2048'],includes=[R,P/'inference'])
    for name in ('original','swap'):
        note('compile_replay',name=name)
        prior=B/f'{name}.compile.json'
        if name=='original' and prior.exists():
            old=read(prior);target=P/'build'/old['binary_name']
            assert sha(G/'replay_original.cpp')==old['replay_sha256'] and sha(G/'c08_original.hpp')==old['carrier_sha256'] and sha(target)==old['binary_sha256']
            bins[name]=target
            continue
        bins[name]=native.build(G/f'replay_{name}.cpp','t0004_'+name,deps=[*sources,*G.glob('*.hpp'),R/'swap_hook.hpp'],
                                 defines=['CARRIER_C08','FUSION_TIER_PXXL_VM','FUSION_MAX_WIDTH=2048','PONLY_STATS=1','PONLY_FIXED_SAMPLES=128'],
                                 includes=[G,R,P/'inference',P/'bots/p_only/include',P/'bots/p_only/bots'])
        write(prior,dict(binary_name=bins[name].name,binary_sha256=sha(bins[name]),replay_sha256=sha(G/f'replay_{name}.cpp'),carrier_sha256=sha(G/f'c08_{name}.hpp')))
    return bins


def module_parity(selected,bins,files):
    from fusion.labels import label_match
    from fusion.schema import Layout
    import numpy as np
    sys.path.insert(0,str(R))
    from score_records import pack
    layout=Layout(native.schema());nll=read(R/'FROZEN_NATIVE.json');cases=[]
    env=os.environ.copy();env['T0004_DLL_S']=str(bins['S']);env['T0004_DLL_L']=str(bins['L'])
    for n in range(3,16):
        e=next(e for e in selected if e['teams']==n)
        assert sha(files[e['name']])==e['sha256']
        m=matchlog.load(files[e['name']]);rec=native.extract(matchlog.seat_stream(m,e['reference_seat']))
        y=label_match(m,rec,layout);valid=np.flatnonzero(y['valid'])
        ix=valid[np.linspace(0,len(valid)-1,16,dtype=int)]
        payload=pack(rec[ix],{k:v[ix] for k,v in y.items()},n)
        for size in ('S','L'):
            actual=subprocess.run([str(bins['check_'+size]),'--raw'],input=payload,capture_output=True,env=env,check=True).stdout
            direct=P/'build'/nll['binaries'][size]['name'];assert sha(direct)==nll['binaries'][size]['sha256']
            expected=subprocess.run([str(direct),'--raw'],input=payload,capture_output=True,check=True).stdout
            assert actual==expected and len(actual)==16*320*4
            cases.append(dict(teams=n,size=size,rows=16,exact_raw_values=16*320))
    write(R/'MODULE_VALIDATION.json',dict(passed=True,cases=cases))


def replay(entry,name,bins,files,identity):
    path=B/'swaps'/f'{entry["match_id"]}.{name}.json'
    if path.exists():
        old=read(path);assert old['identity']==identity and old['input_sha256']==entry['sha256'];return old
    raw=files[entry['name']];assert sha(raw)==entry['sha256']
    m=matchlog.load(raw);stream=matchlog.seat_stream(m,entry['reference_seat'])
    env=os.environ.copy();env['T0004_DLL_S']=str(bins['S']);env['T0004_DLL_L']=str(bins['L'])
    env['T0004_SWAP_MODE']=str(MODES.get(name,0))
    exe=bins['original' if name=='original' else 'swap']
    cp=subprocess.run([str(exe)],input=stream.encode(),capture_output=True,env=env,timeout=7200)
    (B/'swaps'/f'{entry["match_id"]}.{name}.stdout').write_bytes(cp.stdout)
    (B/'swaps'/f'{entry["match_id"]}.{name}.stderr').write_bytes(cp.stderr)
    assert cp.returncode==0,cp.stderr.decode(errors='replace')[-3000:]
    lines=cp.stdout.decode().splitlines();assert lines[-1]=='truncated 0'
    rows=[line.split(' ',2) for line in lines[:-1]];assert len(rows)==400
    assert [int(row[0]) for row in rows]==list(range(1,401))
    counters={}
    for line in cp.stderr.decode().splitlines():
        if line.startswith(('T0004 ','SEARCH ','PONLY ','PLY2 ','PORTFOLIO ')):
            head,body=line.split(' ',1)
            for key,value in re.findall(r'(\w+) (-?[0-9.]+(?:e-?\d+)?)',body):counters[head+'.'+key]=float(value)
    result=dict(identity=identity,input_sha256=entry['sha256'],match_id=entry['match_id'],teams=entry['teams'],mode=name,
                stdout_sha256=hashlib.sha256(cp.stdout).hexdigest(),stderr_sha256=hashlib.sha256(cp.stderr).hexdigest(),
                rows=rows,counters=counters,truncated=0)
    write(path,result);return result


def main():
    lease=read(Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json');assert lease['task']=='T0004'
    assert not (Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==SOURCE
    inputs=read(R/'FROZEN_INPUTS.json')
    selected=[e for n in range(3,16) for e in sorted((x for x in inputs['matches'] if x['teams']==n),key=lambda e:e['match_id'])[:2]]
    assert len(selected)==26
    (B/'swaps').mkdir(exist_ok=True)
    bins=build()
    frozen=dict(source_commit=SOURCE,selection=selected,samples=128,workers=4,root_registry='inference/models_pxxl_vm',
                root_registry_sha256=sha(P/'inference/models_pxxl_vm/registry.hpp'),modes={'original':'unmodified routing',**MODES},
                root_inputs={p.relative_to(P).as_posix():sha(p) for p in native.model_dependencies(P/'inference/models_pxxl_vm/registry.hpp')},
                shared_source_identity=read(R/'FROZEN_NATIVE.json')['identity'],
                model_modules={s:inputs['models'][s] for s in ('S','L')},
                generated={p.name:sha(p) for p in G.iterdir() if p.suffix in ('.cpp','.hpp')},
                recipe={p.name:sha(p) for p in [R/'swap_decisions.py',R/'swap_hook.hpp']},
                binaries={k:dict(name=p.name,sha256=sha(p)) for k,p in bins.items()},
                compiler=subprocess.check_output([native.compiler(),'--version'],text=True).splitlines()[0],
                constraints='Same original C08 depth/scoring/random keys/history/wanted-team mask; deadline-off128 samples; separate replay processes per mode. Each fresh View pins one immutable served model per team for joint/mover/action/target; caches belong to that View and mode only. No cross-size memo reuse.')
    identity=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest();frozen['identity']=identity
    target=R/'FROZEN_SWAPS.json'
    if target.exists():assert read(target)==frozen
    else:write(target,frozen)
    files={p.name:p for p in Path(os.environ['T0004_RETAINED_MATCHES']).rglob('*.json.gz')}
    if not (R/'MODULE_VALIDATION.json').exists():
        module_parity(selected,bins,files)
    else:assert read(R/'MODULE_VALIDATION.json')['passed']
    per={}
    # Prove disabled instrumentation parity across every admitted count first.
    for name in ('original','disabled', 'branch_s','far_l','near_l','all_l'):
        results={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs=[pool.submit(replay,e,name,bins,files,identity) for e in selected]
            for f in concurrent.futures.as_completed(futs):
                r=f.result();results[r['match_id']]=r;note('replay',mode=name,completed=len(results),matches=26)
        per[name]=results
        if name=='disabled':
            for e in selected:
                a,c=per['original'][e['match_id']],results[e['match_id']]
                assert a['rows']==c['rows'] and a['counters'].get('SEARCH.samples')==c['counters'].get('SEARCH.samples')
    summary=dict(identity=identity,turns_per_mode=10400,baseline_parity=True,modes={})
    for name,results in per.items():
        ref=per['original'] if name in ('original','disabled') else per['branch_s']
        changes=[]
        for e in selected:
            r=results[e['match_id']];a=ref[e['match_id']]
            assert all(x[1]==y[1] for x,y in zip(r['rows'],per['original'][e['match_id']]['rows']))
            changes.append(dict(match_id=e['match_id'],teams=e['teams'],changed=sum(x[2]!=y[2] for x,y in zip(r['rows'],a['rows'])),
                                turns=400,counters=r['counters'],stdout_sha256=r['stdout_sha256'],stderr_sha256=r['stderr_sha256']))
        summary['modes'][name]=dict(reference='original' if name in ('original','disabled') else 'branch_s',
                                   changed=sum(x['changed'] for x in changes),turns=10400,root_digest_mismatches=0,truncated=0,matches=changes)
    write(R/'DECISION_SUMMARY.json',summary);note('complete',turns=10400)


if __name__=='__main__':main()
