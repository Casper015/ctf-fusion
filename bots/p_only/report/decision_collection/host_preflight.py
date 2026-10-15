"""Recorded-input runtime/admission and load preflight; never plays new maps."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from fusion import matchlog, runner
from fusion.protocol import parse_reply
from training import decision_collect as C, decision_opponents as O
from training.decision_branch_collect import order_of
from host_common import runtime_registry

OUT=REPO/'build/decision_host_20261014'
PROBES=REPO.parent/'T0010/data/p_only/v_data/matches'
COUNTS=(2,5,10,15)
TURNS=96


def write_gz(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8') as h: json.dump(obj,h)
    tmp.replace(path)
    C.write_json(path.with_suffix('.sha256.json'),{'sha256':O.sha256_file(path),'preflight_identity':obj['preflight_identity']})


def read_probe(path,identity):
    receipt=json.loads(path.with_suffix('.sha256.json').read_text())
    assert receipt['sha256']==O.sha256_file(path) and receipt['preflight_identity']==identity
    with gzip.open(path,'rt',encoding='utf-8') as h:result=json.load(h)
    assert result['preflight_identity']==identity
    return result


def probe(job):
    label,commands,names,mpath,repeat=job
    m=matchlog.load(mpath)
    frames=m['blocks'][:TURNS]
    bots=[runner.Bot(command,name,C.LIMITS) for command,name in zip(commands,names)]
    rows=[]
    try:
        for turn,block in enumerate(frames,1):
            inputs=[(matchlog.init_for_seat(m['init'],i if len(bots)>1 else 0) if turn==1 else '')+block for i in range(len(bots))]
            started=[bot.send(text) for bot,text in zip(bots,inputs)]
            replies=[bot.receive(start,turn) for bot,start in zip(bots,started)]
            for bot,line in zip(bots,replies):
                if line is not None:
                    if parse_reply(line,turn) is None: bot.strike(turn,'malformed probe reply')
                    else: bot.ok()
            rows.append(replies)
        reports=[bot.report() for bot in bots]
        faults=[{'seat':i,'name':bot.name,'turn':turn,'reason':reason} for i,bot in enumerate(bots) for turn,reason in bot.faults]
        latencies=[bot.latencies for bot in bots]
        first=[bot.first_seconds for bot in bots]
        return {'label':label,'repeat':repeat,'count':m['teams'],'match_path':C.rel(Path(mpath)),'match_sha256':O.sha256_file(mpath),'turns':len(frames),'programs':names,'commands':commands,'replies':rows,'faults':faults,'reports':reports,'latencies_seconds':latencies,'first_seconds':first,'limits_ms':[2000,50]}
    finally:
        for bot in bots:bot.close()


def percentile(values,q):
    x=sorted(values)
    return x[min(len(x)-1,int(q*(len(x)-1)))] if x else None


def main():
    build=json.loads((OUT/'BUILD_ALL.json').read_text())
    assert build['complete']
    programs=build['programs']
    for b in programs.values():assert O.sha256_file(REPO/b['binary'])==b['binary_sha256']
    source={n:next((PROBES/f'n{n:02d}').glob('*.json.gz')) for n in COUNTS}
    recipe={'version':'ctf-host-preflight-recipe-1','counts':list(COUNTS),'turns':TURNS,'standalone_repeats':2,'load_workers':[1,2,4,6],'load_fixture_repeats':2,'load_matches_per_worker':2,'admission_rule':'build failure or any retained standalone official-limit/protocol fault quarantines that program globally; designated/baseline failure stops. No data-dependent timing thresholds. Load concurrency: largest tested level with zero faults in its sampled compositions; retain all failed levels.','source_probes':{str(n):{'path':str(p),'sha256':O.sha256_file(p)} for n,p in source.items()},'program_binaries':{name:b['binary_sha256'] for name,b in programs.items()},'driver_sha256':O.sha256_file(__file__),'common_sha256':O.sha256_file(Path(__file__).with_name('host_common.py'))}
    identity=O.digest(recipe)
    path=OUT/'PREFLIGHT_RECIPE.json'
    if path.exists():assert json.loads(path.read_text())==recipe
    else:C.write_json(path,recipe)
    results={}
    for name,b in programs.items():
        for n in COUNTS:
            for repeat in range(2):
                label=f'standalone/{name}/n{n:02d}/r{repeat}'
                raw=OUT/'preflight_raw'/f'{label}.json.gz'
                if raw.exists():
                    result=read_probe(raw,identity)
                else:
                    result=probe((label,[[str(REPO/b['binary'])]],[name],str(source[n]),repeat))
                    result['preflight_identity']=identity
                    write_gz(raw,result)
                results[label]=result
                C.write_json(OUT/'preflight_state.json',{'phase':'standalone','completed':len(results),'expected':len(programs)*len(COUNTS)*2,'last':label})
                print('probe',label,'faults',len(result['faults']),flush=True)
    quarantine={name:[{'label':key,'faults':row['faults']} for key,row in results.items() if row['programs']==[name] and row['faults']] for name in programs}
    quarantine={name:rows for name,rows in quarantine.items() if rows}
    C.write_json(OUT/'QUARANTINE.json',quarantine)
    if any(name in quarantine for name in ('designated','baseline')):
        raise SystemExit('designated/baseline failed standalone preflight; receipts retained')
    reg=runtime_registry(Path(__import__('os').environ['AI_TEST_ROOT']),build,quarantine)
    mix=O.mixture(reg)
    units=mix['units']
    load=[]
    for workers in (1,2,4,6):
        jobs=[]
        for n in COUNTS:
            for index in range(workers*2):
                chosen=O.draw_opponents(random.Random(O.digest(['preflight-load',n,index])),mix,n-1)
                names=['designated']+[units[k]['name'] for k in chosen]
                for repeat in range(2):
                    label=f'load/w{workers}/n{n:02d}/m{index:02d}/r{repeat}'
                    raw=OUT/'preflight_raw'/f'{label}.json.gz'
                    if raw.exists():
                        row=read_probe(raw,identity)
                        load.append(row)
                    else:jobs.append((label,[[str(REPO/programs[name]['binary'])] for name in names],names,str(source[n]),repeat))
        if jobs:
            with ProcessPoolExecutor(workers) as pool:
                for row in pool.map(probe,jobs,chunksize=1):
                    row['preflight_identity']=identity
                    write_gz(OUT/'preflight_raw'/f"{row['label']}.json.gz",row)
                    load.append(row)
                    C.write_json(OUT/'preflight_state.json',{'phase':'load','workers':workers,'completed':len(load),'last':row['label']})
                    print('load',row['label'],'faults',len(row['faults']),flush=True)
    levels={w:{'jobs':sum(row['label'].startswith(f'load/w{w}/') for row in load),'faults':sum(len(row['faults']) for row in load if row['label'].startswith(f'load/w{w}/'))} for w in (1,2,4,6)}
    eligible=[w for w,d in levels.items() if d['jobs'] and d['faults']==0]
    if not eligible:raise SystemExit('no tested concurrency passed sampled load preflight; receipts retained')
    selected=max(eligible)
    comparisons=[]
    for name in programs:
        for n in COUNTS:
            a=results[f'standalone/{name}/n{n:02d}/r0'];b=results[f'standalone/{name}/n{n:02d}/r1']
            comparisons.append({'program':name,'count':n,'total':len(a['replies']),'different_orders':sum(order_of(x[0])!=order_of(y[0]) for x,y in zip(a['replies'],b['replies']))})
    group_differences=[]
    neural=[m for m in reg['members'] if m['component']=='neural' and m['status']=='enabled']
    for n in COUNTS:
        for i,a in enumerate(neural):
            for b in neural[i+1:]:
                ra=results[f"standalone/{a['name']}/n{n:02d}/r0"]['replies'];rb=results[f"standalone/{b['name']}/n{n:02d}/r0"]['replies']
                group_differences.append({'count':n,'a':a['name'],'b':b['name'],'a_group':a['group'],'b_group':b['group'],'different_orders':sum(order_of(x[0])!=order_of(y[0]) for x,y in zip(ra,rb)),'total':min(len(ra),len(rb))})
    lat=[v for row in [*results.values(),*load] for seat in row['latencies_seconds'] for v in seat]
    report={'version':'ctf-host-decision-preflight-1','identity':identity,'recipe':recipe,'quarantine':quarantine,'runtime_pool_identity':reg['pool_identity'],'load_levels':levels,'selected_workers':selected,'A_A':comparisons,'neural_response_differences':group_differences,'latency_seconds':{'n':len(lat),'p50':percentile(lat,.5),'p99':percentile(lat,.99),'max':max(lat) if lat else None},'simulation':O.simulate(reg,2000),'registry':C.strip_private(reg),'baseline_log_order_equivalence':[{'count':n,'repeat':r,'different_orders':sum(order_of(a[0])!=order_of(b[0]) for a,b in zip(results[f'standalone/designated/n{n:02d}/r{r}']['replies'],results[f'standalone/baseline/n{n:02d}/r{r}']['replies'])),'total':TURNS} for n in COUNTS for r in range(2)]}
    C.write_json(OUT/'PREFLIGHT.json',report)
    C.write_json(OUT/'preflight_state.json',{'phase':'complete','selected_workers':selected,'quarantined':list(quarantine)})
    print(json.dumps({'selected_workers':selected,'quarantined':list(quarantine),'levels':levels}),flush=True)


if __name__=='__main__':main()
