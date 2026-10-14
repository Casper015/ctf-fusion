"""Complete only missing probe coverage; preserve all original successful bytes."""
from __future__ import annotations
import json
import host_preflight as P
from training import decision_collect as C,decision_opponents as O
from training.decision_branch_collect import order_of


def main():
    out=P.OUT
    original=json.loads((out/'PREFLIGHT.json').read_text())
    build=json.loads((out/'BUILD_ALL.json').read_text())
    recipe=original['recipe']
    assert O.sha256_file(P.__file__)==recipe['driver_sha256']
    rows={};missing=[]
    for name,b in build['programs'].items():
        assert O.sha256_file(P.REPO/b['binary'])==b['binary_sha256']
        for n in P.COUNTS:
            for repeat in range(2):
                label=f'standalone/{name}/n{n:02d}/r{repeat}'
                path=out/'preflight_raw'/f'{label}.json.gz'
                row=P.read_probe(path,original['identity']) if path.exists() else None
                command=[[str(P.REPO/b['binary'])]]
                if row is not None and row['programs']==[name] and row['commands']==command:
                    rows[label]=row
                else:missing.append({'label':label,'name':name,'count':n,'repeat':repeat,'commands':command,'reason':'Windows case-insensitive path collision: exact program/command mismatch' if row else 'not recorded','retained_path':C.rel(path) if path.exists() else None})
    repair_recipe={'version':'ctf-host-preflight-completion-1','original_preflight_identity':original['identity'],'original_report_sha256':O.sha256_file(out/'PREFLIGHT.json'),'driver_sha256':O.sha256_file(__file__),'missing':missing,'rule':'reuse only SHA-verified original rows with exact matching names/commands; play each missing job once at one worker and official 2000/50ms; preserve original bytes'}
    identity=O.digest(repair_recipe)
    rp=out/'PREFLIGHT_COMPLETION_RECIPE.json'
    if rp.exists():assert json.loads(rp.read_text())==repair_recipe
    else:C.write_json(rp,repair_recipe)
    for entry in missing:
        label=entry['label'];name=entry['name'];n=entry['count'];repeat=entry['repeat']
        path=out/'preflight_completion_raw'/O.digest(name)/f'n{n:02d}-r{repeat}.json.gz'
        if path.exists():row=P.read_probe(path,identity)
        else:
            row=P.probe((label,entry['commands'],[name],recipe['source_probes'][str(n)]['path'],repeat))
            row['preflight_identity']=identity
            P.write_gz(path,row)
        assert row['programs']==[name] and row['commands']==entry['commands']
        rows[label]=row
        print('completed missing probe',label,'faults',len(row['faults']),flush=True)
    assert len(rows)==len(build['programs'])*len(P.COUNTS)*2
    quarantine={name:[{'label':key,'faults':row['faults']} for key,row in rows.items() if row['programs']==[name] and row['faults']] for name in build['programs']}
    quarantine={name:items for name,items in quarantine.items() if items}
    assert quarantine==json.loads((out/'QUARANTINE.json').read_text()),'completion changed admission; a separately declared load stage is required'
    load=[]
    for path in (out/'preflight_raw/load').rglob('*.json.gz'):
        row=P.read_probe(path,original['identity'])
        for name,command in zip(row['programs'],row['commands']):assert command==[str(P.REPO/build['programs'][name]['binary'])]
        load.append(row)
    levels={w:{'jobs':sum(row['label'].startswith(f'load/w{w}/') for row in load),'faults':sum(len(row['faults']) for row in load if row['label'].startswith(f'load/w{w}/'))} for w in (1,2,4,6)}
    for w in levels:assert levels[w]['jobs']==w*2*len(P.COUNTS)*2,'load coverage incomplete'
    eligible=[w for w,item in levels.items() if item['faults']==0]
    assert eligible,'no tested concurrency passes'
    latency=[v for row in [*rows.values(),*load] for seat in row['latencies_seconds'] for v in seat]
    repaired=dict(original)
    repaired.update(identity=O.digest([original['identity'],identity]),original_preflight_identity=original['identity'],completion_identity=identity,completion_recipe=repair_recipe,physical_standalone_probes=len(rows),physical_load_probes=len(load),load_levels=levels,selected_workers=max(eligible),latency_seconds={'n':len(latency),'p50':P.percentile(latency,.5),'p99':P.percentile(latency,.99),'max':max(latency) if latency else None})
    repaired['A_A']=[{'program':name,'count':n,'total':len(rows[f'standalone/{name}/n{n:02d}/r0']['replies']),'different_orders':sum(order_of(a[0])!=order_of(b[0]) for a,b in zip(rows[f'standalone/{name}/n{n:02d}/r0']['replies'],rows[f'standalone/{name}/n{n:02d}/r1']['replies']))} for name in build['programs'] for n in P.COUNTS]
    C.write_json(out/'PREFLIGHT_VERIFIED.json',repaired)
    print(json.dumps({'verified':True,'missing_probes_completed':len(missing),'standalone':len(rows),'load':len(load),'selected_workers':max(eligible),'quarantine':list(quarantine)}),flush=True)


if __name__=='__main__':main()
