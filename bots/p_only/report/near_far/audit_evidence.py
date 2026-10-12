"""Independent receipt, denominator, interval and complete-order audit.

Run after both measurement stages exit, under a new T0004 CPU lease.
Reads preserved arrays/replies; never reruns inference or recorded decisions.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

P=Path.cwd();sys.path.insert(0,str(P))
from fusion.protocol import parse_reply
R=P/'bots/p_only/report/near_far';B=P/'build/t0004';INF=1<<20
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)


def main():
    assert read(Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json')['task']=='T0004'
    assert not (Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
    inputs=read(R/'FROZEN_INPUTS.json');native=read(R/'FROZEN_NATIVE.json');receipts=read(R/'SCORE_RECEIPTS.json')
    nll=read(R/'NLL_SUMMARY.json');swaps=read(R/'FROZEN_SWAPS.json');decision=read(R/'DECISION_SUMMARY.json')
    assert native['identity']==receipts['identity']==nll['identity']==swaps['shared_source_identity']
    assert swaps['identity']==decision['identity']
    for name,h in native['sources'].items():assert sha(P/name)==h,name
    for name,h in swaps['root_inputs'].items():assert sha(P/name)==h,name
    for v in [*native['binaries'].values(),native['geometry'],native['extractor'],*swaps['binaries'].values()]:
        target=B/v['name'] if (B/v['name']).exists() else P/'build'/v['name']
        assert sha(target)==v['sha256']
    for name,h in swaps['generated'].items():assert sha(B/'swapsrc'/name)==h
    for name,h in swaps['recipe'].items():assert sha(R/name)==h
    assert read(R/'NATIVE_VALIDATION.json')['passed'] and len(read(R/'NATIVE_VALIDATION.json')['cases'])==39
    assert read(R/'MODULE_VALIDATION.json')['passed'] and len(read(R/'MODULE_VALIDATION.json')['cases'])==26
    points={k:[] for k in ('all_primary','reference_0_3','reference_4_6','reference_7_plus')}
    concentration={1:[],2:[]}
    raw={p.name:p for p in Path(os.environ['T0004_RETAINED_MATCHES']).rglob('*.json.gz')}
    assert len(receipts['matches'])==len(inputs['matches'])==260
    for e in inputs['matches']:assert sha(raw[e['name']])==e['sha256']
    for e in receipts['matches']:
        target=B/'scores'/(e['match_id']+'.npz');assert sha(target)==e['npz_sha256']
        with np.load(target) as data:
            g,m,primary=data['geometry'],data['metrics'],data['primary']
            entry=next(x for x in inputs['matches'] if x['match_id']==e['match_id'])
            assert len(g)==400*(e['teams']-1) and (g[:,1]!=entry['reference_seat']).all()
            close(primary,(g[:,8]>0)&(g[:,9]>0)&(g[:,2]<INF))
            assert np.isfinite(m[:,:,:6]).all() and (m[:,:,0]>=0).all()
            close(np.isnan(m[:,0,6]),((data['y_act']==3)&(data['y_tgt']<0)).any(1))
            for size in range(3):
                close(m[:,size,2],(data['y_act']>=0).sum(1));close(m[:,size,4],(data['y_tgt']>=0).sum(1))
                ok=np.isfinite(m[:,size,6]);close(m[ok,size,6],m[ok,size,0]+m[ok,size,1]+m[ok,size,3])
            for group,mask in [('all_primary',primary),('reference_0_3',primary&(g[:,2]<=3)),
                               ('reference_4_6',primary&(g[:,2]>=4)&(g[:,2]<=6)),('reference_7_plus',primary&(g[:,2]>=7)&(g[:,2]<INF))]:
                points[group].append((e['teams'],int(mask.sum()),m[mask,:,0].sum(0)))
            near_mask=primary&(g[:,2]<=3);far_mask=primary&(g[:,2]>=7)&(g[:,2]<INF)
            if near_mask.any() and far_mask.any():
                for size in (1,2):
                    delta=m[:,size,0]-m[:,0,0]
                    value=np.average(delta[near_mask])-np.average(delta[far_mask])
                    concentration[size].append((e['teams'],e['map_seed'],float(value),int(near_mask.sum()),int(far_mask.sum())))
    # Recompute paired joint means and map-cluster variance via weighted covariance.
    for group,rows in points.items():
        assert sum(x[1] for x in rows)==nll['groups'][group]['rows']
        for size,name in [(1,'L-S'),(2,'XXL-S')]:
            means,variances=[],[]
            for n in range(3,16):
                pr=[x for x in rows if x[0]==n and x[1]>0]
                weight=np.array([x[1] for x in pr],dtype=float);delta=np.array([(x[2][size]-x[2][0])/x[1] for x in pr])
                mean=np.average(delta,weights=weight)
                influence=(delta-mean)*weight/weight.sum()
                variance=len(pr)/(len(pr)-1)*np.dot(influence,influence)
                means.append(mean);variances.append(variance)
                stored=nll['groups'][group]['metrics']['joint'][name]['counts'][str(n)]
                close(mean,stored['value']);close([mean-1.96*np.sqrt(variance),mean+1.96*np.sqrt(variance)],stored['ci95'])
            mean=np.mean(means);half=1.96*np.sqrt(sum(variances))/13
            stored=nll['groups'][group]['metrics']['joint'][name]['overall']
            close(mean,stored['value']);close([mean-half,mean+half],stored['ci95'])
    for size,name in [(1,'L-S'),(2,'XXL-S')]:
        rows=concentration[size];stored=nll['concentration'][name]
        assert len(rows)==stored['maps']==260
        for n,map_seed,value,near_rows,far_rows in rows:
            original=next(x for x in stored['points'] if x['map_seed']==map_seed)
            assert original['teams']==n and original['near_rows']==near_rows and original['far_rows']==far_rows
            close(value,original['near_minus_far'])
        means=[];variance=[]
        for n in range(3,16):
            values=np.array([x[2] for x in rows if x[0]==n]);assert len(values)==20
            center=values.sum()/20
            var=np.dot(values-center,values-center)/(20*19)
            close(center,stored['counts'][str(n)]['value'])
            close([center-1.96*np.sqrt(var),center+1.96*np.sqrt(var)],stored['counts'][str(n)]['ci95'])
            means.append(center);variance.append(var)
        center=sum(means)/13;half=1.96*np.sqrt(sum(variance))/13
        close(center,stored['overall']['value']);close([center-half,center+half],stored['overall']['ci95'])
    modes={}
    for name,summary in decision['modes'].items():
        changes=move=act=target=invalid=0;counter_totals={};by_count={}
        for e in swaps['selection']:
            current=read(B/'swaps'/f'{e["match_id"]}.{name}.json')
            baseline_name=summary['reference'];base=read(B/'swaps'/f'{e["match_id"]}.{baseline_name}.json')
            original=read(B/'swaps'/f'{e["match_id"]}.original.json')
            assert current['identity']==swaps['identity'] and current['truncated']==0
            for stream in ('stdout','stderr'):
                assert sha(B/'swaps'/f'{e["match_id"]}.{name}.{stream}')==current[stream+'_sha256']
            cc=0
            for row,ref,root in zip(current['rows'],base['rows'],original['rows']):
                assert row[0]==ref[0]==root[0] and row[1]==root[1]
                t=int(row[0]);a,b=parse_reply(row[2],t),parse_reply(ref[2],t)
                if a is None or b is None:invalid+=1;continue
                a=[(x.move,x.act,x.target) for x in a];b=[(x.move,x.act,x.target) for x in b]
                changed=a!=b;assert changed==(row[2]!=ref[2])
                changes+=changed;cc+=changed
                move+=any(x[0]!=y[0] for x,y in zip(a,b));act+=any(x[1]!=y[1] for x,y in zip(a,b));target+=any(x[2]!=y[2] for x,y in zip(a,b))
            assert len(current['rows'])==400
            by_count.setdefault(str(e['teams']),{'changed':0,'turns':0});by_count[str(e['teams'])]['changed']+=cc;by_count[str(e['teams'])]['turns']+=400
            for k,v in current['counters'].items():
                if k not in ('T0004.mode','SEARCH.fixed'):counter_totals[k]=counter_totals.get(k,0)+v
        assert invalid==0 and changes==summary['changed'] and summary['turns']==10400
        if name=='disabled':assert changes==0
        modes[name]=dict(changed=changes,movement_changed=move,action_changed=act,target_changed=target,
                         valid_complete_orders=10400,invalid_reply_exclusions=invalid,by_count=by_count,counters=counter_totals)
    out=dict(passed=True,source_and_binary_hashes=True,raw_matches=260,score_rows=832000,primary_rows=789836,
             joint_means_and_cluster_intervals_independently_recomputed=True,orders=modes,
             paired_map_near_far_concentration_independently_recomputed=True,
             limitation='Branch eligible prediction-call counters are observed; turn-level branch eligibility was not traced.10400 is the complete recorded-decision denominator per mode, including states with no eligible branch call. No strength or latency inference.')
    (R/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=1)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in out.items() if k!='orders'}),flush=True)


if __name__=='__main__':main()
