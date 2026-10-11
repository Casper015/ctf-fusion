"""Read-only raw audit; output only to the T0019 report lane. No games replayed."""
import collections as C
import gzip, hashlib, importlib.util, json, math, os, pathlib, re, statistics, sys, time
P=pathlib.Path.cwd(); R=P/'bots/p_only/report/search_portfolio'
sys.path.insert(0,str(P))
from fusion.protocol import parse_reply, MOVES, ATTACK, PICKUP, DROP
def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def uh(*args): return int.from_bytes(hashlib.sha256('/'.join(map(str,args)).encode()).digest()[:8],'little')
def utility(scores,seat): return sum((s<scores[seat])+0.5*(s==scores[seat]) for i,s in enumerate(scores) if i!=seat)/(len(scores)-1)
def close(a,b): assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def canon(line,t):
 q=parse_reply(line,t) if line is not None else None
 if q is None: return None
 return ' '.join([str(t)]+[f'{MOVES[x.move]} '+({ATTACK:str(x.target),PICKUP:'P',DROP:'X'}.get(x.act,'-')) for x in q])
def cluster(values,maps):
 n=len(values); mean=sum(values)/n; residual=C.defaultdict(float)
 for v,m in zip(values,maps): residual[m]+=(v-mean)/n
 k=len(residual); se=math.sqrt(k/(k-1)*sum(x*x for x in residual.values())) if k>1 else float('nan')
 return {'mean':mean,'lo':mean-1.96*se,'hi':mean+1.96*se,'clusters':k}
start=time.monotonic()
lease=read(pathlib.Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json')
assert lease['task']=='T0019'
assert not (pathlib.Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
spec=importlib.util.spec_from_file_location('frozen_controller',P.parent.parent/'t0019_pipeline.py')
ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
assert ctl.source_identity()==read(R/'FROZEN_INPUTS.json')['measurement']
assert ctl.diagnostic_inputs()==read(R/'FROZEN_INPUTS.json')['diagnostic_inputs']
frozen=read(R/'FROZEN_SCREEN.json');current=ctl.frozen_screen()
assert {k:v for k,v in current.items() if k!='manifest_raw_sha256'}=={k:v for k,v in frozen.items() if k!='manifest_raw_sha256'}
bundle={}
for arm,v in frozen['manifest_stable']['variants'].items():
 raw=(P/v['bundle']).read_bytes();logical=raw.replace(b'\r\n',b'\n');assert hashlib.sha256(logical).hexdigest()==v['bundle_sha256']; assert len(logical)==v['bundle_bytes']
 bundle[arm]={'raw_sha256':hashlib.sha256(raw).hexdigest(),'logical_lf_sha256':hashlib.sha256(logical).hexdigest()}
screen=read(R/'screen.json');arms=list(screen['arms']);keys={hashlib.sha256(a.encode()).hexdigest()[:10]:a for a in arms}
sched=read(P/'data/p_only/search/schedule.json');fx={f['fixture_id']:f for f in sched['fixtures']}
rows=[json.loads(x) for x in (P/'data/p_only/search/results.jsonl').read_text().splitlines() if x.strip()]
index={(r['candidate'],r['fixture_id']):r for r in rows};assert len(rows)==len(index)==2240
assert set(index)=={(a,f) for a in arms for f in fx}
raws={}; receipts=[]; faults=[]; all_faults=[]; counter_totals=C.defaultdict(C.Counter); settings=C.defaultdict(dict); line_counts=C.defaultdict(C.Counter)
for num,path in enumerate(sorted((P/'data/p_only/search/matches').rglob('*.json.gz')),1):
 m=json.load(gzip.open(path,'rt',encoding='utf-8'));mid=m['match_id'];assert mid not in raws;raws[mid]=m
 arm=keys[mid.rsplit('-',1)[1]];fid=mid.rsplit('-',1)[0];row=index[arm,fid];f=fx[fid];seat=f['seat']
 assert m['teams']==f['teams'] and m['map_seed']==f['map_seed'] and m['flag_seed']==f['flag_seed']
 assert m['seats'][seat]['name']=='@candidate' and m['seats'][seat]['sha256']==row['source_sha256']==screen['arms'][arm]['binary_sha256']
 assert [s['name'] for i,s in enumerate(m['seats']) if i!=seat]==f['opponents']
 assert row['identity']==frozen['identities'][arm]['digest'] and row['schedule']==screen['schedule']
 assert m['limits_ms']==[2000,50] and m['engine_sha256']==row['engine_sha256'] and len(m['blocks'])==400
 assert m['result']['score']==row['score'];close(utility(row['score'],seat),row['utility'])
 bot=m['bots'][seat]
 for k in ['faults','dead','late','first_ms','p99_ms','max_ms']: assert bot[k]==row[k],(mid,k)
 for ev in m['faults']:
  all_faults.append({'match_id':mid,'arm':arm,'event':ev})
  if ev[1]==seat: faults.append({'match_id':mid,'arm':arm,'event':ev})
 seen=C.Counter()
 for line in bot['stderr_tail']:
  typ=line.split(' ',1)[0];
  if typ not in ['PONLY','SEARCH','PLY2','PORTFOLIO','VALUE','VBUDGET']:continue
  seen[typ]+=1;line_counts[arm][typ]+=1
  for name,val in re.findall(r'(\w+) (-?[0-9.]+(?:e-?\d+)?)',line):
   key=typ.lower()+'.'+name;v=float(val)
   if name in ['fixed','mode','cache','k','moves','max_calls','deadline_ms'] or name.startswith('coef_'):
    if key in settings[arm]:assert settings[arm][key]==v,(arm,key)
    settings[arm][key]=v
   else:counter_totals[arm][key]+=v
 assert seen['PONLY']==1 and max(seen.values())==1,(mid,seen)
 receipts.append([path.relative_to(P).as_posix(),sha(path)])
 if num%200==0:print('raw matches audited',num,flush=True)
assert len(raws)==2240
paired={}
bands={'2':[2],'3-5':[3,4,5],'6-10':list(range(6,11)),'11-15':list(range(11,16))}
for name,g in {**screen['comparisons'],**screen['rerun_noise']}.items():
 if name.startswith('C08 interaction'):terms=[(1,'c08_search_d'),(-1,'c08_search_c'),(-1,'c08_search_b'),(1,'c08_search_a')]
 else:
  a,b=name.split(' - ');terms=[(1,a),(-1,b)]
 per=C.defaultdict(list)
 for fid,f in fx.items():per[f['teams']].append(sum(w*index[a,fid]['utility'] for w,a in terms))
 def estimate(counts):
  mean=statistics.mean(statistics.mean(per[n]) for n in counts);se=math.sqrt(sum(statistics.variance(per[n])/len(per[n]) for n in counts))/len(counts)
  return {'mean':round(mean,5),'ci95':[round(mean-1.96*se,5),round(mean+1.96*se,5)],'se':round(se,5)}
 check={'overall':estimate(list(range(2,16))),'bands':{b:estimate(ns) for b,ns in bands.items()}}
 for k,v in check['overall'].items():assert v==g['overall'][k],(name,k)
 for b,z in check['bands'].items():
  for k,v in z.items():assert v==g['bands'][b][k],(name,b,k)
 for n,vs in per.items():assert round(statistics.mean(vs),5)==g['per_count'][str(n)]
 paired[name]=check
branch_audit={}
for tag,arm,other in [('v12','v12_search_a','v12_search_b'),('c08','c08_search_a','c08_search_d')]:
 paths=sorted((P/('data/branches/search_'+tag)).rglob('*.json.gz'));summary=read(R/('branch_summary_'+tag+'.json'));regret=read(R/('regret_'+tag+'.json'))
 selected=sorted([mid for mid in raws if mid.endswith('-'+hashlib.sha256(arm.encode()).hexdigest()[:10])],key=lambda mid:uh('pick',mid+'.json.gz'))[:200]
 expected={(mid,min(range(6,max(len(raws[mid]['blocks'])-5,7)),key=lambda t:uh(mid,'root',t))) for mid in selected}
 actual=set();drops=C.Counter();statuses=C.Counter();sources=C.Counter();pairs=C.Counter();metrics=C.defaultdict(list);maps=[];teams=[];different=[];mismatches=[];prefixfault=livefault=branchcount=0
 for path in paths:
  d=json.load(gzip.open(path,'rt',encoding='utf-8'));key=(d['match_id'],d['turn']);assert key not in actual;actual.add(key);m=raws[d['match_id']];assert d['arm']==arm
  receipts.append([path.relative_to(P).as_posix(),sha(path)])
  if 'dropped' in d:drops[d['dropped']]+=1;continue
  assert d['map_seed']==m['map_seed'] and d['seat']==fx[d['match_id'].rsplit('-',1)[0]]['seat'] and d['carrier_sha256']==screen['arms'][arm]['binary_sha256']
  orders=[canon(x,d['turn']) for x in d['candidates']];assert None not in orders and len(orders)==len(set(orders))==len(d['sources']);sources.update(d['sources'])
  assert orders[0]==canon(m['replies'][d['turn']-1][d['seat']],d['turn'])
  close(d['root_outcome'],utility(m['result']['score'],d['seat']))
  seen=set();ys=C.defaultdict(list)
  for b in d['branches']:
   ck=(b['candidate'],b['repeat']);assert ck not in seen;seen.add(ck);assert b['line']==d['candidates'][ck[0]] and b['source']==d['sources'][ck[0]]
   close(utility(b['score'],d['seat']),b['outcome']);ys[ck[0]].append(b['outcome']);assert b['root_block']==m['blocks'][d['turn']-1]
   mismatches.append(b['prefix_mismatch']);prefixfault+=b['prefix_faults'];livefault+=len(b['faults']);branchcount+=1
  assert seen=={(i,j) for i in range(len(orders)) for j in range(2)}
  means={i:statistics.mean(v) for i,v in ys.items()};reply=canon(d['compare_reply'],d['turn']);assert reply in orders
  chosen=orders.index(reply);status='identical' if chosen==0 else 'different';assert status==d['compare_status'] and chosen==d['compare_candidate'];statuses[status]+=1
  for i in means:
   for j in means:
    if i<j:pairs['resolved' if abs(means[i]-means[j])>0.5/(d['teams']-1) else 'unresolved']+=1
  best=max(means.values());stats={'difference':means[chosen]-means[0],'regret_incumbent':best-means[0],'regret_compare':best-means[chosen],'rank_incumbent':1+sum(v>means[0] for v in means.values()),'rank_compare':1+sum(v>means[chosen] for v in means.values())}
  stats['regret_difference']=stats['regret_compare']-stats['regret_incumbent']
  for k,v in stats.items():metrics[k].append(v)
  maps.append(d['map_seed']);teams.append(d['teams']);different.append(chosen!=0)
 assert actual==expected and len(paths)==200 and dict(drops)==summary['dropped'];assert dict(statuses)==summary['compare_status']==regret['root_statuses']
 assert dict(sources)==summary['candidates_by_source'] and dict(pairs)==summary['pairs'];assert branchcount==summary['branches'];assert prefixfault==summary['prefix_faults']==0 and livefault==summary['live_faults']==0
 assert {'sum':sum(mismatches),'max':max(mismatches),'branches_with_any':sum(x>0 for x in mismatches)}==summary['prefix_mismatch']
 checks={}
 for k in ['difference','regret_incumbent','regret_compare','regret_difference']:
  z=cluster(metrics[k],maps)
  for sub,v in z.items():close(v,regret[k][sub])
  checks[k]=z
 z=cluster([v for v,d in zip(metrics['difference'],different) if d],[m for m,d in zip(maps,different) if d])
 for k,v in z.items():close(v,regret['difference_where_differs'][k])
 for k in ['rank_incumbent','rank_compare']:close(statistics.mean(metrics[k]),regret[k+'_mean'])
 for t in set(teams):
  by=cluster([v for v,n in zip(metrics['difference'],teams) if n==t],[m for m,n in zip(maps,teams) if n==t])
  for k,v in by.items():close(v,regret['by_teams'][str(t)][k])
 branch_audit[tag]={'selected_roots':200,'dropped':dict(drops),'valid_roots':len(maps),'statuses':dict(statuses),'continuations':branchcount,'pairs':dict(pairs),'prefix_mismatch':summary['prefix_mismatch'],'prefix_faults':prefixfault,'live_faults':livefault,'recomputed':checks,'where_differs':z}
 print('branch audit passed',tag,flush=True)
diag=read(R/'diag_fixed_work.json');assert diag['games']==60 and len(diag['configs'])==7
assert all(x['turns']==24000 and x['root_digest_mismatches']==0 and x['truncated']==0 for x in diag['configs'].values())
out={'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_commit':ctl.SOURCE,'raw_matches':2240,'paired_intervals':paired,'candidate_fault_events':faults,'all_seat_fault_events':all_faults,'counter_totals':{a:dict(v) for a,v in counter_totals.items()},'counter_settings':dict(settings),'counter_line_counts':{a:dict(v) for a,v in line_counts.items()},'branches':branch_audit,'bundle_hashes':bundle,'raw_receipts_digest':hashlib.sha256(json.dumps(sorted(receipts)).encode()).hexdigest(),'wall_seconds':round(time.monotonic()-start,3),'limits':'Latency summaries cover successful replies; timeout games remain in every screen denominator. Prefix reply mismatches are not reproduced state mismatches. No completed game or continuation was rerun.'}
(R/'INDEPENDENT_AUDIT.json').write_text(json.dumps(out,indent=1)+'\n',encoding='utf-8',newline='\n')
print('independent audit passed',out['wall_seconds'],flush=True)
