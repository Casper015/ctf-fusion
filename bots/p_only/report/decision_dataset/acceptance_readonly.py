"""Read-only supplementary acceptance of PR46. No bots, games, training, dataset rebuilds or original output writes."""
from pathlib import Path
from datetime import datetime,timezone
import collections,gzip,hashlib,json,os,sys,time
import numpy as np
R=Path.cwd();O=R/'build/t0038_acceptance';O.mkdir(parents=True,exist_ok=True)
D=Path(os.environ['T38_SOURCE_REPO']);B=D/'build/decision_host_20261014';RAW=D/'data/decision/host_20261014';DS=D/'data/decision/host_pilot_dataset'
sys.path.insert(0,str(R))
from fusion import decision_features as F
from fusion.protocol import parse_block
from fusion.schema import layout
from training.decision_dataset import root_record,split_of_map
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):return json.load(gzip.open(p,'rt',encoding='utf-8'))
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(s.partition('#')[0].split()) if s is not None else None
def u(score,seat):return sum(1 if score[seat]>x else .5 if score[seat]==x else 0 for i,x in enumerate(score) if i!=seat)/(len(score)-1)
def save(name,x):
 tmp=O/(name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');tmp.replace(O/name)
def main():
 lease=read(Path.home()/'.ctf-fusion/locks/cpu.json');assert lease['task']=='T0038' and 't0038_acceptance_run.ps1' in lease['command']
 assert not (Path.home()/'.ctf-fusion/locks/gpu.json').exists()
 frozen=read(B/'FROZEN.json');plan=read(RAW/'PLAN.json');schedule=read(B/'SCHEDULE.json')
 assert plan['identity']=='8a4a0417ae7d765cf54a2b9b705b0bd79a6e3bd36b1ba18c66876c1e223cadb5'
 assert frozen['identity']==plan['frozen_identity'];assert schedule['seed']==20261701
 binaries={}
 for name,entry in {'@candidate':frozen['designated'],**frozen['opponents']}.items():
  p=D/entry['binary'];assert h(p)==entry['binary_sha256'],f'actual binary drift {name}';binaries[name]=entry['binary_sha256']
 enginepath=Path(os.environ['AI_TEST_ROOT'])/'ctf-arena/Player7/workspace/engine.py';assert h(enginepath)==frozen['engine']['sha256']
 matches={};mhash={};matchturns=0
 for p in sorted((RAW/'matches').rglob('*.json.gz')):
  m=load(p);mid=m['match_id'];assert mid not in matches;matches[mid]=m;mhash[mid]=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  assert len(m['blocks'])==len(m['replies'])==400 and m['limits_ms']==[2000,50]
  assert parse_block(m['final_block']).score==m['result']['score']
  assert all(len(row)==m['teams'] and all(x is not None for x in row) for row in m['replies'])
  assert not m['faults'];matchturns+=400
 fixtures=schedule['fixtures'];assert len(fixtures)==336==len(matches)
 assert {f['fixture_id'] for f in fixtures}==set(matches)
 for f in fixtures:
  m=matches[f['fixture_id']];assert f['teams']==m['teams'] and f['map_seed']==m['map_seed'] and f['flag_seed']==m['flag_seed']
 npzs={s:np.load(DS/s/'decision.npz') for s in ['train','val','test']};index=read(DS/'index.json');assert index['authoritative']
 audit=read(R/'bots/p_only/report/decision_dataset/DATA_AUDIT.json');npzhashes={s:h(DS/s/'decision.npz') for s in npzs}
 assert all(npzhashes[s]==audit['rank_npz'][s]['verified_npz']['sha256'] for s in npzs)
 cursors=collections.Counter();candidates=collections.Counter();rootsseen=set();trajectories=0;blocks=0;prefix_norm=0;rawreceipts=[]
 L=layout()
 for path in sorted((RAW/'branches').rglob('*.json.gz')):
  row=load(path);mid=row['match_id'];assert mid not in rootsseen;rootsseen.add(mid);m=matches[mid];seat=row['seat'];turn=row['turn'];n=row['teams'];k=len(row['candidates'])
  assert row['plan_identity']==plan['identity'] and row['map_seed']==m['map_seed'] and row['init']==m['init']
  assert row['input_binding']['match_sha256']==mhash[mid]
  assert row['input_binding']['root_block_sha256']==hashlib.sha256(json.dumps(m['blocks'][turn-1],sort_keys=True,separators=(',',':')).encode()).hexdigest()
  assert row['carrier_scores']['orders']==row['candidates'] and row['carrier_scores']['scores']==row['input_binding']['scores']
  coords={(b['candidate'],b['repeat']) for b in row['branches']};assert len(row['branches'])==k*4 and coords=={(c,j) for c in range(k) for j in range(4)}
  s=split_of_map(m['map_seed']);npz=npzs[s];ri=cursors[s];ci=int(npz['root_first'][ri]);assert str(npz['root_match'][ri])==mid
  assert int(npz['root_map'][ri])==m['map_seed'] and int(npz['root_seat'][ri])==seat and int(npz['root_turn'][ri])==turn and int(npz['root_count'][ri])==k
  rec=root_record(m,seat,turn);assert np.array_equal(rec,npz['root_rec'][ri]),f'root native prefix record mismatch {mid}'
  units=parse_block(m['blocks'][turn-1]).units;size=int(m['init'].splitlines()[0].split()[3])
  for c,line in enumerate(row['candidates']):
   cc=F.canonicalize(rec,L,line,turn,seat,units,size);assert str(npz['cand_key'][ci+c])==cc['key']
   assert np.array_equal(npz['cand_move'][ci+c],cc['move']) and np.array_equal(npz['cand_act'][ci+c],cc['act'])
   assert np.array_equal(npz['cand_slot'][ci+c],cc['slot']) and np.array_equal(npz['cand_target'][ci+c],cc['target'])
   assert int(npz['cand_flags'][ci+c])==cc['flags'] and bool(npz['cand_support'][ci+c])==F.supported(cc['flags'])
  for branch in row['branches']:
   c=branch['candidate'];j=branch['repeat'];p=D/branch['trajectory_path'];assert h(p)==branch['trajectory_sha256'];x=load(p)
   for key,value in x.items():
    if key!='trajectory':assert branch[key]==value,f'raw/branch metadata mismatch {mid} {c} {j} {key}'
   tr=x['trajectory'];assert tr['init']==m['init'] and len(tr['blocks'])==len(tr['played_replies'])==len(tr['computed_replies'])==400
   assert tr['blocks'][:turn]==m['blocks'][:turn] and tr['played_replies'][:turn-1]==m['replies'][:turn-1]
   assert all(int(block.splitlines()[0].split()[1])==t for t,block in enumerate(tr['blocks'],1))
   expected=list(m['replies'][turn-1]);expected[seat]=row['candidates'][c];assert tr['played_replies'][turn-1]==expected
   assert tr['played_replies'][turn:]==tr['computed_replies'][turn:]
   assert x['root_block']==tr['blocks'][turn-1] and (turn==400 or x['next_block']==tr['blocks'][turn])
   assert parse_block(tr['final_block']).score==x['score']==branch['score']
   assert abs(u(x['score'],seat)-x['outcome'])<1e-12 and abs(x['outcome']-float(npz['cand_outcome'][ci+c,j]))<1e-12
   assert np.array_equal(npz['cand_score'][ci+c,j,:n],x['score']) and np.all(npz['cand_score'][ci+c,j,n:]==-1)
   mismatch=sum(norm(a)!=norm(b) for aa,bb in zip(tr['computed_replies'][:turn-1],m['replies'][:turn-1]) for a,b in zip(aa,bb))
   assert mismatch==x['prefix_mismatch'];prefix_norm+=mismatch
   assert not x['partial'] and not x['all_phase_faults'] and not x['faults'] and not x['discarded_root_computed_faults'] and not x['prefix_faults']
   trajectories+=1;blocks+=400;rawreceipts.append({'path':str(p.relative_to(D)).replace('\\','/'),'sha256':branch['trajectory_sha256']})
  cursors[s]+=1;candidates[s]+=k
  save('state.json',{'roots':len(rootsseen),'trajectories':trajectories,'phase':'read_only_acceptance'})
 assert len(rootsseen)==336 and trajectories==5268 and rootsseen==set(matches)
 assert all(cursors[s]==len(npzs[s]['root_match']) and candidates[s]==len(npzs[s]['cand_root']) for s in npzs)
 checks=read(R/'bots/p_only/report/decision_dataset/CHECKS.json')
 for p,expected in checks['file_hashes'].items():assert hashlib.sha256((R/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected
 result={'success':True,'utc':datetime.now(timezone.utc).isoformat(),'reviewed_head':'2d91c1825f94011a0b5c1a4da480294e8246a549','plan_identity':plan['identity'],'actual_frozen_binaries_verified':len(binaries),'engine_sha256':frozen['engine']['sha256'],
 'matches':336,'match_turns':matchturns,'roots':336,'continuations':trajectories,'trajectory_turn_blocks':blocks,'raw_branch_metadata_score_repeat_input_linkage':True,'played_prefix_and_opponent_root_and_live_reply_protocol':True,'npz_scores_and_repeat_id_exact':True,'npz_native_full_prefix_root_records_and_canonical_actions_exact':True,'all_phase_faults':0,'prefix_order_mismatches_retained':prefix_norm,'npz_sha256':npzhashes,'published_checksum_receipt_exact':True,
 'trajectory_receipts_digest':hashlib.sha256(json.dumps(rawreceipts,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'scope':'supplementary raw/dataset/source acceptance; no bot games or dataset rebuild. No training/deployment/strength admission.'}
 save('ACCEPTANCE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
