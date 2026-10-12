"""Freeze retained, unseen-map C08 observations and committed S/L/XXL P2 artifacts.

Run under the T0004 CPU lease. This reads raw records; no game is replayed.
The two external locations are environment variables and are never reported.
"""
import collections, gzip, hashlib, json, os, pathlib, subprocess, time
P=pathlib.Path.cwd();R=P/'bots/p_only/report/near_far'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
lease=read(pathlib.Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/cpu.json')
assert lease['task']=='T0004'
assert not (pathlib.Path(os.environ['USERPROFILE'])/'.ctf-fusion/locks/gpu.json').exists()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert source=='8a4eab4bef5bf65e971ec6712eebb89821738130'
branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
assert branch=='codex/t0004-near-far-heldout-evidence'
root=pathlib.Path(os.environ['T0004_RETAINED_MATCHES']);index_path=pathlib.Path(os.environ['T0004_TRAIN_INDEX'])
idx=read(index_path);known={m['map_seed'] for m in idx['matches']}
arm='c08_ponly_pxxl_vm';key=hashlib.sha256(arm.encode()).hexdigest()[:10]
files=sorted(p for p in root.rglob('*.json.gz') if p.name.split('.')[0].endswith('-'+key))
models={}
for size,folder in [('S','models'),('L','models_l'),('XXL','models_xxl')]:
 directory=P/'inference'/folder;manifest=read(directory/'MANIFEST.json');entries={}
 for name,v in manifest.items():
  header=directory/v['header'];npz=directory/('weights_'+name+'.npz')
  data=header.read_bytes();actual=sha(header)
  # Older export receipts hash Windows CRLF bytes; Git stores logical LF.
  restored_crlf=hashlib.sha256(data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
  assert v['header_sha256'] in (actual,restored_crlf),name
  assert sha(npz)==v['npz_sha256'],name
  assert v['identity']['spec_version']==2
  entries[name]={'committed_header_sha256':actual,'declared_export_header_sha256':v['header_sha256'],'header_transport':'identical' if actual==v['header_sha256'] else 'original CRLF export to committed LF, exactly reconstructed','npz_sha256':sha(npz),'policy_sha256':v['identity']['policy_sha256'],'weights_digest':v['identity']['weights_digest'],'joint_policy_macs':v['macs']['joint_policy_per_team']}
 models[size]={'directory':directory.relative_to(P).as_posix(),'manifest_sha256':sha(directory/'MANIFEST.json'),'registry_sha256':sha(directory/'registry.hpp'),'entries':entries}
selected=[];excluded=[];counts=collections.Counter();engine=collections.Counter();carriers=collections.Counter();bad_replies=0
for f in files:
 m=json.load(gzip.open(f,'rt',encoding='utf-8'));n=m['teams']
 if n==2:continue
 assert 3<=n<=15 and len(m['blocks'])==len(m['replies'])==400 and m['match_id'].startswith('s20261108-')
 seats=[i for i,s in enumerate(m['seats']) if s['name']=='@candidate'];assert len(seats)==1
 if m['map_seed'] in known:
  excluded.append({'match_id':m['match_id'],'reason':'map appears in original training/calibration/evaluation dataset index'});continue
 counts[n]+=1;engine[m['engine_sha256']]+=1;carriers[m['seats'][seats[0]]['sha256']]+=1
 bad_replies+=sum(x is None for row in m['replies'] for i,x in enumerate(row) if i!=seats[0])
 selected.append({'name':f.name,'sha256':sha(f),'bytes':f.stat().st_size,'match_id':m['match_id'],'teams':n,'map_seed':m['map_seed'],'flag_seed':m['flag_seed'],'reference_seat':seats[0],'reference_binary_sha256':m['seats'][seats[0]]['sha256'],'opponent_seats':[{'seat':i,'name':s['name'],'family':s['family'],'sha256':s['sha256']} for i,s in enumerate(m['seats']) if i!=seats[0]]})
assert selected and len(engine)==1 and len(carriers)==1
out={'task':'T0004','source_commit':source,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'input_arm':arm,'retained_seed':20261108,'existing_dataset_index_sha256':sha(index_path),'existing_dataset_maps':len(known),'selected_matches':len(selected),'selected_maps':len({x['map_seed'] for x in selected}),'counts':dict(counts),'excluded_overlap':excluded,'missing_opponent_replies':bad_replies,'engine_sha256':next(iter(engine)),'reference_binary_sha256':next(iter(carriers)),'models':models,'matches':selected,'scope':'Read-only inventory. No new official game seed, training, calibration fit, model export or strength claim.'}
R.mkdir(parents=True,exist_ok=True)
target=R/'FROZEN_INPUTS.json'
if target.exists():
 old=read(target);assert {k:v for k,v in old.items() if k!='created_utc'}=={k:v for k,v in out.items() if k!='created_utc'}
else:target.write_text(json.dumps(out,indent=1)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:out[k] for k in ['selected_matches','selected_maps','counts','excluded_overlap','missing_opponent_replies']}),flush=True)
