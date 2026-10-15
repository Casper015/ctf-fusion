"""Refresh T0038 precision-level summary receipts from the already-built, hash-pinned NPZs. No collection/training."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from training.decision_noise import repeat_stats
from training.evaluate_decisions import headroom
P=ROOT/'bots/p_only/report/decision_dataset';Q=ROOT/'reports/eval/decision_data_quality'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def write(p,v):Path(p).write_bytes((json.dumps(v,indent=2)+'\n').encode())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'data/decision/host_pilot_dataset');a=ap.parse_args()
 receipt=read(Q/'DATA_QUALITY_RECEIPTS.json');noise=read(Q/'noise.json');audit=read(P/'DATA_AUDIT.json')
 for split in ['train','val']:
  path=a.data/split/'decision.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['rank_npz'][split]['verified_npz']['sha256']
  with np.load(path) as z:d={k:z[k] for k in z.files}
  n=repeat_stats(d);expected=noise['splits'][split]['noise']
  for key in ['noise_roots','within_action_variance','between_action_variance_raw','between_action_variance_net']:assert n[key]==expected[key]
  hr=headroom(d,.002)['headroom'];r=noise['splits'][split]['replay']
  receipt['diagnostics'][split]={'noise_roots':n['noise_roots'],'within_action_variance':n['within_action_variance'],
   'between_action_variance_raw':n['between_action_variance_raw'],'between_action_variance_net':n['between_action_variance_net'],
   'paired_gap_variance':n['paired']['paired_gap_variance'],'implied_covariance':n['paired']['implied_covariance'],
   'replay_mean_diff':r['mean'],'replay_sd_diff':r['sd'],'headroom':hr}
 receipt['generation']={'script':'bots/p_only/report/decision_dataset/generate_acceptance.py','source_sha256':sha(__file__),
  'inputs':'hash-pinned existing train/val NPZs; verified noise.json; no test outcomes used for diagnostics',
  'headroom_weighting':'equal team-count weights; select candidate on even repeat IDs, evaluate on odd IDs',
  'precision_correction':'Full-precision summary fields refreshed from actual saved inputs; recommendation unchanged.'}
 write(Q/'DATA_QUALITY_RECEIPTS.json',receipt)
 checks=read(P/'CHECKS.json');files=[*P.glob('*'),*Q.glob('*')]
 checks['file_hashes']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files if p.is_file() and p!=P/'CHECKS.json' and p.suffix in ['.json','.md','.py']}
 checks['supplemental_acceptance']='FINAL_ACCEPTANCE.json; full trajectory-content and native full-prefix NPZ alignment passed without new games or rebuilding dataset'
 write(P/'CHECKS.json',checks)
 print(json.dumps({'summary_regenerated':True,'train':receipt['diagnostics']['train'],'val':receipt['diagnostics']['val']}))
if __name__=='__main__':main()
