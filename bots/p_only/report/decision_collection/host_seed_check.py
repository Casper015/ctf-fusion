"""Read-only master-seed audit of current board/ledger and retained host metadata."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from training import decision_collect as C,decision_opponents as O


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    candidate=args.seed
    assert candidate not in set(O.consumed_seeds())|{20261210,20261220,20261230,20261240,20261250,20261260,20261501,*range(20261301,20261365)}
    paths=[];observed=set();collisions=[];maps=set();failures=[]
    root=REPO.parent.parent
    excluded={'.git','__pycache__','node_modules','preflight_raw','preflight_completion_raw','branches','rank','continuations','src','bin','raw','raw_logs','targets_train'}
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in excluded and not d.startswith(('models','runs','cache','checkpoints'))]
        for name in files:
            p=Path(directory)/name
            for value in re.findall(r'(?<!\d)2026\d{4}(?!\d)',name):
                observed.add(int(value))
                if int(value)==candidate:collisions.append(str(p))
            relevant=p.suffix.lower()=='.md' or (p.suffix.lower()=='.json' and any(k in name.lower() for k in ('seed','schedule','manifest','plan','config','receipt','identit','state')))
            if not relevant or p.stat().st_size>8_000_000:continue
            try:
                data=p.read_bytes();text=data.decode('utf-8-sig')
            except (OSError,UnicodeError) as exc:failures.append({'path':str(p),'type':type(exc).__name__});continue
            values={int(v) for v in re.findall(r'(?<!\d)2026\d{4}(?!\d)',text)}
            observed.update(values)
            if candidate in values:collisions.append(str(p))
            paths.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest(),'seed_literals':sorted(values)})
            if p.suffix=='.json':
                try:
                    obj=json.loads(text)
                    def visit(value):
                        if isinstance(value,dict):
                            for key,x in value.items():
                                if key=='map_seed' and isinstance(x,int):maps.add(x)
                                visit(x)
                        elif isinstance(value,list):
                            for x in value:visit(x)
                    visit(obj)
                except ValueError:pass
    assert not collisions,f'candidate already found: {collisions[:5]}'
    assert not failures,f'read errors prevent authority: {failures[:5]}'
    for path in ('docs/DIGEST.md','docs/RL_PLAN.md','tasks/SUMMARY.md'):
        text=subprocess.check_output(['git','show','origin/main:'+path],cwd=REPO).decode()
        assert str(candidate) not in text,f'fresh remote reserves candidate: {path}'
    report={'version':'ctf-host-decision-seed-check-1','checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'selected_seed':candidate,'status':'unused_before_reservation','fresh_remote_main':subprocess.check_output(['git','rev-parse','origin/main'],cwd=REPO,text=True).strip(),'scope':'current remote ledger/RL plan/task summary plus all retained local worktree file names and readable schedule/config/manifest/receipt/identity/state/Markdown metadata; fixed previously consumed seed exclusions','checked_files':paths,'observed_seed_literals':sorted(observed),'known_derived_map_seeds':sorted(maps),'read_failures':failures,'candidate_collisions':collisions}
    C.write_json(REPO/'build/decision_host_20261014/SEED_CHECK.json',report)
    print(json.dumps({'candidate':candidate,'status':report['status'],'metadata_files':len(paths),'observed_literals':len(observed),'derived_map_seeds':len(maps)}),flush=True)


if __name__=='__main__':main()
