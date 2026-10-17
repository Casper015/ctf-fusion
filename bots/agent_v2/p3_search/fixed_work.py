"""Materialize matched-work P2/P3 trees. Offline model diagnostics, never strength or timing evidence."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from fusion import runner


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stream',type=Path,required=True,help='one full causal seat stream, optionally .gz')
    ap.add_argument('--turns',type=int,nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(argv)
    if not a.turns or min(a.turns)<1:ap.error('turns must be positive')
    cxx=runner.find_compiler()
    if not cxx:ap.error('a C++20 compiler is required')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    text=a.stream.read_bytes()
    if a.stream.suffix=='.gz':text=gzip.decompress(text)
    stream=out/'stream.txt';stream.write_bytes(text)
    actor=REPO/'inference/meow_rl/aligned/p2_large/actor.hpp'
    critic=REPO/'inference/meow_rl/aligned/p2_large/critic_fp32.hpp'
    all_trees={}
    for family,tier in [('p2','FUSION_TIER_P2GENERAL_VM'),('p3','FUSION_TIER_P3XXL_VM')]:
        exe=out/f'check_{family}{runner.EXE}'
        flags=['-std=c++20','-O2','-I',str(REPO/'inference'),f'-D{tier}=1',
               '-DAGENT_V2_SEARCH_CANDIDATES=8','-DAGENT_V2_SEARCH_SAMPLES=4',
               '-DAGENT_V2_SEARCH_TAU=0.002',f'-DACTOR_HEADER="{actor.as_posix()}"',
               f'-DCRITIC_HEADER="{critic.as_posix()}"']
        subprocess.run([cxx,*flags,str(REPO/'tools/cpp/agent_v2/check.cpp'),'-o',str(exe)],check=True)
        env=dict(os.environ,AGENT_V2_TREE_LEAVES=str(out/f'{family}.f32'))
        r=subprocess.run([str(exe),'tree',str(stream),*map(str,a.turns)],text=True,capture_output=True,check=True,env=env)
        rows=[json.loads(x) for x in r.stdout.splitlines()]
        if sorted(x['turn'] for x in rows)!=sorted(set(a.turns)):
            raise SystemExit('some requested turns were absent/refused; do not compare incomplete roots')
        all_trees[family]=rows
        (out/f'{family}.jsonl').write_text(r.stdout,encoding='utf-8')
    # Inputs/actor are identical: only opponent-model forecasts and resulting tactics/successors may differ.
    for x,y in zip(all_trees['p2'],all_trees['p3']):
        if (x['turn'],x['actor'],x['roots'],x['samples'])!=(y['turn'],y['actor'],y['roots'],y['samples']):
            raise SystemExit('root candidates or actor changed between families')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipts={'schema':'ctf-agent-v2-p3-fixed-work-1','scope':'offline model diagnostics; not strength evidence',
              'stream_sha256':sha(stream),'actor_sha256':sha(actor),'critic_sha256':sha(critic),
              'candidates':8,'samples':4,'turns':a.turns,'trees':len(all_trees['p2']),
              'outputs':{f:sha(out/f) for f in ['p2.jsonl','p3.jsonl','p2.f32','p3.f32']}}
    (out/'RECEIPT.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
