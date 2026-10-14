"""Build all requested host opponents without choosing a collection seed."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from training import decision_collect as C, decision_opponents as O
from fusion.paths import ai_test_root
from fusion.runner import find_compiler


def dependencies(source: Path) -> list[dict]:
    result = subprocess.run([find_compiler(), '-std=c++20', '-MM', '-MT', 'deps', '-I'+str(REPO/'inference'), source.as_posix()], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('dependency extraction failed: '+result.stderr[-1000:])
    body = re.sub(r'\\\r?\n', ' ', result.stdout).partition('deps:')[2]
    paths = [Path(tok.replace('\\ ', ' ')) for tok in re.findall(r'(?:\\.|[^\s])+', body)]
    return [{'path': C.rel(p.resolve()), 'sha256': O.sha256_file(p)} for p in paths]


def main():
    out = REPO / 'build/decision_host_20261014'
    out.mkdir(parents=True, exist_ok=True)
    reg = O.build_registry(ai_test_root())
    variant, source = C.default_builders()
    receipt_path = out/'BUILD_ALL.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'version':'ctf-host-decision-build-1','registry':C.strip_private(reg),'compiler':C.compiler_identity(),'engine':C.engine_identity(ai_test_root()),'runner':C.runner_identity(),'programs':{},'failures':{}}
    assert receipt['registry']['pool_identity'] == reg['pool_identity']
    requested = [('designated', reg['designated']['variant'], None), ('baseline','v12_ponly_xxl',None)]
    requested += [(m['name'], m['name'], m) for m in reg['members'] if m['status']=='enabled']
    for key, name, member in requested:
        if key in receipt['programs']:
            old=receipt['programs'][key]
            assert O.sha256_file(REPO/old['binary']) == old['binary_sha256']
            for item in old.get('inputs',[]):
                p=Path(item['path']);p=p if p.is_absolute() else REPO/p
                assert O.sha256_file(p)==item['sha256']
            print('verified build cache',key,flush=True)
            continue
        try:
            if member is None or member['kind']=='variant':
                build = variant(name, out/('designated' if key=='designated' else 'baseline' if key=='baseline' else 'bots'), key in ('designated','baseline'))
                build['source']=C.catalog_entry(name)['source']
            else:
                src=Path(member['resolution']['path'])
                build=source(src,out/'roster')
                build['source_sha256']=member['source_sha256']
                build['source']=C.rel(src)
                build['inputs']=dependencies(src)
                build['resolution']=member['resolution']
            if member:
                build.update({k:member[k] for k in ('component','group','lineage','unit','family','aliases')})
            receipt['programs'][key]=build
            receipt['failures'].pop(key,None)
            C.write_json(receipt_path,receipt)
            C.write_json(out/'build_state.json',{'completed':len(receipt['programs']),'expected':len(requested),'last':key})
            print('built',key,build['binary_sha256'][:12],flush=True)
        except BaseException as exc:
            receipt['failures'][key]={'error':str(exc),'traceback':traceback.format_exc()}
            C.write_json(receipt_path,receipt)
            print('build failure retained',key,flush=True)
    aliases={}
    for name,b in receipt['programs'].items():
        aliases.setdefault(b['binary_sha256'],[]).append(name)
    receipt['binary_aliases']=[names for names in aliases.values() if len(names)>1]
    receipt['complete']=len(receipt['programs'])==len(requested) and not receipt['failures']
    C.write_json(receipt_path,receipt)
    print(json.dumps({'complete':receipt['complete'],'built':len(receipt['programs']),'failures':list(receipt['failures']),'aliases':receipt['binary_aliases']}),flush=True)
    raise SystemExit(0 if receipt['complete'] else 1)


if __name__=='__main__':
    main()
