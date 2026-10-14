"""Preserve completed host raw/build receipts and verify every archived byte.

This is an offline I/O operation. It never executes bots, deletes originals,
extracts archive paths, reconstructs the original freeze, or uploads anything.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from host_verify import read,sha


def verify_archive(path,inventory):
    seen=set()
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in inventory and member.name not in seen
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            expected=inventory[member.name];assert member.size==expected['bytes']
            h=hashlib.sha256()
            with archive.extractfile(member) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            assert h.hexdigest()==expected['sha256'],member.name
            seen.add(member.name)
    assert seen==set(inventory)
    return len(seen)


def backup(repo,destination):
    build=repo/'build/decision_host_20261014';data=repo/'data/decision/host_20261014'
    assert read(build/'COLLECTION_COMPLETE.json')['stage']=='collection_complete'
    assert read(build/'collection_process/COMPLETE.json')['exit_code']==0
    lines=(build/'collection_process/observer.jsonl').read_text(encoding='utf-8-sig').splitlines()
    assert not json.loads(lines[-1])['verified_wrapper_alive'],'observer still running'
    assert (build/'DELIVERY_VERIFICATION.json').exists()
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/'HOST_RAW_seed20261701.tar.gz';receipt_path=destination/'RAW_BACKUP_RECEIPT.json'
    if path.exists():
        assert receipt_path.exists(),'incomplete archive retained; review before making another'
        receipt=read(receipt_path);assert sha(path)==receipt['archive_sha256']
        verify_archive(path,receipt['files']);print('existing verified raw archive reused');return receipt
    files=sorted(p for folder in (build,data) for p in folder.rglob('*') if p.is_file() and p.name not in ('RAW_BACKUP_RECEIPT.json','RAW_INVENTORY.json') and not p.name.endswith('.pyc'))
    inventory={p.relative_to(repo).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    assert not any('.tmp' in p.name for p in files),'unfinished stage files present'
    temp=path.with_suffix('.partial')
    assert not temp.exists(),'partial backup retained for inspection'
    with tarfile.open(temp,'w:gz',compresslevel=1) as archive:
        for i,p in enumerate(files):
            name=p.relative_to(repo).as_posix();assert sha(p)==inventory[name]['sha256']
            archive.add(p,arcname=name,recursive=False)
            if (i+1)%1000==0:print(f'archived {i+1}/{len(files)}',flush=True)
    verified=verify_archive(temp,inventory)
    temp.replace(path)
    receipt={'schema':'ctf-host-raw-backup-1','plan_identity':read(build/'PLAN.json')['identity'],'archive':path.name,'archive_sha256':sha(path),'archive_bytes':path.stat().st_size,'source_bytes':sum(i['bytes'] for i in inventory.values()),'verified_members':verified,'files':inventory,'method':'every tar member size and SHA256 independently read back; originals retained; source/build/engine original freeze preserved by separate pre-game input archive','original_input_archive':read(destination/'INPUT_BACKUP_RECEIPT.json')['archive_sha256']}
    receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=1)+'\n',encoding='utf-8')
    (build/'RAW_BACKUP_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4]);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();backup(a.repo,a.destination)
