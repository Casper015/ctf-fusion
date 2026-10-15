"""Explicit runtime admission and exact-binary exposure units for the host pilot."""
from __future__ import annotations
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from training import decision_opponents as O


def runtime_registry(root:Path,build:dict,quarantine:dict)->dict:
    reg=O.build_registry(root)
    groups={}
    for member in reg['members']:
        if member['name'] in quarantine:
            member['status']='quarantined'
            member['reason']='official-limit standalone probe fault; retained in QUARANTINE.json'
        if member['status']=='enabled':
            groups.setdefault(build['programs'][member['name']]['binary_sha256'],[]).append(member)
    ledger=[]
    for binary, members in groups.items():
        if len(members)<2:continue
        members=sorted(members,key=lambda m:m['name'])
        names=[m['name'] for m in members]
        ledger.append({'binary_sha256':binary,'members':names,'original_units':[m['unit'] for m in members],'original_groups':[m['group'] for m in members],'original_lineages':[m['lineage'] for m in members]})
        for member in members:
            member['unit']='binary:'+binary
            member['aliases']=names
            member['representative']=member is members[0]
    reg['binary_alias_ledger']=ledger
    reg['pool_identity']=O.pool_identity(reg)
    return reg
