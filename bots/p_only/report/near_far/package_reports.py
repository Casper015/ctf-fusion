"""Package completed verification; normalize report metadata paths, never data.

Run only after all three delivery checks return0. Keeps original frozen receipt
bytes in ignored build and publishes explicit hashes for metadata transport.
"""
import hashlib
import json
from pathlib import Path
import re

P=Path.cwd();R=P/'bots/p_only/report/near_far';B=P/'build/t0004'
def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=1)+'\n',encoding='utf-8',newline='\n')


def main():
    checks=read(B/'checks.json');assert len(checks)==3 and all(x['exit_code']==0 for x in checks)
    audit=read(R/'INDEPENDENT_AUDIT.json')
    assert audit['passed'] and audit['paired_map_near_far_concentration_independently_recomputed']
    logs=[(B/f'check_{i}.log').read_text(encoding='utf-8-sig',errors='replace') for i in range(3)]
    totals=[]
    for log in logs[:2]:
        matches=re.findall(r'Ran (\d+) tests?',log);assert matches and re.search(r'^OK(?:\s|$)',log,re.MULTILINE)
        totals.append(int(matches[-1]))
    release=read(B/'check_bundles/MANIFEST.json');assert release['variants'] and all(v['syntax_check']=='ok' for v in release['variants'].values())
    commands=dict(checks=checks,actual_core_tests=totals[0],actual_ponly_tests=totals[1],actual_bundles=len(release['variants']),
                  ignored_log_sha256={f'check_{i}.log':sha((B/f'check_{i}.log').read_bytes()) for i in range(3)})
    write(R/'CHECK_COMMANDS.json',commands)
    text='# T0004 verification\n\nFrozen measurement source:8a4eab4bef5bf65e971ec6712eebb89821738130. Changes are limited to this report lane.\n\n'
    text+='- Native P2 scorer:39 size/count cases; exact320 integer policy-head accumulators per probe; NLL within1e-12; independent BFS/line-of-fire geometry validation.\n'
    text+='- Loaded S/L model modules:26 size/count cases,16 probes each, exact320 policy accumulators, same compiler/2048 scratch ABI.\n'
    text+='- Full scan:260 frozen maps,832000 valid opponent rows,789836 primary; canonical labels and primary/conditional denominators verified.\n'
    text+='- Decisions: six modes x26maps x400states; original/disabled control identical; root digests invariant; no truncation or invalid complete replies. Complete-order movement/action/target changes independently audited.\n'
    text+='- Independent audit: raw/source/binary/array/replay hashes, paired joint means and map-cluster intervals, label coverage and complete protocol orders passed.\n'
    text+=f'- Core suite:{totals[0]} tests run; P-only suite:{totals[1]} tests run; all returned0. Platform skips, if any, remain in the original logs.\n'
    text+=f'- Single-file package syntax:{len(release["variants"])} bundles passed.\n\n'
    text+='Checks ran sequentially under a separate exclusive CPU lease after measurement completion. Test fixtures are verification, not new tuning data. Wall time is not a capacity or latency comparison. No model export, main/task-board change or production-default change occurred.\n\n'
    for item in checks:text+='- `python '+' '.join(item['command'])+'`: exit0, '+str(round(item['wall_seconds'],3))+' wall seconds.\n'
    text+='\nNATIVE_METADATA_TRANSPORT.json makes the published include flag portable while preserving original receipt hashes and measurement identity. RECEIPT_TRANSPORT.json describes report CRLF-to-LF transport only. Original arrays, game records, binaries and replay bytes are unchanged.\n'
    (R/'CHECKS.md').write_text(text,encoding='utf-8',newline='\n')
    native_path=R/'FROZEN_NATIVE.json';native=read(native_path)
    original=B/'FROZEN_NATIVE.original.json'
    if any('Users' in flag or ':/' in flag or ':\\' in flag for flag in native['flags']):
        raw=native_path.read_bytes()
        if original.exists():assert original.read_bytes()==raw
        else:original.write_bytes(raw)
        changes={}
        for i,flag in enumerate(native['flags']):
            if flag.startswith('-I'):
                assert flag[2:].replace('\\','/').rstrip('/').endswith('/inference')
                native['flags'][i]='-I${REPO}/inference';changes[f'flags[{i}]']='Host include path replaced by the portable REPO placeholder; compiler semantics unchanged.'
        write(native_path,native)
        write(R/'NATIVE_METADATA_TRANSPORT.json',dict(original_receipt_sha256=sha(raw),published_receipt_sha256=sha(native_path.read_bytes()),
                                                    measurement_identity=native['identity'],fields=changes,
                                                    scope='Report metadata transport only. Original receipt retained in ignored build; no score, source, model or binary change.'))
    else:
        assert original.exists() and read(R/'NATIVE_METADATA_TRANSPORT.json')['published_receipt_sha256']==sha(native_path.read_bytes())
    for name in ('RESULTS.md','NLL_RESULTS.md','METHOD.md','BRANCH_STATUS.md'):
        path=R/name;s=path.read_text(encoding='utf-8')
        if name=='RESULTS.md':s=s.replace('Core/P-only/bundle delivery checks are running; CHECKS.md and PR delivery will follow their completion.','Local delivery checks have passed; CHECKS.md records their actual totals. Independent PR delivery and exact-head CI tracking are the final steps.')
        if name=='NLL_RESULTS.md':s=s.replace('Decision-swap diagnostics remain in progress; these are prediction-loss results, not strength evidence.','Decision-swap diagnostics and the independent audit have completed; these are offline prediction-loss results, not strength evidence.')
        if name=='METHOD.md':
            s=s.replace('Decision-swap diagnostics remain in progress; no strength claim is made.','Decision-swap diagnostics, independent audit and local delivery checks are complete; no strength claim is made.')
            s=s.replace('This is a possible implementation, not a validated interface or a production export.','The report-only module interface passed actual loaded-pointer raw-output validation for all served counts (MODULE_VALIDATION.json). It is not a production export.')
        path.write_text(s,encoding='utf-8',newline='\n')
    mapping={}
    for p in sorted(R.iterdir()):
        if p.is_file() and p.name!='RECEIPT_TRANSPORT.json':
            raw=p.read_bytes();logical=raw.replace(b'\r\n',b'\n')
            assert not re.search(rb'(?:C:[/\\]|Users[/\\]caspe|OneDrive[/\\])',logical,re.I),p.name
            mapping[p.name]=dict(raw_sha256=sha(raw),logical_lf_sha256=sha(logical),raw_bytes=len(raw),logical_lf_bytes=len(logical))
    write(R/'RECEIPT_TRANSPORT.json',dict(scope='Report-copy CRLF to LF only; original ignored data/build and measurement hashes unchanged.',files=mapping))
    print(json.dumps(dict(report_files=len(mapping)+1,checks=commands)),flush=True)


if __name__=='__main__':main()
