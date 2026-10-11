"""Package successful local checks and report-only transport receipts."""
import hashlib,json,pathlib,re,shutil
P=pathlib.Path.cwd();R=P/'bots/p_only/report/search_portfolio';B=P/'build/t0019'
checks=json.loads((B/'checks.json').read_text(encoding='utf-8-sig'));assert len(checks)==3 and all(x['exit_code']==0 for x in checks)
sha=lambda b:hashlib.sha256(b).hexdigest()
logs=[(B/f'check_{i}.log').read_text(encoding='utf-8-sig',errors='replace') for i in range(3)]
assert re.search(r'Ran 189 tests',logs[0]) and 'OK (skipped=1)' in logs[0]
assert re.search(r'Ran 43 tests',logs[1]) and 'OK' in logs[1]
release=json.loads((B/'check_bundles/MANIFEST.json').read_text(encoding='utf-8'))
assert len(release['variants'])==29 and all(v['syntax_check']=='ok' for v in release['variants'].values())
text='''# T0019 validation

The measurements use frozen source `05f4aca0acd2f4d31b101b9b71b4afa95c888330`; this branch adds only the task report lane.

| Check | Result |
|---|---|
| Independent raw audit under exclusive CPU lease | Passed:2240 raw games,400 selected branch roots, exact scheduled coverage, frozen identities, utilities, faults, candidate/repeat/source/root blocks, resolved pairs and clustered regret intervals |
| Paired statistics | Passed: all overall/band/per-count contrasts, both A/A studies and interaction match generated JSON |
| Aggregates | Passed: all eight arms' primary timing, PONLY/SEARCH/PLY2/PORTFOLIO counters and consistent settings |
| Core unittest suite |189 tests run:188 passed,1 platform skip |
| P-only unittest suite |43 tests passed |
| Single-file release package check | Passed;29 bundles |

Local tests and package checks ran sequentially under a separate exclusive CPU lease after formal measurement completion. They are verification fixtures, not additional tuning data or strength measurements. The recorded stage/check durations are wall time receipts, not same-condition capacity comparisons. Runtime warnings about test fixture file handles are retained in ignored logs; commands returned0.

No main/task-board mutation, model export, production default change, formal game replay or interference with the user desktop applications occurred. T0010 PR23 is handled by other reviewers and is not merged by this task.

`RECEIPT_TRANSPORT.json` maps report files' original raw bytes to logical LF bytes for Git transport. It does not alter the frozen ignored data/build files or redefine their measurement hashes. The independent audit uses the original frozen checkout and retained data as documented in REPRODUCE.md.

## Check command receipts

'''
for x in checks:
 text+='- `python '+' '.join(x['command'])+'`: exit0, '+str(round(x['wall_seconds'],3))+' wall seconds.\n'
(R/'CHECKS.md').write_text(text,encoding='utf-8',newline='\n')
(R/'CHECK_COMMANDS.json').write_text(json.dumps({'checks':checks,'ignored_log_sha256':{f'check_{i}.log':sha((B/f'check_{i}.log').read_bytes()) for i in range(3)}},indent=1)+'\n',encoding='utf-8',newline='\n')
mapping={}
for p in sorted(R.iterdir()):
 if p.is_file() and p.name!='RECEIPT_TRANSPORT.json':
  raw=p.read_bytes();logical=raw.replace(b'\r\n',b'\n');mapping[p.name]={'raw_sha256':sha(raw),'logical_lf_sha256':sha(logical),'raw_bytes':len(raw),'logical_lf_bytes':len(logical)}
(R/'RECEIPT_TRANSPORT.json').write_text(json.dumps({'purpose':'Report-copy CRLF to LF transport only; original frozen data/build unchanged.','files':mapping},indent=1)+'\n',encoding='utf-8',newline='\n')
print('packaged',len(mapping)+1,'report files')
