"""Source-check and aggregate exact raw probes without executing any bot."""
from __future__ import annotations
import json
import host_preflight as P
from training import decision_collect as C,decision_opponents as O
from training.decision_branch_collect import order_of


def main():
    out=P.OUT
    original=json.loads((out/'PREFLIGHT.json').read_text())
    completion=json.loads((out/'PREFLIGHT_VERIFIED.json').read_text())
    build=json.loads((out/'BUILD_ALL.json').read_text())
    rows={};source_receipts=[]
    for name,b in build['programs'].items():
        for n in P.COUNTS:
            for repeat in range(2):
                label=f'standalone/{name}/n{n:02d}/r{repeat}'
                path=out/'preflight_raw'/f'{label}.json.gz'
                row=P.read_probe(path,original['identity'])
                commands=[[str(P.REPO/b['binary'])]]
                if row['programs']!=[name] or row['commands']!=commands:
                    path=out/'preflight_completion_raw'/O.digest(name)/f'n{n:02d}-r{repeat}.json.gz'
                    row=P.read_probe(path,completion['completion_identity'])
                assert row['programs']==[name] and row['commands']==commands
                assert row['match_sha256']==original['recipe']['source_probes'][str(n)]['sha256']
                rows[label]=row
                source_receipts.append({'path':C.rel(path),'sha256':O.sha256_file(path),'logical_probe':label,'program_binary_sha256':b['binary_sha256']})
    final=dict(completion)
    final['baseline_log_order_equivalence']=[{'count':n,'repeat':r,'different_orders':sum(order_of(a[0])!=order_of(b[0]) for a,b in zip(rows[f'standalone/designated/n{n:02d}/r{r}']['replies'],rows[f'standalone/baseline/n{n:02d}/r{r}']['replies'])),'total':len(rows[f'standalone/designated/n{n:02d}/r{r}']['replies'])} for n in P.COUNTS for r in range(2)]
    final['source_checked_standalone_receipts']=source_receipts
    final['aggregation']={'driver_sha256':O.sha256_file(__file__),'original_report_sha256':O.sha256_file(out/'PREFLIGHT.json'),'completion_report_sha256':O.sha256_file(out/'PREFLIGHT_VERIFIED.json'),'scope':'all requested exact-name/exact-command standalone probes; correct diagnostic baseline versus reference Baseline; no bot replay'}
    final['identity']=O.digest([completion['identity'],final['aggregation'],source_receipts])
    C.write_json(out/'PREFLIGHT_FINAL.json',final)
    print(json.dumps({'identity':final['identity'],'selected_workers':final['selected_workers'],'standalone':len(rows),'baseline_log_order_equivalence':final['baseline_log_order_equivalence']}),flush=True)


if __name__=='__main__':main()
