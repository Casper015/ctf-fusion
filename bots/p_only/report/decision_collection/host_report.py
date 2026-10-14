"""Generate the T0037 English host report from preserved completed receipts.

Host paths are normalized only in published metadata. Original byte hashes and
original freeze manifests remain in the pre-game and raw archives; no normalized
manifest is represented as an original collection freeze.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from host_verify import read,sha


def generate(repo,backup_dir):
    sys.path.insert(0,str(repo))
    from training import decision_opponents as O
    build=repo/'build/decision_host_20261014'
    ver=read(build/'DELIVERY_VERIFICATION.json');plan=read(build/'PLAN.json');frozen=read(build/'FROZEN.json')
    pre=read(build/'PREFLIGHT_FINAL.json');built=read(build/'BUILD_ALL.json');sched=read(build/'SCHEDULE.json')
    raw_backup=read(backup_dir/'RAW_BACKUP_RECEIPT.json');input_backup=read(backup_dir/'INPUT_BACKUP_RECEIPT.json')
    assert raw_backup['plan_identity']==input_backup['plan_identity']==plan['identity']==ver['plan_identity']
    assert raw_backup['archive_sha256']==sha(backup_dir/raw_backup['archive'])
    assert input_backup['archive_sha256']==sha(backup_dir/input_backup['archive'])
    reg=frozen['registry'];mix=O.mixture(reg)
    def normalized(obj):
        if isinstance(obj,dict):return {k:normalized(v) for k,v in obj.items()}
        if isinstance(obj,list):return [normalized(v) for v in obj]
        if not isinstance(obj,str):return obj
        value=obj.replace('\\','/')
        roots=[(str(repo).replace('\\','/'),'<repo>')]
        for count,probe in pre['recipe']['source_probes'].items():
            probe_path=probe['path'].replace('\\','/')
            roots.append((probe_path.split('/data/')[0],'<recorded-probes/T0010>'))
        # Original external source resolution receipts supply the exact arena root.
        for program in built['programs'].values():
            source=str(program.get('source','')).replace('\\','/')
            if 'ctf-arena/' in source and ':/' in source:roots.append((source.split('ctf-arena/')[0].rstrip('/'),'<ai-test>'))
        compiler=str(built['compiler'].get('path','')).replace('\\','/')
        for prefix,replacement in sorted(set(roots),key=lambda x:-len(x[0])):value=value.replace(prefix,replacement)
        if compiler:value=value.replace(compiler,'<compiler>')
        if ':/Users/' in value or ':/users/' in value:raise ValueError('unmapped private host path in publication')
        return value
    actual={};simulation={}
    for n in range(2,16):
        fixtures=[fx for fx in sched['fixtures'] if fx['teams']==n];den=24*(n-1)
        actual[str(n)]={'opponent_seats':den,'maps':24}
        for field in ('components','groups','lineages','units','opponents'):
            cnt=Counter(x for fx in fixtures for x in fx[field]);assert sum(cnt.values())==den
            actual[str(n)][field]={x:{'seats':v,'mass':v/den} for x,v in sorted(cnt.items())}
        rng=random.Random(O.derive(0,n,'simulate'));tallies={field:Counter() for field in ('component','group','lineage','unit')}
        for _ in range(2000):
            for unit in O.draw_opponents(rng,mix,n-1):
                row=mix['units'][unit]
                for field in tallies:tallies[field][row[field]]+=1
        simulation[str(n)]={'games':2000,'rng_seed':0,'opponent_seats':2000*(n-1),'masses':{field:{key:value/(2000*(n-1)) for key,value in sorted(cnt.items())} for field,cnt in tallies.items()}}
    assert all(not set(fx['lineages'])&set(reg['reserved_lineages']) for fx in sched['fixtures'])
    disabled=[m['name'] for m in reg['members'] if m['status']=='disabled']
    assert len(disabled)==5 and not any(set(f['opponents'])&set(disabled) for f in sched['fixtures'])
    observation=[]
    for line in (build/'collection_process/observer.jsonl').read_text(encoding='utf-8-sig').splitlines():observation.append(json.loads(line))
    alive=[r for r in observation if r['verified_wrapper_alive']]
    assert alive and not observation[-1]['verified_wrapper_alive']
    assert all(r['leases']['cpu'] and r['leases']['cpu']['task']=='T0037' and r['leases']['cpu']['agent']=='codex-decision-pilot@desktop-g60ojud' and not r['leases']['gpu'] for r in alive)
    host={'observer_samples':len(observation),'live_samples':len(alive),'first_utc':observation[0]['at_utc'],'last_utc':observation[-1]['at_utc'],'interval_seconds':15,'physical_bytes':alive[0]['physical_bytes'],'max_committed_bytes':max(r['committed_bytes'] for r in alive),'min_free_physical_bytes':min(r['free_physical_bytes'] for r in alive),'commit_limit_bytes':alive[0]['commit_limit_bytes'],'private_birth_unverified_samples':sum(not p['private_birth_verified'] for r in alive for p in r['processes']),'max_chain_private_bytes':max(sum(p['private_bytes'] or 0 for p in r['processes']) for r in alive),'cpu_lease_verified_all_observed_live_samples':True,'gpu_lease_absent_all_observed_live_samples':True,'limitations':'15-second passive snapshots cannot prove the absence of unrelated unleased workloads between observations; short-lived bots may not appear in each snapshot; initial build observer failure preserved separately'}
    publication={'schema':'ctf-host-decision-pilot-report-1','generator':'bots/p_only/report/decision_collection/host_report.py','generator_sha256':sha(Path(__file__)),'task':'T0037','host':'DESKTOP-G60OJUD','status':'completed_collection_ready_for_independent_T0038_audit','identities':{'plan':plan['identity'],'frozen':frozen['identity'],'pool':frozen['pool_identity'],'schedule':sched['schedule_identity'],'source_commit':plan['source_commit'],'source_files':plan['source_files'],'original_receipt_sha256':{name:sha(build/f'{name}.json') for name in ('PLAN','FROZEN','SCHEDULE','BUILD_ALL','PREFLIGHT_FINAL','SEED_CHECK','COLLECTION_COMPLETE','DELIVERY_VERIFICATION')}},'plan':plan,'build':built,'preflight':pre,'seed_check':{k:v for k,v in read(build/'SEED_CHECK.json').items() if k not in ('known_derived_map_seeds','checked_files')},'coverage':{k:v for k,v in ver.items() if k!='raw_inventory'},'raw_inventory_digest':O.digest(ver['raw_inventory']),'raw_inventory_files':len(ver['raw_inventory']),'preserved_common_probes':read(build/'COMMON_PROBE_BACKUP.json'),'disabled_source_commit_receipts':read(build/'DISABLED_SOURCE_COMMIT.json'),'actual_sampler':actual,'sampler':{'nominal':mix,'simulation':simulation,'method':'sequential weighted draws without replacement of exposure units, exact production sampler; simulations use the same preflight seed0 (no games executed)','cap_limitations':'nominal first-draw lineage cap is not a per-fixture quota or a realized aggregate cap; without-replacement conditional normalization changes later-seat masses, especially at large counts; actual sampled integer counts and all-lineage shares are retained'},'host_receipts':host,'backup':{'inputs':{k:v for k,v in input_backup.items() if k!='files'},'raw':{k:v for k,v in raw_backup.items() if k!='files'}},'published_metadata_normalization':'absolute source/command/compiler paths replaced with <repo>/<ai-test>/<compiler>/<recorded-probes/T0010> only for publication. Original PLAN/FROZEN/source receipts remain byte-for-byte in pre-game archive. Identity strings identify originals, not this normalized display.','t0038_handoff':{'raw_root':'data/decision/host_20261014','build_root':'build/decision_host_20261014','matches':'matches/n*/*.json.gz','roots':'roots/n*/*.json.gz','continuations':'continuations/n*/*.json.gz','assembled_rows':'branches/n*/*.json.gz','prerequisite':'reviewed host T0037 report PR merged and truthful done bookkeeping','checks_remaining':'independent full-prefix causal features, legal/canonical/target support, collisions/unavailable targets, matching noise populations, independent repeat subsets for headroom selection/evaluation, map-consistent dataset splits and original-replay memory audit; never replay successful games'},'limitations':['336 independent maps, one root each; repeats and candidates are clustered within maps, not additional independent maps.','Every fork initializes the same original map_seed and flag_seed and replays the common prefix. Future flag placements/RNG consumption can differ with state; wall-clock search and bot RNG are not controlled by an explicit repeat seed.','Original engine prefix orders are forced after fresh bots compute replies; forced candidate does not rewrite bot memory. Full computed/played replies and prefix mismatch/fault receipts permit an independent audit.','Root opportunities and fallback shortages remain; no root selected from final utility or future replies.','Shortlist is a bounded set of logged complete requested orders, not exhaustive legal-action enumeration. Target/canonical aliases and learnability remain unaudited until T0038.','Preflight recorded probes do not guarantee fault-free long matches. All collection faults and partials are reported, not quarantined retroactively.','Colab PR #37/seed20261501 and306map historical stage are separate. Current downloaded T0032 models are unadmitted and were not used.','No training, strength comparison, BT estimate, RL guarantee or deployment claim.']}
    correction=read(build/'FLAG_PROTOCOL_COVERAGE.json')
    publication['flag_protocol_correction']=correction
    publication['known_selection_defect']=correction['defect']
    publication['status']='completed_immutable_pilot_with_flag_selection_defect_ready_for_independent_audit'
    publication['backup']['source_commit']=read(backup_dir/'SOURCE_BACKUP_RECEIPT.json')
    checks=read(backup_dir/'CHECKS.json')
    assert checks.get('completed_utc') and all(row['exit_code']==0 for row in checks['checks'])
    publication['delivery_checks']={**checks,'receipt_sha256':sha(backup_dir/'CHECKS.json'),'logs_sha256':{row['log']:sha(backup_dir/row['log']) for row in checks['checks']}}
    supplement_path=backup_dir/'SUPPLEMENT_BACKUP_RECEIPT.json'
    supplement=read(supplement_path) if supplement_path.exists() else None
    if supplement:
        assert supplement['archive_sha256']==sha(backup_dir/supplement['archive'])
        publication['backup']['delivery_supplement']={k:v for k,v in supplement.items() if k!='files'}
    supplement_text=(f"Post-collection delivery supplement: `{supplement['archive']}`, SHA `{supplement['archive_sha256']}`, {supplement['verified_members']} independently read-back verified members. It preserves the corrected observable-only audit, delivery scripts and validation receipts separately from the original sealed inputs/raw/source bundle." if supplement else 'Delivery supplement receipt is pending; original sealed inputs/raw/source bundle are unchanged.')
    for name in ('prefix_faults','root_discarded_faults','continuation_faults','partial_continuations','root_drops'):
        publication['coverage']['counts'][name]=ver['counts'].get(name,0)
    publication['t0038_handoff']['checks_remaining']+='; explicitly account for flag/quiet misclassification and original selection bias, using supplemental corrected context receipts'
    publication=normalized(publication)
    out=repo/'reports/host/decision_opponent_preflight_DESKTOP-G60OJUD.json'
    out.write_text(json.dumps(publication,sort_keys=True,indent=1)+'\n',encoding='utf-8')
    c=ver['counts'];table='\n'.join(f"| {n} | {v['maps']} | {v['roots']} | {v['continuations']} |" for n,v in ver['per_count'].items())
    text=f'''# T0037 Windows expanded-pool decision pilot

Generated by `bots/p_only/report/decision_collection/host_report.py` from preserved raw receipts. Machine evidence: [JSON report](decision_opponent_preflight_DESKTOP-G60OJUD.json).

The bounded Windows collection is complete: **{c['matches']} independent maps, {c['roots']} selected roots and {c['continuations']} continuations**, across counts 2–15. Offline review found a flag-selector protocol mistake: the declared flag-targeted quota was not achieved. Actual selected roots still cover all six contexts, including 52 with nearby ground flags and 54 truly quiet roots. It is ready for the independent T0038 data-quality audit after review and merge, with this selection defect explicitly retained. This delivery establishes provenance and actual coverage; it does not establish balanced intended strata, training sufficiency or playing strength.

## Frozen design and admission

Training-only master seed 20261701 was checked against fresh remote ledger/task evidence, retained schedules and host receipts, then reserved before the first fixture. Original source commit `{plan['source_commit']}`; plan `{plan['identity']}`; frozen identity `{frozen['identity']}`. Full source/compiler/engine/runner/model/calibration/binary hashes are pinned in JSON and the original pre-game archive. All measured sources and commands stayed frozen; 6 workers were selected before collection from the 1/2/4/6 ladder.

All 52 available opponents plus designated and diagnostic baseline built (54 binaries), with zero build failures and no exact binary aliases. Five missing-weight entries remain disabled with zero sampling. Reserved lineages were excluded; admission/quarantine, semantic groups, source aliases and all transitive model includes are retained. The designated carrier is `v12_ponly_xxl_dlog`, admitted P2 XXL, V off, statistics enabled; binary `{frozen['designated']['binary_sha256']}`. The diagnostic production baseline is distinct. No T0032 replacement was admitted.

Common probes used 96 recorded turns at counts 2/5/10/15, two standalone repetitions, followed by the complete official 2000/50 ms load ladder including n15. Final preflight retains 432 physical standalone and 208 load probes, all observed probe faults zero. Log versus correct diagnostic baseline orders agree on 768/768 comparisons. Per-program A/A and neural-group response differences are in JSON; they are descriptive, with no arbitrary A/A exclusion rule. Aggregate accepted latency p50={pre['latency_seconds']['p50']:.6f}s, p99={pre['latency_seconds']['p99']:.6f}s, max={pre['latency_seconds']['max']:.6f}s; these recorded-input distributions do not establish full-match fault absence.

Windows case-insensitive paths initially collided for reference `Baseline` and diagnostic `baseline`. The reference probes completed; eight diagnostic probes had not executed. Original raw and reports were retained, those eight missing probes ran once under the CPU lease, and an offline exact-name/command aggregation produced `PREFLIGHT_FINAL.json`. No successful probes were replayed. An initial build observer failed due to birth-string conversion; its failed receipt remains. The corrected passive observer was verified before preflight/formal collection.

The sampler chooses distinct exposure units by sequential weighted draws without replacement. JSON publishes every actual component/group/lineage/unit/name count and mass at each count, plus exact-sampler 2000-fixture simulations per count. Nominal lineage cap applies to the initial weight distribution; later conditional draws and realized samples need not obey that numeric cap. Disabled/reserved entries have zero exposure.

## Collection coverage

| Teams | Independent maps | Selected roots | Continuations |
|---|---:|---:|---:|
{table}

Each count requested four roots per carrier/flag/contact/interception/defense/quiet category. The frozen implementation used observable state and valid logged incumbent scores after turn 6, deterministic root hashing and a declared fallback rotation. Original opportunity labels and selections are retained verbatim for provenance, but its flag/quiet opportunity labels are incorrect (see below). Requested counts: `{json.dumps(ver['requested_strata'],sort_keys=True)}`. Original selected-category counts: `{json.dumps(ver['selected_strata'],sort_keys=True)}`. Fallbacks: {c['fallbacks']}; root drops: {c.get('root_drops',0)}. Collected K distribution: `{json.dumps(ver['collected_k'],sort_keys=True)}`; original logged K is separate in JSON. Every candidate is an actual logged complete requested order with its aligned carrier score; K1 and shortened shortlists remain.

### Flag protocol defect and corrected context coverage

The collector incorrectly tested for flag status `ground`. The actual engine protocol uses `home` and `dropped` for flags on the ground, and `carried`/`cooldown` otherwise. Consequently all 56 requested flag targets entered the declared fallback rotation. This was an implementation error, not a real opportunity shortage: all 336 maps have ground-flag opportunities after turn 6 (24,519 observable opportunity turns). The synthetic root contract fixture had also used the invented status, so it did not catch this mistake.

`host_flag_coverage.py` supplies the corrected protocol predicate and independently reclassifies the preserved observable blocks. Its fail-fast status validator and three real-protocol regression checks reject invented statuses. It changes no roots, candidates, repetitions, outcomes or original freeze; the original collector remains historical provenance and must not be used for new collections without correcting this predicate under a new freeze.

Actual selected-root context counts (overlapping): `{json.dumps(correction['summary']['selected_root_contexts'],sort_keys=True)}`. Actual nearby ground flags occur at 52/336 selected roots (15.48%); only 10/56 flag-requested maps happened to select a flag-context root (17.86%). Actual quiet occurs at 54 roots; 50/56 quiet-requested maps meet the corrected quiet definition. Requested-target hit rates and opportunity-map/turn counts at each team count, plus every unchanged root's corrected contexts, are in JSON and `FLAG_PROTOCOL_COVERAGE.json`. The original nominal flag quota and quiet labels must not be used as actual coverage. This supplemental audit executes no bots and does not reselect roots or add favorable samples. T0038 must assess selection bias and usable stratum/group precision from these corrected contexts.

Four repeats were completed for each selected candidate, without sample expansion or favorable early stopping. Original matches retain all-seat replies, faults and terminal scores. Continuations retain full computed and played reply trajectories, original prefix, common opponent root actions and live reactive opponents thereafter. Fresh processes replay the prefix with the declared 5000 ms replay allowance (first reply 2000 ms); the root and future live replies use 50 ms. Forcing the engine's action does not restore internal carrier memory. Each fork initializes the original map_seed and flag_seed and replays the common engine prefix. Future state and RNG consumption may differ across actions; repeat numbers do not seed native-bot RNG. Repeats are not new independent maps.

## Faults and verification

Original-match faults: **{c.get('match_faults',0)}**; missing reply slots: {c.get('match_missing_replies',0)}. Continuation phase faults: prefix {c.get('prefix_faults',0)}, discarded computed root {c.get('root_discarded_faults',0)}, live continuation {c.get('continuation_faults',0)}. Partial continuations: {c.get('partial_continuations',0)}. Prefix requested-order mismatches: {c.get('prefix_order_mismatches',0)}; raw/log reply mismatches: {c.get('prefix_raw_mismatches',0)}. These observations remain in raw and are not removed or converted into clean examples.

Offline verification checked every declared fixture/seat/source identity, successful file SHA, root shortlist/score binding, candidate/repeat linkage, assembled row, all-seat trajectory shape and actual played prefix/root orders. Utility was independently recomputed from raw final scores (opponent wins=1, ties=.5, losses=0, averaged), and final score blocks agree. It executed no bots and replayed no successful games. This is collection delivery verification; T0038 must independently audit causal features, legal/canonical target support, collisions, uncertainty and noise using matching root populations and separate selection/evaluation repeats.

Formal collection held the exclusive CPU lease. Passive 15-second receipts observed {host['live_samples']} live snapshots, verified actual birth/command/parent and CPU/private memory, and found no GPU lease in each. Peak observed committed memory {host['max_committed_bytes']} bytes; physical memory {host['physical_bytes']} bytes; commit limit {host['commit_limit_bytes']} bytes. Private-memory birth checks unavailable/unverified for {host['private_birth_unverified_samples']} sampled processes. Short-lived processes and activity between snapshots are not continuously observed. The observer never altered resources, paging, applications or processes.

Required repository checks passed after collection under a separate CPU lease: core 316 tests (one skipped), P-only 50 tests, four collection contracts, board validation, Markdown index and bundle/compile checks. Three corrected flag-protocol regressions also passed. `CHECKS.json` and its hashed logs are preserved in the delivery supplement. These functional checks are distinct from the immutable pilot; they did not replay its successful fixtures.

## Preserved evidence and reproduction

Original inputs were archived and every 258 member hash verified before the first fixture: `{input_backup['archive']}`, SHA `{input_backup['archive_sha256']}`. Completed raw/build archive: `{raw_backup['archive']}`, SHA `{raw_backup['archive_sha256']}`, {raw_backup['verified_members']} independently read-back verified members. Both archives and full inventories reside in the chat's `outputs/T0037_host/`; raw and binaries remain ignored in Git. JSON contains archive sizes, hash receipts and raw inventory digest. Original freeze manifests are preserved, not reconstructed from later bytes. Published metadata normalizes private paths only; quoted identity values identify the original manifests.

{supplement_text}

Preserved repository-relative raw: `data/decision/host_20261014/`; build/freeze/probe/host receipts: `build/decision_host_20261014/`. Existing successful raw files must never be recollected to reproduce this report. Restore the exact measured 2313043 source tree from the source bundle, add the preserved delivery scripts, and supply the exact retained raw/freeze/binaries and recorded engine root before running source-sensitive verification:

```powershell
python bots/p_only/report/decision_collection/host_verify.py
python bots/p_only/report/decision_collection/host_flag_coverage.py
python bots/p_only/report/decision_collection/host_backup.py --destination <delivery-directory>
python bots/p_only/report/decision_collection/host_report.py --backup-dir <delivery-directory>
```

The complete measured source tree/history is also preserved in `SOURCE_2313043.bundle`, with its SHA and independent bare-clone/fsck receipt in JSON. Source-sensitive verification requires the exact measured source commit, not current main: preserve/restore the bundle's 2313043 tree, add the delivery verification scripts, and read the exact retained raw/freeze/binaries. Current main has independent later changes which must not be interpreted as the original collection source. Neither the original freeze nor missing individual pre-game receipts were reconstructed from later bytes.

`host_collect.py` is preserved as historical source with the flag-selector defect; do not use it for a new collection or replay completed fixtures. `host_flag_coverage.py` is the corrected diagnostic implementation. Independent T0038 starts only after this host PR is reviewed, merged and T0037 truthfully marked done. Colab PR #37, consumed 20261501 and 306 maps remain a separately identified historical/exploratory source; none were recovered or mixed into this collection. Recommendation: proceed to independent data-quality audit with corrected flag/quiet contexts and explicit selection bias; balanced flag-targeted selection was not demonstrated, and training/strength/deployment decisions remain unassessed.
'''
    out.with_suffix('.md').write_text(text,encoding='utf-8')
    print('generated report and JSON',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4]);p.add_argument('--backup-dir',type=Path,required=True);a=p.parse_args();generate(a.repo,a.backup_dir)
