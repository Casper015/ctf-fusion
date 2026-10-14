"""Frozen 336-map Windows pilot with one observable root and atomic repeat files.

Prepare only after seed/lease/source checks. Running this file without arguments
collects the declared whole-match and continuation stages. It never trains,
rebuilds, changes concurrency, expands the sample, or repeats a successful raw file.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from fusion import matchlog,runner
from fusion.labels import pairwise_outcome
from fusion.paths import ai_test_root,load_engine
from fusion.protocol import parse_block
from training import decision_collect as C,decision_opponents as O,decision_branch_collect as B
from training.branch_collect import recorded_orders
from host_common import runtime_registry

OUT=REPO/'build/decision_host_20261014'
DATA=REPO/'data/decision/host_20261014'
CATEGORIES=('carrier','flag','contact','interception','defense','quiet')
REPEATS=4
ALTERNATIVES=3


def load_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as h:return json.load(h)


def save_gz(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp{os.getpid()}')
    with gzip.open(temp,'wt',encoding='utf-8') as h:json.dump(obj,h)
    temp.replace(path)


def source_snapshot():
    paths=[*C.RUNNER_FILES,*B.COLLECTOR_FILES,'fusion/protocol.py','fusion/labels.py','fusion/paths.py','training/dataset.py']
    paths+= [(Path(__file__).parent/name).relative_to(REPO).as_posix() for name in ('host_build.py','host_common.py','host_preflight.py','host_preflight_repair.py','host_preflight_aggregate.py','host_seed_check.py','host_collect.py','host_helper.ps1','host_observer.ps1')]
    return {p:O.sha256_file(REPO/p) for p in sorted(set(paths))}


def actual_registry():
    return runtime_registry(ai_test_root(),json.loads((OUT/'BUILD_ALL.json').read_text()),json.loads((OUT/'QUARANTINE.json').read_text()))


def verify_plan():
    plan=json.loads((OUT/'PLAN.json').read_text())
    assert O.digest({k:v for k,v in plan.items() if k!='identity'})==plan['identity'],'plan identity changed'
    assert source_snapshot()==plan['source_files'],'source changed after freeze'
    frozen=json.loads((OUT/'FROZEN.json').read_text())
    bad=C.verify(frozen,ai_test_root(),registry=actual_registry())
    assert not bad,';'.join(bad)
    for name,b in json.loads((OUT/'BUILD_ALL.json').read_text())['programs'].items():
        assert O.sha256_file(REPO/b['binary'])==b['binary_sha256']
        for item in b.get('inputs',[]):
            p=Path(item['path']);p=p if p.is_absolute() else REPO/p
            assert O.sha256_file(p)==item['sha256'],f'build input changed: {name} {p}'
    assert O.sha256_file(OUT/'PREFLIGHT_FINAL.json')==plan['preflight_sha256']
    return plan,frozen


def prepare(seed):
    assert not (OUT/'PLAN.json').exists(),'a frozen host plan already exists; do not replace it'
    assert (OUT/'SEED_CHECK.json').exists(),'fresh seed inspection receipt required'
    seed_receipt=json.loads((OUT/'SEED_CHECK.json').read_text())
    assert seed_receipt['selected_seed']==seed and seed_receipt['status']=='unused_before_reservation'
    pre=json.loads((OUT/'PREFLIGHT_FINAL.json').read_text())
    build=json.loads((OUT/'BUILD_ALL.json').read_text())
    reg=actual_registry()
    schedule=O.make_schedule(reg,seed,24,range(2,16),allow_consumed=True)
    # allow_consumed reflects the required ledger reservation BEFORE this prepare,
    # not a bypass of the independently saved fresh-seed inspection.
    assert len(schedule['fixtures'])==336
    assert len({fx['map_seed'] for fx in schedule['fixtures']})==336
    assert not ({fx['map_seed'] for fx in schedule['fixtures']} & set(seed_receipt['known_derived_map_seeds'])), 'new schedule reuses a retained historical map seed'
    schedule['realized_component_shares']=C.schedule_shares(schedule)
    schedule['seed_check_sha256']=O.sha256_file(OUT/'SEED_CHECK.json')
    schedule['schedule_identity']=O.digest({k:v for k,v in schedule.items() if k not in ('created','schedule_identity')})
    C.write_json(OUT/'SCHEDULE.json',schedule)
    lookup=lambda name,out,stats:dict(build['programs']['designated' if name==reg['designated']['variant'] else name])
    frozen=C.prepare(OUT/'SCHEDULE.json',OUT,ai_test_root(),builders=(lookup,None),registry=reg)
    plan={'version':'ctf-host-decision-pilot-plan-1','maps':336,'maps_per_count':24,'counts':list(range(2,16)),'seed':seed,'workers':pre['selected_workers'],'repeats':REPEATS,'alternatives':ALTERNATIVES,'max_continuations':336*4*4,'limits_ms':[2000,50],'prefix_replay_limit_ms':5000,'policy':'admitted P2 XXL / V off / v12_ponly_xxl_dlog log-only, its distinct host binary identity','root_selection':'each count has four requested maps per category (carrier, flag, contact, interception, defense, quiet); among observable TURNs t>=6 with a valid scored logged incumbent (K1 allowed), pick minimum SHA hash in the requested category; otherwise deterministic category rotation fallback. No final score, future reply or realized successor selects a root. Retain all stratum opportunities, rejected log reasons, actual logged K and collected K, shortages and fallback.','root_categories':list(CATEGORIES),'category_definitions':{'carrier':'a living own unit holds a flag','flag':'ground flag within Manhattan4 of a living own unit','contact':'living enemy within Manhattan4 of a living own unit','interception':'living enemy carrier within Manhattan8 of a living own unit','defense':'living enemy within Manhattan6 of own base','quiet':'none of the preceding opportunities'},'continuation_policy':'common recorded opponent root actions, live/reactive after root; fresh process full-prefix replay; forced action does not rewrite carrier memory; repeats retain original flag stream and are not IID new maps','expansion_and_stop_rule':'run every declared map and selected candidate four times; no expansion or early favorable stop. Preserve faults/errors/partial rows. A stopped job is not restarted automatically. Resume verifies existing byte hashes and only attempts missing stages.','split_rule':'training.decision_dataset.split_of_map; all candidates and repeats of one map remain in its split','frozen_identity':frozen['identity'],'preflight_sha256':O.sha256_file(OUT/'PREFLIGHT_FINAL.json'),'build_receipt_sha256':O.sha256_file(OUT/'BUILD_ALL.json'),'source_files':source_snapshot(),'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    plan['identity']=O.digest(plan)
    C.write_json(OUT/'PLAN.json',plan)
    C.write_json(DATA/'PLAN.json',plan)
    print(json.dumps({'plan_identity':plan['identity'],'workers':plan['workers'],'maps':336,'seed':seed}),flush=True)


def match_job(job):
    fx,frozen,plan=job
    path=C.match_path(DATA,fx)
    receipt_path=DATA/'rows'/f"{fx['fixture_id']}.json"
    if path.exists() or receipt_path.exists():
        assert path.exists() and receipt_path.exists(),'incomplete match files retained; manual verification required'
        row=json.loads(receipt_path.read_text())
        assert row['plan_identity']==plan['identity'] and row['match_sha256']==O.sha256_file(path)
        return {'stage':'match','id':fx['fixture_id'],'reused':True}
    commands,info=C.commands_and_info(frozen)
    result=C.play_fixture((fx,commands,info,str(ai_test_root()),frozen['pool_identity'],None))
    if 'error' in result:
        C.write_json(DATA/'errors'/f"{fx['fixture_id']}.json",{'fixture':fx,'plan_identity':plan['identity'],'error':result['error']})
        return {'stage':'match','id':fx['fixture_id'],'error':True}
    match=result['match']
    matchlog.save(path,match)
    scores=match['result']['score']
    row={'schema':C.SCHEMA_ROW,'identity':frozen['identity'],'plan_identity':plan['identity'],'fixture_id':fx['fixture_id'],'teams':fx['teams'],'map_seed':fx['map_seed'],'seat':fx['seat'],'match_sha256':O.sha256_file(path),'score':scores,'outcome':float(pairwise_outcome(scores)[fx['seat']]),'faults':len(match['faults'])}
    C.write_json(receipt_path,row)
    return {'stage':'match','id':fx['fixture_id'],'faults':row['faults'],'reused':False}


def categories(init,block,seat):
    parsed=parse_block(block)
    own=[u for u in parsed.units if u[1]==seat and u[4]>0 and u[2]>=0]
    enemy=[u for u in parsed.units if u[1]!=seat and u[4]>0 and u[2]>=0]
    distance=lambda a,b:abs(a[0]-b[0])+abs(a[1]-b[1])
    lines=init.splitlines();idx=next(i for i,s in enumerate(lines) if s.startswith('BASES '))
    base=tuple(map(int,lines[idx+1+seat].split()))
    flags=[]
    for line in block.splitlines():
        tokens=line.split()
        if tokens and tokens[0]=='F' and tokens[2]=='ground':flags.append((int(tokens[3]),int(tokens[4])))
    active=[]
    if any(u[5]>=0 for u in own):active.append('carrier')
    if any(distance(u[2:4],f)<=4 for u in own for f in flags):active.append('flag')
    if any(distance(u[2:4],v[2:4])<=4 for u in own for v in enemy):active.append('contact')
    if any(v[5]>=0 and distance(u[2:4],v[2:4])<=8 for u in own for v in enemy):active.append('interception')
    if any(distance(v[2:4],base)<=6 for v in enemy):active.append('defense')
    return active or ['quiet']


def scored_with_k1(log):
    scored,reason=B.deciding_scores(log)
    if reason=='fewer than two scored orders (K = 1)':
        scale='refined' if log['stage']=='refine' else 'beam'
        entries=[dict(e,score=e[scale]) for e in log['entries'] if e[scale] is not None]
        if entries and entries[0]['order']==log['entries'][0]['order']:
            scored={'scale':scale,'stage':log['stage'],'checked':log['checked'],'entries':entries};reason=None
    return scored,reason


def select_root(match,requested):
    seat=B.designated_seat(match);eligible=[];opportunities=Counter();rejected=Counter()
    for turn,block in enumerate(match['blocks'],1):
        if turn<6:continue
        strata=categories(match['init'],block,seat)
        opportunities.update(strata)
        try:log=B.parse_log(match['replies'][turn-1][seat],turn)
        except (ValueError,KeyError) as exc:rejected['malformed log:'+str(exc)[:80]]+=1;continue
        if not log:rejected['no shortlist log']+=1;continue
        if log['root']!=B.root_digest(block):rejected['root digest mismatch']+=1;continue
        scored,why=scored_with_k1(log)
        if scored is None:rejected[why]+=1;continue
        eligible.append({'turn':turn,'log':log,'scored':scored,'strata':strata,'key':O.digest([match['match_id'],'host-observable-root',turn])})
    offset=CATEGORIES.index(requested)
    order=CATEGORIES[offset:]+CATEGORIES[:offset]
    chosen=None;selected=None
    for category in order:
        possible=[root for root in eligible if category in root['strata']]
        if possible:chosen=min(possible,key=lambda root:root['key']);selected=category;break
    selection={'requested_stratum':requested,'selected_stratum':selected,'fallback':selected is not None and selected!=requested,'opportunity_turns':dict(opportunities),'eligible_turns':dict(Counter(c for root in eligible for c in root['strata'])),'rejected':dict(rejected),'total_turns':len(match['blocks']),'eligible_root_count':len(eligible),'uses_final_outcome':False}
    return chosen,selection


def continuation(job):
    match_path,root_path,candidate,repeat,plan=job
    root=load_gz(root_path)
    path=DATA/'continuations'/f"n{root['teams']:02d}"/f"{root['match_id']}-t{root['turn']:03d}-c{candidate}-r{repeat}.json.gz"
    receipt=path.with_suffix('.sha256.json')
    if path.exists() or receipt.exists():
        assert path.exists() and receipt.exists(),'partial continuation receipt retained; manual verification required'
        check=json.loads(receipt.read_text());assert check['plan_identity']==plan['identity'] and check['sha256']==O.sha256_file(path)
        return {'stage':'continuation','id':path.name,'reused':True,'partial':not check['success']}
    match=matchlog.load(match_path);seat=root['seat'];turn=root['turn']
    frozen=json.loads((OUT/'FROZEN.json').read_text());commands,_=C.commands_and_info(frozen)
    engine=load_engine(ai_test_root());game=engine.Game(match['teams'],match['map_seed'],flag_seed=match['flag_seed'])
    bots=[runner.Bot(commands['@candidate' if i==seat else s['name']],s['name'],(2.0,B.PREFIX_LIMIT)) for i,s in enumerate(match['seats'])]
    blocks=[];computed=[];used=[];faults=[];mismatch=raw_mismatch=0;examples=[];init=None;root_computed=None;root_block=None;next_block=None;partial=None;score=None
    try:
        while not game.done:
            t=game.turn+1;inputs=[game.encode(i) for i in range(match['teams'])]
            if t==1:init,block=matchlog.split_init(inputs[0])
            else:block=inputs[0]
            blocks.append(block)
            if t<=turn:assert block==match['blocks'][t-1],f'engine recorded-prefix mismatch at {t}'
            if t==turn:
                root_block=block
                for bot in bots:bot.turn_limit=.05;bot.strikes=0
            if t==turn+1:next_block=block
            starts=[bot.send(text) for bot,text in zip(bots,inputs)]
            lines=[bot.receive(start,t) for bot,start in zip(bots,starts)]
            computed.append(lines)
            phase='prefix' if t<turn else 'root_discarded' if t==turn else 'continuation'
            for i,(bot,line) in enumerate(zip(bots,lines)):
                reasons=[reason for tt,reason in bot.faults if tt==t]
                if line is None and not reasons:reasons.append(f'no reply ({bot.dead})')
                if line is not None:
                    try:game.decode(i,line)
                    except ValueError as exc:
                        reasons.append('malformed output:'+str(exc))
                        if t>turn:bot.strike(t,reasons[-1])
                faults += [{'turn':t,'seat':i,'phase':phase,'reason':reason} for reason in reasons]
            if t<turn:
                for i,(got,rec) in enumerate(zip(lines,match['replies'][t-1])):
                    raw_mismatch += got!=rec
                    if B.order_of(got)!=B.order_of(rec):
                        mismatch+=1
                        if len(examples)<5:examples.append({'turn':t,'seat':i,'recorded':rec,'replayed':got})
                played=list(match['replies'][t-1]);orders=recorded_orders(game,match,t)
            else:
                if t==turn:
                    root_computed=list(lines);played=list(match['replies'][t-1]);played[seat]=root['candidates'][candidate]
                else:played=list(lines)
                orders={}
                for i,(bot,line) in enumerate(zip(bots,played)):
                    if line is None:orders[i]={};continue
                    try:orders[i]=game.decode(i,line);bot.ok()
                    except ValueError:orders[i]={}
            used.append(played)
            game.step(orders)
        score=game.result()['score']
    except BaseException as exc:partial={'error':str(exc),'traceback':traceback.format_exc()}
    finally:
        reports=[bot.report() for bot in bots]
        for bot in bots:bot.close()
    phase_counts=Counter(f['phase'] for f in faults)
    row={'schema':'ctf-host-decision-continuation-1','plan_identity':plan['identity'],'candidate':candidate,'repeat':repeat,'repeat_key':f"{match['match_id']}/t{turn}/c{candidate}/r{repeat}",'on_policy':candidate==0,'source':root['sources'][candidate],'line':root['candidates'][candidate],'score':score,'outcome':float(pairwise_outcome(score)[seat]) if score is not None else None,'prefix_mismatch':mismatch,'prefix_mismatch_raw':raw_mismatch,'prefix_examples':examples,'prefix_faults':phase_counts['prefix'],'faults':[[f['turn'],f['seat'],f['reason']] for f in faults if f['phase']=='continuation'],'discarded_root_computed_faults':[[f['turn'],f['seat'],f['reason']] for f in faults if f['phase']=='root_discarded'],'fault_counts':{'prefix':phase_counts['prefix'],'root_discarded':phase_counts['root_discarded'],'continuation':phase_counts['continuation']},'root_computed_replies':root_computed,'root_block':root_block,'next_block':next_block,'all_phase_faults':faults,'trajectory':{'init':init,'blocks':blocks,'computed_replies':computed,'played_replies':used,'final_block':game.encode(0) if score is not None else None,'bots':reports},'input_binding':root['input_binding'],'bot_state_mechanism':B.BOT_STATE,'rng':B.RNG_NOTE,'partial':partial}
    save_gz(path,row);C.write_json(receipt,{'sha256':O.sha256_file(path),'plan_identity':plan['identity'],'input_binding':root['input_binding'],'success':partial is None})
    return {'stage':'continuation','id':path.name,'reused':False,'partial':partial is not None,'faults':dict(phase_counts)}


def run():
    plan,frozen=verify_plan()
    schedule=json.loads((OUT/'SCHEDULE.json').read_text())
    state={'plan_identity':plan['identity'],'stage':'matches','scheduled_maps':336,'workers':plan['workers'],'matches':0,'continuations':0,'errors':[]}
    C.write_json(OUT/'state.json',state)
    with ProcessPoolExecutor(plan['workers']) as pool:
        for result in pool.map(match_job,[(fx,frozen,plan) for fx in schedule['fixtures']],chunksize=1):
            state['matches']+=1
            if result.get('error'):state['errors'].append(result)
            C.write_json(OUT/'state.json',state)
            print('match',state['matches'],'/336',result,flush=True)
    verify_plan()
    jobs=[];root_paths=[];selection_ledger=[]
    for index,fx in enumerate(schedule['fixtures']):
        mp=C.match_path(DATA,fx)
        if not mp.exists():continue
        match=matchlog.load(mp);seat=B.designated_seat(match);B.check_sources(match,seat,frozen)
        requested=CATEGORIES[(index%24)%len(CATEGORIES)]
        chosen,selection=select_root(match,requested)
        selection_ledger.append({'match_id':match['match_id'],'map_seed':match['map_seed'],'teams':match['teams'],**selection})
        if chosen is None:
            C.write_json(DATA/'drops'/f"{match['match_id']}.json",{'match_sha256':O.sha256_file(mp),'selection':selection,'reason':'no scored logged incumbent','plan_identity':plan['identity']})
            continue
        cands=B.select_candidates(chosen['scored'],ALTERNATIVES);t=chosen['turn']
        binding=B.input_binding(match,seat,t,chosen,cands)
        row={'schema':B.SCHEMA,'match_id':match['match_id'],'teams':match['teams'],'map_seed':match['map_seed'],'seat':seat,'turn':t,'init':match['init'],'carrier_sha256':match['seats'][seat]['sha256'],'plan_identity':plan['identity'],'collector_identity':O.digest([B.collector_identity(frozen,REPEATS,ALTERNATIVES,'official_50ms'),plan['identity']]),'frozen_identity':frozen['identity'],'input_binding':binding,'candidates':[e['line'] for e in cands],'sources':['recorded']+[f"shortlist:{e['source']}" for e in cands[1:]],'candidate_mode':'shortlist','root_outcome':float(pairwise_outcome(match['result']['score'])[seat]),'carrier_scores':{'version':B.SCORES_VERSION,'k':len(cands),'orders':[e['line'] for e in cands],'scores':[e['score'] for e in cands],'scale':chosen['scored']['scale'],'stage':chosen['scored']['stage'],'checked':chosen['scored']['checked'],'root_digest':chosen['log']['root']},'shortlist':chosen['log'],'logged_shortlist_k':chosen['log']['k'],'collected_k':len(cands),'selection':selection,'repeats':REPEATS,'bot_state_mechanism':B.BOT_STATE,'rng':B.RNG_NOTE,'continuation_policy':'official_50ms'}
        rp=DATA/'roots'/f"n{match['teams']:02d}"/f"{match['match_id']}-t{t:03d}.json.gz"
        if rp.exists():assert load_gz(rp)==row,'selected root input changed'
        else:save_gz(rp,row)
        root_paths.append((rp,mp))
        for candidate in range(len(cands)):
            for repeat in range(REPEATS):jobs.append((str(mp),str(rp),candidate,repeat,plan))
    C.write_json(DATA/'SELECTION_LEDGER.json',selection_ledger)
    state.update(stage='continuations',selected_roots=len(root_paths),scheduled_continuations=len(jobs))
    C.write_json(OUT/'state.json',state)
    with ProcessPoolExecutor(plan['workers']) as pool:
        for result in pool.map(continuation,jobs,chunksize=1):
            state['continuations']+=1
            if result.get('partial'):state['errors'].append(result)
            C.write_json(OUT/'state.json',state)
            if state['continuations']%16==0:print('continuations',state['continuations'],'/',len(jobs),flush=True)
    verify_plan()
    for rp,mp in root_paths:
        root=load_gz(rp);branches=[]
        for c in range(len(root['candidates'])):
            for repeat in range(REPEATS):
                cp=DATA/'continuations'/f"n{root['teams']:02d}"/f"{root['match_id']}-t{root['turn']:03d}-c{c}-r{repeat}.json.gz"
                branch=load_gz(cp)
                if branch['partial'] is None:branches.append({k:v for k,v in branch.items() if k!='trajectory'}|{'trajectory_path':C.rel(cp),'trajectory_sha256':O.sha256_file(cp)})
        out=B.row_path(DATA/'branches',matchlog.load(mp),root['turn'])
        save_gz(out,root|{'branches':branches,'partial_repeat_count':len(root['candidates'])*REPEATS-len(branches)})
    state['stage']='collection_complete';state['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    C.write_json(OUT/'state.json',state)
    C.write_json(OUT/'COLLECTION_COMPLETE.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',nargs='?',default='collect',choices=['prepare','collect']);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.seed)
    else:run()
