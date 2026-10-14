"""Offline collection completeness checks; never starts bots or replays games.

Run after COLLECTION_COMPLETE and helper exit. T0038 remains the independent
causal-feature, target-support and learnability audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def utility(scores,seat):
    own=scores[seat]
    return sum(1. if own>s else .5 if own==s else 0. for i,s in enumerate(scores) if i!=seat)/(len(scores)-1)


def verify(repo):
    sys.path.insert(0,str(repo))
    sys.path.insert(0,str(repo/'bots/p_only/report/decision_collection'))
    import host_collect as H
    from training import decision_collect as C,decision_branch_collect as B,decision_opponents as O
    from fusion.protocol import parse_block,parse_reply
    build=repo/'build/decision_host_20261014';data=repo/'data/decision/host_20261014'
    complete=read(build/'COLLECTION_COMPLETE.json')
    assert complete['stage']=='collection_complete'
    assert read(build/'collection_process/COMPLETE.json')['exit_code']==0
    plan,frozen=H.verify_plan()
    schedule=read(build/'SCHEDULE.json');ledger=read(data/'SELECTION_LEDGER.json')
    assert complete['plan_identity']==plan['identity']
    assert len(schedule['fixtures'])==336 and len(ledger)==336
    assert Counter(f['teams'] for f in schedule['fixtures'])==Counter({n:24 for n in range(2,16)})
    assert len({f['map_seed'] for f in schedule['fixtures']})==336
    counters=Counter();counts={str(n):Counter() for n in range(2,16)}
    requested=Counter();selected=Counter();opportunities=Counter();eligible=Counter();rejected=Counter();k=Counter();logged_k=Counter();seats={str(n):Counter() for n in range(2,16)}
    roots=list(data.glob('roots/n*/*.json.gz'));root_by_match={read(p)['match_id']:p for p in roots}
    assert len(root_by_match)==len(roots)
    expected_continuations=set();expected_branches=set();inventory={}
    strata_by_count={str(n):{name:Counter() for name in ('requested','selected','opportunities','eligible','collected_k','logged_k')} for n in range(2,16)}
    def bind(p):inventory[p.relative_to(repo).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
    for index,fx in enumerate(schedule['fixtures']):
        n=fx['teams'];seat=fx['seat'];mid=fx['fixture_id'];stratum=H.CATEGORIES[(index%24)%6]
        mp=C.match_path(data,fx);m=read(mp);mr=read(data/'rows'/f'{mid}.json')
        assert m['match_id']==mid and m['teams']==n and m['map_seed']==fx['map_seed'] and m['flag_seed']==fx['flag_seed']
        assert m['limits_ms']==[2000,50] and m['engine_sha256']==frozen['engine']['sha256']
        assert mr['match_sha256']==sha(mp) and mr['plan_identity']==plan['identity'] and mr['identity']==frozen['identity']
        names=list(fx['opponents']);names.insert(seat,'@candidate')
        assert [s['name'] for s in m['seats']]==names
        assert m['sampling']=={'regime':'decision-expanded','components':C.fixture_seats(fx)[1],'registry':frozen['pool_identity']}
        B.check_sources(m,seat,frozen)
        assert len(m['blocks'])==len(m['replies']) and all(len(row)==n for row in m['replies'])
        assert len(m['result']['score'])==n and parse_block(m['final_block']).score==m['result']['score']
        assert abs(utility(m['result']['score'],seat)-mr['outcome'])<1e-12
        chosen,selection=H.select_root(m,stratum)
        assert ledger[index]=={'match_id':mid,'map_seed':fx['map_seed'],'teams':n,**selection}
        requested[stratum]+=1;opportunities.update(selection['opportunity_turns']);eligible.update(selection['eligible_turns']);rejected.update(selection['rejected'])
        group=strata_by_count[str(n)];group['requested'][stratum]+=1;group['opportunities'].update(selection['opportunity_turns']);group['eligible'].update(selection['eligible_turns'])
        counters['matches']+=1;counts[str(n)]['maps']+=1;seats[str(n)][str(seat)]+=1
        counters['match_faults']+=len(m['faults']);counts[str(n)]['match_faults']+=len(m['faults']);counters['match_all_seat_reply_slots']+=len(m['blocks'])*n
        counters['match_missing_replies']+=sum(line is None for row in m['replies'] for line in row)
        bind(mp);bind(data/'rows'/f'{mid}.json')
        if chosen is None:
            assert mid not in root_by_match
            drop=data/'drops'/f'{mid}.json';assert read(drop)['match_sha256']==sha(mp);bind(drop);counters['root_drops']+=1;continue
        rp=root_by_match[mid];root=read(rp);cands=B.select_candidates(chosen['scored'],3)
        assert root['input_binding']==B.input_binding(m,seat,chosen['turn'],chosen,cands)
        assert root['candidates']==[e['line'] for e in cands] and root['carrier_scores']['scores']==[e['score'] for e in cands]
        assert root['carrier_scores']['k']==root['collected_k']==len(cands) and root['logged_shortlist_k']==chosen['log']['k']
        assert root['carrier_scores']['version']==B.SCORES_VERSION and root['carrier_scores']['orders']==root['candidates']
        assert all(math.isfinite(v) for v in root['carrier_scores']['scores'])
        assert root['repeats']==4 and root['candidate_mode']=='shortlist'
        assert root['shortlist']==chosen['log'] and root['selection']==selection and root['plan_identity']==plan['identity'] and root['frozen_identity']==frozen['identity']
        assert root['collector_identity']==O.digest([B.collector_identity(frozen,4,3,'official_50ms'),plan['identity']])
        assert root['bot_state_mechanism']==B.BOT_STATE and root['rng']==B.RNG_NOTE and root['continuation_policy']=='official_50ms'
        assert root['turn']==chosen['turn'] and root['seat']==seat and root['map_seed']==fx['map_seed'] and root['teams']==n
        assert root['carrier_sha256']==frozen['designated']['binary_sha256'] and root['root_outcome']==utility(m['result']['score'],seat)
        assert len(set(root['candidates']))==len(cands) and all(parse_reply(line,root['turn']) is not None for line in root['candidates'])
        t=root['turn'];selected[selection['selected_stratum']]+=1;counters['fallbacks']+=int(selection['fallback']);k[str(len(cands))]+=1;logged_k[str(root['logged_shortlist_k'])]+=1
        group['selected'][selection['selected_stratum']]+=1;group['collected_k'][str(len(cands))]+=1;group['logged_k'][str(root['logged_shortlist_k'])]+=1
        counts[str(n)]['fallbacks']+=int(selection['fallback'])
        counters['roots']+=1;counts[str(n)]['roots']+=1;bind(rp)
        bp=B.row_path(data/'branches',m,t);assembled=read(bp);bind(bp);expected_branches.add(bp)
        assert {key:value for key,value in assembled.items() if key not in ('branches','partial_repeat_count')}==root
        by_pair={(b['candidate'],b['repeat']):b for b in assembled['branches']};assert len(by_pair)==len(assembled['branches'])
        successful=0
        for candidate in range(len(cands)):
            for repeat in range(4):
                cp=data/'continuations'/f'n{n:02d}'/f'{mid}-t{t:03d}-c{candidate}-r{repeat}.json.gz'
                expected_continuations.add(cp);branch=read(cp);receipt=read(cp.with_suffix('.sha256.json'));bind(cp);bind(cp.with_suffix('.sha256.json'))
                assert receipt['sha256']==sha(cp) and receipt['plan_identity']==plan['identity'] and receipt['input_binding']==root['input_binding']
                assert branch['input_binding']==root['input_binding'] and branch['plan_identity']==plan['identity']
                assert branch['bot_state_mechanism']==root['bot_state_mechanism'] and branch['rng']==root['rng']
                assert branch['candidate']==candidate and branch['repeat']==repeat and branch['line']==root['candidates'][candidate] and branch['source']==root['sources'][candidate]
                assert branch['repeat_key']==f'{mid}/t{t}/c{candidate}/r{repeat}' and branch['on_policy']==(candidate==0)
                traj=branch['trajectory'];length=len(traj['blocks'])
                assert length==len(traj['computed_replies'])==len(traj['played_replies'])
                assert all(len(row)==n for key in ('computed_replies','played_replies') for row in traj[key])
                assert traj['init']==m['init'] and traj['blocks'][:t]==m['blocks'][:t]
                assert traj['played_replies'][:t-1]==m['replies'][:t-1]
                root_played=list(m['replies'][t-1]);root_played[seat]=root['candidates'][candidate]
                assert traj['played_replies'][t-1]==root_played
                assert traj['computed_replies'][t-1]==branch['root_computed_replies']
                old=m['replies'][:t-1];new=traj['computed_replies'][:t-1]
                assert branch['prefix_mismatch_raw']==sum(a!=b for row1,row2 in zip(old,new) for a,b in zip(row1,row2))
                assert branch['prefix_mismatch']==sum(B.order_of(a)!=B.order_of(b) for row1,row2 in zip(old,new) for a,b in zip(row1,row2))
                assert traj['played_replies'][t:]==traj['computed_replies'][t:]
                assert branch['root_block']==m['blocks'][t-1]
                if length>t:assert branch['next_block']==traj['blocks'][t]
                phase=Counter(f['phase'] for f in branch['all_phase_faults'])
                assert branch['fault_counts']=={name:phase[name] for name in ('prefix','root_discarded','continuation')}
                assert all(f['phase']==('prefix' if f['turn']<t else 'root_discarded' if f['turn']==t else 'continuation') and 0<=f['seat']<n for f in branch['all_phase_faults'])
                for name,amount in phase.items():
                    counters[f'{name}_faults']+=amount;counts[str(n)][f'{name}_faults']+=amount
                assert branch['prefix_faults']==phase['prefix']
                for key,name in (('faults','continuation'),('discarded_root_computed_faults','root_discarded')):
                    assert branch[key]==[[f['turn'],f['seat'],f['reason']] for f in branch['all_phase_faults'] if f['phase']==name]
                counters['prefix_order_mismatches']+=branch['prefix_mismatch'];counters['prefix_raw_mismatches']+=branch['prefix_mismatch_raw']
                counts[str(n)]['prefix_order_mismatches']+=branch['prefix_mismatch'];counts[str(n)]['prefix_raw_mismatches']+=branch['prefix_mismatch_raw']
                counters['continuation_reply_slots']+=length*n;counters['continuations']+=1;counts[str(n)]['continuations']+=1
                assert receipt['success']==(branch['partial'] is None)
                if branch['partial'] is None:
                    successful+=1;assert parse_block(traj['final_block']).score==branch['score']
                    assert abs(utility(branch['score'],seat)-branch['outcome'])<1e-12
                    assert by_pair[(candidate,repeat)]==({key:value for key,value in branch.items() if key!='trajectory'}|{'trajectory_path':C.rel(cp),'trajectory_sha256':sha(cp)})
                else:counters['partial_continuations']+=1;counts[str(n)]['partial_continuations']+=1
        assert assembled['partial_repeat_count']==len(cands)*4-successful and len(by_pair)==successful
        if (index+1)%24==0:print(f'verified {index+1}/336 maps',flush=True)
    assert set(data.glob('continuations/n*/*.json.gz'))==expected_continuations
    assert set(data.glob('branches/n*/*.json.gz'))==expected_branches
    assert complete['continuations']==counters['continuations']==complete['scheduled_continuations']
    for name in ('PLAN','SELECTION_LEDGER'):bind(data/f'{name}.json')
    assert read(data/'PLAN.json')==plan
    result={'schema':'ctf-host-delivery-verification-1','plan_identity':plan['identity'],'frozen_identity':frozen['identity'],'schedule_identity':schedule['schedule_identity'],'source_commit':plan['source_commit'],'verification_script_sha256':sha(Path(__file__)),'counts':dict(counters),'per_count':{n:dict(v) for n,v in counts.items()},'designated_seats':{n:dict(v) for n,v in seats.items()},'collected_k':dict(k),'logged_shortlist_k':dict(logged_k),'requested_strata':dict(requested),'selected_strata':dict(selected),'opportunity_turns':dict(opportunities),'eligible_turns':dict(eligible),'rejected_logs':dict(rejected),'raw_inventory':inventory,'scope':'complete raw hash/linkage, original prefix/root played orders, all-seat reply slots, final-block scores and independently recomputed pairwise utility; no bot execution or engine replay; independent causal features/target support/noise audit remains T0038'}
    result['strata_by_count']={n:{key:dict(value) for key,value in group.items()} for n,group in strata_by_count.items()}
    output=build/'DELIVERY_VERIFICATION.json';output.write_text(json.dumps(result,sort_keys=True,indent=1)+'\n',encoding='utf-8')
    print(json.dumps(result['counts'],sort_keys=True),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4]);args=ap.parse_args();verify(args.repo)
