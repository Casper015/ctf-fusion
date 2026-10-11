import json,pathlib,statistics,math
P=pathlib.Path.cwd();R=P/'bots/p_only/report/search_portfolio'
read=lambda p:json.loads(p.read_text())
a=read(R/'INDEPENDENT_AUDIT.json');s=read(R/'screen.json');rows=[json.loads(x) for x in (P/'data/p_only/search/results.jsonl').read_text().splitlines() if x.strip()]
for arm,g in s['arms'].items():
 rs=[r for r in rows if r['candidate']==arm];p99=sorted(r['p99_ms'] for r in rs if r['p99_ms'] is not None)
 measured={'games':len(rs),'utility':round(statistics.mean(r['utility'] for r in rs),5),'faults':sum(r['faults'] for r in rs),'dead_games':sum(bool(r['dead']) for r in rs),'late_replies':sum(r['late'] or 0 for r in rs),'first_ms_max':max(r['first_ms'] for r in rs if r['first_ms'] is not None),'p99_ms_median':p99[len(p99)//2],'p99_ms_worst':p99[-1],'max_ms':max(r['max_ms'] for r in rs if r['max_ms'] is not None)}
 for k,v in measured.items():assert v==g[k],(arm,k,v,g[k])
 c=a['counter_totals'][arm];settings=a['counter_settings'].get(arm,{});out=s['network_calls'][arm];turns=112000
 assert a['counter_line_counts'][arm]['PONLY']==280
 for k,raw in [('views_per_turn','builds'),('team_encodes_per_turn','teams'),('act_heads_per_turn','act'),('target_heads_per_turn','target')]:assert round(c['ponly.'+raw]/turns,2)==out[k]
 for k,raw in [('fallbacks','fallback'),('inconsistent','inconsistent'),('skipped','skipped')]:assert c['ponly.'+raw]==out[k]
 assert out['turns']==turns
 if 'search_calls' in out:
  assert a['counter_line_counts'][arm]['SEARCH']==260
  calls=c['search.calls'];assert calls==out['search_calls']
  for k,raw,prec in [('samples_per_search','samples',1),('requested_per_search','requested',1),('truncated_share','truncated',4),('rollouts_per_search','rollouts',1),('plans_per_search','plans',1),('survivors_per_search','survivors',2)]:assert round(c['search.'+raw]/calls,prec)==out[k]
  assert settings['search.fixed']==out['fixed_samples']
  p=out['ply2'];assert settings['ply2.mode']==p['mode'] and settings['ply2.cache']==p['cache']
  for k,raw,mul,prec in [('predictions_per_search','predictions',1,1),('prediction_ms_per_search','seconds',1000,3),('view_builds_per_search','view_builds',1,1)]:assert round(mul*c['ply2.'+raw]/calls,prec)==p[k]
  assert c['ply2.aborted_sweeps']==p['aborted_sweeps'] and c['ply2.discarded_predictions']==p['discarded_predictions']
  assert round(c['search.refine_cuts']/c['search.refines'],4)==out['refine_cut_share']
  assert round(c['ply2.view_hits']/c['ply2.view_calls'],4)==p['view_hit_share']
  assert round(c['ply2.memo_hits']/(c['ply2.memo_hits']+c['ply2.memo_misses']),4)==p['memo_hit_share']
 if 'portfolio' in out:
  po=out['portfolio'];assert settings['portfolio.k']==po['k'] and settings['portfolio.moves']==po['moves']
  for k,raw,den in [('assignments_per_call','assignments','calls'),('waits_per_call','waits','calls'),('searched_per_turn','searched','turns'),('refined_per_turn','refined','turns')]:
   denominator=c['portfolio.calls'] if den=='calls' else a['counter_line_counts'][arm]['PORTFOLIO']*400
   assert round(c['portfolio.'+raw]/denominator,2)==po[k]
  for k,raw in [('same_move_sets','same_move'),('cuts','cut'),('overkill_skipped','overkill_skipped')]:assert c['portfolio.'+raw]==po[k]
a['timing_and_primary_counter_aggregates_match']=True
(R/'INDEPENDENT_AUDIT.json').write_text(json.dumps(a,indent=1)+'\n',newline='\n')
print('Timing and primary counters independently match all eight arms.')
