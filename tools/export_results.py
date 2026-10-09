import csv
import gzip
import json
import os
import shutil
from pathlib import Path

repo = Path('C:/Users/caspe/OneDrive/Code/ctf-fusion')
results_dir = repo / 'roster' / 'report' / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

# 1. Copy raw matches into reports/results/matches
raw_probes_src = repo / 'data' / 'raw_probes'
matches_dst = results_dir / 'matches'
if matches_dst.exists():
    shutil.rmtree(matches_dst)
shutil.copytree(raw_probes_src, matches_dst)
print(f"Copied matches to {matches_dst}")

# 2. Generate matches_summary.csv
matches_csv = results_dir / 'matches_summary.csv'
with open(matches_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['match_id', 'teams', 'map_seed', 'flag_seed', 'duration_turns',
                     'seats', 'scores', 'winner_seat', 'winner_score', 'total_faults'])
    
    for gz_path in sorted(matches_dst.glob('n*/*.json.gz')):
        with gzip.open(gz_path, 'rt', encoding='utf-8') as gf:
            m = json.load(gf)
        scores = m.get('scores', [])
        winner_seat = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1
        winner_score = scores[winner_seat] if winner_seat >= 0 else 0
        faults_cnt = len(m.get('faults', []))
        seats_str = ";".join(s['name'] for s in m.get('seats', []))
        scores_str = ";".join(map(str, scores))
        writer.writerow([
            m['match_id'], m['teams'], m['map_seed'], m['flag_seed'],
            len(m.get('blocks', [])), seats_str, scores_str, winner_seat,
            winner_score, faults_cnt
        ])
print(f"Generated {matches_csv}")

# 3. Generate probe_suite.csv
suite_path = repo / 'data' / 'roster' / 'probe_suite.json'
with open(suite_path, 'r', encoding='utf-8') as f:
    suite = json.load(f)

probe_csv = results_dir / 'probe_suite.csv'
with open(probe_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    tags = suite.get('tags', [])
    writer.writerow(['probe_id', 'teams', 'map_seed', 'flag_seed', 'seat', 'turns'] + tags)
    for p in suite['probes']:
        row = [p['probe_id'], p['teams'], p['map_seed'], p['flag_seed'], p['seat'], p['turns']]
        for t in tags:
            row.append(p['tag_counts'].get(t, 0))
        writer.writerow(row)
print(f"Generated {probe_csv}")

# 4. Generate behavior_distance_matrix.csv
dist_path = repo / 'data' / 'roster' / 'behavior_distance.json'
with open(dist_path, 'r', encoding='utf-8') as f:
    dist = json.load(f)

matrix_csv = results_dir / 'behavior_distance_matrix.csv'
with open(matrix_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['bot_id'] + dist['ids'])
    for b, row in zip(dist['ids'], dist['overall']):
        writer.writerow([b] + [round(x, 4) for x in row])
print(f"Generated {matrix_csv}")
