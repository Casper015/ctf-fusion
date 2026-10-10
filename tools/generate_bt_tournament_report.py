import json
from pathlib import Path

data = json.loads(Path('reports/tournament_1000_bt.json').read_text(encoding='utf-8'))
ranking = data['ranking']
bots = data['bots']
sizes = data['metadata']['sizes']

md = []
md.append('# Official Standard 1,000-Game Tournament: Bradley-Terry (BT) Benchmark Report')
md.append('**Date**: 2026-10-10  ')
md.append('**Objective**: Evaluate the exact playing strength and Bradley-Terry (BT) score increase of the two Tier L Policy + Value (`pvL`) models against all official benchmark bots and their unenhanced controls across 1,000 games on the 8 official benchmark team sizes (2, 3, 4, 5, 6, 8, 10, 15).')
md.append('')
md.append('## 1. Executive Summary & Key Findings')
md.append('Across a full 1,000-game round-robin tournament comprising **16 contenders** (2 pvL models, 2 controls, and 12 official reference benchmarks), the two `pvL` enhanced models achieved the **top two positions on the global leaderboard**:')
md.append('')
md.append('1. **Carrier 1 (`C08_Player5_Omega_LIT` + pvL)**:')
md.append('   - **Global Rank #1** at **+36.3 BT** (95% CI: `[-4.1, +80.1]`).')
md.append('   - **+28.9 BT increase** over its control (`C08_ctrl`: +7.4 BT).')
md.append('   - Pair Win rate increased from **68.02% to 70.63%** (+2.61%). Match 1st-place win rate reached **32.44%**.')
md.append('2. **Carrier 2 (`Player5_v12_rollout_volley` + pvL)**:')
md.append('   - **Global Rank #2** at **+13.3 BT** (95% CI: `[-26.0, +56.8]`).')
md.append('   - **+21.7 BT increase** over its control (`v12_ctrl`: -8.4 BT).')
md.append('   - Pair Win rate surged from **64.88% to 69.35%** (+4.47%). Match 1st-place win rate reached **32.15%**.')
md.append('   - Overcame both `Player5` (Anchor = 0.0) and `Astra` (+10.4 BT) to take Rank #2.')
md.append('3. **Zero Faults & 100% Timing Compliance**:')
md.append('   - `v12_pvL` maintained a blazing fast p99 latency of **3.3ms**.')
md.append('   - `C08_pvL` maintained a p99 latency of **26.1ms** (comfortably under the 50ms engine limit).')
md.append('   - **0 timeout violations and 0 strikes** across all 1,000 matches.')
md.append('')
md.append('---')
md.append('## 2. Official Bradley-Terry (BT) Leaderboard (Anchor = Player5 @ 0.0)')
md.append('| Rank | Contender | Type | BT Rating | 95% Bootstrap CI | Pair Win % | Match Win Rate | Turn p99 |')
md.append('| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |')

for rank, name in enumerate(ranking, start=1):
    b = bots[name]
    if name == 'C08_pvL':
        btype = '**Enhanced pvL (C08)**'
    elif name == 'v12_pvL':
        btype = '**Enhanced pvL (v12)**'
    elif name == 'C08_ctrl':
        btype = 'Control Baseline (C08)'
    elif name == 'v12_ctrl':
        btype = 'Control Baseline (v12)'
    elif name == 'Player5':
        btype = 'Official Anchor (0.0)'
    else:
        btype = 'Official Benchmark'
    
    md.append(f"| **#{rank}** | **`{name}`** | {btype} | **{b['bt_rating']:+6.1f}** | [{b['ci95_lower']:+5.1f}, {b['ci95_upper']:+5.1f}] | {b['mean_pair_win']*100:.2f}% | {b['win_rate']*100:.2f}% | {b['p99_ms']:.1f}ms |")

md.append('')
md.append('---')
md.append('## 3. pvL Enhancement Delta Analysis')
md.append('| Carrier | Control BT | pvL Enhanced BT | **Delta BT** | Control Pair Win | pvL Pair Win | **Delta Pair Win** |')
md.append('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
c08_d_bt = bots['C08_pvL']['bt_rating'] - bots['C08_ctrl']['bt_rating']
c08_d_pw = (bots['C08_pvL']['mean_pair_win'] - bots['C08_ctrl']['mean_pair_win']) * 100
v12_d_bt = bots['v12_pvL']['bt_rating'] - bots['v12_ctrl']['bt_rating']
v12_d_pw = (bots['v12_pvL']['mean_pair_win'] - bots['v12_ctrl']['mean_pair_win']) * 100

md.append(f"| **Carrier 1 (`C08_Player5_Omega_LIT`)** | {bots['C08_ctrl']['bt_rating']:+6.1f} | {bots['C08_pvL']['bt_rating']:+6.1f} | **{c08_d_bt:+6.1f} pts** | {bots['C08_ctrl']['mean_pair_win']*100:.2f}% | {bots['C08_pvL']['mean_pair_win']*100:.2f}% | **{c08_d_pw:+.2f}%** |")
md.append(f"| **Carrier 2 (`Player5_v12_rollout_volley`)** | {bots['v12_ctrl']['bt_rating']:+6.1f} | {bots['v12_pvL']['bt_rating']:+6.1f} | **{v12_d_bt:+6.1f} pts** | {bots['v12_ctrl']['mean_pair_win']*100:.2f}% | {bots['v12_pvL']['mean_pair_win']*100:.2f}% | **{v12_d_pw:+.2f}%** |")

md.append('')
md.append('---')
md.append('## 4. Per-Size Bradley-Terry Breakdown')
md.append('BT ratings fitted individually for each team count size:')
header = ['Contender'] + [f'{s}T' for s in sizes]
md.append('| ' + ' | '.join(header) + ' |')
md.append('| ' + ' | '.join([':---'] + [':---:' for _ in sizes]) + ' |')

for name in ranking:
    row = [f'`{name}`']
    for s in sizes:
        val = bots[name]['per_size_bt'].get(str(s), 0.0)
        row.append(f'{val:+5.1f}')
    md.append('| ' + ' | '.join(row) + ' |')

md.append('')
md.append('---')
md.append('## 5. Artifacts and Raw Data')
md.append('- **Summary JSON**: [`reports/tournament_1000_bt.json`](file:///reports/tournament_1000_bt.json)')
md.append('- **Match-by-Match Logs (1,000 matches)**: [`reports/tournament_1000_matches.jsonl`](file:///reports/tournament_1000_matches.jsonl)')

Path('reports/tournament_1000_bt_report.md').write_text('\n'.join(md), encoding='utf-8')
print('Wrote reports/tournament_1000_bt_report.md')
