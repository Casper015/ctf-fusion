import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

repo = Path('C:/Users/caspe/OneDrive/Code/ctf-fusion')
dist_path = repo / 'data/roster/behavior_distance.json'
with open(dist_path, 'r', encoding='utf-8') as f:
    bd = json.load(f)

ids = bd['ids']
D = np.array(bd['overall'], dtype=float)
n = len(D)

# 1. Classical MDS (PCoA)
D2 = D ** 2
H = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * H @ D2 @ H
evals, evecs = np.linalg.eigh(B)

idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]
coords = evecs[:, :2] * np.sqrt(np.maximum(evals[:2], 0))

# 2. Categorize bots into 6 distinct behavioral & lineage groups
def get_category(bid):
    stem = bid.split('/')[-1]
    if 'judge/ctf/bots' in bid:
        return 'LLM Judge Bots'
    if any(k in stem for k in ['Omega', 'EXPECTIMAX', 'Astra_Full', 'Astra_Carrier', 'Astra_Joint']):
        return 'Omega & Search Tier'
    if 'c_series' in bid or stem.startswith('C0') or stem.startswith('C1') or stem.startswith('C2'):
        return 'C-Series Gene Mutants'
    if 's1-controls' in bid or stem.startswith('S1_') or stem.startswith('S2_'):
        return 'S1/S2 Controls'
    if any(k in stem for k in ['baseline', 'random_bot', 'noise_bot']):
        return 'Baselines & Random'
    return 'Candidate Bots (P2-P5)'

cats = [get_category(b) for b in ids]

cat_colors = {
    'LLM Judge Bots': '#E63946',         # Crimson Red
    'Omega & Search Tier': '#1D3557',     # Deep Navy Blue
    'C-Series Gene Mutants': '#457B9D',   # Muted Blue
    'S1/S2 Controls': '#2A9D8F',          # Teal Green
    'Candidate Bots (P2-P5)': '#F4A261',  # Warm Sand/Orange
    'Baselines & Random': '#9E9E9E'       # Neutral Gray
}

cat_markers = {
    'LLM Judge Bots': '*',
    'Omega & Search Tier': 's',
    'C-Series Gene Mutants': 'o',
    'S1/S2 Controls': '^',
    'Candidate Bots (P2-P5)': 'v',
    'Baselines & Random': 'X'
}

cat_sizes = {
    'LLM Judge Bots': 180,
    'Omega & Search Tier': 70,
    'C-Series Gene Mutants': 45,
    'S1/S2 Controls': 65,
    'Candidate Bots (P2-P5)': 50,
    'Baselines & Random': 80
}

# 3. Plotting
fig, ax = plt.subplots(figsize=(15, 11), dpi=300)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#FFFFFF')
ax.grid(True, linestyle='--', alpha=0.5, color='#D0D7DE')

# Plot non-judge bots first
for category in ['Baselines & Random', 'Candidate Bots (P2-P5)', 'C-Series Gene Mutants', 'S1/S2 Controls', 'Omega & Search Tier']:
    mask = [c == category for c in cats]
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=cat_colors[category], marker=cat_markers[category],
               s=cat_sizes[category], alpha=0.75, edgecolors='white', linewidth=0.6,
               label=category, zorder=3)

# Plot LLM Judge Bots on top
mask_judge = [c == 'LLM Judge Bots' for c in cats]
ax.scatter(coords[mask_judge, 0], coords[mask_judge, 1],
           c=cat_colors['LLM Judge Bots'], marker='*',
           s=cat_sizes['LLM Judge Bots'], alpha=0.95, edgecolors='#780016', linewidth=1.2,
           label='LLM Judge Bots', zorder=5)

# Label Clean Names for Judge bots with separated offsets to prevent collision
judge_labels = {
    'deepseek-v4.1-flash.cpp': ('DeepSeek-v4.1', (15, -12)),
    'doubao-seed-2.1-turbo.cpp': ('Doubao-2.1 (Defensive)', (15, -15)),
    'fable-5.1.cpp': ('Fable-5.1', (12, -12)),
    'glm-5.3.cpp': ('GLM-5.3 (Defensive)', (-120, -14)),
    'gpt-6-astra.cpp': ('GPT-6-Astra', (-85, 16)),
    'gpt-6.1-sol.cpp': ('GPT-6.1-Sol', (18, -6)),
    'kimi-k3.cpp': ('Kimi-k3', (-65, -20)),
    'mimo-v2.6-pro.cpp': ('MiMo-v2.6', (12, 14)),
    'opus-5.5.cpp': ('Opus-5.5', (14, 8)),
    'qwen-3.8-max.cpp': ('Qwen-3.8', (-75, 5)),
    'sonnet-5.5.cpp': ('Sonnet-5.5', (14, 16))
}

for i, bid in enumerate(ids):
    stem = bid.split('/')[-1]
    if stem in judge_labels:
        lbl, offset = judge_labels[stem]
        x, y = coords[i, 0], coords[i, 1]
        ax.annotate(lbl, (x, y), xytext=offset, textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#111111',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF3CD', edgecolor='#FFEBAA', alpha=0.92),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.08', color='#B02A37', lw=1.2),
                    zorder=6)

# Annotate closest stylistic connections between LLM Judge bots
closest_judge_pairs = [
    ('sonnet-5.5.cpp', 'opus-5.5.cpp', 'd=0.203 (Claude Cluster)'),
    ('qwen-3.8-max.cpp', 'mimo-v2.6-pro.cpp', 'd=0.243 (APAC Cluster)'),
    ('gpt-6-astra.cpp', 'gpt-6.1-sol.cpp', 'd=0.250 (GPT Cluster)')
]

for b1, b2, text in closest_judge_pairs:
    i1 = [ids.index(b) for b in ids if b1 in b][0]
    i2 = [ids.index(b) for b in ids if b2 in b][0]
    x1, y1 = coords[i1, 0], coords[i1, 1]
    x2, y2 = coords[i2, 0], coords[i2, 1]
    ax.plot([x1, x2], [y1, y2], linestyle=':', color='#E63946', linewidth=2.0, alpha=0.85, zorder=4)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my + 0.012, text, fontsize=8.5, fontweight='bold', color='#780016',
            ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', edgecolor='#E63946', alpha=0.9),
            zorder=6)

# Label key structural anchors (exact ID matching to avoid duplicates)
anchor_ids = {
    'ctf-arena/Player7/workspace/experiments/round3/export/Player5_Omega.cpp': ('Player5_Omega (Search Ceiling)', (15, 10)),
    'ctf-arena/Player7/workspace/experiments/round3/export/Control_NG_foundation.cpp': ('Foundation (11-Bot Cluster 7)', (-145, -18)),
    '@repo/tools/random_bot.py': ('Random Bot (Uniform)', (10, -10)),
    'New_AI_Test/Player0/judge/ctf/workspace/submission/bot.cpp': ('Baseline Bot (Naive)', (10, 5))
}
for bid, (lbl, offset) in anchor_ids.items():
    if bid in ids:
        i = ids.index(bid)
        x, y = coords[i, 0], coords[i, 1]
        ax.annotate(lbl, (x, y), xytext=offset, textcoords='offset points',
                    fontsize=9.5, fontweight='bold', color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#E9ECEF', edgecolor='#CED4DA', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#6C757D', lw=1.0),
                    zorder=5)

# Axis labels & Title
ax.set_title('CTF-Fusion: 2D Behavioral Style Space & Taxonomic Clusters (122 Models)\nMultidimensional Scaling (MDS) over Pairwise Action Disagreement on 42 Canonical Tactical Probes',
             fontsize=13.5, fontweight='bold', pad=18)
ax.set_xlabel('Style Dimension 1: Strategic Sophistication & Search Depth (Lookahead/Expectimax vs Simple Rules)',
              fontsize=11.5, fontweight='semibold', labelpad=10)
ax.set_ylabel('Style Dimension 2: Tactical Skirmish Posture (High-Mobility Aggression vs Conservative Base Defense)',
              fontsize=11.5, fontweight='semibold', labelpad=10)

ax.legend(loc='upper right', framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CED4DA', fontsize=10.5, shadow=True)

plt.tight_layout()
out_img = repo / 'roster' / 'report' / 'style_space_2d.png'
plt.savefig(out_img, dpi=300)
plt.close()
print(f'Saved: {out_img}')
