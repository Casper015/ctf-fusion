import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load distance matrix data
repo = Path('C:/Users/caspe/OneDrive/Code/ctf-fusion')
dist_path = repo / 'data/roster/behavior_distance.json'
with open(dist_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

ids = data['ids']
overall = np.array(data['overall'], dtype=float)

# 1. Filter judge bots
judge_keys = [b for b in ids if 'judge/ctf/bots' in b]
# Standard clean names for display
clean_names = {
    'deepseek-v4.1-flash.cpp': 'DeepSeek-V4.1',
    'doubao-seed-2.1-turbo.cpp': 'Doubao-Seed-2.1',
    'fable-5.1.cpp': 'Fable-5.1',
    'glm-5.3.cpp': 'GLM-5.3',
    'gpt-6-astra.cpp': 'GPT-6-Astra',
    'gpt-6.1-sol.cpp': 'GPT-6.1-Sol',
    'kimi-k3.cpp': 'Kimi-K3',
    'mimo-v2.6-pro.cpp': 'MiMo-V2.6',
    'opus-5.5.cpp': 'Opus-5.5',
    'qwen-3.8-max.cpp': 'Qwen-3.8',
    'sonnet-5.5.cpp': 'Sonnet-5.5'
}

judge_names = [clean_names.get(b.split('/')[-1], b.split('/')[-1]) for b in judge_keys]
judge_indices = [ids.index(b) for b in judge_keys]
sub_matrix = overall[np.ix_(judge_indices, judge_indices)]

# Plot 1: Judge Bots Matrix
fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
cax = ax.imshow(sub_matrix, cmap='YlGnBu', vmin=0, vmax=0.5)

# Ticks and labels
ax.set_xticks(np.arange(len(judge_names)))
ax.set_yticks(np.arange(len(judge_names)))
ax.set_xticklabels(judge_names, rotation=45, ha='right', fontsize=11, fontweight='medium')
ax.set_yticklabels(judge_names, fontsize=11, fontweight='medium')

# Annotations inside cells
for i in range(len(judge_names)):
    for j in range(len(judge_names)):
        val = sub_matrix[i, j]
        color = 'white' if val > 0.32 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9.5,
                fontweight='bold' if (0 < val <= 0.25) else 'normal')

# Colorbar
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Behavioral Distance (Disagreement Rate)', fontsize=11, labelpad=10)

plt.title('CTF-Fusion: Pairwise Behavioral Distance Matrix\nOfficial LLM Judge Bots (42 Probe Scenarios, 14 Team Sizes)',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()

reports_dir = repo / 'reports'
reports_dir.mkdir(parents=True, exist_ok=True)
judge_img_path = reports_dir / 'judge_behavior_matrix.png'
plt.savefig(judge_img_path, dpi=300)
plt.close()
print(f'Saved: {judge_img_path}')

# Plot 2: All 87 Clusters Representative Matrix
# Select one representative from each cluster
clusters = data['clusters']
cluster_reps = [c[0] for c in clusters]
rep_indices = [ids.index(r) for r in cluster_reps]
cluster_matrix = overall[np.ix_(rep_indices, rep_indices)]

fig2, ax2 = plt.subplots(figsize=(12, 10.5), dpi=300)
cax2 = ax2.imshow(cluster_matrix, cmap='magma', vmin=0, vmax=0.6)
cbar2 = fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Pairwise Excess Distance', fontsize=11, labelpad=10)

ax2.set_title('CTF-Fusion: Macro Behavioral Distance Heatmap\nAcross 87 Distinct Behavior Groups (122 Eligible Models)',
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Behavior Group Index (1 to 87)', fontsize=11, labelpad=8)
ax2.set_ylabel('Behavior Group Index (1 to 87)', fontsize=11, labelpad=8)

plt.tight_layout()
cluster_img_path = reports_dir / 'all_clusters_behavior_matrix.png'
plt.savefig(cluster_img_path, dpi=300)
plt.close()
print(f'Saved: {cluster_img_path}')
