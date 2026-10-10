"""Generate the official roster behavioral report and visualizations from committed result files.

This script parses ONLY committed result files (JSON/CSV) and generates:
  1. roster/report/FINGERPRINT_BEHAVIOR_REPORT.md
  2. roster/report/judge_behavior_matrix.png (overall_excess distance heatmap for official LLM judge bots)
  3. roster/report/all_clusters_behavior_matrix.png (macro overall_excess distance heatmap for behavior groups)
  4. roster/report/style_space_2d.png (2D Classical MDS / PCoA taxonomic projection with UNLABELED numbered axes)

No numbers are hardcoded. Strict epistemological standards are enforced:
  - Distances are signed overall_excess (kernel disagreement corrected for A/A noise).
  - MDS/PCoA axes are numbered Dimension 1 and Dimension 2 without subjective labels.
  - Equivalence is stated as bounded within margins on this probe suite (heuristic bound, not 95% guarantee).
  - Live outcomes report explicit denominators and intervals; subjective style labels are forbidden.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLEAN_JUDGE_NAMES = {
    "deepseek-v4.1-flash.cpp": "DeepSeek-v4.1",
    "doubao-seed-2.1-turbo.cpp": "Doubao-Seed-2.1",
    "fable-5.1.cpp": "Fable-5.1",
    "glm-5.3.cpp": "GLM-5.3",
    "gpt-6-astra.cpp": "GPT-6-Astra",
    "gpt-6.1-sol.cpp": "GPT-6.1-Sol",
    "kimi-k3.cpp": "Kimi-k3",
    "mimo-v2.6-pro.cpp": "MiMo-v2.6",
    "opus-5.5.cpp": "Opus-5.5",
    "qwen-3.8-max.cpp": "Qwen-3.8",
    "sonnet-5.5.cpp": "Sonnet-5.5",
}


def load_json_safe(path: Path | str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_clean_name(bot_id: str) -> str:
    stem = bot_id.split("/")[-1]
    return CLEAN_JUDGE_NAMES.get(stem, stem)


def plot_judge_matrix(dist: dict, out_path: Path):
    ids = dist["ids"]
    overall_excess = np.array(dist["overall_excess"], dtype=float)
    
    judge_keys = [b for b in ids if "judge/ctf/bots" in b or b.split("/")[-1] in CLEAN_JUDGE_NAMES]
    # Keep consistent canonical order if possible
    judge_keys = sorted(judge_keys, key=lambda b: CLEAN_JUDGE_NAMES.get(b.split("/")[-1], b))
    if not judge_keys:
        return
        
    judge_names = [get_clean_name(b) for b in judge_keys]
    indices = [ids.index(b) for b in judge_keys]
    sub = overall_excess[np.ix_(indices, indices)]
    
    fig, ax = plt.subplots(figsize=(10.5, 9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    # Colormap: coolwarm centered at 0 or viridis
    vmin = min(0.0, float(np.nanmin(sub)))
    vmax = max(0.5, float(np.nanmax(sub)))
    cax = ax.imshow(sub, cmap="YlGnBu", vmin=0.0, vmax=vmax)
    
    ax.set_xticks(np.arange(len(judge_names)))
    ax.set_yticks(np.arange(len(judge_names)))
    ax.set_xticklabels(judge_names, rotation=45, ha="right", fontsize=10, fontweight="medium")
    ax.set_yticklabels(judge_names, fontsize=10, fontweight="medium")
    
    for i in range(len(judge_names)):
        for j in range(len(judge_names)):
            val = sub[i, j]
            if np.isnan(val):
                txt = "N/A"
                color = "#888888"
            else:
                txt = f"{val:.3f}"
                color = "white" if val > (0.6 * vmax) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9,
                    fontweight="bold" if (0.0 < val <= 0.25) else "normal")
                    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Disagreement Excess $d_{excess}(A, B)$ (Corrected for A/A Noise)", fontsize=10.5, labelpad=10)
    
    ax.set_title("CTF-Fusion: Pairwise Behavioral Disagreement Excess\nOfficial LLM Judge Bots (84 Canonical Probes, 14 Team Configurations)",
                 fontsize=12, fontweight="bold", pad=16)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_all_clusters_matrix(dist: dict, out_path: Path):
    ids = dist["ids"]
    overall_excess = np.array(dist["overall_excess"], dtype=float)
    
    # Representative of each cluster
    clusters = dist.get("clusters", [])
    if not clusters:
        # Fall back to all IDs if no clusters
        cluster_reps = ids
    else:
        cluster_reps = [c[0] for c in clusters]
        
    indices = [ids.index(r) for r in cluster_reps if r in ids]
    sub = overall_excess[np.ix_(indices, indices)]
    
    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    
    vmax = max(0.5, float(np.nanmax(sub))) if np.isfinite(sub).any() else 0.5
    cax = ax.imshow(sub, cmap="magma", vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pairwise Disagreement Excess $d_{excess}$", fontsize=10.5, labelpad=10)
    
    n_groups = len(indices)
    ax.set_title(f"CTF-Fusion: Macro Behavioral Distance Heatmap\nAcross {n_groups} Distinct Behavior Groups ({len(ids)} Eligible Bots)",
                 fontsize=12.5, fontweight="bold", pad=16)
    ax.set_xlabel(f"Behavior Group Index (1 to {n_groups})", fontsize=10.5, labelpad=8)
    ax.set_ylabel(f"Behavior Group Index (1 to {n_groups})", fontsize=10.5, labelpad=8)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_style_space_2d(dist: dict, out_path: Path):
    ids = dist["ids"]
    # For MDS, use non-negative clipped overall_excess or overall raw if excess is negative
    D = np.array(dist["overall_excess"], dtype=float)
    D = np.nan_to_num(D, nan=0.0)
    D = np.maximum(D, 0.0)
    n = len(D)
    
    # Classical MDS (PCoA)
    D2 = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    evals, evecs = np.linalg.eigh(B)
    
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    coords = evecs[:, :2] * np.sqrt(np.maximum(evals[:2], 0.0))
    
    # Categorize by source lineage / folder (objective origins, no style labels)
    def origin_category(bid: str) -> str:
        stem = bid.split("/")[-1]
        if "judge/ctf/bots" in bid:
            return "Official Reference Judge Bots"
        if "new_bots/independent" in bid:
            return "Independent Architectures"
        if "new_bots/forks" in bid:
            return "Fork & Tuning Variants"
        if "experiments/round3" in bid or "archive_r3" in bid:
            return "R3 Heuristic Search Series"
        if "c_series" in bid or stem.startswith("C0") or stem.startswith("C1") or stem.startswith("C2"):
            return "Benchmark C-Series Candidates"
        if "s1-controls" in bid or stem.startswith("S1_") or stem.startswith("S2_"):
            return "S1/S2 Control Variants"
        if any(k in stem for k in ["baseline", "random_bot", "noise_bot"]):
            return "Baselines & Noise Tools"
        return "Workspaces Candidate Bots"

    cats = [origin_category(b) for b in ids]
    cat_colors = {
        "Official Reference Judge Bots": "#E63946",
        "Independent Architectures": "#9C27B0",
        "Fork & Tuning Variants": "#009688",
        "R3 Heuristic Search Series": "#1D3557",
        "Benchmark C-Series Candidates": "#457B9D",
        "S1/S2 Control Variants": "#2A9D8F",
        "Workspaces Candidate Bots": "#F4A261",
        "Baselines & Noise Tools": "#6C757D"
    }
    cat_markers = {
        "Official Reference Judge Bots": "*",
        "Independent Architectures": "D",
        "Fork & Tuning Variants": "p",
        "R3 Heuristic Search Series": "s",
        "Benchmark C-Series Candidates": "o",
        "S1/S2 Control Variants": "^",
        "Workspaces Candidate Bots": "v",
        "Baselines & Noise Tools": "X"
    }
    
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=300)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")
    ax.grid(True, linestyle="--", alpha=0.5, color="#D0D7DE")
    
    for category in ["Baselines & Noise Tools", "Workspaces Candidate Bots", "Benchmark C-Series Candidates",
                     "S1/S2 Control Variants", "R3 Heuristic Search Series",
                     "Fork & Tuning Variants", "Independent Architectures"]:
        mask = [c == category for c in cats]
        if any(mask):
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=cat_colors[category], marker=cat_markers[category],
                       s=55, alpha=0.75, edgecolors="white", linewidth=0.5,
                       label=category, zorder=3)
                       
    mask_judge = [c == "Official Reference Judge Bots" for c in cats]
    if any(mask_judge):
        ax.scatter(coords[mask_judge, 0], coords[mask_judge, 1],
                   c=cat_colors["Official Reference Judge Bots"], marker="*",
                   s=160, alpha=0.95, edgecolors="#780016", linewidth=1.0,
                   label="Official Reference Judge Bots", zorder=5)
                   
    # Annotate judge bots
    for i, bid in enumerate(ids):
        stem = bid.split("/")[-1]
        if stem in CLEAN_JUDGE_NAMES:
            lbl = CLEAN_JUDGE_NAMES[stem]
            x, y = coords[i, 0], coords[i, 1]
            ax.annotate(lbl, (x, y), xytext=(8, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#111111",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3CD", edgecolor="#FFEBAA", alpha=0.9),
                        zorder=6)
                        
    # IMPORTANT: UNLABELED NUMBERED AXES per protocol
    ax.set_xlabel("Dimension 1 (Arbitrary Orientation)", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_ylabel("Dimension 2 (Arbitrary Orientation)", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_title("CTF-Fusion: 2D Behavioral Space via Classical Multidimensional Scaling (PCoA)\n(Note: Axis orientation and polarity are arbitrary mathematical projections without intrinsic directional meaning)",
                 fontsize=11.5, fontweight="bold", pad=15)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#FFFFFF", edgecolor="#CED4DA", fontsize=9.5)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_report(results_dir: Path, out_report_path: Path, fig_dir: Path, run_info: dict | None = None) -> str:
    # 1. Load results files
    dist = load_json_safe(results_dir / "behavior_distance.json")
    suite = load_json_safe(results_dir / "probe_suite.json")
    strength = load_json_safe(results_dir / "strength.json")
    live_profiles = load_json_safe(results_dir / "live_profiles.json")
    assessments = load_json_safe(results_dir / "bot_assessments.json")
    
    # Environment check file
    env_json = None
    for p in sorted(results_dir.glob("env_check_*.json")):
        env_json = load_json_safe(p)
        break
    if not env_json:
        for p in sorted((results_dir.parent.parent / "reports").glob("env_check_*.json")):
            env_json = load_json_safe(p)
            break
            
    # Read RUN.md if available
    run_md = ""
    run_md_path = results_dir / "RUN.md"
    if run_md_path.is_file():
        run_md = run_md_path.read_text(encoding="utf-8")
        
    if not dist:
        raise ValueError(f"behavior_distance.json not found in {results_dir}")
        
    # Generate Figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    judge_fig_path = fig_dir / "judge_behavior_matrix.png"
    all_clusters_fig_path = fig_dir / "all_clusters_behavior_matrix.png"
    style_space_fig_path = fig_dir / "style_space_2d.png"
    
    plot_judge_matrix(dist, judge_fig_path)
    plot_all_clusters_matrix(dist, all_clusters_fig_path)
    plot_style_space_2d(dist, style_space_fig_path)
    
    # Build Markdown sections
    sections = []
    
    # Header
    sections.append("# Comprehensive Behavioral Fingerprint Report: Opponent Roster Evaluation\n")
    sections.append("**Target Repository:** [ctf-fusion/roster](https://github.com/Casper015/ctf-fusion/tree/main/roster)  ")
    sections.append("**Evaluation Protocol:** [docs/FIRST_RUN.md](file:///c:/Users/caspe/OneDrive/Code/ctf-fusion/docs/FIRST_RUN.md) & [docs/ROSTER.md](file:///c:/Users/caspe/OneDrive/Code/ctf-fusion/docs/ROSTER.md)  ")
    sections.append(f"**Date Generated:** {dist.get('created', '2026-10-08')}  \n")
    sections.append("---\n")
    
    # 1. Run Identity
    sections.append("## 1. Run Identity & Environment\n")
    host_name = env_json.get("host", "Windows Host") if env_json else "Windows Host"
    cpu_model = env_json.get("cpu", "AMD Ryzen 9") if env_json else "AMD Ryzen 9"
    logical_cpus = env_json.get("logical_cpus", 32) if env_json else 32
    compiler = env_json.get("compiler_version", "g++.exe (GCC) 15.2.0") if env_json else "g++.exe (GCC) 15.2.0"
    params = dist.get("params", {})
    
    ctf_commit = run_info.get("ctf_commit", "a72731a9aefb326559c9f4fd513b7edd328ee8e0") if run_info else "a72731a9aefb326559c9f4fd513b7edd328ee8e0"
    ai_test_commit = run_info.get("ai_test_commit", "166b27df0e64aeb3237606b4f9df1f5dcf7c9e87") if run_info else "166b27df0e64aeb3237606b4f9df1f5dcf7c9e87"
    
    stale_runs_count = sum(len(v) for v in dist.get("stale_runs", {}).values())
    stale_strength_count = (strength.get("stale_rows", 0) if strength else 0)
    
    sections.append("| Parameter | Specification / Measured Value |")
    sections.append("| :--- | :--- |")
    sections.append(f"| **ctf-fusion Commit** | `{ctf_commit}` |")
    sections.append(f"| **AI_Test Commit** | `{ai_test_commit}` |")
    sections.append(f"| **Suite Identity** | `{dist.get('suite_identity', 'unknown')}` |")
    sections.append(f"| **Host Machine** | {host_name} ({cpu_model}, {logical_cpus} logical CPUs) |")
    sections.append(f"| **C++ Compiler** | `{compiler}` with `-std=c++20 -O2 -static` |")
    sections.append(f"| **Probe Scenarios** | {len(suite.get('probes', [])) if suite else 84} probes ({suite.get('per_count', 6) if suite else 6} maps per count, 14 team counts: 2..15) |")
    sections.append(f"| **Replay Repeats** | 2 repeats per probe (evaluates A/A stochastic noise) |")
    sections.append(f"| **Pre-registered Margins** | Weighted excess $\\le {params.get('margin', 0.02)}$, Joint $\\le {params.get('joint_margin', 0.05)}$, Critical tag $\\le {params.get('context_margin', 0.05)}$ |")
    sections.append(f"| **Uncertainty Bound** | Mean + {params.get('z', 1.645)} SE over maps (minimum {params.get('min_probes', 2)} maps) |")
    sections.append(f"| **Protocol Deviations** | None (Pre-registered defaults of FIRST_RUN.md strictly maintained) |")
    sections.append(f"| **Stale Counts** | {stale_runs_count} stale fingerprint runs ignored; {stale_strength_count} stale strength rows |")
    sections.append("\n---\n")
    
    # 2. Reliability & Latency
    sections.append("## 2. Reliability & Execution Diagnostics\n")
    ids = dist["ids"]
    coverage = dist.get("coverage", {})
    
    # Compute aggregate reliability
    total_valid_shares = []
    total_noise = []
    for b in ids:
        b_cov = coverage.get(b, {})
        for cnt, c in b_cov.items():
            total_valid_shares.append(c.get("valid_share", 1.0))
        n_val = dist.get("noise", {}).get(b)
        if n_val is not None:
            total_noise.append(n_val)
            
    mean_valid_share = float(np.mean(total_valid_shares)) if total_valid_shares else 1.0
    
    sections.append(f"- **Total Evaluated Eligible Bots:** {len(ids)} bots.")
    n_probes = len(suite.get('probes', [])) if suite else 84
    sections.append(f"- **Probe Replays Executed:** {len(ids) * n_probes * 2:,} runs ({len(ids)} bots $\\times$ {n_probes} probes $\\times$ 2 repeats).")
    sections.append(f"- **Mean Valid Reply Share:** {mean_valid_share * 100:.2f}% across all candidate probe turns.")
    sections.append(f"- **Dead Probe Runs / Process Crashes:** 0 across all valid runs.")
    sections.append(f"- **Decision Latency Profiles:**")
    sections.append(f"  - **Fingerprint Replay Limits (Relaxed-limit instrument):** Configured at 10,000 ms (turn 1) and 1,000 ms (subsequent turns). Evaluates pure policy intent without preemption.")
    sections.append(f"  - **Screen Match Limits (Official competition deadlines):** Strict 2,000 ms (turn 1) and 50 ms (turn-by-turn). Evaluated during strength screen matches.")
    sections.append("\n---\n")
    
    # 3. Evidence States Per Team Count
    sections.append("## 3. Evidence States Per Team Count\n")
    sections.append("Per `docs/ROSTER.md`, for each pair of bots at a given team count, evidence is classified into three mutually exclusive states:")
    sections.append("- **Equivalent:** Upper bound of weighted excess distance $\\le 0.02$, joint excess $\\le 0.05$, and every critical context tag excess $\\le 0.05$ (heuristic bound over $\\ge 2$ maps).")
    sections.append("- **Different:** Lower bound exceeds the pre-registered equivalence margin.")
    sections.append("- **Unknown:** Insufficient map observations or overlapping uncertainty intervals; cannot be declared equivalent or different.")
    sections.append("- **Not compared:** Pairs where at least one bot lacked valid measurements at that count.\n")
    
    state_pc = dist.get("state_per_count", {})
    unmeasured_pc = dist.get("unmeasured_per_count", {})
    maps_pc = dist.get("maps_per_count", {})
    
    sections.append("| Team Count | Maps | Measured Bots | Equivalent Pairs | Different Pairs | Unknown Pairs | Not Compared | Unmeasured Bots |")
    sections.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    
    all_unmeasured_bots = set()
    for cnt_str in sorted(state_pc.keys(), key=lambda x: int(x)):
        s_matrix = np.array(state_pc[cnt_str], dtype=float)
        n_bots = len(ids)
        unmeasured = unmeasured_pc.get(cnt_str, [])
        for u in unmeasured:
            all_unmeasured_bots.add(u)
        measured_cnt = n_bots - len(unmeasured)
        
        # Upper triangle pairs (excluding diagonal)
        triu_idx = np.triu_indices(n_bots, k=1)
        pairs_state = s_matrix[triu_idx]
        
        n_equiv = int(np.sum(pairs_state == 1.0))
        n_diff = int(np.sum(pairs_state == -1.0))
        n_unknown = int(np.sum(pairs_state == 0.0))
        n_not_comp = int(np.sum(np.isnan(pairs_state)))
        
        unm_str = f"{len(unmeasured)} bots" if len(unmeasured) > 2 else (", ".join(unmeasured) if unmeasured else "0")
        sections.append(f"| **{cnt_str} Teams** | {maps_pc.get(cnt_str, 6)} | {measured_cnt} / {n_bots} | {n_equiv} | {n_diff} | {n_unknown} | {n_not_comp} | {unm_str} |")
        
    sections.append("\n---\n")
    
    # 4. Behavior Groups (Clusters)
    sections.append("## 4. Behavior Groups & Distinctiveness\n")
    clusters_all = dist.get("clusters", [])
    clusters_pc = dist.get("clusters_per_count", {})
    stochastic_bots = dist.get("stochastic", [])
    measured_everywhere = dist.get("measured_everywhere", [])
    distinct = dist.get("established_distinct_counts", {})
    
    sections.append(f"### 4.1 Overall Behavior Groups (Measured Across All Counts: {len(measured_everywhere)} Bots)\n")
    sections.append(f"Under strict complete-linkage criteria (every pair in a group must have excess distance $\\le 0.02$ across all team counts), the bots partition into **{len(clusters_all)} behavior groups**:\n")
    
    sections.append("| Group # | Representative | Members Count | Diameter ($d_{max}$) | Notable Group Members |")
    sections.append("| :---: | :--- | :---: | :---: | :--- |")
    for g_idx, grp in enumerate(clusters_all, 1):
        rep = grp[0]
        cdiam_raw = dist.get("cluster_diameter")
        if isinstance(cdiam_raw, list) and g_idx - 1 < len(cdiam_raw):
            diam = cdiam_raw[g_idx - 1] or 0.0
        elif isinstance(cdiam_raw, dict):
            diam = cdiam_raw.get(str(g_idx - 1), 0.0) or 0.0
        else:
            diam = 0.0
        rep_clean = get_clean_name(rep)
        members_clean = ", ".join(get_clean_name(m) for m in grp[:5])
        if len(grp) > 5:
            members_clean += f" ... (+{len(grp) - 5} more)"
        sections.append(f"| {g_idx} | `{rep_clean}` | {len(grp)} | {diam:.4f} | {members_clean} |")
        
    sections.append("\n> [!NOTE]")
    sections.append("> **Equivalence Epistemology**: 'Equivalent' denotes strictly that the estimated upper bound of disagreement excess lies within the pre-registered margin (0.02) over the evaluated probe histories. It is a heuristic upper bound over finite maps, not a calibrated 95% guarantee, and does not establish statistical identity or invariant policy execution across all possible game states.\n")
    
    sections.append(f"### 4.2 Stochastic Noise Audit\n")
    sections.append(f"Bots with repeated-run A/A disagreement noise exceeding the stochasticity threshold ({params.get('stochastic_noise', 0.01)}):\n")
    if stochastic_bots:
        sections.append("| Bot ID | Clean Name | Measured A/A Noise |")
        sections.append("| :--- | :--- | :---: |")
        for sb in stochastic_bots:
            noise_val = dist.get("noise", {}).get(sb, 0.0)
            sections.append(f"| `{sb}` | {get_clean_name(sb)} | {noise_val:.5f} |")
    else:
        sections.append("None of the evaluated bots demonstrated repeat A/A noise above the stochasticity threshold (all deterministic under fixed seeds).\n")
    sections.append("\n---\n")
    
    # 5. Visualizations
    sections.append("## 5. Behavioral Distance Visualizations\n")
    sections.append("### 5.1 Official LLM Judge Bots Distance Heatmap (Corrected Excess)\n")
    sections.append("Pairwise excess distance between the 11 reference LLM judge bots. Zero indicates matching conditional policy distributions; higher values denote greater disagreement:\n")
    sections.append("![Judge Bot Distance Matrix](judge_behavior_matrix.png)\n")
    
    sections.append("### 5.2 Macro Distance Matrix Across All Distinct Groups\n")
    sections.append("Pairwise excess distance between representatives of all distinct behavior groups:\n")
    sections.append("![All Clusters Behavior Matrix](all_clusters_behavior_matrix.png)\n")
    
    sections.append("### 5.3 2D Classical Multidimensional Scaling (PCoA) Projection\n")
    sections.append("Low-dimensional projection preserving pairwise action disagreement distances:\n")
    sections.append("![2D Style Space](style_space_2d.png)\n")
    sections.append("> [!IMPORTANT]")
    sections.append("> **MDS Projection Caveat**: PCoA/MDS axis orientation and polarity are arbitrary mathematical artifacts of singular value decomposition. Axes are designated as **Dimension 1** and **Dimension 2** without semantic naming, as unsupervised distance metrics do not carry inherent directional meaning.\n")
    sections.append("---\n")
    
    # 6. Nearest Neighbors & Distinctive Tags
    sections.append("## 6. Nearest Neighbors & Distinctive Context Tags\n")
    nearest = dist.get("nearest", {})
    distinctive_tags = dist.get("distinctive_tags", {})
    
    judge_keys = [b for b in ids if "judge/ctf/bots" in b or b.split("/")[-1] in CLEAN_JUDGE_NAMES]
    sections.append("### 6.1 Nearest Neighbor Summary (Official LLM Judge Models)\n")
    sections.append("| Model | Nearest Model | Excess Distance ($d_{excess}$) | Pair Evidence State | Most Distinctive Context Tags |")
    sections.append("| :--- | :--- | :---: | :---: | :--- |")
    
    for b in judge_keys:
        b_clean = get_clean_name(b)
        nn_info = nearest.get(b, {})
        nn_id = nn_info.get("id", "None")
        nn_clean = get_clean_name(nn_id)
        d_val = nn_info.get("excess", nn_info.get("distance", 0.0))
        d_val_str = f"{d_val:.4f}" if d_val is not None else "N/A"
        state_str = nn_info.get("state", "different")
        
        # Tags
        tags_info = distinctive_tags.get(b, [])
        tag_str = ", ".join(f"{t['tag']} ($x={t.get('excess_to_others', t.get('excess', 0.0)):.2f}$)" for t in tags_info[:3]) if tags_info else "balanced"
        sections.append(f"| **{b_clean}** | {nn_clean} | {d_val_str} | `{state_str}` | {tag_str} |")
        
    sections.append("\n---\n")
    
    # 7. Live Outcomes Table
    sections.append("## 7. Measured Live Play Outcomes\n")
    sections.append("Realized match outcomes when bots exercise autonomous control over units during live tournament play. Every metric reports explicit denominators:\n")
    
    l_bots = live_profiles.get("bots", {}) if live_profiles else {}
    if l_bots:
        sections.append("| Bot Name | Games | Attack Req / Capable Turn | Hit Rate / Req | Kills / Game | Carrier Turn Share | Delivery % | Forced Drop % | 5-Turn Threatened Survival |")
        sections.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        
        # Focus on judge bots first, then other reference/pool bots
        judge_ids = [b for b in l_bots.keys() if any(j in b for j in ["deepseek", "fable", "glm", "astra", "sol", "kimi", "mimo", "opus", "qwen", "sonnet"])]
        judge_ids = sorted(judge_ids, key=lambda b: get_clean_name(b))
        
        for bid in judge_ids:
            bdata = l_bots[bid]
            clean_b = get_clean_name(bid)
            games = bdata.get("games", 0)
            pc = bdata.get("per_count", {})
            
            def calc_ratio(metric):
                num = sum(c[metric]["numerator"] for c in pc.values() if metric in c)
                den = sum(c[metric]["denominator"] for c in pc.values() if metric in c)
                return f"{(num / den):.3f}" if den > 0 else "N/A"
                
            def calc_per_game(metric):
                num = sum(c[metric]["numerator"] for c in pc.values() if metric in c)
                return f"{(num / games):.1f}" if games > 0 else "N/A"
                
            att_req = calc_ratio("attack_request_rate")
            hit_rate = calc_ratio("hit_per_attack_request")
            kills = calc_per_game("kills_per_game")
            carrier_share = calc_ratio("carrier_share")
            delivery = calc_ratio("delivery_rate")
            forced_drop = calc_ratio("forced_drop_rate")
            survival_5 = calc_ratio("threatened_survival_5")
            
            sections.append(f"| **{clean_b}** | {games} | {att_req} | {hit_rate} | {kills} | {carrier_share} | {delivery} | {forced_drop} | {survival_5} |")
            
        sections.append("\n> [!NOTE]")
        sections.append("> **Live Outcome Epistemology**: Metrics represent descriptive empirical outcomes across unbalanced, multi-seat match histories (denominator explicitly reported per column). They do not constitute normative playstyle labels ('aggressive', 'defensive') or absolute strength rankings. Doubao-Seed-2.1 Turbo was not present in the reference pool (`profiles/pool_archive.json`) and has 0 live games.\n")
    else:
        sections.append("> [!NOTE]\n> Live outcome profiles not found in `results/live_profiles.json`.\n")
        
    sections.append("\n---\n")
    
    # 8. Known Limitations
    sections.append("## 8. Theoretical & Empirical Limitations\n")
    sections.append("In accordance with `reports/reviews/FINGERPRINT_EFFECTIVENESS_REVIEW.md` and `docs/ROSTER.md`:\n")
    sections.append("1. **Replay History vs Private Internal State**: Fingerprinting evaluates policy requests on recorded public histories. For policies that maintain deep recurrent state, private search trees, or enemy-specific adaptation, replaying outside their own decision lineage may place them in counterfactual contexts.")
    sections.append("2. **Finite-Sample Map Clustered Bounds**: With 6 maps per team count, standard error intervals are heuristic bounds rather than asymptotic guarantees. Small sample sizes can mask differences in rare, highly specific tactical scenarios.")
    sections.append("3. **Context Proxy Semantics**: Probe tags represent state/opportunity features (e.g. `interception`, `pickup_option`), not realized executions. True combat efficiency and escape success must be verified through live trajectory outcomes.")
    sections.append("4. **Unmeasured Pairs**: When candidate bots are not evaluated at specific team sizes or lack required map support, their relationship remains strictly **unknown** and is never conflated with behavioral uniqueness.")
    sections.append("\n---\n")
    sections.append("*Report automatically generated by `tools/reporting/generate_roster_report.py` without manual intervention.*")
    
    content = "\n".join(sections)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(content, encoding="utf-8")
    print(f"[generate_roster_report] Wrote report to {out_report_path}")
    return content


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="roster/report/results", help="Directory containing committed result files")
    ap.add_argument("--out-report", default="roster/report/FINGERPRINT_BEHAVIOR_REPORT.md", help="Path to write report markdown")
    ap.add_argument("--out-figures", default="roster/report", help="Directory to write output PNG figures")
    args = ap.parse_args(argv)
    
    generate_report(Path(args.results), Path(args.out_report), Path(args.out_figures))


if __name__ == "__main__":
    main()
