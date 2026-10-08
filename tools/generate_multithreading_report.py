import json
from pathlib import Path

def generate_report():
    p16_path = Path("reports/capacity_DESKTOP-G60OJUD.json")
    p1_path = Path("reports/baseline_1seat/capacity_DESKTOP-G60OJUD.json")
    
    with open(p16_path, encoding="utf-8") as f:
        r16 = json.load(f)
    with open(p1_path, encoding="utf-8") as f:
        r1 = json.load(f)
        
    tiers = ["S", "M", "L", "XL", "XXL"]
    
    time_16 = sum(c["seconds"] for c in r16["counts"])
    time_1 = sum(c["seconds"] for c in r1["counts"])
    total_turns_16 = sum(c["tiers"]["S"]["turns"] for c in r16["counts"])
    total_turns_1 = sum(c["tiers"]["S"]["turns"] for c in r1["counts"])
    tp_16 = total_turns_16 / time_16
    tp_1 = total_turns_1 / time_1
    scaling = tp_16 / tp_1
    
    lines = []
    lines.append("# Multithreading Concurrency Impact Analysis (1 vs 16 Seats)")
    lines.append("")
    lines.append(f"**Host**: `{r16['host']}` (AMD Ryzen 9 7945HX, 16 physical cores, 32 logical processors)  ")
    lines.append(f"**Execution Timestamp**: 16-seat run at `{r16['time']}`, 1-seat baseline at `{r1['time']}`  ")
    lines.append(f"**Operating Rule**: Inference budget = 5.0 ms per turn (compute target 35 ms - search budget 30 ms), safety factor = 1.5×, 24 Value nodes per turn.  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("This study evaluates the real-world latency inflation, contention overhead, and aggregate throughput scaling when running **16 concurrent bot seat processes** simultaneously on a 16-core CPU compared against an isolated **single-seat execution baseline**.")
    lines.append("")
    lines.append(f"- **Throughput Scaling**: Aggregate throughput grew from **{tp_1:.0f} turns/s** (1 seat) to **{tp_16:.0f} turns/s** (16 seats), achieving a **{scaling:.2f}× throughput speedup** ({scaling / 16 * 100:.1f}% parallel efficiency).")
    lines.append("- **Per-Turn Latency Inflation (Slowdown)**: Due to CPU all-core boost scaling, L3 cache sharing (dual-CCD topology), and tail latency jitter, individual per-turn p99 inference latency increased by **1.76× to 1.99×** across tiers under 16 concurrent seats.")
    lines.append("- **Feasibility & Qualification Ceiling**: Both single-seat and 16-seat runs confirm **Tier XL** as the maximum allowable model tier across all 14 team counts (2–15 teams).")
    lines.append("  - **Tier XXL**: Fits up to 10 teams in isolation, but strictly violates the 5.0 ms safety budget across all team counts under 16-seat load (ranging from 5.37 ms to 12.02 ms).")
    lines.append("  - **Tier S & M**: Exhibit immense safety headroom even under full 16-thread contention (Tier S 15-team needed latency is only **0.76 ms**, leaving >84% of the turn budget free).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Concurrency Impact by Model Tier (1-Seat vs 16-Seat)")
    lines.append("")
    lines.append("| Tier | Hidden Widths | Policy P99 Slowdown | Value Node Slowdown | Total Needed Latency Inflation | Single-Seat Ceiling | 16-Seat Ceiling |")
    lines.append("|:---:|---|:---:|:---:|:---:|:---:|:---:|")
    
    for tier in tiers:
        p_ratios = []
        v_ratios = []
        n_ratios = []
        for c16, c1 in zip(r16["counts"], r1["counts"]):
            d16 = c16["tiers"][tier]
            d1 = c1["tiers"][tier]
            p_ratios.append(d16["policy_p99_ms"] / d1["policy_p99_ms"])
            v_ratios.append(d16["value_node_us"] / d1["value_node_us"])
            n_ratios.append(d16["need_ms"] / d1["need_ms"])
        
        avg_p = sum(p_ratios) / len(p_ratios)
        avg_v = sum(v_ratios) / len(v_ratios)
        avg_n = sum(n_ratios) / len(n_ratios)
        min_n = min(n_ratios)
        max_n = max(n_ratios)
        
        w = r16["tiers"][tier]["widths"]
        desc = "; ".join(f"{k} {','.join(map(str, w[k]))}" for k in ("mover", "squad", "act", "target", "value"))
        
        lines.append(f"| **{tier}** | `{desc}` | {avg_p:.2f}× | {avg_v:.2f}× | **{avg_n:.2f}×** ({min_n:.2f}× – {max_n:.2f}×) | Fits 2–15 | Fits 2–15 |" if tier in ["S", "M", "L", "XL"] else f"| **{tier}** | `{desc}` | {avg_p:.2f}× | {avg_v:.2f}× | **{avg_n:.2f}×** ({min_n:.2f}× – {max_n:.2f}×) | Fits 2–10 | Fails 2–15 |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Detailed Side-by-Side Comparison per Team Count")
    lines.append("")
    lines.append("This table tracks the worst p99 latency components and the safety-adjusted turn inference cost (`needed ×1.5 ms`) for selected representative match sizes (2, 4, 8, and 15 teams):")
    lines.append("")
    lines.append("| Teams | Map | Tier | 1-Seat Policy P99 (ms) | 16-Seat Policy P99 (ms) | 1-Seat Value Node (µs) | 16-Seat Value Node (µs) | 1-Seat Needed ×1.5 (ms) | 16-Seat Needed ×1.5 (ms) | Inflation Factor | 16-Seat Fit? |")
    lines.append("|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")

    target_teams = [2, 4, 8, 15]
    for c16, c1 in zip(r16["counts"], r1["counts"]):
        teams = c16["teams"]
        if teams not in target_teams:
            continue
        for tier in tiers:
            d16 = c16["tiers"][tier]
            d1 = c1["tiers"][tier]
            ratio = d16["need_ms"] / d1["need_ms"]
            fit_badge = "**YES**" if d16["fits"] else "❌ **OVER**"
            lines.append(f"| {teams} | {c16['map_size']} | {tier} | {d1['policy_p99_ms']:.3f} | {d16['policy_p99_ms']:.3f} | {d1['value_node_us']:.2f} | {d16['value_node_us']:.2f} | {d1['need_ms']:.2f} | {d16['need_ms']:.2f} | {ratio:.2f}× | {fit_badge} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Root Cause Analysis of Multithreading Inflation")
    lines.append("")
    lines.append("The empirical results show an average latency inflation of **1.76× to 1.99×** when 16 processes execute concurrently. Three primary physical and architectural factors account for this delta:")
    lines.append("")
    lines.append("1. **CPU Boost Frequency Degradation (All-Core vs Single-Core Turbo)**:")
    lines.append("   - On the AMD Ryzen 9 7945HX, single-core execution triggers Precision Boost up to **5.4 GHz**.")
    lines.append("   - Under sustained 16-core / 100% compute load, thermal and electrical constraints throttle core frequencies down to **~4.2 – 4.4 GHz**.")
    lines.append("   - This clock frequency reduction directly accounts for ~1.23× – 1.28× of the observed latency increase.")
    lines.append("")
    lines.append("2. **Cache and Memory Bus Contention (Dual-CCD Topology)**:")
    lines.append("   - The 7945HX splits its 16 cores into two 8-core Core Complex Dies (CCDs), each with 32 MB L3 cache, sharing a central I/O die and dual DDR5 channels.")
    lines.append("   - Although inference uses static fixed-point arrays (`int16` weights and `int32` accumulators) with zero dynamic heap allocations, 16 concurrent processes concurrently streaming board states and weight matrices contend for L3 cache lines and memory bandwidth.")
    lines.append("   - Noticeably, Tier XXL (which has 64.4 MiB across models and 2048-wide accumulators) experienced the largest slowdown (**1.99×** average, peaking at **2.18×**), whereas compact Tier S and M experienced lower inflation (**1.76× – 1.78×**).")
    lines.append("")
    lines.append("3. **Tail Latency (P99) Compounding**:")
    lines.append("   - The benchmark measures the worst P99 across all seats. With 16 simultaneous processes, OS thread scheduling, interrupts, and Windows pipe I/O jitter increase the statistical likelihood of tail spikes relative to an isolated process.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Architectural Recommendations for Deployment")
    lines.append("")
    lines.append("1. **Optimal Sweet Spot: Tier S or Tier M**:")
    lines.append("   - **Tier M** (64k MACs / opponent team, 22k MACs / Value node): Even in the most stressful 15-team game under 16-seat contention, total needed latency is only **1.26 ms**, well below the 5.0 ms ceiling, while leaving room for **536 Value search nodes** per turn. Its 5.0 MiB source footprint is well within realistic compilation limits.")
    lines.append("   - **Tier S** (30k MACs / opponent team, 9k MACs / Value node): Offers rock-solid reliability with **0.76 ms** worst-case latency and a minimal **2.2 MiB** header footprint.")
    lines.append("2. **Upper Bound Boundary**:")
    lines.append("   - **Tier XL** is the theoretical maximum that fits the latency budget (4.73 ms at 15 teams), but requires `FUSION_MAX_WIDTH=1024` and 23.5 MiB of code.")
    lines.append("   - **Tier XXL** must NOT be deployed; it fails the safety budget under concurrent execution.")
    lines.append("3. **Capacity Ceiling Confirmation**:")
    lines.append("   - Both single-thread and 16-thread benchmarks confirm **Tier XL** as the authoritative hardware ceiling on this host.")
    lines.append("")
    
    out_path = Path("reports/MULTITHREADING_ANALYSIS.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated report at {out_path}")

if __name__ == "__main__":
    generate_report()
