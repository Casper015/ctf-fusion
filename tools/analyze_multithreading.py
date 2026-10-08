import json
from pathlib import Path

def analyze():
    p16_path = Path("reports/capacity_DESKTOP-G60OJUD.json")
    p1_path = Path("reports/baseline_1seat/capacity_DESKTOP-G60OJUD.json")
    
    with open(p16_path, encoding="utf-8") as f:
        r16 = json.load(f)
    with open(p1_path, encoding="utf-8") as f:
        r1 = json.load(f)
        
    print(f"Host: {r16['host']}")
    print(f"Time 16-seat: {r16['time']}")
    print(f"Time 1-seat: {r1['time']}")
    
    tiers = ["S", "M", "L", "XL", "XXL"]
    
    summary = {}
    for tier in tiers:
        summary[tier] = {
            "policy_slowdowns": [],
            "value_slowdowns": [],
            "need_slowdowns": [],
            "p1_needs": [],
            "p16_needs": [],
            "p1_policy_p99": [],
            "p16_policy_p99": [],
            "p1_value_us": [],
            "p16_value_us": [],
        }

    for c16, c1 in zip(r16["counts"], r1["counts"]):
        teams = c16["teams"]
        for tier in tiers:
            d16 = c16["tiers"][tier]
            d1 = c1["tiers"][tier]
            
            p_ratio = d16["policy_p99_ms"] / d1["policy_p99_ms"]
            v_ratio = d16["value_node_us"] / d1["value_node_us"]
            need_ratio = d16["need_ms"] / d1["need_ms"]
            
            summary[tier]["policy_slowdowns"].append((teams, p_ratio))
            summary[tier]["value_slowdowns"].append((teams, v_ratio))
            summary[tier]["need_slowdowns"].append((teams, need_ratio))
            summary[tier]["p1_needs"].append((teams, d1["need_ms"]))
            summary[tier]["p16_needs"].append((teams, d16["need_ms"]))
            summary[tier]["p1_policy_p99"].append((teams, d1["policy_p99_ms"]))
            summary[tier]["p16_policy_p99"].append((teams, d16["policy_p99_ms"]))
            summary[tier]["p1_value_us"].append((teams, d1["value_node_us"]))
            summary[tier]["p16_value_us"].append((teams, d16["value_node_us"]))

    print("\n--- Summary Slowdowns (16-thread vs 1-thread) ---")
    for tier in tiers:
        avg_need_sd = sum(x[1] for x in summary[tier]["need_slowdowns"]) / len(summary[tier]["need_slowdowns"])
        min_need_sd = min(x[1] for x in summary[tier]["need_slowdowns"])
        max_need_sd = max(x[1] for x in summary[tier]["need_slowdowns"])
        
        avg_p_sd = sum(x[1] for x in summary[tier]["policy_slowdowns"]) / len(summary[tier]["policy_slowdowns"])
        avg_v_sd = sum(x[1] for x in summary[tier]["value_slowdowns"]) / len(summary[tier]["value_slowdowns"])
        
        print(f"Tier {tier:3s}: Total Time Inflation = {avg_need_sd:.2f}x (range {min_need_sd:.2f}x - {max_need_sd:.2f}x) | Policy P99 Inflation = {avg_p_sd:.2f}x | Value Node Inflation = {avg_v_sd:.2f}x")

    print("\n--- Aggregate Throughput Estimation ---")
    # In 16 seats, 16 seats run concurrently.
    # Total elapsed wall clock time comparison:
    time_16 = sum(c["seconds"] for c in r16["counts"])
    time_1 = sum(c["seconds"] for c in r1["counts"])
    total_turns_16 = sum(c["tiers"]["S"]["turns"] for c in r16["counts"])
    total_turns_1 = sum(c["tiers"]["S"]["turns"] for c in r1["counts"])
    print(f"Total benchmark wall-clock time across all 14 team counts:")
    print(f"  16-seat run: {time_16:.1f} s (evaluated {total_turns_16} total seat-turns)")
    print(f"  1-seat run:  {time_1:.1f} s (evaluated {total_turns_1} total seat-turns)")
    print(f"Throughput (seat-turns/s):")
    print(f"  16-seat: {total_turns_16 / time_16:.0f} turns/s")
    print(f"  1-seat:  {total_turns_1 / time_1:.0f} turns/s")
    print(f"Throughput scaling factor: {(total_turns_16 / time_16) / (total_turns_1 / time_1):.2f}x on 16 cores")

if __name__ == "__main__":
    analyze()
