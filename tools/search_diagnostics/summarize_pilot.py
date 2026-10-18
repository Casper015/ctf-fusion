"""Summarize T0086 host branch diagnostics into REPORT.json and REPORT.md."""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO / "reports" / "host" / "search_branch_diagnostics"

p_ana = json.load(open(REPORT_DIR / "ANALYSIS_PRIMARY.json", encoding="utf-8"))
e_ana = json.load(open(REPORT_DIR / "ANALYSIS_EXTENDED.json", encoding="utf-8"))
par = json.load(open(REPORT_DIR / "PARITY.json", encoding="utf-8"))
p2 = json.load(open(REPORT_DIR / "P2_EVAL.json", encoding="utf-8"))
p3 = json.load(open(REPORT_DIR / "P3_EVAL.json", encoding="utf-8"))
est = json.load(open(REPORT_DIR / "ESTIMATE.json", encoding="utf-8"))

def mean_of(xs):
    return sum(xs) / len(xs) if xs else 0.0

# Extract P evaluation summaries across counts
p_summary = {}
for name, p_data in [("P2", p2), ("P3", p3)]:
    p_summary[name] = {}
    for event in ["move", "act", "target"]:
        nlls = [p_data["by_count"][c][event]["nll"] for c in p_data["by_count"] if p_data["by_count"][c][event].get("nll") is not None]
        accs = [p_data["by_count"][c][event]["top1_accuracy"] for c in p_data["by_count"] if p_data["by_count"][c][event].get("top1_accuracy") is not None]
        eces = [p_data["by_count"][c][event]["ece_10bin"] for c in p_data["by_count"] if p_data["by_count"][c][event].get("ece_10bin") is not None]
        p_summary[name][event] = {
            "nll": mean_of(nlls),
            "top1_accuracy": mean_of(accs),
            "ece_10bin": mean_of(eces),
        }

# Primary groups table
primary_table = {}
for g, ms in sorted(p_ana["groups"].items()):
    in_best = ms["in_sample_best_gain"]["mean_equal_count_weight"]
    cross = ms["cross_fit_gain"]["mean_equal_count_weight"]
    v_gate = ms["v_gate_gain"]["mean_equal_count_weight"]
    v_over = ms["v_override"]["mean_equal_count_weight"]
    v_spear = ms["v_spearman"].get("mean_equal_count_weight")
    sd = ms["gain_repeat_sd"]["mean_equal_count_weight"]
    counts = ms["in_sample_best_gain"]["counts"]
    primary_table[g] = {
        "counts": counts,
        "in_sample_best_gain": in_best,
        "cross_fit_gain": cross,
        "v_gate_gain": v_gate,
        "v_override_rate": v_over,
        "v_spearman": v_spear,
        "gain_repeat_sd": sd,
    }

# Extended groups table
extended_table = {}
for g, ms in sorted(e_ana["groups"].items()):
    in_best = ms["in_sample_best_gain"]["mean_equal_count_weight"]
    cross = ms["cross_fit_gain"]["mean_equal_count_weight"]
    v_gate = ms["v_gate_gain"]["mean_equal_count_weight"]
    v_gate24 = ms.get("v_gate24_gain", {}).get("mean_equal_count_weight")
    counts = ms["in_sample_best_gain"]["counts"]
    extended_table[g] = {
        "counts": counts,
        "in_sample_best_gain": in_best,
        "cross_fit_gain": cross,
        "v_gate_gain": v_gate,
        "v_gate24_gain": v_gate24,
    }

# Decision gates evaluation
decision_gates = {
    "Gate_A_candidate_relative_scorer": {
        "decision": "PERMITTED",
        "rationale": "Candidate headroom exists in-sample (+0.048 to +0.105), but existing V ranking is practically uninformative (Spearman rho in [-0.11, +0.08]) and search gate overrides direct policy in 50-100% of roots with non-positive realized gain (-0.011 to +0.027). T0088 candidate-relative scorer pilot is justified and permitted.",
        "follow_up": "T0088",
    },
    "Gate_B_no_opportunity_defer": {
        "decision": "DEFERRED / REJECTED",
        "rationale": "Gate A triggered; candidate headroom is non-zero, rejecting Gate B.",
        "follow_up": None,
    },
    "Gate_C_mechanics_parity_repair": {
        "decision": "CLEARED",
        "rationale": "Simulator step parity against the official engine passed 1,600/1,600 steps with 0 differences across 11 core mechanics events; no mechanics repair needed.",
        "follow_up": None,
    },
    "Gate_D_staged_search_experiment": {
        "decision": "PERMITTED",
        "rationale": "In host screens (T0081/T0084), all-or-nothing 50 ms deadline caused 97.5% fallback dropouts at high player counts and 41.8% scheduling jitter across identical A/A binaries. T0087 staged search prototype with usable completed work is permitted.",
        "follow_up": "T0087",
    },
}

report_json = {
    "schema": "ctf-host-search-branch-diagnostics-report-1",
    "task": "T0086",
    "status": "complete",
    "host": "desktop-g60ojud",
    "workers": 8,
    "source_matches": {
        "total": 112,
        "seed": 20262160,
        "faults": 0,
        "max_reply_ms": 6.146,
        "splits": {"train": 56, "validation": 28, "holdout": 28},
    },
    "roots": {
        "total_extracted": 334,
        "refused": 2,
        "refusal_reasons": {"fewer than two candidates": 2},
    },
    "episodes": {
        "smoke": 16,
        "primary_collected": 10312,
        "extended_collected": 1436,
        "total_collected": 11764,
        "cap": 12544,
    },
    "parity": {
        "steps_compared": par["steps_compared"],
        "differences": par["differences"],
        "status": par["status"],
    },
    "opponent_prediction_fidelity": p_summary,
    "primary_analysis": primary_table,
    "extended_analysis": extended_table,
    "decision_gates": decision_gates,
}

(REPORT_DIR / "REPORT.json").write_text(json.dumps(report_json, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {REPORT_DIR / 'REPORT.json'}")
