// The agent_v2 staged-search bot (task T0087, README.md beside this file): bots/agent_v2/search_gate/bot.hpp's loop
// with agent_v2::StagedSearch (inference/agent_v2/search.hpp) in place of Search. Opt-in, never a default.
//
// A variant (variants/<name>.cpp) defines FUSION_MAX_WIDTH, its FUSION_TIER_*, the tree's AGENT_V2_SEARCH_CANDIDATES /
// AGENT_V2_SEARCH_SAMPLES / AGENT_V2_SEARCH_TAU and optionally AGENT_V2_STAGED_SCHEDULE, includes its actor header
// (namespace meow_rl_model) and critic header (namespace meow_rl_critic), then this file. The parse, P, the actor, the
// resolver, the default order, the tree and the admission estimates are the agent_v2 carrier's, unchanged.
// stderr: the identity on the first turn ("... search 1 greedy 0 staged schedule <k>:<s>,... gate tau <tau> candidates
// <K> samples <S> fixed_work <n>"), then every turn the cumulative counters. The line starts as bot.hpp's
// ("AGENTV2 turns .. refused .. inconsistent .. search_used .. search_fallback .. leaves ..") so existing parsers read
// it; search_used counts turns decided by a completed checkpoint, search_fallback turns that played the default order
// because none completed. Then: trajectories, checkpoints (completed, summed over turns), complete (turns that finished
// the schedule), eff_candidates / eff_scenarios (summed over decided turns: the decision checkpoint's k and s),
// overrides, baseline_kept, gate_kept (baseline kept although a root beat it by <= tau), discarded (trajectories
// finished after the last checkpoint), the stop causes refuse_setup / refuse_trajectory / refuse_expand /
// refuse_critic / refuse_late / stopped_work_cap, the max_*_us timings, branches and the critic format.
#pragma once
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include "../../../inference/agent_v2/runtime.hpp"
#include "../../../inference/agent_v2/search.hpp"

namespace agent_v2_staged_bot {

// One protocol block (through its END line); false at the end of input.
inline bool read_block(std::string& block) {
    block.clear();
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() && block.empty()) continue;
        block += line;
        block += '\n';
        if (line.rfind("END", 0) == 0) return true;
    }
    return false;
}

inline std::string schedule_text() {
    std::string s;
    for (int i = 0; i < agent_v2::kStagedSchedule.n; i++)
        s += (i ? "," : "") + std::to_string(agent_v2::kStagedSchedule.at[i].k) + ":" + std::to_string(agent_v2::kStagedSchedule.at[i].s);
    return s;
}

inline int run() {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    static agent_v2::Runtime<agent_v2::StagedSearch> rt;
    std::string block, why;
    while (read_block(block)) {
        const auto started = std::chrono::steady_clock::now();
        const std::string reply = rt.decide(block, started, why);
        if (rt.tel.turns == 1)
            std::fprintf(stderr, "AGENTV2 carrier %s actions %s resolver %s/%s scenarios %d actor %s digest %s features %s/%s "
                         "predictor %s model %s search 1 greedy 0 staged schedule %s gate tau %.17g candidates %d samples %d "
                         "fixed_work %ld\n", agent_v2::kCarrier, agent_v2::kActionSchema, agent_v2::kResolverRule,
                         rt.resolver.profile.name, agent_v2::kResolverScenarios, meow_rl_model::kName, meow_rl_model::kDigest,
                         agent_v2::kPolicyFeatures, agent_v2::kValueFeatures, agent_v2::PredictorP::kKind, rt.pred.model_id(),
                         schedule_text().c_str(), agent_v2::kTau, agent_v2::kCandidates, agent_v2::kSamples, rt.search.work_cap);
        if (!why.empty() && rt.tel.refused == 1) std::fprintf(stderr, "AGENTV2 refused turn %d: %s\n", rt.last_turn_seen, why.c_str());
        auto& q = rt.search;
        std::fprintf(stderr,
                     "AGENTV2 turns %ld refused %ld inconsistent %ld search_used %ld search_fallback %ld leaves %ld trajectories %ld "
                     "checkpoints %ld complete %ld eff_candidates %ld eff_scenarios %ld overrides %ld baseline_kept %ld gate_kept %ld "
                     "discarded %ld refuse_setup %ld refuse_trajectory %ld refuse_expand %ld refuse_critic %ld refuse_late %ld "
                     "stopped_work_cap %ld max_setup_us %ld max_traj_us %ld max_expand_us %ld max_critic_us %ld max_search_us %ld "
                     "branches %lld critic %s\n",
                     rt.tel.turns, rt.tel.refused, rt.tel.inconsistent, rt.tel.search_used, rt.tel.search_fallback, rt.tel.leaves,
                     q.trajectories, q.checkpoints, q.complete, q.eff_candidates, q.eff_scenarios, q.overrides, q.baseline_kept,
                     q.gate_kept, q.discarded, q.refuse_setup, q.refuse_trajectory, q.refuse_expand, q.refuse_critic, q.refuse_late,
                     q.stopped_work_cap, q.max_setup_us, q.max_traj_us, q.max_expand_us, q.max_critic_us, q.max_search_us,
                     rt.pred.counters.branches, meow_rl_critic::kFormat);
        std::fflush(stderr);
        std::cout << reply << std::endl;
    }
    return 0;
}

}  // namespace agent_v2_staged_bot

int main() { return agent_v2_staged_bot::run(); }
