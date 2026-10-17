// The agent_v2 search-gate bot (task T0076, docs/AGENT_V2_SEARCH_GATE.md): bots/agent_v2/bot.hpp's loop with the
// search always on, plus the gate's identity and counters.
//
// A variant (bots/agent_v2/search_gate/variants/<name>.cpp) defines FUSION_MAX_WIDTH, its FUSION_TIER_*, the tree's
// AGENT_V2_SEARCH_CANDIDATES / AGENT_V2_SEARCH_SAMPLES and AGENT_V2_SEARCH_TAU, includes its actor header (namespace
// meow_rl_model) and its critic header (namespace meow_rl_critic), then this file. Everything else (the parse, P, the
// actor, the resolver, the default order and the admission) is the agent_v2 carrier's, unchanged.
// stderr: the identity on the first turn ("AGENTV2 carrier ... search 1 greedy 0 gate tau <tau> candidates <K> samples
// <S>"), then every turn bot.hpp's cumulative search line followed by "overrides <n> gate_kept <n>": overrides counts
// completed trees that replaced the actor, gate_kept completed trees whose best root beat the actor by a margin <= tau.
#pragma once
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include "../../../inference/agent_v2/runtime.hpp"
#include "../../../inference/agent_v2/search.hpp"

namespace agent_v2_gate_bot {

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

inline int run() {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    static agent_v2::Runtime<agent_v2::Search> rt;
    std::string block, why;
    while (read_block(block)) {
        const auto started = std::chrono::steady_clock::now();
        const std::string reply = rt.decide(block, started, why);
        if (rt.tel.turns == 1)
            std::fprintf(stderr, "AGENTV2 carrier %s actions %s resolver %s/%s scenarios %d actor %s digest %s features %s/%s "
                         "predictor %s model %s search 1 greedy 0 gate tau %.17g candidates %d samples %d\n", agent_v2::kCarrier,
                         agent_v2::kActionSchema, agent_v2::kResolverRule, rt.resolver.profile.name, agent_v2::kResolverScenarios,
                         meow_rl_model::kName, meow_rl_model::kDigest, agent_v2::kPolicyFeatures, agent_v2::kValueFeatures,
                         agent_v2::PredictorP::kKind, rt.pred.model_id(), agent_v2::kTau, agent_v2::kCandidates, agent_v2::kSamples);
        if (!why.empty() && rt.tel.refused == 1) std::fprintf(stderr, "AGENTV2 refused turn %d: %s\n", rt.last_turn_seen, why.c_str());
        auto& q = rt.search;
        std::fprintf(stderr,
                     "AGENTV2 turns %ld refused %ld inconsistent %ld search_used %ld search_fallback %ld leaves %ld trajectories %ld "
                     "refuse_setup %ld refuse_trajectory %ld refuse_expand %ld refuse_critic %ld refuse_late %ld max_setup_us %ld "
                     "max_traj_us %ld max_expand_us %ld max_critic_us %ld max_search_us %ld branches %lld critic %s "
                     "overrides %ld gate_kept %ld\n",
                     rt.tel.turns, rt.tel.refused, rt.tel.inconsistent, rt.tel.search_used, rt.tel.search_fallback, rt.tel.leaves,
                     q.trajectories, q.refuse_setup, q.refuse_trajectory, q.refuse_expand, q.refuse_critic, q.refuse_late,
                     q.max_setup_us, q.max_traj_us, q.max_expand_us, q.max_critic_us, q.max_search_us, rt.pred.counters.branches,
                     meow_rl_critic::kFormat, q.overrides, q.gate_kept);
        std::fflush(stderr);
        std::cout << reply << std::endl;
    }
    return 0;
}

}  // namespace agent_v2_gate_bot

int main() { return agent_v2_gate_bot::run(); }
