// The agent_v2 CPU bot (task T0065, docs/AGENT_V2.md): our opponent model P, the PPO actor, the mathematical resolver
// assign_v1 and, optionally, the bounded value search, on one engine-mirrored public state.
//
// A variant (bots/agent_v2/variants/<name>.cpp) defines FUSION_MAX_WIDTH and its FUSION_TIER_* (P's model registry),
// includes its actor header (namespace meow_rl_model) and, for AGENT_V2_SEARCH=1, a critic header (namespace
// meow_rl_critic), then this file. AGENT_V2_GREEDY=1 plays the actor's greedy joint (a diagnostic, not the deployment).
// The bot reads each block as text (up to its END line), so a malformed block is refused and answered with the safe
// all-stay reply instead of desynchronizing the stream. stderr: the identity on the first turn and every turn the
// cumulative counters (AGENTV2 turns ...).
#pragma once
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#ifndef AGENT_V2_SEARCH
#define AGENT_V2_SEARCH 0
#endif
#ifndef AGENT_V2_GREEDY
#define AGENT_V2_GREEDY 0
#endif

#if defined(FUSION_TIER_M)
#include "models_m/registry.hpp"
#elif defined(FUSION_TIER_L)
#include "models_l/registry.hpp"
#elif defined(FUSION_TIER_XL)
#include "models_xl/registry.hpp"
#elif defined(FUSION_TIER_XXL)
#include "models_xxl/registry.hpp"
#elif defined(FUSION_TIER_PXL_VM)
#include "models_pxl_vm/registry.hpp"
#elif defined(FUSION_TIER_PXXL_VM)
#include "models_pxxl_vm/registry.hpp"
#else
#error "an agent_v2 variant selects our opponent model registry with FUSION_TIER_*"
#endif

#include "../../inference/agent_v2/runtime.hpp"
#if AGENT_V2_SEARCH
#include "../../inference/agent_v2/search.hpp"
using AgentV2Runtime = agent_v2::Runtime<agent_v2::Search>;
#else
using AgentV2Runtime = agent_v2::Runtime<agent_v2::NoSearch>;
#endif

namespace agent_v2_bot {

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
    static AgentV2Runtime rt;
    rt.greedy = AGENT_V2_GREEDY;
    std::string block, why;
    while (read_block(block)) {
        const auto started = std::chrono::steady_clock::now();
        const std::string reply = rt.decide(block, started, why);
        if (rt.tel.turns == 1)
            std::fprintf(stderr, "AGENTV2 carrier %s actions %s resolver %s/%s scenarios %d actor %s digest %s features %s/%s "
                         "predictor %s model %s search %d greedy %d\n", agent_v2::kCarrier, agent_v2::kActionSchema,
                         agent_v2::kResolverRule, rt.resolver.profile.name, agent_v2::kResolverScenarios, meow_rl_model::kName,
                         meow_rl_model::kDigest, agent_v2::kPolicyFeatures, agent_v2::kValueFeatures, agent_v2::PredictorP::kKind,
                         rt.pred.model_id(), AGENT_V2_SEARCH, AGENT_V2_GREEDY);
        if (!why.empty() && rt.tel.refused == 1) std::fprintf(stderr, "AGENTV2 refused turn %d: %s\n", rt.last_turn_seen, why.c_str());
#if AGENT_V2_SEARCH
        auto& q = rt.search;
        std::fprintf(stderr,
                     "AGENTV2 turns %ld refused %ld inconsistent %ld search_used %ld search_fallback %ld leaves %ld trajectories %ld "
                     "refuse_setup %ld refuse_trajectory %ld refuse_expand %ld refuse_critic %ld refuse_late %ld max_setup_us %ld "
                     "max_traj_us %ld max_expand_us %ld max_critic_us %ld max_search_us %ld branches %lld critic %s\n",
                     rt.tel.turns, rt.tel.refused, rt.tel.inconsistent, rt.tel.search_used, rt.tel.search_fallback, rt.tel.leaves,
                     q.trajectories, q.refuse_setup, q.refuse_trajectory, q.refuse_expand, q.refuse_critic, q.refuse_late,
                     q.max_setup_us, q.max_traj_us, q.max_expand_us, q.max_critic_us, q.max_search_us, rt.pred.counters.branches,
                     meow_rl_critic::kFormat);
#else
        std::fprintf(stderr, "AGENTV2 turns %ld refused %ld inconsistent %ld search_used %ld search_fallback %ld leaves %ld\n",
                     rt.tel.turns, rt.tel.refused, rt.tel.inconsistent, rt.tel.search_used, rt.tel.search_fallback, rt.tel.leaves);
#endif
        std::fflush(stderr);
        std::cout << reply << std::endl;
    }
    return 0;
}

}  // namespace agent_v2_bot

int main() { return agent_v2_bot::run(); }
