// The frozen direct continuation policy pi_c of the branch diagnostics (task T0085, tools/search_diagnostics/README.md).
//
// bots/agent_v2/bot.hpp's direct loop (AGENT_V2_SEARCH 0, greedy 0) with one diagnostic switch: an actor scenario key.
//   continuation_bot <key> <from_turn>
// From turn `from_turn` on, the runtime's game seed (which keys the actor's per-turn uniform and the resolver's
// scenario tables, inference/agent_v2/runtime.hpp) becomes splitmix64(INIT seed ^ key). Before that turn, and with
// key 0, every reply is the production bot's: the branch collector replays the causal prefix and the root turn with
// the deployed randomness, then repeats differ through registered keys shared by every candidate of a repeat. The
// runtime's persistent state is a function of the observed blocks and the seed only (History keeps observed states,
// not the orders sent), so a forced root order leaves no hidden inconsistency in this process.
// Built per package by tools/search_diagnostics/build.py: a generated wrapper defines FUSION_MAX_WIDTH and FUSION_TIER_*,
// includes the package's actor header (namespace meow_rl_model), then this file.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../../inference/agent_v2/runtime.hpp"

namespace agent_v2_bot {
inline int g_active_teams = 0;   // read by count-routed actors (bots/agent_v2/variants/agent_v2_rl_csp.cpp)
}

namespace {

bool read_block(std::string& block) {
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: continuation_bot <key> <from_turn>\n");
        return 2;
    }
    const uint64_t key = std::strtoull(argv[1], nullptr, 10);
    const int from = std::atoi(argv[2]);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    static agent_v2::Runtime<agent_v2::NoSearch> rt;
    uint64_t base = 0;
    std::string block;
    while (read_block(block)) {
        if (agent_v2_bot::g_active_teams == 0 && block.rfind("INIT", 0) == 0) {
            const char* p = block.c_str() + 4;
            while (*p == ' ') p++;
            agent_v2_bot::g_active_teams = std::atoi(p);
        }
        const std::string why = rt.observe_block(block);
        if (rt.tel.turns == 1) base = rt.seed;
        std::string reply;
        if (!why.empty()) {
            reply = rt.refuse();
        } else {
            rt.seed = key && rt.s.turn >= from ? agent_v2::splitmix64(base ^ key) : base;
            int64_t sc[3][5];
            agent_v2::Actor::scores(rt.obs, sc);
            const int joint = agent_v2::Actor::sample(sc, rt.obs, agent_v2::uniform(rt.seed, rt.s.turn));
            rt.tactics = rt.resolve(joint);
            reply = rt.commit();
        }
        if (rt.tel.turns == 1)
            std::fprintf(stderr, "AGENTV2 carrier %s actor %s digest %s predictor %s model %s search 0 greedy 0 diag_key %llu from %d\n",
                         agent_v2::kCarrier, meow_rl_model::kName, meow_rl_model::kDigest, agent_v2::PredictorP::kKind,
                         rt.pred.model_id(), (unsigned long long)key, from);
        if (!why.empty() && rt.tel.refused == 1) std::fprintf(stderr, "AGENTV2 refused turn %d: %s\n", rt.last_turn_seen, why.c_str());
        std::fprintf(stderr, "AGENTV2 turns %ld refused %ld inconsistent %ld\n", rt.tel.turns, rt.tel.refused, rt.tel.inconsistent);
        std::cout << reply << std::endl;
    }
    return 0;
}
