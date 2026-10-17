// T0075 test probe for the agent_v2 controls (tests/test_next_controls.py). Include it after a control's bundle,
// compiled with `main` renamed to agent_v2_variant_main, and with CONTROL_EXPECTED_ACTOR naming the fusion::Net that
// every count must route to:
//
//   #define main agent_v2_variant_main
//   #include "<bundle>.cpp"
//   #undef main
//   #define CONTROL_EXPECTED_ACTOR fusion::models::rl_general::kActor
//   #include "probe.hpp"
//
//   probe play     the bundle's own agent_v2_bot::run() on stdin (the deployed loop, unchanged)
//   probe routing  per team count 0..16: the actor kActor resolves to, and its outExp; nonzero exit on a mismatch
//   probe dump     agent_v2_bot::run()'s loop on stdin, writing per answered turn to stdout the observation
//                  (float32 feats[3][5][112], uint8 joint_mask[125]) and the integer unit scores (int64 [3][5]) the
//                  actor computed; replies and counters go to stderr. Nothing here is a timing measurement.
#pragma once
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifndef CONTROL_EXPECTED_ACTOR
#error "define CONTROL_EXPECTED_ACTOR (the fusion::Net every count routes to)"
#endif

namespace control_probe {

inline int routing() {
    int bad = 0;
    const fusion::Net& expected = CONTROL_EXPECTED_ACTOR;
    for (int teams = 0; teams <= 16; teams++) {
        agent_v2_bot::g_active_teams = teams;
        const fusion::Net& net = meow_rl_model::kActor;
        const bool same = &net == &expected;
        std::printf("ROUTE teams %d expected %d outExp %d header_outExp %d\n", teams, int(same), meow_rl_model::kActor.outExp,
                    net.outExp);
        bad += !same || net.outExp != meow_rl_model::kActor.outExp;
    }
    agent_v2_bot::g_active_teams = 0;
    return bad ? 1 : 0;
}

inline int dump() {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);  // raw records: no CRLF translation
#endif
    static AgentV2Runtime rt;
    rt.greedy = AGENT_V2_GREEDY;
    std::string block, why;
    long answered = 0;
    while (agent_v2_bot::read_block(block)) {
        if (agent_v2_bot::g_active_teams == 0 && block.rfind("INIT", 0) == 0) {
            const char* p = block.c_str() + 4;
            while (*p == ' ') p++;
            agent_v2_bot::g_active_teams = std::atoi(p);
        }
        const std::string reply = rt.decide(block, std::chrono::steady_clock::now(), why);
        std::fprintf(stderr, "%s\n", reply.c_str());
        if (!why.empty()) continue;
        int64_t sc[3][5];
        agent_v2::Actor::scores(rt.obs, sc);
        std::fwrite(rt.obs.feats, sizeof(float), 3 * 5 * 112, stdout);
        std::fwrite(rt.obs.joint_mask, 1, 125, stdout);
        std::fwrite(sc, sizeof(int64_t), 15, stdout);
        answered++;
    }
    std::fprintf(stderr, "DUMP teams %d answered %ld refused %ld\n", agent_v2_bot::g_active_teams, answered, rt.tel.refused);
    return 0;
}

}  // namespace control_probe

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "play") return agent_v2_variant_main();
    if (mode == "routing") return control_probe::routing();
    if (mode == "dump") return control_probe::dump();
    std::fprintf(stderr, "usage: probe play|routing|dump\n");
    return 2;
}
