// The Meowmind RL CPU bot (task T0056, docs/MEOWMIND_RL_DEPLOYMENT.md): the unmodified Meowmind reference with its
// search replaced by an actor-direct choice of the joint movement, and the reference's frozen resolver.
//
// A variant (bots/meow_rl/variants/<name>.cpp) includes its generated actor header (inference/meow_rl/<model>/
// actor.hpp, namespace meow_rl_model, the repository's INT16 format) and then this file. Per turn:
//   1. read_turn and meow_rl::Session::observe: exactly the trainer's bridge code (tools/cpp/meow_rl/session.hpp);
//   2. the joint movement: one draw from the integer actor's softmax over the feasible mask at T = 1.05 with the
//      common per-turn uniform of sample.hpp (the predeclared deployment: PPO optimizes the sampled policy), or its
//      greedy joint with MEOW_RL_GREEDY=1 (a diagnostic mode, ties to the lowest index); with MEOW_RL_SEARCH, the
//      bounded value search of search.hpp, which plays that actor choice whenever it cannot finish in time;
//   3. Session::resolve: the reference's resolver and reply format.
// A block the session refuses (malformed input) is answered with every unit staying and waiting, counted, and the
// movement history is reset (Session::forget_history): the next turn never compares against stale positions.
// stderr: the model identity on the first turn, counters on the last (MEOWRL ...).
#pragma once

#define main meowmind_reference_main
#include "../meowmind_p/meowmind-ctf-374-20261011.cpp"
#undef main

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include "../../tools/cpp/meow_rl/session.hpp"
#include "../../tools/cpp/meow_rl/actor_int.hpp"
#include "sample.hpp"

#ifndef MEOW_RL_GREEDY
#define MEOW_RL_GREEDY 0
#endif

#ifndef MEOW_RL_SEARCH
#define MEOW_RL_SEARCH 0
#endif
#if MEOW_RL_SEARCH
#include "search.hpp"
#endif

namespace meow_rl_bot {

struct Counters {
    long turns = 0, refused = 0, search_used = 0, search_fallback = 0, leaves = 0;
};

inline std::string safe_reply(int turn) { return std::to_string(turn) + " S - S - S -"; }

inline int run() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    meow_rl::Session s;
    meow_rl::Observation obs;
    Counters c;
    uint64_t seed = 0;
#if MEOW_RL_SEARCH
    meow_rl_search::Search search;
#endif
    while (read_turn(s.g)) {
        auto started = std::chrono::steady_clock::now();
        if (c.turns++ == 0) {
            seed = meow_rl_sample::fnv1a(meow_rl_sample::canonical_init(s.g));
            std::fprintf(stderr, "MEOWRL model %s scheme %s digest %s identity %s greedy %d search %d\n", meow_rl_model::kName,
                         meow_rl_model::kScheme, meow_rl_model::kDigest, meow_rl_model::kIdentity, MEOW_RL_GREEDY,
                         MEOW_RL_SEARCH);
        }
        std::string reply, why = s.observe(obs);
        if (why.empty()) {
            int64_t sc[3][5];
            for (int k = 0; k < 3; k++)
                for (int m = 0; m < 5; m++) sc[k][m] = meow_rl_int::score(obs.feats[k][m]);
            int joint = MEOW_RL_GREEDY ? meow_rl_int::greedy(obs.feats, obs.joint_mask)
                                       : meow_rl_sample::draw(sc, obs.joint_mask, meow_rl_model::kActor.outExp,
                                                              meow_rl_model::kTemperature,
                                                              meow_rl_sample::uniform(seed, s.g.turn));
#if MEOW_RL_SEARCH
            auto r = search.choose(s, obs, sc, joint, started);
            c.leaves += r.leaves;
            if (r.used) c.search_used++, joint = r.joint;
            else c.search_fallback++;
#endif
            int actions[meow_rl::kUnits];
            why = s.resolve(joint, reply, actions);
        }
        if (!why.empty()) {
            if (c.refused++ == 0) std::fprintf(stderr, "MEOWRL refused turn %d: %s\n", s.g.turn, why.c_str());
            s.forget_history();
            reply = safe_reply(s.g.turn);
        }
        std::cout << reply << std::endl;
        if (s.g.turn == s.g.turns)
            std::fprintf(stderr, "MEOWRL turns %ld refused %ld search_used %ld search_fallback %ld leaves %ld\n", c.turns,
                         c.refused, c.search_used, c.search_fallback, c.leaves);
    }
    return 0;
}

}  // namespace meow_rl_bot

int main() { return meow_rl_bot::run(); }
