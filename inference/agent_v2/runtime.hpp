// The agent_v2 runtime: one episode of one seat (task T0065, docs/AGENT_V2.md).
//
//   block text --parse (stream.hpp)--> Static (INIT) + raw State          one public state
//        |  validate; MapCache (Board) and the feature schema's geometry on the first block
//        |-> P.observe(raw, previous real State)                         our opponent model
//        |-> makeView(raw, parent): origins, respawn immunity             engine facts
//        |-> encoder (meow_policy112_v1 / meow_public416_v1)             the frozen actor/V inputs
//        |-> actor pi: one sampled joint over the feasible mask (common per-turn uniform), or greedy
//        |-> resolver assign_v1 on P's scenario table -> COMPLETE DEFAULT ORDER, ready before any planning
//        |-> [optional search: a joint; any refusal keeps the default]
//        '-> reply (protocol text) ; History.advance(raw) exactly once
//
// The PPO bridge calls observe_block() and answer(joint) itself (tools/cpp/agent_v2/bridge.cpp): the same parse,
// predictor, encoder, resolver and reply as the bot, with the trainer's sampled joint. Include the actor header
// (namespace meow_rl_model) and the model registry before this file.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "core.hpp"
#include "features_meow_v1.hpp"
#include "predictor.hpp"
#include "resolver.hpp"
#include "../../tools/cpp/stream.hpp"
#include "../../tools/cpp/meow_rl/actor_int.hpp"

#ifndef AGENT_V2_RESOLVER_SCENARIOS
#define AGENT_V2_RESOLVER_SCENARIOS 8
#endif

namespace agent_v2 {

inline constexpr int kResolverScenarios = AGENT_V2_RESOLVER_SCENARIOS;

// The actor's per-turn uniforms depend only on the public INIT (bots/meow_rl/sample.hpp's rule).
inline uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}
inline std::string canonical_init(const mm::Game& g) {
    std::string s = "meow_rl_actor_v1\n" + std::to_string(g.teams) + " " + std::to_string(g.me) + " " + std::to_string(g.size) +
                    " " + std::to_string(g.turns) + "\n";
    for (auto& row : g.map) s += row + "\n";
    for (auto& b : g.bases) s += std::to_string(b.first) + " " + std::to_string(b.second) + "\n";
    for (auto& p : g.spots) s += std::to_string(p.first) + " " + std::to_string(p.second) + "\n";
    return s;
}
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
inline double uniform(uint64_t seed, int turn) {
    return double(splitmix64(seed + uint64_t(turn) * 0x9E3779B97F4A7C15ull) >> 11) * 0x1.0p-53;
}

// The actor pi over the meow schema: integer unit-move scores, the joint softmax at T over the feasible mask.
struct Actor {
    static void scores(const Observation& obs, int64_t (&sc)[3][5]) {
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < 5; m++) sc[k][m] = meow_rl_int::score(obs.feats[k][m]);
    }
    // One inverse-CDF draw with uniform u (sample.hpp's draw).
    static int sample(const int64_t (&s)[3][5], const Observation& obs, double u) {
        double z[125], top = -1e300;
        for (int j = 0; j < 125; j++)
            if (obs.joint_mask[j]) {
                z[j] = std::ldexp(double(s[0][j / 25] + s[1][j / 5 % 5] + s[2][j % 5]), -meow_rl_model::kActor.outExp) /
                       meow_rl_model::kTemperature;
                if (z[j] > top) top = z[j];
            }
        double total = 0, p[125];
        for (int j = 0; j < 125; j++) total += p[j] = obs.joint_mask[j] ? std::exp(z[j] - top) : 0.0;
        double target = u * total, acc = 0;
        int last = 0;
        for (int j = 0; j < 125; j++)
            if (obs.joint_mask[j]) {
                acc += p[j];
                last = j;
                if (acc > target) return j;
            }
        return last;
    }
    static int greedy(const Observation& obs) { return meow_rl_int::greedy(obs.feats, obs.joint_mask); }
};

// Protocol text of complete orders: the turn, then per unit the move letter and the action token
// (target id, P pickup, - wait, X drop).
inline std::string reply_text(int turn, const std::array<fusion::Order, 3>& orders) {
    std::string r = std::to_string(turn);
    for (auto& o : orders) {
        r += ' ';
        r += fusion::kMoveChar[o.move];
        r += ' ';
        r += o.act == fusion::kOrderAttack ? std::to_string(o.target) : o.act == fusion::kOrderPickup ? "P" : o.act == fusion::kOrderDrop ? "X" : "-";
    }
    return r;
}

struct Telemetry {
    long turns = 0, refused = 0, search_used = 0, search_fallback = 0, leaves = 0, inconsistent = 0;
};

struct NoSearch {
    template <class R>
    int choose(R&, const int64_t (&)[3][5], int joint, std::chrono::steady_clock::time_point) { return joint; }
};

// The resolver's scenario salt of a real or simulated turn: the game seed, the turn and a purpose.
inline uint32_t resolver_salt(uint64_t game_seed, int turn, uint64_t purpose) {
    return uint32_t(splitmix64(game_seed ^ (uint64_t(turn) << 20) ^ purpose * 0xA24BAED4963EE407ull));
}

template <class Search = NoSearch>
struct Runtime {
    MapCache cache;
    MeowEncoder enc;
    History hist;
    PredictorP pred;
    Resolver resolver;
    Search search;
    fusion::State s;                 // the raw state of the turn being answered
    fusion::TurnView view;           // its engine facts (origins, immunity), with the real parent
    Observation obs;
    Tactics tactics;                 // the order that will be sent
    Telemetry tel;
    uint64_t seed = 0;
    int observed_turn = -1, last_turn_seen = 0;
    bool answered = true, greedy = false;

    // Parses one block (INIT + TURN on the first call, then a TURN block) and observes it. "" or why it is refused.
    std::string observe_block(const std::string& text) {
        tel.turns++;
        try {
            std::istringstream in(text);
            if (text.rfind("INIT", 0) == 0) {
                fusion::Static st;
                if (!fusion::readInit(in, st)) return "empty INIT block";
                if (cache.ready) return "a second INIT block";
                cache.init(st);
                enc.init(st);
                pred.begin(cache);
                seed = fnv1a(canonical_init(enc.base));
            }
            fusion::State next;
            if (!fusion::readTurn(in, next)) return "block without TURN";
            s = std::move(next);
            last_turn_seen = s.turn;
        } catch (const std::exception& e) {
            return std::string("malformed block: ") + e.what();
        }
        return observe_turn();
    }

    std::string observe_turn() {
        if (!cache.ready) return "no INIT block before the first TURN";
        if (auto why = invalid(cache.st, s); !why.empty()) return "malformed block: " + why;
        if (s.turn <= observed_turn) return "turns must increase";
        pred.observe(cache, s, hist.parent());
        fusion::makeView(cache.board, s, hist.parent(), view);
        if (!enc.encode(s, hist.parent(), cache.st.me, obs)) return "the all-stay joint move is infeasible";
        answered = false;
        return "";
    }

    // The complete orders of joint `joint` on the real turn (assign_v1 on P's scenario table).
    Tactics resolve(int joint) {
        std::vector<Scenario> table;
        if (!pred.scenarios(pred.root, s, kResolverScenarios, resolver_salt(seed, s.turn, 1), table)) tel.inconsistent++;
        const std::array<int, 3> moves{joint / 25, joint / 5 % 5, joint % 5};
        return resolver.resolve(cache, s, view, cache.st.me, moves, table, splitmix64(seed ^ uint64_t(s.turn) ^ 0xF1A6ull));
    }

    // The bot's turn. A refusal answers with every unit staying and waiting.
    std::string decide(const std::string& block, std::chrono::steady_clock::time_point started, std::string& why) {
        why = observe_block(block);
        if (!why.empty()) return refuse();
        int64_t sc[3][5];
        Actor::scores(obs, sc);
        const int joint = greedy ? Actor::greedy(obs) : Actor::sample(sc, obs, uniform(seed, s.turn));
        tactics = resolve(joint);                                    // the complete default order, ready
        const int chosen = search.choose(*this, sc, joint, started);
        if (chosen != joint) tactics = resolve(chosen);
        return commit();
    }

    // The bridge's answer to an observed turn with feasible joint `joint` (validated by the caller).
    std::string answer(int joint) {
        tactics = resolve(joint);
        return commit();
    }

    std::string commit() {
        std::string reply = reply_text(s.turn, tactics.orders);
        hist.advance(s);
        observed_turn = s.turn;
        answered = true;
        return reply;
    }

    std::string refuse() {
        tel.refused++;
        hist.forget();
        answered = true;
        return std::to_string(last_turn_seen) + " S - S - S -";
    }
};

}  // namespace agent_v2
