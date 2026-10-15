// Version 2 of the Meowmind RL bounded value search (MEOW_RL_SEARCH=2; task T0061, docs/MEOWMIND_RL_ALIGNED.md).
//
// The same two-ply rule as search.hpp (version 1, the T0056/T0057 search, unchanged and still built by MEOW_RL_SEARCH=1):
// MEOW_RL_SEARCH_CANDIDATES roots (the actor's own choice, then the other feasible joints by integer joint
// accumulator), MEOW_RL_SEARCH_SAMPLES opponent scenarios, two simultaneous plies on the reference's World, our ply-2
// continuation one draw of the integer actor, value r + V(leaf) with gamma 1 and the terminal 2 U - 1, the highest
// mean over scenarios, ties to the actor's order. Two things change.
//
// 1. Common random numbers. Every random stream is a function of (game seed, turn, scenario) only, never of the root
//    candidate, so all roots of a turn face the same opponent draws, simulator seeds and continuation uniform:
//        T      = splitmix64(game_seed ^ (turn * 0xD6E8FEB86659FD93))      game_seed: sample.hpp's FNV of the INIT
//        stream = splitmix64(T ^ (kind * 0x9E3779B97F4A7C15) ^ (scenario + 1) * 0xBF58476D1CE4E5B9)
//    kinds: 1 ply-1 opponent orders (Planner::rng before enemyOrders, drawn once per scenario and shared by the
//    roots), 2 ply-1 simulator seed, 3 ply-2 opponent orders (Planner::rng before enemyOrders at the successor),
//    4 the continuation uniform, 5 ply-2 simulator seed. Planner::rng is set before each use (it is read by nothing
//    else), so a turn's tree depends only on the observed state and the INIT, not on earlier searches.
//    tools/cpp/meow_rl/tree.cpp builds exactly these trees offline (no deadline) for the fixed-work diagnostics.
//
// 2. Admission. The block is read before the clock starts (bot.hpp), so the deadline MEOW_RL_SEARCH_DEADLINE_MS
//    (MEOW_RL_SEARCH_FIRST_MS on turn 1) leaves the rest of the 50 ms for parsing, the reply and scheduling. Work is
//    admitted only if its estimated cost fits before the deadline:
//      * before the scenarios and before each trajectory: the whole trajectory (ply 1, the ply-2 expansion, ply 2
//        and the critic);
//      * before the critic call: one critic call;
//      * after the evaluation: the deadline itself.
//    An estimate is max(prior, the largest cost measured so far in this game) x MEOW_RL_SEARCH_SAFETY_PCT / 100; the
//    priors (MEOW_RL_SEARCH_TRAJ_US, MEOW_RL_SEARCH_CRITIC_US) are per-variant settings frozen by the eight-worker
//    preflight, which records the worst trajectory and critic costs it saw. Any refusal discards the whole search and
//    plays the actor's choice (already computed, so the fallback is always ready); a partial comparison is never used.
//    This is an empirical gate, not a real-time proof: the preflight is what shows that no reply was late.
// Counters (bot.hpp's MEOWRL line): trajectories evaluated, refusals per stage, the largest trajectory, critic and
// search cost in microseconds.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#ifndef MEOW_RL_SEARCH_CANDIDATES
#define MEOW_RL_SEARCH_CANDIDATES 4
#endif
#ifndef MEOW_RL_SEARCH_SAMPLES
#define MEOW_RL_SEARCH_SAMPLES 2
#endif
#ifndef MEOW_RL_SEARCH_DEADLINE_MS
#define MEOW_RL_SEARCH_DEADLINE_MS 30
#endif
#ifndef MEOW_RL_SEARCH_FIRST_MS
#define MEOW_RL_SEARCH_FIRST_MS 1500
#endif
#ifndef MEOW_RL_SEARCH_TRAJ_US
#define MEOW_RL_SEARCH_TRAJ_US 4000
#endif
#ifndef MEOW_RL_SEARCH_CRITIC_US
#define MEOW_RL_SEARCH_CRITIC_US 2000
#endif
#ifndef MEOW_RL_SEARCH_SAFETY_PCT
#define MEOW_RL_SEARCH_SAFETY_PCT 150
#endif

namespace meow_rl_search2 {

constexpr int kCandidates = MEOW_RL_SEARCH_CANDIDATES, kSamples = MEOW_RL_SEARCH_SAMPLES, kVF = learning::VF;
static_assert(kCandidates >= 1 && kSamples >= 1, "at least one candidate and one sample");

enum Stream : uint64_t { kOpp1 = 1, kSim1 = 2, kOpp2 = 3, kCont = 4, kSim2 = 5 };

inline uint64_t turn_seed(uint64_t game_seed, int turn) {
    return meow_rl_sample::splitmix64(game_seed ^ (uint64_t(turn) * 0xD6E8FEB86659FD93ull));
}
inline uint64_t stream(uint64_t T, Stream kind, int scenario) {
    uint64_t s = meow_rl_sample::splitmix64(T ^ (uint64_t(kind) * 0x9E3779B97F4A7C15ull) ^
                                            (uint64_t(scenario) + 1) * 0xBF58476D1CE4E5B9ull);
    return s ? s : 1;   // xorshift state must not be zero
}

// The roots: the actor's choice first, then the other feasible joints by joint accumulator (stable: ties keep index
// order), kCandidates in all (fewer when fewer joints are feasible).
inline std::vector<int> roots(const meow_rl::Observation& obs, const int64_t (&sc)[3][5], int actor_choice) {
    auto z = [&](int j) { return sc[0][j / 25] + sc[1][j / 5 % 5] + sc[2][j % 5]; };
    std::vector<int> out{actor_choice}, rest;
    for (int j = 0; j < meow_rl::kJoint; j++)
        if (obs.joint_mask[j] && j != actor_choice) rest.push_back(j);
    std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) { return z(a) > z(b); });
    for (int j : rest) {
        if (int(out.size()) >= kCandidates) break;
        out.push_back(j);
    }
    return out;
}

// The ply-1 opponent orders of each scenario (shared by every root).
inline std::array<std::vector<Order>, kSamples> scenarios(meow_rl::Session& s, uint64_t T) {
    std::array<std::vector<Order>, kSamples> out;
    for (int smp = 0; smp < kSamples; smp++) {
        s.planner.rng = stream(T, kOpp1, smp);
        out[smp] = s.planner.enemyOrders(s.observation, s.prob, true);
    }
    return out;
}

inline double terminal_reward(const Game& state, int me) {
    double u = 0;
    for (int t = 0; t < state.teams; t++)
        if (t != me) u += state.score[me] > state.score[t] ? 1.0 : state.score[me] == state.score[t] ? 0.5 : 0.0;
    return 2.0 * u / (state.teams - 1) - 1.0;
}

// One trajectory's leaf: a terminal reward, or the 416 value features of the prepared leaf state for the critic.
struct Leaf {
    bool terminal = false;
    int plies = 0;            // 1: the game ended in ply 1; 2: two plies
    double reward = 0;        // terminal only
    int cont = -1;            // our ply-2 continuation joint (-1: all stay, the safe fallback)
    std::array<float, kVF> feats{};
};

// Ply 1 (root joint `joint`, scenario orders `opp1`), the ply-2 expansion and ply 2, as search.hpp, with the
// scenario's common streams. The planner's history, new-life flags and prepared turn are restored before returning.
inline void trajectory(meow_rl::Session& s, int joint, const std::vector<Order>& opp1, uint64_t T, int smp, Leaf& leaf) {
    const Game& g = s.g;
    const Game& o = s.observation;
    Planner& p = s.planner;
    Planner::Candidate ca;
    ca.moves = {joint / 25, joint / 5 % 5, joint % 5};
    ca.actions = p.shots(o, g.me, ca.moves, s.prob);
    Game next = p.step(g, ca, opp1, stream(T, kSim1, smp));
    leaf = Leaf{};
    if (next.turn > g.turns) {
        leaf.terminal = true, leaf.plies = 1, leaf.reward = terminal_reward(next, g.me);
        return;
    }
    auto savedHistory = p.previous;
    auto savedLife = p.newLife;
    int savedTurn = p.preparedTurn;
    for (auto& u : o.units) p.previous[u.id] = p.geo.cell(u);
    Game fo = learning::prepared(next);
    p.preparedTurn = next.turn;
    for (auto& u : fo.units) p.newLife[u.id] = next.units[u.id].x < 0 && u.x >= 0;
    auto pp = p.predict(fo);
    meow_rl::Observation succ;
    Planner::Candidate cont;
    if (meow_rl::fill(p, fo, g.me, nullptr, succ)) {
        int64_t s2[3][5];
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < 5; m++) s2[k][m] = meow_rl_int::score(succ.feats[k][m]);
        double u = double(stream(T, kCont, smp) >> 11) * 0x1.0p-53;
        int j2 = meow_rl_sample::draw(s2, succ.joint_mask, meow_rl_model::kActor.outExp, meow_rl_model::kTemperature, u);
        cont.moves = {j2 / 25, j2 / 5 % 5, j2 % 5};
        cont.actions = p.shots(fo, g.me, cont.moves, pp);
        leaf.cont = j2;
    }                                                        // else all stay and wait (the safe fallback)
    p.rng = stream(T, kOpp2, smp);
    auto eo = p.enemyOrders(fo, pp, true);
    p.previous = savedHistory;
    p.newLife = savedLife;
    p.preparedTurn = savedTurn;
    Game end = p.step(next, cont, eo, stream(T, kSim2, smp));
    leaf.plies = 2;
    if (end.turn > g.turns) {
        leaf.terminal = true, leaf.reward = terminal_reward(end, g.me);
        return;
    }
    auto f = p.geo.value(learning::prepared(end), g.me);
    std::copy(f.begin(), f.end(), leaf.feats.begin());
}

#ifndef MEOW_RL_TREE_ONLY

struct Result {
    bool used = false;   // false: the actor's choice is played
    int joint = -1;
    long leaves = 0;     // critic evaluations
    long trajectories = 0;
    int refused = 0;     // 0 none, 1 before a trajectory, 2 before the critic, 3 after an evaluation
};

struct Search {
    long max_traj_us = 0, max_critic_us = 0, max_search_us = 0;   // the largest costs measured in this game

    static long us(std::chrono::steady_clock::duration d) {
        return long(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
    }
    static long estimate(long prior, long seen) { return std::max(prior, seen) * MEOW_RL_SEARCH_SAFETY_PCT / 100; }

    Result choose(meow_rl::Session& s, const meow_rl::Observation& obs, const int64_t (&sc)[3][5], int actor_choice,
                  uint64_t game_seed, std::chrono::steady_clock::time_point started) {
        using clock = std::chrono::steady_clock;
        Result r;
        auto t_begin = clock::now();
        auto deadline = started + std::chrono::milliseconds(s.g.turn == 1 ? MEOW_RL_SEARCH_FIRST_MS : MEOW_RL_SEARCH_DEADLINE_MS);
        auto admit = [&](long cost_us) { return clock::now() + std::chrono::microseconds(cost_us) <= deadline; };
        auto finish = [&](Result res) {
            max_search_us = std::max(max_search_us, us(clock::now() - t_begin));
            return res;
        };
        const long traj_est = estimate(MEOW_RL_SEARCH_TRAJ_US, max_traj_us);
        if (!admit(traj_est)) return finish(Result{false, -1, 0, 0, 1});
        uint64_t T = turn_seed(game_seed, s.g.turn);
        auto rs = roots(obs, sc, actor_choice);
        auto opp = scenarios(s, T);
        double best = -1e300;
        Leaf leaf;
        for (int ri = 0; ri < int(rs.size()); ri++) {
            double sum = 0;
            for (int smp = 0; smp < kSamples; smp++) {
                if (!admit(estimate(MEOW_RL_SEARCH_TRAJ_US, max_traj_us))) return finish(Result{false, -1, r.leaves, r.trajectories, 1});
                auto t0 = clock::now();
                trajectory(s, rs[ri], opp[smp], T, smp, leaf);
                double v = leaf.reward;
                if (!leaf.terminal) {
                    if (!admit(estimate(MEOW_RL_SEARCH_CRITIC_US, max_critic_us)))
                        return finish(Result{false, -1, r.leaves, r.trajectories, 2});
                    auto t1 = clock::now();
                    v = meow_rl_critic::value(leaf.feats.data());
                    max_critic_us = std::max(max_critic_us, us(clock::now() - t1));
                    r.leaves++;
                }
#ifdef MEOW_RL_SEARCH_TEST_INJECT_DELAY_MS
                std::this_thread::sleep_for(std::chrono::milliseconds(MEOW_RL_SEARCH_TEST_INJECT_DELAY_MS));
#endif
                max_traj_us = std::max(max_traj_us, us(clock::now() - t0));
                r.trajectories++;
                if (clock::now() > deadline) return finish(Result{false, -1, r.leaves, r.trajectories, 3});
                sum += v;
            }
            double v = sum / kSamples;
            if (v > best) best = v, r.joint = rs[ri];
        }
        r.used = true;
        return finish(r);
    }
};

#endif  // MEOW_RL_TREE_ONLY

}  // namespace meow_rl_search2
