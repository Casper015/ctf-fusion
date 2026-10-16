// Optional bounded two-ply value search of the agent_v2 carrier (AGENT_V2_SEARCH=1; task T0065, docs/AGENT_V2.md).
//
// Search v2's machinery (bots/meow_rl/search_v2.hpp, T0061) on this carrier's state, predictor, resolver and simulator:
//   * roots: the actor's joint, then the next feasible joints by integer accumulator (kCandidates in all), each with
//     its complete orders from the resolver (assign_v1 on the real turn's P scenario table, the same call the default
//     order uses);
//   * scenarios: kSamples complete opponent order sets drawn from our P at the root, shared by every root;
//   * ply 1: fusion::step (Simulator, lawful sampled flag refreshes);
//   * the successor: its own P prediction and engine facts whose parent is the real root state (the immediate raw
//     parent), the actor's continuation (one draw over its feasible mask, history = the root's raw cells), its
//     complete orders from the resolver on the branch's own scenario table, the opponents' ply-2 orders from the branch
//     prediction; nothing of a branch is written back (the root prediction, the history and the real state stay);
//   * ply 2: fusion::step; value = terminal 2U - 1 or V(meow_public416_v1 of the leaf), gamma 1;
//   * choice: the highest scenario mean; ties keep the actor's order. Common streams per (game seed, turn, scenario).
// Admission (staged, v2): setup, each trajectory, the ply-2 expansion and the critic are admitted only if their
// estimate (max(prior, worst seen this game) x safety, plus a reply reserve) fits before the deadline; a refusal keeps
// the default order, which the runtime resolved before the search started. Include the critic header
// (namespace meow_rl_critic) before this file.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include "runtime.hpp"
#include "simulate.hpp"

#ifndef AGENT_V2_SEARCH_CANDIDATES
#define AGENT_V2_SEARCH_CANDIDATES 4
#endif
#ifndef AGENT_V2_SEARCH_SAMPLES
#define AGENT_V2_SEARCH_SAMPLES 2
#endif
#ifndef AGENT_V2_BRANCH_SCENARIOS
#define AGENT_V2_BRANCH_SCENARIOS 4
#endif
#ifndef AGENT_V2_SEARCH_DEADLINE_MS
#define AGENT_V2_SEARCH_DEADLINE_MS 30
#endif
#ifndef AGENT_V2_SEARCH_FIRST_MS
#define AGENT_V2_SEARCH_FIRST_MS 1500
#endif
#ifndef AGENT_V2_SEARCH_SETUP_US
#define AGENT_V2_SEARCH_SETUP_US 3000
#endif
#ifndef AGENT_V2_SEARCH_TRAJ_US
#define AGENT_V2_SEARCH_TRAJ_US 8000
#endif
#ifndef AGENT_V2_SEARCH_EXPAND_US
#define AGENT_V2_SEARCH_EXPAND_US 6000
#endif
#ifndef AGENT_V2_SEARCH_CRITIC_US
#define AGENT_V2_SEARCH_CRITIC_US 2000
#endif
#ifndef AGENT_V2_SEARCH_REPLY_US
#define AGENT_V2_SEARCH_REPLY_US 1000
#endif
#ifndef AGENT_V2_SEARCH_SAFETY_PCT
#define AGENT_V2_SEARCH_SAFETY_PCT 150
#endif

namespace agent_v2 {

constexpr int kCandidates = AGENT_V2_SEARCH_CANDIDATES, kSamples = AGENT_V2_SEARCH_SAMPLES;
constexpr int kBranchScenarios = AGENT_V2_BRANCH_SCENARIOS;

enum Stream : uint64_t { kOpp1 = 1, kSim1 = 2, kOpp2 = 3, kCont = 4, kSim2 = 5, kBranchResolver = 6, kBranchRefresh = 7 };
inline uint64_t turn_seed(uint64_t game_seed, int turn) { return splitmix64(game_seed ^ (uint64_t(turn) * 0xD6E8FEB86659FD93ull)); }
inline uint64_t stream(uint64_t T, Stream kind, int scenario) {
    uint64_t s = splitmix64(T ^ (uint64_t(kind) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(scenario) + 1) * 0xBF58476D1CE4E5B9ull);
    return s ? s : 1;
}

inline std::vector<int> roots(const Observation& obs, const int64_t (&sc)[3][5], int actor_choice) {
    auto z = [&](int j) { return sc[0][j / 25] + sc[1][j / 5 % 5] + sc[2][j % 5]; };
    std::vector<int> out{actor_choice}, rest;
    for (int j = 0; j < kJoint; j++)
        if (obs.joint_mask[j] && j != actor_choice) rest.push_back(j);
    std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) { return z(a) > z(b); });
    for (int j : rest) {
        if (int(out.size()) >= kCandidates) break;
        out.push_back(j);
    }
    return out;
}

inline double terminal_reward(const fusion::State& s, int me) {
    double u = 0;
    const int teams = int(s.score.size());
    for (int t = 0; t < teams; t++)
        if (t != me) u += s.score[me] > s.score[t] ? 1.0 : s.score[me] == s.score[t] ? 0.5 : 0.0;
    return 2.0 * u / (teams - 1) - 1.0;
}

struct Leaf {
    bool terminal = false;
    int plies = 0;
    double reward = 0;
    int cont = -1;
    std::array<fusion::Order, 3> root_orders{}, cont_orders{};
    std::array<float, kVF> feats{};
};

struct Always {
    bool operator()() const { return true; }
};

// The shared setup of one turn's tree: roots, their complete orders, and the ply-1 opponent orders per scenario.
struct Tree {
    uint64_t T = 0;
    std::vector<int> roots;
    std::vector<std::array<fusion::Order, 3>> root_orders;
    std::array<std::vector<fusion::Order>, kSamples> opp;
};

template <class R>
void setup(R& rt, const int64_t (&sc)[3][5], int actor_choice, Tree& tree) {
    tree.T = turn_seed(rt.seed, rt.s.turn);
    tree.roots = roots(rt.obs, sc, actor_choice);
    tree.root_orders.clear();
    for (int j : tree.roots) tree.root_orders.push_back(rt.resolve(j).orders);
    for (int smp = 0; smp < kSamples; smp++)
        PredictorP::opponent_orders(rt.pred.root, rt.cache.st.me, smp, uint32_t(stream(tree.T, kOpp1, smp)), tree.opp[smp]);
}

// One trajectory of root ri in scenario smp. Returns false when `expand` refuses the ply-2 expansion.
template <class R, class Gate = Always>
bool trajectory(R& rt, const Tree& tree, int ri, int smp, Simulator& sim, Leaf& leaf, Gate&& expand = Gate{}) {
    const int me = rt.cache.st.me;
    const fusion::State& s = rt.s;
    leaf = Leaf{};
    leaf.root_orders = tree.root_orders[ri];
    std::vector<fusion::Order> orders = tree.opp[smp];
    for (int k = 0; k < 3; k++) orders[3 * me + k] = leaf.root_orders[k];
    fusion::State next = sim.step(rt.cache, s, orders, stream(tree.T, kSim1, smp));
    if (next.turn > rt.cache.st.turns) {
        leaf.terminal = true, leaf.plies = 1, leaf.reward = terminal_reward(next, me);
        return true;
    }
    if (!expand()) return false;
    Prediction branch;
    rt.pred.branch(rt.cache, next, branch);                         // parent: the real root state
    fusion::TurnView view;
    fusion::makeView(rt.cache.board, next, &s, view);
    Observation succ;
    std::array<int, 3> cont{0, 0, 0};
    if (rt.enc.encode(next, &s, me, succ)) {
        int64_t s2[3][5];
        Actor::scores(succ, s2);
        const int j2 = Actor::sample(s2, succ, double(stream(tree.T, kCont, smp) >> 11) * 0x1.0p-53);
        cont = {j2 / 25, j2 / 5 % 5, j2 % 5};
        leaf.cont = j2;
    }                                                               // else all stay (the safe fallback)
    std::vector<Scenario> table;
    rt.pred.scenarios(branch, next, kBranchScenarios, uint32_t(stream(tree.T, kBranchResolver, smp)), table);
    leaf.cont_orders = rt.resolver.resolve(rt.cache, next, view, me, cont, table, stream(tree.T, kBranchRefresh, smp)).orders;
    std::vector<fusion::Order> orders2;
    PredictorP::opponent_orders(branch, me, smp, uint32_t(stream(tree.T, kOpp2, smp)), orders2);
    for (int k = 0; k < 3; k++) orders2[3 * me + k] = leaf.cont_orders[k];
    fusion::State end = sim.step(rt.cache, next, orders2, stream(tree.T, kSim2, smp));
    leaf.plies = 2;
    if (end.turn > rt.cache.st.turns) {
        leaf.terminal = true, leaf.reward = terminal_reward(end, me);
        return true;
    }
    leaf.feats = rt.enc.value(end, me);
    return true;
}

#ifndef AGENT_V2_TREE_ONLY
struct Search {
    Simulator sim;
    long max_setup_us = 0, max_traj_us = 0, max_expand_us = 0, max_critic_us = 0, max_search_us = 0;
    long trajectories = 0, refuse_setup = 0, refuse_trajectory = 0, refuse_expand = 0, refuse_critic = 0, refuse_late = 0;

    static long us(std::chrono::steady_clock::duration d) { return long(std::chrono::duration_cast<std::chrono::microseconds>(d).count()); }
    static long estimate(long prior, long seen) { return std::max(prior, seen) * AGENT_V2_SEARCH_SAFETY_PCT / 100; }

    template <class R>
    int choose(R& rt, const int64_t (&sc)[3][5], int joint, std::chrono::steady_clock::time_point started) {
        using clock = std::chrono::steady_clock;
        const auto t_begin = clock::now();
        const auto deadline = started + std::chrono::milliseconds(rt.s.turn == 1 ? AGENT_V2_SEARCH_FIRST_MS : AGENT_V2_SEARCH_DEADLINE_MS);
        const long reserve = estimate(AGENT_V2_SEARCH_REPLY_US, 0);
        auto admit = [&](long cost) { return clock::now() + std::chrono::microseconds(cost + reserve) <= deadline; };
        auto fallback = [&](long& counter) {
            counter++;
            rt.tel.search_fallback++;
            max_search_us = std::max(max_search_us, us(clock::now() - t_begin));
            return joint;
        };
        if (!admit(estimate(AGENT_V2_SEARCH_SETUP_US, max_setup_us) + estimate(AGENT_V2_SEARCH_TRAJ_US, max_traj_us)))
            return fallback(refuse_setup);
        Tree tree;
        setup(rt, sc, joint, tree);
        max_setup_us = std::max(max_setup_us, us(clock::now() - t_begin));
        double best = -1e300;
        int choice = joint;
        Leaf leaf;
        for (int ri = 0; ri < int(tree.roots.size()); ri++) {
            double sum = 0;
            for (int smp = 0; smp < kSamples; smp++) {
                if (!admit(estimate(AGENT_V2_SEARCH_TRAJ_US, max_traj_us))) return fallback(refuse_trajectory);
                const auto t0 = clock::now();
                clock::time_point t_expand{};
                auto expand = [&] {
                    t_expand = clock::now();
                    return admit(estimate(AGENT_V2_SEARCH_EXPAND_US, max_expand_us));
                };
                auto spent = [&] { max_traj_us = std::max(max_traj_us, us(clock::now() - t0)); };
                if (!trajectory(rt, tree, ri, smp, sim, leaf, expand)) {
                    spent();
                    return fallback(refuse_expand);
                }
                if (leaf.plies == 2) max_expand_us = std::max(max_expand_us, us(clock::now() - t_expand));
                double v = leaf.reward;
                if (!leaf.terminal) {
                    if (!admit(estimate(AGENT_V2_SEARCH_CRITIC_US, max_critic_us))) {
                        spent();
                        return fallback(refuse_critic);
                    }
                    const auto t1 = clock::now();
                    v = meow_rl_critic::value(leaf.feats.data());
                    max_critic_us = std::max(max_critic_us, us(clock::now() - t1));
                    rt.tel.leaves++;
                }
                spent();
                trajectories++;
                if (clock::now() > deadline) return fallback(refuse_late);
                sum += v;
            }
            const double v = sum / kSamples;
            if (v > best) best = v, choice = tree.roots[ri];
        }
        rt.tel.search_used++;
        max_search_us = std::max(max_search_us, us(clock::now() - t_begin));
        return choice;
    }
};
#endif

}  // namespace agent_v2
