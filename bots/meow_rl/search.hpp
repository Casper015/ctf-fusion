// The separately switchable bounded value search of the Meowmind RL bot (MEOW_RL_SEARCH=1; docs/MEOWMIND_RL_DEPLOYMENT.md).
//
// Two-ply here means two complete simultaneous turns, the reference's own deep-search shape (Planner::run's beam):
//   * roots: the actor's own choice for this turn (the sampled joint, or greedy in MEOW_RL_GREEDY mode), then the
//     highest-scoring other feasible joints by integer joint accumulator, MEOW_RL_SEARCH_CANDIDATES in all; each is
//     completed by the frozen resolver (Planner::shots) exactly as Session::resolve would;
//   * scenarios: MEOW_RL_SEARCH_SAMPLES stochastic opponent order sets from the reference's Planner::enemyOrders,
//     drawn once per turn and shared by every root;
//   * ply 1: Planner::step on the raw block (the reference's World and seeds);
//   * ply 2 (unless ply 1 ended the game): the opponent forecasts are recomputed at the successor (predict without
//     saving, as the reference's beam does: the real turn's evidence is frozen), our continuation is one draw of the
//     integer actor at the successor under its own mask (history: the root's positions), completed by the resolver
//     with the successor's forecasts, and the opponents' orders are drawn there; then Planner::step again;
//   * value of a trajectory: r1 + r2 + V(leaf) with gamma 1. Rewards are 0 except the terminal 2 U - 1, counted once
//     and never bootstrapped; V is the trained critic on the public value features of the prepared leaf, in the same
//     2 U - 1 units. The reference's own reward(), value() and potential (capture/kill terms, discount 0.97) are not
//     used;
//   * choice: the highest mean over scenarios; equal values keep the actor's order (the actor's choice first).
// State: the movement history, new-life flags and prepared turn are saved before and restored after every ply-2
// expansion; only per-turn scratch arrays and the reference's search RNG (Planner::rng, used by enemyOrders) advance,
// as in the reference. Our own continuation draws use a separate generator seeded from the turn. Nothing is cached
// across branches; only the immutable map geometry is shared.
// Deadline: MEOW_RL_SEARCH_DEADLINE_MS after the block was read (MEOW_RL_SEARCH_FIRST_MS on turn 1), checked before
// every trajectory. If any trajectory is missing, the whole search is discarded and the actor's choice is played
// (search_fallback): a partial comparison is never used.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

namespace meow_rl_search {

static_assert(MEOW_RL_SEARCH_CANDIDATES >= 1 && MEOW_RL_SEARCH_SAMPLES >= 1, "at least one candidate and one sample");

struct Result {
    bool used = false;   // false: the actor's choice is played
    int joint = -1;
    long leaves = 0;     // critic evaluations
};

// The critic's value of team `me` in a raw state (2 U - 1 units).
inline double critic_value(const learning::Geometry& geo, const Game& state, int me) {
    auto f = geo.value(learning::prepared(state), me);
    int32_t x[learning::VF], acc[1];
    for (int i = 0; i < learning::VF; i++) x[i] = meow_rl_int::quantize_input(f[i]);
    fusion::forwardRaw(meow_rl_critic::kCritic, x, acc);
    return std::ldexp(double(acc[0]), -meow_rl_critic::kCritic.outExp);
}

inline double terminal_reward(const Game& state, int me) {
    double u = 0;
    for (int t = 0; t < state.teams; t++)
        if (t != me) u += state.score[me] > state.score[t] ? 1.0 : state.score[me] == state.score[t] ? 0.5 : 0.0;
    return 2.0 * u / (state.teams - 1) - 1.0;
}

inline uint64_t seed(int turn, int sample, int root) {
    return uint64_t(turn) * 7919 + uint64_t(sample) * 65537 + uint64_t(root) * 101 + 7;
}

struct Search {
    Result choose(meow_rl::Session& s, const meow_rl::Observation& obs, const int64_t (&sc)[3][5], int actor_choice,
                  std::chrono::steady_clock::time_point started) {
        using clock = std::chrono::steady_clock;
        Result r;
        const Game& g = s.g;
        const Game& o = s.observation;
        Planner& p = s.planner;
        auto deadline = started + std::chrono::milliseconds(g.turn == 1 ? MEOW_RL_SEARCH_FIRST_MS : MEOW_RL_SEARCH_DEADLINE_MS);
        auto z = [&](int j) { return sc[0][j / 25] + sc[1][j / 5 % 5] + sc[2][j % 5]; };
        std::vector<int> roots{actor_choice};
        std::vector<int> rest;
        for (int j = 0; j < meow_rl::kJoint; j++)
            if (obs.joint_mask[j] && j != actor_choice) rest.push_back(j);
        std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) { return z(a) > z(b); });
        for (int j : rest) {
            if (int(roots.size()) >= MEOW_RL_SEARCH_CANDIDATES) break;
            roots.push_back(j);
        }
        std::array<std::vector<Order>, MEOW_RL_SEARCH_SAMPLES> opponent;
        for (int smp = 0; smp < MEOW_RL_SEARCH_SAMPLES; smp++) opponent[smp] = p.enemyOrders(o, s.prob, true);
        double best = -1e300;
        for (int ri = 0; ri < int(roots.size()); ri++) {
            int j = roots[ri];
            Planner::Candidate ca;
            ca.moves = {j / 25, j / 5 % 5, j % 5};
            ca.actions = p.shots(o, g.me, ca.moves, s.prob);
            double sum = 0;
            for (int smp = 0; smp < MEOW_RL_SEARCH_SAMPLES; smp++) {
                if (clock::now() > deadline) return Result{false, -1, r.leaves};   // incomplete: the actor plays
                Game next = p.step(g, ca, opponent[smp], seed(g.turn, smp, ri));
                if (next.turn > g.turns) {
                    sum += terminal_reward(next, g.me);
                    continue;
                }
                // ply 2: the reference's beam bookkeeping (Planner::run), restored afterwards
                auto savedHistory = p.previous;
                auto savedLife = p.newLife;
                int savedTurn = p.preparedTurn;
                for (auto& u : o.units) p.previous[u.id] = p.geo.cell(u);
                Game fo = learning::prepared(next);
                p.preparedTurn = next.turn;
                for (auto& u : fo.units) p.newLife[u.id] = next.units[u.id].x < 0 && u.x >= 0;
                auto pp = p.predict(fo);
                meow_rl::Observation succ;
                bool ok = meow_rl::fill(p, fo, g.me, nullptr, succ);
                Planner::Candidate cont;
                if (ok) {
                    int64_t s2[3][5];
                    for (int k = 0; k < 3; k++)
                        for (int m = 0; m < 5; m++) s2[k][m] = meow_rl_int::score(succ.feats[k][m]);
                    uint64_t h = meow_rl_sample::splitmix64(seed(next.turn, smp, ri) ^ 0x5EA4C4ull);
                    int j2 = meow_rl_sample::draw(s2, succ.joint_mask, meow_rl_model::kActor.outExp, meow_rl_model::kTemperature,
                                                  double(h >> 11) * 0x1.0p-53);
                    cont.moves = {j2 / 25, j2 / 5 % 5, j2 % 5};
                    cont.actions = p.shots(fo, g.me, cont.moves, pp);
                }                                                        // else all stay and wait (the safe fallback)
                auto eo = p.enemyOrders(fo, pp, true);
                p.previous = savedHistory;
                p.newLife = savedLife;
                p.preparedTurn = savedTurn;
                Game leaf = p.step(next, cont, eo, seed(next.turn, smp, ri) + 11);
                sum += leaf.turn > g.turns ? terminal_reward(leaf, g.me) : critic_value(p.geo, leaf, g.me);
                if (leaf.turn <= g.turns) r.leaves++;
            }
            double v = sum / MEOW_RL_SEARCH_SAMPLES;
            if (v > best) best = v, r.joint = j;
        }
        r.used = true;
        return r;
    }
};

}  // namespace meow_rl_search
