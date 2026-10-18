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
//   * choice: the override gate (T0076, docs/AGENT_V2_SEARCH_GATE.md): the best root replaces the actor's joint only
//     when its scenario mean beats the actor's by more than AGENT_V2_SEARCH_TAU over the same completed scenarios;
//     ties, a margin <= tau and a non-finite mean keep the actor's order. With finite means, tau 0 is the ungated
//     choice of the search before T0076 (the highest mean, ties keep the earlier root). Common streams per (game seed, turn, scenario).
// Admission (staged, v2): setup, each trajectory, the ply-2 expansion and the critic are admitted only if their
// estimate (max(prior, worst seen this game) x safety, plus a reply reserve) fits before the deadline; a refusal keeps
// the default order, which the runtime resolved before the search started: incomplete work never overrides. Include the critic header
// (namespace meow_rl_critic) before this file.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
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
#ifndef AGENT_V2_SEARCH_TAU
#define AGENT_V2_SEARCH_TAU 0.0
#endif
#ifndef AGENT_V2_SEARCH_TEST_WORK_CAP
#define AGENT_V2_SEARCH_TEST_WORK_CAP 0   // tests only: > 0 refuses the trajectory after this many in one turn
#endif

namespace agent_v2 {

constexpr int kCandidates = AGENT_V2_SEARCH_CANDIDATES, kSamples = AGENT_V2_SEARCH_SAMPLES;
constexpr int kBranchScenarios = AGENT_V2_BRANCH_SCENARIOS;
constexpr double kTau = AGENT_V2_SEARCH_TAU;
static_assert(kCandidates >= 1 && kSamples >= 1, "the tree needs a root and a scenario");
static_assert(kTau >= 0.0, "a negative tau would override the actor on a loss");

// The override gate over one completed tree: mean[0] is the actor's joint, mean[i] root i's mean over the same scenarios.
// Returns the root to play: the first root with the highest mean if it beats the actor's mean by more than tau, else 0.
// A non-finite mean anywhere makes the tree invalid: the actor's joint.
inline int gated_choice(const double* mean, int n, double tau) {
    int best = 0;
    for (int i = 0; i < n; i++)
        if (!std::isfinite(mean[i])) return 0;
    for (int i = 1; i < n; i++)
        if (mean[i] > mean[best]) best = i;
    return best != 0 && mean[best] - mean[0] > tau ? best : 0;
}

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

// salt 0 is the deployed tree; a nonzero salt draws an independent repeat of its scenarios (offline diagnostics only).
template <class R>
void setup(R& rt, const int64_t (&sc)[3][5], int actor_choice, Tree& tree, uint64_t salt = 0) {
    tree.T = turn_seed(rt.seed, rt.s.turn);
    if (salt) tree.T = splitmix64(tree.T ^ salt);
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
    rt.pred.scenarios(branch, next, &s, kBranchScenarios, uint32_t(stream(tree.T, kBranchResolver, smp)), table);
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
    long overrides = 0, gate_kept = 0;   // completed trees: the actor replaced; a better root kept out by the gate (margin <= tau)

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
        std::array<double, kCandidates> mean{};
        Leaf leaf;
        int done = 0;
        for (int ri = 0; ri < int(tree.roots.size()); ri++) {
            double sum = 0;
            for (int smp = 0; smp < kSamples; smp++, done++) {
                if (!admit(estimate(AGENT_V2_SEARCH_TRAJ_US, max_traj_us)) ||
                    (AGENT_V2_SEARCH_TEST_WORK_CAP > 0 && done >= AGENT_V2_SEARCH_TEST_WORK_CAP))
                    return fallback(refuse_trajectory);
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
            mean[ri] = sum / kSamples;
        }
        const int n = int(tree.roots.size());
        const int pick = gated_choice(mean.data(), n, kTau);
        if (pick) overrides++;
        else if (gated_choice(mean.data(), n, 0.0)) gate_kept++;
        rt.tel.search_used++;
        max_search_us = std::max(max_search_us, us(clock::now() - t_begin));
        return tree.roots[pick];
    }
};
#endif

// ---- Staged search (task T0087, bots/agent_v2/staged_search/README.md): opt-in, StagedSearch below. ---------------
// The same tree as Search (roots, scenarios, trajectories, critic, gate and tau, admission estimates and reply reserve,
// all unchanged), worked through a frozen schedule of checkpoints. Checkpoint i compares the first k_i roots (the
// actor's joint first) over scenarios 0..s_i-1, so every compared root has the same completed scenarios. Work runs
// scenario-major: each round gives every root of the checkpoint its next scenario. A completed checkpoint's decision
// (gated_choice over its means) is fixed and never revised by later partial work. When work stops (a refused
// admission, the deadline, or the fixed-work cap), the bot plays the last completed checkpoint's decision, or the
// default order when none completed; trajectories finished after that checkpoint are discarded and counted. The last
// checkpoint is the whole kCandidates x kSamples tree, summed in Search's order, so a run that completes the schedule
// decides exactly as Search does. Nothing is cached: each trajectory is a function of (tree, root, scenario) alone,
// which tools/search_diagnostics/staged checks.
//
// AGENT_V2_STAGED_SCHEDULE: the checkpoints as "{k, s}, ..." (empty: full-width rounds {kCandidates, 1} ...
// {kCandidates, kSamples}); k and s never decrease, each step raises one, the last is {kCandidates, kSamples}.
// AGENT_V2_STAGED_FIXED_WORK >= 0: the deterministic mode for offline checks: no clock, work stops after that many
// trajectories (StagedSearch::work_cap; -1, the default, is the timed mode). It is never a timing or strength arm.
#ifndef AGENT_V2_STAGED_SCHEDULE
#define AGENT_V2_STAGED_SCHEDULE
#endif
#ifndef AGENT_V2_STAGED_FIXED_WORK
#define AGENT_V2_STAGED_FIXED_WORK -1
#endif

struct Checkpoint {
    int k = 0, s = 0;
};
struct Schedule {
    std::array<Checkpoint, 64> at{};
    int n = 0;
};

constexpr Schedule staged_schedule() {
    constexpr Checkpoint user[] = {{0, 0}, AGENT_V2_STAGED_SCHEDULE};   // a sentinel keeps the list non-empty
    constexpr int given = int(sizeof user / sizeof user[0]) - 1;
    Schedule out;
    if (given == 0)
        for (int s = 1; s <= kSamples; s++) out.at[out.n++] = {kCandidates, s};
    else
        for (int i = 1; i <= given && out.n < int(out.at.size()); i++) out.at[out.n++] = user[i];
    return out;
}

constexpr bool valid_schedule(const Schedule& sch) {
    if (sch.n < 1 || sch.n > int(sch.at.size())) return false;
    const int k_min = kCandidates < 2 ? 1 : 2;
    for (int i = 0; i < sch.n; i++) {
        const Checkpoint c = sch.at[i];
        if (c.k < k_min || c.k > kCandidates || c.s < 1 || c.s > kSamples) return false;
        if (i && (c.k < sch.at[i - 1].k || c.s < sch.at[i - 1].s || (c.k == sch.at[i - 1].k && c.s == sch.at[i - 1].s)))
            return false;
    }
    return sch.at[sch.n - 1].k == kCandidates && sch.at[sch.n - 1].s == kSamples;
}

inline constexpr Schedule kStagedSchedule = staged_schedule();
static_assert(valid_schedule(kStagedSchedule), "AGENT_V2_STAGED_SCHEDULE: monotone checkpoints ending at the full tree");

// The mean of root ri over scenarios 0..s-1, summed in scenario order (Search's order, so the full tree matches it).
inline double checkpoint_mean(const double (&val)[kCandidates][kSamples], int ri, int s) {
    double sum = 0;
    for (int smp = 0; smp < s; smp++) sum += val[ri][smp];
    return sum / s;
}

#ifndef AGENT_V2_TREE_ONLY
struct StagedSearch {
    enum Point { kAdmitSetup = 0, kAdmitTrajectory, kAdmitExpand, kAdmitCritic, kLate, kWorkCap, kComplete };
    Simulator sim;
    long work_cap = AGENT_V2_STAGED_FIXED_WORK;   // >= 0: fixed-work mode (no clock); -1: timed
    // Test hooks, off by default: refuse admission `force_point` at trajectory index `force_at` (kLate: discard that
    // trajectory as if it finished after the deadline); give trajectory `nan_at` a NaN value.
    int force_point = -1;
    long force_at = -1, nan_at = -1;
    // Cumulative over the game. Per turn with a search: decided (a checkpoint's decision) or a fallback (none).
    long max_setup_us = 0, max_traj_us = 0, max_expand_us = 0, max_critic_us = 0, max_search_us = 0;
    long trajectories = 0, discarded = 0, checkpoints = 0, complete = 0, decided = 0, eff_candidates = 0, eff_scenarios = 0;
    long overrides = 0, baseline_kept = 0, gate_kept = 0;
    long refuse_setup = 0, refuse_trajectory = 0, refuse_expand = 0, refuse_critic = 0, refuse_late = 0, stopped_work_cap = 0;
    // The last turn: completed checkpoints, the decision checkpoint's (k, s), means and pick, trajectories, why it stopped.
    struct Record {
        int checkpoints = 0, k = 0, s = 0, pick = 0, stop = -1;
        long done = 0;
        std::vector<int> roots;
        std::vector<double> mean;
    } last;

    static long us(std::chrono::steady_clock::duration d) { return long(std::chrono::duration_cast<std::chrono::microseconds>(d).count()); }
    static long estimate(long prior, long seen) { return std::max(prior, seen) * AGENT_V2_SEARCH_SAFETY_PCT / 100; }

    template <class R>
    int choose(R& rt, const int64_t (&sc)[3][5], int joint, std::chrono::steady_clock::time_point started) {
        using clock = std::chrono::steady_clock;
        const auto t_begin = clock::now();
        const bool timed = work_cap < 0;
        const auto deadline = started + std::chrono::milliseconds(rt.s.turn == 1 ? AGENT_V2_SEARCH_FIRST_MS : AGENT_V2_SEARCH_DEADLINE_MS);
        const long reserve = estimate(AGENT_V2_SEARCH_REPLY_US, 0);
        long cell = 0, at_checkpoint = 0;
        last = Record{};
        Tree tree;
        auto admit = [&](long cost, Point p) {
            if (force_point == p && force_at == cell) return false;
            return !timed || clock::now() + std::chrono::microseconds(cost + reserve) <= deadline;
        };
        auto stop = [&](long& counter, Point why) {
            counter++;
            last.stop = why, last.done = cell;
            max_search_us = std::max(max_search_us, us(clock::now() - t_begin));
            if (!last.checkpoints) {
                rt.tel.search_fallback++;
                return joint;
            }
            discarded += cell - at_checkpoint;
            rt.tel.search_used++;
            decided++, eff_candidates += last.k, eff_scenarios += last.s;
            if (last.pick) overrides++;
            else {
                baseline_kept++;
                if (gated_choice(last.mean.data(), last.k, 0.0)) gate_kept++;
            }
            return tree.roots[last.pick];
        };
        if (!admit(estimate(AGENT_V2_SEARCH_SETUP_US, max_setup_us) + estimate(AGENT_V2_SEARCH_TRAJ_US, max_traj_us), kAdmitSetup))
            return stop(refuse_setup, kAdmitSetup);
        setup(rt, sc, joint, tree);
        last.roots = tree.roots;
        max_setup_us = std::max(max_setup_us, us(clock::now() - t_begin));
        const int n = int(tree.roots.size());
        double val[kCandidates][kSamples] = {};
        bool have[kCandidates][kSamples] = {};
        Leaf leaf;
        for (int c = 0; c < kStagedSchedule.n; c++) {
            const int k = std::min(kStagedSchedule.at[c].k, n), s = kStagedSchedule.at[c].s;
            if (last.checkpoints && k == last.k && s == last.s) continue;   // fewer feasible roots: the same comparison
            for (int smp = 0; smp < s; smp++)
                for (int ri = 0; ri < k; ri++) {
                    if (have[ri][smp]) continue;
                    if (!timed && cell >= work_cap) return stop(stopped_work_cap, kWorkCap);
                    if (!admit(estimate(AGENT_V2_SEARCH_TRAJ_US, max_traj_us), kAdmitTrajectory)) return stop(refuse_trajectory, kAdmitTrajectory);
                    const auto t0 = clock::now();
                    clock::time_point t_expand{};
                    auto expand = [&] {
                        t_expand = clock::now();
                        return admit(estimate(AGENT_V2_SEARCH_EXPAND_US, max_expand_us), kAdmitExpand);
                    };
                    auto spent = [&] { max_traj_us = std::max(max_traj_us, us(clock::now() - t0)); };
                    if (!trajectory(rt, tree, ri, smp, sim, leaf, expand)) {
                        spent();
                        return stop(refuse_expand, kAdmitExpand);
                    }
                    if (leaf.plies == 2) max_expand_us = std::max(max_expand_us, us(clock::now() - t_expand));
                    double v = leaf.reward;
                    if (!leaf.terminal) {
                        if (!admit(estimate(AGENT_V2_SEARCH_CRITIC_US, max_critic_us), kAdmitCritic)) {
                            spent();
                            return stop(refuse_critic, kAdmitCritic);
                        }
                        const auto t1 = clock::now();
                        v = meow_rl_critic::value(leaf.feats.data());
                        max_critic_us = std::max(max_critic_us, us(clock::now() - t1));
                        rt.tel.leaves++;
                    }
                    spent();
                    if (cell == nan_at) v = std::numeric_limits<double>::quiet_NaN();
                    trajectories++;
                    const bool late = (timed && clock::now() > deadline) || (force_point == kLate && force_at == cell);
                    cell++;
                    if (late) return stop(refuse_late, kLate);           // its work is discarded with the rest
                    val[ri][smp] = v, have[ri][smp] = true;
                }
            std::array<double, kCandidates> mean{};
            for (int ri = 0; ri < k; ri++) mean[ri] = checkpoint_mean(val, ri, s);
            last.checkpoints++, last.k = k, last.s = s;
            last.mean.assign(mean.begin(), mean.begin() + k);
            last.pick = gated_choice(mean.data(), k, kTau);
            at_checkpoint = cell;
            checkpoints++;
        }
        return stop(complete, kComplete);
    }
};
#endif

}  // namespace agent_v2
