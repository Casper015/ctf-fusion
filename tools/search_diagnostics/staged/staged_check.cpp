// Offline checks of the staged search (task T0087; tests/test_staged.py beside this file builds and runs them).
//
//   build: c++ -std=c++20 -O2 -I inference -I tools/cpp/meow_rl -DFUSION_TIER_P2GENERAL_VM=1 -DFUSION_MAX_WIDTH=1024
//          -DAGENT_V2_SEARCH_CANDIDATES=8 -DAGENT_V2_SEARCH_SAMPLES=4 -DAGENT_V2_SEARCH_TAU=0.002
//          [-DAGENT_V2_STAGED_SCHEDULE='{4,1},{4,2},{8,2},{8,3},{8,4}']
//          -DACTOR_HEADER='"<actor.hpp>"' -DCRITIC_HEADER='"<critic.hpp>"' tools/search_diagnostics/staged/staged_check.cpp
//   staged_check match <stream> <turn>...      the full schedule decides as Search's whole tree (same means, bit for
//                                              bit); every trajectory's value is the same when the cells are computed
//                                              in another order with another simulator (branch isolation); the search
//                                              leaves the runtime's prediction, history and default order unchanged
//   staged_check interrupt <stream> <turn>...  every work cap 0..all and every forced refusal (trajectory, expansion,
//                                              critic, late) at every trajectory: the decision is the last completed
//                                              checkpoint's, or the actor's joint before the first; discarded work and
//                                              the counters add up
//   staged_check nan <stream> <turn>...        a NaN at any trajectory: every checkpoint containing it keeps the actor
//   staged_check repeat <stream>               two fixed-work runtimes over the whole stream reply identically
//   staged_check schedule                      valid_schedule accepts and refuses crafted schedules
// Fixed-work mode throughout: no clock is read for decisions. Each mode prints "ok ..." or the first mismatch, exit 0/1.
#ifndef FUSION_MAX_WIDTH
#define FUSION_MAX_WIDTH 1024
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include ACTOR_HEADER
#include CRITIC_HEADER
#include "../../../inference/agent_v2/runtime.hpp"
#include "../../../inference/agent_v2/search.hpp"

namespace {

using RT = agent_v2::Runtime<agent_v2::NoSearch>;
using agent_v2::kCandidates;
using agent_v2::kSamples;
using SS = agent_v2::StagedSearch;
constexpr long kAll = 1L << 40;   // a work cap no turn reaches

int fail(const std::string& why) {
    std::printf("FAIL %s\n", why.c_str());
    return 1;
}

std::string slurp(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> blocks(const std::string& text) {
    std::vector<std::string> out;
    std::string cur, line;
    std::istringstream in(text);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        cur += line + "\n";
        if (line.rfind("END", 0) == 0) out.push_back(cur), cur.clear();
    }
    return out;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// One turn's whole tree, cell by cell, in the given order (root-major, Search's, or reversed scenario-major).
struct Cells {
    agent_v2::Tree tree;
    double val[kCandidates][kSamples] = {};
    bool expands[kCandidates][kSamples] = {}, critic[kCandidates][kSamples] = {};
};

void compute(RT& rt, const int64_t (&sc)[3][5], int joint, Cells& out, bool reversed) {
    agent_v2::setup(rt, sc, joint, out.tree);
    agent_v2::Simulator sim;
    agent_v2::Leaf leaf;
    const int n = int(out.tree.roots.size());
    auto one = [&](int ri, int smp) {
        bool expanded = false;
        agent_v2::trajectory(rt, out.tree, ri, smp, sim, leaf, [&] { return expanded = true; });
        out.expands[ri][smp] = expanded;
        out.critic[ri][smp] = !leaf.terminal;
        out.val[ri][smp] = leaf.terminal ? leaf.reward : meow_rl_critic::value(leaf.feats.data());
    };
    if (!reversed) {
        for (int ri = 0; ri < n; ri++)
            for (int smp = 0; smp < kSamples; smp++) one(ri, smp);
    } else {
        for (int smp = kSamples - 1; smp >= 0; smp--)
            for (int ri = n - 1; ri >= 0; ri--) one(ri, smp);
    }
}

// The schedule as the search walks it for n roots: per effective checkpoint, its (k, s), the trajectories needed
// to complete it (cumulative), and the cell order.
struct Walk {
    struct Step { int k, s; long cells; };
    std::vector<Step> steps;
    std::vector<std::pair<int, int>> order;   // (root, scenario) in the order the search runs them
};

Walk walk(int n) {
    Walk w;
    bool have[kCandidates][kSamples] = {};
    for (int c = 0; c < agent_v2::kStagedSchedule.n; c++) {
        const int k = std::min(agent_v2::kStagedSchedule.at[c].k, n), s = agent_v2::kStagedSchedule.at[c].s;
        if (!w.steps.empty() && w.steps.back().k == k && w.steps.back().s == s) continue;
        for (int smp = 0; smp < s; smp++)
            for (int ri = 0; ri < k; ri++)
                if (!have[ri][smp]) have[ri][smp] = true, w.order.push_back({ri, smp});
        w.steps.push_back({k, s, long(w.order.size())});
    }
    return w;
}

// The expected decision after `done` usable trajectories: the last step completed by then (-1: none).
int expected_step(const Walk& w, long done) {
    int at = -1;
    for (int i = 0; i < int(w.steps.size()); i++)
        if (w.steps[i].cells <= done) at = i;
    return at;
}

int pick_of(const Cells& c, const Walk::Step& st, std::vector<double>& mean) {
    mean.assign(st.k, 0);
    for (int ri = 0; ri < st.k; ri++) mean[ri] = agent_v2::checkpoint_mean(c.val, ri, st.s);
    return agent_v2::gated_choice(mean.data(), st.k, agent_v2::kTau);
}

// Runs one staged search and compares it with the expectation for `usable` completed trajectories.
std::string expect(RT& rt, const int64_t (&sc)[3][5], int joint, const Cells& cells, const Walk& w, SS& ss, long usable,
                   int want_stop, const std::string& what) {
    const long before_used = rt.tel.search_used, before_fallback = rt.tel.search_fallback;
    const int got = ss.choose(rt, sc, joint, std::chrono::steady_clock::now());
    const int at = expected_step(w, usable);
    if (ss.last.stop != want_stop) return what + ": stop " + std::to_string(ss.last.stop) + " want " + std::to_string(want_stop);
    if (at < 0) {
        if (got != joint || ss.last.checkpoints != 0 || rt.tel.search_fallback != before_fallback + 1 || rt.tel.search_used != before_used)
            return what + ": no checkpoint, the actor's joint and one fallback expected";
        return "";
    }
    std::vector<double> mean;
    const int pick = pick_of(cells, w.steps[at], mean);
    if (ss.last.checkpoints != at + 1 || ss.last.k != w.steps[at].k || ss.last.s != w.steps[at].s)
        return what + ": checkpoint " + std::to_string(ss.last.checkpoints) + " (" + std::to_string(ss.last.k) + "," +
               std::to_string(ss.last.s) + ") want " + std::to_string(at + 1);
    for (int ri = 0; ri < w.steps[at].k; ri++)
        if (!same_bits(ss.last.mean[ri], mean[ri]) && !(std::isnan(ss.last.mean[ri]) && std::isnan(mean[ri])))
            return what + ": mean of root " + std::to_string(ri);
    if (ss.last.pick != pick || got != cells.tree.roots[pick]) return what + ": pick";
    if (rt.tel.search_used != before_used + 1 || rt.tel.search_fallback != before_fallback) return what + ": used/fallback";
    return "";
}

template <class F>
int each_turn(const char* path, int argc, char** argv, F&& f) {
    std::set<int> wanted;
    for (int i = 0; i < argc; i++) wanted.insert(std::atoi(argv[i]));
    RT rt;
    int seen = 0;
    for (auto& blk : blocks(slurp(path))) {
        if (!rt.observe_block(blk).empty()) {
            rt.refuse();
            continue;
        }
        int64_t sc[3][5];
        agent_v2::Actor::scores(rt.obs, sc);
        const int joint = agent_v2::Actor::sample(sc, rt.obs, agent_v2::uniform(rt.seed, rt.s.turn));
        if (wanted.count(rt.s.turn)) {
            rt.tactics = rt.resolve(joint);
            if (std::string why = f(rt, sc, joint); !why.empty()) return fail("turn " + std::to_string(rt.s.turn) + " " + why);
            seen++;
        }
        rt.answer(joint);
    }
    if (seen != int(wanted.size())) return fail("some requested turns were absent or refused");
    return 0;
}

int check_match(const char* path, int argc, char** argv) {
    long cells_total = 0, terminal = 0;
    int turns = 0;
    const int r = each_turn(path, argc, argv, [&](RT& rt, const int64_t (&sc)[3][5], int joint) -> std::string {
        Cells a, b;
        compute(rt, sc, joint, a, false);
        compute(rt, sc, joint, b, true);
        const int n = int(a.tree.roots.size());
        for (int ri = 0; ri < n; ri++)
            for (int smp = 0; smp < kSamples; smp++)
                if (!same_bits(a.val[ri][smp], b.val[ri][smp])) return "cell order changed root " + std::to_string(ri) + " scenario " + std::to_string(smp);
        // Search's whole tree: sum in scenario order per root, / kSamples, the gate over all roots.
        std::array<double, kCandidates> mean{};
        for (int ri = 0; ri < n; ri++) {
            double sum = 0;
            for (int smp = 0; smp < kSamples; smp++) sum += a.val[ri][smp];
            mean[ri] = sum / kSamples;
        }
        const int search_pick = agent_v2::gated_choice(mean.data(), n, agent_v2::kTau);
        const auto key = rt.pred.root.key;
        const std::string reply_before = agent_v2::reply_text(rt.s.turn, rt.tactics.orders);
        const int hist_turn = rt.hist.parent() ? rt.hist.parent()->turn : -1;
        SS ss;
        ss.work_cap = kAll;
        const int got = ss.choose(rt, sc, joint, std::chrono::steady_clock::now());
        if (ss.last.stop != SS::kComplete) return "the full schedule did not complete";
        if (got != a.tree.roots[search_pick] || ss.last.pick != search_pick) return "decision differs from Search's";
        for (int ri = 0; ri < n; ri++)
            if (!same_bits(ss.last.mean[ri], mean[ri])) return "final mean differs from Search's, root " + std::to_string(ri);
        if (!(rt.pred.root.key == key) || agent_v2::reply_text(rt.s.turn, rt.tactics.orders) != reply_before ||
            (rt.hist.parent() ? rt.hist.parent()->turn : -1) != hist_turn)
            return "the search wrote back into the runtime";
        if (ss.discarded != 0 || ss.trajectories != long(walk(n).order.size())) return "complete run counters";
        for (int ri = 0; ri < n; ri++)
            for (int smp = 0; smp < kSamples; smp++) terminal += !a.critic[ri][smp];
        cells_total += ss.trajectories, turns++;
        return "";
    });
    if (!r) std::printf("ok match turns %d trajectories %ld terminal %ld schedule %d\n", turns, cells_total, terminal,
                        agent_v2::kStagedSchedule.n);
    return r;
}

int check_interrupt(const char* path, int argc, char** argv) {
    long runs = 0;
    const int r = each_turn(path, argc, argv, [&](RT& rt, const int64_t (&sc)[3][5], int joint) -> std::string {
        Cells cells;
        compute(rt, sc, joint, cells, false);
        const Walk w = walk(int(cells.tree.roots.size()));
        const long all = long(w.order.size());
        SS ss;                                                      // one search object across runs: counters accumulate
        long want_discarded = 0, want_decided = 0, want_fallback = 0;
        auto tally = [&](long usable, long finished) {             // finished: trajectories completed, used or not
            const int at = expected_step(w, usable);
            if (at < 0) want_fallback++;
            else want_decided++, want_discarded += finished - w.steps[at].cells;
        };
        // Setup refused: the actor's joint.
        ss.work_cap = kAll, ss.force_point = SS::kAdmitSetup, ss.force_at = 0;
        if (auto e = expect(rt, sc, joint, cells, w, ss, -1, SS::kAdmitSetup, "setup"); !e.empty()) return e;
        tally(-1, 0), runs++;
        for (long cap = 0; cap <= all; cap++) {
            ss.work_cap = cap, ss.force_point = -1, ss.force_at = -1;
            const int want_stop = cap == all ? SS::kComplete : SS::kWorkCap;
            if (auto e = expect(rt, sc, joint, cells, w, ss, cap, want_stop, "cap " + std::to_string(cap)); !e.empty()) return e;
            tally(cap, cap), runs++;
        }
        for (long i = 0; i < all; i++) {
            const auto [ri, smp] = w.order[i];
            for (int point : {int(SS::kAdmitTrajectory), int(SS::kAdmitExpand), int(SS::kAdmitCritic), int(SS::kLate)}) {
                const bool fires = point == SS::kAdmitTrajectory || point == SS::kLate ||
                                   (point == SS::kAdmitExpand && cells.expands[ri][smp]) ||
                                   (point == SS::kAdmitCritic && cells.critic[ri][smp]);
                ss.work_cap = kAll, ss.force_point = point, ss.force_at = i;
                const long usable = fires ? i : all;
                const int want_stop = fires ? point : SS::kComplete;
                if (auto e = expect(rt, sc, joint, cells, w, ss, usable, want_stop,
                                    "force " + std::to_string(point) + " at " + std::to_string(i)); !e.empty()) return e;
                tally(usable, fires && point == SS::kLate ? i + 1 : usable), runs++;
            }
        }
        if (ss.decided != want_decided || ss.discarded != want_discarded ||
            ss.decided != ss.overrides + ss.baseline_kept || ss.gate_kept > ss.baseline_kept)
            return "counters: decided " + std::to_string(ss.decided) + "/" + std::to_string(want_decided) + " discarded " +
                   std::to_string(ss.discarded) + "/" + std::to_string(want_discarded);
        const long stops = ss.refuse_setup + ss.refuse_trajectory + ss.refuse_expand + ss.refuse_critic + ss.refuse_late +
                           ss.stopped_work_cap + ss.complete;
        if (stops != ss.decided + want_fallback) return "stop causes do not add up";
        return "";
    });
    if (!r) std::printf("ok interrupt runs %ld\n", runs);
    return r;
}

int check_nan(const char* path, int argc, char** argv) {
    long runs = 0;
    const int r = each_turn(path, argc, argv, [&](RT& rt, const int64_t (&sc)[3][5], int joint) -> std::string {
        Cells cells;
        compute(rt, sc, joint, cells, false);
        const Walk w = walk(int(cells.tree.roots.size()));
        for (long i = 0; i < long(w.order.size()); i++) {
            SS ss;
            ss.work_cap = kAll, ss.nan_at = i;
            const int got = ss.choose(rt, sc, joint, std::chrono::steady_clock::now());
            if (ss.last.stop != SS::kComplete) return "a NaN stopped the work";
            // The final checkpoint contains every cell: its means are not all finite, so the actor's joint.
            if (got != joint || ss.last.pick != 0) return "NaN at " + std::to_string(i) + " overrode the actor";
            // A cap at each step boundary: checkpoints without the NaN cell decide as without it.
            for (const auto& st : w.steps) {
                SS cut;
                cut.work_cap = st.cells, cut.nan_at = i;
                cut.choose(rt, sc, joint, std::chrono::steady_clock::now());
                std::vector<double> mean;
                const int clean = pick_of(cells, st, mean);
                const bool contains = i < st.cells;
                if (contains ? cut.last.pick != 0 : cut.last.pick != clean) return "NaN at " + std::to_string(i) + " step pick";
                runs++;
            }
        }
        return "";
    });
    if (!r) std::printf("ok nan runs %ld\n", runs);
    return r;
}

int check_repeat(const char* path) {
    for (long cap : {1L, 7L, 13L, kAll}) {
        agent_v2::Runtime<SS> a, b;
        a.search.work_cap = b.search.work_cap = cap;
        std::string why;
        int turns = 0;
        for (auto& blk : blocks(slurp(path))) {
            const auto t = std::chrono::steady_clock::now();
            const std::string ra = a.decide(blk, t, why), rb = b.decide(blk, t, why);
            if (ra != rb) return fail("cap " + std::to_string(cap) + " turn " + std::to_string(turns + 1) + " replies differ");
            turns++;
        }
        const auto& x = a.search;
        const auto& y = b.search;
        if (x.trajectories != y.trajectories || x.decided != y.decided || x.discarded != y.discarded || x.overrides != y.overrides)
            return fail("cap " + std::to_string(cap) + " counters differ");
        if (a.tel.search_used + a.tel.search_fallback != a.tel.turns - a.tel.refused) return fail("used + fallback != answered turns");
        std::printf("ok repeat cap %ld turns %d decided %ld overrides %ld\n", cap == kAll ? -1 : cap, turns, x.decided, x.overrides);
    }
    return 0;
}

int check_schedule() {
    using agent_v2::Schedule;
    auto make = [](std::initializer_list<agent_v2::Checkpoint> cs) {
        Schedule s;
        for (auto c : cs) s.at[s.n++] = c;
        return s;
    };
    const int K = kCandidates, S = kSamples;
    struct Case { Schedule s; bool ok; const char* what; };
    const Case cases[] = {
        {make({{K, S}}), true, "the whole tree alone"},
        {make({{2, 1}, {K, S}}), true, "a small start"},
        {make({{K, 1}, {K, S}}), true, "rounds"},
        {make({{2, 1}, {2, 1}, {K, S}}), false, "a repeated checkpoint"},
        {make({{K, 2}, {K, 1}, {K, S}}), false, "fewer scenarios later"},
        {make({{K, 1}, {2, 2}, {K, S}}), false, "fewer roots later"},
        {make({{2, 1}, {K, S - 1}}), false, "not ending at the whole tree"},
        {make({{1, 1}, {K, S}}), K < 2, "the actor alone"},
        {make({{K + 1, 1}, {K, S}}), false, "more roots than candidates"},
        {make({{2, 0}, {K, S}}), false, "no scenario"},
        {make({}), false, "empty"},
    };
    for (const auto& c : cases)
        if (agent_v2::valid_schedule(c.s) != c.ok) return fail(std::string("schedule: ") + c.what);
    if (!agent_v2::valid_schedule(agent_v2::kStagedSchedule)) return fail("the compiled schedule");
    std::printf("ok schedule cases %d compiled", int(std::size(cases)));
    for (int i = 0; i < agent_v2::kStagedSchedule.n; i++)
        std::printf(" %d:%d", agent_v2::kStagedSchedule.at[i].k, agent_v2::kStagedSchedule.at[i].s);
    std::printf("\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return fail("usage: staged_check match|interrupt|nan|repeat|schedule ...");
    const std::string mode = argv[1];
    if (mode == "schedule") return check_schedule();
    if (argc < 3) return fail("a stream is required");
    if (mode == "match") return check_match(argv[2], argc - 3, argv + 3);
    if (mode == "interrupt") return check_interrupt(argv[2], argc - 3, argv + 3);
    if (mode == "nan") return check_nan(argv[2], argc - 3, argv + 3);
    if (mode == "repeat") return check_repeat(argv[2]);
    return fail("unknown mode " + mode);
}
