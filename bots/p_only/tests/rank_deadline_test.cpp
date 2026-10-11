// Deadline contract of C08's sampled search (search::rank, bots/p_only/tools/derive/search_experiment.py) with
// an injected clock and a controlled ply-2 source: every prediction advances the clock by a fixed cost, so the
// test is deterministic and measures no real time. Compiled once per PONLY_PLY2_MODE; prints one line per case:
//
//   CASE <name> calls <predictions started> late <started at or after the deadline> samples <committed>
//        rollouts <committed> aborted <sweeps> discarded <predictions> truncated <0|1> returned <plans> finite <0|1>
//        counts <committed sample count per returned plan, comma-separated>
//
// bots/p_only/tests/test_search_portfolio.py checks the lines.
//
//   g++ -std=c++20 -O2 -Iinference -Ibots/p_only/include -Ibots/p_only/bots -DFUSION_TIER_XXL -DPONLY_FAMILIES \
//       -DPONLY_PLY2_MODE=1 bots/p_only/tests/rank_deadline_test.cpp
#define PONLY_NO_MAIN 1
#include "c08_ponly_f.hpp"

#include <cmath>
#include <cstdio>

namespace {

using Clock = std::chrono::steady_clock;
Clock::time_point fakeNow{};
Clock::time_point fakeNowFn() { return fakeNow; }

// Every prediction costs `cost` of the injected clock; it records whether it started past `deadline`.
struct SlowPly2 : search::Ply2Source {
    mutable int calls = 0, late = 0;
    Clock::duration cost{};
    Clock::time_point deadline{};
    void predict(const search::Input&, const search::Plan&, const search::Ply1Outcome&, int, search::Ply2& out) const override {
        calls++;
        late += fakeNow >= deadline;
        fakeNow += cost;
        out.move.assign(out.move.size(), -1);
        for (auto& a : out.aim) a = {0, 0, 0};
    }
};

struct Setup {
    int n = 9;
    std::vector<std::array<int, 5>> nb;
    std::vector<std::vector<int>> home;
    std::vector<std::string> map;
    std::vector<double> pick;
    search::Input in;

    explicit Setup(int plans) {
        const int N = n * n;
        map.assign(n, std::string(n, '.'));
        nb.assign(N, {});
        const int dx[5] = {0, 0, 0, -1, 1}, dy[5] = {0, -1, 1, 0, 0};
        for (int c = 0; c < N; c++)
            for (int m = 0; m < 5; m++) {
                const int x = c % n + dx[m], y = c / n + dy[m];
                nb[c][m] = x >= 0 && y >= 0 && x < n && y < n ? y * n + x : -1;
            }
        home.assign(2, std::vector<int>(N, 5));
        pick.assign(N, 0.0);
        in.board = {n, &nb, &home, &map};
        in.me = 0;
        in.turn = 5;
        in.pickCell = &pick;
        for (int k = 0; k < 3; k++) {
            auto& a = in.allies[k];
            a.f = {k, 0, 4 * n + 2 + k, 100, false, false};
            for (int m = 0; m < 5; m++) a.options.push_back({nb[a.f.cell][m], 0.1 * m, 0, 0});
        }
        search::Rival r;
        r.f = {3, 1, 6 * n + 3, 100, false, false};
        for (int m = 0; m < 5; m++) { r.cell.push_back(nb[r.f.cell][m]); r.prob.push_back(0.2); r.aim.push_back({0.1, 0.1, 0.1}); }
        in.rivals.push_back(r);
        for (int p = 0; p < plans; p++) {
            search::Plan pl;
            pl.move = {p % 5, (p / 5) % 5, 0};
            pl.target = {-2, -2, -2};
            pl.value = 1.0 - 0.01 * p;
            pl.source = p;
            in.plans.push_back(pl);
        }
    }
};

void run(const char* name, int plans, bool halving, int fixed, double costMs, double deadlineMs) {
    Setup s(plans);
    SlowPly2 slow;
    slow.cost = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(costMs));
    fakeNow = Clock::time_point{} + std::chrono::hours(1);
    const auto deadline = fakeNow + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(deadlineMs));
    slow.deadline = deadline;
    s.in.ply2 = &slow;
    search::Config cfg;
    cfg.halving = halving;
    cfg.maxPlans = plans;
    cfg.maxSamples = 64;
    search::Stats st;
    search::clockHook = &fakeNowFn;
    const auto ranked = search::rank(s.in, cfg, deadline, &st, fixed);
    search::clockHook = nullptr;
    bool finite = true;
    for (const auto& p : ranked) finite = finite && std::isfinite(p.value);
    std::printf("CASE %s calls %d late %d samples %d rollouts %lld aborted %lld discarded %lld truncated %d returned %zu finite %d\n",
                name, slow.calls, slow.late, st.samples, st.rollouts, st.abortedSweeps, st.discardedPredictions, int(st.truncated),
                ranked.size(), int(finite));
}

}  // namespace

int main() {
    // Uniform sweeps over 3 plans, 1 ms per prediction.
    run("uniform_first", 3, false, 0, 1.0, 0.5);      // the first prediction exhausts the budget: nothing committed
    run("uniform_partial", 3, false, 0, 1.0, 7.5);    // two complete samples, the third interrupted
    run("uniform_fixed", 3, false, 4, 1.0, 1e6);      // fixed work, no deadline: exactly 4 samples
    // Sequential halving over 4 plans (2 rounds), fixed samples so the hard deadline binds.
    run("halving_first", 4, true, 8, 1.0, 0.5);
    run("halving_partial", 4, true, 8, 1.0, 10.5);    // two complete samples of round 0, the third interrupted
    run("halving_fixed", 4, true, 8, 1.0, 1e6);
    return 0;
}
