// P-only contract test: replays a recorded seat input stream (stdin) through a
// P-only carrier, exactly as its main() would, and checks after every real
// decision:
//   history   the root view's previous observation is the previous real turn:
//             every unit's previous cell equals its real cell one turn earlier,
//             so no simulated branch reached the real history (the defect of
//             the C08 Fusion carrier); the root view's state is this real turn
//             and the view the planner reads at the end of the turn is the root;
//   dists     per active opponent unit: move probabilities sum to 1 over legal
//             moves and are 0 elsewhere; act probabilities sum to 1 and are 0
//             for acts without effect; target probabilities sum to 1 whenever
//             an attack has an effect; P(attack and target v) <= P(attack);
//   orders    the mode orders are legal (move, act, target among candidates)
//             and two draws with the same keys are identical;
//   fallback  no prediction fell back to uniform (every team count has a model).
// Prints one summary line; exit status 1 on any failure.
//
//   g++ -std=c++20 -O2 -Iinference -Ip_only/include -Ip_only/bots -DCARRIER_C08 p_only/tests/contract_test.cpp
//   gzip -dc p_only/tests/fixtures/stream_n04.txt.gz | ./a.out
#define PONLY_NO_MAIN 1
#if defined(CARRIER_C08)
#include "c08_ponly.hpp"
#elif defined(CARRIER_V12)
#include "v12_ponly.hpp"
#else
#error "define CARRIER_C08 or CARRIER_V12"
#endif

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;
long long checks = 0;

void check(bool ok, int turn, const char* what) {
    checks++;
    if (!ok && ++failures <= 20) std::fprintf(stderr, "turn %d: %s\n", turn, what);
}

void distributions(ponly::View v, int me, int turn) {  // a copy: the act/target caches fill lazily
    const auto& s = v.state();
    for (const auto& u : s.units) {
        if (u.team == me || !v.has(u.team) || v.origin(u.id) < 0) continue;
        double total = 0;
        for (int m = 0; m < ponly::kMoves; m++) {
            const double p = v.move(u.id, m);
            if (!v.legal(u.id, m)) { check(p == 0, turn, "illegal move with probability"); continue; }
            total += p;
            const auto& a = v.act(u.id, m);
            double at = 0;
            for (int k = 0; k < ponly::kActs; k++) {
                if (!v.actLegal(u.id, m, k)) check(a[k] == 0, turn, "act without effect has probability");
                at += a[k];
            }
            check(std::fabs(at - 1) < 1e-9, turn, "act probabilities do not sum to 1");
            if (v.actLegal(u.id, m, fusion::kAttack)) {
                const auto& t = v.target(u.id, m);
                double tt = 0;
                for (int c = 0; c < ponly::kCand; c++) {
                    if (v.candidate(u.id, c) < 0) check(t[c] == 0, turn, "empty target slot has probability");
                    tt += t[c];
                }
                check(std::fabs(tt - 1) < 1e-9, turn, "target probabilities do not sum to 1");
            }
            for (int k = 0; k < 3; k++)
                check(v.aim(u.id, m, 3 * me + k) <= v.fire(u.id, m) + 1e-12, turn, "aim exceeds attack probability");
        }
        check(std::fabs(total - 1) < 1e-9, turn, "move probabilities do not sum to 1");
    }
}

void orders(ponly::View v, int me, int turn) {
    ponly::View w = v;
    for (int t = 0; t < (int)v.state().score.size(); t++) {
        if (t == me || !v.has(t)) continue;
        const auto mode = v.teamOrders(t, -1, 0);
        for (int k = 0; k < 3; k++) {
            const int id = 3 * t + k;
            if (v.origin(id) < 0) continue;
            const auto& o = mode[k];
            check(v.legal(id, o.move), turn, "mode move is illegal");
            check(v.actLegal(id, o.move, o.act), turn, "mode act has no effect");
            if (o.act == fusion::kAttack) check(v.canAim(id, o.target), turn, "mode target is not a candidate");
        }
        const auto a = v.teamOrders(t, 5, 3), b = w.teamOrders(t, 5, 3);
        for (int k = 0; k < 3; k++)
            check(a[k].move == b[k].move && a[k].act == b[k].act && a[k].target == b[k].target, turn, "draw is not deterministic");
    }
}

void history(const fusion_hook::PSession& fs, const Game& g, const std::vector<Unit>& previous) {
    const auto& v = fs.view;
    check(v.state().turn == g.turn, g.turn, "root view is not this turn");
    bool same = v.state().units.size() == g.units.size();
    for (size_t i = 0; same && i < g.units.size(); i++)
        same = v.state().units[i].x == g.units[i].x && v.state().units[i].y == g.units[i].y;
    check(same, g.turn, "root view state differs from the real turn");
    const auto& prev = v.turnView().prevCell;
    for (size_t i = 0; i < g.units.size(); i++) {
        int want = -1;
        if (!previous.empty() && previous[i].x >= 0) want = previous[i].y * g.size + previous[i].x;
        check(prev[i] == want, g.turn, "history is not the previous real turn");
    }
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    Game g;
#if defined(CARRIER_C08)
    multi::Planner many;
    duel::Player two;
#else
    Planner p;
#endif
    std::vector<Unit> previous;
    int turns = 0;
    long long branches = 0, fallback = 0;
    while (read_turn(g)) {
#if defined(CARRIER_C08)
        const std::string out = g.teams == 2 ? two.decide(g) : many.decide(g);
        const fusion_hook::PSession& fs = g.teams == 2 ? two.session() : many.fSession;
        check(fs.cur == &fs.view, g.turn, "a branch view is still current after the turn");
#else
        const std::string out = p.decide(g);
        const fusion_hook::PSession& fs = p.fSession;
#endif
        check(!out.empty() && std::stoi(out) == g.turn, g.turn, "reply does not start with the turn");
        history(fs, g, previous);
        distributions(fs.view, g.me, g.turn);
        orders(fs.view, g.me, g.turn);
        branches = fs.counters.branches;
        fallback = fs.counters.fallback;
        previous = g.units;
        turns++;
    }
    check(fallback == 0, g.turn, "a prediction fell back to uniform");
    std::printf("contract: %d turns, %d teams, %lld checks, %d failures, %lld branch views, %lld fallbacks\n",
                turns, g.teams, checks, failures, branches, fallback);
    return failures ? 1 : 0;
}
