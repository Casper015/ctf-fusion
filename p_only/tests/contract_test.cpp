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
//   fallback  no prediction fell back to uniform (every team count has a model);
//   coherent  every state handed to the network keeps the rules' invariants and
//             agrees with the carrier's search (ponly Counters::inconsistent == 0);
//   engine    with the recorded orders (argument 1, orders_nXX.txt.gz unzipped),
//             the carrier's own turn kernel (C08 stepExact, v12 simStep) applied
//             to each real turn reproduces the engine's next state field by field:
//             positions, HP, carried flags, respawn times, flags, scores and attack
//             events. Turns with a pending random flag return are skipped.
// Prints one summary line; exit status 1 on any failure.
//
//   g++ -std=c++20 -O2 -Iinference -Ip_only/include -Ip_only/bots -DCARRIER_C08 p_only/tests/contract_test.cpp
//   gzip -dc p_only/tests/fixtures/orders_n04.txt.gz > orders.txt
//   gzip -dc p_only/tests/fixtures/stream_n04.txt.gz | ./a.out orders.txt
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
#include <fstream>
#include <map>
#include <set>

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

// Recorded replies: turn -> (team, reply line).
std::map<int, std::vector<std::pair<int, std::string>>> loadOrders(const char* path) {
    std::map<int, std::vector<std::pair<int, std::string>>> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        int turn, team;
        if (!(s >> turn >> team)) continue;
        std::string rest;
        std::getline(s, rest);
        out[turn].push_back({team, rest});
    }
    return out;
}

// Requested cells and acts (target id, -2 pickup, -3 drop, -1 nothing) of every unit.
template <typename NavT>
void parseOrders(const std::vector<std::pair<int, std::string>>& replies, const NavT& nav, const std::vector<int>& pos,
                 std::vector<int>& want, std::vector<int>& acts) {
    want.assign(pos.size(), -1);
    acts.assign(pos.size(), -1);
    for (int i = 0; i < (int)pos.size(); i++) want[i] = pos[i];
    for (const auto& [team, reply] : replies) {
        std::istringstream s(reply);
        int turn;
        s >> turn;
        for (int k = 0; k < 3; k++) {
            std::string mv, act;
            if (!(s >> mv >> act)) break;
            const int id = 3 * team + k;
            if (id >= (int)pos.size() || pos[id] < 0) continue;
            const int m = std::string("SUDLR").find(mv[0]);
            if (m >= 0 && m < 5 && nav.nb[pos[id]][m] >= 0) want[id] = nav.nb[pos[id]][m];
            acts[id] = act == "P" ? -2 : act == "X" ? -3 : act == "-" ? -1 : std::stoi(act);
        }
    }
}

bool flagReturnPending(const Game& g) {
    for (const auto& f : g.flags)
        if ((f.status == "dropped" || f.status == "cooldown") && f.return_at <= g.turn) return true;
    return false;
}

// The kernel's successor of a real turn against the engine's actual next state.
void compare(const std::vector<Unit>& units, const std::vector<Flag>& flags, const std::vector<int>& score,
             const std::set<std::pair<int, int>>& attacks, const Game& real, int turn) {
    const char* bad = nullptr;
    for (size_t i = 0; !bad && i < units.size(); i++) {
        const auto &a = units[i], &b = real.units[i];
        if (a.x != b.x || a.y != b.y) bad = "unit position";
        else if (a.hp != b.hp) bad = "unit HP";
        else if (a.flag != b.flag) bad = "carried flag";
        else if (a.respawn_at != b.respawn_at) bad = "respawn time";
    }
    for (size_t i = 0; !bad && i < flags.size(); i++) {
        const auto &a = flags[i], &b = real.flags[i];
        if (a.status != b.status) bad = "flag status";
        else if (a.x != b.x || a.y != b.y) bad = "flag position";
        else if (a.holder != b.holder) bad = "flag holder";
        else if (a.return_at != b.return_at) bad = "flag return time";
    }
    if (!bad && score != real.score) bad = "score";
    std::set<std::pair<int, int>> realAttacks;
    for (const auto& e : real.events)
        if (e.type == "attack" && e.args.size() >= 2) realAttacks.insert({e.args[0], e.args[1]});
    if (!bad && attacks != realAttacks) bad = "attack events";
    static char msg[96];
    std::snprintf(msg, sizeof msg, "engine mismatch after turn %d: %s", turn, bad ? bad : "");
    check(bad == nullptr, turn + 1, msg);
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    Game g;
#if defined(CARRIER_C08)
    multi::Planner many;
    duel::Player two;
#else
    Planner p;
#endif
    std::vector<Unit> previous;
    int turns = 0, stepped = 0, skipped = 0;
    long long branches = 0, fallback = 0, inconsistent = 0;
    const auto recorded = argc > 1 ? loadOrders(argv[1]) : std::map<int, std::vector<std::pair<int, std::string>>>{};
    // The kernel's prediction of the next real turn, compared once that turn is read.
    bool pending = false;
    int pendingTurn = 0;
    std::vector<Unit> nextUnits;
    std::vector<Flag> nextFlags;
    std::vector<int> nextScore;
    std::set<std::pair<int, int>> nextAttacks;
#if defined(CARRIER_C08)
    multi::Planner kernel;
#endif
    while (read_turn(g)) {
        if (pending && g.turn == pendingTurn + 1) {
            compare(nextUnits, nextFlags, nextScore, nextAttacks, g, pendingTurn);
            stepped++;
        }
        pending = false;
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
        inconsistent = fs.counters.inconsistent;
        auto it = recorded.find(g.turn);
        if (it != recorded.end() && g.turn < g.turns) {
            if (flagReturnPending(g)) {
                skipped++;
            } else {
                std::vector<int> want, acts;
#if defined(CARRIER_C08)
                if (!kernel.initialized) { kernel.nav.init(g); kernel.initialized = true; }
                kernel.respawns(g);
                parseOrders(it->second, kernel.nav, kernel.pos, want, acts);
                Game next;
                if (kernel.stepExact(g, want, acts, next)) {
                    nextUnits = next.units; nextFlags = next.flags; nextScore = next.score; nextAttacks.clear();
                    for (const auto& e : next.events)
                        if (e.type == "attack" && e.args.size() >= 2) nextAttacks.insert({e.args[0], e.args[1]});
                    pending = true;
                }
#else
                parseOrders(it->second, p.nav, p.pos, want, acts);
                auto s = p.initialSim(g);
                std::array<int, 3> ours{};
                for (int k = 0; k < 3; k++) ours[k] = acts[3 * g.me + k];
                p.simStep(g, s, g.turn, want, acts, ours);
                nextUnits = s.u; nextFlags = s.f; nextScore = s.score;
                nextAttacks = std::set<std::pair<int, int>>(s.attacks.begin(), s.attacks.end());
                pending = true;
#endif
                pendingTurn = g.turn;
            }
        }
        previous = g.units;
        turns++;
    }
    check(fallback == 0, g.turn, "a prediction fell back to uniform");
    check(inconsistent == 0, g.turn, "a state handed to the network was inconsistent");
    std::printf("contract: %d turns, %d teams, %lld checks, %d failures, %lld branch views, %lld fallbacks, "
                "%lld inconsistent, engine steps %d (skipped %d)\n",
                turns, g.teams, checks, failures, branches, fallback, inconsistent, stepped, skipped);
    return failures ? 1 : 0;
}
