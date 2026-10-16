// Contract checks of the agent_v2 carrier (task T0065; tests/test_agent_v2.py runs them).
//
//   build: c++ -std=c++20 -O2 -I inference -I tools/cpp/meow_rl -DFUSION_TIER_PXXL_VM=1
//          (or -DFUSION_TIER_CSP=1 -DFUSION_MAX_WIDTH=2048: T0067's per-count specialists)
//          -DACTOR_HEADER='"<actor.hpp>"' -DCRITIC_HEADER='"<critic.hpp>"' tools/cpp/agent_v2/check.cpp
//   check resolver <stream.txt>   on every turn of a recorded seat stream and several feasible joints: assign_v1 with
//                                 its candidate pruning == the unpruned brute force over every target tuple, and the
//                                 engine's rules hold (no carrier or immune unit attacks, only living non-immune
//                                 enemies are targeted, carriers wait, nobody drops, an attack has positive value)
//   check p <stream.txt>          our P through the carrier: every predicted unit's movement marginals sum to 1,
//                                 every target slot is empty or a unit of another team, the scenario table is a set of
//                                 complete opponent orders, our own team is never predicted, and a prediction is
//                                 refused for a state it was not made for (cache identity)
//   check isolation <stream.txt>  a search runtime and an actor-direct runtime read the same stream; after every turn
//                                 the search one's root prediction, history and default order equal the other's
//   check tree <stream.txt> <turn>...  the fixed-work tree of each turn (no deadline): one JSON line per turn on stdout
//                                 and the leaves' 416 float32 value features in $AGENT_V2_TREE_LEAVES
//   check bank                    the compiled-in P bank (inference/agent_v2/p_bank.hpp) serves every team count 2-15:
//                                 "ok bank <bank> models <count>:<model id> ... distinct <n>" (n: distinct policy models)
//   check step                    stdin: a block (INIT + TURN), a refresh seed, one "move act target" line per unit,
//                                 then optionally "spots" and one "x y" per flag (-1 -1: not returning); stdout: the
//                                 successor (turn, score, units, flags, attacks) through Simulator / fusion::step
// Each mode prints "ok ..." or the first mismatch, and exits 0 / 1.
#ifndef FUSION_MAX_WIDTH
#define FUSION_MAX_WIDTH 1024
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include ACTOR_HEADER
#include CRITIC_HEADER
#include "../../../inference/agent_v2/runtime.hpp"
#include "../../../inference/agent_v2/search.hpp"

namespace {

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

int fail(const std::string& what) {
    std::printf("mismatch %s\n", what.c_str());
    return 1;
}

using RT = agent_v2::Runtime<agent_v2::NoSearch>;

int check_resolver(const char* path) {
    RT rt;
    long cases = 0, attacks = 0;
    for (auto& b : blocks(slurp(path))) {
        if (!rt.observe_block(b).empty()) {
            rt.refuse();
            continue;
        }
        std::vector<agent_v2::Scenario> table;
        rt.pred.scenarios(rt.pred.root, rt.s, rt.hist.parent(), agent_v2::kResolverScenarios, agent_v2::resolver_salt(rt.seed, rt.s.turn, 1), table);
        const int me = rt.cache.st.me;
        for (int j = 0; j < 125; j += 3 + rt.s.turn % 4) {
            if (!rt.obs.joint_mask[j]) continue;
            const std::array<int, 3> moves{j / 25, j / 5 % 5, j % 5};
            auto a = rt.resolver.resolve(rt.cache, rt.s, rt.view, me, moves, table, 99, true);
            auto f = rt.resolver.resolve(rt.cache, rt.s, rt.view, me, moves, table, 99, false);
            if (a.target != f.target || a.score != f.score) return fail("pruning changed the choice at turn " + std::to_string(rt.s.turn));
            for (int k = 0; k < 3; k++) {
                const int uid = 3 * me + k, t = a.target[k];
                const auto& o = a.orders[k];
                if (o.move != moves[k]) return fail("an order changed the move");
                if (o.act == fusion::kOrderDrop) return fail("assign_v1 dropped a flag");
                if (rt.s.units[uid].flag >= 0 && o.act != fusion::kOrderWait) return fail("a carrier did not wait");
                if (t < 0) continue;
                attacks++;
                if (rt.view.immune[uid] || rt.s.units[uid].flag >= 0 || rt.view.origin[uid] < 0) return fail("an unable unit attacks");
                if (t / 3 == me || rt.view.origin[t] < 0 || rt.view.immune[t]) return fail("an invalid target");
                if (o.act != fusion::kOrderAttack || o.target != t) return fail("attack order mismatch");
            }
            if (a.target != std::array<int, 3>{-1, -1, -1} && !(a.score > 0)) return fail("a non-positive attack was chosen");
            cases++;
        }
        rt.answer(0);
    }
    std::printf("ok resolver cases %ld attacks %ld\n", cases, attacks);
    return 0;
}

int check_p(const char* path) {
    RT rt;
    long teams = 0, turns = 0;
    fusion::State other;
    bool have_other = false;
    for (auto& b : blocks(slurp(path))) {
        if (!rt.observe_block(b).empty()) {
            rt.refuse();
            continue;
        }
        const auto& v = rt.pred.root.view;
        const int U = int(rt.s.units.size());
        if (agent_v2::state_key(v.state()) != agent_v2::state_key(rt.s)) return fail("P was not given the raw state");
        for (int t = 0; t < rt.cache.st.teams; t++) {
            if (t == rt.cache.st.me) {
                if (v.has(t)) return fail("our own team was predicted");
                continue;
            }
            if (!v.has(t)) continue;
            teams++;
            for (int k = 0; k < 3; k++) {
                const int uid = 3 * t + k;
                double total = 0;
                for (int m = 0; m < 5; m++) total += v.move(uid, m);
                if (v.origin(uid) >= 0 && std::abs(total - 1.0) > 1e-9) return fail("marginals do not sum to 1");
                for (int c = 0; c < fusion::kCand; c++) {
                    const int victim = v.candidate(uid, c);
                    if (victim != -1 && (victim < 0 || victim >= U || victim / 3 == t)) return fail("a target slot names no enemy");
                }
            }
        }
        std::vector<agent_v2::Scenario> table;
        if (!rt.pred.scenarios(rt.pred.root, rt.s, rt.hist.parent(), 4, 7, table) || int(table.size()) != 4) return fail("scenario table refused");
        for (auto& sc : table) {
            if (int(sc.size()) != U) return fail("scenario not complete");
            for (int uid = 0; uid < U; uid++)
                if (sc[uid].move < 0 || sc[uid].move >= 5 ||
                    (sc[uid].act == fusion::kOrderAttack && (sc[uid].target < 0 || sc[uid].target / 3 == uid / 3)))
                    return fail("scenario order invalid");
        }
        if (have_other) {                               // cache identity: the root prediction is not the previous state's
            agent_v2::Marginals q;
            const long long before = rt.pred.counters.inconsistent;
            if (rt.pred.marginals(rt.pred.root, other, rt.hist.parent(), q) || rt.pred.counters.inconsistent != before + 1)
                return fail("a prediction was served for another state");
            if (rt.pred.marginals(rt.pred.root, rt.s, nullptr, q) || rt.pred.counters.inconsistent != before + 2)
                return fail("a prediction was served for another history");
        }
        other = rt.s, have_other = true;
        rt.answer(0);
        turns++;
    }
    std::printf("ok p turns %ld teams %ld model %s\n", turns, teams, rt.pred.model_id());
    return 0;
}

int check_isolation(const char* path) {
    agent_v2::Runtime<agent_v2::Search> a;
    RT b;
    long turns = 0;
    for (auto& blk : blocks(slurp(path))) {
        const auto far = std::chrono::steady_clock::now() + std::chrono::hours(1);   // no deadline pressure
        const std::string wa = a.observe_block(blk), wb = b.observe_block(blk);
        if (wa != wb) return fail("refusals differ");
        if (!wa.empty()) {
            a.refuse(), b.refuse();
            continue;
        }
        int64_t sc[3][5];
        agent_v2::Actor::scores(a.obs, sc);
        const int joint = agent_v2::Actor::sample(sc, a.obs, agent_v2::uniform(a.seed, a.s.turn));
        const auto def_a = a.resolve(joint), def_b = b.resolve(joint);
        if (def_a.target != def_b.target || def_a.score != def_b.score) return fail("default orders differ");
        a.search.choose(a, sc, joint, far);
        agent_v2::Marginals qa, qb;
        a.pred.marginals(a.pred.root, a.s, a.hist.parent(), qa);
        b.pred.marginals(b.pred.root, b.s, b.hist.parent(), qb);
        if (qa != qb) return fail("search changed the root prediction");
        const auto again = a.resolve(joint);
        if (again.target != def_a.target || again.score != def_a.score) return fail("search changed the default order");
        a.answer(joint), b.answer(joint);
        if (agent_v2::state_key(a.hist.previous) != agent_v2::state_key(b.hist.previous)) return fail("search changed the history");
        turns++;
    }
    std::printf("ok isolation turns %ld trajectories %ld branches %lld\n", turns, a.search.trajectories, a.pred.counters.branches);
    return 0;
}

uint64_t fnv(uint64_t h, const void* data, size_t n) {
    auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

int check_tree(const char* path, int argc, char** argv) {
    std::set<int> wanted;
    for (int i = 0; i < argc; i++) wanted.insert(std::atoi(argv[i]));
    const char* lp = std::getenv("AGENT_V2_TREE_LEAVES");
    FILE* leaves = std::fopen(lp ? lp : "leaves.f32", "wb");
    RT rt;
    agent_v2::Simulator sim;
    long rows = 0;
    for (auto& blk : blocks(slurp(path))) {
        if (!rt.observe_block(blk).empty()) {
            rt.refuse();
            continue;
        }
        int64_t sc[3][5];
        agent_v2::Actor::scores(rt.obs, sc);
        const int joint = agent_v2::Actor::sample(sc, rt.obs, agent_v2::uniform(rt.seed, rt.s.turn));
        if (wanted.count(rt.s.turn)) {
            agent_v2::Tree tree;
            agent_v2::setup(rt, sc, joint, tree);
            uint64_t h = fnv(1469598103934665603ull, &tree.T, sizeof tree.T);
            for (int r : tree.roots) h = fnv(h, &r, sizeof r);
            std::string js = "{\"turn\":" + std::to_string(rt.s.turn) + ",\"teams\":" + std::to_string(rt.cache.st.teams) +
                             ",\"actor\":" + std::to_string(joint) + ",\"samples\":" + std::to_string(agent_v2::kSamples) +
                             ",\"predictor\":\"" + rt.pred.model_id() + "\",\"roots\":[";
            for (size_t i = 0; i < tree.roots.size(); i++) js += (i ? "," : "") + std::to_string(tree.roots[i]);
            js += "],\"leaves\":[";
            agent_v2::Leaf leaf;
            for (int ri = 0; ri < int(tree.roots.size()); ri++)
                for (int smp = 0; smp < agent_v2::kSamples; smp++) {
                    agent_v2::trajectory(rt, tree, ri, smp, sim, leaf);
                    long at = -1;
                    if (!leaf.terminal) {
                        std::fwrite(leaf.feats.data(), sizeof(float), leaf.feats.size(), leaves);
                        at = rows++;
                        h = fnv(h, leaf.feats.data(), sizeof(float) * leaf.feats.size());
                    }
                    int place[5] = {ri, smp, int(at), int(leaf.terminal), leaf.plies};
                    h = fnv(h, place, sizeof place);
                    h = fnv(h, &leaf.reward, sizeof leaf.reward);
                    for (auto* os : {&leaf.root_orders, &leaf.cont_orders})
                        for (auto& o : *os) {
                            int f3[3] = {o.move, o.act, o.target};
                            h = fnv(h, f3, sizeof f3);
                        }
                    char buf[200];
                    std::snprintf(buf, sizeof buf, "%s{\"root\":%d,\"sample\":%d,\"terminal\":%s,\"plies\":%d,\"reward\":%.17g,\"cont\":%d,\"row\":%ld}",
                                  ri || smp ? "," : "", ri, smp, leaf.terminal ? "true" : "false", leaf.plies, leaf.reward, leaf.cont, at);
                    js += buf;
                }
            char tail[64];
            std::snprintf(tail, sizeof tail, "],\"hash\":\"%016llx\"}", (unsigned long long)h);
            std::printf("%s%s\n", js.c_str(), tail);
        }
        rt.answer(joint);
    }
    std::fclose(leaves);
    return 0;
}

int check_step() {
    std::string text, line;
    while (std::getline(std::cin, line)) {
        text += line + "\n";
        if (line.rfind("END", 0) == 0) break;
    }
    std::istringstream in(text);
    fusion::Static st;
    fusion::State s;
    if (!fusion::readInit(in, st) || !fusion::readTurn(in, s)) return fail("no block");
    agent_v2::MapCache c;
    c.init(st);
    uint64_t seed = 1;
    std::cin >> seed;
    std::vector<fusion::Order> orders(s.units.size());
    for (auto& o : orders) std::cin >> o.move >> o.act >> o.target;
    std::vector<std::array<int, 2>> spots;
    std::string word;
    fusion::State next;
    fusion::Workspace ws;
    if (std::cin >> word && word == "spots") {
        spots.assign(s.flags.size(), {-1, -1});
        for (auto& p : spots) std::cin >> p[0] >> p[1];
        fusion::step(c.board, c.rules, s, orders.data(), next, ws, nullptr, spots.data());
    } else {
        agent_v2::Simulator sim;
        next = sim.step(c, s, orders, seed);
    }
    std::printf("%d", next.turn);
    for (int v : next.score) std::printf(" %d", v);
    for (auto& u : next.units) std::printf(" %d %d %d %d %d", u.x, u.y, u.hp, u.flag, u.respawnAt);
    for (auto& f : next.flags) std::printf(" %d %d %d %d %d", f.status, f.x, f.y, f.holder, f.returnAt);
    std::printf(" |");
    for (auto& a : next.attacks) std::printf(" %d %d", a.unit, a.target);
    std::printf("\n");
    return 0;
}

int check_bank() {
    std::set<const fusion::PolicyModel*> distinct;
    std::string line = std::string("ok bank ") + agent_v2::kPBank + " models";
    for (int teams = 2; teams <= 15; teams++) {
        const auto* set = agent_v2::p_bank::forTeams(teams);
        if (!set || !set->policy) {
            std::printf("no model for %d teams in bank %s\n", teams, agent_v2::kPBank);
            return 1;
        }
        distinct.insert(set->policy);
        line += " " + std::to_string(teams) + ":" + agent_v2::kPModelPrefix + set->id;
    }
    std::printf("%s distinct %d\n", line.c_str(), int(distinct.size()));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "step") return check_step();
    if (mode == "bank") return check_bank();
    if (mode == "resolver" && argc > 2) return check_resolver(argv[2]);
    if (mode == "p" && argc > 2) return check_p(argv[2]);
    if (mode == "isolation" && argc > 2) return check_isolation(argv[2]);
    if (mode == "tree" && argc > 2) return check_tree(argv[2], argc - 3, argv + 3);
    std::fprintf(stderr, "usage: check resolver|p|isolation <stream.txt> | check tree <stream.txt> <turn>... | check step|bank\n");
    return 2;
}
