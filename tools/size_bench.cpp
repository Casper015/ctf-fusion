// Capacity benchmark driver (tools/capacity_bench.py runs it on the target host).
//
//   size_bench [--verify] <repeat> <value_calls> <tiers.txt> < seat_stream.txt
//
// tiers.txt holds one architecture per line, hidden widths per network:
//   L mover=64 squad=256,64 act=64 target=16 value=512,64
// Each architecture is built at runtime with random int16 weights: integer
// inference costs the same whatever the values, so one binary sweeps every
// size. It is compiled with -DFUSION_MAX_WIDTH at least as wide as the widest
// layer. On every recorded turn it runs what a fusion bot would run:
//
//   P  once per turn: encode and jointPolicy for every opponent team, plus the
//      act and target heads at every legal move (an upper bound: joint-v2
//      needs them only at the moves a learned draw samples);
//   V  value_calls search nodes per turn: encode the node's state, move the
//      team's accumulator to it, run the tail. Nodes alternate between this
//      turn's and the next turn's state, so every update has real deltas.
//
// Output: one JSON line per architecture with P per turn (p50/p95/p99/max ms,
// joint part and heads), V per node (encode, incremental, full pass in us) and
// V per turn at value_calls nodes.
//
// --verify does no timing: every V node's incremental result must equal the
// full pass bit for bit, which also exercises buffers wider than 256. It
// prints one "verified" line per architecture; exit 1 on any mismatch.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../inference/policy.hpp"
#include "../inference/value.hpp"
#include "stream.hpp"

using namespace fusion;
using Clock = std::chrono::steady_clock;

namespace {

uint64_t g_rng = 0x2545F4914F6CDD1Dull;

int32_t nextInt(int lo, int hi) {  // xorshift64, inclusive range
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return lo + static_cast<int32_t>(g_rng % static_cast<uint64_t>(hi - lo + 1));
}

// A network that owns its weights.
struct OwnedNet {
    std::vector<std::vector<int16_t>> w;
    std::vector<std::vector<int32_t>> b;
    std::vector<Layer> layers;
    Net net;
    long params = 0;
};

void build(OwnedNet& n, int in, const std::vector<int>& hidden, int out) {
    std::vector<int> dims{in};
    dims.insert(dims.end(), hidden.begin(), hidden.end());
    dims.push_back(out);
    const int L = static_cast<int>(dims.size()) - 1;
    n.w.assign(L, {});
    n.b.assign(L, {});
    n.layers.assign(L, {});
    for (int l = 0; l < L; l++) {
        n.w[l].resize(static_cast<size_t>(dims[l]) * dims[l + 1]);
        n.b[l].resize(dims[l + 1]);
        for (auto& v : n.w[l]) v = static_cast<int16_t>(nextInt(-31, 31));
        for (auto& v : n.b[l]) v = nextInt(-512, 512);
        n.layers[l] = Layer{dims[l], dims[l + 1], n.w[l].data(), n.b[l].data(), l == 0 ? 9 : 7, l + 1 < L};
        n.params += static_cast<long>(n.w[l].size() + n.b[l].size());
    }
    n.net = Net{n.layers.data(), L, 14};
}

struct Tier {
    std::string name;
    std::vector<int> mover, squad, act, target, value;
};

std::vector<int> widths(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

std::vector<Tier> readTiers(const char* path) {
    std::ifstream in(path);
    std::vector<Tier> tiers;
    for (std::string line; std::getline(in, line);) {
        std::istringstream ss(line);
        Tier t;
        if (!(ss >> t.name)) continue;
        for (std::string kv; ss >> kv;) {
            const auto eq = kv.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = kv.substr(0, eq), val = kv.substr(eq + 1);
            if (key == "mover") t.mover = widths(val);
            else if (key == "squad") t.squad = widths(val);
            else if (key == "act") t.act = widths(val);
            else if (key == "target") t.target = widths(val);
            else if (key == "value") t.value = widths(val);
        }
        tiers.push_back(t);
    }
    return tiers;
}

int widest(const Tier& t) {
    int w = std::max({kMoverIn, kSquadIn, kJoint, kV});
    for (const auto* v : {&t.mover, &t.squad, &t.act, &t.target, &t.value})
        for (int x : *v) w = std::max(w, x);
    return w;
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * v.size()))];
}

}  // namespace

int main(int argc, char** argv) {
    int a = 1;
    const bool verify = argc > 1 && std::strcmp(argv[1], "--verify") == 0;
    if (verify) a++;
    if (argc - a != 3) {
        std::fprintf(stderr, "usage: size_bench [--verify] repeat value_calls tiers.txt < seat_stream\n");
        return 2;
    }
    const int repeat = std::max(1, std::atoi(argv[a]));
    const int valueCalls = std::max(0, std::atoi(argv[a + 1]));
    const std::vector<Tier> tiers = readTiers(argv[a + 2]);
    Static st;
    if (!readInit(std::cin, st)) return 2;
    std::vector<State> blocks;
    for (State s; readTurn(std::cin, s);) blocks.push_back(s);
    if (blocks.size() < 2) return 2;
    Board board;
    board.init(st);
    std::vector<TurnView> views(blocks.size());
    for (size_t t = 0; t < blocks.size(); t++) makeView(board, blocks[t], t ? &blocks[t - 1] : nullptr, views[t]);

    int failures = 0;
    for (const Tier& tier : tiers) {
        if (widest(tier) > kMaxWidth) {
            std::fprintf(stderr, "%s: a layer is %d wide; rebuild with -DFUSION_MAX_WIDTH >= %d\n", tier.name.c_str(),
                         widest(tier), widest(tier));
            return 2;
        }
        OwnedNet mover, squad, act, target, value;
        build(mover, kMoverIn, tier.mover, 1);
        build(squad, kSquadIn, tier.squad, kJoint);
        build(act, kMoverIn, tier.act, kActs);
        build(target, kTargetIn, tier.target, 1);
        build(value, kV, tier.value, 1);
        const PolicyModel policy{mover.net, squad.net, act.net, target.net};
        const ValueModel valueModel{value.net};

        std::vector<double> turnMs, jointMs, valueTurnMs;
        double encodeUs = 0, incrementalUs = 0, fullUs = 0, sink = 0;
        long nodes = 0, mismatches = 0;
        PolicyFeatures pf;
        PolicyOutput out;
        ValueFeatures vf;
        std::vector<ValueAccumulator> acc(st.teams);
        const size_t T = blocks.size() - 1;  // the last block only serves as a successor state
        for (int r = 0; r < repeat; r++)
            for (size_t t = 0; t < T; t++) {
                const State& s = blocks[t];
                const auto t0 = Clock::now();
                double joint = 0;
                for (int team = 0; team < st.teams; team++) {
                    if (team == st.me) continue;
                    const auto j0 = Clock::now();
                    encodePolicy(board, s, views[t], team, pf);
                    jointPolicy(policy, pf, out);
                    joint += std::chrono::duration<double, std::milli>(Clock::now() - j0).count();
                    double probs[kCand];
                    for (int k = 0; k < 3; k++)
                        for (int m = 0; m < kMoves; m++)
                            if (pf.legal[k][m]) {
                                actPolicy(policy, pf, k, m, probs);
                                sink += probs[0];
                                targetPolicy(policy, pf, k, m, probs);
                                sink += probs[0];
                            }
                    sink += out.prob[0];
                }
                turnMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                jointMs.push_back(joint);

                double turnValue = 0;
                for (int i = 0; i < valueCalls; i++) {
                    const int team = i % st.teams;
                    const size_t at = t + (i / st.teams) % 2;  // this turn's or the next turn's state
                    const auto v0 = Clock::now();
                    encodeValue(board, blocks[at], views[at], team, vf);
                    const auto v1 = Clock::now();
                    update(valueModel, vf, acc[team]);
                    const int32_t inc = valueRaw(valueModel, acc[team]);
                    const auto v2 = Clock::now();
                    sink += inc;
                    encodeUs += std::chrono::duration<double, std::micro>(v1 - v0).count();
                    incrementalUs += std::chrono::duration<double, std::micro>(v2 - v1).count();
                    turnValue += std::chrono::duration<double, std::milli>(v2 - v0).count();
                    nodes++;
                    if (verify || i < st.teams) {  // a full pass on a sample of nodes (every node when verifying)
                        int32_t x[kMaxWidth], raw[kMaxWidth];
                        for (int f = 0; f < kV; f++) x[f] = vf[f];
                        const auto f0 = Clock::now();
                        forwardRaw(valueModel.net, x, raw);
                        fullUs += std::chrono::duration<double, std::micro>(Clock::now() - f0).count();
                        mismatches += raw[0] != inc;
                    }
                }
                valueTurnMs.push_back(turnValue);
            }
        const long sampled = verify ? nodes : static_cast<long>(T) * repeat * std::min(valueCalls, st.teams);
        if (verify) {
            std::printf("{\"tier\": \"%s\", \"verified\": %s, \"nodes\": %ld, \"mismatches\": %ld, \"turns\": %zu}\n",
                        tier.name.c_str(), mismatches ? "false" : "true", nodes, mismatches, turnMs.size());
            failures += mismatches > 0;
            continue;
        }
        std::printf("{\"tier\": \"%s\", \"teams\": %d, \"map_size\": %d, \"turns\": %zu, \"value_calls\": %d, "
                    "\"params\": {\"mover\": %ld, \"squad\": %ld, \"act\": %ld, \"target\": %ld, \"value\": %ld}, "
                    "\"policy_turn_ms\": {\"p50\": %.4f, \"p95\": %.4f, \"p99\": %.4f, \"max\": %.4f}, "
                    "\"policy_joint_ms\": {\"p50\": %.4f, \"p99\": %.4f}, "
                    "\"value_node_us\": {\"encode\": %.3f, \"incremental\": %.3f, \"full\": %.3f}, "
                    "\"value_turn_ms\": {\"p50\": %.4f, \"p99\": %.4f, \"max\": %.4f}, \"mismatches\": %ld, \"sink\": %d}\n",
                    tier.name.c_str(), st.teams, st.size, turnMs.size(), valueCalls, mover.params, squad.params, act.params,
                    target.params, value.params, pct(turnMs, 0.5), pct(turnMs, 0.95), pct(turnMs, 0.99),
                    pct(turnMs, 1.0), pct(jointMs, 0.5), pct(jointMs, 0.99), encodeUs / std::max(1L, nodes),
                    incrementalUs / std::max(1L, nodes), fullUs / std::max(1L, sampled), pct(valueTurnMs, 0.5),
                    pct(valueTurnMs, 0.99), pct(valueTurnMs, 1.0), mismatches, sink == 12345.678 ? 1 : 0);
        std::fflush(stdout);
    }
    return failures ? 1 : 0;
}
