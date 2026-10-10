// Value-network cost on real states with a real exported model (tools/host/value_cost.py runs it).
//
//   value_bench [--verify] <calls_per_turn> < seat_stream.txt
//
// Compiled against one exported header (fusion.native.model_tool: FUSION_MODEL_HEADER, FUSION_MODEL_NS,
// FUSION_MAX_WIDTH and the feature families). On every recorded turn, for each of `calls_per_turn` value
// calls (the carrier's invocation pattern: v12 evaluates a settled successor per rollout scenario, C08 per
// refine scenario or surviving plan), alternating between this turn's and the next turn's state as a search
// would:
//
//   view     makeView of the state (the carriers build it per call)
//   encode   the family's features (v3: the combat summary, then the 108 features)
//   full     a full forward pass (what the P-only carriers run today)
//   incr     the accumulator moved from the previous call's state, then the tail (first-layer reuse)
//
// The tail (every layer after the first) is the same work in both: for a wide-first network it is most of
// the cost, and no first-layer cache removes it. The model's policy is timed too, once per turn for every
// opponent team as a P-only root view does it: the shared view and (p3) combat summary, encoding, and the
// joint movement head (policy_view_us per turn, policy_encode_us and policy_joint_us per team). Output: one JSON line with p50/p95/p99/max microseconds per
// call for each part, per-turn totals at `calls_per_turn`, the share of the first layer in a full pass, the
// count of changed first-layer inputs per incremental update (the reuse actually available), and the static
// weight bytes. --verify does no timing: incremental and full results must be equal bit for bit.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "family.hpp"
#include "stream.hpp"
#include FUSION_MODEL_HEADER

using namespace fusion;
using namespace fusion::tools;
using Clock = std::chrono::steady_clock;

namespace {

double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

struct Series {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double q(double p) {
        if (v.empty()) return 0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))];
    }
    double mean() const { double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); }
};

void print(const char* name, Series& s, bool last = false) {
    std::printf("\"%s\": {\"mean\": %.3f, \"p50\": %.3f, \"p95\": %.3f, \"p99\": %.3f, \"max\": %.3f}%s", name, s.mean(), s.q(0.5),
                s.q(0.95), s.q(0.99), s.q(1.0), last ? "" : ", ");
}

long weightBytes(const Net& n) {
    long b = 0;
    for (int i = 0; i < n.count; i++) b += 2L * n.layers[i].in * n.layers[i].out + 4L * n.layers[i].out + (n.layers[i].wT ? 2L * n.layers[i].in * n.layers[i].out : 0);
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    bool verify = argc > 1 && std::strcmp(argv[1], "--verify") == 0;
    const int calls = argc > 1 + verify ? std::max(1, std::atoi(argv[1 + verify])) : 8;
    const VModel& model = FUSION_MODEL_NS::kValue;
    Static st;
    if (!readInit(std::cin, st)) return 2;
    std::vector<State> blocks;
    for (State s; readTurn(std::cin, s);) blocks.push_back(s);
    Board board;
    board.init(st);
    const Rules rules = rulesOf(st);
    TurnView view;
    CombatSummary cs;
    VFeatures f;
    VAccumulator acc;
    Series tView, tEncode, tFull, tIncr, tFirst, changed, turnFull, turnIncr, pView, pEncode, pJoint;
    const PModel& policy = FUSION_MODEL_NS::kPolicy;
    PFeatures pf;
    PolicyOutput po;
    long mismatches = 0, checks = 0;
    double sink = 0;
    const Layer& L0 = model.net.layers[0];
    for (size_t t = 0; t + 1 < blocks.size(); t++) {
        double full = 0, incr = 0;
        {
            auto a = Clock::now();
            makeView(board, blocks[t], t ? &blocks[t - 1] : nullptr, view);
#if FUSION_POLICY_FAMILY == 3
            summarize(board, rules, blocks[t], view, cs);  // once per state, shared by every team
#endif
            auto b = Clock::now();
            if (!verify) pView.add(us(a, b));
            for (int team = 0; team < st.teams; team++) {
                if (team == st.me) continue;
                auto c = Clock::now();
                encodeP(board, blocks[t], view, cs, team, pf);
                auto d = Clock::now();
                jointPolicy(policy, pf, po);
                auto e = Clock::now();
                sink += po.prob[0];
                if (!verify) { pEncode.add(us(c, d)); pJoint.add(us(d, e)); }
            }
        }
        for (int c = 0; c < calls; c++) {
            const State& s = blocks[t + (c & 1)];
            const int team = st.me;
            auto a = Clock::now();
            makeView(board, s, nullptr, view);
            auto b = Clock::now();
#if FUSION_VALUE_FAMILY == 3
            summarize(board, rules, s, view, cs);  // part of the v3 encoding cost
#endif
            encodeV(board, s, view, cs, team, f);
            auto e = Clock::now();
            int moved = 0;
            if (acc.valid)
                for (int i = 0; i < kVIn; i++) moved += f[i] != acc.features[i];
            const double vf = valueLogit(model, f);
            auto g = Clock::now();
            update(model, f, acc);
            const double vi = valueLogit(model, acc);
            auto h = Clock::now();
            int32_t pre[kMaxWidth];
            kernel::layer(L0.w, L0.b, f.data(), L0.in, L0.out, pre);
            auto k = Clock::now();
            sink += vf + vi + pre[0];
            checks++;
            mismatches += vf != vi;
            if (verify) continue;
            tView.add(us(a, b));
            tEncode.add(us(b, e));
            tFull.add(us(e, g));
            tIncr.add(us(g, h));
            tFirst.add(us(h, k));
            changed.add(moved);
            full += us(a, g);
            incr += us(a, b) + us(b, e) + us(g, h);
        }
        if (!verify) { turnFull.add(full / 1000.0); turnIncr.add(incr / 1000.0); }
    }
    if (verify) {
        std::printf("verified checks %ld mismatches %ld (%g)\n", checks, mismatches, sink > 0 ? 0.0 : 1.0);
        return mismatches ? 1 : 0;
    }
    std::printf("{\"calls_per_turn\": %d, \"turns\": %zu, \"value_in\": %d, \"first_layer\": [%d, %d], \"layers\": %d, "
                "\"weight_bytes\": %ld, \"max_width\": %d, \"mismatches\": %ld, ",
                calls, blocks.size() - 1, kVIn, L0.in, L0.out, model.net.count, weightBytes(model.net), kMaxWidth, mismatches);
    print("view_us", tView);
    print("encode_us", tEncode);
    print("full_us", tFull);
    print("incremental_us", tIncr);
    print("first_layer_us", tFirst);
    print("changed_inputs", changed);
    print("turn_full_ms", turnFull);
    print("turn_incremental_ms", turnIncr);
    print("policy_view_us", pView);
    print("policy_encode_us", pEncode);
    print("policy_joint_us", pJoint);
    std::printf("\"first_layer_share\": %.4f, \"sink\": %g}\n", tFull.mean() > 0 ? tFirst.mean() / tFull.mean() : 0.0, sink > 0 ? 0.0 : 1.0);
    return mismatches ? 1 : 0;
}
