// Runtime contract check for an exported model (see tests/test_pipeline.py):
//
//   1. No heap allocation on the per-turn path after the first turn: view,
//      policy and value encoders, joint/act/target policy, value (full and
//      incremental) and one simulator step for every team.
//   2. The incremental value accumulator gives the same raw output as the full
//      forward pass, for every team on every turn (moved from the previous turn).
//
// Built and run by tests/test_pipeline.py (fusion.native.model_tool):
//   g++ -std=c++20 -O2 -DFUSION_MODEL_HEADER="<header>" -DFUSION_MODEL_NS=<namespace>
//       -Iinference tools/cpp/runtime_check.cpp -o build/runtime_check
//   runtime_check < seat_stream.txt
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>

#include "../../inference/policy.hpp"
#include "../../inference/sim.hpp"
#include "../../inference/value.hpp"
#include "stream.hpp"
#include FUSION_MODEL_HEADER

namespace {
long g_allocations = 0;
bool g_counting = false;
}  // namespace

// Counting replacements of the global allocation functions. (GCC flags the
// malloc/free pairing inside them as mismatched; both sides are replaced, so
// the pairing is consistent.)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    if (g_counting) g_allocations++;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace fusion;

int main() {
    const PolicyModel& policy = FUSION_MODEL_NS::kPolicy;
    const ValueModel& valueModel = FUSION_MODEL_NS::kValue;
    Static st;
    if (!readInit(std::cin, st)) return 2;
    std::vector<State> blocks;
    for (State s; readTurn(std::cin, s);) blocks.push_back(s);
    Board board;
    board.init(st);
    const Rules rules = rulesOf(st);
    TurnView view;
    PolicyFeatures pf;
    ValueFeatures vf;
    PolicyOutput out;
    std::vector<ValueAccumulator> acc(st.teams);
    std::vector<Order> orders(blocks.empty() ? 0 : blocks[0].units.size());
    Workspace ws;
    State next;
    long turnsAllocating = 0, accumulatorMismatches = 0, checks = 0;
    double sink = 0;
    for (size_t t = 0; t < blocks.size(); t++) {
        g_allocations = 0;
        g_counting = t > 0;  // the first turn sizes every buffer
        makeView(board, blocks[t], t ? &blocks[t - 1] : nullptr, view);
        for (int team = 0; team < st.teams; team++) {
            encodePolicy(board, blocks[t], view, team, pf);
            jointPolicy(policy, pf, out);
            double probs[kCand];
            for (int k = 0; k < 3; k++)
                for (int m = 0; m < kMoves; m++)
                    if (pf.legal[k][m]) {
                        actPolicy(policy, pf, k, m, probs);
                        targetPolicy(policy, pf, k, m, probs);
                        sink += probs[0];
                    }
            encodeValue(board, blocks[t], view, team, vf);
            const double full = valueLogit(valueModel, vf);
            update(valueModel, vf, acc[team]);
            const double incremental = valueLogit(valueModel, acc[team]);
            accumulatorMismatches += full != incremental;
            checks++;
            sink += out.prob[0] + full;
        }
        step(board, rules, blocks[t], orders.data(), next, ws);
        g_counting = false;
        turnsAllocating += g_allocations > 0;
    }
    std::printf("turns %zu allocating_turns %ld accumulator_checks %ld accumulator_mismatches %ld (%g)\n", blocks.size(),
                turnsAllocating, checks, accumulatorMismatches, sink > 0 ? 0.0 : 1.0);
    return turnsAllocating || accumulatorMismatches ? 1 : 0;
}
