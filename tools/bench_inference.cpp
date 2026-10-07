// Inference cost on the host CPU, against the ceilings in training/specs/MODEL_MATRIX.md:
//   P: opponent policy for every other team, once per turn       target <= 5 ms per turn
//   V: value per search node (incremental accumulator + tail)    target single-digit microseconds
//
// Reads a seat stream (INIT + TURN blocks) on stdin and replays it `repeat`
// times. Run it on the machine that will host the bot, not on a development
// laptop; tools/env_check.py builds and runs it with any exported model.
//
//   bench_inference [repeat] < seat_stream.txt
// Output: one JSON object with per-turn P timings (p50/p95/p99/max ms) and
// per-node V timings (mean us for the incremental and the full path).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../inference/policy.hpp"
#include "../inference/value.hpp"
#include "stream.hpp"
#include FUSION_MODEL_HEADER

using namespace fusion;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const int repeat = argc > 1 ? std::max(1, std::atoi(argv[1])) : 3;
    const PolicyModel& policy = FUSION_MODEL_NS::kPolicy;
    const ValueModel& valueModel = FUSION_MODEL_NS::kValue;
    Static st;
    if (!readInit(std::cin, st)) return 2;
    std::vector<State> blocks;
    for (State s; readTurn(std::cin, s);) blocks.push_back(s);
    Board board;
    board.init(st);
    TurnView view;
    PolicyFeatures pf;
    PolicyOutput out;
    ValueFeatures vf;
    std::vector<ValueAccumulator> acc(st.teams);
    std::vector<double> turnMs;
    double sink = 0, incrementalUs = 0, fullUs = 0, encodeUs = 0;
    long nodes = 0;
    for (int r = 0; r < repeat; r++)
        for (size_t t = 0; t < blocks.size(); t++) {
            // P: what the bot does once per turn (every opponent team, all heads at every legal move).
            const auto t0 = Clock::now();
            makeView(board, blocks[t], t ? &blocks[t - 1] : nullptr, view);
            for (int team = 0; team < st.teams; team++) {
                if (team == st.me) continue;
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
                sink += out.prob[0];
            }
            turnMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            // V: one node per team per turn, as a search would see consecutive states.
            for (int team = 0; team < st.teams; team++) {
                const auto a = Clock::now();
                encodeValue(board, blocks[t], view, team, vf);
                const auto b = Clock::now();
                update(valueModel, vf, acc[team]);
                sink += valueLogit(valueModel, acc[team]);
                const auto c = Clock::now();
                sink += valueLogit(valueModel, vf);
                const auto d = Clock::now();
                encodeUs += std::chrono::duration<double, std::micro>(b - a).count();
                incrementalUs += std::chrono::duration<double, std::micro>(c - b).count();
                fullUs += std::chrono::duration<double, std::micro>(d - c).count();
                nodes++;
            }
        }
    std::sort(turnMs.begin(), turnMs.end());
    auto pct = [&](double q) { return turnMs.empty() ? 0.0 : turnMs[std::min(turnMs.size() - 1, size_t(q * turnMs.size()))]; };
    std::printf("{\"teams\": %d, \"turns\": %zu, \"policy_turn_ms\": {\"p50\": %.4f, \"p95\": %.4f, \"p99\": %.4f, \"max\": %.4f}, "
                "\"value_node_us\": {\"encode\": %.3f, \"incremental\": %.3f, \"full\": %.3f}, \"nodes\": %ld, \"sink\": %d}\n",
                st.teams, turnMs.size(), pct(0.5), pct(0.95), pct(0.99), turnMs.empty() ? 0.0 : turnMs.back(),
                encodeUs / std::max(1L, nodes), incrementalUs / std::max(1L, nodes), fullUs / std::max(1L, nodes), nodes,
                sink == 12345.678 ? 1 : 0);
    return 0;
}
