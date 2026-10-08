// Parity driver: runs an exported model on every (turn, team) of a match
// stream and writes the raw int32 accumulators to stdout, so the Python
// emulation (fusion/quant.py) can be compared bit for bit.
//
// Built by fusion.native.infer_check (tests/test_pipeline.py):
//   g++ -std=c++20 -O2 -DFUSION_MODEL_HEADER="<weights_n08.hpp>" -DFUSION_MODEL_NS=fusion::models::n08
//       -Iinference tools/infer_check.cpp -o build/infer_check
//
// Record per (turn, team), int32: mover[3][5], squad[125], act[3][5][4],
// target[3][kCand][5], value[1]; entries that the model does not compute
// (illegal moves, empty candidate slots) are 0.
#include <cstdio>
#include <iostream>
#include <vector>

#include "../inference/policy.hpp"
#include "../inference/value.hpp"
#include "stream.hpp"
#include FUSION_MODEL_HEADER

using namespace fusion;

int main() {
    std::ios::sync_with_stdio(false);
    binaryStdout();  // raw int32 records
    const PolicyModel& policy = FUSION_MODEL_NS::kPolicy;
    const ValueModel& valueModel = FUSION_MODEL_NS::kValue;
    Static st;
    if (!readInit(std::cin, st)) return 1;
    Board board;
    board.init(st);
    State prev, now;
    bool havePrev = false;
    PolicyFeatures pf;
    TurnView view;
    ValueFeatures vf;
    int32_t x[kMaxWidth], acc[kMaxWidth];
    std::vector<int32_t> rec;
    while (readTurn(std::cin, now)) {
        makeView(board, now, havePrev ? &prev : nullptr, view);
        for (int team = 0; team < st.teams; team++) {
            encodePolicy(board, now, view, team, pf);
            encodeValue(board, now, view, team, vf);
            rec.clear();
            for (int k = 0; k < 3; k++)
                for (int m = 0; m < kMoves; m++) {
                    acc[0] = 0;
                    if (pf.legal[k][m]) { moverInput(pf, k, m, x); forwardRaw(policy.mover, x, acc); }
                    rec.push_back(acc[0]);
                }
            squadInput(pf, x);
            forwardRaw(policy.squad, x, acc);
            rec.insert(rec.end(), acc, acc + kJoint);
            for (int k = 0; k < 3; k++)
                for (int m = 0; m < kMoves; m++) {
                    std::fill(acc, acc + kActs, 0);
                    if (pf.legal[k][m]) { moverInput(pf, k, m, x); forwardRaw(policy.act, x, acc); }
                    rec.insert(rec.end(), acc, acc + kActs);
                }
            for (int k = 0; k < 3; k++)
                for (int c = 0; c < kCand; c++)
                    for (int m = 0; m < kMoves; m++) {
                        acc[0] = 0;
                        if (pf.cand[k][c] >= 0) { targetInput(pf, k, c, m, x); forwardRaw(policy.target, x, acc); }
                        rec.push_back(acc[0]);
                    }
            for (int i = 0; i < kV; i++) x[i] = vf[i];
            forwardRaw(valueModel.net, x, acc);
            rec.push_back(acc[0]);
            std::fwrite(rec.data(), sizeof(int32_t), rec.size(), stdout);
        }
        prev = now;
        havePrev = true;
    }
    return 0;
}
