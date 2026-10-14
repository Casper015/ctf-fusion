// Own-action feature check (inference/own_action.hpp against fusion/decision_features.py; T0043).
//
//   own_action_check --identity                         version, pinned digest and dimensions as JSON
//   own_action_check <seat> <turn> <candidates.txt> < prefix-stream
//
// The prefix stream is INIT plus TURN 1..turn of the seat's input (training.decision_dataset.prefix_stream); blocks
// after the root turn are not read. Every line of candidates.txt is one requested reply (the first is the
// incumbent). Prints JSON: the root features, and per candidate its canonical order, flags, duplicate pointer,
// support and candidate features. Floats are printed with 9 significant digits, which round-trips float32.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../inference/own_action.hpp"
#include "stream.hpp"

using namespace fusion;
namespace oa = fusion::own_action;

namespace {

void floats(const float* x, int n) {
    std::printf("[");
    for (int i = 0; i < n; i++) std::printf("%s%.9g", i ? ", " : "", static_cast<double>(x[i]));
    std::printf("]");
}

void ints(const std::array<int, 3>& x) { std::printf("[%d, %d, %d]", x[0], x[1], x[2]); }

std::string quoted(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--identity") == 0) {
        std::printf("{\"version\": \"%s\", \"digest\": \"%s\", \"layout_version\": %d, \"per_unit\": %d, \"cand_dim\": %d, "
                    "\"root_dim\": %d, \"slots\": %d, \"out_of_cap_range\": %d}\n",
                    oa::kVersion, oa::kIdentityDigest, oa::kLayoutVersion, oa::kPerUnit, oa::kCandDim, oa::kRootDim, kCand,
                    oa::kOutOfCapRange);
        return 0;
    }
    if (argc != 4) {
        std::fprintf(stderr, "usage: own_action_check <seat> <turn> <candidates.txt> < prefix-stream\n");
        return 2;
    }
    const int seat = std::atoi(argv[1]), turn = std::atoi(argv[2]);
    std::vector<std::string> lines;
    {
        std::ifstream in(argv[3]);
        for (std::string line; std::getline(in, line);) lines.push_back(line);
    }
    Static st;
    if (!readInit(std::cin, st)) return 1;
    Board board;
    board.init(st);
    const Rules rules = rulesOf(st);
    State prev, now;
    bool havePrev = false, found = false;
    TurnView view;
    CombatSummary cs;
    oa::Root root;
    while (readTurn(std::cin, now)) {
        if (now.turn == turn) {
            makeView(board, now, havePrev ? &prev : nullptr, view);
            summarize(board, rules, now, view, cs);
            oa::encodeRoot(board, now, view, cs, seat, root);
            found = true;
            break;  // nothing after the root is read
        }
        prev = now;
        havePrev = true;
    }
    if (!found) {
        std::fprintf(stderr, "turn %d not in the stream\n", turn);
        return 1;
    }
    std::vector<oa::Canon> cands;
    for (const auto& line : lines) cands.push_back(oa::canonicalizeLine(root, line));
    const std::vector<int> dup = oa::markDuplicates(cands, 0);
    std::vector<float> x(oa::kRootDim > oa::kCandDim ? oa::kRootDim : oa::kCandDim);
    oa::rootFeatures(root, x.data());
    std::printf("{\"turn\": %d, \"team\": %d, \"root\": ", root.turn, root.team);
    floats(x.data(), oa::kRootDim);
    std::printf(", \"cands\": [");
    for (size_t i = 0; i < cands.size(); i++) {
        const auto& c = cands[i];
        std::printf("%s{\"key\": %s, \"flags\": %d, \"dup\": %d, \"support\": %s, \"move\": ", i ? ", " : "",
                    quoted(c.key).c_str(), c.flags, dup[i], c.supported() ? "true" : "false");
        ints(c.move);
        std::printf(", \"act\": ");
        ints(c.act);
        std::printf(", \"target\": ");
        ints(c.target);
        std::printf(", \"slot\": ");
        ints(c.slot);
        std::printf(", \"x\": ");
        oa::candidateFeatures(root, c, x.data());
        floats(x.data(), oa::kCandDim);
        std::printf("}");
    }
    std::printf("]}\n");
    return 0;
}
