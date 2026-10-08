// Feature extractor: reads one match stream (INIT + TURN blocks) on stdin and
// writes one int16 record per (turn, team) to stdout, ordered by turn then
// team. `extract_features --schema` prints the record layout and the feature
// specs as JSON instead; fusion/schema.py reads it, so the layout below is
// defined in exactly one place.
//
//   g++ -std=c++20 -O2 tools/extract_features.cpp -o build/extract_features
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "../inference/features.hpp"
#include "stream.hpp"

using namespace fusion;

namespace {

struct Block { const char* name; std::vector<int> shape; };

const std::vector<Block>& layout() {
    static const std::vector<Block> blocks = {
        {"turn", {1}},           {"team", {1}},
        {"g", {kG}},             {"u", {3, kU}},
        {"m", {3, kMoves, kM}},  {"legal", {3, kMoves}},
        {"act_legal", {3, kMoves, kActs}},
        {"origin", {3}},         {"cand", {3, kCand}},
        {"cand_f", {3, kCand, kTF}},
        {"v", {kV}},
    };
    return blocks;
}

int width() {
    int w = 0;
    for (const auto& b : layout()) {
        int n = 1;
        for (int d : b.shape) n *= d;
        w += n;
    }
    return w;
}

void printSpec(const char* name, const FeatureSpec* spec, int n, bool last) {
    std::printf("  \"%s\": [", name);
    for (int i = 0; i < n; i++)
        std::printf("%s[\"%s\", %d, %d, %d]", i ? ", " : "", spec[i].name, spec[i].lo, spec[i].hi, spec[i].scale);
    std::printf("]%s\n", last ? "" : ",");
}

void printSchema() {
    std::printf("{\n  \"spec_version\": %d,\n  \"record_width\": %d,\n  \"moves\": %d,\n  \"cand\": %d,\n  \"acts\": %d,\n",
                kSpecVersion, width(), kMoves, kCand, kActs);
    std::printf("  \"layout\": [");
    int offset = 0;
    for (size_t i = 0; i < layout().size(); i++) {
        const auto& b = layout()[i];
        int n = 1;
        std::printf("%s[\"%s\", %d, [", i ? ", " : "", b.name, offset);
        for (size_t d = 0; d < b.shape.size(); d++) {
            std::printf("%s%d", d ? ", " : "", b.shape[d]);
            n *= b.shape[d];
        }
        std::printf("]]");
        offset += n;
    }
    std::printf("],\n");
    printSpec("global", kGlobalSpec, kG, false);
    printSpec("unit", kUnitSpec, kU, false);
    printSpec("move", kMoveSpec, kM, false);
    printSpec("cand_f", kCandSpec, kTF, false);
    printSpec("value_unit", kValueUnitSpec, kVU, false);
    printSpec("value_agg", kValueAggSpec, kVA, true);
    std::printf("}\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--schema") == 0) {
        printSchema();
        return 0;
    }
    std::ios::sync_with_stdio(false);
    binaryStdout();  // raw int16 records; --schema above stays text
    Static st;
    if (!readInit(std::cin, st)) return 1;
    Board board;
    board.init(st);
    State prev, now;
    bool havePrev = false;
    std::vector<int16_t> rec(width());
    PolicyFeatures pf;
    TurnView view;
    ValueFeatures vf;
    while (readTurn(std::cin, now)) {
        makeView(board, now, havePrev ? &prev : nullptr, view);
        for (int team = 0; team < st.teams; team++) {
            encodePolicy(board, now, view, team, pf);
            encodeValue(board, now, view, team, vf);
            int16_t* r = rec.data();
            *r++ = static_cast<int16_t>(now.turn);
            *r++ = static_cast<int16_t>(team);
            for (int v : pf.g) *r++ = v;
            for (const auto& row : pf.u) for (int v : row) *r++ = v;
            for (const auto& unit : pf.m) for (const auto& mv : unit) for (int v : mv) *r++ = v;
            for (const auto& row : pf.legal) for (int v : row) *r++ = v;
            for (const auto& unit : pf.actLegal) for (const auto& mv : unit) for (int v : mv) *r++ = v;
            for (int k = 0; k < 3; k++) *r++ = static_cast<int16_t>(view.origin[3 * team + k]);
            for (const auto& row : pf.cand) for (int v : row) *r++ = static_cast<int16_t>(v);
            for (const auto& unit : pf.candF) for (const auto& c : unit) for (int v : c) *r++ = v;
            for (int v : vf) *r++ = v;
            std::fwrite(rec.data(), sizeof(int16_t), rec.size(), stdout);
        }
        prev = now;
        havePrev = true;
    }
    return 0;
}
