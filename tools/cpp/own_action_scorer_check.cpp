// Own-action scorer check (inference/own_action_scorer.hpp with an exported scorer_model.hpp; task T0044).
//
//   own_action_scorer_check --model                     the exported model's identities and validate() as JSON
//   own_action_scorer_check <seat> <turn> <candidates.txt> < prefix-stream
//
// Built with -I<export directory> so that "scorer_model.hpp" is the export under test. The prefix stream is INIT
// plus TURN 1..turn of the seat's input; blocks after the root turn are not read. Every line of candidates.txt is
// one requested reply (the first is the incumbent). Prints JSON: validate(), and per candidate its canonical key,
// support and Q(s, a) with 17 significant digits.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../inference/own_action_scorer.hpp"
#include "scorer_model.hpp"
#include "stream.hpp"

using namespace fusion;
namespace oa = fusion::own_action;

int main(int argc, char** argv) {
    const oa::ScorerModel& m = oa::exported::kScorer;
    const char* why = "ok";
    const bool ok = oa::validate(m, &why);
    if (argc > 1 && std::strcmp(argv[1], "--model") == 0) {
        std::printf("{\"valid\": %s, \"why\": \"%s\", \"kind\": \"%s\", \"model\": \"%s\", \"feature_digest\": \"%s\", "
                    "\"layers\": %d, \"lambda\": %.17g, \"norm\": \"%s\", \"selection\": \"%s\", \"carrier_variant\": \"%s\", "
                    "\"carrier_status\": \"%s\", \"carrier_config\": \"%s\", \"carrier_value\": %d, \"model_sha256\": \"%s\", "
                    "\"config_sha256\": \"%s\"}\n",
                    ok ? "true" : "false", why, m.kind, m.model, m.featureDigest, m.count, m.lambda, m.norm, m.selectionSha256,
                    m.carrierVariant, m.carrierStatus, m.carrierConfigSha256, m.carrierValue, m.modelSha256,
                    m.runConfigSha256);
        return 0;
    }
    if (argc != 4) {
        std::fprintf(stderr, "usage: own_action_scorer_check <seat> <turn> <candidates.txt> < prefix-stream\n");
        return 2;
    }
    if (!ok) {
        std::fprintf(stderr, "invalid scorer: %s\n", why);
        return 3;
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
            break;
        }
        prev = now;
        havePrev = true;
    }
    if (!found) return 1;
    std::vector<oa::Canon> cands;
    for (const auto& line : lines) cands.push_back(oa::canonicalizeLine(root, line));
    oa::markDuplicates(cands, 0);
    std::vector<float> rx(oa::kRootDim), cx(oa::kCandDim);
    oa::rootFeatures(root, rx.data());
    std::printf("{\"q\": [");
    for (size_t i = 0; i < cands.size(); i++) {
        oa::candidateFeatures(root, cands[i], cx.data());
        std::printf("%s%.17g", i ? ", " : "", oa::q(m, rx.data(), cx.data()));
    }
    std::printf("], \"support\": [");
    for (size_t i = 0; i < cands.size(); i++) std::printf("%s%s", i ? ", " : "", cands[i].supported() ? "true" : "false");
    std::printf("]}\n");
    return 0;
}
