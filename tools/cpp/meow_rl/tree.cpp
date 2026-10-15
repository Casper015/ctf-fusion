// Fixed-work search trees for the Meowmind RL search diagnosis (task T0061, docs/MEOWMIND_RL_ALIGNED.md).
//
//     c++ -std=c++20 -O2 -I inference -DACTOR_HEADER='"inference/meow_rl/<name>/actor.hpp"' tree.cpp -o tree
//     tree <stream.txt> <turn> [<turn> ...] > roots.jsonl 3> leaves.f32      (leaves: fd 3, or MEOW_RL_TREE_LEAVES path)
//
// Replays one seat stream (rebuilt from a match record: the seat's INIT and the shared TURN blocks, exactly what the
// runner sent that seat) through meow_rl::Session exactly as the deployed
// bot reads it, and at each requested turn builds the search_v2.hpp tree of that turn with no deadline: the roots
// (the sampled actor choice of sample.hpp, then the next joints by accumulator), the shared scenarios and every
// trajectory, through the same functions the deployed MEOW_RL_SEARCH=2 bot calls. Each turn writes one JSON line
// (roots, actor choice, feasible joints, leaves in root-major order with terminal flags, rewards, continuation joints
// and the offset of their 416 float32 value features in the leaves file) and a hash of the whole tree. Critics are
// not evaluated here: bots/meow_rl/diagnose.py scores the same leaves with every backend.
//
// The history of a replay does not depend on the joints answered (Session::resolve's history is the observed
// positions), and the search's random streams depend only on the INIT and the turn, so a tree here is the tree the
// deployed bot would have searched at that turn.
#define main meowmind_reference_main
#include "../../../bots/meowmind_p/meowmind-ctf-374-20261011.cpp"
#undef main

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>

#include "session.hpp"
#include ACTOR_HEADER
#include "actor_int.hpp"
#include "../../../bots/meow_rl/sample.hpp"
#define MEOW_RL_TREE_ONLY 1
#include "../../../bots/meow_rl/search_v2.hpp"

namespace {

uint64_t fnv(uint64_t h, const void* data, size_t n) {
    auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: tree <stream.txt> <turn> [<turn> ...]\n");
        return 2;
    }
    std::set<int> wanted;
    for (int i = 2; i < argc; i++) wanted.insert(std::atoi(argv[i]));
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    const char* leaves_path = std::getenv("MEOW_RL_TREE_LEAVES");
    FILE* leaves = leaves_path ? std::fopen(leaves_path, "wb") : fdopen(3, "wb");
    if (!leaves) {
        std::fprintf(stderr, "no leaves output (fd 3 or MEOW_RL_TREE_LEAVES)\n");
        return 2;
    }
    std::cin.rdbuf(in.rdbuf());
    meow_rl::Session s;
    meow_rl::Observation obs;
    uint64_t seed = 0;
    long offset = 0, turns = 0;
    while (read_turn(s.g)) {
        if (turns++ == 0) seed = meow_rl_sample::fnv1a(meow_rl_sample::canonical_init(s.g));
        std::string reply, why = s.observe(obs);
        int joint = 0;
        if (why.empty()) {
            int64_t sc[3][5];
            for (int k = 0; k < 3; k++)
                for (int m = 0; m < 5; m++) sc[k][m] = meow_rl_int::score(obs.feats[k][m]);
            joint = meow_rl_sample::draw(sc, obs.joint_mask, meow_rl_model::kActor.outExp, meow_rl_model::kTemperature,
                                         meow_rl_sample::uniform(seed, s.g.turn));
            if (wanted.count(s.g.turn)) {
                uint64_t T = meow_rl_search2::turn_seed(seed, s.g.turn);
                auto rs = meow_rl_search2::roots(obs, sc, joint);
                auto opp = meow_rl_search2::scenarios(s, T);
                uint64_t h = fnv(1469598103934665603ull, &T, sizeof T);
                std::string js = "{\"turn\":" + std::to_string(s.g.turn) + ",\"turns\":" + std::to_string(s.g.turns) +
                                 ",\"teams\":" + std::to_string(s.g.teams) + ",\"me\":" + std::to_string(s.g.me) +
                                 ",\"valid\":" + std::to_string(obs.valid) + ",\"actor\":" + std::to_string(joint) +
                                 ",\"samples\":" + std::to_string(meow_rl_search2::kSamples) + ",\"roots\":[";
                for (size_t i = 0; i < rs.size(); i++) js += (i ? "," : "") + std::to_string(rs[i]);
                js += "],\"leaves\":[";
                meow_rl_search2::Leaf leaf;
                bool first = true;
                for (size_t ri = 0; ri < rs.size(); ri++)
                    for (int smp = 0; smp < meow_rl_search2::kSamples; smp++) {
                        meow_rl_search2::trajectory(s, rs[ri], opp[smp], T, smp, leaf);
                        long at = -1;
                        if (!leaf.terminal) {
                            std::fwrite(leaf.feats.data(), sizeof(float), leaf.feats.size(), leaves);
                            at = offset++;
                            h = fnv(h, leaf.feats.data(), sizeof(float) * leaf.feats.size());
                        }
                        h = fnv(h, &leaf.reward, sizeof leaf.reward);
                        h = fnv(h, &leaf.cont, sizeof leaf.cont);
                        char buf[200];
                        std::snprintf(buf, sizeof buf, "%s{\"root\":%zu,\"sample\":%d,\"terminal\":%s,\"plies\":%d,\"reward\":%.17g,"
                                      "\"cont\":%d,\"row\":%ld}", first ? "" : ",", ri, smp, leaf.terminal ? "true" : "false",
                                      leaf.plies, leaf.reward, leaf.cont, at);
                        js += buf;
                        first = false;
                    }
                char tail[64];
                std::snprintf(tail, sizeof tail, "],\"hash\":\"%016" PRIx64 "\"}", h);
                std::printf("%s%s\n", js.c_str(), tail);
            }
        }
        if (!why.empty()) {
            s.forget_history();
            continue;
        }
        int actions[meow_rl::kUnits];
        if (auto w = s.resolve(joint, reply, actions); !w.empty()) {
            std::fprintf(stderr, "turn %d: %s\n", s.g.turn, w.c_str());
            return 1;
        }
    }
    std::fclose(leaves);
    std::fprintf(stderr, "turns %ld leaves %ld\n", turns, offset);
    return 0;
}
