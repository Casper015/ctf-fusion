// Fixed-work search trees for the Meowmind RL search diagnosis (task T0061, docs/MEOWMIND_RL_ALIGNED.md).
//
//     c++ -std=c++20 -O2 -I inference -DACTOR_HEADER='"inference/meow_rl/<name>/actor.hpp"' tree.cpp -o tree
//     tree <stream.txt> <turn> [<turn> ...] > roots.jsonl 3> leaves.f32      (leaves: fd 3, or MEOW_RL_TREE_LEAVES path)
//
// Replays one seat stream (rebuilt from a match record: the seat's INIT and the shared TURN blocks, exactly what the
// runner sent that seat) through meow_rl::Session exactly as the deployed
// bot reads it, and at each requested turn builds the search_v2.hpp tree of that turn with no deadline: the roots
// (the sampled actor choice of sample.hpp, then the next joints by accumulator), the shared scenarios and every
// trajectory, through the same functions the deployed MEOW_RL_SEARCH=2 bot calls. Each turn writes one JSON line:
// the feasible-joint mask, the actor's uniform and choice, the roots, the turn seed and every scenario's streams
// (ply-1 and ply-2 opponent RNG states, both simulator seeds, the continuation uniform's 64 bits) with the hash of
// its ply-1 opponent orders, and the leaves in root-major order (terminal flag, plies, reward, the resolver's own
// actions at the root and at the continuation, the continuation joint, the ply-2 opponent orders' hash and the
// offset of the 416 float32 value features in the leaves file). "hash" is FNV-1a 64 over every one of those fields
// and the features, in that order. Critics are not evaluated here: bots/meow_rl/aligned/diagnose.py scores the same
// leaves with every backend.
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
                namespace S2 = meow_rl_search2;
                uint64_t T = S2::turn_seed(seed, s.g.turn);
                auto rs = S2::roots(obs, sc, joint);
                auto opp = S2::scenarios(s, T);
                double u = meow_rl_sample::uniform(seed, s.g.turn);
                std::string mask(meow_rl::kJoint, '0');
                for (int j = 0; j < meow_rl::kJoint; j++) mask[j] = obs.joint_mask[j] ? '1' : '0';
                auto exact = [](double v) { char b[40]; std::snprintf(b, sizeof b, "%.17g", v); return std::string(b); };
                auto hex = [](uint64_t v) { char b[20]; std::snprintf(b, sizeof b, "\"%016" PRIx64 "\"", v); return std::string(b); };
                uint64_t h = fnv(1469598103934665603ull, &T, sizeof T);
                h = fnv(h, mask.data(), mask.size());
                h = fnv(h, &u, sizeof u);
                h = fnv(h, &joint, sizeof joint);
                for (int r : rs) h = fnv(h, &r, sizeof r);
                std::string js = "{\"turn\":" + std::to_string(s.g.turn) + ",\"turns\":" + std::to_string(s.g.turns) +
                                 ",\"teams\":" + std::to_string(s.g.teams) + ",\"me\":" + std::to_string(s.g.me) +
                                 ",\"valid\":" + std::to_string(obs.valid) + ",\"mask\":\"" + mask + "\"" +
                                 ",\"actor_uniform\":" + exact(u) + ",\"actor\":" + std::to_string(joint) +
                                 ",\"samples\":" + std::to_string(S2::kSamples) + ",\"turn_seed\":" + hex(T) + ",\"roots\":[";
                for (size_t i = 0; i < rs.size(); i++) js += (i ? "," : "") + std::to_string(rs[i]);
                js += "],\"scenarios\":[";
                for (int smp = 0; smp < S2::kSamples; smp++) {
                    uint64_t st[5] = {S2::stream(T, S2::kOpp1, smp), S2::stream(T, S2::kSim1, smp), S2::stream(T, S2::kOpp2, smp),
                                      S2::stream(T, S2::kCont, smp), S2::stream(T, S2::kSim2, smp)};
                    uint64_t oh = S2::order_hash(opp[smp]);
                    h = fnv(h, st, sizeof st);
                    h = fnv(h, &oh, sizeof oh);
                    js += std::string(smp ? "," : "") + "{\"opp1_rng\":" + hex(st[0]) + ",\"sim1\":" + hex(st[1]) +
                          ",\"opp2_rng\":" + hex(st[2]) + ",\"cont\":" + hex(st[3]) + ",\"sim2\":" + hex(st[4]) +
                          ",\"opp1_orders\":" + hex(oh) + "}";
                }
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
                        int flags[5] = {int(ri), smp, int(at), int(leaf.terminal), leaf.plies};   // the leaf's place, then its kind
                        h = fnv(h, flags, sizeof flags);
                        h = fnv(h, &leaf.reward, sizeof leaf.reward);
                        h = fnv(h, &leaf.cont, sizeof leaf.cont);
                        h = fnv(h, leaf.root_actions.data(), sizeof leaf.root_actions);
                        h = fnv(h, leaf.cont_actions.data(), sizeof leaf.cont_actions);
                        h = fnv(h, &leaf.opp2_hash, sizeof leaf.opp2_hash);
                        char buf[400];
                        std::snprintf(buf, sizeof buf, "%s{\"root\":%zu,\"sample\":%d,\"terminal\":%s,\"plies\":%d,\"reward\":%.17g,"
                                      "\"root_actions\":[%d,%d,%d],\"cont\":%d,\"cont_actions\":[%d,%d,%d],\"opp2_orders\":\"%016" PRIx64 "\","
                                      "\"row\":%ld}", first ? "" : ",", ri, smp, leaf.terminal ? "true" : "false", leaf.plies, leaf.reward,
                                      leaf.root_actions[0], leaf.root_actions[1], leaf.root_actions[2], leaf.cont,
                                      leaf.cont_actions[0], leaf.cont_actions[1], leaf.cont_actions[2], leaf.opp2_hash, at);
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
