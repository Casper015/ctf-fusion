// Root and P extraction through the real agent_v2 runtime (task T0085, tools/search_diagnostics/README.md).
//
// Built per package by tools/search_diagnostics/build.py: a generated wrapper defines the package's FUSION_MAX_WIDTH,
// FUSION_TIER_* and the tree's AGENT_V2_SEARCH_CANDIDATES / AGENT_V2_SEARCH_SAMPLES, includes its actor header
// (namespace meow_rl_model) and critic header (namespace meow_rl_critic), then this file; the bundle is the identity. The direct bot's code path is replayed exactly: observe the
// block, refuse it (all-stay) or draw the actor's joint with uniform(seed, turn), resolve the complete default order
// with assign_v1, commit. Nothing here has a deadline.
//
//   diag_tool root <stream> <turn>   one JSON line: the replies of turns < turn, then at `turn` the actor's scores,
//                                    feasible mask, sampled joint, complete default order, and the search tree's roots
//                                    (the sampled joint first, then the feasible joints by prior; inference/agent_v2/
//                                    search.hpp roots()) with each root's complete order and per-scenario values
//                                    (terminal reward, or the critic of the two-ply leaf; same streams as the bot's tree)
//   diag_tool pmarg <stream>         one JSON line per answered turn: for every predicted opponent unit P's movement
//                                    marginals, the feasible moves, and per move P(wait, pickup, drop, attack) and
//                                    P(target slot), with the slots' unit ids
//   diag_tool gate <stream> <tau>    one JSON line per answered turn: whether the frozen search gate (inference/agent_v2/
//                                    search.hpp gated_choice over the first 8 roots x the tree's samples, fixed work, no
//                                    deadline) would replace the actor's joint, and its pick. Root selection's "override"
//                                    stratum reads this; it is the gate's offline decision, not what a timed bot did
// Exit 0 on success; 1 with "error ..." on stdout when the stream ends before `turn` or the root turn is refused.
#define AGENT_V2_TREE_ONLY 1
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../../../inference/agent_v2/runtime.hpp"
#include "../../../inference/agent_v2/search.hpp"

namespace {

using RT = agent_v2::Runtime<agent_v2::NoSearch>;

std::vector<std::string> blocks(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::string> out;
    std::string cur, line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() && cur.empty()) continue;
        cur += line + "\n";
        if (line.rfind("END", 0) == 0) out.push_back(cur), cur.clear();
    }
    return out;
}

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
    return b;
}

std::string num(double v) {
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

std::string quoted(const std::string& s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') r += '\\';
        r += c;
    }
    return r + "\"";
}

uint64_t fnv(uint64_t h, const void* data, size_t n) {
    auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

std::string identity(const RT& rt) {
    return std::string("\"carrier\":") + quoted(agent_v2::kCarrier) + ",\"actor\":" + quoted(meow_rl_model::kName) +
           ",\"actor_digest\":" + quoted(meow_rl_model::kDigest) + ",\"predictor_kind\":" + quoted(agent_v2::PredictorP::kKind) +
           ",\"predictor\":" + quoted(rt.pred.model_id()) + ",\"critic\":" + quoted(meow_rl_critic::kFormat) +
           ",\"resolver\":" + quoted(std::string(agent_v2::kResolverRule) + "/" + rt.resolver.profile.name) +
           ",\"resolver_scenarios\":" + std::to_string(agent_v2::kResolverScenarios) +
           ",\"tree_candidates\":" + std::to_string(agent_v2::kCandidates) + ",\"tree_samples\":" + std::to_string(agent_v2::kSamples);
}

// The direct bot's answer to an observed turn (bots/agent_v2/bot.hpp through Runtime::decide with NoSearch).
std::string answer(RT& rt, int64_t (&sc)[3][5], int& joint) {
    agent_v2::Actor::scores(rt.obs, sc);
    joint = agent_v2::Actor::sample(sc, rt.obs, agent_v2::uniform(rt.seed, rt.s.turn));
    rt.tactics = rt.resolve(joint);
    return rt.commit();
}

int root(const char* path, int turn) {
    RT rt;
    std::string replies = "[", refused = "[";
    int64_t sc[3][5];
    int joint = -1, answered = 0;
    for (auto& b : blocks(path)) {
        const std::string why = rt.observe_block(b);
        const bool at_root = why.empty() && rt.s.turn == turn;
        if (!why.empty() && rt.last_turn_seen == turn) {
            std::printf("error the root turn %d is refused: %s\n", turn, why.c_str());
            return 1;
        }
        if (!why.empty()) {
            const std::string r = rt.refuse();
            refused += std::string(refused.size() > 1 ? "," : "") + std::to_string(rt.last_turn_seen);
            replies += std::string(answered++ ? "," : "") + quoted(r);
            continue;
        }
        if (!at_root) {
            replies += std::string(answered++ ? "," : "") + quoted(answer(rt, sc, joint));
            continue;
        }
        agent_v2::Actor::scores(rt.obs, sc);
        joint = agent_v2::Actor::sample(sc, rt.obs, agent_v2::uniform(rt.seed, rt.s.turn));
        const double u = agent_v2::uniform(rt.seed, rt.s.turn);
        const auto def = rt.resolve(joint);
        std::string mask;
        for (int j = 0; j < agent_v2::kJoint; j++) mask += rt.obs.joint_mask[j] ? '1' : '0';
        std::string scores = "[";
        for (int k = 0; k < 3; k++) {
            scores += k ? ",[" : "[";
            for (int m = 0; m < 5; m++) scores += (m ? "," : "") + std::to_string(sc[k][m]);
            scores += "]";
        }
        scores += "]";
        uint64_t feat_hash = 1469598103934665603ull;
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < 5; m++) feat_hash = fnv(feat_hash, rt.obs.feats[k][m], sizeof rt.obs.feats[k][m]);
        agent_v2::Tree tree;
        agent_v2::setup(rt, sc, joint, tree);
        agent_v2::Simulator sim;
        agent_v2::Leaf leaf;
        std::string cands = "[";
        for (int ri = 0; ri < int(tree.roots.size()); ri++) {
            const int j = tree.roots[ri];
            std::string values = "[", terminal = "[", leaves = "[";
            for (int smp = 0; smp < agent_v2::kSamples; smp++) {
                agent_v2::trajectory(rt, tree, ri, smp, sim, leaf);
                double v = leaf.reward;
                uint64_t lh = 0;
                if (!leaf.terminal) {
                    v = meow_rl_critic::value(leaf.feats.data());
                    lh = fnv(1469598103934665603ull, leaf.feats.data(), sizeof(float) * leaf.feats.size());
                }
                values += (smp ? "," : "") + num(v);
                terminal += std::string(smp ? "," : "") + (leaf.terminal ? "1" : "0");
                leaves += std::string(smp ? "," : "") + quoted(hex(lh));
            }
            const auto z = sc[0][j / 25] + sc[1][j / 5 % 5] + sc[2][j % 5];
            cands += std::string(ri ? "," : "") + "{\"joint\":" + std::to_string(j) + ",\"prior\":" + std::to_string(z) +
                     ",\"reply\":" + quoted(agent_v2::reply_text(rt.s.turn, tree.root_orders[ri])) + ",\"values\":" + values +
                     "],\"terminal\":" + terminal + "],\"leaf_feature_fnv\":" + leaves + "]}";
        }
        cands += "]";
        std::printf("{\"schema\":\"ctf-diag-native-root-1\",%s,\"turn\":%d,\"teams\":%d,\"me\":%d,\"seed\":\"%s\",\"uniform\":%s,"
                    "\"state_key\":\"%s\",\"parent_key\":\"%s\",\"policy_feature_fnv\":\"%s\",\"mask\":\"%s\",\"scores\":%s,"
                    "\"sampled_joint\":%d,\"default_reply\":%s,\"tree_T\":\"%s\",\"replies\":%s],\"refused_turns\":%s],"
                    "\"candidates\":%s}\n",
                    identity(rt).c_str(), rt.s.turn, rt.cache.st.teams, rt.cache.st.me, hex(rt.seed).c_str(), num(u).c_str(),
                    hex(agent_v2::state_key(rt.s)).c_str(), hex(rt.hist.parent() ? agent_v2::state_key(*rt.hist.parent()) : 0).c_str(),
                    hex(feat_hash).c_str(), mask.c_str(), scores.c_str(), joint,
                    quoted(agent_v2::reply_text(rt.s.turn, def.orders)).c_str(), hex(tree.T).c_str(), replies.c_str(),
                    refused.c_str(), cands.c_str());
        return 0;
    }
    std::printf("error the stream ends before turn %d\n", turn);
    return 1;
}

int pmarg(const char* path) {
    RT rt;
    int64_t sc[3][5];
    int joint = -1;
    bool first = true;
    for (auto& b : blocks(path)) {
        const std::string why = rt.observe_block(b);
        if (!why.empty()) {
            rt.refuse();
            std::printf("{\"turn\":%d,\"refused\":%s}\n", rt.last_turn_seen, quoted(why).c_str());
            continue;
        }
        auto& v = rt.pred.root.view;
        std::string units = "[";
        int n = 0;
        for (int t = 0; t < rt.cache.st.teams; t++) {
            if (t == rt.cache.st.me || !v.has(t)) continue;
            for (int k = 0; k < 3; k++) {
                const int uid = 3 * t + k;
                if (v.origin(uid) < 0) continue;
                std::string mv = "[", legal, act_legal = "[", act = "[", tgt = "[", cand = "[";
                for (int m = 0; m < 5; m++) {
                    mv += (m ? "," : "") + num(v.move(uid, m));
                    legal += v.legal(uid, m) ? '1' : '0';
                    act_legal += std::string(m ? ",\"" : "\"");
                    for (int i = 0; i < 4; i++) act_legal += v.actLegal(uid, m, i) ? '1' : '0';
                    act_legal += "\"";
                    act += m ? ",[" : "[";
                    tgt += m ? ",[" : "[";
                    if (v.legal(uid, m)) {
                        const auto& a = v.act(uid, m);
                        const auto& p = v.target(uid, m);
                        for (int i = 0; i < 4; i++) act += (i ? "," : "") + num(a[i]);
                        for (int c = 0; c < fusion::kCand; c++) tgt += (c ? "," : "") + num(p[c]);
                    }
                    act += "]", tgt += "]";
                }
                for (int c = 0; c < fusion::kCand; c++) cand += (c ? "," : "") + std::to_string(v.candidate(uid, c));
                units += std::string(n++ ? "," : "") + "{\"uid\":" + std::to_string(uid) + ",\"move\":" + mv + "],\"legal\":\"" +
                         legal + "\",\"act_legal\":" + act_legal + "],\"act\":" + act + "],\"target\":" + tgt + "],\"cand\":" + cand + "]}";
            }
        }
        std::printf("{%s%s,\"turn\":%d,\"state_key\":\"%s\",\"units\":%s]}\n", first ? "\"schema\":\"ctf-diag-native-p-1\"," : "",
                    identity(rt).c_str(), rt.s.turn, hex(agent_v2::state_key(rt.s)).c_str(), units.c_str());
        first = false;
        answer(rt, sc, joint);
    }
    return 0;
}

int gate(const char* path, double tau) {
    RT rt;
    agent_v2::Simulator sim;
    agent_v2::Leaf leaf;
    int64_t sc[3][5];
    int joint = -1;
    for (auto& b : blocks(path)) {
        if (!rt.observe_block(b).empty()) {
            rt.refuse();
            continue;
        }
        agent_v2::Actor::scores(rt.obs, sc);
        joint = agent_v2::Actor::sample(sc, rt.obs, agent_v2::uniform(rt.seed, rt.s.turn));
        agent_v2::Tree tree;
        agent_v2::setup(rt, sc, joint, tree);
        const int n = std::min(8, int(tree.roots.size()));
        std::vector<double> mean(n, 0.0);
        for (int ri = 0; ri < n; ri++) {
            for (int smp = 0; smp < agent_v2::kSamples; smp++) {
                agent_v2::trajectory(rt, tree, ri, smp, sim, leaf);
                mean[ri] += leaf.terminal ? leaf.reward : meow_rl_critic::value(leaf.feats.data());
            }
            mean[ri] /= agent_v2::kSamples;
        }
        const int pick = agent_v2::gated_choice(mean.data(), n, tau);
        std::printf("{\"turn\":%d,\"override\":%d,\"pick\":%d,\"roots\":%d}\n", rt.s.turn, pick != 0, pick, n);
        rt.tactics = rt.resolve(joint);
        rt.commit();
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "root" && argc == 4) return root(argv[2], std::atoi(argv[3]));
    if (mode == "pmarg" && argc == 3) return pmarg(argv[2]);
    if (mode == "gate" && argc == 4) return gate(argv[2], std::atof(argv[3]));
    std::fprintf(stderr, "usage: diag_tool root <stream> <turn> | diag_tool pmarg <stream> | diag_tool gate <stream> <tau>\n");
    return 2;
}
