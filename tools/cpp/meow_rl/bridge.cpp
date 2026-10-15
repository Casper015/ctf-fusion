// Native bridge for the Meowmind RL trainer (task T0054, docs/MEOWMIND_RL_IMPLEMENTATION.md).
//
// Compiles the unmodified Meowmind reference (bots/meowmind_p/meowmind-ctf-374-20261011.cpp, sha256 cc7b730c...) as a
// shared library with a C API, so the trainer uses the reference's own code for everything except the learned actor:
//   * per-episode state: geometry, movement history (Planner::previous), the opponent experts' online belief
//     (Planner::adapt / predict, frozen weights) and the new-life flags the resolver reads;
//   * the 5 x 112 per-unit move features (learning::Geometry::policyMoves), the 416 public value features
//     (learning::Geometry::value) and the move feasibility the reference applies (wall bumps, dead units stay,
//     no two own units on one destination);
//   * the frozen attack/pickup resolver (Planner::shots with the reference's defaults) and the reply format.
// The reference's own actor probabilities are exported only for the warm-start parity test.
//
// One session is one episode of one seat. Sessions read their block through std::cin, which is process-global:
// use one session at a time per process (rollout workers are processes). Every entry point catches exceptions.

#define main meowmind_reference_main
#include "../../../bots/meowmind_p/meowmind-ctf-374-20261011.cpp"
#undef main

#include <cstring>
#include <exception>
#include <string>

#if defined(_WIN32)
#define MR_API extern "C" __declspec(dllexport)
#else
#define MR_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int kUnits = 3, kMoves = 5, kJoint = 125, kPF = learning::PF, kVF = learning::VF;
static_assert(kPF == 112 && kVF == 416, "the bridge serves the reference's 112/416 feature schema");

struct Session {
    Game g;                              // the raw blocks as read (INIT data persists across turns)
    Planner planner;                     // the reference's per-episode state
    Game observation;                    // learning::prepared(g): predicted respawns
    std::vector<Planner::Prob> prob;     // predict(): reference own policy, opponents' belief mixture
    int observed_turn = -1;              // the turn observed and not yet answered
    int answered_turn = -1;
    std::string reply;
    std::string error;
};

int fail(Session* s, const std::string& what) {
    if (s) s->error = what;
    return -1;
}

// Reads one INIT/TURN block through the reference parser.
bool read_block(Game& g, const char* text) {
    std::istringstream in(text);
    auto* saved = std::cin.rdbuf(in.rdbuf());
    bool ok = false;
    try {
        ok = read_turn(g);
    } catch (...) {
        std::cin.rdbuf(saved);
        throw;
    }
    std::cin.rdbuf(saved);
    std::cin.clear();
    return ok;
}

}  // namespace

MR_API int mr_abi_version() { return 1; }
MR_API int mr_policy_features() { return kPF; }
MR_API int mr_value_features() { return kVF; }

MR_API void* mr_create() {
    try {
        return new Session();
    } catch (...) {
        return nullptr;
    }
}

MR_API void mr_destroy(void* handle) { delete static_cast<Session*>(handle); }

MR_API const char* mr_error(void* handle) {
    auto* s = static_cast<Session*>(handle);
    return s ? s->error.c_str() : "null session";
}

// Observes one block (the first must carry INIT). Writes, for the three own units k (id = 3 * me + k):
//   feats      [3][5][112] float   the reference's per-move features (policyMoves with the unit's previous cell)
//   unit_legal [3][5] uint8        1 when the move is feasible for that unit
//   joint_mask [125] uint8         joint (a, b, c) at a*25 + b*5 + c: every move feasible, no shared destination
//   value_feats[416] float         the public value features of the own team
//   ref_prob   [3][5] float        the reference actor's own move probabilities (warm-start parity only)
//   info       [8] int             turn, turns, teams, me, size, alive mask (bit k), valid joints, units
// Returns 0, or -1 with mr_error set. Each observed turn must be answered (mr_resolve) before the next block.
MR_API int mr_observe(void* handle, const char* text, float* feats, unsigned char* unit_legal,
                      unsigned char* joint_mask, float* value_feats, float* ref_prob, int* info) {
    auto* s = static_cast<Session*>(handle);
    if (!s) return -1;
    try {
        if (s->observed_turn >= 0 && s->answered_turn != s->observed_turn)
            return fail(s, "turn " + std::to_string(s->observed_turn) + " was observed and not answered");
        if (!read_block(s->g, text)) return fail(s, "block without END");
        Game& g = s->g;
        if (g.teams < 2 || g.size <= 0 || g.map.empty()) return fail(s, "no INIT block before the first TURN");
        if (int(g.units.size()) != 3 * g.teams) return fail(s, "expected 3 units per team");
        if (g.turn <= s->observed_turn) return fail(s, "turns must increase");
        Planner& p = s->planner;
        // Planner::run up to the search, unchanged.
        if (!p.geo.n) {
            p.geo.init(g);
            p.previous.assign(g.units.size(), -1);
        }
        p.adapt(g);
        s->observation = learning::prepared(g);
        p.preparedTurn = g.turn;
        p.newLife.assign(g.units.size(), false);
        for (auto& u : s->observation.units) p.newLife[u.id] = g.units[u.id].x < 0 && u.x >= 0;
        s->prob = p.predict(s->observation, true);

        const Game& o = s->observation;
        int alive = 0, dest[kUnits][kMoves];
        for (int k = 0; k < kUnits; k++) {
            int id = 3 * g.me + k, cell = p.geo.cell(o.units[id]);
            auto all = p.geo.policyMoves(o, id, p.previous[id]);
            for (int m = 0; m < kMoves; m++) {
                std::memcpy(feats + (k * kMoves + m) * kPF, all[m].data(), sizeof(float) * kPF);
                bool legal = cell >= 0 ? (m == 0 || all[m][7] <= .5f) : m == 0;   // Planner::policy's -100 rule
                unit_legal[k * kMoves + m] = legal;
                dest[k][m] = cell < 0 ? -1 : p.geo.nb[cell][m];
                ref_prob[k * kMoves + m] = s->prob[id][m];
            }
            if (cell >= 0) alive |= 1 << k;
        }
        int valid = 0;
        for (int a = 0; a < kMoves; a++)
            for (int b = 0; b < kMoves; b++)
                for (int c = 0; c < kMoves; c++) {
                    int mv[3] = {a, b, c};
                    bool ok = true;
                    for (int k = 0; k < kUnits && ok; k++) ok = unit_legal[k * kMoves + mv[k]];
                    for (int k = 0; k < kUnits && ok; k++)
                        for (int l = k + 1; l < kUnits && ok; l++)
                            if (dest[k][mv[k]] >= 0 && dest[k][mv[k]] == dest[l][mv[l]]) ok = false;   // Planner::moves
                    joint_mask[a * 25 + b * 5 + c] = ok;
                    valid += ok;
                }
        if (!joint_mask[0]) return fail(s, "the all-stay joint move is infeasible");   // the safe fallback
        auto v = p.geo.value(o, g.me);
        std::memcpy(value_feats, v.data(), sizeof(float) * kVF);
        int fields[8] = {g.turn, g.turns, g.teams, g.me, g.size, alive, valid, int(g.units.size())};
        std::memcpy(info, fields, sizeof fields);
        s->observed_turn = g.turn;
        s->error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(s, std::string("exception: ") + e.what());
    } catch (...) {
        return fail(s, "unknown exception");
    }
}

// Answers the observed turn with joint move `joint` (a*25 + b*5 + c, which must be feasible): the frozen resolver
// assigns attacks and pickups for exactly these moves (Planner::rollout's shots call) and the reply is formatted as
// Planner::run does. The movement history then advances as in Planner::run. Writes the reply (no newline) into
// `out` (capacity `cap`) and the three resolved actions (target id, -1 pickup or wait, -3 drop) into `actions`.
MR_API int mr_resolve(void* handle, int joint, char* out, int cap, int* actions) {
    auto* s = static_cast<Session*>(handle);
    if (!s) return -1;
    try {
        if (s->observed_turn < 0 || s->answered_turn == s->observed_turn) return fail(s, "no observed turn to answer");
        if (joint < 0 || joint >= kJoint) return fail(s, "joint index out of range");
        const Game& g = s->g;
        const Game& o = s->observation;
        Planner& p = s->planner;
        Planner::Candidate ca;
        ca.moves = {joint / 25, joint / 5 % 5, joint % 5};
        // Re-check feasibility here: the trainer must never execute an action outside the mask it sampled from.
        int dest[kUnits];
        for (int k = 0; k < kUnits; k++) {
            int id = 3 * g.me + k, cell = p.geo.cell(o.units[id]), m = ca.moves[k];
            if (cell < 0 && m != 0) return fail(s, "a dead unit must stay");
            if (cell >= 0 && m != 0 && p.geo.nb[cell][m] == cell) return fail(s, "move into a wall");
            dest[k] = cell < 0 ? -1 : p.geo.nb[cell][m];
        }
        for (int k = 0; k < kUnits; k++)
            for (int l = k + 1; l < kUnits; l++)
                if (dest[k] >= 0 && dest[k] == dest[l]) return fail(s, "two units share a destination");
        ca.actions = p.shots(o, g.me, ca.moves, s->prob);
        std::string reply = std::to_string(g.turn);
        for (int k = 0; k < kUnits; k++) {
            int id = g.me * 3 + k;
            reply += ' ';
            reply += std::string("SUDLR")[ca.moves[k]];
            reply += ' ';
            reply += ca.actions[k] == -3 ? "X" : g.units[id].flag >= 0 ? "-" : ca.actions[k] >= 0 ? std::to_string(ca.actions[k]) : "P";
            actions[k] = ca.actions[k];
        }
        for (auto& u : g.units) p.previous[u.id] = p.geo.cell(u);
        if (int(reply.size()) + 1 > cap) return fail(s, "reply buffer too small");
        std::memcpy(out, reply.c_str(), reply.size() + 1);
        s->reply = reply;
        s->answered_turn = s->observed_turn;
        s->error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(s, std::string("exception: ") + e.what());
    } catch (...) {
        return fail(s, "unknown exception");
    }
}

// Test only: the reference's scalar per-move features (learning::Geometry::policy) of the observed turn, for the
// parity test against the batched policyMoves the reference uses (NN_BATCH_FEATURES) and mr_observe returns.
MR_API int mr_scalar_features(void* handle, float* feats) {
    auto* s = static_cast<Session*>(handle);
    if (!s) return -1;
    try {
        if (s->observed_turn < 0) return fail(s, "nothing observed");
        const Game& g = s->g;
        Planner& p = s->planner;
        for (int k = 0; k < kUnits; k++) {
            int id = 3 * g.me + k;
            // previous[] still holds the last answered turn's cells until mr_resolve.
            int old = s->answered_turn == s->observed_turn ? -2 : p.previous[id];
            if (old == -2) return fail(s, "call before mr_resolve");
            for (int m = 0; m < kMoves; m++) {
                auto f = p.geo.policy(s->observation, id, m, old);
                std::memcpy(feats + (k * kMoves + m) * kPF, f.data(), sizeof(float) * kPF);
            }
        }
        return 0;
    } catch (...) {
        return fail(s, "exception");
    }
}

// Copies one of the reference's weight arrays (policy_w0 ... value_b2, opponent_*) into `out`. Returns its length,
// or -1 for an unknown name; with out == nullptr or cap too small only the length is returned.
MR_API int mr_weights(const char* name, float* out, int cap) {
    struct Entry { const char* name; const float* data; int size; };
#define MR_W(x) {#x, weights::x, int(std::size(weights::x))}
    static const Entry table[] = {MR_W(policy_w0), MR_W(policy_b0), MR_W(policy_w1), MR_W(policy_b1), MR_W(policy_w2),
                                  MR_W(policy_b2), MR_W(value_w0),  MR_W(value_b0),  MR_W(value_w1),  MR_W(value_b1),
                                  MR_W(value_w2),  MR_W(value_b2)};
#undef MR_W
    for (auto& e : table)
        if (std::strcmp(e.name, name) == 0) {
            if (out && cap >= e.size) std::memcpy(out, e.data, sizeof(float) * e.size);
            return e.size;
        }
    return -1;
}
