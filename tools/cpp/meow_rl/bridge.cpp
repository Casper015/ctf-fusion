// Native bridge for the Meowmind RL trainer (task T0054, docs/MEOWMIND_RL_IMPLEMENTATION.md).
//
// Compiles the unmodified Meowmind reference (bots/meowmind_p/meowmind-ctf-374-20261011.cpp, sha256 cc7b730c...) as a
// shared library with a C API around meow_rl::Session (session.hpp), the same code the deployed bot runs: the
// trainer uses the reference's own code for everything except the learned actor. The reference's own actor
// probabilities are exported only for the warm-start parity test.
//
// One session is one episode of one seat. Sessions read their block through std::cin, which is process-global:
// use one session at a time per process (rollout workers are processes). Every entry point catches exceptions.

#define main meowmind_reference_main
#include "../../../bots/meowmind_p/meowmind-ctf-374-20261011.cpp"
#undef main

#include <cstring>
#include <exception>
#include <string>

#include "session.hpp"

#if defined(_WIN32)
#define MR_API extern "C" __declspec(dllexport)
#else
#define MR_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

using meow_rl::kJoint, meow_rl::kMoves, meow_rl::kPF, meow_rl::kUnits, meow_rl::kVF;

struct Handle {
    meow_rl::Session session;
    meow_rl::Observation obs;
    std::string error;
};

int fail(Handle* h, const std::string& what) {
    if (h) h->error = what;
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
        return new Handle();
    } catch (...) {
        return nullptr;
    }
}

MR_API void mr_destroy(void* handle) { delete static_cast<Handle*>(handle); }

MR_API const char* mr_error(void* handle) {
    auto* h = static_cast<Handle*>(handle);
    return h ? h->error.c_str() : "null session";
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
    auto* h = static_cast<Handle*>(handle);
    if (!h) return -1;
    try {
        auto& s = h->session;
        if (s.observed_turn >= 0 && s.answered_turn != s.observed_turn)
            return fail(h, "turn " + std::to_string(s.observed_turn) + " was observed and not answered");
        if (!read_block(s.g, text)) return fail(h, "block without END");
        auto& o = h->obs;
        if (auto why = s.observe(o); !why.empty()) return fail(h, why);
        std::memcpy(feats, o.feats, sizeof o.feats);
        std::memcpy(unit_legal, o.unit_legal, sizeof o.unit_legal);
        std::memcpy(joint_mask, o.joint_mask, sizeof o.joint_mask);
        std::memcpy(value_feats, o.value_feats, sizeof o.value_feats);
        std::memcpy(ref_prob, o.ref_prob, sizeof o.ref_prob);
        int fields[8] = {o.turn, o.turns, o.teams, o.me, o.size, o.alive, o.valid, o.units};
        std::memcpy(info, fields, sizeof fields);
        h->error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(h, std::string("exception: ") + e.what());
    } catch (...) {
        return fail(h, "unknown exception");
    }
}

// Answers the observed turn with joint move `joint` (Session::resolve). Writes the reply (no newline) into `out`
// (capacity `cap`) and the three resolved actions (target id, -1 pickup or wait, -3 drop) into `actions`.
MR_API int mr_resolve(void* handle, int joint, char* out, int cap, int* actions) {
    auto* h = static_cast<Handle*>(handle);
    if (!h) return -1;
    try {
        std::string reply;
        int acts[kUnits];
        if (auto why = h->session.resolve(joint, reply, acts); !why.empty()) return fail(h, why);
        if (int(reply.size()) + 1 > cap) return fail(h, "reply buffer too small");
        std::memcpy(out, reply.c_str(), reply.size() + 1);
        std::memcpy(actions, acts, sizeof acts);
        h->error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(h, std::string("exception: ") + e.what());
    } catch (...) {
        return fail(h, "unknown exception");
    }
}

// After a refused block: Session::forget_history, as the deployed bot does.
MR_API int mr_forget(void* handle) {
    auto* h = static_cast<Handle*>(handle);
    if (!h) return -1;
    h->session.forget_history();
    return 0;
}

// Test only: the reference's scalar per-move features of the observed turn (Session::scalar_features).
MR_API int mr_scalar_features(void* handle, float* feats) {
    auto* h = static_cast<Handle*>(handle);
    if (!h) return -1;
    try {
        if (auto why = h->session.scalar_features(feats); !why.empty()) return fail(h, why);
        return 0;
    } catch (...) {
        return fail(h, "exception");
    }
}

// Copies one of the reference's weight arrays (policy_w0 ... value_b2) into `out`. Returns its length, or -1 for an
// unknown name; with out == nullptr or cap too small only the length is returned.
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
