// Our opponent model P in the agent_v2 carrier (task T0065, docs/AGENT_V2.md).
//
// P is an OPPONENT model. For every opponent team it predicts the 125-way joint movement (mover + squad residual, so
// the three units' moves stay coordinated), consistent per-unit move marginals and, conditional on a unit's move,
// P(wait, pickup, drop, attack) and P(target slot) over the encoder's eight attack candidates (enemy ids nearest
// first; -1 empty; candidates beyond eight are truncated: a missing slot is missing support, never a target). The
// implementation is the existing one, unchanged: bots/p_only/include/ponly.hpp (ponly::View) over inference/policy.hpp,
// with the model registry the variant includes (FUSION_TIER_*; fusion::models::forTeams). P2 XXL
// (inference/models_pxxl_vm: dedicated n02-n08, n10, n12, n15; general at 9, 11, 13, 14) is the first reference
// candidate, not a chosen best. No mixture with a hand-written forecast is applied (ponly::View ignores alpha).
//
// Inputs, exactly as P was trained: the raw State (never a prepared one: P distinguishes a dead unit about to respawn)
// and the previous REAL State as history (previous cells, attacks). Teams are predicted independently; within a team
// the movement is coordinated, the action/target heads are per unit given its move. P does not mask squad collisions.
//
// Cache identity: every prediction (a View) carries the key of what it was computed from: the state, its parent and
// the model. The root View lives for one real turn; a branch View is built for one simulated state with the root
// state as parent and dropped afterwards; nothing a branch computes is written back. Reading a View for a state it
// was not computed for is refused (Counters::inconsistent, and an empty result).
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "core.hpp"
#include "resolver.hpp"
#include "../../bots/p_only/include/ponly.hpp"

namespace agent_v2 {

inline uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 1099511628211ull; }

// A content key of a State (turn, scores, units, flags, attacks).
inline uint64_t state_key(const fusion::State& s) {
    uint64_t h = mix(1469598103934665603ull, uint64_t(s.turn));
    for (int v : s.score) h = mix(h, uint64_t(uint32_t(v)));
    for (auto& u : s.units)
        for (int v : {u.id, u.team, u.x, u.y, u.hp, u.flag, u.respawnAt}) h = mix(h, uint64_t(uint32_t(v)));
    for (auto& f : s.flags)
        for (int v : {f.id, f.status, f.x, f.y, f.holder, f.returnAt}) h = mix(h, uint64_t(uint32_t(v)));
    for (auto& a : s.attacks) h = mix(mix(h, uint64_t(uint32_t(a.unit))), uint64_t(uint32_t(a.target)));
    return h;
}

struct PredictionKey {
    uint64_t state = 0, parent = 0;
    const fusion::PolicyModel* model = nullptr;
    bool operator==(const PredictionKey& o) const { return state == o.state && parent == o.parent && model == o.model; }
};

struct Prediction {
    ponly::View view;
    PredictionKey key;
};

struct PredictorP {
    static constexpr const char* kKind = "P";
    ponly::Counters counters;
    Prediction root;
    const fusion::models::ModelSet* set = nullptr;
    int me = -1;

    void begin(const MapCache& c) {
        set = fusion::models::forTeams(c.st.teams);
        me = c.st.me;
    }
    const char* model_id() const { return set ? set->id : "none"; }
    const fusion::PolicyModel* model() const { return set ? set->policy : nullptr; }

    void predict(const MapCache& c, const fusion::State& s, const fusion::State* parent, Prediction& out) {
        out.view.counters = &counters;
        out.view.build(c.board, s, parent, me, model());
        out.key = {state_key(s), parent ? state_key(*parent) : 0, model()};
    }
    // The real turn: raw state and the previous real state.
    void observe(const MapCache& c, const fusion::State& s, const fusion::State* parent) { predict(c, s, parent, root); }
    // A simulated successor: its parent is the real root state.
    void branch(const MapCache& c, const fusion::State& s, Prediction& out) {
        counters.branches++;
        predict(c, s, &root.view.state(), out);
    }

    // The resolver's scenario table: K complete opponent order sets drawn from prediction `p` (computed for state
    // `s`), keyed by `salt` (scenario i uses sample i). Refused, empty, for a state the prediction was not made for.
    bool scenarios(Prediction& p, const fusion::State& s, int K, uint32_t salt, std::vector<Scenario>& out) {
        out.clear();
        if (p.key.state != state_key(s) || p.key.model != model()) {
            counters.inconsistent++;
            return false;
        }
        out.resize(K);
        for (int i = 0; i < K; i++) opponent_orders(p, me, i, salt, out[i]);
        return true;
    }

    // Movement marginals of every unit of state `s` from prediction `p` (0 for our team and unpredicted teams).
    bool marginals(const Prediction& p, const fusion::State& s, Marginals& q) {
        q.assign(s.units.size(), {});
        if (p.key.state != state_key(s) || p.key.model != model()) {
            counters.inconsistent++;
            return false;
        }
        for (int uid = 0; uid < int(s.units.size()); uid++)
            for (int m = 0; m < kMoves; m++) q[uid][m] = p.view.move(uid, m);
        return true;
    }

    // Complete orders of every opponent unit for scenario `sample` (joint movement draw, then action and target
    // draws), keyed by (turn, team, unit, sample, salt): all root candidates face the same draws. Our units wait.
    static void opponent_orders(Prediction& p, int me, int sample, uint32_t salt, std::vector<fusion::Order>& out) {
        const int U = int(p.view.state().units.size());
        out.assign(U, fusion::Order{});
        for (int t = 0; t * 3 < U; t++) {
            if (t == me || !p.view.has(t)) continue;
            auto o = p.view.teamOrders(t, sample, salt);
            for (int k = 0; k < 3; k++) out[3 * t + k] = {o[k].move, o[k].act, o[k].target};
        }
    }
};

}  // namespace agent_v2
