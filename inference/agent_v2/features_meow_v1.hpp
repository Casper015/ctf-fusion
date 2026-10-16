// The meow feature schemas, as an input adapter of the agent_v2 carrier (task T0065, docs/AGENT_V2.md).
//
//   actor  meow_policy112_v1: per own unit and move the 112 features of learning::Geometry::policyMoves
//   critic meow_public416_v1: the 416 public value features of learning::Geometry::value (RICH_VALUE)
// These are the inputs the existing PPO actors and critics (T0055/T0059, the aligned exports) were trained on, so the
// carrier computes them with the original code (meowmind_features.hpp, a verbatim extract) on the same data that
// code read: the raw TURN state converted to the extracted Game struct, learning::prepared applied (deterministic
// respawns), and each unit's cell in the previous REAL observation as its movement history (-1 when there is none).
// A future actor or critic with other inputs is a new schema next to this one, never a silent change of it.
//
// The feasible joint mask is a POLICY restriction of our actor, not an engine rule: walls and dead units stay, as the
// reference's feasibility, and no two own units request one destination (the engine would resolve that collision;
// our actor never asks for it). The reference's other per-move pruning is not applied.
#pragma once
#include <cstring>
#include <string>
#include <vector>

#include "core.hpp"
#include "meowmind_features.hpp"

namespace agent_v2 {

inline constexpr const char* kPolicyFeatures = "meow_policy112_v1";
inline constexpr const char* kValueFeatures = "meow_public416_v1";
constexpr int kPF = mm::learning::PF, kVF = mm::learning::VF;
static_assert(kPF == 112 && kVF == 416, "the meow 112/416 feature schemas");

struct Observation {
    int turn = 0, turns = 0, teams = 0, me = 0, size = 0, alive = 0, valid = 0, units = 0;
    float feats[kUnits][kMoves][kPF];
    unsigned char unit_legal[kUnits][kMoves];
    unsigned char joint_mask[kJoint];
    float value_feats[kVF];
};

inline const char* flag_text(int status) {
    return status == fusion::kCarried ? "carried" : status == fusion::kDropped ? "dropped" : status == fusion::kCooldown ? "cooldown" : "home";
}

struct MeowEncoder {
    mm::Game base;                      // the INIT data in the extracted struct
    mm::learning::Geometry geo;

    void init(const fusion::Static& st) {
        base = mm::Game{};
        base.teams = st.teams, base.me = st.me, base.size = st.size, base.turns = st.turns, base.hp = st.hp;
        base.damage = st.damage, base.respawn = st.respawn, base.flag_return = st.flagReturn, base.flag_cooldown = st.flagCooldown;
        base.map.assign(st.size, std::string(st.size, '.'));
        for (int y = 0; y < st.size; y++)
            for (int x = 0; x < st.size; x++)
                if (st.wall[y * st.size + x]) base.map[y][x] = '#';
        for (auto& b : st.baseCenter) base.bases.push_back({b[0], b[1]});
        for (auto& p : st.spots) base.spots.push_back({p[0], p[1]});
        geo.init(base);
    }

    mm::Game game(const fusion::State& s) const {
        mm::Game g = base;
        g.turn = s.turn;
        g.score = s.score;
        for (auto& u : s.units) g.units.push_back({u.id, u.team, u.x, u.y, u.hp, u.flag, u.respawnAt});
        for (auto& f : s.flags) g.flags.push_back({f.id, flag_text(f.status), f.x, f.y, f.holder, f.returnAt});
        for (auto& a : s.attacks) g.events.push_back({"attack", {a.unit, a.target}});
        return g;
    }

    // Each unit's cell in `parent` (the previous real observation), -1 without one.
    std::vector<int> history(const fusion::State& s, const fusion::State* parent) const {
        std::vector<int> prev(s.units.size(), -1);
        if (parent && parent->units.size() == s.units.size())
            for (auto& u : parent->units) prev[u.id] = u.x < 0 ? -1 : u.y * base.size + u.x;
        return prev;
    }

    // The schema's inputs of raw state `s` for team `me`. Returns false if the all-stay joint is infeasible.
    bool encode(const fusion::State& s, const fusion::State* parent, int me, Observation& out) const {
        const mm::Game o = mm::learning::prepared(game(s));
        const std::vector<int> previous = history(s, parent);
        int alive = 0, dest[kUnits][kMoves];
        for (int k = 0; k < kUnits; k++) {
            int id = 3 * me + k, cell = geo.cell(o.units[id]);
            auto all = geo.policyMoves(o, id, previous[id]);
            for (int m = 0; m < kMoves; m++) {
                std::memcpy(out.feats[k][m], all[m].data(), sizeof(float) * kPF);
                out.unit_legal[k][m] = cell >= 0 ? (m == 0 || all[m][7] <= .5f) : m == 0;
                dest[k][m] = cell < 0 ? -1 : geo.nb[cell][m];
            }
            if (cell >= 0) alive |= 1 << k;
        }
        int valid = 0;
        for (int a = 0; a < kMoves; a++)
            for (int b = 0; b < kMoves; b++)
                for (int c = 0; c < kMoves; c++) {
                    int mv[3] = {a, b, c};
                    bool ok = true;
                    for (int k = 0; k < kUnits && ok; k++) ok = out.unit_legal[k][mv[k]];
                    for (int k = 0; k < kUnits && ok; k++)
                        for (int l = k + 1; l < kUnits && ok; l++)
                            if (dest[k][mv[k]] >= 0 && dest[k][mv[k]] == dest[l][mv[l]]) ok = false;
                    out.joint_mask[a * 25 + b * 5 + c] = ok;
                    valid += ok;
                }
        auto v = geo.value(o, me);
        std::memcpy(out.value_feats, v.data(), sizeof(float) * kVF);
        out.turn = o.turn, out.turns = o.turns, out.teams = o.teams, out.me = me, out.size = o.size;
        out.units = int(o.units.size()), out.alive = alive, out.valid = valid;
        return out.joint_mask[0];
    }

    // The critic's input alone (a simulated leaf).
    std::array<float, kVF> value(const fusion::State& s, int me) const { return geo.value(mm::learning::prepared(game(s)), me); }
};

}  // namespace agent_v2
