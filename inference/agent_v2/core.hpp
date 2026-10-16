// Public state, map cache, causal history and versions of the agent_v2 carrier (task T0065, docs/AGENT_V2.md).
//
// One public state for every module: fusion::Static (the INIT block) and fusion::State (one TURN block), the
// engine-mirrored representation of inference/state.hpp, parsed once (tools/cpp/stream.hpp). Each model reads it
// through its own adapter and its own versioned input contract:
//   * our opponent model P: the raw State and the previous real State (bots/p_only/include/ponly.hpp);
//   * the actor and the critic: the meow feature schemas (features_meow_v1.hpp), which convert the same State;
//   * the resolver and the simulator: fusion::Board geometry and fusion::makeView respawn/immunity facts.
// The map cache (Board, and the feature schema's own geometry) is built once from INIT and never changes.
// History is the previous REAL State only: it advances once per answered real turn; a search branch keeps its own
// parent pointer and never writes it back; a refused block clears it (the next turn is read without a parent).
#pragma once
#include <string>
#include <vector>

#include "sim.hpp"     // fusion::Rules, fusion::step (and state.hpp)
#include "state.hpp"

namespace agent_v2 {

// Versions of everything a run, an export or a bot must agree on (training/agent_v2/identity.py records them).
inline constexpr const char* kCarrier = "agent_v2_v1";
inline constexpr const char* kActionSchema = "agent_v2_actions_v1";  // pi's joint movement (125) + resolver assign_v1
inline constexpr const char* kResolverRule = "assign_v1";
inline constexpr int kUnits = 3, kMoves = 5, kJoint = 125;

// What the rules index without checking must be in range; returns "" or why the turn is refused.
inline std::string invalid(const fusion::Static& st, const fusion::State& s) {
    if (st.teams < 2 || st.teams > 15) return "teams outside 2-15";
    if (st.me < 0 || st.me >= st.teams) return "seat outside the team range";
    if (st.size <= 0 || st.size * st.size > 2500 || int(st.wall.size()) != st.size * st.size) return "bad map size";
    if (int(st.baseCenter.size()) != st.teams || st.spots.empty()) return "bad bases or spots";
    if (int(s.units.size()) != 3 * st.teams || int(s.score.size()) != st.teams) return "expected 3 units and a score per team";
    auto in = [&](int x, int y) { return x >= 0 && y >= 0 && x < st.size && y < st.size; };
    for (auto& b : st.baseCenter)
        if (b[0] < 1 || b[1] < 1 || b[0] > st.size - 2 || b[1] > st.size - 2) return "base not inside the map";
    for (auto& p : st.spots)
        if (!in(p[0], p[1])) return "spot outside the map";
    for (int i = 0; i < int(s.units.size()); i++) {
        auto& u = s.units[i];
        if (u.id != i || u.team != i / 3) return "unit ids must be 0.. in team order";
        if (u.x >= 0 && (!in(u.x, u.y) || st.wall[u.y * st.size + u.x])) return "unit outside the open map";
        if (u.flag < -1 || u.flag >= int(s.flags.size())) return "unit carries an unknown flag";
    }
    for (int i = 0; i < int(s.flags.size()); i++) {
        auto& f = s.flags[i];
        if (f.id != i) return "flag ids must be 0..";
        if (f.holder < -1 || f.holder >= int(s.units.size())) return "flag held by an unknown unit";
        if (f.x >= 0 && !in(f.x, f.y)) return "flag outside the map";
    }
    for (auto& a : s.attacks)
        if (a.unit < 0 || a.unit >= int(s.units.size()) || a.target < 0 || a.target >= int(s.units.size()))
            return "attack event for an unknown unit";
    return "";
}

// The immutable map cache of one episode.
struct MapCache {
    fusion::Static st;
    fusion::Board board;
    fusion::Rules rules;
    bool ready = false;
    void init(const fusion::Static& s) {
        st = s;
        board.init(s);
        rules = {s.hp, s.damage, s.respawn, s.flagReturn, s.flagCooldown};
        ready = true;
    }
};

// Causal memory: the previous real observation.
struct History {
    fusion::State previous;
    bool has_previous = false;
    const fusion::State* parent() const { return has_previous ? &previous : nullptr; }
    void advance(const fusion::State& real) { previous = real, has_previous = true; }
    void forget() { has_previous = false; }
};

}  // namespace agent_v2

