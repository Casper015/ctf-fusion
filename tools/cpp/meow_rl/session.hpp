// One episode of one seat on the unmodified Meowmind reference: the code the trainer's bridge (bridge.cpp) and the
// deployed bot (bots/meow_rl/) share, so the two cannot drift (task T0054, docs/MEOWMIND_RL_IMPLEMENTATION.md).
//
// Include the reference first (bots/meowmind_p/meowmind-ctf-374-20261011.cpp with `main` renamed): this header uses
// its Game, Planner and learning:: code unchanged for
//   * per-episode state: geometry, movement history (Planner::previous), the opponent experts' online belief
//     (Planner::adapt / predict, frozen weights) and the new-life flags the resolver reads;
//   * the 5 x 112 per-unit move features (learning::Geometry::policyMoves), the 416 public value features
//     (learning::Geometry::value) and the reference's move feasibility (wall bumps, dead units stay, no two own
//     units on one destination);
//   * the frozen attack/pickup resolver (Planner::shots with the reference's defaults) and the reply format.
#pragma once
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace meow_rl {

constexpr int kUnits = 3, kMoves = 5, kJoint = 125, kPF = learning::PF, kVF = learning::VF;
static_assert(kPF == 112 && kVF == 416, "the reference's 112/416 feature schema");

struct Observation {
    int turn = 0, turns = 0, teams = 0, me = 0, size = 0, alive = 0, valid = 0, units = 0;
    float feats[kUnits][kMoves][kPF];
    unsigned char unit_legal[kUnits][kMoves];
    unsigned char joint_mask[kJoint];
    float value_feats[kVF];
    float ref_prob[kUnits][kMoves];
};

// Everything the reference indexes without checking: seat, ids, teams, flags, holders and cells must be in range.
inline std::string invalid(const Game& g) {
    if (g.teams < 2 || g.teams > 15) return "teams outside 2-15";
    if (g.me < 0 || g.me >= g.teams) return "seat outside the team range";
    if (g.size <= 0 || g.size * g.size > 2500 || int(g.map.size()) != g.size) return "bad map size";
    for (auto& row : g.map)
        if (int(row.size()) != g.size) return "bad map row";
    if (int(g.bases.size()) != g.teams || g.spots.empty()) return "bad bases or spots";
    if (int(g.units.size()) != 3 * g.teams || int(g.score.size()) != g.teams) return "expected 3 units and a score per team";
    auto cell = [&](int x, int y) { return x >= 0 && y >= 0 && x < g.size && y < g.size; };
    for (auto& b : g.bases)        // Geometry::init reads the 3x3 base around each center
        if (b.first < 1 || b.second < 1 || b.first > g.size - 2 || b.second > g.size - 2) return "base not inside the map";
    for (auto& s : g.spots)
        if (!cell(s.first, s.second)) return "spot outside the map";
    for (int i = 0; i < int(g.units.size()); i++) {
        auto& u = g.units[i];
        if (u.id != i || u.team != i / 3) return "unit ids must be 0.. in team order";
        if (u.x >= 0 && !cell(u.x, u.y)) return "unit outside the map";
        if (u.flag < -1 || u.flag >= int(g.flags.size())) return "unit carries an unknown flag";
    }
    for (int i = 0; i < int(g.flags.size()); i++) {
        auto& f = g.flags[i];
        if (f.id != i) return "flag ids must be 0..";
        if (f.holder < -1 || f.holder >= int(g.units.size())) return "flag held by an unknown unit";
        if (f.x >= 0 && !cell(f.x, f.y)) return "flag outside the map";
    }
    for (auto& e : g.events)
        if (!e.args.empty() && e.type != "return" && (e.args[0] < 0 || e.args[0] >= int(g.units.size())))
            return "event for an unknown unit";
    return "";
}

// The learner's inputs in state `o` (a prepared state) for team `me`, with the planner's movement history:
// the move features, unit feasibility, joint mask, value features and (with `prob`) the reference actor's own
// probabilities. Returns false if the all-stay joint (the safe fallback) is infeasible. Session::observe uses it for
// the real turn; a search uses it at simulated states (with the history it set for that branch).
inline bool fill(const Planner& p, const Game& o, int me, const std::vector<Planner::Prob>* prob, Observation& out) {
    int alive = 0, dest[kUnits][kMoves];
    for (int k = 0; k < kUnits; k++) {
        int id = 3 * me + k, cell = p.geo.cell(o.units[id]);
        auto all = p.geo.policyMoves(o, id, p.previous[id]);
        for (int m = 0; m < kMoves; m++) {
            std::memcpy(out.feats[k][m], all[m].data(), sizeof(float) * kPF);
            out.unit_legal[k][m] = cell >= 0 ? (m == 0 || all[m][7] <= .5f) : m == 0;   // Planner::policy's -100 rule
            dest[k][m] = cell < 0 ? -1 : p.geo.nb[cell][m];
            out.ref_prob[k][m] = prob ? (*prob)[id][m] : 0.f;
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
                        if (dest[k][mv[k]] >= 0 && dest[k][mv[k]] == dest[l][mv[l]]) ok = false;   // Planner::moves
                out.joint_mask[a * 25 + b * 5 + c] = ok;
                valid += ok;
            }
    auto v = p.geo.value(o, me);
    std::memcpy(out.value_feats, v.data(), sizeof(float) * kVF);
    out.alive = alive, out.valid = valid;
    return out.joint_mask[0];
}

struct Session {
    Game g;                              // the raw blocks as read (INIT data persists across turns)
    Planner planner;                     // the reference's per-episode state
    Game observation;                    // learning::prepared(g): predicted respawns
    std::vector<Planner::Prob> prob;     // predict(): reference own policy, opponents' belief mixture
    int observed_turn = -1;              // the turn observed and not yet answered
    int answered_turn = -1;

    // After the caller parsed the next block into `g` (read_turn): Planner::run up to its search, unchanged, and
    // the learner's observation. Returns "" or why the block is refused (nothing in the state changed then).
    std::string observe(Observation& out) {
        if (observed_turn >= 0 && answered_turn != observed_turn)
            return "turn " + std::to_string(observed_turn) + " was observed and not answered";
        if (g.teams < 2 || g.size <= 0 || g.map.empty()) return "no INIT block before the first TURN";
        if (auto why = invalid(g); !why.empty()) return "malformed block: " + why;
        if (g.turn <= observed_turn) return "turns must increase";
        Planner& p = planner;
        if (!p.geo.n) {
            p.geo.init(g);
            p.previous.assign(g.units.size(), -1);
        }
        p.adapt(g);
        observation = learning::prepared(g);
        p.preparedTurn = g.turn;
        p.newLife.assign(g.units.size(), false);
        for (auto& u : observation.units) p.newLife[u.id] = g.units[u.id].x < 0 && u.x >= 0;
        prob = p.predict(observation, true);

        if (!fill(p, observation, g.me, &prob, out)) return "the all-stay joint move is infeasible";   // the safe fallback
        out.turn = g.turn, out.turns = g.turns, out.teams = g.teams, out.me = g.me, out.size = g.size;
        out.units = int(g.units.size());
        observed_turn = g.turn;
        return "";
    }

    // Answers the observed turn with joint move `joint` (a*25 + b*5 + c, which must be feasible): the frozen
    // resolver assigns attacks and pickups for exactly these moves (Planner::rollout's shots call), the reply is
    // formatted as Planner::run does and the movement history advances as there. `actions` receives the resolved
    // actions (target id, -1 pickup or wait, -3 drop). Returns "" or why the joint is refused (no repair).
    std::string resolve(int joint, std::string& reply, int actions[kUnits]) {
        if (observed_turn < 0 || answered_turn == observed_turn) return "no observed turn to answer";
        if (joint < 0 || joint >= kJoint) return "joint index out of range";
        const Game& o = observation;
        Planner& p = planner;
        Planner::Candidate ca;
        ca.moves = {joint / 25, joint / 5 % 5, joint % 5};
        int dest[kUnits];
        for (int k = 0; k < kUnits; k++) {
            int id = 3 * g.me + k, cell = p.geo.cell(o.units[id]), m = ca.moves[k];
            if (cell < 0 && m != 0) return "a dead unit must stay";
            if (cell >= 0 && m != 0 && p.geo.nb[cell][m] == cell) return "move into a wall";
            dest[k] = cell < 0 ? -1 : p.geo.nb[cell][m];
        }
        for (int k = 0; k < kUnits; k++)
            for (int l = k + 1; l < kUnits; l++)
                if (dest[k] >= 0 && dest[k] == dest[l]) return "two units share a destination";
        ca.actions = p.shots(o, g.me, ca.moves, prob);
        reply = std::to_string(g.turn);
        for (int k = 0; k < kUnits; k++) {
            int id = g.me * 3 + k;
            reply += ' ';
            reply += std::string("SUDLR")[ca.moves[k]];
            reply += ' ';
            reply += ca.actions[k] == -3 ? "X" : g.units[id].flag >= 0 ? "-" : ca.actions[k] >= 0 ? std::to_string(ca.actions[k]) : "P";
            actions[k] = ca.actions[k];
        }
        for (auto& u : g.units) p.previous[u.id] = p.geo.cell(u);
        answered_turn = observed_turn;
        return "";
    }

    // After a refused block: the movement history is unknown, so it is reset to the episode start (every unit's
    // previous cell -1: features 13-14 read 0 and Planner::adapt skips every unit next turn) and the turn counts
    // as answered. Opponent beliefs (evidence) are kept. The deployed bot calls this; the bridge exposes it.
    void forget_history() {
        // Keep the trusted length (set at initialization): a refused block's own unit count is not trusted.
        std::fill(planner.previous.begin(), planner.previous.end(), -1);
        answered_turn = observed_turn;
    }

    // Test only: the reference's scalar per-move features (learning::Geometry::policy) of the observed turn, before
    // resolve (the batched policyMoves of observe must equal them).
    std::string scalar_features(float* feats) {
        if (observed_turn < 0) return "nothing observed";
        if (answered_turn == observed_turn) return "call before resolve";
        for (int k = 0; k < kUnits; k++) {
            int id = 3 * g.me + k;
            for (int m = 0; m < kMoves; m++) {
                auto f = planner.geo.policy(observation, id, m, planner.previous[id]);
                std::memcpy(feats + (k * kMoves + m) * kPF, f.data(), sizeof(float) * kPF);
            }
        }
        return "";
    }
};

}  // namespace meow_rl
