// Combat context: the facts about fire, kills and exposure that the policy
// family p3 and the value family v3 add to the legacy encoders
// (docs/FEATURE_FAMILIES.md, training/specs/FEATURES.md).
//
// The legacy policy inputs (features.hpp, family p2) describe a unit and each
// destination, and the target head alone sees a target's hit points. Whether a
// destination puts a one-hit kill, a carrier or a coordinated volley in reach
// never reaches the mover, squad or act heads, so the decision to approach,
// retreat, attack or pick up cannot depend on it. The combat block per
// (unit, destination) carries exactly that, and the value extension carries
// the matching public facts about carriers, flags and rivals.
//
// Every rule comes from the engine (inference/sim.hpp mirrors engine
// Game.step): damage per hit and starting hit points from INIT, hits add up
// within one settlement, a unit respawning in this settlement can neither
// shoot nor be shot, a carrier cannot shoot (attacks settle before drops),
// and Board::canAttack is engine Game.can_attack from the post-move cells.
//
// Two kinds of quantity, named apart:
//   *_stay   a fact about the current positions: true if the enemies named
//            stay where they stand (their orders are unknown);
//   *_reach  an upper bound over every legal move of the units named
//            (any destination they could stand on after this move).
// Movement blocking is not modelled: a destination counts as reached.
// Nothing here uses future actions or outcomes.
//
// CombatSummary is computed once per state and shared by every team's
// encoding (and by the value extension). Its per-cell buffers are sized once
// per board; after that nothing on the per-turn path allocates.
#ifndef FUSION_COMBAT_HPP
#define FUSION_COMBAT_HPP
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "features.hpp"
#include "sim.hpp"

namespace fusion {

// Semantics version of the combat summary shared by families p3 and v3.
inline constexpr int kCombatVersion = 1;

struct CombatUnit {
    int cell = -1;           // origin this turn, -1 off the board
    bool hittable = false;   // on the board and not respawning now
    bool shooter = false;    // on the board, not respawning now, not carrying
    bool carrier = false;    // on the board and carrying a flag
    int hits = 0;            // hits that kill it this settlement: ceil(hp / damage); 0 if not hittable
    int home = 99;           // a carrier's BFS distance to its own base (capped at 99); 99 otherwise
    int ndest = 0;           // cells it can stand on after its move (stay included)
    std::array<int, kMoves> dest{};
};

struct CombatSummary {
    int damage = 34;
    int units = 0;
    std::array<CombatUnit, kMaxUnits> u{};
    std::vector<int> threat;   // per cell: shooters of any team able to fire at the cell after their move
    std::vector<int> stamp;    // per cell: last shooter that counted it
    std::vector<int> touched;  // cells whose threat is nonzero (reset list)
    int touchedCount = 0;

    // True when shooter w can fire at cell c from some cell it can stand on after its move.
    bool reach(const Board& b, int w, int c) const {
        const CombatUnit& x = u[w];
        if (!x.shooter || c < 0) return false;
        for (int i = 0; i < x.ndest; i++)
            if (b.canAttack(x.dest[i], c)) return true;
        return false;
    }
};

inline int requiredHits(int hp, int damage) { return hp <= 0 ? 0 : (hp + damage - 1) / damage; }

// Fill `cs` for state `s` (TurnView `v` from makeView). Engine damage comes from INIT (Rules).
inline void summarize(const Board& b, const Rules& r, const State& s, const TurnView& v, CombatSummary& cs) {
    const int N = b.n * b.n;
    cs.damage = std::max(1, r.damage);
    cs.units = std::min(static_cast<int>(s.units.size()), kMaxUnits);
    if (static_cast<int>(cs.threat.size()) != N) {
        cs.threat.assign(N, 0);
        cs.stamp.assign(N, -1);
        cs.touched.assign(N, 0);
        cs.touchedCount = 0;
    }
    for (int i = 0; i < cs.touchedCount; i++) {
        cs.threat[cs.touched[i]] = 0;
        cs.stamp[cs.touched[i]] = -1;
    }
    cs.touchedCount = 0;
    for (int id = 0; id < cs.units; id++) {
        CombatUnit& x = cs.u[id];
        x = CombatUnit{};
        const Unit& w = s.units[id];
        x.cell = v.origin[id];
        if (x.cell < 0) continue;
        x.hittable = detail::hittable(v, id);
        x.shooter = detail::shooter(s, v, id);
        x.carrier = w.flag >= 0;
        x.hits = x.hittable ? requiredHits(w.hp, cs.damage) : 0;
        x.home = x.carrier ? detail::capped(b.home[w.team][x.cell]) : 99;
        for (int m = 0; m < kMoves; m++)
            if (b.nb[x.cell][m] >= 0) x.dest[x.ndest++] = b.nb[x.cell][m];
        if (!x.shooter) continue;
        for (int i = 0; i < x.ndest; i++) {
            const int d = x.dest[i], dx = d % b.n, dy = d / b.n;
            for (int oy = -2; oy <= 2; oy++)
                for (int ox = -2 + std::abs(oy); ox <= 2 - std::abs(oy); ox++) {
                    const int cx = dx + ox, cy = dy + oy;
                    if (cx < 0 || cy < 0 || cx >= b.n || cy >= b.n) continue;
                    const int c = b.cell(cx, cy);
                    if (cs.stamp[c] == id || !b.canAttack(d, c)) continue;
                    cs.stamp[c] = id;
                    if (cs.threat[c] == 0) cs.touched[cs.touchedCount++] = c;
                    cs.threat[c]++;
                }
        }
    }
}

// Per team: which of its own shooters can fire at each unit's current cell
// after their move (the coordinated-volley support), computed once per team.
struct TeamCombat {
    int team = -1;
    std::array<std::array<unsigned char, kMaxUnits>, 3> reach{};

    void build(const Board& b, const CombatSummary& cs, int t) {
        team = t;
        for (auto& row : reach) row.fill(0);
        for (int j = 0; j < 3; j++) {
            const int a = 3 * t + j;
            if (a >= cs.units || !cs.u[a].shooter) continue;
            for (int e = 0; e < cs.units; e++)
                if (e / 3 != t && cs.u[e].hittable && b.man(cs.u[a].cell, cs.u[e].cell) <= 3)
                    reach[j][e] = cs.reach(b, a, cs.u[e].cell);
        }
    }
    // Teammates of own unit k (k excluded) able to fire at enemy e this turn.
    int support(int k, int e) const {
        int n = 0;
        for (int j = 0; j < 3; j++) n += j != k && reach[j][e];
        return n;
    }
    int shotsReach(int e) const { return reach[0][e] + reach[1][e] + reach[2][e]; }
};

// Enemy shooters (not of `team`) that can fire at cell c after their move.
inline int exposureReach(const Board& b, const CombatSummary& cs, int team, int c) {
    if (c < 0) return 0;
    int n = cs.threat[c];
    for (int j = 0; j < 3; j++) {
        const int a = 3 * team + j;
        if (a < cs.units && cs.reach(b, a, c)) n--;
    }
    return n;
}

// ---------------------------------------------------------------- policy p3

// Per own unit k and move m (destination q), appended to the legacy move
// features in the mover, act and squad inputs. Own-fire features (kill1,
// volley, focus) are 0 when unit k cannot shoot this turn (carrying or
// respawning). An illegal move or a unit off the board has an all-zero block,
// as its legacy move features are.
inline constexpr FeatureSpec kCombatSpec[] = {
    {"kill1_stay", 0, 15, 5},          {"kill1_carrier_stay", 0, 8, 2},
    {"volley_stay", 0, 15, 5},         {"volley_carrier_stay", 0, 8, 2},
    {"focus_present", 0, 1, 1},        {"focus_hits", 0, 4, 4},
    {"focus_home", 0, 99, 50},         {"focus_support", 0, 2, 2},
    {"chase_present", 0, 1, 1},        {"chase_dist", 0, 6, 6},
    {"chase_hits", 0, 4, 4},           {"chase_home", 0, 99, 50},
    {"exposure_reach", 0, 15, 5},      {"lethal_stay", 0, 1, 1},
    {"lethal_reach", 0, 1, 1},
};
inline constexpr int kPC = sizeof(kCombatSpec) / sizeof(FeatureSpec);
inline constexpr int kChaseRange = 6;

using CombatBlock = std::array<std::array<std::array<int16_t, kPC>, kMoves>, 3>;

inline void encodeCombatPolicy(const Board& b, const State& s, const TurnView& v, const CombatSummary& cs, int team,
                               CombatBlock& out) {
    for (auto& unit : out)
        for (auto& mv : unit) mv.fill(0);
    TeamCombat tc;
    tc.build(b, cs, team);
    for (int k = 0; k < 3; k++) {
        const int id = 3 * team + k;
        if (id >= cs.units) continue;
        const CombatUnit& me = cs.u[id];
        const int p = me.cell;
        if (p < 0) continue;
        const int hp = s.units[id].hp;
        for (int m = 0; m < kMoves; m++) {
            const int q = b.nb[p][m];
            if (q < 0) continue;
            int kill1 = 0, kill1c = 0, volley = 0, volleyc = 0;
            int focus = -1, chase = -1, chaseDist = 0, shootersStay = 0;
            for (int e = 0; e < cs.units; e++) {
                if (e / 3 == team) continue;
                const CombatUnit& x = cs.u[e];
                if (x.cell < 0) continue;
                if (!v.immune[id] && x.shooter && b.canAttack(x.cell, q)) shootersStay++;
                if (!x.hittable) continue;
                if (x.carrier) {
                    const int d = b.man(q, x.cell);
                    if (d <= kChaseRange) {
                        const CombatUnit* c = chase >= 0 ? &cs.u[chase] : nullptr;
                        if (!c || d < chaseDist || (d == chaseDist && (x.hits < c->hits || (x.hits == c->hits && x.home < c->home)))) {
                            chase = e;
                            chaseDist = d;
                        }
                    }
                }
                if (!me.shooter || !b.canAttack(q, x.cell)) continue;
                const bool one = x.hits == 1, volleyKill = 1 + tc.support(k, e) >= x.hits;
                kill1 += one;
                volley += volleyKill;
                if (x.carrier) {
                    kill1c += one;
                    volleyc += volleyKill;
                    const CombatUnit* f = focus >= 0 ? &cs.u[focus] : nullptr;
                    if (!f || x.hits < f->hits || (x.hits == f->hits && x.home < f->home)) focus = e;
                }
            }
            const int exposure = v.immune[id] ? 0 : exposureReach(b, cs, team, q);
            const int lethalStay = !v.immune[id] && shootersStay * cs.damage >= hp;
            const int lethalReach = !v.immune[id] && exposure * cs.damage >= hp;
            const int vals[kPC] = {kill1, kill1c, volley, volleyc,
                                   focus >= 0, focus >= 0 ? cs.u[focus].hits : 0, focus >= 0 ? cs.u[focus].home : 0,
                                   focus >= 0 ? tc.support(k, focus) : 0,
                                   chase >= 0, chase >= 0 ? chaseDist : 0, chase >= 0 ? cs.u[chase].hits : 0,
                                   chase >= 0 ? cs.u[chase].home : 0,
                                   exposure, lethalStay, lethalReach};
            for (int i = 0; i < kPC; i++) out[k][m][i] = clip(vals[i], kCombatSpec[i]);
        }
    }
}

// ---------------------------------------------------------------- value v3

// Appended to the 54 legacy value features (encodeValue). Absent slots are 0
// with their `*_present` bit 0; "none" for a timer or distance is the value
// past its range (16, 9, 99), so larger always means later or farther.
inline constexpr FeatureSpec kValueExtSpec[] = {
    // Own units, by id: hits that kill it now, enemy shooters able to reach a firing cell, whether that kills it.
    {"u0_hits", 0, 4, 4}, {"u1_hits", 0, 4, 4}, {"u2_hits", 0, 4, 4},
    {"u0_exposure", 0, 15, 5}, {"u1_exposure", 0, 15, 5}, {"u2_exposure", 0, 15, 5},
    {"u0_lethal", 0, 1, 1}, {"u1_lethal", 0, 1, 1}, {"u2_lethal", 0, 1, 1},
    // Two enemy carriers, nearest delivery first (then fewer hits, then id).
    {"ec0_present", 0, 1, 1}, {"ec0_hits", 0, 4, 4}, {"ec0_home", 0, 99, 50}, {"ec0_shots_stay", 0, 3, 3},
    {"ec0_shots_reach", 0, 3, 3}, {"ec0_killable", 0, 1, 1}, {"ec0_escort", 0, 2, 2}, {"ec0_gap", -20, 20, 10},
    {"ec1_present", 0, 1, 1}, {"ec1_hits", 0, 4, 4}, {"ec1_home", 0, 99, 50}, {"ec1_shots_stay", 0, 3, 3},
    {"ec1_shots_reach", 0, 3, 3}, {"ec1_killable", 0, 1, 1}, {"ec1_escort", 0, 2, 2}, {"ec1_gap", -20, 20, 10},
    {"own_carrier_escort", 0, 2, 2},
    // Flags: timers, distances, races.
    {"dropped_return_min", 0, 16, 16}, {"cooldown_ready_min", 0, 9, 9}, {"flag_own_dist", 0, 99, 50},
    {"flag_enemy_dist", 0, 99, 50}, {"flag_contest", -20, 20, 10}, {"dropped_own_closer", 0, 8, 4},
    {"dropped_enemy_closer", 0, 8, 4},
    // Three rivals by score (then team id).
    {"r0_present", 0, 1, 1}, {"r0_gap", -20, 20, 10}, {"r0_carriers", 0, 3, 3}, {"r0_active", 0, 3, 3}, {"r0_carrier_home", 0, 99, 50},
    {"r1_present", 0, 1, 1}, {"r1_gap", -20, 20, 10}, {"r1_carriers", 0, 3, 3}, {"r1_active", 0, 3, 3}, {"r1_carrier_home", 0, 99, 50},
    {"r2_present", 0, 1, 1}, {"r2_gap", -20, 20, 10}, {"r2_carriers", 0, 3, 3}, {"r2_active", 0, 3, 3}, {"r2_carrier_home", 0, 99, 50},
    // Resources and races between scores and carriers.
    {"own_respawning", 0, 3, 3}, {"enemy_respawning", 0, 42, 10}, {"rivals_ahead_carrying", 0, 14, 7},
    {"rivals_close_carrying", 0, 14, 7}, {"enemy_carriers_killable", 0, 8, 4}, {"own_carriers_lethal", 0, 3, 3},
};
inline constexpr int kVX = sizeof(kValueExtSpec) / sizeof(FeatureSpec);

inline void encodeValueExt(const Board& b, const State& s, const TurnView& v, const CombatSummary& cs, int team, int16_t* out) {
    int vals[kVX] = {};
    int i = 0;
    TeamCombat tc;
    tc.build(b, cs, team);
    std::array<int, 3> exposure{};
    for (int k = 0; k < 3; k++) {
        const int id = 3 * team + k;
        if (id < cs.units && cs.u[id].cell >= 0 && cs.u[id].hittable)
            exposure[k] = exposureReach(b, cs, team, cs.u[id].cell);
    }
    for (int k = 0; k < 3; k++) vals[i++] = 3 * team + k < cs.units ? cs.u[3 * team + k].hits : 0;
    for (int k = 0; k < 3; k++) vals[i++] = exposure[k];
    int ownCarriersLethal = 0;
    for (int k = 0; k < 3; k++) {
        const int id = 3 * team + k;
        const bool lethal = id < cs.units && cs.u[id].hittable && exposure[k] * cs.damage >= s.units[id].hp;
        vals[i++] = lethal;
        ownCarriersLethal += lethal && cs.u[id].carrier;
    }

    // Enemy carriers: the two nearest deliveries.
    std::array<int, 2> ec{-1, -1};
    int killable = 0;
    for (int e = 0; e < cs.units; e++) {
        const CombatUnit& x = cs.u[e];
        if (e / 3 == team || !x.carrier) continue;
        killable += tc.shotsReach(e) >= x.hits;
        auto better = [&](int a) {
            if (a < 0) return true;
            const CombatUnit& y = cs.u[a];
            return x.home < y.home || (x.home == y.home && x.hits < y.hits);
        };
        if (better(ec[0])) { ec[1] = ec[0]; ec[0] = e; }
        else if (better(ec[1])) ec[1] = e;
    }
    for (int slot = 0; slot < 2; slot++) {
        const int e = ec[slot];
        if (e < 0) { i += 8; continue; }
        const CombatUnit& x = cs.u[e];
        int stay = 0, escort = 0;
        for (int j = 0; j < 3; j++) {
            const int a = 3 * team + j;
            if (a < cs.units && cs.u[a].shooter && b.canAttack(cs.u[a].cell, x.cell)) stay++;
            const int mate = 3 * (e / 3) + j;
            if (mate != e && mate < cs.units && cs.u[mate].cell >= 0 && b.man(cs.u[mate].cell, x.cell) <= 3) escort++;
        }
        const int reach = tc.shotsReach(e);
        const int vs[8] = {1, x.hits, x.home, stay, reach, reach >= x.hits, escort, s.score[e / 3] - s.score[team]};
        for (int z = 0; z < 8; z++) vals[i++] = vs[z];
    }

    // Our best-placed carrier's escort: teammates within 3.
    int ownCarrier = -1;
    for (int k = 0; k < 3; k++) {
        const int id = 3 * team + k;
        if (id < cs.units && cs.u[id].carrier && (ownCarrier < 0 || cs.u[id].home < cs.u[ownCarrier].home)) ownCarrier = id;
    }
    int ownEscort = 0;
    if (ownCarrier >= 0)
        for (int k = 0; k < 3; k++) {
            const int id = 3 * team + k;
            if (id != ownCarrier && id < cs.units && cs.u[id].cell >= 0 && b.man(cs.u[id].cell, cs.u[ownCarrier].cell) <= 3) ownEscort++;
        }
    vals[i++] = ownEscort;

    // Flags.
    int droppedReturn = 16, cooldownReady = 9, ownCloser = 0, enemyCloser = 0;
    for (const Flag& f : s.flags) {
        if (f.status == kCooldown) cooldownReady = std::min(cooldownReady, std::max(0, f.returnAt - s.turn));
        if (f.status != kDropped || !available(f, s.turn)) continue;
        droppedReturn = std::min(droppedReturn, f.returnAt - s.turn);
        const int c = b.cell(f.x, f.y);
        int own = kUnreachable, enemy = kUnreachable;
        for (int id = 0; id < cs.units; id++) {
            if (cs.u[id].cell < 0) continue;
            int& best = id / 3 == team ? own : enemy;
            best = std::min(best, b.man(cs.u[id].cell, c));
        }
        ownCloser += own < enemy;
        enemyCloser += enemy < own;
    }
    int flagOwn = 99, flagEnemy = 99;
    for (int id = 0; id < cs.units; id++) {
        const int c = cs.u[id].cell;
        if (c < 0) continue;
        const int d = detail::capped(v.flagDist[c]);
        int& best = id / 3 == team ? flagOwn : flagEnemy;
        best = std::min(best, d);
    }
    const int fl[7] = {droppedReturn, cooldownReady, flagOwn, flagEnemy, flagEnemy - flagOwn, ownCloser, enemyCloser};
    for (int z = 0; z < 7; z++) vals[i++] = fl[z];

    // Rivals by score.
    std::array<int, kMaxTeams> carriers{}, active{}, carrierHome{};
    carrierHome.fill(99);
    int ownRespawning = 0, enemyRespawning = 0;
    for (int id = 0; id < cs.units; id++) {
        const int t = id / 3;
        if (cs.u[id].cell < 0) {
            (t == team ? ownRespawning : enemyRespawning)++;
            continue;
        }
        active[t]++;
        if (cs.u[id].carrier) {
            carriers[t]++;
            carrierHome[t] = std::min(carrierHome[t], cs.u[id].home);
        }
    }
    std::array<int, kMaxTeams> order{};
    int rivals = 0;
    for (int t = 0; t < b.teams && t < kMaxTeams; t++)
        if (t != team) order[rivals++] = t;
    for (int r = 1; r < rivals; r++)  // insertion sort by score, stable (std::stable_sort may allocate)
        for (int q = r; q > 0 && s.score[order[q]] > s.score[order[q - 1]]; q--) std::swap(order[q], order[q - 1]);
    int aheadCarrying = 0, closeCarrying = 0;
    for (int r = 0; r < rivals; r++) {
        const int t = order[r], gap = s.score[t] - s.score[team];
        aheadCarrying += gap > 0 && carriers[t] > 0;
        closeCarrying += (gap == 0 || gap == -1) && carriers[t] > 0;
    }
    for (int r = 0; r < 3; r++) {
        if (r >= rivals) { i += 5; continue; }
        const int t = order[r];
        const int vs[5] = {1, s.score[t] - s.score[team], carriers[t], active[t], carrierHome[t]};
        for (int z = 0; z < 5; z++) vals[i++] = vs[z];
    }
    const int misc[6] = {ownRespawning, enemyRespawning, aheadCarrying, closeCarrying, killable, ownCarriersLethal};
    for (int z = 0; z < 6; z++) vals[i++] = misc[z];
    for (int z = 0; z < kVX; z++) out[z] = clip(vals[z], kValueExtSpec[z]);
}

}  // namespace fusion
#endif  // FUSION_COMBAT_HPP
