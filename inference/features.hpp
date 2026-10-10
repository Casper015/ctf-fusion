// Feature encoders, feature spec v2 (see training/specs/FEATURES.md).
//
// This header is the single source of truth for every model input: the
// training extractor (tools/cpp/extract_features.cpp) and the bot call the same
// functions, so training and inference cannot drift apart. Every feature is a
// clipped small integer; training divides by `scale`, and the exporter folds
// that division into the first layer, so inference stays integer and exact.
// No encoder allocates: all scratch is fixed-size and on the stack.
#ifndef FUSION_FEATURES_HPP
#define FUSION_FEATURES_HPP
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "state.hpp"

namespace fusion {

inline constexpr int kSpecVersion = 2;  // v2: score-distribution value features

struct FeatureSpec { const char* name; int lo, hi, scale; };

inline constexpr FeatureSpec kGlobalSpec[] = {
    {"teams", 2, 15, 15},           {"turn", 0, 400, 400},          {"remaining", 0, 400, 400},
    {"own_score", 0, 63, 10},       {"best_enemy_score", 0, 63, 10}, {"own_rank", 1, 15, 15},
    {"margin", -20, 20, 10},        {"flags_home", 0, 8, 8},        {"flags_dropped", 0, 8, 8},
    {"flags_carried_own", 0, 3, 3}, {"flags_carried_enemy", 0, 8, 8}, {"flags_unavailable", 0, 8, 8},
    {"map_size", 23, 49, 49},       {"own_active", 0, 3, 3},        {"enemy_active", 0, 42, 42},
};
inline constexpr FeatureSpec kUnitSpec[] = {
    {"active", 0, 1, 1},        {"immune", 0, 1, 1},       {"dead_wait", 0, 10, 10},   {"hp", 0, 100, 100},
    {"carrying", 0, 1, 1},      {"home_dist", 0, 99, 50},  {"flag_dist", 0, 99, 50},   {"nearest_enemy", 0, 20, 20},
    {"enemies_r3", 0, 15, 5},   {"allies_r3", 0, 2, 2},    {"threatened_by", 0, 15, 5}, {"last_dx", -1, 1, 1},
    {"last_dy", -1, 1, 1},      {"hits_taken", 0, 15, 3},
};
inline constexpr FeatureSpec kMoveSpec[] = {
    {"legal", 0, 1, 1},      {"d_home", -1, 1, 1},    {"d_flag", -1, 1, 1},       {"shooters", 0, 15, 5},
    {"enemies_r2", 0, 15, 5}, {"flag_cell", 0, 1, 1},  {"own_base", 0, 1, 1},      {"occupied", 0, 1, 1},
    {"continues", 0, 1, 1},  {"carrier_targets", 0, 8, 2}, {"targets", 0, 15, 5},
};
inline constexpr FeatureSpec kCandSpec[] = {
    {"dist", 0, 4, 4},      {"hp", 0, 100, 100},   {"carrying", 0, 1, 1}, {"support", 0, 2, 2},
    {"hit_S", 0, 1, 1},     {"hit_U", 0, 1, 1},    {"hit_D", 0, 1, 1},    {"hit_L", 0, 1, 1},
    {"hit_R", 0, 1, 1},     {"attacked_by_team", 0, 3, 3}, {"target_home", 0, 99, 50},
};
inline constexpr FeatureSpec kValueUnitSpec[] = {
    {"active", 0, 1, 1},     {"hp", 0, 100, 100},      {"carrying", 0, 1, 1},     {"dead_wait", 0, 10, 10},
    {"home_dist", 0, 99, 50}, {"flag_dist", 0, 99, 50}, {"nearest_enemy", 0, 20, 20}, {"threatened_by", 0, 15, 5},
    {"enemies_r3", 0, 15, 5},
};
inline constexpr FeatureSpec kValueAggSpec[] = {
    {"enemy_carriers", 0, 8, 8},     {"enemy_carrier_home", 0, 99, 50}, {"enemies_near_base", 0, 42, 10},
    {"escort_threat", 0, 30, 10},    {"own_carrier_home", 0, 99, 50},   {"dropped_near", 0, 8, 4},
    {"enemies_r6", 0, 42, 10},       {"carrier_shots", 0, 3, 3},
    // Score distribution (v2). Own score, best rival, rank and margin alone
    // cannot tell [5,6,4] from [5,6,5] for team 0, yet the outcome target
    // differs (0.5 vs 0.25 if nobody scores again).
    {"rivals_tied", 0, 14, 7},       {"rivals_1_2_ahead", 0, 14, 7},    {"rivals_1_2_behind", 0, 14, 7},
    {"pairwise_now", 0, 100, 100},   // current pairwise outcome x 100, rounded half up
};

inline constexpr int kG = sizeof(kGlobalSpec) / sizeof(FeatureSpec);
inline constexpr int kU = sizeof(kUnitSpec) / sizeof(FeatureSpec);
inline constexpr int kM = sizeof(kMoveSpec) / sizeof(FeatureSpec);
inline constexpr int kTF = sizeof(kCandSpec) / sizeof(FeatureSpec);
inline constexpr int kVU = sizeof(kValueUnitSpec) / sizeof(FeatureSpec);
inline constexpr int kVA = sizeof(kValueAggSpec) / sizeof(FeatureSpec);
inline constexpr int kV = kG + 3 * kVU + kVA;
inline constexpr int kCand = 8;      // attack candidates per unit
inline constexpr int kCandRange = 4; // farther targets cannot come within range 2 this turn
inline constexpr int kActs = 4;      // wait, pickup, drop, attack
enum Act { kWait = 0, kPickup = 1, kDrop = 2, kAttack = 3 };

struct PolicyFeatures {
    std::array<int16_t, kG> g{};
    std::array<std::array<int16_t, kU>, 3> u{};
    std::array<std::array<std::array<int16_t, kM>, kMoves>, 3> m{};
    std::array<std::array<unsigned char, kMoves>, 3> legal{};
    std::array<std::array<std::array<unsigned char, kActs>, kMoves>, 3> actLegal{};
    std::array<std::array<int, kCand>, 3> cand{};  // enemy unit ids, -1 = empty slot
    std::array<std::array<std::array<int16_t, kTF>, kCand>, 3> candF{};
};
using ValueFeatures = std::array<int16_t, kV>;

inline int16_t clip(int v, const FeatureSpec& s) { return static_cast<int16_t>(std::clamp(v, s.lo, s.hi)); }

namespace detail {

inline int sign(int v) { return (v > 0) - (v < 0); }
inline int capped(int d) { return std::min(d, 99); }

// A unit that can fire this turn: on the board, not respawning, not carrying.
inline bool shooter(const State& s, const TurnView& v, int id) {
    return v.origin[id] >= 0 && !v.immune[id] && s.units[id].flag < 0;
}
// A unit that can be hit this turn.
inline bool hittable(const TurnView& v, int id) { return v.origin[id] >= 0 && !v.immune[id]; }

inline void encodeGlobal(const Board& b, const State& s, const TurnView& v, int team, int16_t* g) {
    int best = 0, rank = 1;
    for (int t = 0; t < b.teams; t++)
        if (t != team) {
            best = std::max(best, s.score[t]);
            rank += s.score[t] > s.score[team];
        }
    int home = 0, dropped = 0, ownCarried = 0, enemyCarried = 0, unavailable = 0;
    for (const Flag& f : s.flags) {
        if (f.status == kCarried) {
            if (f.holder >= 0 && s.units[f.holder].team == team) ownCarried++;
            else enemyCarried++;
        } else if (available(f, s.turn)) {
            (f.status == kHome ? home : dropped)++;
        } else {
            unavailable++;
        }
    }
    int ownActive = 0, enemyActive = 0;
    for (const Unit& u : s.units)
        if (v.origin[u.id] >= 0) (u.team == team ? ownActive : enemyActive)++;
    const int vals[kG] = {b.teams, s.turn, b.turns - s.turn + 1, s.score[team], best, rank, s.score[team] - best,
                          home, dropped, ownCarried, enemyCarried, unavailable, b.n, ownActive, enemyActive};
    for (int i = 0; i < kG; i++) g[i] = clip(vals[i], kGlobalSpec[i]);
}

struct UnitFacts { int nearestEnemy = 20, enemiesR3 = 0, alliesR3 = 0, threatenedBy = 0; };

inline UnitFacts unitFacts(const Board& b, const State& s, const TurnView& v, int id) {
    UnitFacts f;
    const int p = v.origin[id], team = s.units[id].team;
    if (p < 0) return f;
    for (const Unit& w : s.units) {
        const int q = v.origin[w.id];
        if (w.id == id || q < 0) continue;
        const int d = b.man(p, q);
        if (w.team == team) {
            f.alliesR3 += d <= 3;
        } else {
            f.enemiesR3 += d <= 3;
            if (!v.immune[w.id]) f.nearestEnemy = std::min(f.nearestEnemy, d);
            if (!v.immune[id] && shooter(s, v, w.id) && b.canAttack(q, p)) f.threatenedBy++;
        }
    }
    return f;
}

}  // namespace detail

inline void encodePolicy(const Board& b, const State& s, const TurnView& v, int team, PolicyFeatures& out) {
    out = PolicyFeatures{};
    detail::encodeGlobal(b, s, v, team, out.g.data());
    for (int k = 0; k < 3; k++) {
        const int id = 3 * team + k;
        const Unit& u = s.units[id];
        const int p = v.origin[id];
        auto& uf = out.u[k];
        for (auto& c : out.cand[k]) c = -1;
        if (p < 0) {
            uf[2] = clip(u.respawnAt >= 0 ? u.respawnAt - s.turn : 0, kUnitSpec[2]);
            out.legal[k][0] = 1;  // a unit off the board can only stay
            continue;
        }
        const detail::UnitFacts f = detail::unitFacts(b, s, v, id);
        int lastDx = 0, lastDy = 0;
        if (v.prevCell[id] >= 0 && u.x >= 0 && b.man(v.prevCell[id], p) == 1) {
            lastDx = p % b.n - v.prevCell[id] % b.n;
            lastDy = p / b.n - v.prevCell[id] / b.n;
        }
        const int carrying = u.flag >= 0;
        const int unitVals[kU] = {1, v.immune[id], 0, v.immune[id] ? 100 : u.hp, carrying,
                                  detail::capped(b.home[team][p]), detail::capped(v.flagDist[p]), f.nearestEnemy,
                                  f.enemiesR3, f.alliesR3, f.threatenedBy, lastDx, lastDy, v.hitsTaken[id]};
        for (int i = 0; i < kU; i++) uf[i] = clip(unitVals[i], kUnitSpec[i]);

        for (int m = 0; m < kMoves; m++) {
            const int q = b.nb[p][m];
            if (q < 0) continue;
            out.legal[k][m] = 1;
            int shooters = 0, enemiesR2 = 0, carrierTargets = 0, targets = 0;
            for (const Unit& w : s.units) {
                const int r = v.origin[w.id];
                if (w.team == team || r < 0) continue;
                enemiesR2 += b.man(q, r) <= 2;
                if (!v.immune[id] && detail::shooter(s, v, w.id) && b.canAttack(r, q)) shooters++;
                if (detail::hittable(v, w.id) && b.canAttack(q, r)) {
                    targets++;
                    carrierTargets += w.flag >= 0;
                }
            }
            const int occupied = v.occupant[q] >= 0 && v.occupant[q] != id;
            const int continues = m != 0 && kDX[m] == lastDx && kDY[m] == lastDy;
            const int flagGain = v.flagDist[p] >= kUnreachable ? 0 : detail::sign(v.flagDist[q] - v.flagDist[p]);
            const int moveVals[kM] = {1, detail::sign(b.home[team][q] - b.home[team][p]), flagGain, shooters, enemiesR2,
                                      v.flagCell[q], b.baseOf[q] == team, occupied, continues, carrierTargets, targets};
            for (int i = 0; i < kM; i++) out.m[k][m][i] = clip(moveVals[i], kMoveSpec[i]);
        }

        // Attack candidates: hittable enemies within kCandRange, nearest first.
        std::array<std::pair<int, int>, kMaxUnits> near{};
        int found = 0;
        for (const Unit& w : s.units)
            if (w.team != team && detail::hittable(v, w.id) && b.man(p, v.origin[w.id]) <= kCandRange && found < kMaxUnits)
                near[found++] = {b.man(p, v.origin[w.id]), w.id};
        std::sort(near.begin(), near.begin() + found);
        const int count = std::min(kCand, found);
        for (int c = 0; c < count; c++) {
            const int t = near[c].second, r = v.origin[t];
            int support = 0, attackedBy = 0;
            for (int j = 0; j < 3; j++) {
                const int ally = 3 * team + j;
                if (ally != id && detail::shooter(s, v, ally) && b.canAttack(v.origin[ally], r)) support++;
            }
            for (const Attack& a : s.attacks)
                if (a.target == t && a.unit >= 0 && s.units[a.unit].team == team) attackedBy++;
            const Unit& w = s.units[t];
            int vals[kTF] = {near[c].first, w.hp, w.flag >= 0, support, 0, 0, 0, 0, 0, attackedBy,
                             w.flag >= 0 ? detail::capped(b.home[w.team][r]) : 99};
            for (int m = 0; m < kMoves; m++) vals[4 + m] = b.nb[p][m] >= 0 && b.canAttack(b.nb[p][m], r);
            out.cand[k][c] = t;
            for (int i = 0; i < kTF; i++) out.candF[k][c][i] = clip(vals[i], kCandSpec[i]);
        }

        // Effective actions per destination (anything else is a no-op in the engine).
        for (int m = 0; m < kMoves; m++) {
            const int q = b.nb[p][m];
            if (q < 0) continue;
            auto& a = out.actLegal[k][m];
            a[kWait] = 1;
            a[kPickup] = !carrying && v.flagCell[q];
            a[kDrop] = carrying;
            a[kAttack] = !carrying && !v.immune[id] && count > 0;
        }
    }
}

inline void encodeValue(const Board& b, const State& s, const TurnView& v, int team, ValueFeatures& out) {
    out.fill(0);
    detail::encodeGlobal(b, s, v, team, out.data());
    int16_t* unit = out.data() + kG;
    int ownCarrierHome = 99, escortThreat = 0, carrierShots = 0;
    std::array<int, 3> ownCells{};
    int owned = 0;
    for (int k = 0; k < 3; k++) {
        const int id = 3 * team + k;
        const Unit& u = s.units[id];
        const int p = v.origin[id];
        int16_t* f = unit + k * kVU;
        if (p < 0) {
            f[3] = clip(u.respawnAt >= 0 ? u.respawnAt - s.turn : 0, kValueUnitSpec[3]);
            continue;
        }
        ownCells[owned++] = p;
        const detail::UnitFacts uf = detail::unitFacts(b, s, v, id);
        const int vals[kVU] = {1, v.immune[id] ? 100 : u.hp, u.flag >= 0, 0, detail::capped(b.home[team][p]),
                               detail::capped(v.flagDist[p]), uf.nearestEnemy, uf.threatenedBy, uf.enemiesR3};
        for (int i = 0; i < kVU; i++) f[i] = clip(vals[i], kValueUnitSpec[i]);
        if (u.flag >= 0) {
            ownCarrierHome = std::min(ownCarrierHome, detail::capped(b.home[team][p]));
            escortThreat += uf.enemiesR3;
        }
    }
    int enemyCarriers = 0, enemyCarrierHome = 99, nearBase = 0, r6 = 0;
    for (const Unit& w : s.units) {
        const int r = v.origin[w.id];
        if (w.team == team || r < 0) continue;
        if (w.flag >= 0) {
            enemyCarriers++;
            enemyCarrierHome = std::min(enemyCarrierHome, detail::capped(b.home[w.team][r]));
            for (int k = 0; k < 3; k++) {
                const int ally = 3 * team + k;
                if (detail::shooter(s, v, ally) && b.canAttack(v.origin[ally], r)) carrierShots++;
            }
        }
        nearBase += b.man(r, b.baseCell[team]) <= 5;
        bool close = false;
        for (int i = 0; i < owned; i++) close = close || b.man(r, ownCells[i]) <= 6;
        r6 += close;
    }
    int droppedNear = 0;
    for (const Flag& f : s.flags)
        if (f.status == kDropped && available(f, s.turn)) {
            bool close = false;
            for (int i = 0; i < owned; i++) close = close || b.man(b.cell(f.x, f.y), ownCells[i]) <= 5;
            droppedNear += close;
        }
    int tied = 0, ahead = 0, behind = 0, below = 0;
    for (int t = 0; t < b.teams; t++) {
        if (t == team) continue;
        const int d = s.score[t] - s.score[team];
        tied += d == 0;
        ahead += d >= 1 && d <= 2;
        behind += d <= -1 && d >= -2;
        below += d < 0;
    }
    const int rivals = std::max(b.teams - 1, 1);
    const int pairwiseNow = (100 * (2 * below + tied) + rivals) / (2 * rivals);
    const int agg[kVA] = {enemyCarriers, enemyCarrierHome, nearBase, escortThreat, ownCarrierHome, droppedNear, r6,
                          carrierShots,  tied,             ahead,    behind,       pairwiseNow};
    for (int i = 0; i < kVA; i++) out[kG + 3 * kVU + i] = clip(agg[i], kValueAggSpec[i]);
}

}  // namespace fusion
#endif  // FUSION_FEATURES_HPP
