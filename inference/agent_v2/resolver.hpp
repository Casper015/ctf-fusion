// The mathematical tactical resolver of the agent_v2 carrier: rule assign_v1 (task T0065, docs/AGENT_V2.md).
//
// Given our team's movement joint and a table of opponent scenarios, it returns complete own orders. The rule is
// declared here and tested against itself (an unpruned brute force over every assignment) and against the engine's
// rules; it is not the Meowmind resolver. It is exact for its declared scenario score, not for winning the game.
//
// Scenarios. A scenario is one complete set of opponent orders (movement from our P's TEAM JOINT distribution, then
// its conditional action and target heads; PredictorP::scenarios), with weight 1/K. Our P's movement marginals alone
// cannot say where units end up, because the engine blocks shared destinations, opposing swaps and chains of moves.
// For each scenario the engine's own movement (fusion::step, inference/sim.hpp) resolves every unit's cell after
// movement (all units waiting: phases 1-2 only) and, separately, the opponents' attacks, deaths and our survival (our
// units waiting, the opponents' actions as drawn: phases 1-6). Attacks are simultaneous, so our own attacks change
// neither where anyone ends nor whether an opponent's shot lands.
//
// Rules taken from the engine: respawned units are immune this settlement (they neither attack nor can be hit); a
// carrier cannot attack; only enemies can be attacked; attacks use the cells after movement and Board::canAttack
// (range 2, walls block straight shots, a diagonal needs one open corner); pickups follow attacks and need a unit that
// did not attack; capture follows pickup in the same settlement, so a pickup on our own base scores immediately.
//
// Declared score of an assignment (each able unit attacks at most one enemy, or none):
//     sum over targeted enemies e of  sum_s p_s * ( hit * min(h_s, need) + [h_s >= need] * (carrier(e) ? carrier_kill : kill) )
//     - sum over attacking units k of C_k
//   h_s: our units assigned to e whose cell after movement in scenario s can attack e's cell after movement;
//   need = ceil(hp / damage); C_k = sum_s p_s * [k survives the opponents' attacks in s and an available flag lies on
//   its cell] * (pickup_capture on our base, pickup elsewhere): what attacking forgoes.
// Able attackers: alive at the settlement's origin, not immune, not carrying. Targets: alive, not immune, other teams.
// Candidates: an enemy is a candidate of unit k only if it lies within Manhattan distance 3 of k's origin or k's
// wall-clamped destination (k fires from one of those two cells, and an enemy moves at most one step); any other
// target scores exactly 0 for every scenario. Chosen: the highest score; ties: fewer attackers, then the
// lexicographically smallest target tuple (unit 0, 1, 2; no attack = -1). Binary64, fixed accumulation order
// (scenarios in order, enemies in id order), exact comparisons.
// Complete orders: an attacker attacks; any other living non-carrier asks to pick up (no effect without a flag); a
// carrier waits. assign_v1 never drops a flag. The profile's numbers are tunable settings, not rules; dp_v1 starts
// from the Meowmind resolver's values.
#pragma once
#include <array>
#include <cstdlib>
#include <vector>

#include "core.hpp"
#include "simulate.hpp"

namespace agent_v2 {

struct ResolverProfile {
    const char* name = "dp_v1";
    double hit = .045, kill = .20, carrier_kill = .65, pickup_capture = 1.2, pickup = .32;
};

using Scenario = std::vector<fusion::Order>;
using Marginals = std::vector<std::array<double, kMoves>>;   // P(unit moves m) per unit id (0 for our own team)   // complete orders of every unit (ours ignored: replaced by waits)

struct Tactics {
    std::array<int, 3> target{{-1, -1, -1}};   // enemy unit id, or -1
    std::array<fusion::Order, 3> orders{};     // complete orders of units 3 * team + k
    double score = 0;
};

// Per scenario: every unit's cell after movement, and whether each of our units survives the opponents' attacks.
struct ScenarioCells {
    std::vector<int> cell;                     // -1: not on the board
    std::array<bool, 3> survives{};
    std::array<bool, 3> flag_here{};           // an available flag lies on our unit's cell after the settlement's returns
};

struct Resolver {
    ResolverProfile profile;
    Simulator sim;

    ScenarioCells cells(const MapCache& c, const fusion::State& s, int team, const std::array<int, 3>& moves,
                        const Scenario& opp, uint64_t refresh_seed) {
        const int U = int(s.units.size());
        ScenarioCells out;
        std::vector<fusion::Order> orders(U);
        for (int i = 0; i < U; i++) orders[i] = {opp[i].move, fusion::kOrderWait, -1};
        for (int k = 0; k < 3; k++) orders[3 * team + k] = {moves[k], fusion::kOrderWait, -1};
        const fusion::State moved = sim.step(c, s, orders, refresh_seed);          // movement only: nobody acts
        out.cell.assign(U, -1);
        for (auto& u : moved.units) out.cell[u.id] = u.x < 0 ? -1 : c.board.cell(u.x, u.y);
        for (int i = 0; i < U; i++)
            if (i / 3 != team) orders[i] = opp[i];
        const fusion::State shot = sim.step(c, s, orders, refresh_seed);           // the opponents' actions as drawn
        for (int k = 0; k < 3; k++) {
            const auto& u = shot.units[3 * team + k];
            out.survives[k] = u.x >= 0 && out.cell[3 * team + k] >= 0;
            if (!out.survives[k]) continue;
            for (auto& f : shot.flags)
                if ((f.status == fusion::kHome || f.status == fusion::kDropped) && f.x == u.x && f.y == u.y) out.flag_here[k] = true;
        }
        return out;
    }

    struct Table {
        std::vector<ScenarioCells> scen;
        std::array<double, 3> forgo{};               // C_k
        std::array<std::vector<int>, 3> cand;
        std::array<bool, 3> able{};
        double weight = 0;
    };

    Table table(const MapCache& c, const fusion::State& s, const fusion::TurnView& v, int team, const std::array<int, 3>& moves,
                const std::vector<Scenario>& opp, uint64_t refresh_seed, bool prune = true) {
        Table t;
        const int U = int(s.units.size());
        t.weight = opp.empty() ? 0.0 : 1.0 / double(opp.size());
        for (size_t i = 0; i < opp.size(); i++) t.scen.push_back(cells(c, s, team, moves, opp[i], refresh_seed + i));
        for (int k = 0; k < 3; k++) {
            const int uid = 3 * team + k;
            for (auto& sc : t.scen)
                if (sc.flag_here[k])
                    t.forgo[k] += t.weight * (c.board.baseOf[sc.cell[uid]] == team ? profile.pickup_capture : profile.pickup);
            t.able[k] = v.origin[uid] >= 0 && !v.immune[uid] && s.units[uid].flag < 0;
            if (!t.able[k]) continue;
            const int o = v.origin[uid], q = c.board.nb[o][moves[k]], d = q < 0 ? o : q;
            for (int e = 0; e < U; e++) {
                if (e / 3 == team || v.origin[e] < 0 || v.immune[e]) continue;
                const int eo = v.origin[e];
                if (!prune || c.board.man(o, eo) <= 3 || c.board.man(d, eo) <= 3) t.cand[k].push_back(e);
            }
        }
        return t;
    }

    double value(const MapCache& c, const fusion::State& s, const Table& t, int e, int team, int mask) const {
        const int need = (s.units[e].hp + c.rules.damage - 1) / c.rules.damage;
        const double kill = s.units[e].flag >= 0 ? profile.carrier_kill : profile.kill;
        double total = 0;
        for (auto& sc : t.scen) {
            const int z = sc.cell[e];
            int h = 0;
            for (int k = 0; k < 3; k++)
                if ((mask >> k & 1) && c.board.canAttack(sc.cell[3 * team + k], z)) h++;
            total += t.weight * (profile.hit * std::min(h, need) + (h >= need ? kill : 0.0));
        }
        return total;
    }

    double score(const MapCache& c, const fusion::State& s, const Table& t, int team, const std::array<int, 3>& target) const {
        double total = 0;
        for (int e = 0; e < int(s.units.size()); e++) {
            int mask = 0;
            for (int k = 0; k < 3; k++)
                if (target[k] == e) mask |= 1 << k;
            if (mask) total += value(c, s, t, e, team, mask);
        }
        for (int k = 0; k < 3; k++)
            if (target[k] >= 0) total -= t.forgo[k];
        return total;
    }

    static bool tie_before(const std::array<int, 3>& a, const std::array<int, 3>& b) {
        int na = 0, nb = 0;
        for (int k = 0; k < 3; k++) na += a[k] >= 0, nb += b[k] >= 0;
        if (na != nb) return na < nb;
        return a < b;
    }

    Tactics resolve(const MapCache& c, const fusion::State& s, const fusion::TurnView& v, int team, const std::array<int, 3>& moves,
                    const std::vector<Scenario>& opp, uint64_t refresh_seed, bool prune = true) {
        const Table t = table(c, s, v, team, moves, opp, refresh_seed, prune);
        std::array<std::vector<int>, 3> opts;
        for (int k = 0; k < 3; k++) {
            opts[k].push_back(-1);
            for (int e : t.cand[k]) opts[k].push_back(e);
        }
        Tactics best;
        bool have = false;
        for (int a : opts[0])
            for (int b1 : opts[1])
                for (int b2 : opts[2]) {
                    const std::array<int, 3> tg{a, b1, b2};
                    const double sc = score(c, s, t, team, tg);
                    if (!have || sc > best.score || (sc == best.score && tie_before(tg, best.target)))
                        best.score = sc, best.target = tg, have = true;
                }
        for (int k = 0; k < 3; k++) {
            const int uid = 3 * team + k;
            auto& o = best.orders[k];
            o.move = moves[k];
            o.target = -1;
            if (best.target[k] >= 0) o.act = fusion::kOrderAttack, o.target = best.target[k];
            else if (s.units[uid].flag >= 0) o.act = fusion::kOrderWait;
            else o.act = fusion::kOrderPickup;
        }
        return best;
    }
};

}  // namespace agent_v2
