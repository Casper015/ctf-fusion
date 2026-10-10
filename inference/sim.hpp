// Standalone forward simulator: one engine settlement (engine Game.step,
// phases 1-6) on fusion::State, for any joint orders of all units.
//
// This is the multi-team planner's `stepExact` (AI_Test nextgen master.cpp)
// decoupled from planner state: no navigation members, no cached respawn
// predictions, no save/restore. It needs only the Board and the rules from
// INIT, so search, trajectory generation and off-policy branching can share
// it. tools/cpp/sim_check.cpp replays recorded matches through it and compares
// every next state with what the engine actually produced.
//
// The one thing the public state cannot know is where a returning flag
// lands (engine phase 1 draws a random free spot from a hidden seed). Such a
// flag comes back `home` at (-1, -1) and `unknownReturns` counts it; a caller
// that knows the spot (trajectory generation) passes it in `returnSpot`.
//
// Allocation: `Workspace` holds every buffer; after its first use for a
// board, step() does not touch the heap.
#ifndef FUSION_SIM_HPP
#define FUSION_SIM_HPP
#include <array>
#include <vector>

#include "state.hpp"

namespace fusion {

enum OrderAct { kOrderWait = 0, kOrderPickup = 1, kOrderDrop = 2, kOrderAttack = 3 };

struct Order { int move = 0, act = kOrderWait, target = -1; };  // move indexes kDX/kDY (S U D L R)

struct Rules { int hp = 100, damage = 34, respawn = 10, flagReturn = 15, flagCooldown = 8; };
inline Rules rulesOf(const Static& s) { return {s.hp, s.damage, s.respawn, s.flagReturn, s.flagCooldown}; }

struct StepEvents {
    int unknownReturns = 0;                       // flags that returned to a spot the state cannot know
    int deaths = 0, pickups = 0, drops = 0, captures = 0, respawns = 0;
};

struct Workspace {
    std::vector<int> origin, start, want, owner, count, damage, hits;
    std::vector<unsigned char> immune, occupied, blocked, live;
};

// next = state after `s` settles with `orders` (indexed by unit id; units
// without an order wait). `returnSpot[f]`, if given, is where flag f lands
// should it return this turn ({-1,-1} = unknown).
inline void step(const Board& b, const Rules& r, const State& s, const Order* orders, State& next, Workspace& w,
                 StepEvents* ev = nullptr, const std::array<int, 2>* returnSpot = nullptr) {
    StepEvents local;
    StepEvents& e = ev ? *ev : local;
    e = StepEvents{};
    const int U = static_cast<int>(s.units.size()), N = b.n * b.n, t = s.turn;
    next.turn = t + 1;
    next.score = s.score;
    next.units = s.units;
    next.flags = s.flags;
    next.attacks.clear();
    next.attacks.reserve(s.units.size());  // allocates only the first time

    // 1. Flag returns, then respawns (immune for this settlement).
    for (Flag& f : next.flags)
        if ((f.status == kDropped || f.status == kCooldown) && t >= f.returnAt) {
            const bool known = returnSpot && returnSpot[f.id][0] >= 0;
            f.status = kHome;
            f.holder = -1;
            f.returnAt = -1;
            f.x = known ? returnSpot[f.id][0] : -1;
            f.y = known ? returnSpot[f.id][1] : -1;
            if (!known) e.unknownReturns++;
        }
    respawnOrigins(b, s, w.origin, w.immune, w.occupied);
    for (Unit& u : next.units)
        if (u.x < 0 && w.origin[u.id] >= 0) {
            u.x = w.origin[u.id] % b.n;
            u.y = w.origin[u.id] / b.n;
            u.hp = r.hp;
            u.respawnAt = -1;
            e.respawns++;
        }

    // 2. Simultaneous movement (engine: shared destinations block, cross-team
    // head-on swaps block, moves into a staying or blocked unit's cell block,
    // to a fixed point).
    w.start.assign(U, -1);
    w.want.assign(U, -1);
    w.owner.assign(N, -1);
    w.count.assign(N, 0);
    w.blocked.assign(U, 0);
    w.live.assign(U, 0);
    for (const Unit& u : next.units)
        if (u.x >= 0) {
            const int c = b.cell(u.x, u.y), m = orders ? orders[u.id].move : 0;
            w.live[u.id] = 1;
            w.start[u.id] = c;
            w.owner[c] = u.id;
            const int q = (m >= 0 && m < kMoves) ? b.nb[c][m] : c;
            w.want[u.id] = q < 0 ? c : q;
            w.count[w.want[u.id]]++;
        }
    for (int i = 0; i < U; i++) {
        if (!w.live[i]) continue;
        if (w.count[w.want[i]] > 1) w.blocked[i] = 1;
        const int j = w.owner[w.want[i]];
        if (j >= 0 && j != i && w.want[j] == w.start[i] && s.units[i].team != s.units[j].team) w.blocked[i] = w.blocked[j] = 1;
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (int i = 0; i < U; i++) {
            if (!w.live[i] || w.blocked[i] || w.want[i] == w.start[i]) continue;
            const int j = w.owner[w.want[i]];
            if (j >= 0 && (w.blocked[j] || w.want[j] == w.start[j])) { w.blocked[i] = 1; changed = true; }
        }
    }
    for (Unit& u : next.units)
        if (w.live[u.id]) {
            const int c = w.blocked[u.id] ? w.start[u.id] : w.want[u.id];
            u.x = c % b.n;
            u.y = c / b.n;
        }

    // 3. Simultaneous attacks (pre-pickup flag state, post-move cells), then deaths.
    w.damage.assign(U, 0);
    w.hits.assign(U, 0);
    for (int i = 0; i < U; i++) {
        if (!w.live[i] || !orders || orders[i].act != kOrderAttack) continue;
        const Unit& a = next.units[i];
        const int tg = orders[i].target;
        if (a.flag >= 0 || w.immune[i] || tg < 0 || tg >= U) continue;
        const Unit& v = next.units[tg];
        if (v.x < 0 || v.team == a.team || w.immune[tg]) continue;
        if (!b.canAttack(b.cell(a.x, a.y), b.cell(v.x, v.y))) continue;
        w.hits[tg]++;
        next.attacks.push_back({i, tg});
    }
    for (Unit& u : next.units) {
        if (!w.live[u.id] || w.hits[u.id] == 0) continue;
        u.hp -= r.damage * w.hits[u.id];
        if (u.hp > 0) continue;
        e.deaths++;
        if (u.flag >= 0) {
            Flag& f = next.flags[u.flag];
            f.status = kDropped;
            f.x = u.x;
            f.y = u.y;
            f.holder = -1;
            f.returnAt = t + r.flagReturn;
        }
        u.x = u.y = -1;
        u.hp = 0;
        u.flag = -1;
        u.respawnAt = t + r.respawn;
    }

    // 4. Pickups, ascending unit id (lowest id wins a contested flag).
    for (Unit& u : next.units) {
        if (u.x < 0 || u.flag >= 0 || !orders || orders[u.id].act != kOrderPickup) continue;
        for (Flag& f : next.flags)
            if ((f.status == kHome || f.status == kDropped) && f.x == u.x && f.y == u.y) {
                u.flag = f.id;
                f.status = kCarried;
                f.x = f.y = -1;
                f.holder = u.id;
                f.returnAt = -1;
                e.pickups++;
                break;
            }
    }
    // 5. Voluntary drops.
    for (Unit& u : next.units) {
        if (u.x < 0 || u.flag < 0 || !orders || orders[u.id].act != kOrderDrop) continue;
        Flag& f = next.flags[u.flag];
        f.status = kDropped;
        f.x = u.x;
        f.y = u.y;
        f.holder = -1;
        f.returnAt = t + r.flagReturn;
        u.flag = -1;
        e.drops++;
    }
    // 6. Captures on any cell of the carrier's own base.
    for (Unit& u : next.units) {
        if (u.x < 0 || u.flag < 0 || b.baseOf[b.cell(u.x, u.y)] != u.team) continue;
        Flag& f = next.flags[u.flag];
        f.status = kCooldown;
        f.x = f.y = -1;
        f.holder = -1;
        f.returnAt = t + r.flagCooldown;
        u.flag = -1;
        next.score[u.team]++;
        e.captures++;
    }
}

}  // namespace fusion
#endif  // FUSION_SIM_HPP
