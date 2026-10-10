// Public game state and static board geometry shared by the feature encoders
// and the simulator.
//
// Pure data and geometry: no I/O, standard library only, so a bot can include
// it next to its own parser (tools/cpp/stream.hpp is the parser the tools use).
// Every rule mirrored here comes from the official engine
// (ctf-arena/Player7/workspace/engine.py); comments name the engine function.
//
// Allocation: Board::init allocates once (turn 1, inside the 2 s budget).
// makeView() reuses a TurnView's buffers, so after the first turn of a match
// nothing on the per-turn path touches the heap.
#ifndef FUSION_STATE_HPP
#define FUSION_STATE_HPP
#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

namespace fusion {

inline constexpr int kMoves = 5;  // protocol letters in this order
inline constexpr int kDX[kMoves] = {0, 0, 0, -1, 1};
inline constexpr int kDY[kMoves] = {0, -1, 1, 0, 0};
inline constexpr char kMoveChar[kMoves] = {'S', 'U', 'D', 'L', 'R'};
inline constexpr int kUnreachable = 1 << 20;
inline constexpr int kMaxTeams = 15;
inline constexpr int kMaxUnits = 3 * kMaxTeams;
inline constexpr int kMaxFlags = 8;  // max(2, ceil(15 / 2))

enum FlagStatus { kHome = 0, kCarried = 1, kDropped = 2, kCooldown = 3 };

struct Unit { int id = -1, team = -1, x = -1, y = -1, hp = 0, flag = -1, respawnAt = -1; };
struct Flag { int id = -1, status = kHome, x = -1, y = -1, holder = -1, returnAt = -1; };
struct Attack { int unit = -1, target = -1; };

// INIT block.
struct Static {
    int teams = 0, me = -1, size = 0, turns = 400, hp = 100, damage = 34, respawn = 10, flagReturn = 15, flagCooldown = 8;
    std::vector<unsigned char> wall;  // size*size, index y*size+x
    std::vector<std::array<int, 2>> baseCenter, spots;
};

// One TURN block: the public state before turn `turn` settles.
struct State {
    int turn = 0;
    std::vector<int> score;
    std::vector<Unit> units;      // ascending id
    std::vector<Flag> flags;      // ascending id
    std::vector<Attack> attacks;  // attack events of the previous settlement
};

struct Board {
    int n = 0, teams = 0, turns = 400, center = 0;
    std::vector<unsigned char> wall;
    std::vector<std::array<int, kMoves>> nb;  // destination per move, -1 = wall
    std::vector<std::vector<int>> home;       // BFS distance to each team's base cells
    std::vector<std::array<int, 9>> spawnOrder;
    std::vector<int> baseOf;                  // owning team of a base cell, -1 elsewhere
    std::vector<int> baseCell;                // base center cell per team
    mutable std::vector<int> queue;           // BFS scratch, n*n

    int cell(int x, int y) const { return y * n + x; }
    int man(int a, int b) const { return std::abs(a % n - b % n) + std::abs(a / n - b / n); }
    bool open(int c) const { return c >= 0 && !wall[c]; }

    // engine Game.can_attack: range 2, a wall between blocks straight shots,
    // a diagonal shot needs one open orthogonal corner.
    bool canAttack(int a, int b) const {
        if (a < 0 || b < 0 || man(a, b) > 2) return false;
        const int x = a % n, y = a / n, dx = b % n - x, dy = b / n - y;
        if (std::abs(dx) == 2 && dy == 0) return !wall[cell(x + dx / 2, y)];
        if (std::abs(dy) == 2 && dx == 0) return !wall[cell(x, y + dy / 2)];
        if (std::abs(dx) == 1 && std::abs(dy) == 1) return !wall[cell(x + dx, y)] || !wall[cell(x, y + dy)];
        return true;
    }

    // Multi-source BFS over open cells into `dist` (resized to n*n; no allocation once sized).
    void bfs(const int* sources, int count, std::vector<int>& dist) const {
        dist.assign(n * n, kUnreachable);
        int head = 0, tail = 0;
        for (int i = 0; i < count; i++) {
            const int s = sources[i];
            if (open(s) && dist[s] != 0) { dist[s] = 0; queue[tail++] = s; }
        }
        while (head < tail) {
            const int c = queue[head++];
            for (int m = 1; m < kMoves; m++) {
                const int t = nb[c][m];
                if (t >= 0 && dist[t] > dist[c] + 1) { dist[t] = dist[c] + 1; queue[tail++] = t; }
            }
        }
    }

    void init(const Static& s) {
        n = s.size;
        teams = s.teams;
        turns = s.turns;
        wall = s.wall;
        center = cell(n / 2, n / 2);
        queue.assign(n * n, 0);
        nb.assign(n * n, {});
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
                for (int m = 0; m < kMoves; m++) {
                    const int px = x + kDX[m], py = y + kDY[m];
                    const bool in = px >= 0 && py >= 0 && px < n && py < n;
                    nb[cell(x, y)][m] = in && !wall[cell(px, py)] ? cell(px, py) : -1;
                }
        baseOf.assign(n * n, -1);
        home.assign(teams, {});
        spawnOrder.assign(teams, {});
        baseCell.assign(teams, -1);
        for (int t = 0; t < teams; t++) {
            const int bx = s.baseCenter[t][0], by = s.baseCenter[t][1];
            baseCell[t] = cell(bx, by);
            std::array<int, 9> order{};
            int k = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) order[k++] = cell(bx + dx, by + dy);
            for (int c : order) baseOf[c] = t;
            bfs(order.data(), 9, home[t]);
            // engine Game.__init__ spawn_order: stable sort by
            // (distance to map center, |dx| + 2|dy| from the base center).
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                const int ka = man(a, center), kb = man(b, center);
                if (ka != kb) return ka < kb;
                return std::abs(a % n - bx) + 2 * std::abs(a / n - by) < std::abs(b % n - bx) + 2 * std::abs(b / n - by);
            });
            spawnOrder[t] = order;
        }
    }
};

// Facts about one state that every team's features share.
struct TurnView {
    std::vector<int> origin;            // cell each unit moves from this turn, -1 = inactive
    std::vector<unsigned char> immune;  // respawns during this settlement: cannot shoot or be shot
    std::vector<int> prevCell;          // cell in the previous block, -1 if unknown or dead
    std::vector<unsigned char> flagCell;
    std::vector<int> flagDist;          // BFS distance to the nearest available flag
    std::vector<int> occupant;          // unit standing on a cell, -1 if none
    std::vector<int> hitsTaken;         // attack events on each unit in the previous settlement
    std::vector<unsigned char> occupied;  // scratch
};

// A flag an unladen unit can pick up this turn: home, or dropped and not
// returning in this settlement's phase 1 (engine Game.step, "Flag returns").
inline bool available(const Flag& f, int turn) {
    return f.status == kHome || (f.status == kDropped && turn < f.returnAt);
}

// engine Game.step phase 1: dead units whose respawn time has come take the
// first free cell of their team's spawn order, in ascending unit id. Fills
// origin (current cell, spawn cell, or -1) and immune.
inline void respawnOrigins(const Board& b, const State& s, std::vector<int>& origin, std::vector<unsigned char>& immune,
                           std::vector<unsigned char>& occupied) {
    const int U = static_cast<int>(s.units.size());
    origin.assign(U, -1);
    immune.assign(U, 0);
    occupied.assign(b.n * b.n, 0);
    for (const Unit& u : s.units)
        if (u.x >= 0) {
            origin[u.id] = b.cell(u.x, u.y);
            occupied[origin[u.id]] = 1;
        }
    for (const Unit& u : s.units)
        if (u.x < 0 && u.respawnAt >= 0 && s.turn >= u.respawnAt)
            for (int c : b.spawnOrder[u.team])
                if (!occupied[c]) {
                    origin[u.id] = c;
                    immune[u.id] = 1;
                    occupied[c] = 1;
                    break;
                }
}

inline void makeView(const Board& b, const State& s, const State* prev, TurnView& v) {
    const int U = static_cast<int>(s.units.size());
    respawnOrigins(b, s, v.origin, v.immune, v.occupied);
    v.prevCell.assign(U, -1);
    v.hitsTaken.assign(U, 0);
    v.occupant.assign(b.n * b.n, -1);
    for (int i = 0; i < U; i++)
        if (v.origin[i] >= 0) v.occupant[v.origin[i]] = i;
    if (prev && prev->units.size() == s.units.size())
        for (const Unit& u : prev->units)
            if (u.x >= 0) v.prevCell[u.id] = b.cell(u.x, u.y);
    for (const Attack& a : s.attacks)
        if (a.target >= 0 && a.target < U) v.hitsTaken[a.target]++;
    v.flagCell.assign(b.n * b.n, 0);
    std::array<int, kMaxFlags> sources{};
    int count = 0;
    for (const Flag& f : s.flags)
        if (available(f, s.turn) && f.x >= 0 && count < kMaxFlags) {
            const int c = b.cell(f.x, f.y);
            v.flagCell[c] = 1;
            sources[count++] = c;
        }
    b.bfs(sources.data(), count, v.flagDist);
}

}  // namespace fusion
#endif  // FUSION_STATE_HPP
