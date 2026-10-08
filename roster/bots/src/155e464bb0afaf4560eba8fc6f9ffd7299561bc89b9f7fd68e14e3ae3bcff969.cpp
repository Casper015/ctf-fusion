// Multi-Faction CTF bot -- workspace/submission/bot.cpp
//
// Final strategy
// --------------
// Turn 1: BFS walk-distance fields from every flag spot and every base centre, plus a
//         per-cell "best step toward X" field. Everything is precomputed here because
//         turn 1 has a 2000 ms budget while every later turn has only 50 ms.
//
// Every turn, in order:
//   1. GOAL ASSIGNMENT. Each of the 3 units gets exactly one objective:
//        - a unit carrying a flag goes HOME (capture is automatic; it never drops in base,
//          because drop is phase 5 and capture phase 6, so X at home forfeits the point);
//        - otherwise it claims a ground flag, ranked by spotDist + homeDist, i.e. cheap to
//          reach AND cheap to carry home (a flag near our own base is worth far more), with
//          one unit per flag when several are loose;
//        - with no loose flag it waits on the spot that minimises the same two-part cost,
//          so the next spawn lands next to us (a unit already standing on a spot takes a
//          flag that spawns under it in the same turn: return is phase 1, pickup phase 4);
//        - the nearest free unit escorts our carrier. The baseline's real edge is carrier
//          kills (12 per match vs ~4), so denying that is worth more than more logistics.
//   2. JOINT MOVE SEARCH. Enumerate the (up to 6)^3 (move, attack) combinations, replicate
//      the engine's own movement resolution (contested destination blocks everyone, the
//      chase/block fixpoint, opposing swaps bounce, same-faction swaps are legal), and
//      score each joint outcome: progress toward the goal, +2500 for landing on a loose
//      flag, attack value (carrier first, then lethal blows), a danger term for a carrier
//      that is hunted, an anti-oscillation toll on recently used cells, and support for a
//      bodyguard closing on whoever is near our carrier. Cells held by an enemy that is
//      expected to stay are rejected outright.
//   3. ACTIONS. Pickup only when a loose flag is actually on our cell; attack only a target
//      the engine will accept (post-movement range <= 2 with wall line-of-sight); never
//      emit an action we cannot execute. Exactly 7 tokens are emitted per turn.
//
// Engine facts this file relies on (workspace/engine.py):
//   MOVES: U=(0,-1) D=(0,1) L=(-1,0) R=(1,0) S=STAY (protocol 'S' is never "south").
//   Walls block; a contested destination blocks all movers; entering a cell occupied at
//   turn start is blocked unless that occupant leaves; opposing swaps bounce.
//   Attack range 2 with wall line-of-sight; 34 damage, 3 hits kill; a flag carrier cannot
//   attack; a respawning unit is immune and still may move and capture.
//   Resolution order: returns/respawns, move, attack, pickup, drop, capture.
//
// The bot is a pure function of the observed state (no RNG, no file or network access),
// which makes it reproducible and diff-testable.

#include <array>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

static const int INF = 1 << 26;
static const int BIG = 1000000;
static const int K_FREE = 1000;   // value of standing next to / holding a flag
static const int MOVER_SCALE = 40;  // per-step value of closing on a free flag
static const int HOME_SCALE = 80;   // per-step value of carrying a flag home
static const int HUNT_SCALE = 30;   // per-step value of closing on a victim
static const int CARRY_HOME_VALUE = 1000;

struct Unit {
    int id = 0, team = 0, x = -1, y = -1, hp = 100, flag = -1, respawn_at = -1;
};
struct Flag {
    int id = 0, status = 0, x = -1, y = -1, holder = -1, return_at = -1;  // status: 0 home, 1 dropped, 2 carried, 3 cooldown
};

struct Bot {
    // ---- static ----
    int teams = 0, me = 0, size = 0, turns = 0, hp0 = 100, dmg = 34, respawnN = 10, flagReturn = 15, flagCool = 8;
    vector<string> map_;
    vector<pair<int, int>> bases;   // base centres
    vector<pair<int, int>> spots;   // flag spots

    // ---- turn state ----
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;

    // ---- precomputed ----
    bool init_done = false;
    vector<vector<int>> dSpot;   // BFS distance field per flag spot
    vector<vector<int>> dHome;   // BFS distance field per base centre (idx = team)
    vector<vector<unsigned char>> spotStep;  // best direction toward each spot
    vector<vector<unsigned char>> homeStep;  // best direction toward each base
    int hist[3][8] = {{0}};      // recently chosen cells per own unit (anti-oscillation)
    int histN = 0;

    inline bool wall(int x, int y) const {
        return x < 0 || y < 0 || x >= size || y >= size || map_[y][x] == '#';
    }

    static const int DX[4], DY[4];
    static const char MOVECH[4];

    vector<int> bfs(int sx, int sy) const {
        vector<int> d((size_t)size * size, INF);
        if (wall(sx, sy)) return d;
        vector<int> q;
        q.reserve((size_t)size * size);
        d[(size_t)sy * size + sx] = 0;
        q.push_back(sy * size + sx);
        for (size_t h = 0; h < q.size(); ++h) {
            int c = q[h], cx = c % size, cy = c / size, nc = d[c] + 1;
            for (int k = 0; k < 4; ++k) {
                int nx = cx + DX[k], ny = cy + DY[k];
                if (wall(nx, ny)) continue;
                int ni = ny * size + nx;
                if (d[ni] > nc) { d[ni] = nc; q.push_back(ni); }
            }
        }
        return d;
    }

    // For every cell, which move gets closer (or stays) relative to a BFS field.
    vector<unsigned char> stepField(const vector<int>& d) const {
        vector<unsigned char> st((size_t)size * size, 4);  // 4 == stay
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                if (wall(x, y)) continue;
                int cur = d[(size_t)y * size + x];
                if (cur <= 0) continue;
                unsigned char b = 4;
                int bv = cur;
                for (int k = 0; k < 4; ++k) {
                    int nx = x + DX[k], ny = y + DY[k];
                    if (wall(nx, ny)) continue;
                    int v = d[(size_t)ny * size + nx];
                    if (v < bv) { bv = v; b = (unsigned char)k; }
                }
                st[(size_t)y * size + x] = b;
            }
        }
        return st;
    }

    // ---------------- input ----------------
    bool readTurn() {
        string w;
        while (cin >> w) {
            if (w == "INIT") {
                cin >> teams >> me >> size >> turns >> hp0 >> dmg >> respawnN >> flagReturn >> flagCool;
            } else if (w == "MAP") {
                map_.assign(size, string());
                for (auto& r : map_) cin >> r;
            } else if (w == "BASES") {
                int k; cin >> k; bases.resize(k);
                for (auto& b : bases) cin >> b.first >> b.second;
            } else if (w == "SPOTS") {
                int k; cin >> k; spots.resize(k);
                for (auto& s : spots) cin >> s.first >> s.second;
            } else if (w == "TURN") {
                cin >> turn;
            } else if (w == "SCORE") {
                score.resize(teams);
                for (auto& s : score) cin >> s;
            } else if (w == "UNITS") {
                int k; cin >> k; units.assign(k, Unit());
                for (auto& u : units) { string tag; cin >> tag >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawn_at; }
            } else if (w == "FLAGS") {
                int k; cin >> k; flags.assign(k, Flag());
                for (auto& f : flags) {
                    string tag, st;
                    cin >> tag >> f.id >> st >> f.x >> f.y >> f.holder >> f.return_at;
                    f.status = (st == "home") ? 0 : (st == "dropped") ? 1 : (st == "carried") ? 2 : 3;
                }
            } else if (w == "EVENTS") {
                int k; cin >> k;
                string line;
                getline(cin, line);
                for (int i = 0; i < k; ++i) getline(cin, line);  // not needed by the policy
            } else if (w == "END") {
                init_done = true;
                return true;
            }
        }
        return false;
    }

    void precompute() {
        dSpot.resize(spots.size());
        spotStep.resize(spots.size());
        for (size_t i = 0; i < spots.size(); ++i) {
            dSpot[i] = bfs(spots[i].first, spots[i].second);
            spotStep[i] = stepField(dSpot[i]);
        }
        dHome.resize(bases.size());
        homeStep.resize(bases.size());
        for (size_t i = 0; i < bases.size(); ++i) {
            dHome[i] = bfs(bases[i].first, bases[i].second);
            homeStep[i] = stepField(dHome[i]);
        }
    }

    // Dead units carry the sentinel (-1,-1): never let that reach the unsigned index.
    inline bool cellOk(int x, int y) const { return x >= 0 && y >= 0 && x < size && y < size; }
    inline int spotDist(size_t i, int x, int y) const {
        if (i >= dSpot.size() || !cellOk(x, y)) return INF;
        return dSpot[i][(size_t)y * size + x];
    }
    inline int homeDist(int team, int x, int y) const {
        if (team < 0 || team >= (int)dHome.size() || !cellOk(x, y)) return INF;
        return dHome[(size_t)team][(size_t)y * size + x];
    }

    static int manh(int x1, int y1, int x2, int y2) { return abs(x1 - x2) + abs(y1 - y2); }

    bool inBase(int team, int x, int y) const {
        int bx = bases[team].first, by = bases[team].second;
        return abs(x - bx) <= 1 && abs(y - by) <= 1;
    }

    // engine can_attack(), evaluated on post-movement positions
    bool canAttack(int ax, int ay, int bx, int by) const {
        int dx = bx - ax, dy = by - ay;
        if (abs(dx) + abs(dy) > 2) return false;
        if (abs(dx) == 2 && dy == 0) return !wall(ax + dx / 2, ay);
        if (abs(dy) == 2 && dx == 0) return !wall(ax, ay + dy / 2);
        if (abs(dx) == 1 && abs(dy) == 1) return !wall(ax + dx, ay) || !wall(ax, ay + dy);
        return true;
    }

    bool spotFree(int id) const {
        for (const auto& f : flags) if (f.id == id) return f.status == 0 || f.status == 1;
        return false;
    }

    // ---------------- decision ----------------
    // ---------------- decision ----------------
    // Deterministic greedy planner. Each unit gets one goal, picks the move that closes
    // BFS distance to that goal while dodging enemy-occupied cells and our own planned
    // cells, then attacks the best legal target. Rule: never fight the movement engine.
    string decide() {
        if (!init_done) return to_string(turn) + " S - S - S -";
        if (dSpot.empty()) precompute();

        const int nU = (int)units.size();
        int ui[3] = {-1, -1, -1};
        for (int i = 0; i < nU; ++i) {
            int k = units[i].id - 3 * me;
            if (units[i].team == me && k >= 0 && k < 3) ui[k] = i;
        }

        int px[3], py[3], hp3[3], carry3[3];
        bool alive3[3], immune3[3];
        for (int s = 0; s < 3; ++s) {
            if (ui[s] < 0) { alive3[s] = false; carry3[s] = false; immune3[s] = false; px[s] = py[s] = -1; hp3[s] = 0; carry3[s] = 0; continue; }
            const Unit& u = units[ui[s]];
            bool alive = (u.respawn_at < 0 || turn >= u.respawn_at);
            alive3[s] = alive;
            carry3[s] = u.flag >= 0;
            immune3[s] = (u.respawn_at >= 0 && turn >= u.respawn_at);
            px[s] = alive ? u.x : -1;
            py[s] = alive ? u.y : -1;
            hp3[s] = u.hp;
        }

        // ---- world snapshot ----
        struct Foe { int id, x, y, hp, flag, team; };
        vector<Foe> foes;
        for (const auto& u : units) if (u.team != me && u.x >= 0) foes.push_back({u.id, u.x, u.y, u.hp, u.flag, u.team});

        struct Ground { int sidx, sx, sy; };   // sidx = index into spots[]
        vector<Ground> grounds;
        for (const auto& f : flags) {
            if (f.status != 0 && f.status != 1) continue;
            for (size_t i = 0; i < spots.size(); ++i)
                if (spots[i].first == f.x && spots[i].second == f.y) { grounds.push_back({(int)i, f.x, f.y}); break; }
        }

        auto spotIndexAt = [&](int x, int y) {
            for (size_t i = 0; i < spots.size(); ++i) if (spots[i].first == x && spots[i].second == y) return (int)i;
            return -1;
        };

        // ---- goals ----
        // gtype: 0 home, 1 a ground flag (by grounds index), 2 a flag spot (by spots index)
        int gtype[3] = {2, 2, 2}, garg[3] = {0, 0, 0};
        bool gClaimed[64] = {false};
        bool anyGroundLeft = !grounds.empty();

        for (int s = 0; s < 3; ++s) {
            if (!alive3[s]) continue;
            if (carry3[s]) { gtype[s] = 0; continue; }   // carriers can only score: go home
            int bestG = -1, bestV = INT_MAX;
            for (size_t gi = 0; gi < grounds.size(); ++gi) {
                if (!anyGroundLeft) break;
                if (gi < 64 && gClaimed[gi] && grounds.size() > 1) continue;  // spread one unit per flag when several exist
                int d = spotDist((size_t)grounds[gi].sidx, px[s], py[s]);
                if (d >= INF) continue;
                // cost = get there + carry it home; a spot near our own base is worth more
                int cost = d + homeDist(me, grounds[gi].sx, grounds[gi].sy);
                if (cost < bestV) { bestV = cost; bestG = (int)gi; }
            }
            if (bestG >= 0) {
                gtype[s] = 1; garg[s] = bestG;
                if (bestG < 64) gClaimed[bestG] = true;
            } else {
                // no reachable ground flag: wait at the spot that is both cheap for this unit
                // to reach and cheap to carry home from (spots near our base are gold)
                int bestS = -1, bestD = INT_MAX;
                for (size_t i = 0; i < spots.size(); ++i) {
                    int d = spotDist(i, px[s], py[s]);
                    if (d >= INF) continue;
                    int cost = d + homeDist(me, spots[i].first, spots[i].second);
                    if (cost < bestD) { bestD = cost; bestS = (int)i; }
                }
                gtype[s] = 2; garg[s] = (bestS < 0 ? 0 : bestS);
            }
        }
        // carriers never attack; also rescue our own stranded carrier? (no: goal is home)

        // ---- per-unit move selection, in unit order (so earlier picks reserve cells) ----
        const char MV[5] = {'S', 'U', 'D', 'L', 'R'};
        char chosenMv[3] = {'S', 'S', 'S'};
        string chosenAct[3] = {"-", "-", "-"};
        int chosenTgt[3] = {-1, -1, -1};
        bool attackOpt[3] = {false, false, false};
        int atkTarget[3] = {-1, -1, -1};
        auto attackTargetFor = [&](int s, int x, int y) {
            int bt = -1, bv = INT_MIN;
            for (const auto& f : foes) {
                if (!canAttack(x, y, f.x, f.y)) continue;
                int v = (f.flag >= 0) ? 1000 : 0;
                v += (f.hp <= dmg) ? 400 : 0;
                for (int c = 0; c < 3; ++c) {
                    if (!alive3[c] || !carry3[c] || inBase(me, px[c], py[c])) continue;
                    if (manh(f.x, f.y, px[c], py[c]) <= 2) v += 600;
                }
                v -= manh(x, y, f.x, f.y);
                if (v > bv) { bv = v; bt = f.id; }
            }
            return bt;
        };
        for (int s = 0; s < 3; ++s) {
            if (!alive3[s] || carry3[s] || immune3[s]) continue;
            int t = attackTargetFor(s, px[s], py[s]);
            if (t >= 0) { attackOpt[s] = true; atkTarget[s] = t; }
        }
        int finalX[3], finalY[3], newHp[3];
        for (int s = 0; s < 3; ++s) { finalX[s] = px[s]; finalY[s] = py[s]; newHp[s] = hp3[s]; }

        // Escort: if one of our units carries a flag, the nearest free unit becomes its
        // bodyguard. Carrier kills are where the baseline beats us (7-14 per melee game
        // against our 1-2), so denying that is worth more than extra flag logistics.
        int escortOf = -1, escortBy = -1;
        {
            int bd = INT_MAX;
            for (int c = 0; c < 3; ++c) {
                if (!alive3[c] || !carry3[c] || inBase(me, px[c], py[c])) continue;
                for (int s = 0; s < 3; ++s) {
                    if (!alive3[s] || s == c || carry3[s]) continue;
                    int d = manh(px[s], py[s], px[c], py[c]);
                    if (d < bd) { bd = d; escortOf = c; escortBy = s; }
                }
            }
            if (escortOf >= 0 && escortBy >= 0) { gtype[escortBy] = 3; garg[escortBy] = escortOf; }
        }

        auto goalDist = [&](int s, int x, int y) {
            if (x < 0) return INF;
            if (gtype[s] == 0) return homeDist(me, x, y);
            if (gtype[s] == 1 && garg[s] >= 0 && garg[s] < (int)grounds.size())
                return spotDist((size_t)grounds[garg[s]].sidx, x, y);
            if (gtype[s] == 3 && garg[s] >= 0 && garg[s] < 3) {
                int c = garg[s];
                if (!alive3[c]) return INF;
                return manh(x, y, px[c], py[c]);     // stay close to the carrier
            }
            if (garg[s] >= 0 && garg[s] < (int)spots.size()) return spotDist((size_t)garg[s], x, y);
            return INF;
        };

        // predict enemies' single greedy step toward whoever is closest (their likely move)
        auto predictFoe = [&](const Foe& f) {
            int bi = -1, bd = INT_MAX;
            for (int s = 0; s < 3; ++s) {
                if (!alive3[s]) continue;
                int d = manh(f.x, f.y, px[s], py[s]);
                if (d < bd) { bd = d; bi = s; }
            }
            if (bi < 0) return make_pair(f.x, f.y);
            int bx = f.x, by = f.y, bv = bd;
            for (int k = 0; k < 4; ++k) {
                int nx = f.x + DX[k], ny = f.y + DY[k];
                if (wall(nx, ny)) continue;
                int d = manh(nx, ny, px[bi], py[bi]);
                if (d < bv) { bv = d; bx = nx; by = ny; }
            }
            return make_pair(bx, by);
        };
        vector<pair<int, int>> foeNext;
        for (const auto& f : foes) foeNext.push_back(predictFoe(f));

        // ---- joint move search ----
        // The engine cancels moves it considers contested (fixpoint over start-cell
        // ownership). Enumerate the (<=5)^3 move combinations, replicate that resolution
        // for our own units, and keep the best joint outcome. This removes the stalls a
        // purely local step gets when the direct route is temporarily blocked.
        int candX[3][5], candY[3][5];
        int candN[3];
        for (int s = 0; s < 3; ++s) {
            candN[s] = 0;
            if (!alive3[s]) continue;
            for (int m = 0; m < 5; ++m) {
                int nx = px[s], ny = py[s];
                if (m > 0) {
                    int k = m - 1;
                    nx = px[s] + DX[k]; ny = py[s] + DY[k];
                    if (wall(nx, ny)) continue;
                }
                candX[s][candN[s]] = nx; candY[s][candN[s]] = ny; ++candN[s];
            }
            if (candN[s] == 0) { candX[s][0] = px[s]; candY[s][0] = py[s]; candN[s] = 1; }
        }

        // movement resolution for our 3 units, mirroring engine.py:281-304
        auto resolve = [&](const int wx[3], const int wy[3], int ox[3], int oy[3]) {
            bool blocked[3] = {false, false, false};
            for (int a = 0; a < 3; ++a) {
                if (!alive3[a]) { ox[a] = oy[a] = -1; continue; }
                for (int b = a + 1; b < 3; ++b) {
                    if (!alive3[b]) continue;
                    if (wx[a] == wx[b] && wy[a] == wy[b]) blocked[a] = blocked[b] = true;
                }
            }
            bool changed = true;
            while (changed) {
                changed = false;
                for (int s = 0; s < 3; ++s) {
                    if (!alive3[s] || blocked[s]) continue;
                    if (wx[s] == px[s] && wy[s] == py[s]) continue;
                    for (int o = 0; o < 3; ++o) {
                        if (o == s || !alive3[o]) continue;
                        if (wx[s] != px[o] || wy[s] != py[o]) continue;   // not entering o's start cell
                        bool vacates = (wx[o] != px[o] || wy[o] != py[o]);
                        bool swap = (wx[o] == px[s] && wy[o] == py[s]);
                        if (!vacates || blocked[o] || swap) { blocked[s] = true; changed = true; break; }
                    }
                }
            }
            for (int s = 0; s < 3; ++s) {
                if (!alive3[s]) { ox[s] = oy[s] = -1; continue; }
                if (blocked[s]) { ox[s] = px[s]; oy[s] = py[s]; }
                else { ox[s] = wx[s]; oy[s] = wy[s]; }
            }
        };

        int carrierIdx = -1;
        for (int c = 0; c < 3; ++c) if (alive3[c] && carry3[c] && !inBase(me, px[c], py[c])) { carrierIdx = c; break; }

        auto evalCell = [&](int s, int nx, int ny) {
            int v = 0;
            int cur = goalDist(s, px[s], py[s]);
            int nd = goalDist(s, nx, ny);
            if (cur < INF && nd < INF) v += (cur - nd) * 100;
            if (nx == px[s] && ny == py[s]) v -= 1;                  // prefer progress
            if (!carry3[s]) {
                for (const auto& g : grounds) if (g.sx == nx && g.sy == ny) { v += 2500; break; }
            }
            int threats = 0, flagThreats = 0;
            for (size_t fi = 0; fi < foes.size(); ++fi) {
                int fx = foeNext[fi].first, fy = foeNext[fi].second;
                if (canAttack(fx, fy, nx, ny)) { ++threats; if (foes[fi].flag >= 0) ++flagThreats; }
            }
            if (attackOpt[s] && nx == px[s] && ny == py[s]) {
                int tid = atkTarget[s];
                int thp = -1, tflag = -1;
                for (const auto& f : foes) if (f.id == tid) { thp = f.hp; tflag = f.flag; }
                if (thp >= 0) {
                    int av = (tflag >= 0) ? 350 : 120;
                    if (thp <= dmg) av += 300;
                    v += av;
                }
            }
            if (carrierIdx >= 0 && !carry3[s] && s != carrierIdx) {
                // Move to cut off whoever is hunting our carrier: the baseline's edge is
                // carrier kills, and 3 simultaneous hits kill a 100 HP unit.
                if (manh(nx, ny, px[carrierIdx], py[carrierIdx]) < manh(px[s], py[s], px[carrierIdx], py[carrierIdx]))
                    v += 25;
            }
            if (threats > 0 && carry3[s]) {
                int hits = (hp3[s] + dmg - 1) / dmg;
                int sev = min(threats, hits) + min(flagThreats, 1);
                v -= 150 * sev;
                // being hunted specifically for the flag is a much stronger signal: prefer a
                // detour that keeps the carrier alive over one step of direct progress.
                if (flagThreats > 0 || min(threats, hits) >= hits) v -= 400;
            }
            int code = nx * 64 + ny, seen = 0;
            for (int h = 0; h < histN; ++h) if (hist[s][h] == code) ++seen;
            if (seen) v -= (carry3[s] ? 30 : 160) * seen;            // carriers must not dither either
            return v;
        };

        std::array<std::array<int, 2>, 3> badCell;                   // cells an enemy occupies and is likely to keep
        int nBad = 0;
        for (size_t fi = 0; fi < foes.size() && nBad < 3; ++fi) {
            bool leaves = (foeNext[fi].first != foes[fi].x || foeNext[fi].second != foes[fi].y);
            if (!leaves) { badCell[nBad][0] = foes[fi].x; badCell[nBad][1] = foes[fi].y; ++nBad; }
        }

        int bestVal = INT_MIN;
        int bx3[3] = {px[0], px[1], px[2]}, by3[3] = {py[0], py[1], py[2]};
        for (int a = 0; a <= candN[0]; ++a) {
            if (a == candN[0] && !attackOpt[0]) continue;
            for (int b = 0; b <= candN[1]; ++b) {
                if (b == candN[1] && !attackOpt[1]) continue;
                for (int c = 0; c <= candN[2]; ++c) {
                    if (c == candN[2] && !attackOpt[2]) continue;
                    int wx[3], wy[3], ox[3], oy[3];
                    wx[0] = candX[0][min(a, candN[0] - 1)]; wy[0] = candY[0][min(a, candN[0] - 1)];
                    wx[1] = candX[1][min(b, candN[1] - 1)]; wy[1] = candY[1][min(b, candN[1] - 1)];
                    wx[2] = candX[2][min(c, candN[2] - 1)]; wy[2] = candY[2][min(c, candN[2] - 1)];
                    for (int s = 0; s < 3; ++s) if (!alive3[s]) { wx[s] = px[s]; wy[s] = py[s]; }
                    bool attacking[3] = {false, false, false};
                    int asel[3] = {a, b, c};
                    for (int s = 0; s < 3; ++s) {
                        if (attackOpt[s] && asel[s] == candN[s]) { attacking[s] = true; wx[s] = px[s]; wy[s] = py[s]; }
                    }
                    resolve(wx, wy, ox, oy);
                    int val = 0;
                    bool ok = true;
                    for (int s = 0; s < 3; ++s) {
                        if (!alive3[s]) continue;
                        for (int q = 0; q < nBad; ++q)
                            if (ox[s] == badCell[q][0] && oy[s] == badCell[q][1]) { ok = false; break; }
                        if (!ok) break;
                        val += evalCell(s, ox[s], oy[s]);
                    }
                    if (!ok) continue;
                    if (val > bestVal) {
                        bestVal = val;
                        for (int s = 0; s < 3; ++s) { bx3[s] = ox[s]; by3[s] = oy[s]; chosenTgt[s] = attacking[s] ? atkTarget[s] : -1; }
                    }
                }
            }
        }
        if (bestVal == INT_MIN) { for (int s = 0; s < 3; ++s) { bx3[s] = px[s]; by3[s] = py[s]; } }

        // ---- moves and actions from the chosen joint outcome ----
        for (int s = 0; s < 3; ++s) {
            if (!alive3[s]) { chosenMv[s] = 'S'; chosenAct[s] = "-"; finalX[s] = px[s]; finalY[s] = py[s]; continue; }
            finalX[s] = bx3[s]; finalY[s] = by3[s];
            chosenMv[s] = 'S';
            for (int k = 0; k < 4; ++k)
                if (px[s] + DX[k] == bx3[s] && py[s] + DY[k] == by3[s]) chosenMv[s] = MOVECH[k];
            if (carry3[s]) { chosenAct[s] = "-"; continue; }   // capture is automatic; never drop at home
            bool onGround = false;
            for (const auto& g : grounds) if (g.sx == bx3[s] && g.sy == by3[s]) { onGround = true; break; }
            if (onGround) { chosenAct[s] = "P"; continue; }
            if (chosenTgt[s] >= 0) { chosenAct[s] = to_string(chosenTgt[s]); continue; }
            if (!immune3[s]) {
                int bt = -1, bv = INT_MIN;
                for (const auto& f : foes) {
                    if (!canAttack(bx3[s], by3[s], f.x, f.y)) continue;
                    int v = (f.flag >= 0) ? 1000 : 0;
                    v += (f.hp <= dmg) ? 400 : 0;
                    // defending our carrier: shoot whoever is about to kill it
                    for (int c = 0; c < 3; ++c) {
                        if (!alive3[c] || !carry3[c] || inBase(me, px[c], py[c])) continue;
                        if (manh(f.x, f.y, px[c], py[c]) <= 2) v += 600;
                    }
                    v -= manh(bx3[s], by3[s], f.x, f.y);
                    if (v > bv) { bv = v; bt = f.id; }
                }
                if (bt >= 0) chosenAct[s] = to_string(bt);
            }
        }

        // ---- emit ----
        for (int s = 0; s < 3; ++s) {
            for (int h = 7; h > 0; --h) hist[s][h] = hist[s][h - 1];
            hist[s][0] = (alive3[s] && finalX[s] >= 0 && finalX[s] < 64 && finalY[s] >= 0 && finalY[s] < 64)
                             ? finalX[s] * 64 + finalY[s] : -1;
        }
        histN = min(8, histN + 1);

        string out = to_string(turn);
        for (int s = 0; s < 3; ++s) {
            if (!alive3[s]) { out += " S -"; continue; }
            out += " ";
            out += chosenMv[s];
            out += " ";
            out += chosenAct[s];
        }
        (void)immune3; (void)newHp;
        return out;
    }
};

const int Bot::DX[4] = {0, 0, -1, 1};    // U D L R
const int Bot::DY[4] = {-1, 1, 0, 0};
const char Bot::MOVECH[4] = {'U', 'D', 'L', 'R'};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Bot bot;
    while (bot.readTurn()) {
        if (!bot.init_done) continue;
        cout << bot.decide() << "\n";
        cout.flush();
    }
    return 0;
}
