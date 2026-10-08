// CTF bot: BFS navigation, task assignment, joint 125-tuple movement scored with a probabilistic combat model
// (enemy move forecast, attention/crossfire, death probability, DP volley assignment).
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct Unit { int id, team, x, y, hp, flag, respawn_at; };      // Dead units have x = y = -1
struct Flag { int id; string status; int x, y, holder, return_at; };
struct Event { string type; vector<int> args; };

struct Game {
    int teams = 0, me = 0, size = 0, turns = 0, hp = 0, damage = 0, respawn = 0, flag_return = 0, flag_cooldown = 0;
    vector<string> map;
    vector<pair<int, int>> bases;
    vector<pair<int, int>> spots;
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;
    vector<Event> events;
};

bool read_turn(Game& g) {
    string w;
    while (cin >> w) {
        if (w == "INIT") {
            cin >> g.teams >> g.me >> g.size >> g.turns >> g.hp >> g.damage >> g.respawn >> g.flag_return >> g.flag_cooldown;
        } else if (w == "MAP") {
            g.map.assign(g.size, "");
            for (auto& row : g.map) cin >> row;
        } else if (w == "BASES") {
            int k; cin >> k; g.bases.resize(k);
            for (auto& b : g.bases) cin >> b.first >> b.second;
        } else if (w == "SPOTS") {
            int k; cin >> k; g.spots.resize(k);
            for (auto& s : g.spots) cin >> s.first >> s.second;
        } else if (w == "TURN") {
            cin >> g.turn;
        } else if (w == "SCORE") {
            g.score.resize(g.teams);
            for (auto& s : g.score) cin >> s;
        } else if (w == "UNITS") {
            int k; cin >> k; g.units.resize(k);
            for (auto& u : g.units) { string tag; cin >> tag >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawn_at; }
        } else if (w == "FLAGS") {
            int k; cin >> k; g.flags.resize(k);
            for (auto& f : g.flags) { string tag; cin >> tag >> f.id >> f.status >> f.x >> f.y >> f.holder >> f.return_at; }
        } else if (w == "EVENTS") {
            int k; cin >> k; g.events.assign(k, {});
            string line; getline(cin, line);
            for (auto& e : g.events) {
                getline(cin, line);
                istringstream in(line);
                string tag; in >> tag >> e.type;
                for (int v; in >> v;) e.args.push_back(v);
            }
        } else if (w == "END") {
            return true;
        }
    }
    return false;
}

// ---------------- Static map data ----------------
static int S = 0, N = 0;
static vector<char> wallc;              // per cell
static vector<uint16_t> Dall;           // all-pairs terrain distances, N*N
static vector<vector<int>> homeDist;    // per team: distance to nearest own base tile
static vector<vector<int>> baseCells;   // per team
static vector<char> myBase;             // per cell
static vector<int> spawnOrder;          // my base cells in respawn preference order
static vector<int> spotCells;
static vector<char> isSpot;
static const int DX[5] = {0, 0, 0, -1, 1}, DY[5] = {0, -1, 1, 0, 0};
static const char MV[5] = {'S', 'U', 'D', 'L', 'R'};
static const uint16_t INF16 = 65535;

inline int cid(int x, int y) { return y * S + x; }
inline int cx(int c) { return c % S; }
inline int cy(int c) { return c / S; }
inline int D(int a, int b) { return Dall[(size_t)a * N + b]; }
inline int stepTo(int c, int m) {
    int x = cx(c) + DX[m], y = cy(c) + DY[m];
    if (x < 0 || y < 0 || x >= S || y >= S) return c;
    int n = cid(x, y);
    return wallc[n] ? c : n;
}

bool canAttack(int a, int b) {
    int ax = cx(a), ay = cy(a), bx = cx(b), by = cy(b);
    int dx = bx - ax, dy = by - ay;
    if (abs(dx) + abs(dy) > 2) return false;
    if (abs(dx) == 2 && dy == 0) return !wallc[cid(ax + dx / 2, ay)];
    if (abs(dy) == 2 && dx == 0) return !wallc[cid(ax, ay + dy / 2)];
    if (abs(dx) == 1 && abs(dy) == 1) return !wallc[cid(ax + dx, ay)] || !wallc[cid(ax, ay + dy)];
    return true;
}

void bfsMulti(const vector<int>& src, vector<int>& out) {
    out.assign(N, INT_MAX / 4);
    vector<int> q; q.reserve(N);
    for (int s : src) if (!wallc[s] && out[s] != 0) { out[s] = 0; q.push_back(s); }
    for (size_t h = 0; h < q.size(); h++) {
        int c = q[h];
        for (int m = 1; m < 5; m++) {
            int n = stepTo(c, m);
            if (n != c && out[n] > out[c] + 1) { out[n] = out[c] + 1; q.push_back(n); }
        }
    }
}

void initStatic(const Game& g) {
    S = g.size; N = S * S;
    wallc.assign(N, 0);
    for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) wallc[cid(x, y)] = g.map[y][x] == '#';
    Dall.assign((size_t)N * N, INF16);
    vector<int> q(N);
    for (int s = 0; s < N; s++) {
        if (wallc[s]) continue;
        uint16_t* row = &Dall[(size_t)s * N];
        int qh = 0, qt = 0;
        row[s] = 0; q[qt++] = s;
        while (qh < qt) {
            int c = q[qh++];
            for (int m = 1; m < 5; m++) {
                int n = stepTo(c, m);
                if (n != c && row[n] == INF16) { row[n] = row[c] + 1; q[qt++] = n; }
            }
        }
    }
    baseCells.assign(g.teams, {});
    homeDist.assign(g.teams, {});
    for (int t = 0; t < g.teams; t++) {
        auto [bx, by] = g.bases[t];
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) baseCells[t].push_back(cid(bx + dx, by + dy));
        bfsMulti(baseCells[t], homeDist[t]);
    }
    myBase.assign(N, 0);
    for (int c : baseCells[g.me]) myBase[c] = 1;
    int cc = S / 2;
    auto [bx, by] = g.bases[g.me];
    spawnOrder = baseCells[g.me];
    stable_sort(spawnOrder.begin(), spawnOrder.end(), [&](int a, int b) {
        int da = abs(cx(a) - cc) + abs(cy(a) - cc), db = abs(cx(b) - cc) + abs(cy(b) - cc);
        if (da != db) return da < db;
        int ka = abs(cx(a) - bx) + 2 * abs(cy(a) - by), kb = abs(cx(b) - bx) + 2 * abs(cy(b) - by);
        return ka < kb;
    });
    isSpot.assign(N, 0);
    for (auto [x, y] : g.spots) { spotCells.push_back(cid(x, y)); isSpot[cid(x, y)] = 1; }
}

// ---------------- Per-turn decision ----------------
struct Enemy { int id, team, c, hp; bool carrier; };
enum Role { NONE, CARRIER, HUNT, INTERCEPT, STAGE, GUARD };

string decide(const Game& g) {
    const int t = g.turn, me = g.me, dmg = g.damage;
    vector<Enemy> enemies;
    vector<int> occupied(N, -1);           // unit id at cell (input positions)
    for (auto& u : g.units) if (u.x >= 0) occupied[cid(u.x, u.y)] = u.id;
    for (auto& u : g.units)
        if (u.x >= 0 && u.team != me) enemies.push_back({u.id, u.team, cid(u.x, u.y), u.hp, u.flag >= 0});
    vector<char> enemyCell(N, 0);
    for (auto& e : enemies) enemyCell[e.c] = 1;

    // Threat: number of enemy shooters that can hit a cell after one move.
    vector<int> threat(N, 0), stamp(N, -1);
    for (size_t k = 0; k < enemies.size(); k++) {
        auto& e = enemies[k];
        if (e.carrier) continue;
        for (int m = 0; m < 5; m++) {
            int p = stepTo(e.c, m);
            int px = cx(p), py = cy(p);
            for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
                if (abs(dx) + abs(dy) > 2) continue;
                int x = px + dx, y = py + dy;
                if (x < 0 || y < 0 || x >= S || y >= S) continue;
                int c = cid(x, y);
                if (wallc[c] || stamp[c] == (int)k) continue;
                if (canAttack(p, c)) { stamp[c] = (int)k; threat[c]++; }
            }
        }
    }

    // My units, including predicted respawns this turn.
    int start[3], hpv[3], flagv[3];
    bool active[3], spawning[3];
    vector<char> occ2(N, 0);
    for (auto& u : g.units) if (u.x >= 0) occ2[cid(u.x, u.y)] = 1;
    for (int k = 0; k < 3; k++) {
        const Unit& u = g.units[3 * me + k];
        start[k] = -1; hpv[k] = u.hp; flagv[k] = u.flag; active[k] = false; spawning[k] = false;
        if (u.x >= 0) { start[k] = cid(u.x, u.y); active[k] = true; }
        else if (u.respawn_at >= 0 && u.respawn_at <= t) {
            for (int c : spawnOrder) if (!occ2[c]) { occ2[c] = 1; start[k] = c; active[k] = true; spawning[k] = true; hpv[k] = g.hp; flagv[k] = -1; break; }
        }
    }

    // Ground flags reliably present this turn.
    vector<pair<int, int>> ground;   // (flag id, cell)
    vector<int> groundAt(N, -1);
    for (auto& f : g.flags) {
        bool ok = f.status == "home" || (f.status == "dropped" && f.return_at > t);
        if (ok && f.x >= 0) { ground.push_back({f.id, cid(f.x, f.y)}); if (groundAt[cid(f.x, f.y)] < 0) groundAt[cid(f.x, f.y)] = f.id; }
    }

    // Enemy forecast: softmax over each enemy's moves from a goal/target model, plus its attention on us.
    vector<int> upos(g.units.size(), -1);
    for (auto& u : g.units) if (u.x >= 0) upos[u.id] = cid(u.x, u.y);
    auto man = [&](int a, int b) { return abs(cx(a) - cx(b)) + abs(cy(a) - cy(b)); };
    struct Fc { int n = 0; int c[5]; double w[5]; bool shoot[5]; double att = 1.0; };
    vector<Fc> fc(enemies.size());
    vector<double> threatP(N, 0.0);
    for (size_t e = 0; e < enemies.size(); e++) {
        const Enemy& en = enemies[e];
        const Unit& eu = g.units[en.id];
        Fc& F = fc[e];
        int goalC = -1;
        if (!en.carrier) {
            double bv = 1e18;
            for (auto& f : g.flags) {
                bool ok = f.x >= 0 && (f.status == "home" || (f.status == "dropped" && f.return_at > t));
                if (!ok) continue;
                int q = cid(f.x, f.y), d = D(en.c, q);
                if (f.status == "dropped" && d > f.return_at - t) continue;
                double v = d + 0.28 * homeDist[en.team][q];
                if (v < bv) { bv = v; goalC = q; }
            }
            if (goalC < 0 && en.team < (int)spotCells.size()) goalC = spotCells[en.team];
        }
        int target = -1; double tv = -1e18;
        for (auto& v : g.units) {
            if (v.team == en.team || upos[v.id] < 0) continue;
            int d = D(en.c, upos[v.id]);
            if (d > 7) continue;
            double val = 8 - d + (v.flag >= 0 ? 6 : 0) + (100 - v.hp) * 0.025;
            if (val > tv) { tv = val; target = v.id; }
        }
        F.att = (target >= 0 && g.units[target].team != me) ? 0.2 : 1.0;
        double vals[5];
        for (int m = 0; m < 5; m++) {
            int q = stepTo(en.c, m);
            if (m > 0 && q == en.c) continue;
            double sc = 0;
            if (en.carrier) {
                const auto& hd = homeDist[en.team];
                sc = 2.2 * (hd[en.c] - hd[q]);
                double risk = 0;
                for (auto& v : g.units) {
                    if (v.team == en.team || v.flag >= 0 || upos[v.id] < 0) continue;
                    int d = man(q, upos[v.id]);
                    risk += d <= 2 ? 1.3 : d == 3 ? 0.7 : d == 4 ? 0.15 : 0.0;
                }
                sc -= risk * (eu.hp <= 34 ? 2.3 : eu.hp <= 68 ? 1.8 : 1.2);
            } else {
                if (goalC >= 0) sc += 1.25 * (D(en.c, goalC) - D(q, goalC));
                if (target >= 0) {
                    int tp = upos[target], d = man(q, tp);
                    sc += (tv > 8 ? 1.2 : 0.5) * (canAttack(q, tp) ? 1.3 : d <= 3 ? 0.5 : 0.0);
                    if (eu.hp <= 34 && g.units[target].flag < 0 && d <= 2) sc -= 0.45;
                }
                if (groundAt[q] >= 0) sc += 2.5;
            }
            if (m == 0) sc -= 0.35;
            if (occupied[q] >= 0 && occupied[q] != en.id) sc -= 0.65;
            F.c[F.n] = q; vals[F.n] = sc; F.shoot[F.n] = !en.carrier && groundAt[q] < 0; F.n++;
        }
        double mx = *max_element(vals, vals + F.n), sum = 0;
        for (int i = 0; i < F.n; i++) { F.w[i] = exp((vals[i] - mx) / 0.75); sum += F.w[i]; }
        for (int i = 0; i < F.n; i++) F.w[i] = 0.88 * F.w[i] / sum + 0.12 / F.n;
        if (en.carrier) continue;
        for (int i = 0; i < F.n; i++) {
            int q = F.c[i];
            for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
                if (abs(dx) + abs(dy) > 2) continue;
                int x = cx(q) + dx, y = cy(q) + dy;
                if (x < 0 || y < 0 || x >= S || y >= S) continue;
                int r = cid(x, y);
                if (!wallc[r] && canAttack(q, r)) threatP[r] += F.w[i] * F.att;
            }
        }
    }

    // Task assignment.
    Role role[3] = {NONE, NONE, NONE};
    int goal[3] = {-1, -1, -1};
    for (int k = 0; k < 3; k++) if (active[k] && flagv[k] >= 0) role[k] = CARRIER;

    struct Cand { double cost; int k, type, idx, cell; };
    vector<Cand> cands;
    for (size_t fi = 0; fi < ground.size(); fi++) {
        int fc = ground[fi].second;
        int rival = INT_MAX / 4;
        for (auto& e : enemies) if (!e.carrier) rival = min(rival, D(e.c, fc));
        int h = homeDist[me][fc];
        for (int k = 0; k < 3; k++) {
            if (!active[k] || role[k] == CARRIER) continue;
            int d = D(start[k], fc);
            if (d >= INF16) continue;
            if (t + max(0, d - 1) + h > g.turns) continue;
            double cost = d + 0.5 * h + (d > rival ? 1.5 * (d - rival) : 0.0);
            cands.push_back({cost, k, HUNT, (int)fi, fc});
        }
    }
    // Enemy carriers: intercept along their greedy route home.
    int icount = 0;
    vector<int> interceptCap;
    for (auto& e : enemies) {
        if (!e.carrier) continue;
        vector<int> path{e.c};
        int c = e.c;
        const auto& hd = homeDist[e.team];
        while (hd[c] > 0 && path.size() < 200) {
            int best = c;
            for (int m = 1; m < 5; m++) { int n = stepTo(c, m); if (hd[n] < hd[best]) best = n; }
            if (best == c) break;
            c = best; path.push_back(c);
        }
        int idx = icount++;
        interceptCap.push_back(2);
        for (int k = 0; k < 3; k++) {
            if (!active[k] || role[k] == CARRIER) continue;
            for (size_t s = 0; s < path.size(); s++) {
                if (D(start[k], path[s]) <= (int)s + 2) {
                    double cost = s + 4.0 + (g.teams > 2 ? 4.0 : 0.0);
                    cands.push_back({cost, k, INTERCEPT, idx, path[s]});
                    break;
                }
            }
        }
    }
    // Guard: enemies close enough to threaten one of our carriers become escort targets.
    vector<char> carrierThreat(enemies.size(), 0);
    for (size_t e = 0; e < enemies.size(); e++) {
        if (enemies[e].carrier) continue;
        for (int k = 0; k < 3; k++)
            if (active[k] && role[k] == CARRIER && D(enemies[e].c, start[k]) <= 4) carrierThreat[e] = 1;
        if (!carrierThreat[e]) continue;
        int idx = icount++;
        interceptCap.push_back(2);
        for (int k = 0; k < 3; k++) {
            if (!active[k] || role[k] == CARRIER) continue;
            int d = D(start[k], enemies[e].c);
            if (d > 8) continue;
            cands.push_back({max(0, d - 2) + 1.0, k, GUARD, idx, enemies[e].c});
        }
    }
    sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        if (a.k != b.k) return a.k < b.k;
        return a.cell < b.cell;
    });
    vector<char> flagTaken(ground.size(), 0);
    for (auto& cd : cands) {
        if (role[cd.k] != NONE) continue;
        if (cd.type == HUNT) {
            if (flagTaken[cd.idx]) continue;
            flagTaken[cd.idx] = 1;
        } else {
            if (interceptCap[cd.idx] <= 0) continue;
            interceptCap[cd.idx]--;
        }
        role[cd.k] = (Role)cd.type; goal[cd.k] = cd.cell;
    }
    // Staging: idle units take distinct flag spots, preferring ones near home.
    vector<char> spotUsed(spotCells.size(), 0);
    for (int k = 0; k < 3; k++) {
        if (!active[k] || role[k] != NONE) continue;
        double best = 1e18; int bi = -1;
        for (size_t s = 0; s < spotCells.size(); s++) {
            if (spotUsed[s]) continue;
            int sc = spotCells[s];
            double cost = D(start[k], sc) + 1.5 * homeDist[me][sc];
            if (cost < best) { best = cost; bi = (int)s; }
        }
        if (bi >= 0) { spotUsed[bi] = 1; role[k] = STAGE; goal[k] = spotCells[bi]; }
    }

    // Danger-weighted distance to home for carriers.
    vector<int> carrierField;
    bool anyCarrier = false;
    for (int k = 0; k < 3; k++) if (role[k] == CARRIER) anyCarrier = true;
    if (anyCarrier) {
        int minHp = 100;
        for (int k = 0; k < 3; k++) if (role[k] == CARRIER) minHp = min(minHp, hpv[k]);
        double hf = minHp <= 34 ? 7.0 : minHp <= 68 ? 5.0 : 3.0;
        carrierField.assign(N, INT_MAX / 4);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        for (int c : baseCells[me]) if (!wallc[c]) { carrierField[c] = 0; pq.push({0, c}); }
        while (!pq.empty()) {
            auto [d, c] = pq.top(); pq.pop();
            if (d != carrierField[c]) continue;
            double shield = 0;
            for (int k = 0; k < 3; k++) if (active[k] && role[k] != CARRIER && !spawning[k] && man(start[k], c) <= 3) shield += 0.3;
            int enter = 10 + (int)min(120.0, 10.0 * max(0.0, threatP[c] - shield) * hf);   // cost of stepping into c
            for (int m = 1; m < 5; m++) {
                int n = stepTo(c, m);
                if (n == c) continue;
                if (carrierField[n] > d + enter) { carrierField[n] = d + enter; pq.push({d + enter, n}); }
            }
        }
    }

    auto unitScore = [&](int k, int p) -> double {
        double sc = 0;
        bool immune = spawning[k];
        if (role[k] == CARRIER) {
            if (myBase[p]) sc += 10000;
            sc -= carrierField[p];
            return sc;
        }
        if (groundAt[p] >= 0) {
            int h = homeDist[me][p];
            if (t + h <= g.turns) sc += 300;
        }
        if (role[k] == HUNT || role[k] == INTERCEPT) sc -= 10.0 * D(goal[k], p);
        else if (role[k] == GUARD) sc -= 10.0 * max(0, D(goal[k], p) - 2) - (canAttack(p, goal[k]) ? 12.0 : 0.0);
        else if (role[k] == STAGE) sc -= 6.0 * D(goal[k], p);
        (void)immune;
        return sc;
    };

    // Combat model inputs: nearby enemies, and crossfire competition from third teams at each forecast cell.
    vector<int> rel;
    for (size_t e = 0; e < enemies.size(); e++) {
        bool r = false;
        for (int k = 0; k < 3; k++) if (active[k] && man(enemies[e].c, start[k]) <= 4) r = true;
        if (r) rel.push_back((int)e);
    }
    auto pri = [&](int hp, bool carrier, int from, int at) {
        double v = carrier ? 5.0 : 1.0;
        if (hp <= 34) v *= 2.4; else if (hp <= 68) v *= 1.35;
        if (man(from, at) <= 1) v *= 1.12;
        return v;
    };
    vector<array<double, 5>> cross(enemies.size());
    for (int e : rel)
        for (int o = 0; o < fc[e].n; o++) {
            double v = 0; int q = fc[e].c[o];
            for (auto& u : g.units)
                if (u.team != me && u.team != enemies[e].team && upos[u.id] >= 0 && canAttack(q, upos[u.id]))
                    v += pri(u.hp, u.flag >= 0, q, upos[u.id]);
            cross[e][o] = v;
        }
    const double CS = 9.5;   // combat values are in movement steps; scores use 10 per step
    vector<array<double, 3>> incoming(rel.size());
    vector<array<double, 8>> util(rel.size());
    int bestTarget[3] = {-1, -1, -1};

    // Joint movement over 5^3 tuples with own-team conflict resolution; enemies treated as stationary blockers.
    int bestMove[3] = {0, 0, 0}, bestPos[3] = {start[0], start[1], start[2]};
    double bestScore = -1e18;
    int ncomb = 125;
    for (int combo = 0; combo < ncomb; combo++) {
        int mv[3] = {combo % 5, (combo / 5) % 5, combo / 25};
        bool skip = false;
        for (int k = 0; k < 3; k++) if (!active[k] && mv[k] != 0) skip = true;
        if (skip) continue;
        int want[3], np[3];
        bool blocked[3] = {false, false, false};
        for (int k = 0; k < 3; k++) {
            if (!active[k]) { want[k] = -1; continue; }
            want[k] = stepTo(start[k], mv[k]);
            if (want[k] != start[k] && enemyCell[want[k]]) blocked[k] = true;
        }
        for (int a = 0; a < 3; a++) for (int b = a + 1; b < 3; b++)
            if (active[a] && active[b] && want[a] == want[b]) blocked[a] = blocked[b] = true;
        for (int it = 0; it < 3; it++)
            for (int a = 0; a < 3; a++) {
                if (!active[a] || blocked[a] || want[a] == start[a]) continue;
                for (int b = 0; b < 3; b++)
                    if (b != a && active[b] && start[b] == want[a] && (blocked[b] || want[b] == start[b])) blocked[a] = true;
            }
        bool dup = false;
        for (int k = 0; k < 3; k++) np[k] = active[k] ? (blocked[k] ? start[k] : want[k]) : -1;
        // Moves that end up blocked are equivalent to staying; skip so the plain stay tuple represents them.
        for (int k = 0; k < 3; k++) if (active[k] && blocked[k] && mv[k] != 0) dup = true;
        if (dup) continue;
        double sc = 0;
        for (int k = 0; k < 3; k++) if (active[k]) sc += unitScore(k, np[k]);

        // Combat: offensive utility per shooter subset and incoming fire, both conditioned on each enemy move outcome.
        int armed = 0;
        for (int k = 0; k < 3; k++) if (active[k] && !spawning[k] && flagv[k] < 0 && groundAt[np[k]] < 0) armed |= 1 << k;
        int worst[3] = {0, 0, 0};
        for (size_t r = 0; r < rel.size(); r++) {
            int e = rel[r];
            const Enemy& en = enemies[e];
            const Fc& F = fc[e];
            util[r].fill(0.0); incoming[r].fill(0.0);
            bool possible[3] = {false, false, false};
            double kv = 11.0 + (en.carrier ? 34.0 : 0.0) + (carrierThreat[e] ? 6.0 : 0.0);
            for (int o = 0; o < F.n; o++) {
                int q = F.c[o]; double w = F.w[o];
                int hitMask = 0;
                for (int k = 0; k < 3; k++) if ((armed >> k & 1) && canAttack(np[k], q)) hitMask |= 1 << k;
                for (int sub = 1; sub < 8; sub++) {
                    if (sub & ~armed) continue;
                    int h = __builtin_popcount(sub & hitMask);
                    util[r][sub] += w * (min(en.hp, dmg * h) * 0.04 + (dmg * h >= en.hp ? kv : 0.0));
                }
                if (!F.shoot[o]) continue;
                double wt[3] = {0, 0, 0}, den = cross[e][o];
                for (int k = 0; k < 3; k++)
                    if (active[k] && !spawning[k] && canAttack(q, np[k])) { wt[k] = pri(hpv[k], flagv[k] >= 0, q, np[k]); den += wt[k]; possible[k] = true; }
                if (den > 0) for (int k = 0; k < 3; k++) incoming[r][k] += w * F.att * wt[k] / den;
            }
            for (int k = 0; k < 3; k++) if (possible[k]) worst[k]++;
        }
        double combat = 0;
        for (int k = 0; k < 3; k++) {
            if (!active[k] || spawning[k]) continue;
            int lethal = (hpv[k] + dmg - 1) / dmg;
            double hc[4] = {1, 0, 0, 0}, eh = 0;
            for (size_t r = 0; r < rel.size(); r++) {
                double p = min(1.0, max(0.0, incoming[r][k]));
                eh += p;
                double nx[4] = {0, 0, 0, 0};
                for (int j = 0; j < 4; j++) { nx[j] += hc[j] * (1 - p); nx[min(3, j + 1)] += hc[j] * p; }
                for (int j = 0; j < 4; j++) hc[j] = nx[j];
            }
            double death = 0;
            for (int j = min(lethal, 3); j < 4; j++) death += hc[j];
            bool car = flagv[k] >= 0;
            combat -= eh * (1.1 + (car ? 3.0 : 0.0)) + death * (12.0 + (car ? 45.0 : 0.0));
            if (car && worst[k] >= lethal) combat -= 10.0 * (1 - death);
        }
        // Assign disjoint shooter subsets to enemies (DP over enemies, 8 shooter masks).
        double dpv[8]; int dpt[8][3];
        for (int m = 0; m < 8; m++) { dpv[m] = -1e18; dpt[m][0] = dpt[m][1] = dpt[m][2] = -1; }
        dpv[0] = 0;
        for (size_t r = 0; r < rel.size(); r++) {
            double nv[8]; int nt[8][3];
            for (int m = 0; m < 8; m++) { nv[m] = dpv[m]; for (int j = 0; j < 3; j++) nt[m][j] = dpt[m][j]; }
            for (int used = 0; used < 8; used++) {
                if (dpv[used] < -1e17) continue;
                int rem = armed & ~used;
                for (int sub = rem; sub; sub = (sub - 1) & rem) {
                    if (util[r][sub] <= 1e-8) continue;
                    int c = used | sub;
                    double cand = dpv[used] + util[r][sub];
                    if (cand > nv[c] + 1e-9) {
                        nv[c] = cand;
                        for (int j = 0; j < 3; j++) nt[c][j] = (sub >> j & 1) ? rel[r] : dpt[used][j];
                    }
                }
            }
            for (int m = 0; m < 8; m++) { dpv[m] = nv[m]; for (int j = 0; j < 3; j++) dpt[m][j] = nt[m][j]; }
        }
        int bm = 0;
        for (int m = 1; m < 8; m++) if (dpv[m] > dpv[bm] + 1e-9) bm = m;
        combat += dpv[bm];
        sc += CS * combat;
        if (sc > bestScore) {
            bestScore = sc;
            for (int k = 0; k < 3; k++) { bestMove[k] = mv[k]; bestPos[k] = np[k]; bestTarget[k] = dpt[bm][k]; }
        }
    }

    // Attacks: the volley assignment chosen together with the moves; idle unladen units try a pickup.
    string act[3] = {"-", "-", "-"};
    for (int k = 0; k < 3; k++) {
        if (!active[k] || flagv[k] >= 0) continue;
        if (groundAt[bestPos[k]] >= 0 || spawning[k]) { act[k] = "P"; continue; }
        act[k] = bestTarget[k] >= 0 ? to_string(enemies[bestTarget[k]].id) : "P";
    }

    string out = to_string(t);
    for (int k = 0; k < 3; k++) {
        out += ' '; out += MV[active[k] ? bestMove[k] : 0];
        out += ' '; out += act[k];
    }
    return out;
}

int main() {
    ios::sync_with_stdio(false);
    Game g;
    bool init = false;
    while (read_turn(g)) {
        if (!init) { initStatic(g); init = true; }
        cout << decide(g) << endl;
    }
    return 0;
}
