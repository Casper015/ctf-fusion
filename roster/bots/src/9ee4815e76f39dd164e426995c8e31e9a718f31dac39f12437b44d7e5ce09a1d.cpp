// CTF bot: BFS navigation, flag/intercept task assignment, joint 125-tuple movement, outcome-conditioned volleys.
#include <algorithm>
#include <array>
#include <climits>
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
            if (active[k] && role[k] == CARRIER && D(enemies[e].c, start[k]) <= 6) carrierThreat[e] = 1;
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
        carrierField.assign(N, INT_MAX / 4);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        for (int c : baseCells[me]) if (!wallc[c]) { carrierField[c] = 0; pq.push({0, c}); }
        while (!pq.empty()) {
            auto [d, c] = pq.top(); pq.pop();
            if (d != carrierField[c]) continue;
            int enter = 10 + 25 * threat[c];   // cost of stepping into c
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
            if (!immune && threat[p] * dmg >= hpv[k]) sc -= 60;
            return sc;
        }
        if (groundAt[p] >= 0) {
            int h = homeDist[me][p];
            if (t + h <= g.turns) sc += 300;
        }
        if (role[k] == HUNT || role[k] == INTERCEPT) sc -= 10.0 * D(goal[k], p);
        else if (role[k] == GUARD) sc -= 10.0 * max(0, D(goal[k], p) - 2) - (canAttack(p, goal[k]) ? 12.0 : 0.0);
        else if (role[k] == STAGE) sc -= 6.0 * D(goal[k], p);
        if (!immune) {
            int hit = 0; bool carrierHit = false;
            for (auto& e : enemies) if (canAttack(p, e.c)) { hit++; if (e.carrier) carrierHit = true; }
            sc += 5.0 * min(hit, 1) + (carrierHit ? 15.0 : 0.0);
            sc -= 7.0 * threat[p];
            if (threat[p] * dmg >= hpv[k]) sc -= 25.0;
        }
        return sc;
    };

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
        if (sc > bestScore) {
            bestScore = sc;
            for (int k = 0; k < 3; k++) { bestMove[k] = mv[k]; bestPos[k] = np[k]; }
        }
    }

    // Attack assignment: evaluate volleys per enemy movement outcome.
    string act[3] = {"-", "-", "-"};
    vector<int> shooters;
    for (int k = 0; k < 3; k++) {
        if (!active[k]) continue;
        if (flagv[k] >= 0) continue;
        if (groundAt[bestPos[k]] >= 0 || spawning[k]) { act[k] = "P"; continue; }
        shooters.push_back(k);
    }
    // Candidate targets per shooter.
    vector<vector<int>> cand(3);
    vector<array<pair<int, double>, 5>> outs(enemies.size());
    vector<int> nouts(enemies.size(), 0);
    for (size_t e = 0; e < enemies.size(); e++) {
        auto& en = enemies[e];
        int pred = en.c;
        if (en.carrier) {
            const auto& hd = homeDist[en.team];
            for (int m = 1; m < 5; m++) { int n = stepTo(en.c, m); if (hd[n] < hd[pred]) pred = n; }
        }
        vector<pair<int, double>> o;
        for (int m = 0; m < 5; m++) {
            int n = stepTo(en.c, m);
            if (n != en.c && occupied[n] >= 0) n = en.c;      // likely bounced
            double w;
            if (en.carrier) w = (n == pred) ? 0.7 : 0.075;
            else w = (m == 0) ? 0.4 : 0.15;
            bool merged = false;
            for (auto& pr : o) if (pr.first == n) { pr.second += w; merged = true; }
            if (!merged) o.push_back({n, w});
        }
        double tot = 0; for (auto& pr : o) tot += pr.second;
        for (auto& pr : o) { pr.second /= tot; outs[e][nouts[e]++] = pr; }
    }
    for (int k : shooters)
        for (size_t e = 0; e < enemies.size(); e++) {
            bool any = false;
            for (int i = 0; i < nouts[e]; i++) if (canAttack(bestPos[k], outs[e][i].first)) any = true;
            if (any) cand[k].push_back((int)e);
        }
    // Enumerate assignments.
    int ns = shooters.size();
    vector<int> choice(ns, -1), bestChoice(ns, -1);
    double bestVal = 0;
    function<void(int)> rec = [&](int i) {
        if (i == ns) {
            double val = 0;
            for (int a = 0; a < ns; a++) {
                int e = choice[a];
                if (e < 0) continue;
                bool first = true;
                for (int b = 0; b < a; b++) if (choice[b] == e) first = false;
                if (!first) continue;
                auto& en = enemies[e];
                double v = 0;
                for (int o = 0; o < nouts[e]; o++) {
                    int pos = outs[e][o].first;
                    int hits = 0;
                    for (int b = 0; b < ns; b++) if (choice[b] == e && canAttack(bestPos[shooters[b]], pos)) hits++;
                    double r;
                    if (hits * dmg >= en.hp) r = 100.0 + (en.carrier ? 120.0 : 0.0) + (carrierThreat[e] ? 80.0 : 0.0);
                    else r = hits * (en.carrier ? 20.0 : carrierThreat[e] ? 20.0 : 10.0);
                    v += outs[e][o].second * r;
                }
                val += v;
            }
            if (val > bestVal + 1e-9) { bestVal = val; bestChoice = choice; }
            return;
        }
        choice[i] = -1; rec(i + 1);
        for (int e : cand[shooters[i]]) { choice[i] = e; rec(i + 1); }
        choice[i] = -1;
    };
    rec(0);
    for (int a = 0; a < ns; a++) {
        int k = shooters[a];
        act[k] = bestChoice[a] >= 0 ? to_string(enemies[bestChoice[a]].id) : "P";
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
