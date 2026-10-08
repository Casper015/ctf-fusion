//  bot 3  2-15 
// fetch_flag/intercept/escort/occupy+ Dijkstra  +  + 
#include <bits/stdc++.h>
using namespace std;

struct Unit { int id, team, x, y, hp, flag, respawn_at; };
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

static const int DX[5] = {0, 0, -1, 1, 0};
static const int DY[5] = {-1, 1, 0, 0, 0};
static const char MVC[5] = {'U', 'D', 'L', 'R', 'S'};

struct Solver {
    Game g;
    int N, S, CELLS;
    vector<int> wall;
    vector<vector<int>> teamBaseDist;
    vector<vector<int>> spawnOrder;
    int centerx, centery;
    // 
    int prevPred[3][2] = {{-1, -1}, {-1, -1}, {-1, -1}};
    int stuck[3] = {0, 0, 0};
    int banCell[3] = {-1, -1, -1};   // 

    int cid(int x, int y) const { return y * S + x; }
    bool isWall(int x, int y) const { return x < 0 || y < 0 || x >= S || y >= S || wall[cid(x, y)]; }

    bool canAtk(int ax, int ay, int bx, int by) const {
        int dx = bx - ax, dy = by - ay;
        if (abs(dx) + abs(dy) > 2) return false;
        if (abs(dx) == 2 && dy == 0) return !isWall(ax + dx / 2, ay);
        if (abs(dy) == 2 && dx == 0) return !isWall(ax, ay + dy / 2);
        if (abs(dx) == 1 && abs(dy) == 1) return !isWall(ax + dx, ay) || !isWall(ax, ay + dy);
        return true;
    }

    vector<int> bfsFromCells(const vector<int>& src) {
        vector<int> d(CELLS, -1);
        queue<int> q;
        for (int c : src) { d[c] = 0; q.push(c); }
        while (!q.empty()) {
            int c = q.front(); q.pop();
            int x = c % S, y = c / S;
            for (int k = 0; k < 4; k++) {
                int nx = x + DX[k], ny = y + DY[k];
                if (isWall(nx, ny)) continue;
                int nc = cid(nx, ny);
                if (d[nc] == -1) { d[nc] = d[c] + 1; q.push(nc); }
            }
        }
        return d;
    }

    void precompute() {
        N = g.teams; S = g.size; CELLS = S * S;
        wall.assign(CELLS, 0);
        for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) wall[cid(x, y)] = (g.map[y][x] == '#');
        centerx = centery = S / 2;
        teamBaseDist.assign(N, {});
        spawnOrder.assign(N, {});
        for (int t = 0; t < N; t++) {
            vector<int> cells;
            auto bc = g.bases[t];
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) cells.push_back(cid(bc.first + dx, bc.second + dy));
            teamBaseDist[t] = bfsFromCells(cells);
            vector<int> so = cells;
            sort(so.begin(), so.end(), [&](int a, int b) {
                int ax = a % S, ay = a / S, bx = b % S, by = b / S;
                int da = abs(ax - centerx) + abs(ay - centery), db = abs(bx - centerx) + abs(by - centery);
                if (da != db) return da < db;
                int ta = abs(ax - bc.first) + 2 * abs(ay - bc.second);
                int tb = abs(bx - bc.first) + 2 * abs(by - bc.second);
                return ta < tb;
            });
            spawnOrder[t] = so;
        }
    }

    struct My {
        Unit u;
        bool alive = false, immune = false;
        int px = -1, py = -1;
        int role = 3;
        int tx = -1, ty = -1;
        int move = 4, act = 0, actArg = -1;
        int fid = -1;                          // role==1 
        string note;
    };

    vector<double> costField(int tx, int ty, const vector<double>& w) {
        const double INF = 1e18;
        vector<double> d(CELLS, INF);
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
        int t = cid(tx, ty);
        d[t] = 0; pq.push({0, t});
        while (!pq.empty()) {
            auto [dc, c] = pq.top(); pq.pop();
            if (dc > d[c] + 1e-9) continue;
            int x = c % S, y = c / S;
            for (int k = 0; k < 4; k++) {
                int nx = x + DX[k], ny = y + DY[k];
                if (isWall(nx, ny)) continue;
                int nc = cid(nx, ny);
                double nd = dc + w[nc];
                if (nd < d[nc] - 1e-9) { d[nc] = nd; pq.push({nd, nc}); }
            }
        }
        return d;
    }

    // return_home
    vector<int> carrierPath(const Unit& e) {
        vector<int> path;
        int x = e.x, y = e.y;
        path.push_back(cid(x, y));
        for (int i = 0; i < 200; i++) {
            int bd = teamBaseDist[e.team][cid(x, y)];
            if (bd <= 0) break;
            int bx = -1, by = -1;
            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d], ny = y + DY[d];
                if (isWall(nx, ny)) continue;
                int nd = teamBaseDist[e.team][cid(nx, ny)];
                if (nd >= 0 && nd < bd) { bd = nd; bx = nx; by = ny; }
            }
            if (bx < 0) break;
            x = bx; y = by;
            path.push_back(cid(x, y));
        }
        return path;
    }

    string solve() {
        vector<My> mine(3);
        vector<int> occupied(CELLS, -1);
        for (auto& u : g.units) if (u.x >= 0) occupied[cid(u.x, u.y)] = u.id;

        //  id 
        vector<pair<int, int>> respawnPos(g.units.size(), {-1, -1});
        {
            vector<int> occ = occupied;
            for (auto& u : g.units) {
                if (u.x < 0 && u.respawn_at == g.turn) {
                    for (int c : spawnOrder[u.team]) {
                        if (occ[c] < 0) { occ[c] = u.id; respawnPos[u.id] = {c % S, c / S}; break; }
                    }
                }
            }
        }

        vector<const Unit*> enemies;
        for (auto& u : g.units) if (u.team != g.me && u.x >= 0) enemies.push_back(&u);

        for (int k = 0; k < 3; k++) {
            int uid = 3 * g.me + k;
            const Unit& u = g.units[uid];
            mine[k].u = u;
            if (u.x >= 0) { mine[k].alive = true; mine[k].px = u.x; mine[k].py = u.y; }
            else if (u.respawn_at == g.turn && respawnPos[uid].first >= 0) {
                mine[k].alive = true; mine[k].immune = true;
                mine[k].px = respawnPos[uid].first; mine[k].py = respawnPos[uid].second;
            }
            // 
            if (mine[k].alive && prevPred[k][0] >= 0 && (prevPred[k][0] != mine[k].px || prevPred[k][1] != mine[k].py)) {
                stuck[k]++;
                banCell[k] = cid(prevPred[k][0], prevPred[k][1]);   // 
            } else {
                stuck[k] = 0; banCell[k] = -1;
            }
        }

        // 
        vector<int> threatNow(CELLS, 0), threatNext(CELLS, 0);
        for (auto e : enemies) {
            if (e->flag < 0) {
                for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
                    int nx = e->x + dx, ny = e->y + dy;
                    if (isWall(nx, ny)) continue;
                    if (canAtk(e->x, e->y, nx, ny)) threatNow[cid(nx, ny)]++;
                }
            }
            for (int dy = -3; dy <= 3; dy++) for (int dx = -3; dx <= 3; dx++) {
                if (abs(dx) + abs(dy) > 3) continue;
                int nx = e->x + dx, ny = e->y + dy;
                if (isWall(nx, ny)) continue;
                threatNext[cid(nx, ny)]++;
            }
        }

        vector<vector<int>> distU(3);
        for (int k = 0; k < 3; k++)
            if (mine[k].alive) distU[k] = bfsFromCells({cid(mine[k].px, mine[k].py)});

        // fetch_flag
        vector<int> availFlags;
        for (auto& f : g.flags) {
            if (f.status == "home" && f.x >= 0) availFlags.push_back(f.id);
            else if (f.status == "dropped" && f.x >= 0 && (f.return_at < 0 || f.return_at > g.turn)) availFlags.push_back(f.id);
        }
        vector<vector<int>> distF(g.flags.size());
        for (int fid : availFlags) distF[fid] = bfsFromCells({cid(g.flags[fid].x, g.flags[fid].y)});
        auto enemyMinDist = [&](const vector<int>& df) {
            int best = 1 << 29;
            for (auto e : enemies) {
                int d = df[cid(e->x, e->y)];
                if (d >= 0) best = min(best, d);
            }
            return best;
        };

        vector<const Unit*> eCarriers;
        for (auto e : enemies) if (e->flag >= 0) eCarriers.push_back(e);
        int myCarrier = -1;
        for (int k = 0; k < 3; k++) if (mine[k].alive && mine[k].u.flag >= 0) myCarrier = k;

        // 
        double crowd = N <= 2 ? 0.25 : (N == 3 ? 0.5 : (N <= 5 ? 0.7 : 1.0));
        //  = 
        vector<int> hotFlag(g.flags.size(), 0);
        for (int fid : availFlags) {
            int fc = cid(g.flags[fid].x, g.flags[fid].y);
            int near = 0;
            for (auto e : enemies) if (abs(e->x - g.flags[fid].x) + abs(e->y - g.flags[fid].y) <= 2) near++;
            hotFlag[fid] = (N >= 3 && (threatNext[fc] >= 2 || near >= 2)) ? 1 : 0;
        }

        // intercept
        vector<vector<int>> ePath(eCarriers.size());
        for (int i = 0; i < (int)eCarriers.size(); i++) ePath[i] = carrierPath(*eCarriers[i]);
        auto interceptEta = [&](int k, int i) -> int {
            // min over path cells:  - <=1 
            int best = 1 << 29;
            for (int j = 0; j < (int)ePath[i].size(); j++) {
                int d = distU[k][ePath[i][j]];
                if (d < 0) continue;
                best = min(best, d - j);
            }
            return best;
        };

        // ----------  ----------
        int F = availFlags.size(), C = eCarriers.size();
        int T_ESC = F + C, T_STAGE = F + C + 1, NT = F + C + 2;

        // 
        int carrierBd = 0, carrierTh = 0, carrierChasers = 0;
        if (myCarrier >= 0) {
            carrierBd = teamBaseDist[g.me][cid(mine[myCarrier].px, mine[myCarrier].py)];
            carrierTh = threatNext[cid(mine[myCarrier].px, mine[myCarrier].py)];
            for (auto e : enemies)
                if (e->flag < 0 && abs(e->x - mine[myCarrier].px) + abs(e->y - mine[myCarrier].py) <= 3) carrierChasers++;
        }
        int escCap = (myCarrier >= 0 && carrierBd >= 4 && carrierTh > 0) ? 2 : 1;

        auto taskScore = [&](int k, int t) -> double {
            if (!mine[k].alive) return -1e17;
            if (t < F) {
                int fid = availFlags[t];
                int fc = cid(g.flags[fid].x, g.flags[fid].y);
                int d = distU[k][fc];
                if (d < 0) return -1e17;
                double v = 100.0 - 2.0 * d - 1.5 * teamBaseDist[g.me][fc];
                int ed = enemyMinDist(distF[fid]);
                if (ed <= d - 2) v -= 35;                  // 
                else if (ed <= d + 1) v += 10;             // 
                else v += 15;                              // 
                // fetch_flag => 
                v -= crowd * (10.0 * threatNext[fc] + 20.0 * threatNow[fc]);
                // 
                if (mine[k].u.hp <= g.damage && ed <= d + 1) v -= 45 * crowd;
                else if (mine[k].u.hp <= 2 * g.damage && ed <= d + 1) v -= 15 * crowd;
                if (g.flags[fid].status == "dropped") {
                    int left = g.flags[fid].return_at - g.turn;
                    if (left >= 0 && left < d + 1) v -= 40;
                }
                return v;
            } else if (t < F + C) {
                int i = t - F;
                auto e = eCarriers[i];
                int eta = interceptEta(k, i);
                int tb = teamBaseDist[e->team][cid(e->x, e->y)];
                if (eta > 1 && eta > tb) return -1e17;     // 
                double progress = 1.0 - tb / 20.0;
                double v = 85.0 + 70.0 * progress - 6.0 * max(0, eta);
                if (eta > 1) v -= 3.0 * eta;               // 
                return v;
            } else if (t == T_ESC) {
                if (myCarrier < 0) return -1e17;
                int d = distU[k][cid(mine[myCarrier].px, mine[myCarrier].py)];
                if (d < 0) return -1e17;
                int th = threatNext[cid(mine[myCarrier].px, mine[myCarrier].py)];
                int bd = teamBaseDist[g.me][cid(mine[myCarrier].px, mine[myCarrier].py)];
                // escortescort
                return 20.0 + 45.0 * min(carrierChasers, 2) + 20.0 * min(th, 2) + 0.5 * bd - 2.0 * d;
            } else {
                return 25.0 - 1.0 * distU[k][cid(centerx, centery)];
            }
        };

        vector<int> bestAsg(3, T_STAGE);
        double bestScore = -1e17;
        vector<int> asg(3);
        function<void(int, double)> dfs = [&](int k, double acc) {
            if (k == 3) {
                if (acc > bestScore) { bestScore = acc; bestAsg = asg; }
                return;
            }
            if (!mine[k].alive) { asg[k] = T_STAGE; dfs(k + 1, acc); return; }
            if (k == myCarrier) { asg[k] = -1; dfs(k + 1, acc); return; }
            for (int t = 0; t < NT; t++) {
                vector<int> cntInt(C, 0);
                int cntEsc = 0;
                for (int j = 0; j < k; j++) {
                    if (asg[j] >= F && asg[j] < F + C) cntInt[asg[j] - F]++;
                    else if (asg[j] == T_ESC) cntEsc++;
                }
                double s = taskScore(k, t);
                if (t < F) {
                    int have = 0;
                    for (int j = 0; j < k; j++) if (asg[j] == t) have++;
                    int fid = availFlags[t];
                    int d = distU[k][cid(g.flags[fid].x, g.flags[fid].y)];
                    int ed = enemyMinDist(distF[fid]);
                    int cap = (ed <= d + 1) ? 2 : 1;       // 
                    if (have >= cap) continue;
                    if (have == 1) s -= 20;                // 
                } else if (t < F + C) {
                    if (cntInt[t - F] >= 2) continue;
                    if (cntInt[t - F] == 1) s += 25;       // intercept
                } else if (t == T_ESC) {
                    if (cntEsc >= escCap) continue;
                    if (cntEsc == 1) s -= 10;
                }
                if (s < -1e16) continue;
                asg[k] = t;
                dfs(k + 1, acc + s);
            }
        };
        dfs(0, 0);

        // occupy
        vector<int> spotTaken(CELLS, 0), intTaken(CELLS, 0);
        for (int k = 0; k < 3; k++) {
            if (!mine[k].alive) continue;
            if (k == myCarrier) {
                mine[k].role = 0;
                mine[k].tx = g.bases[g.me].first; mine[k].ty = g.bases[g.me].second;
                mine[k].note = "return_home";
                continue;
            }
            int t = bestAsg[k];
            if (t < F) {
                mine[k].role = 1;
                mine[k].fid = availFlags[t];
                mine[k].tx = g.flags[availFlags[t]].x; mine[k].ty = g.flags[availFlags[t]].y;
                mine[k].note = "fetch_flag";
            } else if (t < F + C) {
                mine[k].role = 2;
                auto e = eCarriers[t - F];
                auto& path = ePath[t - F];
                // intercept
                int bestc = cid(e->x, e->y), bestv = 1 << 29;
                for (int j = 0; j < (int)path.size(); j++) {
                    if (intTaken[path[j]]) continue;
                    int d = distU[k][path[j]];
                    if (d < 0) continue;
                    if (d - j < bestv) { bestv = d - j; bestc = path[j]; }
                }
                intTaken[bestc] = 1;
                mine[k].tx = bestc % S; mine[k].ty = bestc / S;
                mine[k].note = "intercept";
            } else if (t == T_ESC) {
                mine[k].role = 2;
                // escort
                int bestd = 6, ex = -1, ey = -1;
                for (auto e : enemies) {
                    if (e->flag >= 0) continue;
                    int d = abs(e->x - mine[myCarrier].px) + abs(e->y - mine[myCarrier].py);
                    if (d < bestd) { bestd = d; ex = e->x; ey = e->y; }
                }
                if (ex >= 0) {
                    int cx0 = mine[myCarrier].px, cy0 = mine[myCarrier].py;
                    int bx = cx0, by = cy0, bd2 = 1 << 29;
                    for (int d = 0; d < 5; d++) {
                        int nx = cx0 + DX[d], ny = cy0 + DY[d];
                        if (d < 4 && isWall(nx, ny)) continue;
                        int dd = abs(nx - ex) + abs(ny - ey);
                        if (dd < bd2) { bd2 = dd; bx = nx; by = ny; }
                    }
                    mine[k].tx = bx; mine[k].ty = by; mine[k].note = "block_position";
                } else { mine[k].tx = mine[myCarrier].px; mine[k].ty = mine[myCarrier].py; mine[k].note = "escort"; }
            } else {
                mine[k].role = 3;
                double best = 1e18; int sx = centerx, sy = centery;
                for (auto& sp : g.spots) {
                    int c = cid(sp.first, sp.second);
                    if (spotTaken[c]) continue;
                    bool hasFlag = false;
                    for (auto& f : g.flags) if (f.status == "home" && f.x == sp.first && f.y == sp.second) hasFlag = true;
                    if (hasFlag) continue;
                    int d = distU[k][c];
                    if (d < 0) continue;
                    double cost = d + 0.8 * teamBaseDist[g.me][c] + 5.0 * threatNext[c];
                    if (cost < best) { best = cost; sx = sp.first; sy = sp.second; }
                }
                spotTaken[cid(sx, sy)] = 1;
                mine[k].tx = sx; mine[k].ty = sy;
                mine[k].note = "occupy";
            }
        }

        // ----------  ----------
        vector<int> order = {0, 1, 2};
        sort(order.begin(), order.end(), [&](int a, int b) {
            if (mine[a].alive != mine[b].alive) return mine[a].alive > mine[b].alive;
            return mine[a].role < mine[b].role;
        });
        vector<int> reserved(CELLS, 0), stays(CELLS, 0);
        for (int k = 0; k < 3; k++) if (mine[k].alive) stays[cid(mine[k].px, mine[k].py)] = 1;

        for (int oi = 0; oi < 3; oi++) {
            int k = order[oi];
            if (!mine[k].alive) continue;
            vector<double> w(CELLS, 1.0);
            // 
            double frag = mine[k].u.hp <= g.damage ? 3.0 : (mine[k].u.hp <= 2 * g.damage ? 1.6 : 1.0);
            for (int c = 0; c < CELLS; c++) {
                if (occupied[c] >= 0 && g.units[occupied[c]].team != g.me) w[c] += 15;
                if (mine[k].role == 0) w[c] += ((8.0 + 17.0 * crowd) * threatNow[c] + (1.0 + 2.0 * crowd) * threatNext[c]) * frag;
                else if (mine[k].role == 1) w[c] += (3.0 * threatNow[c] + 0.5 * threatNext[c]) * frag * crowd;
                else if (mine[k].role == 3) w[c] += 2.0 * threatNow[c] * frag * crowd;
            }
            int tc = cid(mine[k].tx, mine[k].ty);
            bool enemyOnTarget = occupied[tc] >= 0 && g.units[occupied[tc]].team != g.me;
            // 
            bool hotSit = mine[k].role == 1 && mine[k].fid >= 0 && hotFlag[mine[k].fid] &&
                          mine[k].px == mine[k].tx && mine[k].py == mine[k].ty;
            bool hotAdj = mine[k].role == 1 && mine[k].fid >= 0 && hotFlag[mine[k].fid] && enemyOnTarget &&
                          abs(mine[k].px - mine[k].tx) + abs(mine[k].py - mine[k].ty) <= 1;
            if (!enemyOnTarget) w[tc] = 1.0;
            auto cf = costField(mine[k].tx, mine[k].ty, w);
            vector<pair<double, int>> cand;
            for (int d = 0; d < 5; d++) {
                int nx = mine[k].px + DX[d], ny = mine[k].py + DY[d];
                if (d < 4 && isWall(nx, ny)) continue;
                int nc = cid(nx, ny);
                double sc = cf[nc] + (d == 4 ? 0.4 : 0.0);
                if (mine[k].role == 0) sc += 3.0 * threatNow[nc];
                if (nc == banCell[k]) sc += 6.0;           // 
                if (hotSit || hotAdj) sc += (d == 4 ? -100.0 : 100.0);  // 
                cand.push_back({sc, d});
            }
            sort(cand.begin(), cand.end());
            int chosen = 4;
            for (auto& [sc, d] : cand) {
                int nx = mine[k].px + DX[d], ny = mine[k].py + DY[d];
                int nc = cid(nx, ny);
                if (d != 4) {
                    if (reserved[nc]) continue;
                    int occ = occupied[nc];
                    if (occ >= 0 && g.units[occ].team == g.me && stays[nc]) continue;
                    if (nc == banCell[k] && stuck[k] >= 2) continue;   // 
                }
                chosen = d;
                break;
            }
            int nx = mine[k].px + DX[chosen], ny = mine[k].py + DY[chosen];
            mine[k].move = chosen;
            if (chosen == 4 || isWall(nx, ny)) {
                reserved[cid(mine[k].px, mine[k].py)] = 1;   // occupy
            } else {
                stays[cid(mine[k].px, mine[k].py)] = 0;
                mine[k].px = nx; mine[k].py = ny;
                reserved[cid(nx, ny)] = 1;
                stays[cid(nx, ny)] = 1;
            }
        }

        // ----------  ----------
        // fetch_flag P
        auto flagContender = [&](int k) -> int {
            if (mine[k].role != 1) return -1;
            int fc = cid(mine[k].tx, mine[k].ty);
            int occ = occupied[fc];
            if (occ >= 0 && g.units[occ].team != g.me) return occ;   // 
            if (stuck[k] >= 1) {
                // /
                for (auto e : enemies) {
                    int d = abs(e->x - mine[k].tx) + abs(e->y - mine[k].ty);
                    if (d <= 1) return e->id;
                }
            }
            return -1;
        };
        for (int k = 0; k < 3; k++) {
            if (!mine[k].alive || mine[k].u.flag >= 0) continue;
            int cont = flagContender(k);
            int onFid = -1;
            for (auto& f : g.flags)
                if ((f.status == "home" || f.status == "dropped") && f.x == mine[k].px && f.y == mine[k].py) onFid = f.id;
            if (onFid >= 0) {
                // 
                int fc = cid(mine[k].px, mine[k].py);
                int enemyNear = 0, myNear = 0;
                for (auto e : enemies) if (abs(e->x - mine[k].px) + abs(e->y - mine[k].py) <= 2) enemyNear++;
                for (int j = 0; j < 3; j++) if (j != k && mine[j].alive && abs(mine[j].px - mine[k].px) + abs(mine[j].py - mine[k].py) <= 3) myNear++;
                int bd = teamBaseDist[g.me][fc];
                //  => 
                bool safe = enemyNear == 0 || myNear >= enemyNear || bd <= 4 ||
                            (enemyNear <= 1 && threatNow[fc] == 0 && bd <= 7);
                if (safe) { mine[k].act = 1; mine[k].note = "pickup"; continue; }
                mine[k].note = "deny_flag";   // 
            }
            if (cont >= 0 && !mine[k].immune) { mine[k].act = 3; mine[k].actArg = cont; mine[k].note = "contest_flag"; }
        }

        // 
        vector<pair<int, int>> ePred(g.units.size(), {-1, -1});
        for (auto& u : g.units) {
            if (u.team == g.me || u.x < 0) continue;
            int ex = u.x, ey = u.y;
            if (u.flag >= 0) {
                int bd = teamBaseDist[u.team][cid(ex, ey)], bx = ex, by = ey;
                for (int d = 0; d < 4; d++) {
                    int nx = ex + DX[d], ny = ey + DY[d];
                    if (isWall(nx, ny)) continue;
                    int nd = teamBaseDist[u.team][cid(nx, ny)];
                    if (nd >= 0 && nd < bd) { bd = nd; bx = nx; by = ny; }
                }
                ex = bx; ey = by;
            }
            ePred[u.id] = {ex, ey};
        }
        vector<int> shooters;
        for (int k = 0; k < 3; k++)
            if (mine[k].alive && mine[k].act == 0 && mine[k].u.flag < 0 && !mine[k].immune) shooters.push_back(k);
        struct Tgt { int uid; double val; int need; vector<int> by; };
        vector<Tgt> tgts;
        for (auto e : enemies) {
            if (e->x < 0) continue;
            auto pp = ePred[e->id];
            vector<int> by;
            for (int k : shooters) if (canAtk(mine[k].px, mine[k].py, pp.first, pp.second)) by.push_back(k);
            // “contest_flag”
            for (int k = 0; k < 3; k++)
                if (mine[k].act == 3 && mine[k].actArg == e->id &&
                    find(by.begin(), by.end(), k) == by.end() &&
                    find(shooters.begin(), shooters.end(), k) == shooters.end()) {
                    if (canAtk(mine[k].px, mine[k].py, pp.first, pp.second)) by.push_back(k);
                }
            if (by.empty()) continue;
            double val = 1.0 * (g.hp - e->hp) + 2.0 * max(0, 3 - (abs(pp.first - mine[by[0]].px) + abs(pp.second - mine[by[0]].py)));
            if (e->flag >= 0) {
                int tb = teamBaseDist[e->team][cid(e->x, e->y)];
                val += 150 + 10.0 * max(0, 20 - tb);
            }
            // 
            for (int k = 0; k < 3; k++)
                if (mine[k].role == 1 && abs(e->x - mine[k].tx) + abs(e->y - mine[k].ty) <= 2) { val += 80; break; }
            // 
            if (teamBaseDist[g.me][cid(e->x, e->y)] <= 1) val += 60;
            int need = (e->hp + g.damage - 1) / g.damage;
            if ((int)by.size() >= need) val += 300;
            tgts.push_back({e->id, val, need, by});
        }
        sort(tgts.begin(), tgts.end(), [](const Tgt& a, const Tgt& b) { return a.val > b.val; });
        vector<int> used(3, 0);
        for (int k = 0; k < 3; k++) if (mine[k].act == 3) used[k] = 1;
        for (auto& t : tgts) {
            int assigned = 0;
            for (int k : t.by) if (mine[k].act == 3 && mine[k].actArg == t.uid) assigned++;
            for (int k : t.by) {
                if (used[k]) continue;
                mine[k].act = 3; mine[k].actArg = t.uid;
                used[k] = 1;
                if (++assigned >= t.need) break;
            }
        }

        // 
        for (int k = 0; k < 3; k++) {
            if (mine[k].alive) { prevPred[k][0] = mine[k].px; prevPred[k][1] = mine[k].py; }
            else { prevPred[k][0] = prevPred[k][1] = -1; stuck[k] = 0; banCell[k] = -1; }
        }

        // ----------  ----------
        string out = to_string(g.turn);
        vector<string> notes(3);
        for (int k = 0; k < 3; k++) {
            char mv = mine[k].alive ? MVC[mine[k].move] : 'S';
            string act = "-";
            if (mine[k].alive) {
                if (mine[k].act == 1) act = "P";
                else if (mine[k].act == 2) act = "X";
                else if (mine[k].act == 3) act = to_string(mine[k].actArg);
            }
            out += " "; out += mv; out += " "; out += act;
            notes[k] = mine[k].note;
        }
        out += " # " + notes[0] + ";" + notes[1] + ";" + notes[2];
        return out;
    }
};

int main() {
    ios::sync_with_stdio(false);
    Solver s;
    bool first = true;
    while (read_turn(s.g)) {
        if (first) { s.precompute(); first = false; }
        cout << s.solve() << endl;
    }
}
