//  bot 2–15 
//
// 
//  · 2–3  3 return_home / steal_flag /  /
//    steal_flag / escort / intercept /  Dijkstra 
//    
//  · ≥4 ——
//    "shortest_roundtrip"pickupreturn_home
//    10 
//
// /pickup
// 
// 
#include <bits/stdc++.h>
using namespace std;

struct Unit { int id, team, x, y, hp, flag, respawn_at; };
struct FlagT { int id; string status; int x, y, holder, return_at; };
struct EventT { string type; vector<int> args; };

struct Game {
    int teams = 0, me = 0, size = 0, totalTurns = 0, maxhp = 0, dmg = 0, respT = 0, flagRet = 0, flagCd = 0;
    vector<string> grid;
    vector<pair<int,int>> bases, spots;
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<FlagT> flags;
    vector<EventT> events;
};

bool read_turn(Game& g) {
    string w;
    while (cin >> w) {
        if (w == "INIT") { cin >> g.teams >> g.me >> g.size >> g.totalTurns >> g.maxhp >> g.dmg >> g.respT >> g.flagRet >> g.flagCd; }
        else if (w == "MAP") { g.grid.assign(g.size, string()); for (auto& r : g.grid) cin >> r; }
        else if (w == "BASES") { int k; cin >> k; g.bases.resize(k); for (auto& b : g.bases) cin >> b.first >> b.second; }
        else if (w == "SPOTS") { int k; cin >> k; g.spots.resize(k); for (auto& s : g.spots) cin >> s.first >> s.second; }
        else if (w == "TURN") { cin >> g.turn; }
        else if (w == "SCORE") { g.score.resize(g.teams); for (auto& s : g.score) cin >> s; }
        else if (w == "UNITS") { int k; cin >> k; g.units.resize(k); for (auto& u : g.units) { string tag; cin >> tag >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawn_at; } }
        else if (w == "FLAGS") { int k; cin >> k; g.flags.resize(k); for (auto& f : g.flags) { string tag; cin >> tag >> f.id >> f.status >> f.x >> f.y >> f.holder >> f.return_at; } }
        else if (w == "EVENTS") { int k; cin >> k; g.events.assign(k, {}); string line; getline(cin, line); for (auto& e : g.events) { getline(cin, line); istringstream in(line); string tag; in >> tag >> e.type; for (int v; in >> v;) e.args.push_back(v); } }
        else if (w == "END") return true;
    }
    return false;
}

// ----------------  ----------------
int S, N, ME, MAXHP, DMG, TOTALTURNS;
vector<string> G;
vector<vector<int>> dB;               // dB[team][cell] = (3x3) BFS 
vector<float> baseDanger;             // 
vector<pair<int,int>> spotsG, myBase9, mySpawnOrder;
pair<int,int> C0;
bool inited = false;

inline int ID(int x, int y) { return y * S + x; }
inline bool openc(int x, int y) { return x >= 0 && x < S && y >= 0 && y < S && G[y][x] == '.'; }
inline int md(int ax, int ay, int bx, int by) { return abs(ax - bx) + abs(ay - by); }

bool canAtk(int ax, int ay, int bx, int by) {
    int dx = bx - ax, dy = by - ay;
    if (abs(dx) + abs(dy) > 2) return false;
    if (abs(dx) == 2 && dy == 0) return openc(ax + dx / 2, ay);
    if (abs(dy) == 2 && dx == 0) return openc(ax, ay + dy / 2);
    if (abs(dx) == 1 && abs(dy) == 1) return openc(ax + dx, ay) || openc(ax, ay + dy);
    return true;
}

vector<int> bfsCells(const vector<pair<int,int>>& src) {
    vector<int> d(S * S, INT_MAX);
    deque<int> q;
    for (auto& p : src) if (openc(p.first, p.second) && d[ID(p.first, p.second)] == INT_MAX) { d[ID(p.first, p.second)] = 0; q.push_back(ID(p.first, p.second)); }
    const int DX[4] = {1,-1,0,0}, DY[4] = {0,0,1,-1};
    while (!q.empty()) {
        int c = q.front(); q.pop_front();
        int x = c % S, y = c / S, dc = d[c] + 1;
        for (int i = 0; i < 4; i++) {
            int nx = x + DX[i], ny = y + DY[i];
            if (openc(nx, ny)) { int nc = ID(nx, ny); if (d[nc] > dc) { d[nc] = dc; q.push_back(nc); } }
        }
    }
    return d;
}

void doInit(Game& g) {
    S = g.size; N = g.teams; ME = g.me; MAXHP = g.maxhp; DMG = g.dmg; TOTALTURNS = g.totalTurns; G = g.grid;
    C0 = {S / 2, S / 2};
    baseDanger.assign(S * S, 0.f);
    for (int k = 0; k < N; k++) {
        int bx = g.bases[k].first, by = g.bases[k].second;
        vector<pair<int,int>> cells;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) cells.push_back({bx + dx, by + dy});
        dB.push_back(bfsCells(cells));
        if (k == ME) myBase9 = cells;
        else for (int c = 0; c < S * S; c++) {
            int dd = dB[k][c];
            baseDanger[c] += dd == 0 ? 0.8f : dd == 1 ? 0.35f : dd == 2 ? 0.12f : 0.f;
        }
    }
    int bcx = g.bases[ME].first, bcy = g.bases[ME].second;
    mySpawnOrder = myBase9;
    sort(mySpawnOrder.begin(), mySpawnOrder.end(), [&](const pair<int,int>& a, const pair<int,int>& b) {
        int da = md(a.first, a.second, C0.first, C0.second), db = md(b.first, b.second, C0.first, C0.second);
        if (da != db) return da < db;
        return abs(a.first - bcx) + 2 * abs(a.second - bcy) < abs(b.first - bcx) + 2 * abs(b.second - bcy);
    });
    spotsG = g.spots;
    inited = true;
}

// 
enum { R_CARRY = 0, R_FETCH, R_CLEAR, R_SUPPORT, R_ESCORT, R_HUNT, R_CAMP, R_IDLE, R_DEAD };


// /return_home
string decideSimple(Game& g) {
    int T = g.turn;
    const int DX[4] = {1,-1,0,0}, DY[4] = {0,0,1,-1};
    Unit raw[3] = {};
    for (auto& u : g.units) if (u.team == ME) raw[u.id - 3 * ME] = u;
    set<int> occ;
    for (auto& u : g.units) if (u.x >= 0) occ.insert(ID(u.x, u.y));
    int ux[3], uy[3], uhp[3], uflag[3]; bool alive[3], respNow[3];
    for (int k = 0; k < 3; k++) {
        ux[k] = raw[k].x; uy[k] = raw[k].y; uhp[k] = raw[k].hp; uflag[k] = raw[k].flag;
        alive[k] = ux[k] >= 0; respNow[k] = false;
        if (!alive[k] && raw[k].respawn_at == T)
            for (auto& p : mySpawnOrder) { int c = ID(p.first, p.second);
                if (!occ.count(c)) { ux[k] = p.first; uy[k] = p.second; uhp[k] = MAXHP; uflag[k] = -1; alive[k] = true; respNow[k] = true; occ.insert(c); break; } }
    }
    vector<Unit> en;
    for (auto& u : g.units) if (u.team != ME && u.x >= 0) en.push_back(u);
    // 
    vector<float> danger(S * S, 0.f);
    for (auto& e : en) {
        if (e.flag >= 0) continue;
        for (int dy = -4; dy <= 4; dy++) for (int dx = -4; dx <= 4; dx++) {
            int d = abs(dx) + abs(dy); if (d == 0 || d > 4) continue;
            int x = e.x + dx, y = e.y + dy;
            if (!openc(x, y)) continue;
            float w = 0;
            if (d <= 2) w = canAtk(e.x, e.y, x, y) ? 1.f : 0.3f;
            else if (d == 3) w = 0.45f; else w = 0.12f;
            danger[ID(x, y)] += w;
        }
    }
    for (int k = 0; k < 3; k++) if (alive[k]) {
        for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
            if (abs(dx) + abs(dy) == 0) continue;
            int x = ux[k] + dx, y = uy[k] + dy;
            if (openc(x, y)) danger[ID(x, y)] -= 0.25f;
        }
    }
    for (int c = 0; c < S * S; c++) if (danger[c] < 0.f) danger[c] = 0.f;
    // 
    vector<pair<int,int>> afv;
    for (auto& f : g.flags) if ((f.status == "home" || f.status == "dropped") && f.return_at != T) afv.push_back({f.x, f.y});
    string out = to_string(T);
    set<int> of;
    for (int k = 0; k < 3; k++) if (alive[k]) of.insert(ID(ux[k], uy[k]));
    for (int k = 0; k < 3; k++) {
        string mv = "S", act = "-";
        if (alive[k]) {
            vector<pair<int,int>> targs;
            if (uflag[k] >= 0) targs = myBase9;
            else {
                int bd = INT_MAX; pair<int,int> best{-1,-1};
                for (auto& p : afv) {
                    int d = md(ux[k], uy[k], p.first, p.second) + dB[ME][ID(p.first, p.second)];
                    if (d < bd) { bd = d; best = p; }
                }
                if (best.first >= 0) targs = {best};
            }
            // BFS 
            int nxk = ux[k], nyk = uy[k];
            if (!targs.empty()) {
                float wD = (uflag[k] >= 0) ? 1.5f : 0.6f;    // 
                vector<float> D(S * S, 1e18f);
                priority_queue<pair<float,int>, vector<pair<float,int>>, greater<pair<float,int>>> pq;
                for (auto& p : targs) { int c = ID(p.first, p.second); if (openc(p.first, p.second) && D[c] > 0) { D[c] = 0.f; pq.push({0.f, c}); } }
                while (!pq.empty()) {
                    pair<float,int> pr = pq.top(); pq.pop();
                    float dd = pr.first; int c = pr.second;
                    if (dd > D[c] + 1e-6f) continue;
                    int x = c % S, y = c / S;
                    for (int i = 0; i < 4; i++) {
                        int ax = x + DX[i], ay = y + DY[i];
                        if (!openc(ax, ay)) continue;
                        int ac = ID(ax, ay);
                        float cost = 1.f + wD * danger[ac];
                        if (dd + cost < D[ac] - 1e-6f) { D[ac] = dd + cost; pq.push({D[ac], ac}); }
                    }
                }
                int cur = ID(ux[k], uy[k]);
                if (D[cur] > 0.f && D[cur] < 1e17f) {
                    float bc = 1e18f; int bs = -1;
                    for (int i = 0; i < 4; i++) {
                        int ax = ux[k] + DX[i], ay = uy[k] + DY[i];
                        if (!openc(ax, ay)) continue;
                        int ac = ID(ax, ay);
                        if (of.count(ac)) continue;
                        bool enemyOn = false;
                        for (auto& e : en) if (e.x == ax && e.y == ay) enemyOn = true;
                        if (enemyOn) continue;
                        float v = 1.f + wD * danger[ac] + D[ac];
                        if (v < bc) { bc = v; bs = ac; }
                    }
                    if (bs >= 0) { of.erase(cur); of.insert(bs); nxk = bs % S; nyk = bs / S; }
                }
            }
            int ddx = nxk - ux[k], ddy = nyk - uy[k];
            if (ddx == 1) mv = "R"; else if (ddx == -1) mv = "L"; else if (ddy == 1) mv = "D"; else if (ddy == -1) mv = "U";
            if (uflag[k] < 0) {
                bool onF = false;
                for (auto& f : g.flags) if ((f.status == "home" || f.status == "dropped") && f.return_at != T && f.x == nxk && f.y == nyk) onF = true;
                if (onF) act = "P";
                else if (!respNow[k]) {
                    int bestT = -1; float bs2 = -1e9;
                    for (auto& e : en) {
                        if (!canAtk(nxk, nyk, e.x, e.y)) continue;
                        float s2 = (e.flag >= 0 ? 100.f : 0.f) + (MAXHP - e.hp) * 0.3f;
                        if (s2 > bs2) { bs2 = s2; bestT = e.id; }
                    }
                    if (bestT >= 0) act = to_string(bestT);
                }
            }
        }
        out += " " + mv + " " + act;
    }
    return out;
}

string decide(Game& g) {
    if (!inited) doInit(g);
    int T = g.turn, total = TOTALTURNS;
    if (N >= 4) return decideSimple(g);   // 4 steal_flag
    const int DX[4] = {1,-1,0,0}, DY[4] = {0,0,1,-1};

    // ---- 
    Unit raw[3] = {};
    for (auto& u : g.units) if (u.team == ME) raw[u.id - 3 * ME] = u;
    int ux[3], uy[3], uhp[3], uflag[3]; bool alive[3], respNow[3];
    set<int> occ;
    for (auto& u : g.units) if (u.x >= 0) occ.insert(ID(u.x, u.y));
    for (int k = 0; k < 3; k++) {
        ux[k] = raw[k].x; uy[k] = raw[k].y; uhp[k] = raw[k].hp; uflag[k] = raw[k].flag;
        alive[k] = ux[k] >= 0; respNow[k] = false;
        if (!alive[k] && raw[k].respawn_at == T) {
            for (auto& p : mySpawnOrder) {
                int c = ID(p.first, p.second);
                if (occ.find(c) == occ.end()) { ux[k] = p.first; uy[k] = p.second; uhp[k] = MAXHP; uflag[k] = -1; alive[k] = true; respNow[k] = true; occ.insert(c); break; }
            }
        }
    }
    vector<Unit> en;
    for (auto& u : g.units) if (u.team != ME && u.x >= 0) en.push_back(u);

    // ---- 
    vector<float> danger(S * S, 0.f);
    for (auto& e : en) {
        if (e.flag >= 0) continue;                    // 
        for (int dy = -4; dy <= 4; dy++) for (int dx = -4; dx <= 4; dx++) {
            int d = abs(dx) + abs(dy); if (d == 0 || d > 4) continue;
            int x = e.x + dx, y = e.y + dy;
            if (!openc(x, y)) continue;
            float w = 0;
            if (d <= 2) w = canAtk(e.x, e.y, x, y) ? 1.f : 0.3f;
            else if (d == 3) w = 0.45f; else w = 0.12f;
            danger[ID(x, y)] += w;
        }
    }
    for (int c = 0; c < S * S; c++) danger[c] += baseDanger[c];
    // escort/
    for (int k = 0; k < 3; k++) if (alive[k]) {
        for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
            int d = abs(dx) + abs(dy); if (d == 0) continue;
            int x = ux[k] + dx, y = uy[k] + dy;
            if (!openc(x, y)) continue;
            danger[ID(x, y)] -= 0.25f;
        }
    }
    for (int c = 0; c < S * S; c++) if (danger[c] < 0.f) danger[c] = 0.f;

    //  3 
    vector<int> enN(S * S, 0), allySplash(S * S, 0);
    for (auto& e : en) {
        if (e.flag >= 0) continue;
        for (int dy = -3; dy <= 3; dy++) for (int dx = -3; dx <= 3; dx++) {
            if (abs(dx) + abs(dy) > 3) continue;
            int x = e.x + dx, y = e.y + dy;
            if (openc(x, y)) enN[ID(x, y)]++;
        }
    }
    for (int k = 0; k < 3; k++) if (alive[k]) {
        for (int dy = -3; dy <= 3; dy++) for (int dx = -3; dx <= 3; dx++) {
            if (abs(dx) + abs(dy) > 3) continue;
            int x = ux[k] + dx, y = uy[k] + dy;
            if (openc(x, y)) allySplash[ID(x, y)]++;
        }
    }

    // ---- 
    struct AF { int x, y, id, dh; };
    vector<AF> af; vector<vector<int>> fdist;
    for (auto& f : g.flags) if ((f.status == "home" || f.status == "dropped") && f.return_at != T) {
        af.push_back({f.x, f.y, f.id, dB[ME][ID(f.x, f.y)]});
        fdist.push_back(bfsCells({{f.x, f.y}}));
    }
    int naf = af.size();

    // ---- 
    int role[3], parm[3], parm2[3];
    for (int k = 0; k < 3; k++) { role[k] = alive[k] ? R_IDLE : R_DEAD; parm[k] = parm2[k] = -1; }
    for (int k = 0; k < 3; k++) if (alive[k] && uflag[k] >= 0) role[k] = R_CARRY;

    // steal_flag =  - 
    struct PR { float cost; int u, f, margin; };
    vector<PR> prs;
    int centerD = dB[ME][ID(C0.first, C0.second)];
    vector<int> enNear(naf, INT_MAX);      // 
    for (int i = 0; i < naf; i++) for (auto& e : en) {
        int d2 = fdist[i][ID(e.x, e.y)];
        if (d2 < enNear[i]) enNear[i] = d2;
    }
    for (int k = 0; k < 3; k++) if (alive[k] && role[k] == R_IDLE) {
        for (int i = 0; i < naf; i++) {
            int d1 = fdist[i][ID(ux[k], uy[k])]; if (d1 == INT_MAX) continue;
            int myT = d1 + af[i].dh;
            if (T + myT + 2 > total) continue;
            int enT = INT_MAX;
            for (auto& e : en) {
                int d2 = fdist[i][ID(e.x, e.y)]; if (d2 == INT_MAX) continue;
                int v = d2 + dB[e.team][ID(af[i].x, af[i].y)];
                if (v < enT) enT = v;
            }
            if (uhp[k] <= 34 && enNear[i] <= 4) continue;                 // pickup
            if (uhp[k] <= 34 && af[i].dh > centerD * 2 / 3 && enNear[i] <= 6) continue;   // 
            int margin = (enT == INT_MAX) ? 8 : enT - myT;
            //  1 
            int closerLim = max(1, N / 3);   // 
            int closer = 0;
            for (auto& e : en) if (e.flag < 0 && fdist[i][ID(e.x, e.y)] < d1) closer++;
            bool homeZone = af[i].dh <= centerD * 2 / 3;
            if (N >= 5 && af[i].dh > centerD * 3 / 4) continue;   // 
                        if (margin < -8 && !(homeZone || (N >= 3 && closer <= closerLim))) continue;
            prs.push_back({(float)(myT - 0.6 * max(min(margin, 8), -8) - (homeZone ? 3 : 0)) + (MAXHP - uhp[k]) * 0.3f, k, i, margin});
        }
    }
    sort(prs.begin(), prs.end(), [](const PR& a, const PR& b) { return a.cost < b.cost; });
    vector<char> fTaken(naf, 0);
    int fMargin[3]; bool supported[3];
    for (int k = 0; k < 3; k++) { fMargin[k] = 1000; supported[k] = false; }
    for (auto& p : prs) if (role[p.u] == R_IDLE && !fTaken[p.f]) { role[p.u] = R_FETCH; parm[p.u] = p.f; fTaken[p.f] = 1; fMargin[p.u] = p.margin; }

    vector<int> left;
    for (int k = 0; k < 3; k++) if (alive[k] && role[k] == R_IDLE) left.push_back(k);

    // 
    if (!left.empty() && N >= 3) {
        int mySpotIdx = 0, bd0 = INT_MAX;
        for (int j = 0; j < (int)spotsG.size(); j++) {
            int d = dB[ME][ID(spotsG[j].first, spotsG[j].second)];
            if (d < bd0) { bd0 = d; mySpotIdx = j; }
        }
        bool fetcherThere = false;
        for (int i = 0; i < naf; i++)
            if (fTaken[i] && af[i].x == spotsG[mySpotIdx].first && af[i].y == spotsG[mySpotIdx].second) fetcherThere = true;
        int bi = 0, bdu = INT_MAX;
        for (int i = 0; i < (int)left.size(); i++) {
            int d = md(ux[left[i]], uy[left[i]], spotsG[mySpotIdx].first, spotsG[mySpotIdx].second);
            if (d < bdu) { bdu = d; bi = i; }
        }
        // steal_flag
        int sitIdx = mySpotIdx;
        if (fetcherThere) {
            int bc2 = -1, bdc = INT_MAX;
            for (int j = 0; j < (int)spotsG.size(); j++) {
                int d = dB[ME][ID(spotsG[j].first, spotsG[j].second)];
                if (d < bdc && d > bd0) { bdc = d; bc2 = j; }
            }
            if (bc2 >= 0) sitIdx = bc2;
        }
        role[left[bi]] = R_CAMP; parm[left[bi]] = sitIdx; left.erase(left.begin() + bi);
    }

    // 
    {
        vector<int> campers;
        for (auto& e : en) if (dB[ME][ID(e.x, e.y)] <= 2) campers.push_back(e.id);
        vector<char> camTaken(campers.size(), 0);
        while (!left.empty()) {
            int bi = -1, bj = -1, bd = INT_MAX;
            for (int i = 0; i < (int)left.size(); i++) for (int j = 0; j < (int)campers.size(); j++) {
                if (camTaken[j]) continue;
                int d = md(ux[left[i]], uy[left[i]], g.units[campers[j]].x, g.units[campers[j]].y);
                if (d < bd) { bd = d; bi = i; bj = j; }
            }
            if (bi < 0) break;
            role[left[bi]] = R_CLEAR; parm[left[bi]] = campers[bj]; camTaken[bj] = 1;
            left.erase(left.begin() + bi);
        }
    }

    // escortsteal_flagsteal_flagescort
    for (int cs = 0; cs < 3 && left.empty() && N >= 3; cs++) if (alive[cs] && uflag[cs] >= 0) {
        int bk = -1; float bcost = -1e9;
        for (int k = 0; k < 3; k++) if (role[k] == R_FETCH) {
            float c = (float)fMargin[k] + af[parm[k]].dh * 0.3f;   // 
            if (c > bcost) { bcost = c; bk = k; }
        }
        if (bk >= 0) { left.push_back(bk); role[bk] = R_IDLE; fTaken[parm[bk]] = 0; }
    }

    // escort
    // steal_flag
    auto stealFetcher = [&]() -> bool {
        int bk = -1; float bcost = -1e9;
        for (int k = 0; k < 3; k++) if (role[k] == R_FETCH) {
            float c = (float)fMargin[k] + af[parm[k]].dh * 0.3f;
            if (c > bcost) { bcost = c; bk = k; }
        }
        if (bk >= 0) { left.push_back(bk); role[bk] = R_IDLE; fTaken[parm[bk]] = 0; return true; }
        return false;
    };
    for (int cs = 0; cs < 3; cs++) if (alive[cs] && uflag[cs] >= 0) {
        int nearE = INT_MAX, threat = 0;
        for (auto& e : en) {
            int d = md(e.x, e.y, ux[cs], uy[cs]);
            if (d < nearE) nearE = d;
            if (d <= 4) threat++;
        }
        if (nearE > (N <= 2 ? 8 : 12)) continue;
        int need = (N >= 5) ? 2 : (threat >= 2 ? 2 : 1);
        int got = 0;
        while (got < need && (!left.empty() || (N >= 3 && stealFetcher()))) {
            int bi = 0, bd = INT_MAX;
            for (int i = 0; i < (int)left.size(); i++) { int d = md(ux[left[i]], uy[left[i]], ux[cs], uy[cs]); if (d < bd) { bd = d; bi = i; } }
            role[left[bi]] = R_ESCORT; parm[left[bi]] = cs; left.erase(left.begin() + bi);
            got++;
        }
    }

    // steal_flag 6 
    while (!left.empty()) {
        int bk = -1; int bestc = INT_MAX;
        for (int k = 0; k < 3; k++) if (role[k] == R_FETCH && !supported[k]) {
            int d1 = fdist[parm[k]][ID(ux[k], uy[k])];
            int contest = (enNear[parm[k]] == INT_MAX) ? INT_MAX : enNear[parm[k]] - d1;
            if (contest < bestc && contest < 6) { bestc = contest; bk = k; }
        }
        if (bk < 0) break;
        supported[bk] = 1;
        int bi = 0, bd = INT_MAX;
        for (int i = 0; i < (int)left.size(); i++) { int d = md(ux[left[i]], uy[left[i]], ux[bk], uy[bk]); if (d < bd) { bd = d; bi = i; } }
        // 
        role[left[bi]] = R_SUPPORT; parm[left[bi]] = bk; parm2[left[bi]] = parm[bk]; left.erase(left.begin() + bi);
    }

    // interceptreturn_home
    for (int li = 0; li < (int)left.size(); ) {
        int k = left[li];
        int bestPrey = -1, bestCell = -1; float bestAdv = 1e9f;
        if (N > 6) { li++; continue; }
        for (auto& e : en) {
            if (e.flag < 0) continue;
            int home = dB[e.team][ID(e.x, e.y)]; if (home < 3) continue;
            int allies = 0;
            for (auto& a : en) if (a.id != e.id && a.team == e.team && a.flag < 0 && md(a.x, a.y, e.x, e.y) <= 2) allies++;
            if (allies >= 2) continue;
            vector<int> myD = bfsCells({{ux[k], uy[k]}});
            for (int c = 0; c < S * S; c++) {
                if (myD[c] == INT_MAX) continue;
                int dp = dB[e.team][c]; if (dp < 2 || dp >= home) continue;
                float adv = (float)myD[c] + dp - home;
                if (adv < bestAdv) { bestAdv = adv; bestPrey = e.id; bestCell = c; }
            }
        }
        if (bestPrey >= 0 && bestAdv <= -1.f) { role[k] = R_HUNT; parm[k] = bestPrey; parm2[k] = bestCell; left.erase(left.begin() + li); }
        else li++;
    }

    // steal_flag
    if (!left.empty()) {
        vector<char> spotTaken(spotsG.size(), 0);
        int nCamp = 0;
        for (int li = 0; li < (int)left.size(); li++) {
            int k = left[li];
            if (nCamp >= 1 && N >= 5) {
                int bi = -1, bc = INT_MAX;
                for (int i = 0; i < naf; i++) {
                    if (fTaken[i] || af[i].dh > centerD * 3 / 4) continue;
                    int d1 = fdist[i][ID(ux[k], uy[k])];
                    if (d1 == INT_MAX || T + d1 + af[i].dh + 2 > total) continue;
                    if (d1 + af[i].dh < bc) { bc = d1 + af[i].dh; bi = i; }
                }
                if (bi >= 0) { role[k] = R_FETCH; parm[k] = bi; parm2[k] = bi; fTaken[bi] = 1; fMargin[k] = 0; supported[k] = true; continue; }
            }
            int bj = -1, bs = INT_MAX;
            for (int j = 0; j < (int)spotsG.size(); j++) {
                if (spotTaken[j]) continue;
                int dhome = dB[ME][ID(spotsG[j].first, spotsG[j].second)];
                if (dhome > centerD + 2) continue;                // 
                bool fetcherGoing = false;
                for (int i = 0; i < naf; i++) if (fTaken[i] && af[i].x == spotsG[j].first && af[i].y == spotsG[j].second) fetcherGoing = true;
                if (fetcherGoing) continue;
                int sc = 2 * dhome + md(ux[k], uy[k], spotsG[j].first, spotsG[j].second);
                if (sc < bs) { bs = sc; bj = j; }
            }
            if (bj >= 0) { role[k] = R_CAMP; parm[k] = bj; spotTaken[bj] = 1; nCamp++; }
            else if (dB[ME][ID(ux[k], uy[k])] > 3) { role[k] = R_CAMP; parm[k] = -1; }
        }
    }

    // 3  2 
    for (int k = 0; k < 3; k++) {
        if (!alive[k] || uflag[k] >= 0 || role[k] == R_CARRY || role[k] == R_CLEAR || N > 6) continue;
        int bc = -1, bdd = 4;
        for (auto& e : en) if (e.flag >= 0) {
            int d = md(ux[k], uy[k], e.x, e.y);
            if (d < bdd) { bdd = d; bc = ID(e.x, e.y); }
        }
        if (bc >= 0) {
            bool onFlag = false;
            for (auto& f : g.flags) if ((f.status == "home" || f.status == "dropped") && f.return_at != T && f.x == ux[k] && f.y == uy[k]) onFlag = true;
            if (!onFlag) { role[k] = R_HUNT; parm[k] = -1; parm2[k] = bc; }
        }
    }

    // ---- 
    int order3[3] = {-1, -1, -1}; int oi = 0;
    for (int r = R_CARRY; r <= R_IDLE; r++) for (int k = 0; k < 3; k++) if (role[k] == r) order3[oi++] = k;
    set<int> of;
    for (int k = 0; k < 3; k++) if (alive[k]) of.insert(ID(ux[k], uy[k]));
    int nx[3], ny[3];
    for (int k = 0; k < 3; k++) { nx[k] = ux[k]; ny[k] = uy[k]; }

    vector<float> D;
    for (int o = 0; o < 3; o++) {
        int k = order3[o];
        if (k < 0 || !alive[k] || role[k] == R_IDLE || role[k] == R_DEAD) continue;
        vector<pair<int,int>> targs; float wD; int allow = -1;
        switch (role[k]) {
            case R_CARRY: targs = myBase9; wD = 2.2f; break;
            case R_FETCH: {
                int fx = af[parm[k]].x, fy = af[parm[k]].y;
                int enF = 0, alF = 0;
                for (auto& e : en) if (md(e.x, e.y, fx, fy) <= 3) enF++;
                for (int j = 0; j < 3; j++) if (alive[j] && j != k && md(ux[j], uy[j], fx, fy) <= 3) alF++;
                if (enF >= alF + 2) {
                    //  2 
                    for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
                        int d = abs(dx) + abs(dy);
                        if (d == 0 || d > 2) continue;
                        int x = fx + dx, y = fy + dy;
                        if (openc(x, y) && canAtk(x, y, fx, fy)) targs.push_back({x, y});
                    }
                    if (targs.empty()) targs.push_back({fx, fy});
                } else {
                    targs = {{fx, fy}};
                    allow = ID(fx, fy);
                }
                wD = 1.1f; break;
            }
            case R_CLEAR: {
                Unit& e = g.units[parm[k]];
                for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
                    int x = e.x + dx, y = e.y + dy;
                    if (openc(x, y) && canAtk(x, y, e.x, e.y)) targs.push_back({x, y});
                }
                wD = 0.8f; break;
            }
            case R_SUPPORT: {
                int p = parm[k];
                if (md(ux[k], uy[k], ux[p], uy[p]) <= 1) targs.push_back({ux[k], uy[k]});
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    if (abs(dx) + abs(dy) != 1) continue;
                    int x = ux[p] + dx, y = uy[p] + dy;
                    if (openc(x, y)) targs.push_back({x, y});
                }
                wD = 1.0f; break;
            }
            case R_ESCORT: {
                int p = parm[k];
                if (md(ux[k], uy[k], ux[p], uy[p]) <= 1) targs.push_back({ux[k], uy[k]});
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    if (abs(dx) + abs(dy) != 1) continue;
                    int x = ux[p] + dx, y = uy[p] + dy;
                    if (openc(x, y)) targs.push_back({x, y});
                }
                wD = role[k] == R_SUPPORT ? 1.0f : 0.8f; break;
            }
            case R_HUNT: targs = {{parm2[k] % S, parm2[k] / S}}; wD = 0.7f; break;
            case R_CAMP:
                if (parm[k] >= 0) targs = {spotsG[parm[k]]};
                else targs = myBase9;
                wD = 1.3f; break;
            default: continue;
        }
        if (targs.empty()) continue;
        // 10 
        float thr = (role[k] == R_CARRY) ? 0.45f * uhp[k] : max(0.85f * uhp[k], 0.6f * MAXHP);
        float imbW = (role[k] == R_CARRY) ? 1.0f : (role[k] == R_FETCH || role[k] == R_CAMP) ? 0.7f : 0.25f;
        auto ldanger = [&](int c) -> float {
            float imb = (float)enN[c] - ((float)allySplash[c] - (md(ux[k], uy[k], c % S, c / S) <= 3 ? 1.f : 0.f));
            return danger[c] + imbW * min(max(imb, 0.f), 2.f);
        };
        D.assign(S * S, 1e18f);
        priority_queue<pair<float,int>, vector<pair<float,int>>, greater<pair<float,int>>> pq;
        for (auto& p : targs) { int c = ID(p.first, p.second); if (openc(p.first, p.second) && D[c] > 0) { D[c] = 0.f; pq.push({0.f, c}); } }
        auto pen = [&](int c) -> bool { return 34.f * ldanger(c) >= thr; };
        while (!pq.empty()) {
            pair<float,int> pr = pq.top(); pq.pop();
            float d = pr.first; int c = pr.second;
            if (d > D[c] + 1e-6f) continue;
            int x = c % S, y = c / S;
            for (int i = 0; i < 4; i++) {
                int ax = x + DX[i], ay = y + DY[i];
                if (!openc(ax, ay)) continue;
                int ac = ID(ax, ay);
                float cost = 1.f + wD * ldanger(ac) + ((pen(ac) && ac != allow) ? 30.f : 0.f);
                if (d + cost < D[ac] - 1e-6f) { D[ac] = d + cost; pq.push({D[ac], ac}); }
            }
        }
        int cur = ID(ux[k], uy[k]);
        if (D[cur] <= 0.f || D[cur] > 1e17f) continue;           //  / 
        int best = -1; float bestv = 1e18f;
        for (int i = 0; i < 4; i++) {
            int ax = ux[k] + DX[i], ay = uy[k] + DY[i];
            if (!openc(ax, ay)) continue;
            int ac = ID(ax, ay);
            if (of.count(ac)) continue;                           // /
            bool enemyOn = false;
            for (auto& e : en) if (e.x == ax && e.y == ay) { enemyOn = true; break; }
            if (enemyOn) continue;
            float v = 1.f + wD * ldanger(ac) + ((pen(ac) && ac != allow) ? 30.f : 0.f) + D[ac];
            if (v < bestv) { bestv = v; best = ac; }
        }
        if (best < 0) continue;
        if (pen(best) && !pen(cur) && ldanger(cur) < ldanger(best)) continue;   // 
        of.erase(cur); of.insert(best);
        nx[k] = best % S; ny[k] = best / S;
    }

    // ---- 
    string out = to_string(T);
    for (int k = 0; k < 3; k++) {
        string mv = "S", act = "-";
        if (alive[k]) {
            int ddx = nx[k] - ux[k], ddy = ny[k] - uy[k];
            if (ddx == 1) mv = "R"; else if (ddx == -1) mv = "L"; else if (ddy == 1) mv = "D"; else if (ddy == -1) mv = "U";
            if (uflag[k] < 0) {
                bool onF = false;
                for (auto& f : g.flags) if ((f.status == "home" || f.status == "dropped") && f.return_at != T && f.x == nx[k] && f.y == ny[k]) onF = true;
                if (onF) act = "P";
                else if (!respNow[k]) {
                    int bestT = -1; float bs = -1e9;
                    for (auto& e : en) {
                        if (!canAtk(nx[k], ny[k], e.x, e.y)) continue;
                        float s = 0;
                        if (e.flag >= 0) s += 100;                                    // 
                        if (dB[ME][ID(e.x, e.y)] <= 2) s += 80;                      // 
                        for (auto& f : g.flags) if ((f.status == "home" || f.status == "dropped") && f.return_at != T && f.x == e.x && f.y == e.y) s += 90;   // 
                        for (int j = 0; j < 3; j++) if (alive[j] && uflag[j] >= 0 && md(e.x, e.y, nx[j], ny[j]) <= 3) s += 50;
                        int hits = 0;
                        for (int j = 0; j < 3; j++) if (alive[j] && uflag[j] < 0 && !respNow[j] && canAtk(nx[j], ny[j], e.x, e.y)) hits++;
                        if (e.hp <= hits * DMG) s += 55;                             // 
                        s += (MAXHP - e.hp) * 0.25f;
                        if (s > bs) { bs = s; bestT = e.id; }
                    }
                    if (bestT >= 0) act = to_string(bestT);
                }
            }
        }
        out += " " + mv + " " + act;
    }
    static const char* rname[9] = {"cy","fe","cl","su","es","hu","ca","id","xx"};
    string note = "#";
    for (int k = 0; k < 3; k++) { if (k) note += ";"; note += rname[role[k] < 9 ? role[k] : 8]; }
    out += " " + note;
    return out;
}

int main() {
    ios::sync_with_stdio(false);
    Game g;
    while (read_turn(g)) cout << decide(g) << endl;
}
