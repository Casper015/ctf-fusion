// 夺旗 bot（2–15 方）
// 策略：全员挣分 —— 每回合给 3 个角色分配互不相同的目标，抢最划算的旗并送回家；
//       携旗者只走生存路线；基地附近出现敌方携旗者时最近角色转拦截。
// 实现：INIT 预计算 BFS 距离场；每回合精确威胁计数（含敌方移动 1 格后的射程）；
//       动作在"移动后位置"上判定（与引擎结算顺序一致）；三人同格仲裁避免自锁。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct Unit { int id = 0, team = 0, x = -1, y = -1, hp = 0, flag = -1, respawn_at = -1; };
struct Flag { int id = 0; string status; int x = -1, y = -1, holder = -1, return_at = -1; };
struct Event { string type; vector<int> args; };

struct Game {
    int teams = 0, me = 0, size = 0, turns = 0, hp0 = 100, damage = 34;
    int respawn_t = 10, flag_return = 15, flag_cooldown = 8;
    vector<string> map;
    vector<pair<int, int>> bases;
    vector<pair<int, int>> spots;
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;
    vector<Event> events;
};

// ---------------- 协议 ----------------
static bool read_turn(Game& g) {
    string w;
    while (cin >> w) {
        if (w == "INIT") {
            cin >> g.teams >> g.me >> g.size >> g.turns >> g.hp0 >> g.damage
                >> g.respawn_t >> g.flag_return >> g.flag_cooldown;
        } else if (w == "MAP") {
            g.map.assign(g.size, string());
            for (auto& row : g.map) { if (!(cin >> row)) return false; }
        } else if (w == "BASES") {
            int k; cin >> k; g.bases.assign(max(0, k), {0, 0});
            for (auto& b : g.bases) cin >> b.first >> b.second;
        } else if (w == "SPOTS") {
            int k; cin >> k; g.spots.assign(max(0, k), {0, 0});
            for (auto& s : g.spots) cin >> s.first >> s.second;
        } else if (w == "TURN") {
            cin >> g.turn;
        } else if (w == "SCORE") {
            g.score.assign(g.teams, 0);
            for (auto& s : g.score) cin >> s;
        } else if (w == "UNITS") {
            int k; cin >> k; g.units.assign(max(0, k), Unit());
            for (auto& u : g.units)
                { string tag; cin >> tag >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawn_at; }
        } else if (w == "FLAGS") {
            int k; cin >> k; g.flags.assign(max(0, k), Flag());
            for (auto& f : g.flags)
                { string tag; cin >> tag >> f.id >> f.status >> f.x >> f.y >> f.holder >> f.return_at; }
        } else if (w == "EVENTS") {
            int k; cin >> k; g.events.clear();
            string line; getline(cin, line);
            for (int i = 0; i < k; i++) {
                if (!getline(cin, line)) break;
                istringstream in(line);
                Event e; string tag; in >> tag >> e.type;
                for (int v; in >> v;) e.args.push_back(v);
                g.events.push_back(e);
            }
        } else if (w == "END") {
            return true;
        } else {
            string skip; getline(cin, skip);
        }
    }
    return false;
}

// ================= 世界 =================
namespace W {

const int INF = 1000000;
const int MAXS = 64;

int S = 0, me = 0, DMG = 34, TURN = 0;
bool wall[MAXS][MAXS];
vector<int> bx, by;
vector<pair<int, int>> spots;
vector<int> ux, uy, uhp, uflag, uteam, ursp;
vector<int> fx, fy, fst;   // fst: 0 home, 1 dropped, 2 carried, 3 cooldown

// 预计算距离场（自家基地 / 中心 / 各旗点）
const int MAXFLD = 64;
int fldCell[MAXFLD][MAXS * MAXS];
int nFld = 0;
int fldX[MAXFLD], fldY[MAXFLD];
int myBaseFld = -1, centerFld = -1;

int threatCnt[MAXS][MAXS];

inline bool inb(int x, int y) { return x >= 0 && y >= 0 && x < S && y < S; }
inline bool wallAt(int x, int y) { return !inb(x, y) || wall[x][y]; }
inline int mdist(int ax, int ay, int bx2, int by2) { return abs(ax - bx2) + abs(ay - by2); }
inline bool live(int i) { return ux[i] >= 0; }
inline bool myBaseCell(int x, int y) { return abs(x - bx[me]) <= 1 && abs(y - by[me]) <= 1; }

// 与 engine.can_attack 完全一致（按移动后的位置判定）
inline bool canAttack(int ax, int ay, int tx, int ty) {
    int dx = tx - ax, dy = ty - ay, adx = abs(dx), ady = abs(dy);
    if (adx + ady > 2 || (dx == 0 && dy == 0)) return false;
    if (adx == 2 && dy == 0) return !wall[ax + dx / 2][ay];
    if (ady == 2 && dx == 0) return !wall[ax][ay + dy / 2];
    if (adx == 1 && ady == 1) return !wall[ax + dx][ay] || !wall[ax][ay + dy];
    return true;
}

int findFld(int x, int y) {
    for (int i = 0; i < nFld; i++)
        if (fldX[i] == x && fldY[i] == y) return i;
    return -1;
}

void buildFld(int idx) {
    int* F = fldCell[idx];
    for (int i = 0; i < S * S; i++) F[i] = INF;
    if (wallAt(fldX[idx], fldY[idx])) return;
    static int q[MAXS * MAXS];
    int h = 0, t = 0;
    int s = fldY[idx] * S + fldX[idx];
    F[s] = 0; q[t++] = s;
    while (h < t) {
        int c = q[h++], x = c % S, y = c / S, d = F[c];
        const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (wallAt(nx, ny)) continue;
            int ni = ny * S + nx;
            if (F[ni] <= d + 1) continue;
            F[ni] = d + 1;
            q[t++] = ni;
        }
    }
}

// 注册一个源；超出上限或非法坐标返回 -1（绝不越界）
int addFld(int x, int y) {
    int e = findFld(x, y);
    if (e >= 0) return e;
    if (nFld >= MAXFLD || !inb(x, y)) return -1;
    fldX[nFld] = x; fldY[nFld] = y;
    buildFld(nFld);
    return nFld++;
}

inline int fdAt(int idx, int x, int y) {
    if (idx < 0 || idx >= nFld || !inb(x, y)) return INF;
    return fldCell[idx][y * S + x];
}

// 精确威胁计数：敌人移动 1 格（含原地）后能打到的所有格
void buildThreat() {
    for (int x = 0; x < S; x++)
        for (int y = 0; y < S; y++) threatCnt[x][y] = 0;
    const int dx[5] = {0, 1, -1, 0, 0}, dy[5] = {0, 0, 0, 1, -1};
    for (size_t i = 0; i < ux.size(); i++) {
        if (!live((int)i) || uteam[i] == me) continue;
        if (ursp[i] == TURN) continue;            // 本回合重生，免疫
        for (int d = 0; d < 5; d++) {
            int qx = ux[i] + dx[d], qy = uy[i] + dy[d];
            if (wallAt(qx, qy)) continue;
            for (int ax = -2; ax <= 2; ax++)
                for (int ay = -2; ay <= 2; ay++) {
                    if (abs(ax) + abs(ay) > 2 || (ax == 0 && ay == 0)) continue;
                    int tx = qx + ax, ty = qy + ay;
                    if (wallAt(tx, ty)) continue;
                    if (canAttack(qx, qy, tx, ty)) threatCnt[tx][ty]++;
                }
        }
    }
}

}  // namespace W

// ================= 决策 =================
namespace D {

using namespace W;

struct Act { int m = 0; int a = -1; };   // act: -1 无, 2 拾旗, >=0 攻击目标 id

const int dirDx[5] = {0, 1, -1, 0, 0};
const int dirDy[5] = {0, 0, 0, 1, -1};
const char moveChar[5] = {'S', 'R', 'L', 'D', 'U'};

int homeFld = -1;       // 自家基地距离场
int homeSpotFld = -1;   // 自家旗点距离场

bool spotOccupied(int x, int y) {
    for (size_t f = 0; f < fx.size(); f++)
        if (fst[f] != 3 && fx[f] == x && fy[f] == y) return true;
    return false;
}

int flagHere(int x, int y) {
    for (size_t f = 0; f < fx.size(); f++) {
        if (fst[f] != 0 && fst[f] != 1) continue;
        if (fx[f] == x && fy[f] == y) return (int)f;
    }
    return -1;
}

// 旗的价值：离我方越近越好、离自家基地越近越好、别人更快则贬值
double flagValue(int f, int self) {
    int dSelf = mdist(ux[self], uy[self], fx[f], fy[f]);
    int dBand = mdist(fx[f], fy[f], bx[me], by[me]);
    int eBest = INF;
    for (size_t i = 0; i < ux.size(); i++) {
        if (!live((int)i) || uteam[i] == me) continue;
        eBest = min(eBest, mdist(ux[i], uy[i], fx[f], fy[f]));
    }
    double v = dBand * 6.0 - dSelf * 10.0;
    if (fst[f] == 1) v += 20;
    if (eBest + 1 < dSelf) v -= 40 + (dSelf - eBest) * 8;
    return v;
}

// goalFld >= 0 用距离场；否则用曼哈顿贴近 (tx,ty)
Act chooseMove(int self, int goalFld, int tx, int ty, bool wantPick, bool carrying) {
    int x = ux[self], y = uy[self];
    Act best; best.m = 0; best.a = -1;
    int bs = -INF;
    for (int d = 0; d < 5; d++) {
        int nx = x + dirDx[d], ny = y + dirDy[d];
        if (wallAt(nx, ny)) continue;
        int pos = 0;
        if (goalFld >= 0) {
            int dd = fdAt(goalFld, nx, ny);
            if (dd >= INF) continue;
            pos -= dd * 100;
        } else {
            pos -= mdist(nx, ny, tx, ty) * 55;
        }
        int fh = carrying ? -1 : flagHere(nx, ny);
        if (fh >= 0) pos += 520 - mdist(nx, ny, bx[me], by[me]) * 12;
        int th = threatCnt[nx][ny];
        if (th > 0) {
            if (carrying) pos -= 300 * th;
            else if (uhp[self] <= DMG * th) pos -= 150 * th;
            else if (uhp[self] <= 2 * DMG) pos -= 40 * th;
            else pos -= 10 * th;
        }
        int tgt = -1, tval = -INF;
        if (!carrying) {
            for (size_t i = 0; i < ux.size(); i++) {
                if (!live((int)i) || uteam[i] == me) continue;
                if (!canAttack(nx, ny, ux[i], uy[i])) continue;
                int v = 50;
                if (uflag[(int)i] >= 0) v += 760;                     // 打携旗者
                if (uhp[(int)i] <= DMG) v += 340;                     // 能补刀
                else if (uhp[(int)i] <= 2 * DMG) v += 110;
                if (mdist(ux[i], uy[i], bx[me], by[me]) <= 2) v += 140;
                if (v > tval) { tval = v; tgt = (int)i; }
            }
        }
        if (tgt >= 0) {
            int total = pos + tval + 200;
            if (total > bs) { bs = total; best.m = d; best.a = tgt; }
        }
        if (pos > bs) { bs = pos; best.m = d; best.a = (fh >= 0 && !carrying) ? 2 : -1; }
    }
    return best;
}

}  // namespace D

int main() {
    Game g;
    bool inited = false;
    while (read_turn(g)) {
        try {
            if (!inited || g.turn <= 1 || (int)g.map.size() != g.size) {
                W::S = g.size < 1 ? 1 : g.size; W::me = g.me;
                for (int x = 0; x < W::S; x++)
                    for (int y = 0; y < W::S; y++)
                        W::wall[x][y] = (x < (int)g.map[y].size() && g.map[y][x] == '#');
                W::bx.assign(g.teams, 0); W::by.assign(g.teams, 0);
                for (int t = 0; t < g.teams && t < (int)g.bases.size(); t++) {
                    W::bx[t] = g.bases[t].first; W::by[t] = g.bases[t].second;
                }
                W::spots = g.spots;
                W::nFld = 0;
                W::myBaseFld = W::addFld(W::bx[W::me], W::by[W::me]);
                W::centerFld = W::addFld(W::S / 2, W::S / 2);
                for (auto& s : W::spots) W::addFld(s.first, s.second);
                D::homeFld = W::myBaseFld;
                D::homeSpotFld = -1;
                int bestD = 1 << 30;   // 自家旗点 = 离自家基地最近的旗点
                for (auto& s : W::spots) {
                    int d = W::mdist(s.first, s.second, W::bx[W::me], W::by[W::me]);
                    if (d < bestD) { bestD = d; D::homeSpotFld = W::findFld(s.first, s.second); }
                }
                inited = true;
            }
            W::TURN = g.turn; W::DMG = g.damage;
            size_t nu = g.units.size();
            W::ux.assign(nu, -1); W::uy.assign(nu, -1); W::uhp.assign(nu, 0);
            W::uflag.assign(nu, -1); W::uteam.assign(nu, 0); W::ursp.assign(nu, -1);
            for (size_t i = 0; i < nu; i++) {
                W::ux[i] = g.units[i].x; W::uy[i] = g.units[i].y; W::uhp[i] = g.units[i].hp;
                W::uflag[i] = g.units[i].flag; W::uteam[i] = g.units[i].team; W::ursp[i] = g.units[i].respawn_at;
            }
            W::fx.clear(); W::fy.clear(); W::fst.clear();
            for (auto& f : g.flags) {
                int st = f.status == "home" ? 0 : f.status == "dropped" ? 1 : f.status == "carried" ? 2 : 3;
                W::fx.push_back(f.x); W::fy.push_back(f.y); W::fst.push_back(st);
            }
            W::buildThreat();

            using namespace W;
            int ids[3] = {3 * me, 3 * me + 1, 3 * me + 2};
            bool alive[3] = {false, false, false};
            int carrier = -1;
            for (int k = 0; k < 3; k++) {
                alive[k] = ids[k] < (int)nu && live(ids[k]);
                if (alive[k] && uflag[ids[k]] >= 0) carrier = ids[k];
            }
            int threat = -1, threatD = INF;       // 逼近自家基地的敌方携旗者
            for (size_t i = 0; i < ux.size(); i++) {
                if (!live((int)i) || uteam[i] == me || uflag[i] < 0) continue;
                int d = mdist(ux[i], uy[i], bx[me], by[me]);
                if (d < threatD) { threatD = d; threat = (int)i; }
            }

            // ---- 目标分配（-3 回家 / -2 拦截 / -4 蹲点 / >=0 抢第 f 面旗）----
            int need[3] = {-4, -4, -4};
            bool assigned[3] = {false, false, false};
            for (int k = 0; k < 3; k++)
                if (alive[k] && uflag[ids[k]] >= 0) { need[k] = -3; assigned[k] = true; }
            // 捕手：主动猎杀敌方携旗者（只要它在向自己基地移动且离我方半场更近）
            if (threat >= 0) {
                int hisBase = W::mdist(ux[threat], uy[threat], bx[1 - me], by[1 - me]);
                if (hisBase <= 12) {                       // 它离自家基地 12 步内 = 正在回家途中
                    int hk = -1, hd = 1 << 30;
                    for (int k = 0; k < 3; k++) {
                        if (!alive[k] || assigned[k] || ids[k] == carrier) continue;
                        int d = W::mdist(ux[ids[k]], uy[ids[k]], ux[threat], uy[threat]);
                        if (d < hd) { hd = d; hk = k; }
                    }
                    if (hk >= 0 && hd <= 10) { need[hk] = -2; assigned[hk] = true; }
                }
            }
            if (threat >= 0 && threatD <= 6) {          // 拦截
                int bestK = -1, bestD = INF;
                for (int k = 0; k < 3; k++) {
                    if (!alive[k] || assigned[k]) continue;
                    int d = mdist(ux[ids[k]], uy[ids[k]], ux[threat], uy[threat]);
                    if (d < bestD) { bestD = d; bestK = k; }
                }
                if (bestK >= 0 && bestD <= 9) { need[bestK] = -2; assigned[bestK] = true; }
            }
            bool used[64] = {false};                    // 每面旗最多一个角色
            for (int round = 0; round < 3; round++) {
                int bestK = -1, bestF = -1;
                double bestV = -1e18;
                for (int k = 0; k < 3; k++) {
                    if (!alive[k] || assigned[k]) continue;
                    for (size_t f = 0; f < fx.size() && f < 64; f++) {
                        if (used[f] || (fst[f] != 0 && fst[f] != 1)) continue;
                        double v = D::flagValue((int)f, ids[k]);
                        if (v > bestV) { bestV = v; bestK = k; bestF = (int)f; }
                    }
                }
                if (bestK < 0 || bestF < 0) break;
                need[bestK] = bestF; assigned[bestK] = true; used[bestF] = true;
            }
            for (int k = 0; k < 3; k++)
                if (alive[k] && !assigned[k]) need[k] = -4;

            // ---- 生成动作 ----
            D::Act act[3];
            for (int k = 0; k < 3; k++) {
                act[k].m = 0; act[k].a = -1;
                if (!alive[k]) continue;
                int id = ids[k];
                bool carrying = uflag[id] >= 0;
                if (need[k] == -3)
                    act[k] = D::chooseMove(id, D::homeFld, bx[me], by[me], false, carrying);
                else if (need[k] == -2)
                    act[k] = D::chooseMove(id, -1, ux[threat], uy[threat], false, carrying);
                else if (need[k] >= 0) {
                    int f = need[k];
                    int gf = (f < (int)fx.size()) ? W::findFld(fx[f], fy[f]) : -1;
                    act[k] = D::chooseMove(id, gf, fx[f], fy[f], true, carrying);
                } else
                    act[k] = D::chooseMove(id, D::homeSpotFld, bx[me], by[me], true, false);
            }
            // ---- 同格仲裁：后者换次优格，避免双方都不动 ----
            for (int a = 0; a < 3; a++) {
                if (!alive[a] || act[a].m == 0) continue;
                int ax = ux[ids[a]] + D::dirDx[act[a].m], ay = uy[ids[a]] + D::dirDy[act[a].m];
                for (int b = a + 1; b < 3; b++) {
                    if (!alive[b]) continue;
                    int bx2 = ux[ids[b]] + D::dirDx[act[b].m], by2 = uy[ids[b]] + D::dirDy[act[b].m];
                    if (bx2 != ax || by2 != ay) continue;
                    int bestM = 0, bestS = -INF;
                    for (int d = 0; d < 5; d++) {
                        int cx = ux[ids[b]] + D::dirDx[d], cy = uy[ids[b]] + D::dirDy[d];
                        if (wallAt(cx, cy) || (cx == ax && cy == ay)) continue;
                        int s = -mdist(cx, cy, bx[me], by[me]);
                        if (s > bestS) { bestS = s; bestM = d; }
                    }
                    act[b].m = bestM; act[b].a = -1;
                }
            }

            string out = to_string(g.turn);
            for (int k = 0; k < 3; k++) {
                out += ' ';
                out += D::moveChar[act[k].m];
                out += ' ';
                if (act[k].a == 2) out += 'P';
                else if (act[k].a >= 0) out += to_string(act[k].a);
                else out += '-';
            }
            cout << out << endl;
        } catch (...) {
            if (g.turn > 0 && g.me >= 0) cout << to_string(g.turn) + " S - S - S -" << endl;
        }
    }
    return 0;
}
