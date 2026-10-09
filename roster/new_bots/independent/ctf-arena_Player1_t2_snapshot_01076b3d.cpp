// 夺旗 bot：支持 2–15 方。
// 决策 = 静态预处理 + 每回合 BFS 距离场 + 角色分工（抢旗/护航/截杀）+ 局部单步评估。
// 移动 U D L R S；动作 - P X 或攻击目标 id。
#include <bits/stdc++.h>
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

static bool read_turn(Game& g) {
    string w;
    while (cin >> w) {
        if (w == "INIT") {
            cin >> g.teams >> g.me >> g.size >> g.turns >> g.hp0 >> g.damage
                >> g.respawn_t >> g.flag_return >> g.flag_cooldown;
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
            g.score.assign(g.teams, 0);
            for (auto& s : g.score) cin >> s;
        } else if (w == "UNITS") {
            int k; cin >> k; g.units.resize(k);
            for (auto& u : g.units)
                { string tag; cin >> tag >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawn_at; }
        } else if (w == "FLAGS") {
            int k; cin >> k; g.flags.resize(k);
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
        }
    }
    return false;
}

// ================= 世界 =================
namespace W {

int N = 0, S = 0, me = 0, TURN = 0, TOThp = 100, DMG = 34, RSP = 10;
vector<string> G;
vector<pair<int, int>> spots;
int spotIdx[64][64];
bool wall[64][64];

vector<int> ux, uy, uhp, uflag, uteam, ursp;
vector<int> bx, by;
vector<int> scol;

const int INF = 1e9;
int dist[64][64];
int queueBuf[64 * 64 * 4];

inline bool inb(int x, int y) { return x >= 0 && y >= 0 && x < S && y < S; }
inline bool wallAt(int x, int y) { return !inb(x, y) || wall[x][y]; }
inline bool live(int i) { return ux[i] >= 0; }
inline int eAt(int x, int y) {   // 该格上的角色，无则 -1（只用于我方避让判断）
    for (size_t i = 0; i < ux.size(); i++)
        if (ux[i] == x && uy[i] == y) return (int)i;
    return -1;
}
inline int mdist(int ax, int ay, int bx2, int by2) { return abs(ax - bx2) + abs(ay - by2); }
inline bool myBaseCell(int x, int y) {
    return abs(x - bx[me]) <= 1 && abs(y - by[me]) <= 1;
}

void bfsFrom(int sx, int sy) {
    for (int x = 0; x < S; x++)
        for (int y = 0; y < S; y++) dist[x][y] = INF;
    if (wallAt(sx, sy)) return;
    int head = 0, tail = 0;
    dist[sx][sy] = 0;
    queueBuf[tail++] = sy * S + sx;
    static const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
    while (head < tail) {
        int c = queueBuf[head++], x = c % S, y = c / S, d = dist[x][y];
        for (int k = 0; k < 4; k++) {
            int nx = x + dx[k], ny = y + dy[k];
            if (wallAt(nx, ny) || dist[nx][ny] <= d + 1) continue;
            dist[nx][ny] = d + 1;
            queueBuf[tail++] = ny * S + nx;
        }
    }
}

// 与 engine.can_attack 完全一致
bool canAttack(int ax, int ay, int tx, int ty) {
    int dx = tx - ax, dy = ty - ay;
    if (abs(dx) + abs(dy) > 2) return false;
    if (dx == 0 && dy == 0) return false;
    if (abs(dx) == 2 && dy == 0) return !wall[ax + dx / 2][ay];
    if (abs(dy) == 2 && dx == 0) return !wall[ax][ay + dy / 2];
    if (abs(dx) == 1 && abs(dy) == 1) return !wall[ax + dx][ay] || !wall[ax][ay + dy];
    return true;
}

}  // namespace W

// ================= 决策 =================
namespace D {

using namespace W;

// 目标类型
enum { GO_NONE = 0, GO_FLAG = 1, GO_BASE = 2, GO_ENEMY = 3 };

struct Plan {
    int type = GO_NONE;    // carrier 去基地 / collector 去旗 / hunter 去敌人
    int tx = -1, ty = -1;  // 目标格
    int enemy = -1;        // hunter 目标角色
    int flagId = -1;       // collector 目标旗（-1 表示去 target 附近等旗）
    int guard = -1;        // 需要保护的携旗友军
};

int dirDx[5] = {0, 1, -1, 0, 0};
int dirDy[5] = {0, 0, 0, 1, -1};
inline char moveCh(int d) {
    switch (d) { case 0: return 'S'; case 1: return 'R'; case 2: return 'L'; case 3: return 'D'; default: return 'U'; }
}

// 对手威胁：某格是否在敌人下回合可攻击范围内
bool dangerAt(int x, int y) {
    for (size_t i = 0; i < ux.size(); i++) {
        if (!live((int)i) || uteam[i] == me) continue;
        // 敌人下回合最多移动 1 格：只考虑它现在能否攻击到射程 3 内的点
        if (abs(ux[i] - x) + abs(uy[i] - y) <= 2 && canAttack(ux[i], uy[i], x, y)) return true;
        if (abs(ux[i] - x) + abs(uy[i] - y) == 3) {
            // 移动一格后可达：近似判断
            if (abs(ux[i] - x) <= 1 && abs(uy[i] - y) <= 2) return true;
            if (abs(ux[i] - x) <= 2 && abs(uy[i] - y) <= 1) return true;
        }
    }
    return false;
}

// 当前距离（BFS，忽略其他角色占位）
int distTo(int x, int y, int tx, int ty) {
    if (x == tx && y == ty) return 0;
    if (wallAt(tx, ty)) return INF;
    bfsFrom(tx, ty);
    return dist[x][y];
}

int stepCost(int d, bool onFlagCell, bool picking, bool onBase) {
    if (d < 0 || d >= INF) return INF;
    int c = d;
    if (picking) c -= 3;
    if (onBase) c -= 1;
    return c;
}

struct Act { int move = 0, act = -1, target = -1, score = -INF, dist = INF; bool pick = false; };

// 每回合刷新的旗子数组
vector<int> fxNow, fyNow, fstat, fholder;

// 对单个我方角色枚举 5 个移动 ×（原地/拾旗/攻击每个可达敌人），返回最优
Act bestAct(unsigned char self, const Plan& p, bool hasCarrierInTeam) {
    Act best;
    int x = ux[self], y = uy[self];
    int fid = uflag[self];
    bool isCarrier = fid >= 0;

    // 攻击候选：可达敌人（含预判下一格，简化：只看当前位置）
    vector<pair<int, int>> atk;   // (enemyId, value)
    for (size_t i = 0; i < ux.size(); i++) {
        if (!live((int)i) || uteam[i] == me) continue;
        int val = 1;
        if (uflag[(int)i] >= 0) val += 12;                       // 携旗者优先
        if (uhp[(int)i] <= DMG) val += 10;                       // 残血可杀
        else if (uhp[(int)i] <= 2 * DMG) val += 4;
        if (myBaseCell(ux[i], uy[i])) val += 6;                  // 打到自家基地里的
        atk.push_back({(int)i, val});
    }

    for (int d = 0; d < 5; d++) {
        int nx = x + dirDx[d], ny = y + dirDy[d];
        if (d == 0) { nx = x; ny = y; }
        if (wallAt(nx, ny)) continue;
        bool pick = false;
        int pickScore = 0;
        if (!isCarrier) {
            for (size_t f = 0; f < fxNow.size(); f++) {
                int st = fstat[f];
                if (st != 0 && st != 1) continue;
                if (fxNow[f] == nx && fyNow[f] == ny) {
                    pick = true;
                    // 离自家基地近的旗更值钱；已经快回位的旗次之
                    int cost = fyNow[f] * 0 + 0;
                    (void)cost;
                    pickScore = 5200;
                    if (st == 1) pickScore += 300;   // 地上的旗直接拿
                    break;
                }
            }
        }
        // 攻击评估
        int atkBest = -1, atkBestVal = 0;
        if (!isCarrier) {
            for (auto& a : atk) {
                if (!canAttack(nx, ny, ux[a.first], uy[a.first])) continue;
                int v = a.second;
                int hpLeft = uhp[a.first] - DMG;
                if (hpLeft <= 0) v += uflag[a.first] >= 0 ? 120 : 30;
                if (v > atkBestVal) { atkBestVal = v; atkBest = a.first; }
            }
        }
        // 位置评分
        int posScore = 0;
        if (p.type == GO_FLAG || p.type == GO_ENEMY) {
            bfsFrom(p.tx, p.ty);
            int dd = dist[nx][ny];
            if (dd < INF) posScore = -dd * 300;
            else posScore = -100000;
        } else if (p.type == GO_BASE) {
            bfsFrom(bx[me], by[me]);
            int dd = dist[nx][ny];
            if (myBaseCell(nx, ny)) dd = 0;
            posScore = (dd >= INF ? -100000 : -dd * 420);
        } else {
            posScore = 0;
        }
        // 被威胁惩罚
        int threat = 0;
        if (dangerAt(nx, ny)) {
            int mul = isCarrier ? 260 : (uhp[self] <= DMG ? 60 : 30);
            threat = -mul;
        }
        int stay = (nx == x && ny == y) ? -25 : 0;
        int age = (int)ux.size();
        (void)age;
        int total = posScore + pickScore + stay + threat + (pick ? 400 : 0);
        if (atkBest >= 0) {
            total += 600 + atkBestVal * 55;
            int actKind = atkBest;   // 攻击目标 id (>=0)
            if (total > best.score) { best = {d, actKind, atkBest, total, 0, pick}; }
        } else {
            if (pick) {
                if (total > best.score) { best = {d, 2, -1, total, 0, true}; }
            } else {
                if (total > best.score) { best = {d, 0, -1, total, 0, false}; }
            }
        }
    }
    (void)fid; (void)hasCarrierInTeam;
    return best;
}

}  // namespace D

int main() {
    Game g;
    while (read_turn(g)) {
        string out = to_string(g.turn);
        // ---- 初始化世界 ----
        W::N = g.teams; W::S = g.size; W::me = g.me; W::TURN = g.turn;
        W::TOThp = g.hp0; W::DMG = g.damage; W::RSP = g.respawn_t;
        W::G = g.map;
        W::spots = g.spots;
        for (int x = 0; x < W::S; x++)
            for (int y = 0; y < W::S; y++)
                W::wall[x][y] = (g.map[y][x] == '#');
        for (int x = 0; x < W::S; x++)
            for (int y = 0; y < W::S; y++)
                W::spotIdx[x][y] = -1;
        for (size_t i = 0; i < g.spots.size(); i++)
            W::spotIdx[g.spots[i].first][g.spots[i].second] = (int)i;
        W::bx.resize(g.teams); W::by.resize(g.teams);
        for (int t = 0; t < g.teams; t++) { W::bx[t] = g.bases[t].first; W::by[t] = g.bases[t].second; }
        W::ux.resize(g.units.size()); W::uy.resize(g.units.size());
        W::uhp.resize(g.units.size()); W::uflag.resize(g.units.size());
        W::uteam.resize(g.units.size()); W::ursp.resize(g.units.size());
        for (size_t i = 0; i < g.units.size(); i++) {
            W::ux[i] = g.units[i].x; W::uy[i] = g.units[i].y; W::uhp[i] = g.units[i].hp;
            W::uflag[i] = g.units[i].flag; W::uteam[i] = g.units[i].team; W::ursp[i] = g.units[i].respawn_at;
        }
        W::scol = g.score;
        D::fxNow.clear(); D::fyNow.clear(); D::fstat.clear(); D::fholder.clear();
        for (auto& f : g.flags) {
            int st = f.status == "home" ? 0 : f.status == "dropped" ? 1 : f.status == "carried" ? 2 : 3;
            D::fxNow.push_back(f.x); D::fyNow.push_back(f.y);
            D::fstat.push_back(st); D::fholder.push_back(f.holder);
        }

        // ---- 角色分工 ----
        using namespace W;
        // 我方三个角色
        int myIds[3] = {3 * me, 3 * me + 1, 3 * me + 2};
        int carrier = -1;
        for (int k = 0; k < 3; k++)
            if (myIds[k] < (int)ux.size() && live(myIds[k]) && uflag[myIds[k]] >= 0) carrier = myIds[k];

        int roles[3] = {0, 0, 0};   // 0 = 抢旗, 1 = 截杀, 2 = 护航
        // 有我方携旗者：1 人护航，其余抢旗；无：2 抢旗 1 截杀
        if (carrier >= 0) {
            int guardAssigned = 0;
            for (int k = 0; k < 3; k++) {
                if (myIds[k] == carrier) { roles[k] = 2; continue; }
                if (!guardAssigned) { roles[k] = 1; guardAssigned = 1; }
                else roles[k] = 0;
            }
        } else {
            for (int k = 0; k < 3; k++) roles[k] = (k == 2) ? 1 : 0;
        }
        // 危险度高（敌人多）时更偏截杀
        if (g.teams >= 6) { roles[0] = 0; roles[1] = 1; roles[2] = 0; }

        for (int k = 0; k < 3; k++) {
            int id = myIds[k];
            if (id >= (int)ux.size()) { out += " S -"; continue; }
            D::Plan p;
            if (live(id) && uflag[id] >= 0) {
                p.type = D::GO_BASE; p.tx = bx[me]; p.ty = by[me];
            } else if (roles[k] == 1) {
                // 截杀：优先敌方携旗者（尤其靠近自家基地的）
                int bestE = -1, bestVal = -1;
                for (size_t i = 0; i < ux.size(); i++) {
                    if (!live((int)i) || uteam[i] == me) continue;
                    int dx = abs(ux[i] - bx[me]), dy = abs(uy[i] - by[me]);
                    int distBase = dx + dy;
                    int v = 0;
                    if (uflag[(int)i] >= 0) v = 1000 - distBase * 3;
                    else v = 300 - mdist(ux[id], uy[id], ux[i], uy[i]) * 4 + (uhp[(int)i] <= 2 * DMG ? 60 : 0);
                    if (mdist(ux[id], uy[id], ux[i], uy[i]) > 22) v -= 200;
                    if (v > bestVal) { bestVal = v; bestE = (int)i; }
                }
                if (bestE >= 0) { p.type = D::GO_ENEMY; p.tx = ux[bestE]; p.ty = uy[bestE]; p.enemy = bestE; }
            } else {
                // 抢旗：对所有可拾的旗评分（到达时间 + 旗到自家基地距离）
                int bestF = -1, bestV = INF;
                for (size_t f = 0; f < D::fxNow.size(); f++) {
                    if (D::fstat[f] != 0 && D::fstat[f] != 1) continue;
                    int fx = D::fxNow[f], fy = D::fyNow[f];
                    int mine = D::distTo(ux[id], uy[id], fx, fy);
                    if (mine >= INF) continue;
                    int hd = mdist(fx, fy, bx[me], by[me]);
                    int v = mine * 10 + hd * 7;
                    if (D::fstat[f] == 1) v -= 25;                  // 地上的旗优先捡
                    if (hd <= 4) v -= 40;                           // 离基地近的更值
                    // 敌人更近就别去送
                    int eBest = INF;
                    for (size_t i = 0; i < ux.size(); i++) {
                        if (!live((int)i) || uteam[i] == me) continue;
                        eBest = min(eBest, mdist(ux[i], uy[i], fx, fy));
                    }
                    if (eBest + 1 < mine) v += 35 + (mine - eBest) * 6;
                    if (v < bestV) { bestV = v; bestF = (int)f; }
                }
                if (bestF >= 0) {
                    p.type = D::GO_FLAG; p.tx = D::fxNow[bestF]; p.ty = D::fyNow[bestF]; p.flagId = bestF;
                } else {
                    p.type = D::GO_FLAG; p.tx = g.spots[0].first; p.ty = g.spots[0].second;
                }
            }
            // 目标就是自己脚下时仍然枚举，bestAct 会选 P
            D::Act a = D::bestAct((unsigned char)id, p, carrier >= 0);
            out += " ";
            out += D::moveCh(a.move);
            out += " ";
            if (a.act == 2) out += "P";
            else if (a.act >= 0) out += to_string(a.act);
            else out += "-";
        }
        cout << out << endl;
    }
    return 0;
}
