// 多阵营夺旗 bot：确定性三层决策器
//   1) 局面解析 + 与 engine.step 一致的推演（重生、移动阻挡、攻击可达）
//   2) 宏角色分配（主/次采集、护送、拦截、防守驻点）
//   3) 每角色候选动作打分 + 两轮坐标下降 + 我方同格/被挡消解
// 所有阈值集中在 Tunables，便于调参。单回合耗时远低于 50ms。
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
using namespace std;

struct Unit { int id = 0, team = 0, x = -1, y = -1, hp = 0, flag = -1, respawn_at = -1; };
struct Flag { int id = 0; string status; int x = -1, y = -1, holder = -1, return_at = -1; };
struct Event { string type; vector<int> args; };

struct Game {
    int teams = 0, me = 0, size = 0, turns = 400, hp = 100, damage = 34, respawn = 10, flag_return = 15, flag_cooldown = 8;
    vector<string> map;
    vector<pair<int, int>> bases, spots;
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;
    vector<Event> events;
};

// ---------------------------------------------------------------- 可调常量表
struct Tunables {
    // 目标价值（以“一步进度 = wProg”为基本单位）
    double wCap = 1400;        // 本回合完成交旗
    double wProg = 10;         // 距任务目标每剩一步的代价
    double wPick = 70;         // 拾旗动作
    double wHit = 14;          // 每点伤害
    double wKill = 170;        // 击杀奖励
    double wKillCarry = 430;   // 击杀旗手（阻止一次交旗）
    double wHunt = 210;        // 拦截任务的基准价值
    // 风险
    double wHitOnMe = 4;       // 每个能打到我的敌人（轻微，避免无谓绕路）
    double wHitOnMeCarry = 6;  // 携旗者略高
    double wWounded = 60;      // 下回合再挨一次就死
    double wDeath = 240;       // 预计本回合阵亡
    double wDeathCarry = 380;  // 携旗阵亡（掉旗）
    double wBlocked = 170;     // 移动被我方/敌方顶住，白走一步
    double wEnemyCamp = 70;    // 原地堵住敌方旗手去路
    // 采集/竞争
    double contestW = 7;       // 敌方比我早到的每一步惩罚
    int    huntRadius = 8;     // 可拦截距离上限
    double fetchMult = 2.4;    // 采集任务最大成本 = fetchMult * 最短往返
    double campW = 1.0;        // 驻点权重
    double centerBonus = 230;  // 中心旗点驻点价值
    double ownBonus = 150;     // 自家旗点驻点价值
    double campUsedPen = 150;  // 同一驻点已有队友
    double wEscort = 120;      // 护送我方旗手的基准价值
    int    escortMin = 4;      // 旗手离基地还剩这么多步以内就不必护送
    int    escortMax = 22;     // 太远也不必护送
    int    lateBy = 2;         // 敌方比我早这么多回合到旗点就放弃争夺
    int    pinTeams = 0;       // 阵营数 <= 此值时启用「钉旗点」战术
    int    pinRace = 1;        // 钉旗点允许比敌方晚到的步数
    int    huntMaxHome = 14;   // 拦截任务离我方基地的最大距离（格）
    int    endSlack = 3;       // 收尾回合余量
};
static Tunables K;
static int HOMELEG = 8;        // 我方基地到自家旗点的距离

static const int DMG = 34;
static const int DIV[5] = {0, 0, 0, -1, 1};    // 0 S 1 U 2 D 3 L 4 R
static const int DJV[5] = {0, -1, 1, 0, 0};
static const char DCH[5] = {'S', 'U', 'D', 'L', 'R'};

static int TEAMS = 0, ME = 0, SZ = 0, TURNS = 400, HPMAX = 100, RESPAWN = 10, FLAGRET = 15, FLAGCD = 8;
static vector<string> MAPROW;
static vector<pair<int, int>> BC, SP;
static vector<char> WALLV;
static vector<array<int, 5>> NBV;
static vector<vector<int>> BCEL, SPAWNORD;
static unordered_map<long long, vector<int>> DC;
static bool INITED = false;
static bool DBGON = false;

// ---------------------------------------------------------------- 基础工具
inline int CIDX(int x, int y) { return y * SZ + x; }
inline bool inside(int x, int y) { return x >= 0 && y >= 0 && x < SZ && y < SZ; }

// 单源 BFS 距离场（按目标格缓存，地图静态，缓存永久有效）
static const vector<int>& DFc(int cell) {
    auto it = DC.find(cell);
    if (it != DC.end()) return it->second;
    vector<int> d(SZ * SZ, -1);
    if (cell >= 0 && cell < SZ * SZ && !WALLV[cell]) {
        deque<int> q; q.push_back(cell); d[cell] = 0;
        while (!q.empty()) {
            int c = q.front(); q.pop_front();
            int nd = d[c] + 1;
            for (int k = 1; k <= 4; k++) { int t = NBV[c][k]; if (t != c && d[t] < 0) { d[t] = nd; q.push_back(t); } }
        }
    }
    return DC.emplace((long long)cell, std::move(d)).first->second;
}

// 多源 BFS：某阵营基地 3x3
static const vector<int>& DFb(int team) {
    long long key = 1000000LL + team;
    auto it = DC.find(key);
    if (it != DC.end()) return it->second;
    vector<int> d(SZ * SZ, -1);
    deque<int> q;
    for (int c : BCEL[team]) if (!WALLV[c]) { d[c] = 0; q.push_back(c); }
    while (!q.empty()) {
        int c = q.front(); q.pop_front();
        int nd = d[c] + 1;
        for (int k = 1; k <= 4; k++) { int t = NBV[c][k]; if (t != c && d[t] < 0) { d[t] = nd; q.push_back(t); } }
    }
    return DC.emplace(key, std::move(d)).first->second;
}

// 沿距离场梯度走一步
static int bestDir(int from, const vector<int>& d) {
    if (from < 0 || d[from] < 0) return 0;
    int bd = d[from], bi = 0;
    for (int k = 1; k <= 4; k++) { int t = NBV[from][k]; if (t == from) continue; if (d[t] >= 0 && d[t] < bd) { bd = d[t]; bi = k; } }
    return bi;
}

// 攻击可达性：与 engine.can_attack 完全一致
static bool canAtk(int a, int b) {
    if (a < 0 || b < 0) return false;
    int ax = a % SZ, ay = a / SZ, bx = b % SZ, by = b / SZ;
    int dx = bx - ax, dy = by - ay;
    if (abs(dx) + abs(dy) > 2) return false;
    if (abs(dx) == 2 && dy == 0) return !WALLV[ay * SZ + ax + dx / 2];
    if (abs(dy) == 2 && dx == 0) return !WALLV[(ay + dy / 2) * SZ + ax];
    if (abs(dx) == 1 && abs(dy) == 1) return !WALLV[ay * SZ + ax + dx] || !WALLV[(ay + dy) * SZ + ax];
    return true;
}

// ---------------------------------------------------------------- 每回合状态
static int TURN = 0;
static vector<Unit> U;
static vector<Flag> F;
static vector<int> PSTART;      // 重生阶段结束后的位置（含本回合重生者）
static vector<char> IMMUNE;     // 本回合重生 => 免疫，不能攻击也不会被攻击
static vector<int> EWANT;       // 敌方本回合想要的落点（预测）
static vector<int> MYID, MYSTART;
static vector<char> MYIMM;
static vector<int> DFH;         // 到己方基地的距离场

struct Mission { int kind = 0; int cell = -1; int flag = -1; int enemy = -1; double cost = 0; };
// kind: 0 驻点 / 1 取旗 / 2 拦截 / 3 回基地交旗
static Mission MIS[3];
static const vector<int>* MISDF[3];

static bool pickable(const Flag& f) {
    if (f.x < 0) return false;
    if (f.status == "home") return true;
    if (f.status == "dropped") return f.return_at < 0 || TURN < f.return_at;
    return false;   // carried / cooldown
}

// 与 engine.step 第 2 步同语义的落点结算（starts/wants 含全部存活角色）
static vector<int> resolveAll(const vector<int>& starts, const vector<int>& wants, const vector<int>& teams) {
    int nu = (int)starts.size();
    vector<int> blk(nu, 0), end = starts;
    static vector<int> cnt, own;
    cnt.assign(SZ * SZ, 0);
    own.assign(SZ * SZ, -1);
    for (int i = 0; i < nu; i++) if (starts[i] >= 0) { cnt[wants[i]]++; own[starts[i]] = i; }
    for (int i = 0; i < nu; i++) if (starts[i] >= 0 && cnt[wants[i]] > 1) blk[i] = 1;
    for (int i = 0; i < nu; i++) {
        if (starts[i] < 0) continue;
        int w = wants[i], j = own[w];
        if (j >= 0 && j != i && wants[j] == starts[i] && teams[i] != teams[j]) { blk[i] = 1; blk[j] = 1; }
    }
    for (;;) {
        int added = 0;
        for (int i = 0; i < nu; i++) {
            if (starts[i] < 0 || blk[i] || wants[i] == starts[i]) continue;
            int j = own[wants[i]];
            if (j >= 0 && (blk[j] || wants[j] == starts[j])) { blk[i] = 1; added = 1; }
        }
        if (!added) break;
    }
    for (int i = 0; i < nu; i++) if (starts[i] >= 0) end[i] = blk[i] ? starts[i] : wants[i];
    return end;
}

static double attackValue(int meCell, int ti, const vector<int>& ends) {
    int ec = ends[ti];
    if (ec < 0 || !canAtk(meCell, ec)) return -1e9;
    int h = U[ti].hp;
    double v = K.wHit * min(DMG, h);
    if (h <= DMG) v += K.wKill + (U[ti].flag >= 0 ? K.wKillCarry : 0);
    if (U[ti].flag >= 0) v += 40;
    return v;
}

// 选动作：携旗只能待机；否则优先拾旗，其次打最划算的敌人
static void pickAction(int k, int endCell, const vector<int>& ends, int& act, int& tgt, double& bonus) {
    act = 0; tgt = -1; bonus = 0;
    int id = MYID[k];
    if (U[id].flag >= 0) return;
    if (MIS[k].kind != 4)   // 钉旗单位：不拾旗（也不让别人拾）
        for (const Flag& f : F) if (pickable(f) && CIDX(f.x, f.y) == endCell) { act = 1; bonus = K.wPick; return; }
    if (MYIMM[k]) return;   // 本回合重生：不能攻击
    int best = -1; double bv = 0;
    for (int i = 0; i < (int)U.size(); i++) {
        if (U[i].team == ME || PSTART[i] < 0 || IMMUNE[i]) continue;
        double v = attackValue(endCell, i, ends);
        if (v > bv) { bv = v; best = i; }
    }
    if (best >= 0) { act = 3; tgt = best; bonus = bv; }
}

static double riskPenalty(int k, int endCell, const vector<int>& ends) {
    if (MYIMM[k]) return 0;
    int id = MYID[k];
    int hits = 0;
    for (int i = 0; i < (int)U.size(); i++) {
        if (U[i].team == ME || PSTART[i] < 0 || IMMUNE[i]) continue;
        if (U[i].flag >= 0) continue;           // 携旗者不能攻击
        int ec = ends[i];
        if (ec < 0) continue;
        if (canAtk(ec, endCell)) hits++;
    }
    if (hits == 0) return 0;
    int dmg = DMG * hits;
    bool carry = U[id].flag >= 0;
    double s = (carry ? K.wHitOnMeCarry : K.wHitOnMe) * hits;
    if (dmg >= U[id].hp) s += carry ? K.wDeathCarry : K.wDeath;
    else if (dmg + DMG >= U[id].hp) s += K.wWounded;
    return s;
}

static double unitScore(int k, int endCell, double bonus, bool blocked, const vector<int>& ends) {
    if (endCell < 0) return 0;
    int id = MYID[k];
    const Unit& u = U[id];
    double s = 0;
    if (u.flag >= 0) {
        if (DFH[endCell] == 0) s += K.wCap;     // 本回合交旗
        else s -= K.wProg * DFH[endCell];
    } else if (MISDF[k]) {
        s -= K.wProg * (*MISDF[k])[endCell];
    }
    s += bonus;
    s -= riskPenalty(k, endCell, ends);
    if (blocked) s -= K.wBlocked;
    if (endCell == MYSTART[k]) {                 // 原地堵住敌方旗手
        for (int i = 0; i < (int)U.size(); i++)
            if (U[i].team != ME && U[i].flag >= 0 && ends[i] == endCell) { s += K.wEnemyCamp; break; }
    }
    return s;
}

// 给定我方 3 个方向，推演全局落点并给我方打分
static double jointScore(const array<int, 3>& dir, vector<int>& myEnd, vector<int>& myAct, vector<int>& myTgt) {
    int nu = (int)U.size();
    vector<int> starts(nu, -1), wants(nu, -1), teams(nu);
    for (int i = 0; i < nu; i++) {
        teams[i] = U[i].team;
        if (PSTART[i] >= 0) { starts[i] = PSTART[i]; wants[i] = EWANT[i] >= 0 ? EWANT[i] : PSTART[i]; }
    }
    for (int k = 0; k < 3; k++) {
        int id = MYID[k];
        if (MYSTART[k] >= 0) { starts[id] = MYSTART[k]; wants[id] = NBV[MYSTART[k]][dir[k]]; }
    }
    vector<int> ends = resolveAll(starts, wants, teams);
    double tot = 0;
    for (int k = 0; k < 3; k++) {
        int id = MYID[k];
        if (MYSTART[k] < 0) { myEnd[k] = -1; myAct[k] = 0; myTgt[k] = -1; continue; }
        int e = ends[id];
        myEnd[k] = e;
        int act, tgt; double bonus;
        pickAction(k, e, ends, act, tgt, bonus);
        myAct[k] = act; myTgt[k] = tgt;
        bool blocked = (e == MYSTART[k] && wants[id] != MYSTART[k]);
        tot += unitScore(k, e, bonus, blocked, ends);
    }
    return tot;
}

// ---------------------------------------------------------------- 输入解析
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
                if (!getline(cin, line)) break;
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

// ---------------------------------------------------------------- 静态初始化
static void buildInit(const Game& g) {
    TEAMS = g.teams; ME = g.me; SZ = g.size; TURNS = g.turns; HPMAX = g.hp;
    RESPAWN = g.respawn; FLAGRET = g.flag_return; FLAGCD = g.flag_cooldown;
    MAPROW = g.map; BC = g.bases; SP = g.spots;
    WALLV.assign(SZ * SZ, 0);
    for (int y = 0; y < SZ; y++) for (int x = 0; x < SZ; x++)
        if (y < (int)MAPROW.size() && x < (int)MAPROW[y].size() && MAPROW[y][x] == '#') WALLV[CIDX(x, y)] = 1;
    NBV.assign(SZ * SZ, {});
    for (int y = 0; y < SZ; y++) for (int x = 0; x < SZ; x++) {
        int c = CIDX(x, y);
        NBV[c][0] = c;
        for (int k = 1; k <= 4; k++) {
            int nx = x + DIV[k], ny = y + DJV[k];
            NBV[c][k] = (inside(nx, ny) && !WALLV[CIDX(nx, ny)]) ? CIDX(nx, ny) : c;
        }
    }
    // 基地 3x3（与 engine 一致）与重生顺序
    int cx = SZ / 2, cy = SZ / 2;
    BCEL.assign(TEAMS, {});
    SPAWNORD.assign(TEAMS, {});
    for (int t = 0; t < TEAMS; t++) {
        int bx = BC[t].first, by = BC[t].second;
        vector<pair<long long, int>> tmp;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            int x = bx + dx, y = by + dy;
            if (!inside(x, y)) continue;
            int c = CIDX(x, y);
            BCEL[t].push_back(c);
            long long key = (long long)(abs(x - cx) + abs(y - cy)) * 1000 + abs(x - bx) + 2 * abs(y - by);
            tmp.push_back({key, c});
        }
        sort(tmp.begin(), tmp.end());
        for (auto& p : tmp) SPAWNORD[t].push_back(p.second);
    }
    INITED = true;
    HOMELEG = 8;
    if (ME < (int)BC.size() && ME < (int)SP.size()) {
        int dc = CIDX(BC[ME].first, BC[ME].second);
        int dsp = CIDX(SP[ME].first, SP[ME].second);
        HOMELEG = max(2, DFc(dsp)[dc]);
    }
}

// ---------------------------------------------------------------- 决策
static string decide(const Game& g) {
    if (!INITED) buildInit(g);
    TURN = g.turn;
    U = g.units; F = g.flags;
    int nu = (int)U.size();
    int nf = (int)F.size();
    if (nu <= 0) return to_string(TURN) + " S - S - S -";

    // 1. 重生阶段推演（与 engine 第 1 步一致）
    PSTART.assign(nu, -1);
    IMMUNE.assign(nu, 0);
    {
        static vector<char> occ;
        occ.assign(SZ * SZ, 0);
        for (auto& u : U) if (u.x >= 0) occ[CIDX(u.x, u.y)] = 1;
        for (int i = 0; i < nu; i++) {
            if (U[i].x >= 0) { PSTART[i] = CIDX(U[i].x, U[i].y); continue; }
            if (U[i].respawn_at >= 0 && TURN >= U[i].respawn_at && U[i].team < TEAMS) {
                for (int c : SPAWNORD[U[i].team]) if (!occ[c]) { PSTART[i] = c; occ[c] = 1; IMMUNE[i] = 1; break; }
            }
        }
    }

    DFH = DFb(ME);

    // 2. 敌方意图预测：携旗者回基地，否则去自己最划算的旗点
    EWANT.assign(nu, -1);
    for (int i = 0; i < nu; i++) {
        if (U[i].team == ME || PSTART[i] < 0) continue;
        const vector<int>* df = nullptr;
        if (U[i].flag >= 0) {
            df = &DFb(U[i].team);
        } else {
            int bestFlag = -1; int bestCost = INT_MAX;
            for (int fi = 0; fi < nf; fi++) {
                if (!pickable(F[fi])) continue;
                int fc = CIDX(F[fi].x, F[fi].y);
                const vector<int>& d1 = DFc(fc);
                if (d1[PSTART[i]] < 0) continue;
                int cost = d1[PSTART[i]] + DFb(U[i].team)[fc];
                if (cost < bestCost) { bestCost = cost; bestFlag = fi; }
            }
            if (bestFlag >= 0) df = &DFc(CIDX(F[bestFlag].x, F[bestFlag].y));
            else if (U[i].team < (int)SP.size()) df = &DFc(CIDX(SP[U[i].team].first, SP[U[i].team].second));
            else df = &DFb(U[i].team);
        }
        int d = bestDir(PSTART[i], *df);
        EWANT[i] = NBV[PSTART[i]][d];
    }

    // 3. 我方角色
    MYID.resize(3); MYSTART.assign(3, -1); MYIMM.assign(3, 0);
    for (int k = 0; k < 3; k++) {
        MYID[k] = 3 * ME + k;
        if (MYID[k] >= 0 && MYID[k] < nu) { MYSTART[k] = PSTART[MYID[k]]; MYIMM[k] = IMMUNE[MYID[k]]; }
        else MYID[k] = -1;
    }

    vector<char> assigned(3, 0), flagUsed(nf, 0);
    for (int k = 0; k < 3; k++) MIS[k] = Mission();
    for (int k = 0; k < 3; k++) MISDF[k] = nullptr;

    // 携旗者：回基地
    for (int k = 0; k < 3; k++) {
        if (MYID[k] < 0 || MYSTART[k] < 0) continue;
        if (U[MYID[k]].flag >= 0) { MIS[k] = {3, -1, -1, -1, (double)DFH[MYSTART[k]]}; MISDF[k] = &DFH; assigned[k] = 1; }
    }

    // 驻点表（中心旗点最高频，自家旗点最短往返）
    vector<int> camps;
    vector<double> campBonus;
    int centerCamp = -1, ownCamp = -1;
    if ((int)SP.size() > TEAMS) { centerCamp = CIDX(SP[TEAMS].first, SP[TEAMS].second); camps.push_back(centerCamp); campBonus.push_back(K.centerBonus); }
    if (ME < (int)SP.size()) { ownCamp = CIDX(SP[ME].first, SP[ME].second); camps.push_back(ownCamp); campBonus.push_back(K.ownBonus); }
    if (ME < (int)BC.size()) { camps.push_back(CIDX(BC[ME].first, BC[ME].second)); campBonus.push_back(0); }
    vector<char> campUsed(camps.size(), 0);

    // 钉旗点（小场专用）：占住敌方最容易得分的旗点；旗子刷新到脚下就不拾，
    // 由 engine.py「同格不可进入 + 只有 act=P 才拾旗」两条规则，永久掐断这面旗的流动性。
    if (TEAMS <= K.pinTeams && nf > 0) {
        int bestK = -1, bestCell = -1; double bestV = -1e18;
        for (int si = 0; si < (int)SP.size(); si++) {
            if (si == ME) continue;                       // 自家旗点留给采集
            int cell = CIDX(SP[si].first, SP[si].second);
            const vector<int>& dc0 = DFc(cell);
            bool hasFlag = false;
            for (const Flag& f : F) if (pickable(f) && CIDX(f.x, f.y) == cell) hasFlag = true;
            if (hasFlag) continue;
            int eMin = INT_MAX;
            for (int i = 0; i < nu; i++)
                if (U[i].team != ME && PSTART[i] >= 0) { int d = dc0[PSTART[i]]; if (d >= 0) eMin = min(eMin, d); }
            if (eMin >= INT_MAX) continue;
            for (int k = 0; k < 3; k++) {
                if (assigned[k] || MYSTART[k] < 0) continue;
                int ourD = dc0[MYSTART[k]];
                if (ourD < 0 || ourD > eMin + K.pinRace) continue;
                double v = -K.wProg * ourD - 0.6 * K.wProg * eMin;
                if (v > bestV) { bestV = v; bestK = k; bestCell = cell; }
            }
        }
        if (bestK >= 0) {
            assigned[bestK] = 1;
            MIS[bestK] = {4, bestCell, -1, -1, bestV};
            MISDF[bestK] = &DFc(bestCell);
        }
    }

    // 采集任务贪心分配（有全局性价比阈值，不做无望的长途奔袭）
    struct FC { double cost; int k; int f; };
    vector<FC> fcs;
    double fetchMax = K.fetchMult * 2.0 * HOMELEG;
    for (int k = 0; k < 3; k++) {
        if (assigned[k] || MYSTART[k] < 0) continue;
        for (int fi = 0; fi < nf; fi++) {
            if (!pickable(F[fi])) continue;
            int fc = CIDX(F[fi].x, F[fi].y);
            const vector<int>& dfc = DFc(fc);
            int d1 = dfc[MYSTART[k]];
            if (d1 < 0) continue;
            int d2 = DFH[fc];
            if (d1 + d2 > TURNS - TURN + 1 + K.endSlack) continue;   // 来不及
            if (d1 + d2 > fetchMax) continue;                        // 太远：不如驻点等下一次刷新
            int ed = INT_MAX;
            for (int i = 0; i < nu; i++)
                if (U[i].team != ME && PSTART[i] >= 0) ed = min(ed, dfc[PSTART[i]] < 0 ? INT_MAX : dfc[PSTART[i]]);
            if (ed < INT_MAX && ed + K.lateBy <= d1) continue;        // 明显抢不到
            double pen = (ed < d1) ? K.contestW * (d1 - ed) : 0;
            fcs.push_back({(double)(d1 + d2) + pen, k, fi});
        }
    }
    sort(fcs.begin(), fcs.end(), [](const FC& a, const FC& b) { return a.cost < b.cost; });
    for (auto& c : fcs) {
        if (assigned[c.k] || flagUsed[c.f]) continue;
        assigned[c.k] = 1; flagUsed[c.f] = 1;
        MIS[c.k] = {1, CIDX(F[c.f].x, F[c.f].y), c.f, -1, c.cost};
        MISDF[c.k] = &DFc(MIS[c.k].cell);
    }

    // 护送：我方有旗手在途时，派一名离得近的队友贴身（旗手阵亡时可立刻捡回掉落的旗）
    {
        int carrierK = -1;
        for (int k = 0; k < 3; k++) if (MIS[k].kind == 3) carrierK = k;
        if (carrierK >= 0) {
            int cc = MYSTART[carrierK];
            int remain = DFH[cc];
            if (remain >= K.escortMin && remain <= K.escortMax) {
                int bestK = -1; double bestV = -1e18;
                for (int k = 0; k < 3; k++) {
                    if (assigned[k] || MYSTART[k] < 0) continue;
                    const vector<int>& dc = DFc(cc);
                    int du = dc[MYSTART[k]];
                    if (du < 0) continue;
                    double v = K.wEscort - K.wProg * du;
                    if (v > bestV) { bestV = v; bestK = k; }
                }
                if (bestK >= 0 && bestV > 0) {
                    assigned[bestK] = 1;
                    MIS[bestK] = {2, cc, -1, -1, bestV};
                    MISDF[bestK] = &DFc(cc);
                }
            }
        }
    }

    // 剩余角色：拦截近处旗手 或 驻点
    for (int k = 0; k < 3; k++) {
        if (assigned[k] || MYSTART[k] < 0) continue;
        double bestVal = -1e18; Mission bm; const vector<int>* bdf = nullptr;
        // 拦截：只在离我方基地不远的范围内，且来得及在旗手回家前追上
        for (int i = 0; i < nu; i++) {
            if (U[i].team == ME || PSTART[i] < 0 || U[i].flag < 0) continue;
            int ec = EWANT[i] >= 0 ? EWANT[i] : PSTART[i];
            if (DFH[ec] > K.huntMaxHome) continue;
            const vector<int>& de = DFc(ec);
            int du = de[MYSTART[k]];
            if (du < 0 || du > K.huntRadius) continue;
            int dh = DFb(U[i].team)[ec];
            if (du > dh + 2) continue;
            double v = K.wHunt - K.wProg * du + (U[i].hp <= DMG ? 150 : 0) - 0.5 * K.wProg * dh;
            if (v > bestVal) { bestVal = v; bm = {2, ec, -1, i, v}; bdf = &de; }
        }
        // 驻点
        for (int ci = 0; ci < (int)camps.size(); ci++) {
            const vector<int>& dc = DFc(camps[ci]);
            int du = dc[MYSTART[k]];
            if (du < 0) continue;
            double v = campBonus[ci] - K.wProg * K.campW * du - (campUsed[ci] ? K.campUsedPen : 0);
            if (v > bestVal) { bestVal = v; bm = {0, camps[ci], -1, -1, v}; bdf = &dc; }
        }
        MIS[k] = bm; MISDF[k] = bdf;
        for (int ci = 0; ci < (int)camps.size(); ci++) if (bm.kind == 0 && bm.cell == camps[ci]) campUsed[ci] = 1;
    }

    // 4. 候选动作 + 两轮坐标下降
    array<int, 3> dir = {0, 0, 0};
    vector<vector<int>> cands(3);
    for (int k = 0; k < 3; k++) {
        if (MYSTART[k] < 0) continue;
        vector<int>& cd = cands[k];
        auto add = [&](int d) { if (find(cd.begin(), cd.end(), d) == cd.end()) cd.push_back(d); };
        if (MISDF[k]) add(bestDir(MYSTART[k], *MISDF[k]));
        if (U[MYID[k]].flag < 0) {
            add(bestDir(MYSTART[k], DFH));
            // 备选旗点
            for (int fi = 0; fi < nf && (int)cd.size() < 8; fi++) {
                if (!pickable(F[fi]) || MIS[k].flag == fi) continue;
                add(bestDir(MYSTART[k], DFc(CIDX(F[fi].x, F[fi].y))));
            }
        }
        add(0);
        for (int d = 1; d <= 4; d++) add(d);
        dir[k] = MISDF[k] ? bestDir(MYSTART[k], *MISDF[k]) : 0;
    }

    // 处理顺序：携旗 > 采集（近的优先）> 拦截 > 驻点
    vector<int> order;
    for (int k = 0; k < 3; k++) if (MYSTART[k] >= 0) order.push_back(k);
    sort(order.begin(), order.end(), [&](int a, int b) {
        auto pr = [&](int k) {
            const Unit& u = U[MYID[k]];
            if (u.flag >= 0) return 0.0;
            if (MIS[k].kind == 1) return 1.0;
            if (MIS[k].kind == 2) return 2.0;
            return 3.0;
        };
        double pa = pr(a), pb = pr(b);
        if (pa != pb) return pa < pb;
        if (MIS[a].kind == 1) return MIS[a].cost < MIS[b].cost;
        return MYID[a] < MYID[b];
    });

    vector<int> myEnd(3, -1), myAct(3, 0), myTgt(3, -1);
    jointScore(dir, myEnd, myAct, myTgt);
    for (int round = 0; round < 2; round++) {
        for (int k : order) {
            if (cands[k].empty()) continue;
            int bd = dir[k]; double bs = -1e18;
            for (int d : cands[k]) {
                dir[k] = d;
                double s = jointScore(dir, myEnd, myAct, myTgt);
                if (s > bs + 1e-9) { bs = s; bd = d; }
            }
            dir[k] = bd;
        }
    }
    jointScore(dir, myEnd, myAct, myTgt);   // 固定最终结果

    // 5. 输出
    string out = to_string(TURN);
    for (int k = 0; k < 3; k++) {
        if (MYSTART[k] < 0) { out += " S -"; continue; }
        out += " "; out += DCH[dir[k]];
        if (myAct[k] == 1) out += " P";
        else if (myAct[k] == 2) out += " X";
        else if (myAct[k] == 3 && myTgt[k] >= 0) out += " " + to_string(myTgt[k]);
        else out += " -";
    }
    if (DBGON) {
        int pk = 0; for (const Flag& f : F) if (pickable(f)) pk++;
        cerr << "[bot] turn " << TURN << " pick=" << pk << " -> " << out << " | missions:";
        for (int k = 0; k < 3; k++) cerr << " " << MYID[k] << ":kind" << MIS[k].kind << "@" << MIS[k].cell;
        cerr << "\n";
    }
    return out;
}

// 开发期调参用：BOTK="name=value;name=value" 覆盖 Tunables（交付前删除本段）
static void applyTuningOverrides() {
    const char* spec = getenv("BOTK");
    if (!spec) return;
    static map<string, double*> tb;
    if (tb.empty()) {
        tb["wCap"] = &K.wCap; tb["wProg"] = &K.wProg; tb["wPick"] = &K.wPick; tb["wHit"] = &K.wHit;
        tb["wKill"] = &K.wKill; tb["wKillCarry"] = &K.wKillCarry; tb["wHunt"] = &K.wHunt;
        tb["wHitOnMe"] = &K.wHitOnMe; tb["wHitOnMeCarry"] = &K.wHitOnMeCarry; tb["wWounded"] = &K.wWounded;
        tb["wDeath"] = &K.wDeath; tb["wDeathCarry"] = &K.wDeathCarry; tb["wBlocked"] = &K.wBlocked;
        tb["wEnemyCamp"] = &K.wEnemyCamp; tb["contestW"] = &K.contestW; tb["campW"] = &K.campW;
        tb["centerBonus"] = &K.centerBonus; tb["ownBonus"] = &K.ownBonus; tb["campUsedPen"] = &K.campUsedPen;
        tb["wEscort"] = &K.wEscort; tb["fetchMult"] = &K.fetchMult;
    }
    static map<string, int*> ti;
    if (ti.empty()) {
        ti["huntRadius"] = &K.huntRadius; ti["escortMin"] = &K.escortMin; ti["escortMax"] = &K.escortMax;
        ti["lateBy"] = &K.lateBy; ti["endSlack"] = &K.endSlack; ti["pinTeams"] = &K.pinTeams; ti["pinRace"] = &K.pinRace;
        ti["huntMaxHome"] = &K.huntMaxHome;
    }
    string s(spec);
    size_t p = 0;
    while (p < s.size()) {
        size_t e = s.find(';', p);
        string kv = s.substr(p, e == string::npos ? string::npos : e - p);
        p = (e == string::npos) ? s.size() : e + 1;
        size_t eq = kv.find('=');
        if (eq == string::npos) continue;
        string name = kv.substr(0, eq);
        double val = atof(kv.c_str() + eq + 1);
        if (tb.count(name)) *tb[name] = val;
        else if (ti.count(name)) *ti[name] = (int)llround(val);
    }
}

int main() {
    ios::sync_with_stdio(false);
    DBGON = getenv("BOTDBG") != nullptr;
    applyTuningOverrides();
    Game g;
    while (read_turn(g)) {
        string line;
        try {
            line = decide(g);
        } catch (...) {
            line = to_string(g.turn) + " S - S - S -";
        }
        cout << line << endl;
    }
    return 0;
}
