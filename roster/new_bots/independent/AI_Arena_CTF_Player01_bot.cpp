#if defined(__has_include) && __has_include(<bits/stdc++.h>)
#include <bits/stdc++.h>
#else
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <algorithm>
#include <deque>
#include <array>
#endif
using namespace std;

struct Unit { int id, team, x, y, hp, flag, respawn_at; };      // 阵亡时 x = y = -1
struct Flag { int id; string status; int x, y, holder, return_at; };
struct Event { string type; vector<int> args; };                // 各类事件的字段见 RULES.md

struct Game {
    // 开局信息（INIT 块）
    int teams = 0, me = 0, size = 0, turns = 0, hp = 0, damage = 0, respawn = 0, flag_return = 0, flag_cooldown = 0;
    vector<string> map;                  // map[y][x]：'#' 墙，'.' 空地
    vector<pair<int, int>> bases;        // 各阵营基地中心，基地是以它为中心的 3×3
    vector<pair<int, int>> spots;        // 全部旗点
    // 每回合更新（TURN 块）
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;
    vector<Event> events;                // 上一回合发生的事件
};

// 读一个块。读到 INIT 时继续读到 TURN 块结束。返回 false 表示输入结束。
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

// ---------------- 策略 ----------------
// 每回合：先按引擎的规则算出敌人能打到的格子（威胁图）、各目标的距离场，
// 然后枚举己方 3 个角色的 移动×动作 组合（约几百到几千种），用同样的移动/攻击/拾旗/交旗规则
// 模拟一步，给结果打分，选分最高的一组。敌人假设原地不动（威胁图已经把它们的一步移动算进去）。

namespace {

constexpr int INF = 1 << 20;
constexpr int DX[5] = {0, 0, 0, -1, 1};      // 0 原地 1 上 2 下 3 左 4 右
constexpr int DY[5] = {0, -1, 1, 0, 0};
constexpr char MCH[5] = {'S', 'U', 'D', 'L', 'R'};

// 评分权重
struct Weights {
    double cap = 1000;       // 交一面旗
    double carry = 1000;     // 持旗本身的价值（要大到即使离基地很远，拾旗也不会被判为亏）
    double home = 20;        // 持旗者离己方基地每一步的代价
    double obj = 30;         // 无旗者离旗子每一步的代价
    double post = 35;        // 无旗者离自己负责的旗点每一步的代价（旗点待命）
    double intercept = 20;   // 无旗者离很近（4 格内）的敌方持旗者每一步的代价
    double esc = 6;          // 无旗者离己方持旗者超过 2 格的每一格代价
    double hit = 8;          // 每个可能命中自己的敌人（无旗）
    double hitCarry = 90;    // 每个可能命中自己的敌人（携旗：旗子会一起丢）
    int wcost = 15;          // 无旗者寻路时，每个会被打到的格子额外的代价（一步 = 10）
    int carrierWCost = 60;   // 携旗者回基地寻路时，每个会被打到的格子额外的代价
    double death = 600;      // 携旗者本回合可能被击杀
    double deathNC = 80;     // 无旗者本回合可能被击杀（威胁按最坏情况算，所以打折）
    double flagloss = 500;   // 可能被击杀且携旗
    double dmg = 25;         // 每次命中敌人
    double fragile = 300;    // 携旗者这回合之后只剩一击就死
    double kill = 600;       // 击杀
    double killc = 500;      // 击杀携旗者的额外奖励
};
const Weights W{};

struct Opt { int mv, act, tgt, cell; };   // act: 0 原地 1 拾旗 2 攻击 tgt

string decide(const Game& g) {
    const int T = g.turn, me = g.me, S = g.size, C = S / 2, N = g.teams;
    const int NU = static_cast<int>(g.units.size());
    const int F = static_cast<int>(g.flags.size());
    const int DMG = g.damage, HPMAX = g.hp;
    auto cellOf = [S](int x, int y) { return y * S + x; };

    // 1. 墙与基地
    vector<char> wall(S * S, 0);
    for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++) wall[cellOf(x, y)] = g.map[y][x] == '#';
    auto isWall = [&](int x, int y) { return x < 0 || y < 0 || x >= S || y >= S || wall[cellOf(x, y)]; };
    auto canAttack = [&](int ax, int ay, int bx, int by) {
        int dx = bx - ax, dy = by - ay;
        int d = abs(dx) + abs(dy);
        if (d == 0 || d > 2) return false;
        if (abs(dx) == 2 && dy == 0) return !isWall(ax + dx / 2, ay);
        if (abs(dy) == 2 && dx == 0) return !isWall(ax, ay + dy / 2);
        if (abs(dx) == 1 && abs(dy) == 1) return !isWall(ax + dx, ay) || !isWall(ax, ay + dy);
        return true;
    };

    vector<int> baseOwner(S * S, -1);
    vector<vector<int>> spawn(N);      // 各队重生格，按引擎的顺序（离地图中心近优先）
    vector<int> homeCells;
    for (int t = 0; t < N; t++) {
        const int bx = g.bases[t].first, by = g.bases[t].second;
        vector<pair<int, int>> cells;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) cells.push_back({bx + dx, by + dy});
        for (auto [x, y] : cells) baseOwner[cellOf(x, y)] = t;
        auto key = [&](pair<int, int> p) {
            return make_pair(abs(p.first - C) + abs(p.second - C), abs(p.first - bx) + 2 * abs(p.second - by));
        };
        stable_sort(cells.begin(), cells.end(), [&](auto a, auto b) { return key(a) < key(b); });
        for (auto [x, y] : cells) spawn[t].push_back(cellOf(x, y));
    }
    for (int c = 0; c < S * S; c++)
        if (baseOwner[c] == me) homeCells.push_back(c);

    // 2. 角色位置（含本回合即将重生的角色，与引擎同一顺序放置）
    vector<int> ucell(NU, -1), uhp(NU, 0), ucarry(NU, -1), uteam(NU, 0);
    vector<char> uimm(NU, 0);
    vector<int> occ(S * S, -1);
    for (const auto& u : g.units) {
        uteam[u.id] = u.team;
        uhp[u.id] = u.hp;
        ucarry[u.id] = u.flag;
        if (u.x >= 0 && u.y >= 0) {
            ucell[u.id] = cellOf(u.x, u.y);
            occ[ucell[u.id]] = u.id;
        }
    }
    for (const auto& u : g.units) {
        if (ucell[u.id] >= 0 || u.respawn_at < 0 || T < u.respawn_at) continue;
        for (int c : spawn[u.team])
            if (occ[c] < 0) {
                ucell[u.id] = c;
                occ[c] = u.id;
                uimm[u.id] = 1;               // 重生当回合不能攻击，也不会被攻击
                uhp[u.id] = HPMAX;
                ucarry[u.id] = -1;
                break;
            }
    }

    // 3. 旗子
    vector<int> fcell(F, -1), fstate(F, 3);   // 0 home 1 dropped 2 carried 3 cooldown
    for (const auto& f : g.flags) {
        fstate[f.id] = f.status == "home" ? 0 : f.status == "dropped" ? 1 : f.status == "carried" ? 2 : 3;
        if (f.x >= 0 && f.y >= 0) fcell[f.id] = cellOf(f.x, f.y);
    }
    auto flagAvail = [&](int f) { return (fstate[f] == 0 || fstate[f] == 1) && fcell[f] >= 0; };

    // 4. 敌方威胁图：每个能在下一回合打到某格的敌人（可先走一步）计一次
    vector<int> threat(S * S, 0), stamp(S * S, -1);
    for (int e = 0; e < NU; e++) {
        if (uteam[e] == me || ucell[e] < 0 || uimm[e] || ucarry[e] >= 0) continue;
        const int ex = ucell[e] % S, ey = ucell[e] / S;
        for (int mv = 0; mv < 5; mv++) {
            const int qx = ex + DX[mv], qy = ey + DY[mv];
            if (mv && isWall(qx, qy)) continue;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++) {
                    const int d = abs(dx) + abs(dy);
                    if (d == 0 || d > 2) continue;
                    const int cx = qx + dx, cy = qy + dy;
                    if (isWall(cx, cy) || !canAttack(qx, qy, cx, cy)) continue;
                    const int c = cellOf(cx, cy);
                    if (stamp[c] != e) { stamp[c] = e; threat[c]++; }
                }
        }
    }

    // 5. 距离场（只看墙）
    auto bfs = [&](const vector<int>& src) {
        vector<int> d(S * S, INF);
        deque<int> q;
        for (int c : src)
            if (d[c] == INF) { d[c] = 0; q.push_back(c); }
        while (!q.empty()) {
            const int c = q.front();
            q.pop_front();
            const int x = c % S, y = c / S;
            for (int k = 1; k < 5; k++) {
                const int nx = x + DX[k], ny = y + DY[k];
                if (isWall(nx, ny)) continue;
                const int nc = cellOf(nx, ny);
                if (d[nc] == INF) { d[nc] = d[c] + 1; q.push_back(nc); }
            }
        }
        return d;
    };
    // 带威胁的最短路：每走一步记 10，每个会被打到的格子额外加 wcost。单位：一步 = 10。
    // 携旗者的路线对威胁更敏感：丢旗的代价远大于多走几步。
    auto wdist = [&](const vector<int>& src, int wcost) {
        vector<int> d(S * S, INF);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        for (int c : src)
            if (d[c] != 0) { d[c] = 0; pq.push({0, c}); }
        while (!pq.empty()) {
            const auto [dc, c] = pq.top();
            pq.pop();
            if (dc != d[c]) continue;
            const int x = c % S, y = c / S;
            for (int k = 1; k < 5; k++) {
                const int nx = x + DX[k], ny = y + DY[k];
                if (isWall(nx, ny)) continue;
                const int nc = cellOf(nx, ny);
                const int nd = dc + 10 + wcost * threat[nc];
                if (nd < d[nc]) { d[nc] = nd; pq.push({nd, nc}); }
            }
        }
        return d;
    };
    const vector<int> homeCost = wdist(homeCells, W.carrierWCost);

    // 6. 己方角色的可选动作
    const int base = 3 * me;
    vector<vector<Opt>> opts(3);
    vector<char> alive(3, 0);
    vector<int> start(3, -1);
    for (int k = 0; k < 3; k++) {
        const int id = base + k;
        if (id >= NU || ucell[id] < 0) continue;
        alive[k] = 1;
        start[k] = ucell[id];
        const int x = ucell[id] % S, y = ucell[id] / S;
        for (int mv = 0; mv < 5; mv++) {
            const int nx = x + DX[mv], ny = y + DY[mv];
            if (mv && isWall(nx, ny)) continue;
            const int c = cellOf(nx, ny);
            opts[k].push_back({mv, 0, -1, c});
            if (ucarry[id] >= 0) continue;               // 携旗者不能拾旗、不能攻击
            for (int f = 0; f < F; f++)
                if (flagAvail(f) && fcell[f] == c) { opts[k].push_back({mv, 1, -1, c}); break; }
            if (uimm[id]) continue;                      // 重生当回合不能攻击
            vector<pair<int, int>> tg;                   // (优先级, 敌人 id)：携旗者优先，其次血少的
            for (int e = 0; e < NU; e++)
                if (uteam[e] != me && ucell[e] >= 0 && !uimm[e] && canAttack(nx, ny, ucell[e] % S, ucell[e] / S))
                    tg.push_back({(ucarry[e] >= 0 ? 0 : 1000) + uhp[e], e});
            sort(tg.begin(), tg.end());
            for (int i = 0; i < static_cast<int>(tg.size()) && i < 3; i++)
                opts[k].push_back({mv, 2, tg[i].second, c});
        }
    }

    vector<int> carrSrc;
    for (int e = 0; e < NU; e++)
        if (uteam[e] != me && ucell[e] >= 0 && !uimm[e] && ucarry[e] >= 0) carrSrc.push_back(ucell[e]);
    const vector<int> carrDist = bfs(carrSrc);

    // 6b. 无旗角色的目标：先把可取的旗分给各自最近的角色（每面旗只分给一人），
    // 剩下的角色去尚未被占的旗点待命，等旗子刷新时第一时间拾取。
    vector<vector<int>> ud(3), tgtDist(3);      // ud：角色到各格的距离；tgtDist：到目标的距离
    vector<int> tgt(3, -1), tgtW(3, 0);         // 目标格；拉力权重（旗 = obj，旗点 = post）
    vector<char> flagClaimed(F, 0), spotClaimed(S * S, 0);
    for (int k = 0; k < 3; k++)
        if (alive[k] && ucarry[base + k] < 0) ud[k] = bfs({start[k]});
    vector<array<int, 3>> pairs;                // (距离, 角色, 旗)
    for (int k = 0; k < 3; k++)
        if (!ud[k].empty())
            for (int f = 0; f < F; f++)
                if (flagAvail(f) && ud[k][fcell[f]] < INF) pairs.push_back({ud[k][fcell[f]], k, f});
    sort(pairs.begin(), pairs.end());
    for (const auto& p : pairs) {
        const int k = p[1], f = p[2];
        if (tgt[k] >= 0 || flagClaimed[f]) continue;
        tgt[k] = fcell[f];
        tgtW[k] = W.obj;
        flagClaimed[f] = 1;
    }
    for (int k = 0; k < 3; k++) {
        if (ud[k].empty() || tgt[k] >= 0) continue;
        int best = -1;
        for (const auto& s : g.spots) {
            const int c = cellOf(s.first, s.second);
            if (!spotClaimed[c] && ud[k][c] < INF && (best < 0 || ud[k][c] < ud[k][best])) best = c;
        }
        if (best >= 0) {
            tgt[k] = best;
            tgtW[k] = W.post;
            spotClaimed[best] = 1;
        }
    }
    for (int k = 0; k < 3; k++)
        if (tgt[k] >= 0) tgtDist[k] = wdist({tgt[k]}, W.wcost);

    // 7. 枚举组合并打分
    vector<int> cntOurs(S * S, 0), enemyAt(S * S, 0), hitsOn(NU, 0), flagTaken(F, 0);
    for (int c = 0; c < S * S; c++)
        if (occ[c] >= 0 && uteam[occ[c]] != me) enemyAt[c] = 1;

    auto evaluate = [&](const array<int, 3>& ch) {
        int fin[3] = {-1, -1, -1}, want[3] = {-1, -1, -1};
        bool blk[3] = {false, false, false};
        // 移动结算（与引擎相同：同格争抢都不动；走进正在原地不动的格子不动；敌人视为原地不动）
        for (int k = 0; k < 3; k++) if (alive[k]) { want[k] = opts[k][ch[k]].cell; cntOurs[want[k]]++; }
        for (int k = 0; k < 3; k++)
            if (alive[k]) blk[k] = cntOurs[want[k]] + enemyAt[want[k]] > 1;
        for (int k = 0; k < 3; k++) if (alive[k]) cntOurs[want[k]] = 0;
        for (bool changed = true; changed;) {
            changed = false;
            for (int k = 0; k < 3; k++) {
                if (!alive[k] || blk[k] || want[k] == start[k]) continue;
                for (int j = 0; j < 3; j++)
                    if (alive[j] && start[j] == want[k] && j != k && (blk[j] || want[j] == start[j])) {
                        blk[k] = true; changed = true; break;
                    }
            }
        }
        for (int k = 0; k < 3; k++) if (alive[k]) fin[k] = blk[k] ? start[k] : want[k];

        double E = 0;
        // 攻击结算（同时生效，阵亡在之后）
        bool landed[3] = {false, false, false};
        int touched[3], nt = 0;
        for (int k = 0; k < 3; k++) {
            if (!alive[k] || opts[k][ch[k]].act != 2) continue;
            const int t = opts[k][ch[k]].tgt;
            if (!canAttack(fin[k] % S, fin[k] / S, ucell[t] % S, ucell[t] / S)) continue;
            landed[k] = true;
            if (hitsOn[t] == 0) touched[nt++] = t;
            hitsOn[t]++;
        }
        for (int i = 0; i < nt; i++) {
            const int t = touched[i];
            if (uhp[t] - DMG * hitsOn[t] <= 0) E += W.kill + (ucarry[t] >= 0 ? W.killc : 0);
            hitsOn[t] = 0;
        }
        for (int k = 0; k < 3; k++) if (landed[k]) E += W.dmg;

        // 致命判断（威胁按最坏情况计算：所有能打到的敌人都打这个角色）
        bool lethal[3] = {false, false, false};
        for (int k = 0; k < 3; k++) {
            if (!alive[k]) continue;
            const int id = base + k;
            lethal[k] = !uimm[id] && DMG * threat[fin[k]] >= uhp[id];
        }
        // 拾旗（先于交旗；同一格取编号最小的）
        int pick[3] = {-1, -1, -1};
        for (int k = 0; k < 3; k++) {
            if (!alive[k] || lethal[k] || opts[k][ch[k]].act != 1 || ucarry[base + k] >= 0) continue;
            for (int f = 0; f < F; f++)
                if (flagAvail(f) && !flagTaken[f] && fcell[f] == fin[k]) {
                    pick[k] = f; flagTaken[f] = 1; break;
                }
        }
        for (int k = 0; k < 3; k++) if (pick[k] >= 0) flagTaken[pick[k]] = 0;

        bool carry[3] = {false, false, false}, cap[3] = {false, false, false};
        for (int k = 0; k < 3; k++) {
            if (!alive[k]) continue;
            carry[k] = ucarry[base + k] >= 0 || pick[k] >= 0;
            cap[k] = carry[k] && !lethal[k] && baseOwner[fin[k]] == me;
        }

        // 局面评估
        bool anyCarrier = false;
        for (int k = 0; k < 3; k++) if (alive[k] && carry[k] && !lethal[k] && !cap[k]) anyCarrier = true;
        for (int k = 0; k < 3; k++) {
            if (!alive[k]) continue;
            const int id = base + k, f = fin[k];
            if (lethal[k]) {
                E -= carry[k] ? W.death : W.deathNC;
                if (carry[k]) E -= W.flagloss;
                continue;
            }
            const int hits = uimm[id] ? 0 : threat[f];
            E -= (carry[k] ? W.hitCarry : W.hit) * hits;
            if (carry[k] && hits && uhp[id] - DMG * hits <= DMG) E -= W.fragile;
            if (cap[k]) { E += W.cap; continue; }
            if (carry[k]) {
                E += W.carry - W.home * min(homeCost[f], 2000) / 10.0;
            } else {
                double pull = 1e18;
                if (tgt[k] >= 0 && tgtDist[k][f] < INF) pull = tgtW[k] * tgtDist[k][f] / 10.0;
                if (carrDist[f] <= 4) pull = min(pull, W.intercept * carrDist[f]);   // 只拦近处的敌方持旗者
                if (pull < 1e17) E -= pull;
                if (anyCarrier) {
                    int best = INF;
                    for (int j = 0; j < 3; j++)
                        if (alive[j] && carry[j] && !lethal[j] && !cap[j])
                            best = min(best, abs(fin[j] % S - f % S) + abs(fin[j] / S - f / S));
                    E -= W.esc * max(0, best - 2);
                }
            }
        }
        return E;
    };

    array<int, 3> sz = {1, 1, 1}, ch = {0, 0, 0}, bestCh = {0, 0, 0};
    for (int k = 0; k < 3; k++) if (alive[k]) sz[k] = static_cast<int>(opts[k].size());
    double bestE = -1e300;
    if (alive[0] || alive[1] || alive[2]) {
        while (true) {
            const double E = evaluate(ch);
            if (E > bestE) { bestE = E; bestCh = ch; }
            int k = 0;
            while (k < 3 && ++ch[k] == sz[k]) { ch[k] = 0; k++; }
            if (k == 3) break;
        }
    }

    // 8. 输出
    // 旗子在回合开头（结算第 1 步）才刷新到旗点上，而拾旗在第 4 步，所以站在旗点上的无旗角色
    // 即使本回合看不到旗，也发 P：旗一刷新就能当回合拾起；没有旗时 P 什么也不做。
    vector<char> spotCell(S * S, 0);
    for (const auto& s : g.spots) spotCell[cellOf(s.first, s.second)] = 1;
    string out = to_string(T);
    string notes;
    for (int k = 0; k < 3; k++) {
        if (!alive[k]) { out += " S -"; notes += (k ? ";" : "") + string("-"); continue; }
        const Opt& o = opts[k][bestCh[k]];
        const bool idleOnSpot = o.act == 0 && ucarry[base + k] < 0 && spotCell[o.cell];
        out += ' ';
        out += MCH[o.mv];
        out += ' ';
        if (o.act == 1 || idleOnSpot) out += 'P';
        else if (o.act == 0) out += '-';
        else out += to_string(o.tgt);
        const char* note = ucarry[base + k] >= 0 ? "carry" : o.act == 1 ? "grab" : o.act == 2 ? "hit" : idleOnSpot ? "post" : "run";
        notes += (k ? ";" : "") + string(note);
    }
    out += " # " + notes;
    return out;
}

}  // namespace

int main() {
    ios::sync_with_stdio(false);
    Game g;
    while (read_turn(g)) {
        cout << decide(g) << endl;   // endl 会立即刷新输出
    }
}
