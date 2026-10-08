// ============================================================================
//  夺旗 bot —— 单一文件 / C++20 / 只用标准库 / 兼容 g++ 与 clang++
//
//  设计要点
//  1) 地图、基地、旗点在整局内恒定 => 所有寻路距离场在首回合一次性预计算，
//     之后每回合不再做整图 BFS（只有掉落旗 / 敌方携旗者这类动态目标才现算一次）。
//  2) 每回合把 3 个角色当成一次小型指派问题：任务 = 交旗 / 拾旗 / 拦截敌方携旗者 /
//     旗点预置，用暴力枚举取全队效用和最大，而不是各走各的"最近的旗"。
//  3) 选步时把"推进距离场 + 打人 + 挨打风险"放在同一个标量里比较，
//     因此会出现"原地不动补最后一刀"这类正确取舍。
//  4) 移动阻塞完全复刻引擎语义（同格互撞、业主不动连锁阻塞、同阵营可对换），
//     并用局部重解修复被阻塞的角色。
//  5) 任何异常都兜底成合法的 "t S - S - S -"，绝不给裁判一个格式错误。
// ============================================================================
#include <algorithm>
#include <climits>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct Unit { int id = 0, team = 0, x = -1, y = -1, hp = 0, flag = -1, respawn_at = -1; };
struct Flag { int id = 0; string status; int x = -1, y = -1, holder = -1, return_at = -1; };

// ---------------------------------------------------------------------------
//  可调参数（集中在这里，便于生成变体做联赛评测）
// ---------------------------------------------------------------------------
struct Params {
    double wProgress    = 3.0;   // 向目标推进一格的权重
    double wAttack      = 1.0;   // 攻击价值权重
    double wDanger      = 1.2;   // 受击风险权重（挨一下不等于死，别把旗点让出去）
    double lethalPen    = 1.5;   // 会被打死的额外惩罚
    double contestFar   = 0.55;  // 明显抢不过别人时的折扣
    double contestNear  = 0.85;  // 势均力敌时的折扣
    double interceptSlack = 4.0; // 拦截可行性宽限
    double denyWeight   = 0.5;   // 拒止（杀携旗者）价值
    double leaderBonus  = 0.8;   // 对手分数高于我方时的额外权重
    double carrierBonus = 2.5;   // 目标是携旗者
    double lethalBonus  = 6.0;   // 这一下能打死
    double finishBonus  = 1.2;   // 残血额外收益
    double standbyOwn   = 0.05;  // 常驻自家旗点的基础价值
    double classCollect = 100.0; // 优先级分层：场上有旗可捡时，捡旗压过一切其它任务
    double urgentDeny   = 45.0;  // 敌方携旗者已逼进其基地时的抢杀价值
    int    standbyLead  = 4;     // 旗子回位前几回合开始预置
};

// 下标 0..4 依次是 S / L / R / U / D —— DC/DR 与 MC 必须严格同序
static const int DR[5] = {0, 0, 0, -1, 1};
static const int DC[5] = {0, -1, 1, 0, 0};
static const char MC[5] = {'S', 'L', 'R', 'U', 'D'};

struct Bot {
    // ---- 开局信息 ----
    int teams = 0, me = 0, size = 0, turns = 400, hpMax = 100, dmg = 34, respawnT = 10, flagReturn = 15, flagCooldown = 8;
    vector<string> grid;
    vector<pair<int, int>> baseCenters, spots;

    // ---- 每回合局面 ----
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;

    // ---- 静态预计算 ----
    int W = 0, H = 0;
    vector<char> wall;
    vector<vector<int>> baseCells, spawnOrder;
    vector<int> dBaseMe;             // 到"我方基地"的距离场
    vector<vector<int>> dSpot;       // 每个旗点的距离场
    vector<vector<int>> dBaseTeam;   // 每个阵营基地的距离场

    Params P;
    bool built = false;

    int cellOf(int x, int y) const { return y * W + x; }
    bool inb(int x, int y) const { return x >= 0 && y >= 0 && x < W && y < H; }

    // ------------------------------------------------------------------
    void bfs(const vector<int>& src, vector<int>& dist) {
        dist.assign((size_t)W * H, -1);
        vector<int> q;
        q.reserve((size_t)W * H);
        for (int c : src)
            if (c >= 0 && c < W * H && !wall[c] && dist[c] < 0) { dist[c] = 0; q.push_back(c); }
        for (size_t h = 0; h < q.size(); ++h) {
            int c = q[h], d = dist[c], x = c % W, y = c / W;
            for (int k = 1; k <= 4; ++k) {
                int nx = x + DC[k], ny = y + DR[k];
                if (!inb(nx, ny)) continue;
                int nc = ny * W + nx;
                if (wall[nc] || dist[nc] >= 0) continue;
                dist[nc] = d + 1;
                q.push_back(nc);
            }
        }
    }

    // 完全复刻引擎的 can_attack
    bool canAttack(int a, int b) const {
        int ax = a % W, ay = a / W, bx = b % W, by = b / W;
        int dx = bx - ax, dy = by - ay;
        if (abs(dx) + abs(dy) > 2) return false;
        if (abs(dx) == 2 && dy == 0) return !wall[ay * W + ax + dx / 2];
        if (abs(dy) == 2 && dx == 0) return !wall[(ay + dy / 2) * W + ax];
        if (abs(dx) == 1 && abs(dy) == 1) return !wall[(ay + dy) * W + ax] || !wall[ay * W + ax + dx];
        return true;
    }

    void buildStatic() {
        W = H = size;
        wall.assign((size_t)W * H, 0);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) wall[y * W + x] = (y < (int)grid.size() && x < (int)grid[y].size() && grid[y][x] == '#') ? 1 : 0;

        int cc = size / 2, cx = cc, cy = cc;
        baseCells.assign(teams, {});
        spawnOrder.assign(teams, {});
        for (int t = 0; t < teams; ++t) {
            int bx = baseCenters[t].first, by = baseCenters[t].second;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (inb(bx + dx, by + dy)) baseCells[t].push_back(cellOf(bx + dx, by + dy));
            spawnOrder[t] = baseCells[t];
            stable_sort(spawnOrder[t].begin(), spawnOrder[t].end(), [&](int p, int q) {
                int px = p % W, py = p / W, qx = q % W, qy = q / W;
                int dp = abs(px - cx) + abs(py - cy), dq = abs(qx - cx) + abs(qy - cy);
                if (dp != dq) return dp < dq;
                return (abs(px - bx) + 2 * abs(py - by)) < (abs(qx - bx) + 2 * abs(qy - by));
            });
        }
        bfs(baseCells[me], dBaseMe);
        dSpot.assign(spots.size(), {});
        for (size_t s = 0; s < spots.size(); ++s)
            bfs(vector<int>{cellOf(spots[s].first, spots[s].second)}, dSpot[s]);
        dBaseTeam.assign(teams, {});
        for (int t = 0; t < teams; ++t) bfs(baseCells[t], dBaseTeam[t]);
        built = true;
    }

    // ------------------------------------------------------------------
    // 预测本回合各角色的有效位置（含本回合将重生的角色，逐 id 复刻引擎占位顺序）
    // ------------------------------------------------------------------
    void predictPositions(vector<int>& ep) const {
        int N = (int)units.size();
        ep.assign(N, -1);
        vector<char> taken((size_t)W * H, 0);
        for (const auto& u : units)
            if (u.x >= 0) { ep[u.id] = cellOf(u.x, u.y); taken[ep[u.id]] = 1; }
        for (const auto& u : units) {
            if (u.x >= 0) continue;
            if (u.respawn_at < 0 || turn < u.respawn_at) continue;
            for (int c : spawnOrder[u.team])
                if (!taken[c]) { ep[u.id] = c; taken[c] = 1; break; }
        }
    }

    bool willRespawnNow(const Unit& u) const { return u.x < 0 && u.respawn_at >= 0 && turn >= u.respawn_at; }

    // 复刻引擎移动结算，标出被阻塞的角色
    void resolveBlocking(const vector<int>& ep, const vector<int>& want, vector<char>& blocked) const {
        int N = (int)units.size();
        blocked.assign(N, 0);
        vector<pair<int, int>> v;
        v.reserve(N);
        for (int i = 0; i < N; ++i)
            if (ep[i] >= 0) v.push_back({want[i], i});
        sort(v.begin(), v.end());
        for (size_t i = 0; i < v.size();) {
            size_t j = i;
            while (j < v.size() && v[j].first == v[i].first) ++j;
            if (j - i > 1)
                for (size_t k = i; k < j; ++k) blocked[v[k].second] = 1;
            i = j;
        }
        vector<int> owner((size_t)W * H, -1);
        for (int i = 0; i < N; ++i)
            if (ep[i] >= 0) owner[ep[i]] = i;
        for (int i = 0; i < N; ++i) {
            if (ep[i] < 0) continue;
            int j = owner[want[i]];
            if (j >= 0 && j != i && want[j] == ep[i] && units[i].team != units[j].team) { blocked[i] = 1; blocked[j] = 1; }
        }
        for (int it = 0; it < N + 2; ++it) {
            bool more = false;
            for (int i = 0; i < N; ++i) {
                if (ep[i] < 0 || blocked[i] || want[i] == ep[i]) continue;
                int j = owner[want[i]];
                if (j < 0 || j == i) continue;
                if (blocked[j] || want[j] == ep[j]) { blocked[i] = 1; more = true; }
            }
            if (!more) break;
        }
    }

    // ------------------------------------------------------------------
    string decide();
};

string Bot::decide() {
    const int N = (int)units.size();
    const int meBase = 3 * me;

    vector<int> ep;
    predictPositions(ep);

    // 本回合开头会有旗子被挪到一个随机空旗点（cooldown/dropped 到期）。我们看不到它落在哪
    // 一格，但"站在旗点上按 P"是免费的握手：旗子正好落在脚下时，第 4 步就 0 距离捡走。
    bool anyReturnThisTurn = false;
    for (const auto& f : flags) {
        if (f.status != "cooldown" && f.status != "dropped") continue;
        if (f.return_at >= 0 && f.return_at <= turn) anyReturnThisTurn = true;
    }
    vector<char> isSpot((size_t)W * H, 0);
    for (const auto& sp : spots)
        if (inb(sp.first, sp.second)) isSpot[cellOf(sp.first, sp.second)] = 1;

    // ---------- 1. 威胁图：敌方本回合能攻击到哪些格 ----------
    vector<int> threat((size_t)W * H, 0);
    for (const auto& e : units) {
        if (e.team == me) continue;
        int c0 = ep[e.id];
        if (c0 < 0) continue;
        if (willRespawnNow(e)) continue;             // 刚重生：免疫，打不到人也不会被打
        int x0 = c0 % W, y0 = c0 / W;
        for (int k = 0; k < 5; ++k) {
            int x = x0 + DC[k], y = y0 + DR[k];
            if (!inb(x, y)) continue;
            int p = y * W + x;
            if (wall[p]) continue;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    if (abs(dx) + abs(dy) > 2) continue;
                    int qx = x + dx, qy = y + dy;
                    if (!inb(qx, qy)) continue;
                    int q = qy * W + qx;
                    if (wall[q]) continue;
                    if (canAttack(p, q)) threat[q]++;
                }
        }
    }

    // ---------- 2. 可拾取的旗子（每面一个距离场） ----------
    vector<vector<int>> dyn;                       // 本回合现算的距离场
    struct Pick { int flagId, cell, spotIdx, dynIdx; };
    vector<Pick> picks;

    vector<char> spotHasFlag(spots.size(), 0);
    for (const auto& f : flags)
        if (f.status == "home" && f.x >= 0)
            for (size_t s = 0; s < spots.size(); ++s)
                if (spots[s].first == f.x && spots[s].second == f.y) spotHasFlag[s] = 1;
    int emptySpots = 0;
    for (size_t s = 0; s < spots.size(); ++s) emptySpots += (spotHasFlag[s] ? 0 : 1);
    if (emptySpots < 1) emptySpots = 1;

    for (const auto& f : flags) {
        if (f.status != "home" && f.status != "dropped") continue;
        if (f.x < 0) continue;
        // 本回合就会被挪走的旗：旧坐标即将失效，跑去旧位置是白跑一整回合
        if (f.return_at >= 0 && f.return_at <= turn) continue;
        Pick pk{f.id, cellOf(f.x, f.y), -1, -1};
        for (size_t s = 0; s < spots.size(); ++s)
            if (spots[s].first == f.x && spots[s].second == f.y) { pk.spotIdx = (int)s; break; }
        if (pk.spotIdx < 0) {
            dyn.push_back({});
            bfs(vector<int>{pk.cell}, dyn.back());
            pk.dynIdx = (int)dyn.size() - 1;
        }
        picks.push_back(pk);
    }
    auto fieldOf = [&](const Pick& pk) -> const vector<int>& {
        return pk.spotIdx >= 0 ? dSpot[pk.spotIdx] : dyn[pk.dynIdx];
    };

    // ---------- 3. 拦截目标：敌方携旗者（按离其基地近的优先，最多 4 个） ----------
    struct Ict { int unitId, dynIdx, theirBase; double pr; };
    vector<Ict> icts;
    {
        vector<pair<int, int>> cand;               // (到其基地距离, 单位 id)
        for (const auto& e : units) {
            if (e.team == me || e.flag < 0) continue;
            int c0 = ep[e.id];
            if (c0 < 0) continue;
            int tb = dBaseTeam[e.team][c0];
            if (tb < 0) continue;
            cand.push_back({tb, e.id});
        }
        sort(cand.begin(), cand.end());
        if (cand.size() > 4) cand.resize(4);
        for (auto& c : cand) {
            int c0 = ep[c.second];
            dyn.push_back({});
            bfs(vector<int>{c0}, dyn.back());
            Ict it{c.second, (int)dyn.size() - 1, c.first, score[units[c.second].team] > score[me] ? 1.0 : 0.0};
            icts.push_back(it);
        }
    }

    // ---------- 4. 预测敌人下一格（携旗者回其基地，否则去最近的可用旗） ----------
    vector<int> estep(N, -1);
    for (const auto& e : units) {
        int c0 = ep[e.id];
        if (c0 < 0 || e.team == me) continue;
        if (willRespawnNow(e)) { estep[e.id] = c0; continue; }
        const vector<int>* F = nullptr;
        if (e.flag >= 0) F = &dBaseTeam[e.team];
        else {
            int bd = INT_MAX;
            for (const auto& pk : picks) {
                int d = fieldOf(pk)[c0];
                if (d >= 0 && d < bd) { bd = d; F = &fieldOf(pk); }
            }
            if (!F) F = &dBaseTeam[e.team];
        }
        int x = c0 % W, y = c0 / W, bx = x, by = y, bd = (*F)[c0];
        for (int k = 1; k <= 4; ++k) {
            int nx = x + DC[k], ny = y + DR[k];
            if (!inb(nx, ny)) continue;
            int nc = ny * W + nx;
            if (wall[nc]) continue;
            int d = (*F)[nc];
            if (d >= 0 && (bd < 0 || d < bd)) { bd = d; bx = nx; by = ny; }
        }
        estep[e.id] = by * W + bx;
    }

    // ---------- 5. 任务生成 ----------
    vector<int> freeU;
    for (int k = 0; k < 3; ++k) {
        int id = meBase + k;
        if (ep[id] < 0) continue;                  // 阵亡且本回合不重生
        if (units[id].flag >= 0) continue;         // 携旗者强制回基地
        freeU.push_back(id);
    }
    const int nf = (int)freeU.size();

    struct Task { int kind, a; };                  // 0=拾旗 1=拦截 2=预置
    vector<Task> tasks;
    vector<vector<double>> util(nf);

    const double NEG = -1e18;

    // 5a 拾旗
    for (size_t pi = 0; pi < picks.size(); ++pi) {
        const Pick& pk = picks[pi];
        const vector<int>& F = fieldOf(pk);
        int back = dBaseMe[pk.cell];
        if (back < 0) continue;
        double bestE = 1e9;
        for (const auto& e : units) {
            if (e.team == me) continue;
            int ec = ep[e.id];
            if (ec < 0) continue;
            int de = F[ec];
            if (de >= 0 && de < bestE) bestE = de;
        }
        vector<double> col(nf, NEG);
        bool any = false;
        for (int i = 0; i < nf; ++i) {
            int c = ep[freeU[i]];
            int d1 = F[c];
            if (d1 < 0) continue;
            if (turn + d1 + back > turns + 1) continue;      // 来不及送回（第 turns 回合末仍可交旗）
            double u = 1.0 / (d1 + back + 2.0);
            if (bestE < d1 - 0.5) u *= P.contestFar;
            else if (bestE <= d1 + 1.0) u *= P.contestNear;
            col[i] = u + P.classCollect;                   // 场上真有旗：捡旗是第一优先
            any = true;
        }
        if (!any) continue;
        tasks.push_back({0, (int)pi});
        for (int i = 0; i < nf; ++i) util[i].push_back(col[i]);
    }

    // 5b 拦截敌方携旗者
    for (size_t ii = 0; ii < icts.size(); ++ii) {
        const Ict& it = icts[ii];
        const vector<int>& F = dyn[it.dynIdx];
        vector<double> col(nf, NEG);
        bool any = false;
        for (int i = 0; i < nf; ++i) {
            int c = ep[freeU[i]];
            int d1 = F[c];
            if (d1 < 0) continue;
            if (d1 > it.theirBase + (int)P.interceptSlack) continue;
            double u = P.denyWeight / (d1 + 2.0) * (1.0 + it.pr * P.leaderBonus) / (1.0 + 0.15 * it.theirBase);
            if (it.theirBase <= 1) u += P.classCollect + 40.0;   // 本回合不杀就丢分：抢占一切
            else if (it.theirBase <= 3) u += P.classCollect * 0.6; // 快到家门口了：优先抢杀
            col[i] = u;
            any = true;
        }
        if (!any) continue;
        tasks.push_back({1, (int)ii});
        for (int i = 0; i < nf; ++i) util[i].push_back(col[i]);
    }

    // 5c 预置：把空闲角色压到旗点上"蹲点"。旗子回位在回合开头、拾旗在同回合第 4 步，
    //    所以站在旗点上的角色能在旗子落下的同一回合 0 距离拿走 —— 这是抢旗速度的关键。
    for (size_t s = 0; s < spots.size(); ++s) {
        if (spotHasFlag[s]) continue;                  // 这格已经有旗，有人会去捡
        double v = 0.02;                               // 基础占位价值：让空闲角色分散到各旗点
        for (const auto& f : flags) {
            if (f.status != "dropped" && f.status != "cooldown") continue;
            if (f.return_at < 0) continue;
            int dt = f.return_at - turn;
            if (dt < 0) dt = 0;
            if (dt <= P.standbyLead) v += 2.0 * (1.0 / emptySpots) / (1.0 + dt);
        }
        if ((int)s == me) v += P.standbyOwn;
        if (v <= 0) continue;
        int sc = cellOf(spots[s].first, spots[s].second);
        int home = dBaseMe[sc];
        if (home < 0) continue;
        vector<double> col(nf, NEG);
        bool any = false;
        for (int i = 0; i < nf; ++i) {
            int d1 = dSpot[s][ep[freeU[i]]];
            if (d1 < 0) continue;
            col[i] = v / (d1 + 1.0) / (home + 2.0);
            any = true;
        }
        if (!any) continue;
        tasks.push_back({2, (int)s});
        for (int i = 0; i < nf; ++i) util[i].push_back(col[i]);
    }

    const int T = (int)tasks.size();

    // ---------- 6. 指派：单射枚举，最大化效用和 ----------
    vector<int> assign(nf, -1);
    {
        double bestTot = 0.0;                       // 全部不接任务 = 0
        vector<int> cur(nf, -1), best(nf, -1);
        vector<char> used(T, 0);
        function<void(int, double)> rec = [&](int i, double tot) {
            if (i == nf) {
                if (tot > bestTot) { bestTot = tot; best = cur; }
                return;
            }
            cur[i] = -1;
            rec(i + 1, tot);
            for (int j = 0; j < T; ++j) {
                if (used[j] || util[i][j] <= NEG * 0.5) continue;
                used[j] = 1;
                cur[i] = j;
                rec(i + 1, tot + util[i][j]);
                used[j] = 0;
            }
            cur[i] = -1;
        };
        rec(0, 0.0);
        assign = best;
    }

    // ---------- 7. 每个角色的距离场 ----------
    vector<const vector<int>*> ufield(3, nullptr);
    vector<string> tag(3, "idle");
    for (int k = 0; k < 3; ++k) {
        int id = meBase + k;
        if (ep[id] < 0) continue;
        if (units[id].flag >= 0) { ufield[k] = &dBaseMe; tag[k] = "home"; continue; }
        int fi = -1;
        for (int i = 0; i < nf; ++i) if (freeU[i] == id) { fi = i; break; }
        if (fi >= 0 && assign[fi] >= 0) {
            const Task& t = tasks[assign[fi]];
            if (t.kind == 0) { ufield[k] = &fieldOf(picks[t.a]); tag[k] = "get"; }
            else if (t.kind == 1) { ufield[k] = &dyn[icts[t.a].dynIdx]; tag[k] = "kill"; }
            else { ufield[k] = &dSpot[t.a]; tag[k] = "spot"; }
        } else {
            ufield[k] = &dBaseMe;                    // 没事干就往家靠，缩短下一趟
            tag[k] = "idle";
        }
    }

    // ---------- 8. 选步 ----------
    // 从落点出发能打出的最佳一击
    auto bestAttack = [&](int q, int& tgt) -> double {
        tgt = -1;
        double bv = 0.0;
        for (const auto& e : units) {
            if (e.team == me) continue;
            int c0 = ep[e.id];
            if (c0 < 0) continue;
            if (willRespawnNow(e)) continue;
            double hit = 0.0;
            if (canAttack(q, c0)) hit += 0.5;
            if (estep[e.id] >= 0 && estep[e.id] != c0 && canAttack(q, estep[e.id])) hit += 0.5;
            if (hit <= 0.0) continue;
            double v = hit;
            if (e.hp <= dmg) v += P.lethalBonus * hit;
            if (e.flag >= 0) v += P.carrierBonus * hit;
            v += P.finishBonus * (1.0 - e.hp / (double)hpMax);
            if (score[e.team] > score[me]) v += P.leaderBonus;
            if (v > bv) { bv = v; tgt = e.id; }
        }
        return bv;
    };

    vector<double> sc(3 * 5, NEG);
    vector<int> dest(3, -1), atkTgt(3, -1);

    for (int k = 0; k < 3; ++k) {
        int id = meBase + k;
        if (ep[id] < 0) continue;
        const vector<int>& F = *ufield[k];
        int p0 = ep[id];
        const Unit& u = units[id];
        bool canAtk = (u.flag < 0) && !willRespawnNow(u);
        for (int m = 0; m < 5; ++m) {
            int x = p0 % W + DC[m], y = p0 / W + DR[m];
            int w = p0;
            if (inb(x, y) && !wall[y * W + x]) w = y * W + x;
            int f0 = F[p0], f1 = F[w];
            double prog = (f0 >= 0 && f1 >= 0) ? (double)(f0 - f1) : (f1 < 0 && f0 >= 0 ? -4.0 : 0.0);
            double danger = 0.0;
            if (!willRespawnNow(u) && threat[w] > 0) {
                danger = threat[w] * (double)dmg / max(1, u.hp);
                if (threat[w] * dmg >= u.hp) danger += P.lethalPen;
                if (u.flag >= 0) danger *= 2.5;       // 携旗者死 = 整趟白跑 + 把旗送人，更谨慎
            }
            double av = 0.0;
            int tg = -1;
            if (canAtk) av = bestAttack(w, tg);
            // 风险只用来"在同样能推进的走法之间选更安全的"，绝不能大到让单位原地挨打：
            // 战场上威胁图经常给出 lethal，若不加盖，携旗者会在半路停下来被点死。
            double dpen = P.wDanger * danger;
            if (dpen > P.wProgress * 0.7) dpen = P.wProgress * 0.7;
            sc[k * 5 + m] = P.wProgress * prog + P.wAttack * av - dpen;
            dest[k] = w;
            if (m == 0) atkTgt[k] = tg;              // 占位，后面按最终落点重算
        }
        // 平局时的稳定偏好：优先动，其次朝目标
        for (int m = 0; m < 5; ++m) sc[k * 5 + m] += 1e-6 * (5 - m);
    }

    // ---------- 9. 去冲突：复刻引擎阻塞语义 + 局部重解 ----------
    vector<int> want(N, -1), chosen(3, 0);
    for (int i = 0; i < N; ++i) want[i] = ep[i];
    vector<char> forbid(3 * 5, 0);
    vector<char> blocked;
    for (int round = 0; round < 4; ++round) {
        for (int k = 0; k < 3; ++k) {
            int id = meBase + k;
            if (ep[id] < 0) { want[id] = -1; continue; }
            int bm = 0;
            double bs = NEG;
            for (int m = 0; m < 5; ++m) {
                if (forbid[k * 5 + m]) continue;
                if (sc[k * 5 + m] > bs) { bs = sc[k * 5 + m]; bm = m; }
            }
            chosen[k] = bm;
            int x = ep[id] % W + DC[bm], y = ep[id] / W + DR[bm];
            want[id] = (!inb(x, y) || wall[y * W + x]) ? ep[id] : y * W + x;
        }
        resolveBlocking(ep, want, blocked);
        bool changed = false;
        for (int k = 0; k < 3; ++k) {
            int id = meBase + k;
            if (ep[id] < 0 || !blocked[id]) continue;
            if (want[id] == ep[id]) continue;         // 本来就没打算动
            forbid[k * 5 + chosen[k]] = 1;            // 这一格走不通，换个方向
            changed = true;
        }
        if (!changed) break;
    }

    // 最终落点
    for (int k = 0; k < 3; ++k) {
        int id = meBase + k;
        dest[k] = (ep[id] < 0) ? -1 : want[id];
    }

    // ---------- 10. 组装输出 ----------
    string out = to_string(turn);
    for (int k = 0; k < 3; ++k) {
        int id = meBase + k;
        const Unit& u = units[id];
        if (ep[id] < 0) { out += " S -"; continue; }
        int d = dest[k];
        char mv = 'S';
        if (d >= 0 && d != ep[id]) {
            int x0 = ep[id] % W, y0 = ep[id] / W, x1 = d % W, y1 = d / W;
            int dx = x1 - x0, dy = y1 - y0;
            for (int m = 1; m <= 4; ++m) if (DC[m] == dx && DR[m] == dy) mv = MC[m];
        }
        string act = "-";
        if (u.flag >= 0) {
            act = "-";                                 // 交旗在结算第 6 步自动完成
        } else {
            bool onFlag = false;
            for (const auto& f : flags)
                if ((f.status == "home" || f.status == "dropped") && f.x >= 0 && cellOf(f.x, f.y) == d) { onFlag = true; break; }
            int tg = -1;
            if (!willRespawnNow(u)) bestAttack(d, tg);
            if (onFlag) {
                act = "P";
            } else if (anyReturnThisTurn && d >= 0 && isSpot[d] && !(tg >= 0 && units[tg].hp <= dmg)) {
                act = "P";   // 站在旗点上、本回合有旗子要落点：免费握手，命中即 0 距离拾取
            } else if (tg >= 0) {
                act = to_string(tg);
            }
        }
        out += " ";
        out += mv;
        out += " ";
        out += act;
    }
    // 注释（只进回放，不影响判定）
    out += " #";
    for (int k = 0; k < 3; ++k) {
        if (k) out += ";";
        out += tag[k];
    }
    return out;
}

// ---------------------------------------------------------------------------
//  输入解析（沿用选手包模板）
// ---------------------------------------------------------------------------
static bool read_turn(Bot& g) {
    string w;
    while (cin >> w) {
        if (w == "INIT") {
            cin >> g.teams >> g.me >> g.size >> g.turns >> g.hpMax >> g.dmg >> g.respawnT >> g.flagReturn >> g.flagCooldown;
        } else if (w == "MAP") {
            g.grid.assign(g.size, "");
            for (auto& row : g.grid) cin >> row;
        } else if (w == "BASES") {
            int k; cin >> k; g.baseCenters.resize(k);
            for (auto& b : g.baseCenters) cin >> b.first >> b.second;
        } else if (w == "SPOTS") {
            int k; cin >> k; g.spots.resize(k);
            for (auto& s : g.spots) cin >> s.first >> s.second;
        } else if (w == "TURN") {
            cin >> g.turn;
        } else if (w == "SCORE") {
            g.score.assign(g.teams, 0);
            for (auto& s : g.score) cin >> s;
        } else if (w == "UNITS") {
            int k; cin >> k; g.units.assign(k, Unit{});
            for (auto& u : g.units) { string tag; cin >> tag >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawn_at; }
        } else if (w == "FLAGS") {
            int k; cin >> k; g.flags.assign(k, Flag{});
            for (auto& f : g.flags) { string tag; cin >> tag >> f.id >> f.status >> f.x >> f.y >> f.holder >> f.return_at; }
        } else if (w == "EVENTS") {
            int k; cin >> k;
            string line; getline(cin, line);
            for (int i = 0; i < k; ++i) getline(cin, line);   // 事件本回合用不到，读掉即可
        } else if (w == "END") {
            return true;
        }
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Bot g;
    while (read_turn(g)) {
        string line;
        try {
            if (!g.built) g.buildStatic();
            line = g.decide();
        } catch (const exception& e) {
            static int shown = 0;
            if (shown++ < 3) cerr << "bot-error turn=" << g.turn << " " << e.what() << endl;
            line = to_string(g.turn) + " S - S - S -";     // 兜底：永远给裁判一个合法行
        } catch (...) {
            static int shown = 0;
            if (shown++ < 3) cerr << "bot-error turn=" << g.turn << " unknown" << endl;
            line = to_string(g.turn) + " S - S - S -";
        }
        cout << line << endl;                                // endl 立即刷新
    }
    return 0;
}
