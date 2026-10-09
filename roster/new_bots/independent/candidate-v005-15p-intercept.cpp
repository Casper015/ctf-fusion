// Starter template: parses the game state every turn and keeps all 3 units stationary.
// The parser may be kept as-is; only modify decide() to implement your strategy.
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>

using namespace std;

struct Unit { int id, team, x, y, hp, flag, respawn_at; };      // Dead units have x = y = -1
struct Flag { int id; string status; int x, y, holder, return_at; };
struct Event { string type; vector<int> args; };

struct Game {
    // Initial configuration (INIT block)
    int teams = 0, me = 0, size = 0, turns = 0, hp = 0, damage = 0, respawn = 0, flag_return = 0, flag_cooldown = 0;
    vector<string> map;                  // map[y][x]: '#' wall, '.' empty
    vector<pair<int, int>> bases;        // Base center coordinates (each base is 3x3 centered here)
    vector<pair<int, int>> spots;        // All flag spots
    // Turn state (TURN block)
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;
    vector<Event> events;                // Events from the previous turn
};

// Reads one input block. Continues until the TURN block finishes with END. Returns false on EOF.
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

struct Pos {
    int x = -1, y = -1;
    bool operator==(const Pos& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Pos& o) const { return !(*this == o); }
};

static const int DX[5] = {0, 0, 0, -1, 1};
static const int DY[5] = {0, -1, 1, 0, 0};
static const char* MOVE[5] = {"S", "U", "D", "L", "R"};
static const int INF = 1'000'000;

int cell_id(const Pos& p, int n) { return p.y * n + p.x; }
bool inside(const Game& g, const Pos& p) {
    return p.x >= 0 && p.y >= 0 && p.x < g.size && p.y < g.size;
}
bool open_cell(const Game& g, const Pos& p) {
    return inside(g, p) && g.map[p.y][p.x] != '#';
}
Pos moved(const Pos& p, int dir) { return {p.x + DX[dir], p.y + DY[dir]}; }

vector<int> distances_from(const Game& g, const vector<Pos>& sources) {
    vector<int> d(g.size * g.size, INF);
    queue<Pos> q;
    for (const Pos& p : sources) {
        if (!open_cell(g, p)) continue;
        int id = cell_id(p, g.size);
        if (d[id] == 0) continue;
        d[id] = 0;
        q.push(p);
    }
    while (!q.empty()) {
        Pos p = q.front(); q.pop();
        int nd = d[cell_id(p, g.size)] + 1;
        for (int dir = 1; dir <= 4; ++dir) {
            Pos z = moved(p, dir);
            if (!open_cell(g, z)) continue;
            int id = cell_id(z, g.size);
            if (d[id] <= nd) continue;
            d[id] = nd;
            q.push(z);
        }
    }
    return d;
}

bool in_base(const Pos& p, const pair<int,int>& b) {
    return p.x >= b.first - 1 && p.x <= b.first + 1 &&
           p.y >= b.second - 1 && p.y <= b.second + 1;
}

bool can_attack(const Game& g, const Pos& a, const Pos& b) {
    if (!open_cell(g, a) || !open_cell(g, b)) return false;
    int dx = abs(a.x - b.x), dy = abs(a.y - b.y), md = dx + dy;
    if (md < 1 || md > 2) return false;
    if (md == 1) return true;
    if (dx == 2) return g.map[a.y][(a.x + b.x) / 2] != '#';
    if (dy == 2) return g.map[(a.y + b.y) / 2][a.x] != '#';
    return g.map[a.y][b.x] != '#' || g.map[b.y][a.x] != '#';
}

struct Goal {
    Pos p;
    vector<int> dist;
    int flag_id = -1;
    bool flag_goal = false;
};

// Movement commands are evaluated jointly. This resolves friendly destination
// conflicts and blocked following chains while allowing same-team swaps/cycles.
array<Pos,3> resolve_friendly_moves(const Game& g, const array<Pos,3>& starts,
                                    const array<bool,3>& alive,
                                    const array<int,3>& dirs,
                                    const vector<Pos>& enemy_positions) {
    array<Pos,3> want = starts;
    array<bool,3> moving_now{};
    for (int i = 0; i < 3; ++i) {
        if (!alive[i] || dirs[i] == 0) continue;
        Pos z = moved(starts[i], dirs[i]);
        if (!open_cell(g, z)) continue;
        bool enemy_occupied = false;
        for (const Pos& ep : enemy_positions) if (ep == z) enemy_occupied = true;
        if (enemy_occupied) continue;
        want[i] = z;
        moving_now[i] = (z != starts[i]);
    }
    bool changed = true;
    while (changed) {
        changed = false;
        // All movers entering the same destination bounce.
        for (int i = 0; i < 3; ++i) if (moving_now[i]) {
            for (int j = i + 1; j < 3; ++j) if (moving_now[j] && want[i] == want[j]) {
                moving_now[i] = moving_now[j] = false;
                want[i] = starts[i]; want[j] = starts[j];
                changed = true;
            }
        }
        // A follower cannot enter a cell whose occupant failed to leave.
        for (int i = 0; i < 3; ++i) if (moving_now[i]) {
            for (int j = 0; j < 3; ++j) if (i != j && alive[j] && want[i] == starts[j] && !moving_now[j]) {
                moving_now[i] = false;
                want[i] = starts[i];
                changed = true;
                break;
            }
        }
    }
    for (int i = 0; i < 3; ++i) if (!moving_now[i]) want[i] = starts[i];
    return want;
}

struct AttackSolver {
    const Game& g;
    const array<Pos,3>& final_pos;
    const array<const Unit*,3>& mine;
    const array<bool,3>& pickup;
    vector<const Unit*> enemies;
    vector<array<Pos,5>> forecasts;
    array<vector<int>,3> choices;
    array<int,3> selected{{-1,-1,-1}};
    array<int,3> best{{-1,-1,-1}};
    double best_value = -1.0;

    AttackSolver(const Game& game, const array<Pos,3>& fp,
                 const array<const Unit*,3>& own, const array<bool,3>& wants_pickup)
        : g(game), final_pos(fp), mine(own), pickup(wants_pickup) {
        for (const Unit& u : g.units) {
            if (u.team == g.me || u.hp <= 0 || u.x < 0 || u.y < 0) continue;
            if (u.respawn_at >= 0) continue;
            enemies.push_back(&u);
            array<Pos,5> f{};
            Pos p{u.x,u.y};
            for (int d = 0; d < 5; ++d) {
                Pos z = moved(p, d);
                f[d] = (d == 0 || !open_cell(g,z)) ? p : z;
            }
            forecasts.push_back(f);
        }
        for (int i = 0; i < 3; ++i) {
            if (!mine[i] || mine[i]->x < 0 || mine[i]->flag >= 0 || pickup[i]) continue;
            for (int e = 0; e < (int)enemies.size(); ++e) {
                bool possible = false;
                for (const Pos& ep : forecasts[e]) if (can_attack(g, final_pos[i], ep)) possible = true;
                if (possible) choices[i].push_back(e);
            }
        }
    }

    double target_value(int e, const vector<int>& shooters) const {
        const Unit& target = *enemies[e];
        double threat = target.flag >= 0 ? 2.3 : 1.0;
        for (int k = 0; k < 3; ++k) {
            if (mine[k] && mine[k]->flag >= 0 && final_pos[k].x >= 0 &&
                abs(final_pos[k].x - target.x) + abs(final_pos[k].y - target.y) <= 5)
                threat += 0.25;
        }
        double total = 0.0;
        for (const Pos& ep : forecasts[e]) {
            int hits = 0;
            for (int s : shooters) if (can_attack(g, final_pos[s], ep)) ++hits;
            int dealt = min(target.hp, hits * g.damage);
            double v = 0.14 * dealt * threat;
            if (dealt >= target.hp && hits > 0) v += (42.0 + (target.flag >= 0 ? 48.0 : 0.0)) * threat;
            total += v / 5.0;
        }
        return total;
    }

    double score_leaf() const {
        double value = 0.0;
        for (int e = 0; e < (int)enemies.size(); ++e) {
            vector<int> shooters;
            for (int i = 0; i < 3; ++i) if (selected[i] == e) shooters.push_back(i);
            if (!shooters.empty()) value += target_value(e, shooters);
        }
        return value;
    }

    void search(int i) {
        if (i == 3) {
            double v = score_leaf();
            if (v > best_value) { best_value = v; best = selected; }
            return;
        }
        if (!mine[i] || mine[i]->x < 0 || mine[i]->flag >= 0 || pickup[i]) {
            selected[i] = -1; search(i + 1); return;
        }
        selected[i] = -1; search(i + 1);
        for (int e : choices[i]) { selected[i] = e; search(i + 1); }
        selected[i] = -1;
    }

    array<int,3> solve() { search(0); return best; }
};

// Generates a move and action for each of your 3 units (IDs: 3*me..3*me+2).
string decide(const Game& g) {
    array<const Unit*,3> mine{};
    for (const Unit& u : g.units) {
        if (u.team == g.me && u.id >= 3*g.me && u.id < 3*g.me+3)
            mine[u.id - 3*g.me] = &u;
    }
    array<Pos,3> starts{};
    array<bool,3> alive{};
    array<bool,3> carrier{};
    for (int i = 0; i < 3; ++i) if (mine[i]) {
        starts[i] = {mine[i]->x, mine[i]->y};
        alive[i] = mine[i]->x >= 0 && mine[i]->y >= 0 && mine[i]->hp > 0;
        carrier[i] = mine[i]->flag >= 0;
    }

    vector<Pos> enemies_now;
    for (const Unit& u : g.units)
        if (u.team != g.me && u.hp > 0 && u.x >= 0 && u.y >= 0) enemies_now.push_back({u.x,u.y});

    vector<Pos> home_cells;
    if (g.me >= 0 && g.me < (int)g.bases.size()) {
        auto b = g.bases[g.me];
        for (int y = b.second - 1; y <= b.second + 1; ++y)
            for (int x = b.first - 1; x <= b.first + 1; ++x) {
                Pos p{x,y}; if (open_cell(g,p)) home_cells.push_back(p);
            }
    }
    vector<int> home_dist = distances_from(g, home_cells);

    vector<Goal> goals;
    vector<Pos> active_flag_positions;
    for (const Flag& f : g.flags) {
        if ((f.status != "home" && f.status != "dropped") || f.x < 0 || f.y < 0) continue;
        Pos p{f.x,f.y};
        if (!open_cell(g,p)) continue;
        active_flag_positions.push_back(p);
        vector<int> d = distances_from(g,{p});
        int goal_idx = cell_id(p,g.size);
        if (home_dist[goal_idx] >= INF) continue;
        int pickup_steps = 0;
        bool feasible = false;
        for (int i = 0; i < 3; ++i) if (alive[i] && !carrier[i]) {
            int from = d[cell_id(starts[i],g.size)];
            if (from >= INF) continue;
            int ps = max(0,from-1);
            if (f.status == "dropped" && g.turn + ps >= f.return_at) continue;
            int eta = ps + home_dist[goal_idx];
            if (g.turn + eta <= g.turns) { feasible = true; pickup_steps = min(pickup_steps, ps); }
        }
        if (!feasible) continue;
        goals.push_back({p,std::move(d),f.id,true});
    }

    // Staging goals let free units wait near likely respawn spots when flags are
    // temporarily carried or cooling down. Active ground flags take precedence.
    for (const auto& s : g.spots) {
        Pos p{s.first,s.second};
        if (!open_cell(g,p)) continue;
        bool active_here = false;
        for (const Pos& q : active_flag_positions) if (p == q) active_here = true;
        if (active_here) continue;
        bool duplicate = false;
        for (const Goal& goal : goals) if (!goal.flag_goal && goal.p == p) duplicate = true;
        if (duplicate) continue;
        goals.push_back({p,distances_from(g,{p}),-1,false});
    }

    array<int,3> assigned{{-1,-1,-1}};
    vector<int> free_units;
    for (int i = 0; i < 3; ++i) if (alive[i] && !carrier[i]) free_units.push_back(i);
    vector<vector<double>> task_cost(free_units.size(), vector<double>(goals.size(), 1e9));
    for (int fi = 0; fi < (int)free_units.size(); ++fi) {
        int unit_i = free_units[fi];
        for (int gi = 0; gi < (int)goals.size(); ++gi) {
            const Goal& goal = goals[gi];
            int d = goal.dist[cell_id(starts[unit_i],g.size)];
            if (d >= INF) continue;
            if (goal.flag_goal) {
                int home_d = home_dist[cell_id(goal.p,g.size)];
                int pickup_steps = max(0,d-1);
                int eta = pickup_steps + home_d;
                if (g.turn + eta > g.turns) continue;
                bool dropped_expiring = false;
                for (const Flag& f : g.flags) if (f.id == goal.flag_id && f.status == "dropped")
                    dropped_expiring = g.turn + pickup_steps >= f.return_at;
                if (dropped_expiring) continue;
                double cost = 0.35 * pickup_steps + 0.65 * home_d - 48.0;
                int enemy_eta = INF;
                for (const Unit& e : g.units) if (e.team != g.me && e.hp > 0 && e.x >= 0 && e.y >= 0) {
                    int ed = goal.dist[cell_id({e.x,e.y},g.size)];
                    if (ed < INF) enemy_eta = min(enemy_eta,max(0,ed-1));
                }
                if (enemy_eta <= pickup_steps) cost += min(35.0, 4.0 * (pickup_steps - enemy_eta + 1));
                task_cost[fi][gi] = cost;
            } else {
                task_cost[fi][gi] = 12.0 + 0.35 * d;
            }
        }
    }
    vector<bool> used(goals.size(),false);
    double best_assignment_cost = numeric_limits<double>::infinity();
    array<int,3> current_assignment{{-1,-1,-1}};
    function<void(int,double)> assign_rec = [&](int index, double cost) {
        if (index == (int)free_units.size()) {
            if (cost < best_assignment_cost) {
                best_assignment_cost = cost;
                assigned = current_assignment;
            }
            return;
        }
        int ui = free_units[index];
        current_assignment[ui] = -1;
        assign_rec(index+1,cost+45.0);
        for (int gi = 0; gi < (int)goals.size(); ++gi) if (!used[gi] && task_cost[index][gi] < 1e8) {
            used[gi] = true; current_assignment[ui] = gi;
            assign_rec(index+1,cost+task_cost[index][gi]);
            used[gi] = false;
        }
        current_assignment[ui] = -1;
    };
    assign_rec(0,0.0);

    array<array<int,5>,3> move_options{};
    array<int,3> option_count{{1,1,1}};
    for (int i = 0; i < 3; ++i) {
        move_options[i][0] = 0;
        if (!alive[i]) continue;
        bool in_home_with_flag = carrier[i] && g.me < (int)g.bases.size() && in_base(starts[i],g.bases[g.me]);
        if (in_home_with_flag) {
            option_count[i] = 1;
            for (int d = 1; d <= 4; ++d) {
                Pos z = moved(starts[i],d);
                if (open_cell(g,z) && in_base(z,g.bases[g.me]))
                    move_options[i][option_count[i]++] = d;
            }
        } else {
            option_count[i] = 5;
            for (int d = 0; d < 5; ++d) move_options[i][d] = d;
        }
    }

    array<int,3> best_dirs{{0,0,0}};
    array<Pos,3> best_positions = starts;
    double best_move_score = -numeric_limits<double>::infinity();
    for (int a = 0; a < option_count[0]; ++a)
    for (int b = 0; b < option_count[1]; ++b)
    for (int c = 0; c < option_count[2]; ++c) {
        array<int,3> dirs{{move_options[0][a],move_options[1][b],move_options[2][c]}};
        array<Pos,3> pos = resolve_friendly_moves(g,starts,alive,dirs,enemies_now);
        double score = 0.0;
        for (int i = 0; i < 3; ++i) if (alive[i]) {
            const vector<int>* field = nullptr;
            if (carrier[i]) field = &home_dist;
            else if (assigned[i] >= 0 && assigned[i] < (int)goals.size()) field = &goals[assigned[i]].dist;
            if (field) {
                int before = (*field)[cell_id(starts[i],g.size)];
                int after = (*field)[cell_id(pos[i],g.size)];
                if (before < INF && after < INF) score += (carrier[i] ? 16.0 : 12.0) * (before-after);
            }
            // Prefer routes that reduce the chance of incoming fire, especially
            // for a carrier. The forecast covers all five enemy move choices.
            int danger = 0;
            for (const Unit& e : g.units) if (e.team != g.me && e.hp > 0 && e.x >= 0 && e.y >= 0 &&
                                                e.respawn_at < 0 && (g.teams != 15 || e.flag < 0)) {
                Pos ep{e.x,e.y};
                int covered = 0;
                for (int ed = 0; ed < 5; ++ed) {
                    Pos q = moved(ep,ed);
                    if (ed == 0 || !open_cell(g,q)) q = ep;
                    if (can_attack(g,q,pos[i])) ++covered;
                }
                danger += covered;
            }
            double hp_factor = (mine[i] && mine[i]->hp <= 68) ? 1.35 : 1.0;
            score -= danger * (carrier[i] ? 0.095 : 0.035) * hp_factor;
        }
        if (g.teams == 15) {
            // In the densest fields, favor positions that let allies volley on
            // enemy carriers. Each volley is evaluated against one shared
            // post-move enemy position, and lethal damage is credited only in
            // outcomes where enough shots connect.
            for (const Unit& e : g.units) if (e.team != g.me && e.hp > 0 && e.flag >= 0 &&
                                                e.x >= 0 && e.y >= 0 && e.respawn_at < 0) {
                Pos ep{e.x,e.y};
                int total_hits = 0, lethal_outcomes = 0;
                for (int ed = 0; ed < 5; ++ed) {
                    Pos q = moved(ep,ed);
                    if (ed == 0 || !open_cell(g,q)) q = ep;
                    int hits = 0;
                    for (int i = 0; i < 3; ++i) if (alive[i] && !carrier[i]) {
                        bool pickup_here = false;
                        for (const Flag& f : g.flags)
                            if ((f.status == "home" || f.status == "dropped") &&
                                f.x == pos[i].x && f.y == pos[i].y) pickup_here = true;
                        if (!pickup_here && can_attack(g,pos[i],q)) ++hits;
                    }
                    total_hits += hits;
                    if (hits * g.damage >= e.hp) ++lethal_outcomes;
                }
                score += 3.0 * (total_hits / 5.0) + 15.0 * (lethal_outcomes / 5.0);
            }
        }
        if (score > best_move_score) {
            best_move_score = score;
            best_dirs = dirs;
            best_positions = pos;
        }
    }

    array<bool,3> pickup{};
    array<int,3> pickup_flag{{-1,-1,-1}};
    for (int i = 0; i < 3; ++i) if (alive[i] && !carrier[i]) {
        for (const Flag& f : g.flags) {
            if ((f.status == "home" || f.status == "dropped") && f.x == best_positions[i].x && f.y == best_positions[i].y) {
                if (pickup_flag[i] < 0 || f.id < pickup_flag[i]) pickup_flag[i] = f.id;
            }
        }
        pickup[i] = pickup_flag[i] >= 0;
    }
    AttackSolver attack_solver(g,best_positions,mine,pickup);
    array<int,3> attacks = attack_solver.solve();

    string out = to_string(g.turn);
    for (int i = 0; i < 3; ++i) {
        out += " "; out += MOVE[best_dirs[i]]; out += " ";
        if (!mine[i] || !alive[i] || carrier[i]) out += "-";
        else if (pickup[i]) out += "P";
        else if (attacks[i] >= 0 && attacks[i] < (int)attack_solver.enemies.size())
            out += to_string(attack_solver.enemies[attacks[i]]->id);
        else out += "-";
    }
    return out;
}

int main() {
    ios::sync_with_stdio(false);
    Game g;
    while (read_turn(g)) {
        cout << decide(g) << endl;  // endl flushes stdout immediately
    }
    return 0;
}
