#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <array>
#include <cstdint>
#include <cstring>

using namespace std;

const uint16_t INF_DIST = 9999;
const int MAX_CELLS = 2500;
const int MAX_TEAMS = 16;

const int DX[5] = {0, 0, -1, 1, 0};
const int DY[5] = {-1, 1, 0, 0, 0};
const char MOVE_CHAR[5] = {'U', 'D', 'L', 'R', 'S'};

struct Point {
    int x, y;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
    bool operator!=(const Point& o) const { return x != o.x || y != o.y; }
};

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

// Global precomputations
static bool precomputed = false;
static uint16_t dist_matrix[MAX_CELLS][MAX_CELLS];
static uint8_t can_hit[MAX_CELLS][MAX_CELLS];
static uint16_t dist_to_base[MAX_TEAMS][MAX_CELLS];
static vector<Point> base_cells[MAX_TEAMS];
static vector<Point> spawn_order[MAX_TEAMS];
static bool is_base_cell[MAX_TEAMS][MAX_CELLS];
static vector<int> attack_cells[MAX_CELLS];

inline int cell_id(int x, int y, int size) {
    return y * size + x;
}

inline Point id_to_point(int id, int size) {
    return {id % size, id / size};
}

bool check_los_attack(int x1, int y1, int x2, int y2, const vector<string>& map, int size) {
    int d = abs(x1 - x2) + abs(y1 - y2);
    if (d > 2) return false;
    int dx = x2 - x1, dy = y2 - y1;
    if (abs(dx) == 2 && dy == 0) {
        return map[y1][x1 + dx / 2] != '#';
    }
    if (abs(dy) == 2 && dx == 0) {
        return map[y1 + dy / 2][x1] != '#';
    }
    if (abs(dx) == 1 && abs(dy) == 1) {
        return map[y1][x1 + dx] != '#' || map[y1 + dy][x1] != '#';
    }
    return true;
}

void precompute(const Game& g) {
    int S = g.size;
    Point center = {S / 2, S / 2};

    // 1. Initialize dist_matrix
    for (int i = 0; i < S * S; i++) {
        for (int j = 0; j < S * S; j++) {
            dist_matrix[i][j] = INF_DIST;
            can_hit[i][j] = 0;
        }
        attack_cells[i].clear();
    }

    // 2. Base cells & spawn order for all teams
    for (int t = 0; t < g.teams; t++) {
        base_cells[t].clear();
        spawn_order[t].clear();
        for (int i = 0; i < S * S; i++) is_base_cell[t][i] = false;

        Point bc = {g.bases[t].first, g.bases[t].second};
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int bx = bc.x + dx, by = bc.y + dy;
                if (bx >= 0 && bx < S && by >= 0 && by < S) {
                    base_cells[t].push_back({bx, by});
                    is_base_cell[t][cell_id(bx, by, S)] = true;
                }
            }
        }

        spawn_order[t] = base_cells[t];
        sort(spawn_order[t].begin(), spawn_order[t].end(), [&](const Point& p1, const Point& p2) {
            int d1 = abs(p1.x - center.x) + abs(p1.y - center.y);
            int d2 = abs(p2.x - center.x) + abs(p2.y - center.y);
            if (d1 != d2) return d1 < d2;
            int t1 = abs(p1.x - bc.x) + 2 * abs(p1.y - bc.y);
            int t2 = abs(p2.x - bc.x) + 2 * abs(p2.y - bc.y);
            return t1 < t2;
        });
    }

    // 3. BFS all-pairs
    int q[MAX_CELLS];
    for (int sy = 0; sy < S; sy++) {
        for (int sx = 0; sx < S; sx++) {
            if (g.map[sy][sx] == '#') continue;
            int src = cell_id(sx, sy, S);
            dist_matrix[src][src] = 0;
            int head = 0, tail = 0;
            q[tail++] = src;

            while (head < tail) {
                int cur = q[head++];
                int cx = cur % S, cy = cur / S;
                int cd = dist_matrix[src][cur];

                for (int dir = 0; dir < 4; dir++) {
                    int nx = cx + DX[dir], ny = cy + DY[dir];
                    if (nx >= 0 && nx < S && ny >= 0 && ny < S && g.map[ny][nx] != '#') {
                        int nxt = cell_id(nx, ny, S);
                        if (dist_matrix[src][nxt] > cd + 1) {
                            dist_matrix[src][nxt] = cd + 1;
                            q[tail++] = nxt;
                        }
                    }
                }
            }
        }
    }

    // 4. Attack LOS table
    for (int y1 = 0; y1 < S; y1++) {
        for (int x1 = 0; x1 < S; x1++) {
            if (g.map[y1][x1] == '#') continue;
            int u = cell_id(x1, y1, S);
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    if (abs(dx) + abs(dy) > 2) continue;
                    int x2 = x1 + dx, y2 = y1 + dy;
                    if (x2 >= 0 && x2 < S && y2 >= 0 && y2 < S && g.map[y2][x2] != '#') {
                        if (check_los_attack(x1, y1, x2, y2, g.map, S)) {
                            int v = cell_id(x2, y2, S);
                            can_hit[u][v] = 1;
                            attack_cells[u].push_back(v);
                        }
                    }
                }
            }
        }
    }

    // 5. Distance to base for each team
    for (int t = 0; t < g.teams; t++) {
        for (int i = 0; i < S * S; i++) {
            uint16_t min_d = INF_DIST;
            for (const auto& b : base_cells[t]) {
                int bid = cell_id(b.x, b.y, S);
                if (dist_matrix[i][bid] < min_d) {
                    min_d = dist_matrix[i][bid];
                }
            }
            dist_to_base[t][i] = min_d;
        }
    }
}

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

struct UnitState {
    int id;
    int team;
    Point pos;
    int hp;
    int flag;
    bool is_respawning;
    bool is_alive;
};

string decide(const Game& g) {
    if (!precomputed) {
        precompute(g);
        precomputed = true;
    }

    int S = g.size;
    Point center = {S / 2, S / 2};

    // 1. Simulate respawns
    bool occupied[MAX_CELLS] = {0};
    for (const auto& u : g.units) {
        if (u.x != -1) occupied[cell_id(u.x, u.y, S)] = true;
    }

    vector<UnitState> all_units(g.units.size());
    for (const auto& u : g.units) {
        UnitState us;
        us.id = u.id;
        us.team = u.team;
        us.hp = u.hp;
        us.flag = u.flag;
        us.is_respawning = false;
        us.is_alive = false;

        if (u.x != -1) {
            us.pos = {u.x, u.y};
            us.is_alive = true;
        } else if (u.respawn_at != -1 && g.turn >= u.respawn_at) {
            for (const auto& p : spawn_order[u.team]) {
                int pid = cell_id(p.x, p.y, S);
                if (!occupied[pid]) {
                    us.pos = p;
                    occupied[pid] = true;
                    us.is_alive = true;
                    us.is_respawning = true;
                    us.hp = g.hp;
                    break;
                }
            }
        }
        all_units[u.id] = us;
    }

    UnitState my_units[3];
    for (int k = 0; k < 3; k++) {
        my_units[k] = all_units[3 * g.me + k];
    }

    vector<UnitState> enemies;
    for (const auto& u : all_units) {
        if (u.team != g.me && u.is_alive) {
            enemies.push_back(u);
        }
    }

    // 2. Identify Flags & Enemy Carriers
    vector<Flag> open_flags;
    for (const auto& f : g.flags) {
        if (f.status == "home" || f.status == "dropped") {
            open_flags.push_back(f);
        }
    }

    vector<int> enemy_carriers;
    for (const auto& e : enemies) {
        if (e.flag != -1) enemy_carriers.push_back(e.id);
    }

    // 3. Enemy Threat Map
    int enemy_threat_count[MAX_CELLS] = {0};
    bool marked[MAX_CELLS];
    for (const auto& e : enemies) {
        if (!e.is_alive || e.flag != -1 || e.is_respawning) continue;
        memset(marked, 0, sizeof(bool) * (S * S));
        for (int dir = 0; dir < 5; dir++) {
            int enx = e.pos.x + DX[dir], eny = e.pos.y + DY[dir];
            if (enx < 0 || enx >= S || eny < 0 || eny >= S || g.map[eny][enx] == '#') continue;
            int encid = cell_id(enx, eny, S);
            for (int v : attack_cells[encid]) {
                if (!marked[v]) {
                    marked[v] = true;
                    enemy_threat_count[v]++;
                }
            }
        }
    }

    // 4. Macro Roles & Targets
    Point target[3];
    int role[3]; // 0: CARRY, 1: FETCH, 2: INTERCEPT, 3: COMBAT/ESCORT, 4: PATROL
    for (int i = 0; i < 3; i++) {
        role[i] = -1;
        target[i] = {-1, -1};
    }

    // Assign carriers
    int my_carrier_idx = -1;
    for (int i = 0; i < 3; i++) {
        if (my_units[i].is_alive && my_units[i].flag != -1) {
            role[i] = 0; // CARRY
            my_carrier_idx = i;
            Point best_b = base_cells[g.me][0];
            int my_cid = cell_id(my_units[i].pos.x, my_units[i].pos.y, S);
            int min_db = INF_DIST;
            for (const auto& b : base_cells[g.me]) {
                int bid = cell_id(b.x, b.y, S);
                if (dist_matrix[my_cid][bid] < min_db) {
                    min_db = dist_matrix[my_cid][bid];
                    best_b = b;
                }
            }
            target[i] = best_b;
        }
    }

    vector<int> unassigned;
    for (int i = 0; i < 3; i++) {
        if (my_units[i].is_alive && role[i] == -1) {
            unassigned.push_back(i);
        }
    }

    vector<bool> flag_taken(open_flags.size(), false);
    vector<bool> carrier_intercepted(enemy_carriers.size(), false);

    vector<Point> patrol_spots;
    patrol_spots.push_back({g.spots[g.me].first, g.spots[g.me].second});
    patrol_spots.push_back(center);
    Point mid_spot = {(g.bases[g.me].first + center.x) / 2, (g.bases[g.me].second + center.y) / 2};
    patrol_spots.push_back(mid_spot);

    int patrol_idx = 0;

    for (int u_idx : unassigned) {
        Point u_pos = my_units[u_idx].pos;
        int u_cid = cell_id(u_pos.x, u_pos.y, S);

        // A. Intercept nearby enemy carrier
        int best_ec = -1;
        Point best_intercept_pt = {-1, -1};
        int best_ec_dist = 999;

        for (size_t ec_i = 0; ec_i < enemy_carriers.size(); ec_i++) {
            if (carrier_intercepted[ec_i]) continue;
            int ec_id = enemy_carriers[ec_i];
            const auto& ec = all_units[ec_id];
            int ec_cid = cell_id(ec.pos.x, ec.pos.y, S);

            int cur_c = ec_cid;
            int step = 0;
            while (cur_c != -1 && step < 20) {
                int dist_to_pt = dist_matrix[u_cid][cur_c];
                if (dist_to_pt - 1 <= step) {
                    if (dist_to_pt < best_ec_dist) {
                        best_ec_dist = dist_to_pt;
                        best_ec = ec_i;
                        best_intercept_pt = id_to_point(cur_c, S);
                    }
                    break;
                }
                int next_c = -1;
                int min_db = dist_to_base[ec.team][cur_c];
                Point cp = id_to_point(cur_c, S);
                for (int dir = 0; dir < 4; dir++) {
                    int nx = cp.x + DX[dir], ny = cp.y + DY[dir];
                    if (nx >= 0 && nx < S && ny >= 0 && ny < S && g.map[ny][nx] != '#') {
                        int nid = cell_id(nx, ny, S);
                        if (dist_to_base[ec.team][nid] < min_db) {
                            min_db = dist_to_base[ec.team][nid];
                            next_c = nid;
                        }
                    }
                }
                if (next_c == -1 || next_c == cur_c) break;
                cur_c = next_c;
                step++;
            }
        }

        if (best_ec != -1 && best_ec_dist <= 12) {
            role[u_idx] = 2; // INTERCEPT
            target[u_idx] = best_intercept_pt;
            carrier_intercepted[best_ec] = true;
            continue;
        }

        // B. Fetch best open flag
        int best_f = -1;
        double best_f_score = -1e9;
        for (size_t f_i = 0; f_i < open_flags.size(); f_i++) {
            if (flag_taken[f_i]) continue;
            const auto& f = open_flags[f_i];
            int fid = cell_id(f.x, f.y, S);
            int d_u = dist_matrix[u_cid][fid];
            if (d_u >= INF_DIST) continue;

            if (f.status == "dropped" && f.return_at != -1) {
                if (f.return_at - g.turn <= d_u) continue;
            }

            int d_base = dist_to_base[g.me][fid];
            int min_e = 999;
            for (const auto& e : enemies) {
                if (!e.is_alive) continue;
                int ed = dist_matrix[cell_id(e.pos.x, e.pos.y, S)][fid];
                if (ed < min_e) min_e = ed;
            }

            double score = 100.0 / (d_u + d_base * 0.5 + 2.0);
            if (d_u <= min_e) score += 40.0;
            else if (d_u > min_e + 2) score -= 20.0;

            if (score > best_f_score) {
                best_f_score = score;
                best_f = f_i;
            }
        }

        if (best_f != -1 && best_f_score > 0) {
            role[u_idx] = 1; // FETCH
            target[u_idx] = {open_flags[best_f].x, open_flags[best_f].y};
            flag_taken[best_f] = true;
            continue;
        }

        // C. Escort / Combat
        if (my_carrier_idx != -1) {
            Point c_pos = my_units[my_carrier_idx].pos;
            int closest_e = -1;
            int min_ed = 999;
            for (const auto& e : enemies) {
                if (!e.is_alive) continue;
                int d = dist_matrix[cell_id(e.pos.x, e.pos.y, S)][cell_id(c_pos.x, c_pos.y, S)];
                if (d < min_ed && d <= 5) {
                    min_ed = d;
                    closest_e = e.id;
                }
            }
            if (closest_e != -1) {
                role[u_idx] = 3;
                target[u_idx] = all_units[closest_e].pos;
                continue;
            }
        }

        // D. Nearby enemies to hunt
        int nearest_enemy = -1;
        int nearest_e_dist = 999;
        for (const auto& e : enemies) {
            if (!e.is_alive || e.is_respawning) continue;
            int d = dist_matrix[u_cid][cell_id(e.pos.x, e.pos.y, S)];
            if (d < nearest_e_dist && d <= 3) {
                nearest_e_dist = d;
                nearest_enemy = e.id;
            }
        }

        if (nearest_enemy != -1 && my_units[u_idx].hp >= 66) {
            role[u_idx] = 3;
            target[u_idx] = all_units[nearest_enemy].pos;
            continue;
        }

        // E. Patrol distinct spots
        role[u_idx] = 4;
        if (patrol_idx < (int)patrol_spots.size()) {
            target[u_idx] = patrol_spots[patrol_idx++];
        } else {
            target[u_idx] = center;
        }
    }

    // 5. Select Move for each unit
    int chosen_moves[3] = {4, 4, 4};
    Point chosen_dest[3];

    for (int i = 0; i < 3; i++) {
        if (!my_units[i].is_alive) {
            chosen_dest[i] = {-1, -1};
            continue;
        }

        Point my_pos = my_units[i].pos;
        int my_cid = cell_id(my_pos.x, my_pos.y, S);
        Point tgt = target[i];
        int tgt_cid = cell_id(tgt.x, tgt.y, S);

        struct MoveOpt {
            int dir;
            double score;
        };
        vector<MoveOpt> opts;

        for (int dir = 0; dir < 5; dir++) {
            int nx = my_pos.x + DX[dir], ny = my_pos.y + DY[dir];
            if (nx < 0 || nx >= S || ny < 0 || ny >= S || g.map[ny][nx] == '#') continue;
            int ncid = cell_id(nx, ny, S);

            double sc = 0;
            int threats = enemy_threat_count[ncid];
            int dmg = threats * g.damage;

            if (role[i] == 0) { // CARRY
                int d_base = dist_to_base[g.me][ncid];
                sc = -d_base * 100.0;
                if (is_base_cell[g.me][ncid]) {
                    if (my_units[i].hp > dmg) sc += 100000.0; // SAFE CAPTURE!
                    else sc -= 20000.0;
                } else {
                    if (dmg >= my_units[i].hp) sc -= 25000.0; // lethal ambush
                    else sc -= threats * 20.0;
                }
            } else { // Non-carrier
                int d_tgt = dist_matrix[ncid][tgt_cid];
                sc = -d_tgt * 20.0;
                if (dir == 4 && d_tgt > 0) sc -= 5.0;

                // Attack opportunity bonus
                for (int v : attack_cells[ncid]) {
                    for (const auto& e : enemies) {
                        if (e.is_alive && !e.is_respawning && cell_id(e.pos.x, e.pos.y, S) == v) {
                            sc += 15.0;
                            if (e.hp <= g.damage) sc += 30.0;
                            break;
                        }
                    }
                }

                if (dmg >= my_units[i].hp && my_units[i].hp <= g.damage) {
                    sc -= 800.0;
                }
            }

            opts.push_back({dir, sc});
        }

        sort(opts.begin(), opts.end(), [](const MoveOpt& a, const MoveOpt& b) {
            return a.score > b.score;
        });

        if (!opts.empty()) {
            chosen_moves[i] = opts[0].dir;
            chosen_dest[i] = {my_pos.x + DX[opts[0].dir], my_pos.y + DY[opts[0].dir]};
        } else {
            chosen_moves[i] = 4;
            chosen_dest[i] = my_pos;
        }
    }

    // 6. Friendly Conflict Resolution
    int p_order[3] = {0, 1, 2};
    sort(p_order, p_order + 3, [&](int a, int b) {
        if (!my_units[a].is_alive) return false;
        if (!my_units[b].is_alive) return true;
        if (role[a] != role[b]) return role[a] < role[b];
        return a < b;
    });

    for (int p1 = 0; p1 < 3; p1++) {
        int u1 = p_order[p1];
        if (!my_units[u1].is_alive) continue;
        for (int p2 = p1 + 1; p2 < 3; p2++) {
            int u2 = p_order[p2];
            if (!my_units[u2].is_alive) continue;

            if (chosen_dest[u1] == chosen_dest[u2]) {
                Point u2_pos = my_units[u2].pos;
                int tgt_cid = cell_id(target[u2].x, target[u2].y, S);
                int best_alt = 4;
                int min_d = 9999;

                for (int dir = 0; dir < 5; dir++) {
                    int nx = u2_pos.x + DX[dir], ny = u2_pos.y + DY[dir];
                    if (nx < 0 || nx >= S || ny < 0 || ny >= S || g.map[ny][nx] == '#') continue;
                    Point np = {nx, ny};
                    if (np == chosen_dest[u1]) continue;
                    int d = dist_matrix[cell_id(nx, ny, S)][tgt_cid];
                    if (d < min_d) {
                        min_d = d;
                        best_alt = dir;
                    }
                }
                chosen_moves[u2] = best_alt;
                chosen_dest[u2] = {u2_pos.x + DX[best_alt], u2_pos.y + DY[best_alt]};
            }
        }
    }

    // 7. Actions: Pickup vs Attack
    string acts[3] = {"-", "-", "-"};

    vector<int> attackers;
    for (int i = 0; i < 3; i++) {
        if (!my_units[i].is_alive) continue;
        if (role[i] == 0) continue; // carrier

        bool on_flag = false;
        for (const auto& f : open_flags) {
            if (f.x == chosen_dest[i].x && f.y == chosen_dest[i].y) {
                on_flag = true;
                break;
            }
        }

        if (on_flag) {
            acts[i] = "P";
            continue;
        }

        if (!my_units[i].is_respawning) {
            attackers.push_back(i);
        }
    }

    if (!attackers.empty()) {
        vector<vector<int>> reachable(attackers.size());
        for (size_t a_i = 0; a_i < attackers.size(); a_i++) {
            int uid = attackers[a_i];
            int u_cid = cell_id(chosen_dest[uid].x, chosen_dest[uid].y, S);
            for (const auto& e : enemies) {
                if (!e.is_alive || e.is_respawning) continue;
                int e_cid = cell_id(e.pos.x, e.pos.y, S);
                if (can_hit[u_cid][e_cid]) {
                    reachable[a_i].push_back(e.id);
                }
            }
        }

        int hits_possible[MAX_TEAMS * 3] = {0};
        for (const auto& list : reachable) {
            for (int eid : list) hits_possible[eid]++;
        }

        vector<bool> attacker_assigned(attackers.size(), false);

        // Priority 1: Killable enemies this turn
        for (const auto& e : enemies) {
            if (!e.is_alive || e.is_respawning) continue;
            int req_hits = (e.hp + g.damage - 1) / g.damage;
            if (hits_possible[e.id] >= req_hits && req_hits > 0) {
                int assigned_count = 0;
                for (size_t a_i = 0; a_i < attackers.size(); a_i++) {
                    if (attacker_assigned[a_i]) continue;
                    if (find(reachable[a_i].begin(), reachable[a_i].end(), e.id) != reachable[a_i].end()) {
                        acts[attackers[a_i]] = to_string(e.id);
                        attacker_assigned[a_i] = true;
                        assigned_count++;
                        if (assigned_count == req_hits) break;
                    }
                }
            }
        }

        // Priority 2: Enemy carriers
        for (size_t a_i = 0; a_i < attackers.size(); a_i++) {
            if (attacker_assigned[a_i]) continue;
            for (int eid : reachable[a_i]) {
                if (all_units[eid].flag != -1) {
                    acts[attackers[a_i]] = to_string(eid);
                    attacker_assigned[a_i] = true;
                    break;
                }
            }
        }

        // Priority 3: Lowest HP enemy reachable
        for (size_t a_i = 0; a_i < attackers.size(); a_i++) {
            if (attacker_assigned[a_i]) continue;
            int best_target = -1;
            int min_hp = 999;
            for (int eid : reachable[a_i]) {
                if (all_units[eid].hp < min_hp) {
                    min_hp = all_units[eid].hp;
                    best_target = eid;
                }
            }
            if (best_target != -1) {
                acts[attackers[a_i]] = to_string(best_target);
                attacker_assigned[a_i] = true;
            }
        }
    }

    string out = to_string(g.turn);
    for (int i = 0; i < 3; i++) {
        out += " ";
        out += MOVE_CHAR[chosen_moves[i]];
        out += " ";
        out += acts[i];
    }
    return out;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Game g;
    while (read_turn(g)) {
        cout << decide(g) << "\n" << flush;
    }
    return 0;
}
