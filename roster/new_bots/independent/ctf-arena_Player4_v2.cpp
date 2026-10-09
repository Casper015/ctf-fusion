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

// Simulate movement for 3 friendly units given candidate moves
// Handles friendly swaps and stops on conflicts
void simulate_friendly_moves(const Point start[3], const bool alive[3], const int moves[3], Point out_pos[3], bool out_blocked[3], int S, const vector<string>& map) {
    Point want[3];
    for (int i = 0; i < 3; i++) {
        out_blocked[i] = false;
        if (!alive[i]) {
            want[i] = {-1, -1};
            out_pos[i] = {-1, -1};
            continue;
        }
        int nx = start[i].x + DX[moves[i]], ny = start[i].y + DY[moves[i]];
        if (nx < 0 || nx >= S || ny < 0 || ny >= S || map[ny][nx] == '#') {
            want[i] = start[i];
        } else {
            want[i] = {nx, ny};
        }
    }

    // Step A: Destination collisions (multiple friendly units want the exact same cell)
    for (int i = 0; i < 3; i++) {
        if (!alive[i]) continue;
        for (int j = i + 1; j < 3; j++) {
            if (!alive[j]) continue;
            if (want[i] == want[j]) {
                out_blocked[i] = true;
                out_blocked[j] = true;
            }
        }
    }

    // Step B: Cascade blocks
    // If unit i wants unit j's start cell, but unit j is blocked or stayed
    // Note: friendly swap (want[i] == start[j] && want[j] == start[i]) is ALLOWED!
    while (true) {
        bool changed = false;
        for (int i = 0; i < 3; i++) {
            if (!alive[i] || out_blocked[i] || want[i] == start[i]) continue;
            for (int j = 0; j < 3; j++) {
                if (i == j || !alive[j]) continue;
                if (want[i] == start[j]) {
                    // Check if it's a mutual swap
                    bool mutual_swap = (want[j] == start[i] && !out_blocked[j]);
                    if (!mutual_swap) {
                        // If j is blocked or j didn't move, i is blocked
                        if (out_blocked[j] || want[j] == start[j]) {
                            out_blocked[i] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        if (!changed) break;
    }

    for (int i = 0; i < 3; i++) {
        if (!alive[i]) out_pos[i] = {-1, -1};
        else out_pos[i] = out_blocked[i] ? start[i] : want[i];
    }
}

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
            // Respawning this turn
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

    // 2. Compute Enemy Threat Map
    // Count distinct enemies that can attack cell c next turn
    int enemy_threat_count[MAX_CELLS] = {0};
    bool marked[MAX_CELLS];
    for (const auto& e : enemies) {
        if (!e.is_alive || e.flag != -1 || e.is_respawning) continue;
        memset(marked, 0, sizeof(bool) * (S * S));
        // Enemy can move to any of 5 positions
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

    // 3. Flags and Enemy Carriers
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

    // 4. Determine Macro Targets for our 3 units
    Point target[3];
    int role[3]; // 0: CARRY, 1: FETCH, 2: INTERCEPT, 3: ESCORT, 4: GUARD
    int assigned_flag[3] = {-1, -1, -1};

    // First assign carriers
    for (int i = 0; i < 3; i++) {
        if (!my_units[i].is_alive) {
            target[i] = {-1, -1};
            role[i] = -1;
            continue;
        }
        if (my_units[i].flag != -1) {
            role[i] = 0; // CARRY
            // Target the base cell that is safest / shortest
            Point best_b = base_cells[g.me][0];
            int my_cid = cell_id(my_units[i].pos.x, my_units[i].pos.y, S);
            int min_cost = 99999;
            for (const auto& b : base_cells[g.me]) {
                int bcid = cell_id(b.x, b.y, S);
                int cost = dist_matrix[my_cid][bcid] * 10 + enemy_threat_count[bcid] * 50;
                if (cost < min_cost) {
                    min_cost = cost;
                    best_b = b;
                }
            }
            target[i] = best_b;
        }
    }

    // Assign non-carriers to (FETCH / INTERCEPT / ESCORT / GUARD)
    vector<int> unassigned;
    for (int i = 0; i < 3; i++) {
        if (my_units[i].is_alive && my_units[i].flag == -1) {
            unassigned.push_back(i);
        }
    }

    // Check if we have a carrier to escort
    int my_carrier_idx = -1;
    for (int i = 0; i < 3; i++) {
        if (my_units[i].is_alive && my_units[i].flag != -1) {
            my_carrier_idx = i;
            break;
        }
    }

    // Greedily match unassigned units to best available targets
    vector<bool> flag_taken(open_flags.size(), false);
    for (int u_idx : unassigned) {
        Point u_pos = my_units[u_idx].pos;
        int u_cid = cell_id(u_pos.x, u_pos.y, S);

        // Option 1: Intercept enemy carrier if close
        int best_ec = -1;
        int best_ec_dist = 999;
        for (int ec_id : enemy_carriers) {
            const auto& ec = all_units[ec_id];
            int ec_cid = cell_id(ec.pos.x, ec.pos.y, S);
            int d = dist_matrix[u_cid][ec_cid];
            if (d < best_ec_dist && d <= 12) {
                best_ec_dist = d;
                best_ec = ec_id;
            }
        }

        if (best_ec != -1 && best_ec_dist <= 8) {
            role[u_idx] = 2; // INTERCEPT
            target[u_idx] = all_units[best_ec].pos;
            continue;
        }

        // Option 2: Escort friendly carrier
        if (my_carrier_idx != -1 && unassigned.size() >= 2 && u_idx == unassigned.back()) {
            role[u_idx] = 3; // ESCORT
            target[u_idx] = target[my_carrier_idx];
            continue;
        }

        // Option 3: Fetch best available flag
        int best_f = -1;
        double best_f_score = -1e9;
        for (size_t f_idx = 0; f_idx < open_flags.size(); f_idx++) {
            if (flag_taken[f_idx]) continue;
            const auto& f = open_flags[f_idx];
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

            double score = 100.0 / (d_u + d_base * 0.7 + 3.0);
            if (d_u <= min_e) score += 35.0;
            else if (d_u > min_e + 2) score -= 25.0;
            score -= d_base * 0.4;

            if (score > best_f_score) {
                best_f_score = score;
                best_f = f_idx;
            }
        }

        if (best_f != -1 && best_f_score > 0) {
            role[u_idx] = 1; // FETCH
            target[u_idx] = {open_flags[best_f].x, open_flags[best_f].y};
            assigned_flag[u_idx] = open_flags[best_f].id;
            flag_taken[best_f] = true;
            continue;
        }

        // Option 4: Guard / Patrol front spot
        role[u_idx] = 4; // GUARD
        Point front_spot = {g.spots[g.me].first, g.spots[g.me].second};
        target[u_idx] = front_spot;
    }

    // 5. Joint Tactical Search: evaluate all 125 joint moves
    Point start_pos[3];
    bool is_alive[3];
    for (int i = 0; i < 3; i++) {
        start_pos[i] = my_units[i].pos;
        is_alive[i] = my_units[i].is_alive;
    }

    int best_joint_moves[3] = {4, 4, 4};
    string best_joint_acts[3] = {"-", "-", "-"};
    double best_joint_score = -1e9;

    int candidate_moves[3][5];
    int num_candidates[3];
    for (int i = 0; i < 3; i++) {
        if (!is_alive[i]) {
            candidate_moves[i][0] = 4;
            num_candidates[i] = 1;
        } else {
            num_candidates[i] = 5;
            for (int m = 0; m < 5; m++) candidate_moves[i][m] = m;
        }
    }

    Point sim_p[3];
    bool sim_blocked[3];

    for (int m0 = 0; m0 < num_candidates[0]; m0++) {
        int move0 = candidate_moves[0][m0];
        for (int m1 = 0; m1 < num_candidates[1]; m1++) {
            int move1 = candidate_moves[1][m1];
            for (int m2 = 0; m2 < num_candidates[2]; m2++) {
                int move2 = candidate_moves[2][m2];
                int trial_moves[3] = {move0, move1, move2};

                // Simulate movement
                simulate_friendly_moves(start_pos, is_alive, trial_moves, sim_p, sim_blocked, S, g.map);

                // Score this joint move
                double joint_score = 0;

                // Penalty for friendly collisions
                for (int i = 0; i < 3; i++) {
                    if (is_alive[i] && sim_blocked[i]) {
                        joint_score -= 800.0;
                    }
                }

                // Progress towards target & threat avoidance
                for (int i = 0; i < 3; i++) {
                    if (!is_alive[i]) continue;
                    Point p = sim_p[i];
                    int pid = cell_id(p.x, p.y, S);
                    int tid = cell_id(target[i].x, target[i].y, S);

                    int threats = enemy_threat_count[pid];
                    int threat_dmg = threats * g.damage;

                    if (role[i] == 0) { // CARRY
                        // Check if p is inside base
                        if (is_base_cell[g.me][pid]) {
                            if (my_units[i].hp > threat_dmg) {
                                // Guaranteed safe capture!
                                joint_score += 25000.0;
                            } else {
                                // Dies in base
                                joint_score -= 10000.0;
                            }
                        } else {
                            // Minimize distance to base
                            int cur_db = dist_to_base[g.me][cell_id(start_pos[i].x, start_pos[i].y, S)];
                            int next_db = dist_to_base[g.me][pid];
                            joint_score += (cur_db - next_db) * 60.0;

                            // Severe penalty for carrier taking damage
                            if (threat_dmg >= my_units[i].hp) {
                                joint_score -= 20000.0; // Fatal!
                            } else {
                                joint_score -= threats * 200.0;
                            }
                        }
                    } else {
                        // Non-carrier progress
                        int cur_dt = dist_matrix[cell_id(start_pos[i].x, start_pos[i].y, S)][tid];
                        int next_dt = dist_matrix[pid][tid];
                        joint_score += (cur_dt - next_dt) * 15.0;

                        // Non-carrier threat penalty
                        if (threat_dmg >= my_units[i].hp) {
                            joint_score -= 2500.0; // Death
                        } else {
                            joint_score -= threats * 30.0;
                        }
                    }
                }

                // Focus Fire Combat & Pickup Evaluation
                string trial_acts[3] = {"-", "-", "-"};
                bool can_unit_attack[3] = {false, false, false};

                for (int i = 0; i < 3; i++) {
                    if (!is_alive[i] || role[i] == 0) continue; // dead or carrier cannot attack/pickup

                    // Check if standing on an open flag
                    bool on_open_flag = false;
                    for (const auto& f : open_flags) {
                        if (f.x == sim_p[i].x && f.y == sim_p[i].y) {
                            on_open_flag = true;
                            break;
                        }
                    }

                    int pid = cell_id(sim_p[i].x, sim_p[i].y, S);
                    int threats = enemy_threat_count[pid];

                    if (on_open_flag) {
                        // Safe pickup?
                        if (threats == 0 || my_units[i].hp > g.damage) {
                            trial_acts[i] = "P";
                            joint_score += 3000.0;
                            continue;
                        }
                    }

                    if (!my_units[i].is_respawning) {
                        can_unit_attack[i] = true;
                    }
                }

                // Among attackers, optimize focus fire
                vector<int> candidate_targets;
                for (const auto& e : enemies) {
                    if (!e.is_alive || e.is_respawning) continue;
                    int ecid = cell_id(e.pos.x, e.pos.y, S);
                    bool reachable = false;
                    for (int i = 0; i < 3; i++) {
                        if (can_unit_attack[i]) {
                            int pid = cell_id(sim_p[i].x, sim_p[i].y, S);
                            if (can_hit[pid][ecid]) {
                                reachable = true;
                                break;
                            }
                        }
                    }
                    if (reachable) candidate_targets.push_back(e.id);
                }

                double best_combat_score = 0;
                if (!candidate_targets.empty()) {
                    vector<int> attackers;
                    for (int i = 0; i < 3; i++) if (can_unit_attack[i]) attackers.push_back(i);

                    int num_atk = attackers.size();
                    if (num_atk == 1) {
                        int a0 = attackers[0];
                        int a0_pid = cell_id(sim_p[a0].x, sim_p[a0].y, S);
                        int best_t = -1;
                        double best_sc = 0;
                        for (int tid : candidate_targets) {
                            int t_cid = cell_id(all_units[tid].pos.x, all_units[tid].pos.y, S);
                            if (!can_hit[a0_pid][t_cid]) continue;
                            double sc = 100.0;
                            if (all_units[tid].flag != -1) sc += 800.0;
                            if (all_units[tid].hp <= g.damage) {
                                sc += (all_units[tid].flag != -1 ? 12000.0 : 3500.0);
                            }
                            if (sc > best_sc) { best_sc = sc; best_t = tid; }
                        }
                        if (best_t != -1) {
                            trial_acts[a0] = to_string(best_t);
                            best_combat_score = best_sc;
                        }
                    } else if (num_atk == 2) {
                        int a0 = attackers[0], a1 = attackers[1];
                        int a0_pid = cell_id(sim_p[a0].x, sim_p[a0].y, S);
                        int a1_pid = cell_id(sim_p[a1].x, sim_p[a1].y, S);
                        int best_t0 = -1, best_t1 = -1;
                        double best_sc = 0;

                        vector<int> t0_opts = {-1}, t1_opts = {-1};
                        for (int tid : candidate_targets) {
                            int t_cid = cell_id(all_units[tid].pos.x, all_units[tid].pos.y, S);
                            if (can_hit[a0_pid][t_cid]) t0_opts.push_back(tid);
                            if (can_hit[a1_pid][t_cid]) t1_opts.push_back(tid);
                        }

                        for (int t0 : t0_opts) {
                            for (int t1 : t1_opts) {
                                double sc = 0;
                                int hits[MAX_TEAMS * 3] = {0};
                                if (t0 != -1) hits[t0]++;
                                if (t1 != -1) hits[t1]++;

                                for (int tid : candidate_targets) {
                                    if (hits[tid] == 0) continue;
                                    sc += hits[tid] * 100.0;
                                    if (all_units[tid].flag != -1) sc += hits[tid] * 800.0;
                                    if (all_units[tid].hp <= hits[tid] * g.damage) {
                                        sc += (all_units[tid].flag != -1 ? 12000.0 : 3500.0);
                                    }
                                }
                                if (sc > best_sc) {
                                    best_sc = sc;
                                    best_t0 = t0; best_t1 = t1;
                                }
                            }
                        }
                        if (best_t0 != -1) trial_acts[a0] = to_string(best_t0);
                        if (best_t1 != -1) trial_acts[a1] = to_string(best_t1);
                        best_combat_score = best_sc;
                    } else if (num_atk == 3) {
                        int a0 = attackers[0], a1 = attackers[1], a2 = attackers[2];
                        int a0_pid = cell_id(sim_p[a0].x, sim_p[a0].y, S);
                        int a1_pid = cell_id(sim_p[a1].x, sim_p[a1].y, S);
                        int a2_pid = cell_id(sim_p[a2].x, sim_p[a2].y, S);
                        int best_t0 = -1, best_t1 = -1, best_t2 = -1;
                        double best_sc = 0;

                        vector<int> t0_opts = {-1}, t1_opts = {-1}, t2_opts = {-1};
                        for (int tid : candidate_targets) {
                            int t_cid = cell_id(all_units[tid].pos.x, all_units[tid].pos.y, S);
                            if (can_hit[a0_pid][t_cid]) t0_opts.push_back(tid);
                            if (can_hit[a1_pid][t_cid]) t1_opts.push_back(tid);
                            if (can_hit[a2_pid][t_cid]) t2_opts.push_back(tid);
                        }

                        for (int t0 : t0_opts) {
                            for (int t1 : t1_opts) {
                                for (int t2 : t2_opts) {
                                    double sc = 0;
                                    int hits[MAX_TEAMS * 3] = {0};
                                    if (t0 != -1) hits[t0]++;
                                    if (t1 != -1) hits[t1]++;
                                    if (t2 != -1) hits[t2]++;

                                    for (int tid : candidate_targets) {
                                        if (hits[tid] == 0) continue;
                                        sc += hits[tid] * 100.0;
                                        if (all_units[tid].flag != -1) sc += hits[tid] * 800.0;
                                        if (all_units[tid].hp <= hits[tid] * g.damage) {
                                            sc += (all_units[tid].flag != -1 ? 12000.0 : 3500.0);
                                        }
                                    }
                                    if (sc > best_sc) {
                                        best_sc = sc;
                                        best_t0 = t0; best_t1 = t1; best_t2 = t2;
                                    }
                                }
                            }
                        }
                        if (best_t0 != -1) trial_acts[a0] = to_string(best_t0);
                        if (best_t1 != -1) trial_acts[a1] = to_string(best_t1);
                        if (best_t2 != -1) trial_acts[a2] = to_string(best_t2);
                        best_combat_score = best_sc;
                    }
                }

                joint_score += best_combat_score;

                if (joint_score > best_joint_score) {
                    best_joint_score = joint_score;
                    for (int i = 0; i < 3; i++) {
                        best_joint_moves[i] = trial_moves[i];
                        best_joint_acts[i] = trial_acts[i];
                    }
                }
            }
        }
    }

    // Output command
    string out = to_string(g.turn);
    for (int i = 0; i < 3; i++) {
        out += " ";
        out += MOVE_CHAR[best_joint_moves[i]];
        out += " ";
        out += best_joint_acts[i];
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
