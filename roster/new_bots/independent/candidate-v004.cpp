// Incumbent: parses the game state every turn. decide() sends each unit to the nearest available
// flag, or to its own base while carrying, picks up flags on arrival, and fires at the weakest
// enemy in range after moving.
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

// Move order: stay, Up, Down, Left, Right. Ties keep the earlier entry, so staying is preferred.
const int DX[5] = {0, 0, 0, -1, 1};
const int DY[5] = {0, -1, 1, 0, 0};
const char MV[5] = {'S', 'U', 'D', 'L', 'R'};

bool open_cell(const Game& g, int x, int y) {
    return x >= 0 && y >= 0 && x < g.size && y < g.size && g.map[y][x] == '.';
}

// Shortest-path distance over open cells from the nearest source; -1 if unreachable.
vector<int> distance_field(const Game& g, const vector<pair<int, int>>& sources) {
    int n = g.size;
    vector<int> d(n * n, -1), q;
    for (auto [x, y] : sources) {
        if (open_cell(g, x, y) && d[y * n + x] < 0) { d[y * n + x] = 0; q.push_back(y * n + x); }
    }
    for (size_t h = 0; h < q.size(); h++) {
        int x = q[h] % n, y = q[h] / n;
        for (int k = 1; k < 5; k++) {
            int nx = x + DX[k], ny = y + DY[k];
            if (open_cell(g, nx, ny) && d[ny * n + nx] < 0) { d[ny * n + nx] = d[q[h]] + 1; q.push_back(ny * n + nx); }
        }
    }
    return d;
}

// Mirrors engine.py can_attack: Manhattan range 2, with the wall rules for straight and diagonal shots.
bool can_attack(const Game& g, int ax, int ay, int bx, int by) {
    int dx = bx - ax, dy = by - ay;
    if (abs(dx) + abs(dy) > 2) return false;
    if (abs(dx) == 2 && dy == 0) return open_cell(g, ax + dx / 2, ay);
    if (abs(dy) == 2 && dx == 0) return open_cell(g, ax, ay + dy / 2);
    if (abs(dx) == 1 && abs(dy) == 1) return open_cell(g, ax + dx, ay) || open_cell(g, ax, ay + dy);
    return true;
}

// Respawn cells for a team, in the engine's preference order: closest to the map center first.
vector<pair<int, int>> spawn_cells(const Game& g, int team) {
    int c = g.size / 2, bx = g.bases[team].first, by = g.bases[team].second;
    vector<pair<int, int>> cells;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) cells.push_back({bx + dx, by + dy});
    auto key = [&](pair<int, int> p) {
        return make_pair(abs(p.first - c) + abs(p.second - c), abs(p.first - bx) + 2 * abs(p.second - by));
    };
    stable_sort(cells.begin(), cells.end(), [&](pair<int, int> a, pair<int, int> b) { return key(a) < key(b); });
    return cells;
}

// Generates a move and action for each of your 3 units (IDs: 3*me, 3*me+1, 3*me+2).
// Moves: U (Up), D (Down), L (Left), R (Right), S (Stay).
// Actions: - (None), P (Pickup), X (Drop), integer ID (Attack target unit).
string decide(const Game& g) {
    int n = g.size, me = g.me, first = 3 * me;
    vector<Unit> u = g.units;                    // Local copy: due respawns below are predictions
    int total = u.size();
    vector<char> occ(n * n, 0), spawned(total, 0);
    for (auto& v : u) if (v.x >= 0) occ[v.y * n + v.x] = 1;

    // The engine respawns due dead units before movement, in ID order, on the first free base cell.
    for (int i = 0; i < total; i++) {
        Unit& v = u[i];
        if (v.x >= 0 || v.respawn_at < 0 || g.turn < v.respawn_at) continue;
        for (auto [x, y] : spawn_cells(g, v.team)) {
            if (occ[y * n + x]) continue;
            v.x = x; v.y = y; occ[y * n + x] = 1; spawned[i] = 1;
            break;
        }
    }

    // Flags a unit can target now: home on a spot, or dropped and not returning this turn.
    vector<int> cand;
    for (int k = 0; k < (int)g.flags.size(); k++) {
        const Flag& f = g.flags[k];
        if (f.x >= 0 && (f.status == "home" || (f.status == "dropped" && f.return_at > g.turn))) cand.push_back(k);
    }
    vector<vector<int>> flag_field(g.flags.size());
    for (int k : cand) flag_field[k] = distance_field(g, {{g.flags[k].x, g.flags[k].y}});

    vector<pair<int, int>> home;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) home.push_back({g.bases[me].first + dx, g.bases[me].second + dy});
    vector<int> home_field = distance_field(g, home);

    // Greedy claims: nearest (unit, flag) pairs first, at most one flag per unit and per flag.
    struct Claim { int d, unit, flag; };
    vector<Claim> claims;
    for (int i = first; i < first + 3 && i < total; i++) {
        if (u[i].x < 0 || u[i].flag >= 0) continue;
        vector<int> dist = distance_field(g, {{u[i].x, u[i].y}});
        for (int k : cand) {
            int d = dist[g.flags[k].y * n + g.flags[k].x];
            if (d >= 0) claims.push_back({d, i, k});
        }
    }
    sort(claims.begin(), claims.end(), [](const Claim& a, const Claim& b) {
        return tie(a.d, a.unit, a.flag) < tie(b.d, b.unit, b.flag);
    });
    vector<int> goal(total, -1);
    vector<char> claimed(g.flags.size(), 0);
    for (auto& c : claims) {
        if (goal[c.unit] < 0 && !claimed[c.flag]) { goal[c.unit] = c.flag; claimed[c.flag] = 1; }
    }

    vector<char> reserved(n * n, 0);
    string out = to_string(g.turn);
    for (int k = 0; k < 3; k++) {
        int i = first + k;
        if (i >= total || u[i].x < 0) { out += " S -"; continue; }

        // Carriers head home. Other units head for their claimed flag, if any.
        bool carrier = u[i].flag >= 0;
        const vector<int>* field = carrier ? &home_field : goal[i] >= 0 ? &flag_field[goal[i]] : nullptr;
        int bx = u[i].x, by = u[i].y, best = 0;
        int bestd = field ? (*field)[by * n + bx] : -1;
        if (field) {
            for (int d = 1; d < 5; d++) {
                int nx = bx + DX[d], ny = by + DY[d];
                if (!open_cell(g, nx, ny) || occ[ny * n + nx] || reserved[ny * n + nx]) continue;
                int dv = (*field)[ny * n + nx];
                if (dv >= 0 && (bestd < 0 || dv < bestd)) { best = d; bestd = dv; }
            }
            // A carrier whose way forward is blocked by an ally slides to an equally close open cell.
            if (carrier && best == 0 && bestd > 0) {
                for (int d = 1; d < 5; d++) {
                    int nx = bx + DX[d], ny = by + DY[d];
                    if (!open_cell(g, nx, ny) || occ[ny * n + nx] || reserved[ny * n + nx]) continue;
                    if ((*field)[ny * n + nx] == bestd) { best = d; break; }
                }
            }
        }
        int dx = bx + DX[best], dy = by + DY[best];
        reserved[dy * n + dx] = 1;

        // Pickup on arrival at a live flag. Carriers never attack, and respawned units cannot attack this turn.
        string act = "-";
        if (!carrier) {
            for (int f : cand) {
                if (g.flags[f].x == dx && g.flags[f].y == dy) { act = "P"; break; }
            }
        }
        if (act == "-" && !carrier && !spawned[i]) {
            int target = -1;
            for (int j = 0; j < total; j++) {
                if (u[j].team == me || u[j].x < 0 || spawned[j]) continue;
                if (!can_attack(g, dx, dy, u[j].x, u[j].y)) continue;
                if (target < 0 || u[j].hp < u[target].hp) target = j;
            }
            if (target >= 0) act = to_string(u[target].id);
        }
        out += " ";
        out += MV[best];
        out += " " + act;
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
