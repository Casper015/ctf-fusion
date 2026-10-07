// Differential check of inference/sim.hpp against recorded engine matches.
//
//   sim_check <stream.txt> <replies.txt>
//
// stream.txt: a seat stream (INIT + every TURN block) followed by the
// post-game block; replies.txt: one line per (turn, team), turn-major, the
// reply exactly as sent or "-" when there was none. For every turn the
// simulator settles the recorded orders and the result must equal the next
// recorded block: units, flags, scores and attack events. Flags that return
// to a random spot get the spot from the recorded block (the public state
// cannot know it): the flag's cell if it lies at home, or its holder's cell if
// a unit camping the spot picked it up in the same settlement (pickups happen
// after movement, on the picker's final cell). Prints one summary line; exit
// code 1 on any mismatch.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../inference/sim.hpp"
#include "stream.hpp"

using namespace fusion;

namespace {

// engine Game.decode for one team; false when the engine would reject the line.
bool decode(const std::string& line, int turn, int team, std::vector<Order>& orders) {
    std::istringstream ss(line.substr(0, line.find('#')));
    std::vector<std::string> tok;
    for (std::string t; ss >> t;) tok.push_back(t);
    if (tok.size() != 7 || tok[0] != std::to_string(turn)) return false;
    Order parsed[3];
    for (int k = 0; k < 3; k++) {
        const std::string& mv = tok[1 + 2 * k];
        const std::string& act = tok[2 + 2 * k];
        const char* moves = "SUDLR";
        if (mv.size() != 1 || !std::strchr(moves, mv[0])) return false;
        parsed[k].move = static_cast<int>(std::strchr(moves, mv[0]) - moves);
        if (act == "-") parsed[k].act = kOrderWait;
        else if (act == "P") parsed[k].act = kOrderPickup;
        else if (act == "X") parsed[k].act = kOrderDrop;
        else if (!act.empty() && std::all_of(act.begin(), act.end(), ::isdigit)) { parsed[k].act = kOrderAttack; parsed[k].target = std::stoi(act); }
        else return false;
    }
    for (int k = 0; k < 3; k++) orders[3 * team + k] = parsed[k];
    return true;
}

int compare(const State& got, const State& want, std::string& why) {
    int bad = 0;
    std::ostringstream os;
    if (got.score != want.score) { bad++; os << " score"; }
    for (size_t i = 0; i < want.units.size(); i++) {
        const Unit &a = got.units[i], &b = want.units[i];
        if (a.x != b.x || a.y != b.y || a.hp != b.hp || a.flag != b.flag || a.respawnAt != b.respawnAt) {
            bad++;
            os << " unit" << i << "(" << a.x << "," << a.y << ",hp" << a.hp << ",f" << a.flag << ",r" << a.respawnAt << " vs "
               << b.x << "," << b.y << ",hp" << b.hp << ",f" << b.flag << ",r" << b.respawnAt << ")";
        }
    }
    for (size_t i = 0; i < want.flags.size(); i++) {
        const Flag &a = got.flags[i], &b = want.flags[i];
        if (a.status != b.status || a.x != b.x || a.y != b.y || a.holder != b.holder || a.returnAt != b.returnAt) {
            bad++;
            os << " flag" << i;
        }
    }
    auto key = [](std::vector<Attack> v) {
        std::vector<std::pair<int, int>> k;
        for (auto& a : v) k.push_back({a.unit, a.target});
        std::sort(k.begin(), k.end());
        return k;
    };
    if (key(got.attacks) != key(want.attacks)) { bad++; os << " attacks"; }
    why = os.str();
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: sim_check stream replies\n"); return 2; }
    std::ifstream in(argv[1]), rep(argv[2]);
    Static st;
    if (!readInit(in, st)) return 2;
    Board board;
    board.init(st);
    std::vector<State> blocks;
    for (State s; readTurn(in, s);) blocks.push_back(s);
    std::vector<std::string> replies;
    for (std::string line; std::getline(rep, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        replies.push_back(line);
    }
    const int T = static_cast<int>(blocks.size()) - 1, n = st.teams;
    if (static_cast<int>(replies.size()) != T * n) { std::fprintf(stderr, "expected %d reply lines, got %zu\n", T * n, replies.size()); return 2; }
    const Rules rules = rulesOf(st);
    Workspace ws;
    State next;
    std::vector<Order> orders;
    long mismatches = 0, compared = 0, skipped = 0, events = 0;
    for (int t = 0; t < T; t++) {
        const State& s = blocks[t];
        const State& actual = blocks[t + 1];
        orders.assign(s.units.size(), Order{});
        for (int team = 0; team < n; team++) {
            const std::string& line = replies[t * n + team];
            if (line != "-") decode(line, s.turn, team, orders);
        }
        std::vector<std::array<int, 2>> spots(s.flags.size(), {-1, -1});
        bool unknowable = false;
        for (const Flag& f : s.flags)
            if ((f.status == kDropped || f.status == kCooldown) && s.turn >= f.returnAt) {
                const Flag& a = actual.flags[f.id];
                if (a.status == kHome && a.x >= 0) spots[f.id] = {a.x, a.y};
                else if (a.status == kCarried && a.holder >= 0 && actual.units[a.holder].x >= 0)
                    spots[f.id] = {actual.units[a.holder].x, actual.units[a.holder].y};
                else unknowable = true;
            }
        if (unknowable) { skipped++; continue; }
        StepEvents ev;
        step(board, rules, s, orders.data(), next, ws, &ev, spots.data());
        events += ev.deaths + ev.pickups + ev.drops + ev.captures + ev.respawns;
        std::string why;
        const int bad = compare(next, actual, why);
        compared++;
        if (bad) {
            if (mismatches < 5) std::fprintf(stderr, "turn %d:%s\n", s.turn, why.c_str());
            mismatches++;
        }
    }
    std::printf("turns %d compared %ld skipped %ld mismatches %ld events %ld\n", T, compared, skipped, mismatches, events);
    return mismatches ? 1 : 0;
}
