// Text protocol parser for the tools (RULES.md "Input"): INIT block, then one
// TURN block per turn. Bots keep their own parsers; this one only fills the
// fusion::Static / fusion::State structs.
#ifndef FUSION_STREAM_HPP
#define FUSION_STREAM_HPP
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../inference/state.hpp"

namespace fusion {

inline int flagStatus(const std::string& s) {
    if (s == "home") return kHome;
    if (s == "carried") return kCarried;
    if (s == "dropped") return kDropped;
    if (s == "cooldown") return kCooldown;
    throw std::runtime_error("unknown flag status " + s);
}

inline bool nextLine(std::istream& in, std::string& line) {
    while (std::getline(in, line))
        if (!line.empty() && line != "\r") return true;
    return false;
}

inline void expect(std::istringstream& ss, const char* word) {
    std::string w;
    ss >> w;
    if (w != word) throw std::runtime_error(std::string("expected ") + word + ", got " + w);
}

// Reads the INIT block (the stream's first block). Returns false at end of input.
inline bool readInit(std::istream& in, Static& s) {
    std::string line;
    if (!nextLine(in, line)) return false;
    std::istringstream head(line);
    expect(head, "INIT");
    head >> s.teams >> s.me >> s.size >> s.turns >> s.hp >> s.damage >> s.respawn >> s.flagReturn >> s.flagCooldown;
    nextLine(in, line);
    if (line.rfind("MAP", 0) != 0) throw std::runtime_error("expected MAP");
    s.wall.assign(s.size * s.size, 0);
    for (int y = 0; y < s.size; y++) {
        nextLine(in, line);
        for (int x = 0; x < s.size; x++) s.wall[y * s.size + x] = line[x] == '#';
    }
    int count = 0;
    nextLine(in, line);
    std::istringstream bases(line);
    expect(bases, "BASES");
    bases >> count;
    s.baseCenter.assign(count, {});
    for (auto& b : s.baseCenter) {
        nextLine(in, line);
        std::istringstream(line) >> b[0] >> b[1];
    }
    nextLine(in, line);
    std::istringstream spots(line);
    expect(spots, "SPOTS");
    spots >> count;
    s.spots.assign(count, {});
    for (auto& p : s.spots) {
        nextLine(in, line);
        std::istringstream(line) >> p[0] >> p[1];
    }
    return true;
}

// Reads one TURN ... END block. Returns false at end of input.
inline bool readTurn(std::istream& in, State& s) {
    std::string line, word;
    if (!nextLine(in, line)) return false;
    std::istringstream t(line);
    expect(t, "TURN");
    t >> s.turn;
    nextLine(in, line);
    std::istringstream sc(line);
    expect(sc, "SCORE");
    s.score.clear();
    for (int v; sc >> v;) s.score.push_back(v);
    int count = 0;
    nextLine(in, line);
    std::istringstream un(line);
    expect(un, "UNITS");
    un >> count;
    s.units.assign(count, {});
    for (int i = 0; i < count; i++) {
        nextLine(in, line);
        std::istringstream ss(line);
        Unit u;
        expect(ss, "U");
        ss >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawnAt;
        s.units.at(u.id) = u;
    }
    nextLine(in, line);
    std::istringstream fl(line);
    expect(fl, "FLAGS");
    fl >> count;
    s.flags.assign(count, {});
    for (int i = 0; i < count; i++) {
        nextLine(in, line);
        std::istringstream ss(line);
        Flag f;
        std::string status;
        expect(ss, "F");
        ss >> f.id >> status >> f.x >> f.y >> f.holder >> f.returnAt;
        f.status = flagStatus(status);
        s.flags.at(f.id) = f;
    }
    nextLine(in, line);
    std::istringstream ev(line);
    expect(ev, "EVENTS");
    ev >> count;
    s.attacks.clear();
    for (int i = 0; i < count; i++) {
        nextLine(in, line);
        std::istringstream ss(line);
        expect(ss, "E");
        ss >> word;
        if (word == "attack") {
            Attack a;
            ss >> a.unit >> a.target;
            s.attacks.push_back(a);
        }
    }
    nextLine(in, line);
    if (line.rfind("END", 0) != 0) throw std::runtime_error("expected END, got " + line);
    return true;
}

}  // namespace fusion
#endif  // FUSION_STREAM_HPP
