// Sampled actor-direct choice of the Meowmind RL bot (the predeclared deployment; bots/meow_rl/sampling.py mirrors it).
//
// The joint distribution is the trained one: softmax over the feasible joints of z_j = acc_j * 2^-outExp / T (the
// integer actor's joint accumulator, T = 1.05), one inverse-CDF draw. The uniform for turn t is common to every arm
// of a fixture: it depends only on the public INIT (teams, seat, size, turns, map, bases, spots) and t, never on the
// model, so control and candidate face the same draws and an A/A copy replays the same choices.
//     seed = FNV-1a 64 of "meow_rl_actor_v1\n" + canonical INIT text
//     u_t  = (splitmix64(seed + t * 0x9E3779B97F4A7C15) >> 11) * 2^-53
// The search RNG (Planner::rng) is separate and never touched here.
#pragma once
#include <cmath>
#include <cstdint>
#include <string>

namespace meow_rl_sample {

inline uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

inline std::string canonical_init(const Game& g) {
    std::string s = "meow_rl_actor_v1\n" + std::to_string(g.teams) + " " + std::to_string(g.me) + " " +
                    std::to_string(g.size) + " " + std::to_string(g.turns) + "\n";
    for (auto& row : g.map) s += row + "\n";
    for (auto& b : g.bases) s += std::to_string(b.first) + " " + std::to_string(b.second) + "\n";
    for (auto& p : g.spots) s += std::to_string(p.first) + " " + std::to_string(p.second) + "\n";
    return s;
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline double uniform(uint64_t seed, int turn) {
    return double(splitmix64(seed + uint64_t(turn) * 0x9E3779B97F4A7C15ull) >> 11) * 0x1.0p-53;
}

// One draw from the feasible joints given the three units' output accumulators per move.
inline int draw(const int64_t (&s)[3][5], const unsigned char* mask, int out_exp, double temperature, double u) {
    double z[125], top = -1e300;
    for (int j = 0; j < 125; j++)
        if (mask[j]) {
            z[j] = std::ldexp(double(s[0][j / 25] + s[1][j / 5 % 5] + s[2][j % 5]), -out_exp) / temperature;
            if (z[j] > top) top = z[j];
        }
    double total = 0, p[125];
    for (int j = 0; j < 125; j++) total += p[j] = mask[j] ? std::exp(z[j] - top) : 0.0;
    double target = u * total, acc = 0;
    int last = 0;
    for (int j = 0; j < 125; j++)
        if (mask[j]) {
            acc += p[j];
            last = j;
            if (acc > target) return j;
        }
    return last;
}

}  // namespace meow_rl_sample
