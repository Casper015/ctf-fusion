// The rules simulator of the agent_v2 carrier (task T0065, docs/AGENT_V2.md): one engine settlement.
//
// fusion::step (inference/sim.hpp, engine Game.step phases 1-6, unchanged) on the public State with complete orders for
// every unit. The one input the public state cannot know is where a returning flag lands: the engine draws a free
// spot with its hidden flag RNG (engine Game.free_spot). Search must not know that draw, so it samples a LAWFUL outcome
// with its own scenario stream, mirroring the engine's rule: flags return in flag-id order; each picks uniformly among
// the designated spots not occupied by a home flag (including flags returned earlier in the same settlement; dropped
// flags stay on their cells and reserve nothing), or among all spots when none is free. Common scenario seeds reduce
// variance across root candidates; they do not guarantee identical refreshes when actions change which spots are free.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "core.hpp"

namespace agent_v2 {

inline uint64_t xorshift(uint64_t& x) {
    x ^= x << 13, x ^= x >> 7, x ^= x << 17;
    return x;
}

// The spots of the flags that return in this settlement of `s` (-1 for the others).
inline void sample_returns(const MapCache& c, const fusion::State& s, uint64_t seed, std::vector<std::array<int, 2>>& spots) {
    uint64_t rng = seed ? seed : 1;
    spots.assign(s.flags.size(), {-1, -1});
    std::vector<std::array<int, 2>> home;
    for (auto& f : s.flags)
        if (f.status == fusion::kHome && f.x >= 0) home.push_back({f.x, f.y});
    std::vector<std::array<int, 2>> free;
    for (auto& f : s.flags) {
        if (!((f.status == fusion::kDropped || f.status == fusion::kCooldown) && s.turn >= f.returnAt)) continue;
        free.clear();
        for (auto& p : c.st.spots) {
            bool taken = false;
            for (auto& h : home) taken |= h == p;
            if (!taken) free.push_back(p);
        }
        const auto& pool = free.empty() ? c.st.spots : free;
        spots[f.id] = pool[xorshift(rng) % pool.size()];
        home.push_back(spots[f.id]);
    }
}

struct Simulator {
    fusion::Workspace ws;
    std::vector<std::array<int, 2>> spots;
    fusion::StepEvents events;

    // The successor of `s` under `orders` (indexed by unit id), with lawful sampled refreshes.
    fusion::State step(const MapCache& c, const fusion::State& s, const std::vector<fusion::Order>& orders, uint64_t seed) {
        sample_returns(c, s, seed, spots);
        fusion::State next;
        fusion::step(c.board, c.rules, s, orders.data(), next, ws, &events, spots.data());
        return next;
    }
};

}  // namespace agent_v2
