// Fixed-work replay: feeds a recorded seat input stream (stdin) to a P-only
// carrier, exactly as its main() would, and prints one line per turn:
//
//   <turn> <digest of the network's root predictions> <the bot's reply>
//
// then "truncated <n>": expectimax calls a deadline stopped before their
// requested samples, plus checked lookaheads a deadline cut short (C08; 0 for
// v12). Only an untruncated replay is fixed work. Two builds of the same carrier and P
// model produce the same digests on the same stream whatever else differs
// (V on or off, coefficients): the network's predictions do not depend on V.
// With PONLY_FIXED_SAMPLES and no truncation, a build with V compiled in but
// all V coefficients 0 must reply exactly as the V-off build (V-off
// equivalence). p_only/tests/test_p_only.py compares the outputs.
//
//   g++ -std=c++20 -O2 -Iinference -Ip_only/include -Ip_only/bots -DCARRIER_C08 -DFUSION_TIER_PXL_VM \
//       -DPONLY_FIXED_SAMPLES=64 p_only/tests/replay_test.cpp
#define PONLY_NO_MAIN 1
#if defined(CARRIER_C08)
#include "c08_ponly.hpp"
#elif defined(CARRIER_V12)
#include "v12_ponly.hpp"
#else
#error "define CARRIER_C08 or CARRIER_V12"
#endif

#include <cstdio>
#include <cstring>

namespace {

// FNV-1a over the bits of every opponent unit's move marginal at the root.
unsigned long long predictionDigest(const ponly::View& v, int me) {
    unsigned long long h = 1469598103934665603ull;
    for (const auto& u : v.state().units) {
        if (u.team == me || !v.has(u.team) || v.origin(u.id) < 0) continue;
        for (int m = 0; m < ponly::kMoves; m++) {
            const double p = v.move(u.id, m);
            unsigned long long bits;
            std::memcpy(&bits, &p, sizeof bits);
            for (int i = 0; i < 8; i++) { h ^= (bits >> (8 * i)) & 0xff; h *= 1099511628211ull; }
        }
    }
    return h;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    Game g;
#if defined(CARRIER_C08)
    multi::Planner many;
    duel::Player two;
#else
    Planner p;
#endif
    while (read_turn(g)) {
#if defined(CARRIER_C08)
        const std::string out = g.teams == 2 ? two.decide(g) : many.decide(g);
        const ponly::View& view = g.teams == 2 ? two.session().view : many.fSession.view;
#else
        const std::string out = p.decide(g);
        const ponly::View& view = p.fSession.view;
#endif
        std::printf("%d %016llx %s\n", g.turn, predictionDigest(view, g.me), out.c_str());
    }
#if defined(CARRIER_C08)
    std::printf("truncated %lld\n", many.statTruncated + many.statRefineCuts);  // search calls and checked lookaheads cut short
#else
    std::printf("truncated 0\n");
#endif
    return 0;
}
