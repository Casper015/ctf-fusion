// The P-view cache contract of C08's per-branch ply-2 predictions (search_experiment.py, PONLY_PLY2_MODE 1/2):
// two successor states are one cache entry only when every field of the public state is equal; a change of one
// unit's HP, a flag, a respawn time or the last attacks makes another entry, and reaches the policy encoder.
// Prints "SAME <case> <0|1>" and "ENCODER <case> <0|1>" (1: the encoded inputs of the changed team differ).
//
//   g++ -std=c++20 -O2 -Iinference -Ibots/p_only/include -Ibots/p_only/bots -DFUSION_TIER_XXL -DPONLY_FAMILIES \
//       bots/p_only/tests/ply2_cache_test.cpp
#define PONLY_NO_MAIN 1
#include "c08_ponly_f.hpp"

#include <cstdio>
#include <cstring>

namespace {

fusion::State base() {
    fusion::State s;
    s.turn = 7;
    s.score = {1, 0, 2};
    for (int id = 0; id < 9; id++) s.units.push_back({id, id / 3, 3 + id, 5 + id % 3, 100, -1, -1});
    s.flags.push_back({0, fusion::kHome, 10, 10, -1, -1});
    s.flags.push_back({1, fusion::kDropped, 12, 12, -1, 20});
    return s;
}

bool encodedDiffers(const fusion::State& a, const fusion::State& b, int team) {
    fusion::Static st;
    st.teams = 3; st.me = 0; st.size = 23;
    st.wall.assign(23 * 23, 0);
    st.baseCenter = {{3, 3}, {19, 3}, {11, 19}};
    st.spots = {{11, 11}};
    fusion::Board board;
    board.init(st);
    fusion::TurnView va, vb;
    fusion::makeView(board, a, &a, va);  // the same parent for both: only the state differs
    fusion::makeView(board, b, &a, vb);
    fusion::PolicyFeatures fa, fb;
    fusion::encodePolicy(board, a, va, team, fa);
    fusion::encodePolicy(board, b, vb, team, fb);
    return std::memcmp(&fa, &fb, sizeof fa) != 0;
}

void report(const char* name, const fusion::State& changed, int team) {
    const fusion::State a = base();
    std::printf("SAME %s %d\n", name, int(multi::Planner::Ply2FromNetwork::sameState(a, changed)));
    std::printf("ENCODER %s %d\n", name, int(encodedDiffers(a, changed, team)));
}

}  // namespace

int main() {
    report("identical", base(), 1);
    fusion::State hp = base(); hp.units[4].hp = 34; report("hp", hp, 1);                   // team 1 sees its own unit's HP
    fusion::State dead = base(); dead.units[4] = {4, 1, -1, -1, 0, -1, 17}; report("death", dead, 1);
    fusion::State flag = base(); flag.flags[1].returnAt = 18; report("flag_timer", flag, 1);
    fusion::State carried = base(); carried.flags[0] = {0, fusion::kCarried, -1, -1, 3, -1}; carried.units[3].flag = 0; report("carried", carried, 1);
    fusion::State hit = base(); hit.attacks.push_back({0, 4}); report("attack", hit, 1);       // hits taken last settlement
    fusion::State score = base(); score.score[2] = 3; report("score", score, 1);
    return 0;
}
