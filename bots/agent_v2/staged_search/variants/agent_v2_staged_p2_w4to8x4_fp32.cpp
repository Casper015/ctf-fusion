// T0087 staged search, widening: the same tree as agent_v2_staged_p2_r8x4_fp32, with a frozen schedule that starts
// with the first 4 roots (the actor's joint first) over 1 and 2 scenarios, then all 8 over 2, 3 and 4. Opt-in; not
// screened.
#define FUSION_MAX_WIDTH 1024
#define FUSION_TIER_P2GENERAL_VM 1
#define AGENT_V2_SEARCH_CANDIDATES 8
#define AGENT_V2_SEARCH_SAMPLES 4
#define AGENT_V2_SEARCH_TAU 0.002
#define AGENT_V2_STAGED_SCHEDULE {4, 1}, {4, 2}, {8, 2}, {8, 3}, {8, 4}
#include "../../../../inference/meow_rl/aligned/p2_large/actor.hpp"
#include "../../../../inference/meow_rl/aligned/p2_large/critic_fp32.hpp"
#include "../bot.hpp"
