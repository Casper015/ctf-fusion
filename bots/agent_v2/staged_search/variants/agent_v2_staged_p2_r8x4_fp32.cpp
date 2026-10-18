// T0087 staged search, full-width rounds: the tree of agent_v2_p3exp_p2_gw8x4_fp32 (general P2 XXL, 8 candidates x 4
// scenarios, tau 0.002, FP32 critic), checkpointed after each scenario round over all 8 roots. Opt-in; not screened.
#define FUSION_MAX_WIDTH 1024
#define FUSION_TIER_P2GENERAL_VM 1
#define AGENT_V2_SEARCH_CANDIDATES 8
#define AGENT_V2_SEARCH_SAMPLES 4
#define AGENT_V2_SEARCH_TAU 0.002
#include "../../../../inference/meow_rl/aligned/p2_large/actor.hpp"
#include "../../../../inference/meow_rl/aligned/p2_large/critic_fp32.hpp"
#include "../bot.hpp"
