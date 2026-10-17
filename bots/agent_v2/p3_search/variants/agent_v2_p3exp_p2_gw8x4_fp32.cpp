// P3 opponent-prediction experiment: only P differs between paired families. No strength admission.
#define FUSION_MAX_WIDTH 1024
#define FUSION_TIER_PXXL_VM 1
#define AGENT_V2_SEARCH_CANDIDATES 8
#define AGENT_V2_SEARCH_SAMPLES 4
#define AGENT_V2_SEARCH_TAU 0.002
#include "../../../../inference/meow_rl/aligned/p2_large/actor.hpp"
#include "../../../../inference/meow_rl/aligned/p2_large/critic_fp32.hpp"
#include "../../search_gate/bot.hpp"
