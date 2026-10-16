// agent_v2 variant agent_v2_p2xxl_search_fp32 (task T0065, docs/AGENT_V2.md, profiles/agent_v2.json). Hand-written: no generated weights here.
// experimental: + bounded two-ply search with the checkpoint's FP32 critic
#define FUSION_MAX_WIDTH 1024
#define FUSION_TIER_PXXL_VM 1
#define AGENT_V2_SEARCH 1
#include "../../../inference/meow_rl/aligned/p2_large/actor.hpp"
#include "../../../inference/meow_rl/aligned/p2_large/critic_fp32.hpp"
#include "../bot.hpp"
