// agent_v2 variant agent_v2_p2xxl_direct (task T0065, docs/AGENT_V2.md, profiles/agent_v2.json). Hand-written: no generated weights here.
// actor-direct: the T0059 pool-v2 large-V actor (aligned export), our P2 XXL opponent model, the extracted resolver
#define FUSION_MAX_WIDTH 1024
#define FUSION_TIER_PXXL_VM 1
#include "../../../inference/meow_rl/aligned/p2_large/actor.hpp"
#include "../bot.hpp"
