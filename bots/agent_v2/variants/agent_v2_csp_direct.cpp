// agent_v2 variant agent_v2_csp_direct (task T0072, docs/AGENT_V2.md). Hand-written: no generated weights here.
// actor-direct: the T0059 pool-v2 large-V actor (aligned export), T0067's per-count opponent P specialists, the extracted
// resolver. The same actor and resolver as agent_v2_p2xxl_direct; only P's bank differs. Not a production default.
#define FUSION_MAX_WIDTH 2048
#define FUSION_TIER_CSP 1
#include "../../../inference/meow_rl/aligned/p2_large/actor.hpp"
#include "../bot.hpp"
