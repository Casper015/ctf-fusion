// agent_v2 variant agent_v2_p2xxl_gw8x4_fp32 (task T0076, docs/AGENT_V2_SEARCH_GATE.md). Hand-written: no generated weights here.
// gated-wide (GW): 8 candidates x 4 scenarios, override only when the best root's mean beats the actor's by more than tau 0.002.
// Breadth and tau frozen by bots/agent_v2/search_gate/diagnostics/DIAGNOSTICS.json; actor, FP32 critic, P2 XXL and the
// admission settings are agent_v2_p2xxl_search_fp32's.
#define FUSION_MAX_WIDTH 1024
#define FUSION_TIER_PXXL_VM 1
#define AGENT_V2_SEARCH_CANDIDATES 8
#define AGENT_V2_SEARCH_SAMPLES 4
#define AGENT_V2_SEARCH_TAU 0.002
#include "../../../../inference/meow_rl/aligned/p2_large/actor.hpp"
#include "../../../../inference/meow_rl/aligned/p2_large/critic_fp32.hpp"
#include "../bot.hpp"
