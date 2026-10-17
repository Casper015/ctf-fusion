// agent_v2 production release variant agent_v2_rl_general_csp (T0077 winner, +0.2 BT).
// actor-direct: T0068's compute-matched general PPO policy (rl_general) on every count,
// T0067's per-count opponent P specialists (inference/count_specialists_p), the extracted resolver.
#define FUSION_MAX_WIDTH 2048
#define FUSION_TIER_CSP 1

#include "../../../inference/agent_v2_controls/general_only.hpp"
#include "../../../inference/mlp.hpp"

namespace agent_v2_bot {
extern int g_active_teams;
}

namespace meow_rl_model {
inline constexpr const char* kName = "count_specialists_rl_general_only";
inline constexpr const char* kScheme = "meow_rl_reference16_v1";
inline constexpr const char* kDigest = fusion::models::rl_general::kDigest;
inline constexpr int kInBits = 12;
inline constexpr double kTemperature = 1.05;
static_assert(fusion::models::rl_general::kInBits == kInBits, "the actor's input bits are agent_v2_rl_csp's");
static_assert(fusion::models::rl_general::kTemperature == kTemperature, "the actor's temperature is agent_v2_rl_csp's");
static_assert(fusion::models::rl_general::kActor.outExp == 16, "the actor's outExp is agent_v2_rl_csp's");

struct DynamicActor {
    int outExp = 16;
    operator const fusion::Net&() const {
        return agent_v2_controls::general_only::get_actor(agent_v2_bot::g_active_teams);
    }
};
inline constexpr DynamicActor kActor{};
}  // namespace meow_rl_model

#include "../bot.hpp"
