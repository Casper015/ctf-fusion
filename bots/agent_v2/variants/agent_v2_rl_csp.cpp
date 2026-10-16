// agent_v2 variant agent_v2_rl_csp (task T0068, docs/AGENT_V2.md, docs/COUNT_SPECIALISTS_RL.md).
// actor-direct: T0068's exact-count PPO RL policy specialists (inference/count_specialists_rl),
// T0067's per-count opponent P specialists (inference/count_specialists_p), the extracted resolver.
#define FUSION_MAX_WIDTH 2048
#define FUSION_TIER_CSP 1

#include "../../../inference/count_specialists_rl/registry.hpp"
#include "../../../inference/mlp.hpp"

inline int g_active_teams = 0;

namespace meow_rl_model {
inline constexpr const char* kName = "count_specialists_rl";
inline constexpr const char* kScheme = "meow_rl_reference16_v1";
inline constexpr const char* kDigest = "count_specialists_rl_v1";
inline constexpr int kInBits = 12;
inline constexpr double kTemperature = 1.05;

struct DynamicActor {
    int outExp = 6;
    operator const fusion::Net&() const {
        return count_specialists_rl::get_actor(g_active_teams);
    }
};
inline constexpr DynamicActor kActor{};
}  // namespace meow_rl_model

#include "../bot.hpp"
