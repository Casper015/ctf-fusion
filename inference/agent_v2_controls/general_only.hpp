// T0075 control routing (bots/agent_v2/controls/): every team count plays T0068's compute-matched general control
// rl_general (inference/count_specialists_rl/weights_general.hpp). Hand-written: no generated weights here.
#pragma once
#include "../count_specialists_rl/weights_general.hpp"

namespace agent_v2_controls::general_only {

inline const fusion::Net& get_actor(int /*teams*/) { return fusion::models::rl_general::kActor; }

}  // namespace agent_v2_controls::general_only
