// T0075 control routing (bots/agent_v2/controls/): every team count plays T0068's common initialization actor
// (data/count_specialists_rl/init_actor.npz), exported by tools/screens/next_controls/export_init_actor.py into
// init_actor.hpp with the scheme, temperature and calibration protocol of the general control. Hand-written.
#pragma once
#include "init_actor.hpp"

namespace agent_v2_controls::init_only {

inline const fusion::Net& get_actor(int /*teams*/) { return fusion::models::rl_init::kActor; }

}  // namespace agent_v2_controls::init_only
