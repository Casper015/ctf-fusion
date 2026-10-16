// Native bridge of the agent_v2 carrier for PPO (task T0065, docs/AGENT_V2.md; training/agent_v2/bridge.py builds it).
//
// training.meow_rl's C ABI 1 (mr_observe / mr_resolve / mr_forget ...), so the existing rollout code (env.run_episode,
// its collector, PPO/GAE, storage) runs unchanged on the new carrier. The trainer's FP32 actor samples each joint
// from the observation written here; mr_resolve executes exactly that joint through the carrier's Runtime::answer
// (assign_v1 on our P's scenario table, the reply, the history commit), the code the deployed bot runs. PPO therefore
// stores the probability of the movement joint actually executed; the rest of the action (attacks, pickups) is the
// resolver's deterministic function of the state, that joint and the frozen scenario seeds (action schema
// agent_v2_actions_v1). The bridge never searches: a search-selected joint is never a PPO sample.
// Shapes: feats [3][5][112], unit_legal [3][5], joint_mask [125] (j = 25 a + 5 b + c), value_feats [416]
// (meow_policy112_v1 / meow_public416_v1). ref_prob is all zero (no Meowmind own-policy network here) and mr_weights
// knows no weights (warm starts come from exported actor arrays).
#ifndef FUSION_MAX_WIDTH
#define FUSION_MAX_WIDTH 1024
#endif
#include <cstring>
#include <exception>
#include <string>

#if defined(FUSION_TIER_XXL)
#include "models_xxl/registry.hpp"
#elif defined(FUSION_TIER_PXL_VM)
#include "models_pxl_vm/registry.hpp"
#elif defined(FUSION_TIER_PXXL_VM)
#include "models_pxxl_vm/registry.hpp"
#else
#error "the agent_v2 bridge selects our opponent model registry with FUSION_TIER_*"
#endif
#include ACTOR_HEADER
#include "../../../inference/agent_v2/runtime.hpp"

#if defined(_WIN32)
#define MR_API extern "C" __declspec(dllexport)
#else
#define MR_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct Handle {
    agent_v2::Runtime<agent_v2::NoSearch> rt;
    std::string error;
};

int fail(Handle* h, const std::string& what) {
    if (h) h->error = what;
    return -1;
}

}  // namespace

MR_API int mr_abi_version() { return 1; }
MR_API int mr_policy_features() { return agent_v2::kPF; }
MR_API int mr_value_features() { return agent_v2::kVF; }
MR_API const char* mr_carrier() { return agent_v2::kCarrier; }
MR_API const char* mr_action_schema() { return agent_v2::kActionSchema; }

MR_API void* mr_create() {
    try {
        return new Handle();
    } catch (...) {
        return nullptr;
    }
}

MR_API void mr_destroy(void* handle) { delete static_cast<Handle*>(handle); }

MR_API const char* mr_error(void* handle) {
    auto* h = static_cast<Handle*>(handle);
    return h ? h->error.c_str() : "null session";
}

MR_API const char* mr_predictor(void* handle) {
    auto* h = static_cast<Handle*>(handle);
    return h ? h->rt.pred.model_id() : "";
}

MR_API int mr_observe(void* handle, const char* text, float* feats, unsigned char* unit_legal, unsigned char* joint_mask,
                      float* value_feats, float* ref_prob, int* info) {
    auto* h = static_cast<Handle*>(handle);
    if (!h) return -1;
    try {
        auto& rt = h->rt;
        if (!rt.answered) return fail(h, "turn " + std::to_string(rt.s.turn) + " was observed and not answered");
        if (auto why = rt.observe_block(text); !why.empty()) return fail(h, why);
        auto& o = rt.obs;
        std::memcpy(feats, o.feats, sizeof o.feats);
        std::memcpy(unit_legal, o.unit_legal, sizeof o.unit_legal);
        std::memcpy(joint_mask, o.joint_mask, sizeof o.joint_mask);
        std::memcpy(value_feats, o.value_feats, sizeof o.value_feats);
        std::memset(ref_prob, 0, sizeof(float) * agent_v2::kUnits * agent_v2::kMoves);
        int fields[8] = {o.turn, o.turns, o.teams, o.me, o.size, o.alive, o.valid, o.units};
        std::memcpy(info, fields, sizeof fields);
        h->error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(h, std::string("exception: ") + e.what());
    } catch (...) {
        return fail(h, "unknown exception");
    }
}

MR_API int mr_resolve(void* handle, int joint, char* out, int cap, int* actions) {
    auto* h = static_cast<Handle*>(handle);
    if (!h) return -1;
    try {
        auto& rt = h->rt;
        if (rt.answered) return fail(h, "no observed turn to answer");
        if (joint < 0 || joint >= agent_v2::kJoint || !rt.obs.joint_mask[joint]) return fail(h, "the joint is not feasible");
        std::string reply = rt.answer(joint);
        if (int(reply.size()) + 1 > cap) return fail(h, "reply buffer too small");
        std::memcpy(out, reply.c_str(), reply.size() + 1);
        for (int k = 0; k < 3; k++) {
            const auto& o = rt.tactics.orders[k];
            actions[k] = o.act == fusion::kOrderAttack ? o.target : o.act == fusion::kOrderDrop ? -3 : -1;
        }
        h->error.clear();
        return 0;
    } catch (const std::exception& e) {
        return fail(h, std::string("exception: ") + e.what());
    } catch (...) {
        return fail(h, "unknown exception");
    }
}

// After a refused block: the history is not trusted.
MR_API int mr_forget(void* handle) {
    auto* h = static_cast<Handle*>(handle);
    if (!h) return -1;
    h->rt.hist.forget();
    h->rt.answered = true;
    return 0;
}

MR_API int mr_scalar_features(void*, float*) { return -1; }
MR_API int mr_weights(const char*, float*, int) { return -1; }
