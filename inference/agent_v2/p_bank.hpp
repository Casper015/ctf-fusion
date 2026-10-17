// The opponent model bank of the agent_v2 carrier (docs/AGENT_V2.md): the one place that turns the variant's
// FUSION_TIER_* into P's model registry. The bot (bots/agent_v2/bot.hpp), the PPO bridge and the contract checks
// (tools/cpp/agent_v2/) all reach it through predictor.hpp, so they cannot select different banks.
//
//   FUSION_TIER_M / L / XL / XXL / PXL_VM / PXXL_VM   inference/models_*: fusion::models::forTeams, as before
//   FUSION_TIER_CSP   T0067's per-count specialists (inference/count_specialists_p: one model per exact team count 2-15,
//                     namespaces csp_nXX): fusion::models::count_specialists_p::forTeams. Its paired value networks
//                     are 2048 wide, so FUSION_MAX_WIDTH must be 2048, defined before the first inference header (the
//                     actor's); the weight headers' static_asserts refuse a narrower build.
//
// FUSION_TIER_P3XXL_VM selects the existing general P3 XXL policy; its paired V is never used by this carrier.
// kPBank names the bank. kPModelPrefix makes model ids unique across banks: the specialists' registry ids (n02 ...) are
// the XXL banks' ids too, so their model_id() is csp_n02 ... (the other banks' ids are unchanged).
#pragma once

#if (defined(FUSION_TIER_M) + defined(FUSION_TIER_L) + defined(FUSION_TIER_XL) + defined(FUSION_TIER_XXL) + defined(FUSION_TIER_PXL_VM) + defined(FUSION_TIER_PXXL_VM) + defined(FUSION_TIER_CSP) + defined(FUSION_TIER_P3XXL_VM)) > 1
#error "agent_v2: define exactly one FUSION_TIER_* (the opponent model bank)"
#endif

#if defined(FUSION_TIER_M)
#include "../models_m/registry.hpp"
#elif defined(FUSION_TIER_L)
#include "../models_l/registry.hpp"
#elif defined(FUSION_TIER_XL)
#include "../models_xl/registry.hpp"
#elif defined(FUSION_TIER_XXL)
#include "../models_xxl/registry.hpp"
#elif defined(FUSION_TIER_PXL_VM)
#include "../models_pxl_vm/registry.hpp"
#elif defined(FUSION_TIER_PXXL_VM)
#include "../models_pxxl_vm/registry.hpp"
#elif defined(FUSION_TIER_CSP)
#include "../count_specialists_p/registry.hpp"
#elif defined(FUSION_TIER_P3XXL_VM)
#include "../models_pf_p3xxl_vm/registry.hpp"
#else
#error "agent_v2 selects our opponent model registry with FUSION_TIER_*"
#endif

namespace agent_v2 {

// A P3 bank must use P3 features and conditional heads, not a cast of its weights into P2.
#if defined(FUSION_TIER_P3XXL_VM)
using OpponentPolicy = fusion::PolicyModelP3;
#else
using OpponentPolicy = fusion::PolicyModel;
#endif

#if defined(FUSION_TIER_CSP)
namespace p_bank = fusion::models::count_specialists_p;
inline constexpr const char* kPBank = "count_specialists_p";
inline constexpr const char* kPModelPrefix = "csp_";
static_assert(fusion::kMaxWidth >= p_bank::kRegistryRequiredMaxWidth, "FUSION_TIER_CSP needs FUSION_MAX_WIDTH 2048");
#else
namespace p_bank = fusion::models;
#if defined(FUSION_TIER_P3XXL_VM)
inline constexpr const char* kPModelPrefix = "p3xxl_";
#else
inline constexpr const char* kPModelPrefix = "";
#endif
#if defined(FUSION_TIER_M)
inline constexpr const char* kPBank = "models_m";
#elif defined(FUSION_TIER_L)
inline constexpr const char* kPBank = "models_l";
#elif defined(FUSION_TIER_XL)
inline constexpr const char* kPBank = "models_xl";
#elif defined(FUSION_TIER_XXL)
inline constexpr const char* kPBank = "models_xxl";
#elif defined(FUSION_TIER_P3XXL_VM)
inline constexpr const char* kPBank = "models_pf_p3xxl_vm";
#elif defined(FUSION_TIER_PXL_VM)
inline constexpr const char* kPBank = "models_pxl_vm";
#else
inline constexpr const char* kPBank = "models_pxxl_vm";
#endif
#endif

}  // namespace agent_v2
