// Feature-family dispatch for the model tools (infer_check, runtime_check,
// value_bench): one source compiles against p2/p3 and v2/v3 models.
//
//   -DFUSION_POLICY_FAMILY=3   policy family p3 (inference/policy_p3.hpp), default 2
//   -DFUSION_VALUE_FAMILY=3    value family v3 (inference/value_v3.hpp), default 2
//
// fusion.native.model_tool reads the families from the exported header's
// preamble and passes these defines; a mismatch fails the header's own
// static_asserts or the type check below at compile time.
#ifndef FUSION_TOOLS_FAMILY_HPP
#define FUSION_TOOLS_FAMILY_HPP
#include "../../inference/combat.hpp"
#include "../../inference/policy_p3.hpp"
#include "../../inference/value_v3.hpp"

#ifndef FUSION_POLICY_FAMILY
#define FUSION_POLICY_FAMILY 2
#endif
#ifndef FUSION_VALUE_FAMILY
#define FUSION_VALUE_FAMILY 2
#endif

namespace fusion::tools {

#if FUSION_POLICY_FAMILY == 3
using PFeatures = PolicyFeaturesP3;
using PModel = PolicyModelP3;
inline void encodeP(const Board& b, const State& s, const TurnView& v, const CombatSummary& cs, int team, PFeatures& f) {
    encodePolicyP3(b, s, v, cs, team, f);
}
#elif FUSION_POLICY_FAMILY == 2
using PFeatures = PolicyFeatures;
using PModel = PolicyModel;
inline void encodeP(const Board& b, const State& s, const TurnView& v, const CombatSummary&, int team, PFeatures& f) {
    encodePolicy(b, s, v, team, f);
}
#else
#error "FUSION_POLICY_FAMILY must be 2 or 3"
#endif

#if FUSION_VALUE_FAMILY == 3
using VFeatures = ValueFeaturesV3;
using VModel = ValueModelV3;
using VAccumulator = ValueAccumulatorV3;
inline constexpr int kVIn = kV3;
inline void encodeV(const Board& b, const State& s, const TurnView& v, const CombatSummary& cs, int team, VFeatures& f) {
    encodeValueV3(b, s, v, cs, team, f);
}
#elif FUSION_VALUE_FAMILY == 2
using VFeatures = ValueFeatures;
using VModel = ValueModel;
using VAccumulator = ValueAccumulator;
inline constexpr int kVIn = kV;
inline void encodeV(const Board& b, const State& s, const TurnView& v, const CombatSummary&, int team, VFeatures& f) {
    encodeValue(b, s, v, team, f);
}
#else
#error "FUSION_VALUE_FAMILY must be 2 or 3"
#endif

}  // namespace fusion::tools
#endif  // FUSION_TOOLS_FAMILY_HPP
