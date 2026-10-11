// Value feature family v3: the 54 legacy value features (features.hpp
// encodeValue, family v2) followed by the combat, carrier, flag and rival
// extension (combat.hpp kValueExtSpec). Same network semantics as value.hpp:
// the output is a logit of the expected final pairwise outcome, and terminal
// states use the exact outcome instead of the network.
//
// The types keep the families apart: a v3 model only accepts
// ValueFeaturesV3. The exported header of a v3 model static_asserts its input
// width against kV3 (training/export_weights.py).
//
// ValueAccumulatorV3 keeps the first layer's int32 pre-activations like
// value.hpp's accumulator: a node that differs from its parent in a few
// features costs one first-layer column per changed feature plus the tail,
// bit-identical to the full pass. For a wide first layer (the 4XL shape,
// 8192 -> 512) the tail dominates the cost and no first-layer cache removes
// it. Nothing here allocates.
#ifndef FUSION_VALUE_V3_HPP
#define FUSION_VALUE_V3_HPP
#include <array>
#include <cstdint>

#include "combat.hpp"
#include "value.hpp"

namespace fusion {

inline constexpr int kValueFamilyV3 = 3;
inline constexpr int kV3 = kV + kVX;
using ValueFeaturesV3 = std::array<int16_t, kV3>;

struct ValueModelV3 { Net net; };

// `cs` must summarize the same (b, s, v); one summary serves every team.
inline void encodeValueV3(const Board& b, const State& s, const TurnView& v, const CombatSummary& cs, int team,
                          ValueFeaturesV3& out) {
    ValueFeatures legacy;
    encodeValue(b, s, v, team, legacy);
    for (int i = 0; i < kV; i++) out[i] = legacy[i];
    encodeValueExt(b, s, v, cs, team, out.data() + kV);
}

inline double valueLogit(const ValueModelV3& model, const ValueFeaturesV3& f) {
    int32_t x[kMaxWidth], acc[kMaxWidth];
    for (int i = 0; i < kV3; i++) x[i] = f[i];
    forwardRaw(model.net, x, acc);
    return toReal(model.net, acc[0]);
}

inline double value(const ValueModelV3& model, const ValueFeaturesV3& f) { return sigmoid(valueLogit(model, f)); }

inline double nodeValue(const ValueModelV3& model, const Board& b, const State& s, int team, const ValueFeaturesV3& f) {
    return terminal(b, s) ? pairwiseOutcome(s, team) : value(model, f);
}

struct ValueAccumulatorV3 {
    // Only refresh initializes the active first-layer rows. Every read is behind valid;
    // avoid zeroing a 64 KiB buffer at each search node in a 5XL build.
    std::array<int32_t, kMaxWidth> pre;
    ValueFeaturesV3 features{};
    bool valid = false;
};

inline void refresh(const ValueModelV3& model, const ValueFeaturesV3& f, ValueAccumulatorV3& a) {
    const Layer& L = model.net.layers[0];
    kernel::layer(L.w, L.b, f.data(), L.in, L.out, a.pre.data());
    a.features = f;
    a.valid = true;
}

inline void update(const ValueModelV3& model, const ValueFeaturesV3& f, ValueAccumulatorV3& a) {
    if (!a.valid) { refresh(model, f, a); return; }
    const Layer& L = model.net.layers[0];
    for (int i = 0; i < L.in; i++) {
        const int32_t delta = int32_t(f[i]) - int32_t(a.features[i]);
        if (delta == 0) continue;
        if (L.wT) kernel::axpy(a.pre.data(), L.wT + static_cast<long>(i) * L.out, delta, L.out);
        else for (int o = 0; o < L.out; o++) a.pre[o] += int32_t(L.w[o * L.in + i]) * delta;
    }
    a.features = f;
}

inline int32_t valueRaw(const ValueModelV3& model, const ValueAccumulatorV3& a) {
    const Layer& L = model.net.layers[0];
    if (model.net.count == 1) return a.pre[0];
    int32_t h[kMaxWidth], acc[kMaxWidth];
    for (int o = 0; o < L.out; o++) h[o] = activate(L, a.pre[o]);
    forwardFrom(model.net, 1, h, acc);
    return acc[0];
}

inline double valueLogit(const ValueModelV3& model, const ValueAccumulatorV3& a) { return toReal(model.net, valueRaw(model, a)); }

}  // namespace fusion
#endif  // FUSION_VALUE_V3_HPP
