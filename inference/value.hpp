// Board value V(s, team): expected pairwise outcome of `team` against every
// other team at the end of the match (win 1, draw 0.5, loss 0), from the
// public state. The network's final output is a logit; value() is its sigmoid.
//
// Two ways to evaluate:
//   value(model, features)                     full forward pass
//   ValueAccumulator: refresh once, then update(model, features) per node
// The accumulator keeps the first layer's int32 pre-activations (the
// "128-wide accumulator"). A search node that differs from its parent in a few
// features (one unit moved, one hit) costs one column update per changed
// feature plus the small tail (128 -> 16 -> 1), instead of the full 54 x 128
// layer. Integer arithmetic makes both paths identical bit for bit
// (tools/runtime_check.cpp proves it). Nothing here allocates.
#ifndef FUSION_VALUE_HPP
#define FUSION_VALUE_HPP
#include <array>
#include <cmath>
#include <cstdint>

#include "features.hpp"
#include "mlp.hpp"

namespace fusion {

struct ValueModel { Net net; };

inline double valueLogit(const ValueModel& model, const ValueFeatures& f) {
    int32_t x[kMaxWidth], acc[kMaxWidth];
    for (int i = 0; i < kV; i++) x[i] = f[i];
    forwardRaw(model.net, x, acc);
    return toReal(model.net, acc[0]);
}

inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

inline double value(const ValueModel& model, const ValueFeatures& f) { return sigmoid(valueLogit(model, f)); }

// A state after the last settlement (the post-game block, turn > turns) has
// nothing left to estimate: its value is the exact pairwise outcome, the
// quantity V is trained on (fusion/labels.py pairwise_outcome). Search code
// must use this at terminal nodes instead of the network.
inline bool terminal(const Board& b, const State& s) { return s.turn > b.turns; }

inline double pairwiseOutcome(const State& s, int team) {
    const int n = static_cast<int>(s.score.size());
    if (n < 2) return 0.5;
    double wins = 0;
    for (int t = 0; t < n; t++)
        if (t != team) wins += s.score[team] > s.score[t] ? 1.0 : s.score[team] == s.score[t] ? 0.5 : 0.0;
    return wins / (n - 1);
}

// The value a search node should use: exact at terminal states, else V.
inline double nodeValue(const ValueModel& model, const Board& b, const State& s, int team, const ValueFeatures& f) {
    return terminal(b, s) ? pairwiseOutcome(s, team) : value(model, f);
}

struct ValueAccumulator {
    std::array<int32_t, kMaxWidth> pre{};  // first-layer accumulators (before activation)
    ValueFeatures features{};              // the features `pre` reflects
    bool valid = false;
};

// Full first layer for `f`.
inline void refresh(const ValueModel& model, const ValueFeatures& f, ValueAccumulator& a) {
    const Layer& L = model.net.layers[0];
    kernel::layer(L.w, L.b, f.data(), L.in, L.out, a.pre.data());
    a.features = f;
    a.valid = true;
}

// Move the accumulator to `f`, touching only the features that changed: one
// column of the first layer per changed feature, contiguous when the export
// carries the transposed copy (wT), strided otherwise. Same bits either way.
inline void update(const ValueModel& model, const ValueFeatures& f, ValueAccumulator& a) {
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

// Remaining layers from the accumulator; same result as valueLogit(model, a.features).
inline double valueLogit(const ValueModel& model, const ValueAccumulator& a) {
    const Layer& L = model.net.layers[0];
    int32_t h[kMaxWidth], acc[kMaxWidth];
    for (int o = 0; o < L.out; o++) h[o] = activate(L, a.pre[o]);
    if (model.net.count == 1) return toReal(model.net, a.pre[0]);
    forwardFrom(model.net, 1, h, acc);
    return toReal(model.net, acc[0]);
}

inline int32_t valueRaw(const ValueModel& model, const ValueAccumulator& a) {
    const Layer& L = model.net.layers[0];
    int32_t h[kMaxWidth], acc[kMaxWidth];
    for (int o = 0; o < L.out; o++) h[o] = activate(L, a.pre[o]);
    if (model.net.count == 1) return a.pre[0];
    forwardFrom(model.net, 1, h, acc);
    return acc[0];
}

}  // namespace fusion
#endif  // FUSION_VALUE_HPP
