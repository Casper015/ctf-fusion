// Board value V(s, team): expected pairwise outcome of `team` against every
// other team at the end of the match (win 1, draw 0.5, loss 0), from the
// public state. The network's final output is a logit; value() is its sigmoid.
#ifndef FUSION_VALUE_HPP
#define FUSION_VALUE_HPP
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

inline double value(const ValueModel& model, const ValueFeatures& f) {
    return 1.0 / (1.0 + std::exp(-valueLogit(model, f)));
}

}  // namespace fusion
#endif  // FUSION_VALUE_HPP
