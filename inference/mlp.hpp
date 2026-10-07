// Integer multilayer perceptron, bit-exact with fusion/quant.py.
//
// Weights are int16, accumulators int32, hidden activations clipped ReLU on
// the fixed-point scale kActOne (= 1.0). The exporter proves that no
// accumulator can leave [-2^30, 2^30] for any input inside the feature spec,
// so int32 never overflows. Only the final layer's accumulator is turned into
// a double, by an exact power-of-two scaling.
#ifndef FUSION_MLP_HPP
#define FUSION_MLP_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fusion {

inline constexpr int32_t kActOne = 128;
inline constexpr int kMaxWidth = 256;

struct Layer {
    int in = 0, out = 0;
    const int16_t* w = nullptr;  // row-major [out][in]
    const int32_t* b = nullptr;
    int shift = 0;               // hidden: activation = clamp(round(acc / 2^shift), 0, kActOne)
    bool hidden = true;          // false only for the last layer
};

struct Net {
    const Layer* layers = nullptr;
    int count = 0;
    int outExp = 0;  // real output = last accumulator / 2^outExp
    int in() const { return layers[0].in; }
    int out() const { return layers[count - 1].out; }
};

inline int32_t roundShift(int32_t v, int s) { return s > 0 ? (v + (int32_t(1) << (s - 1))) >> s : v; }

// One layer's activation of a raw accumulator.
inline int32_t activate(const Layer& L, int32_t acc) {
    return L.hidden ? std::clamp(roundShift(acc, L.shift), int32_t(0), kActOne) : acc;
}

// Layers first..count-1 on `x` (the input of layer `first`); acc receives the
// last layer's raw accumulators. Fixed stack buffers only, no allocation.
inline void forwardFrom(const Net& net, int first, const int32_t* x, int32_t* acc) {
    int32_t bufA[kMaxWidth], bufB[kMaxWidth];
    const int32_t* src = x;
    for (int l = first; l < net.count; l++) {
        const Layer& L = net.layers[l];
        int32_t* dst = l + 1 == net.count ? acc : (l % 2 == 0 ? bufA : bufB);
        for (int o = 0; o < L.out; o++) {
            const int16_t* row = L.w + o * L.in;
            int32_t s = L.b[o];
            for (int i = 0; i < L.in; i++) s += int32_t(row[i]) * src[i];
            dst[o] = activate(L, s);
        }
        src = dst;
    }
}

// x has net.in() entries; acc receives net.out() raw accumulators.
inline void forwardRaw(const Net& net, const int32_t* x, int32_t* acc) { forwardFrom(net, 0, x, acc); }

inline double toReal(const Net& net, int32_t acc) { return std::ldexp(static_cast<double>(acc), -net.outExp); }

}  // namespace fusion
#endif  // FUSION_MLP_HPP
