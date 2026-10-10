// Integer multilayer perceptron, bit-exact with fusion/quant.py.
//
// Weights are int16, accumulators int32, hidden activations clipped ReLU on
// the fixed-point scale kActOne (= 1.0). The exporter proves that no
// accumulator can leave [-2^30, 2^30] for any input inside the feature spec,
// so int32 never overflows. Only the final layer's accumulator is turned into
// a double, by an exact power-of-two scaling.
//
// Layer inputs travel as int16 (features are clipped small integers, hidden
// activations 0..kActOne), so each layer is one call of a SIMD multiply-add
// kernel (simd.hpp). Integer sums are exact in any order: every path gives the
// same bits as the scalar loop (tools/cpp/kernel_check.cpp checks each path).
#ifndef FUSION_MLP_HPP
#define FUSION_MLP_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "simd.hpp"

namespace fusion {

// Widest layer input or output the fixed stack buffers (and the value
// accumulator) hold. 256 covers the default architectures; a larger model is
// built with -DFUSION_MAX_WIDTH=<n> (tools/host/capacity_bench.py measures what a
// size costs). Every exported header static_asserts its layers against it.
#ifndef FUSION_MAX_WIDTH
#if defined(FUSION_TIER_XXL)
#define FUSION_MAX_WIDTH 2048
#elif defined(FUSION_TIER_XL) || defined(FUSION_TIER_PXXL_VM)
#define FUSION_MAX_WIDTH 1024
#elif defined(FUSION_TIER_L) || defined(FUSION_TIER_PXL_VM)
#define FUSION_MAX_WIDTH 512
#else
#define FUSION_MAX_WIDTH 256
#endif
#endif

inline constexpr int32_t kActOne = 128;
inline constexpr int kMaxWidth = FUSION_MAX_WIDTH;

struct Layer {
    int in = 0, out = 0;
    const int16_t* w = nullptr;  // row-major [out][in]
    const int32_t* b = nullptr;
    int shift = 0;               // hidden: activation = clamp(round(acc / 2^shift), 0, kActOne)
    bool hidden = true;          // false only for the last layer
    const int16_t* wT = nullptr; // optional column-major copy [in][out] (value first layer: incremental updates)
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

// Layers first..count-1 on `x` (the input of layer `first`, every value inside
// int16, as features and activations are); acc receives the last layer's raw
// accumulators. Fixed stack buffers only, no allocation.
inline void forwardFrom(const Net& net, int first, const int32_t* x, int32_t* acc) {
    alignas(64) int16_t bufA[kMaxWidth], bufB[kMaxWidth];
    alignas(64) int32_t sum[kMaxWidth];
    const int inputs = net.layers[first].in;
    for (int i = 0; i < inputs; i++) bufA[i] = static_cast<int16_t>(x[i]);
    const int16_t* src = bufA;
    for (int l = first; l < net.count; l++) {
        const Layer& L = net.layers[l];
        if (l + 1 == net.count) {
            kernel::layer(L.w, L.b, src, L.in, L.out, acc);
            for (int o = 0; o < L.out; o++) acc[o] = activate(L, acc[o]);
            break;
        }
        int16_t* dst = src == bufA ? bufB : bufA;
        kernel::layer(L.w, L.b, src, L.in, L.out, sum);
        for (int o = 0; o < L.out; o++) dst[o] = static_cast<int16_t>(activate(L, sum[o]));
        src = dst;
    }
}

// x has net.in() entries; acc receives net.out() raw accumulators.
inline void forwardRaw(const Net& net, const int32_t* x, int32_t* acc) { forwardFrom(net, 0, x, acc); }

inline double toReal(const Net& net, int32_t acc) { return std::ldexp(static_cast<double>(acc), -net.outExp); }

}  // namespace fusion
#endif  // FUSION_MLP_HPP
