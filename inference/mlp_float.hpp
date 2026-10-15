// FP32 multilayer perceptron for the CPU float critic of the Meowmind RL search diagnosis (task T0061, arm F).
//
// The checkpoint's float32 weights, unchanged: y = b + W x per layer, ReLU (y > 0 ? y : +0) on hidden layers, a linear
// output. Every product and every sum is one rounded float32 operation in a fixed order: each row is summed in 8
// interleaved partial sums (lane i % 8, p[k] = p[k] + w * x), the lanes are combined as ((0+4)+(2+6))+((1+5)+(3+7)) and
// the bias is added last. The AVX2 path performs exactly these operations (separate multiply and add intrinsics, no
// FMA in its target). Multiply-add contraction is switched off for these kernels (clang's fp pragma, GCC's
// optimize pragma). With no -ffast-math
// the native result is therefore bit-identical to bots/meow_rl/critics.py's float32 emulation, which the tests check;
// the error against the float64 forward pass of the training code is reported separately.
#ifndef FUSION_MLP_FLOAT_HPP
#define FUSION_MLP_FLOAT_HPP
#include <cstdint>

#include "simd.hpp"

namespace fusion::flt {

#ifndef FUSION_FLOAT_MAX_WIDTH
#define FUSION_FLOAT_MAX_WIDTH 1024
#endif
inline constexpr int kMaxWidth = FUSION_FLOAT_MAX_WIDTH;

struct Layer {
    int in = 0, out = 0;
    const float* w = nullptr;  // row-major [out][in]
    const float* b = nullptr;
    bool hidden = true;
};

struct Net {
    const Layer* layers = nullptr;
    int count = 0;
    int in() const { return layers[0].in; }
    int out() const { return layers[count - 1].out; }
};

using LayerFn = void (*)(const float* W, const float* b, const float* x, int in, int out, float* y);

// No multiply-add contraction in this file's kernels, whatever the compiler's default (GCC contracts on AArch64, and on
// x86-64 whenever FMA is enabled): every product and sum stays one rounded float32 operation.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

inline float combine(const float (&p)[8]) { return ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7])); }

inline void layerScalar(const float* W, const float* b, const float* x, int in, int out, float* y) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (int o = 0; o < out; o++) {
        const float* row = W + static_cast<long>(o) * in;
        float p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int i = 0;
        for (; i + 8 <= in; i += 8)
            for (int k = 0; k < 8; k++) p[k] = p[k] + row[i + k] * x[i + k];
        for (int k = 0; i < in; i++, k++) p[k] = p[k] + row[i] * x[i];
        y[o] = b[o] + combine(p);
    }
}

#if FUSION_X86 && FUSION_AVX2
__attribute__((target("avx2"))) inline void layerAvx2(const float* W, const float* b, const float* x, int in, int out,
                                                      float* y) {
    for (int o = 0; o < out; o++) {
        const float* row = W + static_cast<long>(o) * in;
        __m256 a = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= in; i += 8) a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(x + i)));
        alignas(32) float p[8];
        _mm256_store_ps(p, a);
        for (int k = 0; i < in; i++, k++) p[k] = p[k] + row[i] * x[i];
        y[o] = b[o] + combine(p);
    }
}
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

struct Kernels {
    LayerFn layer;
    const char* name;
};

inline Kernels choose() {
#if FUSION_X86 && FUSION_AVX2
    if (fusion::kernel::cpuHasAvx2()) return {layerAvx2, "avx2"};
#endif
    return {layerScalar, "scalar"};
}

inline const Kernels& active() {
    static const Kernels k = choose();
    return k;
}
inline const char* name() { return active().name; }

// x has net.in() floats; out receives net.out() floats.
inline void forward(const Net& net, const float* x, float* out, LayerFn kernel = nullptr) {
    LayerFn f = kernel ? kernel : active().layer;
    alignas(64) float bufA[kMaxWidth], bufB[kMaxWidth];
    for (int i = 0; i < net.layers[0].in; i++) bufA[i] = x[i];
    const float* src = bufA;
    for (int l = 0; l < net.count; l++) {
        const Layer& L = net.layers[l];
        float* dst = l + 1 == net.count ? out : (src == bufA ? bufB : bufA);
        f(L.w, L.b, src, L.in, L.out, dst);
        if (L.hidden)
            for (int o = 0; o < L.out; o++) dst[o] = dst[o] > 0.f ? dst[o] : 0.f;
        src = dst;
    }
}

}  // namespace fusion::flt
#endif  // FUSION_MLP_FLOAT_HPP
