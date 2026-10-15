// WideAct16 integer multilayer perceptron, bit-exact with fusion/quant_wide.py (task T0061).
//
// int16 weights, int64 biases and accumulators, hidden activations on 0..32767 (2^15 = 1.0) stored as int16: the
// higher-precision sibling of inference/mlp.hpp (reference16, activations 0..128, int32 accumulators). The exporter
// proves, for every input inside the feature spec:
//   * |w| <= 32767 and every layer input |x| <= 32767, so one int16 x int16 pair sum is below 2^31 (2 * 32767^2) and a
//     pairwise multiply-add into int32 lanes (pmaddwd, vmull) is exact before it is widened to int64;
//   * every accumulator, and every partial sum of one, stays inside +-2^52: int64 addition in any order gives the same
//     bits on every path, and the output accumulator converts to a double exactly.
// Activation: acc <= 0 gives 0; otherwise (acc + 2^(s-1)) >> s, clamped to 32767. The shift is applied only to positive
// values, so no implementation-defined shift of a negative number occurs.
//
// Paths, chosen once at startup like simd.hpp: avx2 (runtime check, target attribute), neon (AArch64) and scalar
// (everything else, or -DFUSION_SIMD=0). There is no SSE2-only path: without AVX2 an x86 CPU uses the scalar loop
// (reported by name(), so a timing never mistakes it for SIMD).
#ifndef FUSION_MLP_WIDE_HPP
#define FUSION_MLP_WIDE_HPP
#include <cmath>
#include <cstdint>

#include "simd.hpp"

namespace fusion::wide {

inline constexpr int32_t kActBits = 15;
inline constexpr int32_t kActMax = (1 << kActBits) - 1;

#ifndef FUSION_WIDE_MAX_WIDTH
#define FUSION_WIDE_MAX_WIDTH 1024
#endif
inline constexpr int kMaxWidth = FUSION_WIDE_MAX_WIDTH;

struct Layer {
    int in = 0, out = 0;
    const int16_t* w = nullptr;  // row-major [out][in]
    const int64_t* b = nullptr;
    int shift = 0;
    bool hidden = true;
};

struct Net {
    const Layer* layers = nullptr;
    int count = 0;
    int outExp = 0;  // real output = last accumulator / 2^outExp
    int in() const { return layers[0].in; }
    int out() const { return layers[count - 1].out; }
};

inline int16_t activate(int64_t acc, int s) {
    if (acc <= 0) return 0;
    int64_t r = s > 0 ? (acc + (int64_t(1) << (s - 1))) >> s : acc;
    return static_cast<int16_t>(r < kActMax ? r : kActMax);
}

using LayerFn = void (*)(const int16_t* W, const int64_t* b, const int16_t* x, int in, int out, int64_t* y);

inline void layerScalar(const int16_t* W, const int64_t* b, const int16_t* x, int in, int out, int64_t* y) {
    for (int o = 0; o < out; o++) {
        const int16_t* row = W + static_cast<long>(o) * in;
        int64_t s = b[o];
        for (int i = 0; i < in; i++) s += int64_t(int32_t(row[i]) * int32_t(x[i]));
        y[o] = s;
    }
}

#if FUSION_X86 && FUSION_AVX2
__attribute__((target("avx2"))) inline void layerAvx2(const int16_t* W, const int64_t* b, const int16_t* x, int in,
                                                      int out, int64_t* y) {
    for (int o = 0; o < out; o++) {
        const int16_t* row = W + static_cast<long>(o) * in;
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        int i = 0;
        for (; i + 16 <= in; i += 16) {
            // 8 exact int32 pair sums (|.| <= 2 * 32767^2 < 2^31), widened to int64 before they are added
            const __m256i p = _mm256_madd_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i)),
                                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)));
            a0 = _mm256_add_epi64(a0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p)));
            a1 = _mm256_add_epi64(a1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p, 1)));
        }
        alignas(32) int64_t lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_add_epi64(a0, a1));
        int64_t s = b[o] + lanes[0] + lanes[1] + lanes[2] + lanes[3];
        for (; i < in; i++) s += int64_t(int32_t(row[i]) * int32_t(x[i]));
        y[o] = s;
    }
}
#endif

#if FUSION_NEON
inline void layerNeon(const int16_t* W, const int64_t* b, const int16_t* x, int in, int out, int64_t* y) {
    for (int o = 0; o < out; o++) {
        const int16_t* row = W + static_cast<long>(o) * in;
        int64x2_t a0 = vdupq_n_s64(0), a1 = vdupq_n_s64(0);
        int i = 0;
        for (; i + 8 <= in; i += 8) {
            const int16x8_t w = vld1q_s16(row + i), v = vld1q_s16(x + i);
            // exact int32 products (|.| <= 32767^2), pairwise added into int64 lanes
            a0 = vpadalq_s32(a0, vmull_s16(vget_low_s16(w), vget_low_s16(v)));
            a1 = vpadalq_s32(a1, vmull_high_s16(w, v));
        }
        int64_t s = b[o] + vaddvq_s64(vaddq_s64(a0, a1));
        for (; i < in; i++) s += int64_t(int32_t(row[i]) * int32_t(x[i]));
        y[o] = s;
    }
}
#endif

struct Kernels {
    LayerFn layer;
    const char* name;
};

inline Kernels choose() {
#if FUSION_X86 && FUSION_AVX2
    if (fusion::kernel::cpuHasAvx2()) return {layerAvx2, "avx2"};
    return {layerScalar, "scalar"};
#elif FUSION_NEON
    return {layerNeon, "neon"};
#else
    return {layerScalar, "scalar"};
#endif
}

inline const Kernels& active() {
    static const Kernels k = choose();
    return k;
}
inline const char* name() { return active().name; }

// x has net.in() entries (each inside the proven input range); acc receives net.out() raw accumulators.
// Fixed stack buffers only. `kernel` selects a path explicitly (tests); nullptr uses the active one.
inline void forwardRaw(const Net& net, const int32_t* x, int64_t* acc, LayerFn kernel = nullptr) {
    LayerFn f = kernel ? kernel : active().layer;
    alignas(64) int16_t bufA[kMaxWidth], bufB[kMaxWidth];
    alignas(64) int64_t sum[kMaxWidth];
    for (int i = 0; i < net.layers[0].in; i++) bufA[i] = static_cast<int16_t>(x[i]);
    const int16_t* src = bufA;
    for (int l = 0; l < net.count; l++) {
        const Layer& L = net.layers[l];
        if (l + 1 == net.count) {
            f(L.w, L.b, src, L.in, L.out, acc);
            for (int o = 0; o < L.out && L.hidden; o++) acc[o] = activate(acc[o], L.shift);
            break;
        }
        int16_t* dst = src == bufA ? bufB : bufA;
        f(L.w, L.b, src, L.in, L.out, sum);
        for (int o = 0; o < L.out; o++) dst[o] = activate(sum[o], L.shift);
        src = dst;
    }
}

inline double toReal(const Net& net, int64_t acc) { return std::ldexp(static_cast<double>(acc), -net.outExp); }

}  // namespace fusion::wide
#endif  // FUSION_MLP_WIDE_HPP
