// Experimental low-bit policy layers (task T0030, docs/P_MIXED_PRECISION.md), bit-exact with fusion/precision.py.
//
// A layer stores int16, int8 or packed int4 weights (two per byte, 6-bit block scales), an int32 bias in
// accumulator units, and either a fixed-point requantizer (hidden layers: activation 0..kActOne) or a float64
// scale per output (last layer). Inputs travel as int16, zero-padded to a whole number of 32-column blocks:
// first layers read the extractor's signed integers, hidden layers the clipped-ReLU activations 0..128.
//
//   acc[o] = b[o] + sum_blocks d[o][blk] * sum_{i in blk} q[o][i] * x[i]          int32, |acc| <= 2^30 (proven)
//   hidden: h = clamp((int64(acc) * mult[o] + 2^(r-1)) >> r, 0, kActOne)
//   last:   real = double(acc) * outScale[o]
//
// Packed weights are decoded block by block inside the kernel (sign extension or nibble unpacking into
// registers); no int16 copy of a low-bit matrix is ever made. Every partial sum is bounded by the row bound,
// so all kernel paths give identical bits. Paths, chosen once at startup (as inference/simd.hpp):
//   avx2 (runtime check, target attribute), sse2 (every x86-64), neon (every AArch64), scalar (fallback, or
//   -DFUSION_SIMD=0). This header changes nothing in the INT16 path; nothing includes it by default.
#ifndef FUSION_POLICY_PRECISION_LOWBIT_HPP
#define FUSION_POLICY_PRECISION_LOWBIT_HPP
#include <algorithm>
#include <cstdint>

#include "../mlp.hpp"

namespace fusion::lowbit {

inline constexpr int kBlock = 32;
inline constexpr int kFormatVersion = 1;  // fusion/precision.py FORMAT_VERSION "pprec-1"

struct Layer {
    int bits = 8;                   // 16, 8 or 4
    int in = 0, out = 0, inPad = 0; // inPad = in rounded up to kBlock
    const void* w = nullptr;        // int16/int8 [out][inPad], or uint8 [out][inPad/2] (int4: block-packed nibbles)
    const uint8_t* d = nullptr;     // int4 only: [out][inPad/kBlock] block scales 1..63
    const int32_t* b = nullptr;     // bias in accumulator units
    bool hidden = true;
    const int32_t* mult = nullptr;  // hidden: requantization multipliers
    int rshift = 0;                 //         and shift
    const double* outScale = nullptr;  // last: real = acc * outScale[o]
};

struct Net {
    const Layer* layers = nullptr;
    int count = 0;
    int in() const { return layers[0].in; }
    int out() const { return layers[count - 1].out; }
};

// One row's accumulator, the independent reference: decode every weight, multiply, add. Used by the scalar
// path and by tools/cpp/precision_check.cpp as the oracle the vector paths are compared with.
inline int32_t rowReference(const Layer& L, int o, const int16_t* x) {
    int64_t s = L.b[o];
    const int blocks = L.inPad / kBlock;
    for (int blk = 0; blk < blocks; blk++) {
        int64_t part = 0;
        for (int j = 0; j < kBlock; j++) {
            const int i = blk * kBlock + j;
            int q;
            if (L.bits == 16) q = static_cast<const int16_t*>(L.w)[static_cast<long>(o) * L.inPad + i];
            else if (L.bits == 8) q = static_cast<const int8_t*>(L.w)[static_cast<long>(o) * L.inPad + i];
            else {
                const uint8_t byte = static_cast<const uint8_t*>(L.w)[static_cast<long>(o) * (L.inPad / 2) + blk * (kBlock / 2) + (j % (kBlock / 2))];
                q = (j < kBlock / 2 ? (byte & 15) : (byte >> 4)) - 8;
            }
            part += int64_t(q) * x[i];
        }
        s += (L.bits == 4 ? int64_t(L.d[static_cast<long>(o) * blocks + blk]) : 1) * part;
    }
    return static_cast<int32_t>(s);
}

using RowsFn = void (*)(const Layer& L, const int16_t* x, int32_t* acc);

inline void rowsScalar(const Layer& L, const int16_t* x, int32_t* acc) {
    for (int o = 0; o < L.out; o++) acc[o] = rowReference(L, o, x);
}

#if FUSION_X86
// Sign-extend 16 int8 lanes to two int16 vectors (SSE2: no pmovsx).
inline void widen8Sse2(__m128i v, __m128i& lo, __m128i& hi) {
    const __m128i sign = _mm_cmplt_epi8(v, _mm_setzero_si128());
    lo = _mm_unpacklo_epi8(v, sign);
    hi = _mm_unpackhi_epi8(v, sign);
}

inline void rowsSse2(const Layer& L, const int16_t* x, int32_t* acc) {
    const int blocks = L.inPad / kBlock;
    const __m128i low4 = _mm_set1_epi8(15), eight = _mm_set1_epi8(8);
    for (int o = 0; o < L.out; o++) {
        int32_t s = L.b[o];
        for (int blk = 0; blk < blocks; blk++) {
            const int16_t* xb = x + blk * kBlock;
            __m128i a = _mm_setzero_si128();
            if (L.bits == 16) {
                const int16_t* w = static_cast<const int16_t*>(L.w) + static_cast<long>(o) * L.inPad + blk * kBlock;
                for (int j = 0; j < kBlock; j += 8)
                    a = _mm_add_epi32(a, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + j)),
                                                        _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb + j))));
            } else {
                __m128i v0, v1;  // 32 weights as int8
                if (L.bits == 8) {
                    const int8_t* w = static_cast<const int8_t*>(L.w) + static_cast<long>(o) * L.inPad + blk * kBlock;
                    v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
                    v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16));
                } else {
                    const uint8_t* p = static_cast<const uint8_t*>(L.w) + static_cast<long>(o) * (L.inPad / 2) + blk * (kBlock / 2);
                    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
                    v0 = _mm_sub_epi8(_mm_and_si128(bytes, low4), eight);
                    v1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), eight);
                }
                __m128i w0, w1, w2, w3;
                widen8Sse2(v0, w0, w1);
                widen8Sse2(v1, w2, w3);
                a = _mm_add_epi32(a, _mm_madd_epi16(w0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb))));
                a = _mm_add_epi32(a, _mm_madd_epi16(w1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb + 8))));
                a = _mm_add_epi32(a, _mm_madd_epi16(w2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb + 16))));
                a = _mm_add_epi32(a, _mm_madd_epi16(w3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb + 24))));
            }
            const int32_t part = kernel::hsum128(a);
            s += L.bits == 4 ? int32_t(L.d[static_cast<long>(o) * blocks + blk]) * part : part;
        }
        acc[o] = s;
    }
}

#if FUSION_AVX2
__attribute__((target("avx2"))) inline void rowsAvx2(const Layer& L, const int16_t* x, int32_t* acc) {
    const int blocks = L.inPad / kBlock;
    const __m128i low4 = _mm_set1_epi8(15), eight = _mm_set1_epi8(8);
    for (int o = 0; o < L.out; o++) {
        int32_t s = L.b[o];
        for (int blk = 0; blk < blocks; blk++) {
            const int16_t* xb = x + blk * kBlock;
            const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb));
            const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb + 16));
            __m256i w0, w1;
            if (L.bits == 16) {
                const int16_t* w = static_cast<const int16_t*>(L.w) + static_cast<long>(o) * L.inPad + blk * kBlock;
                w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
                w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
            } else if (L.bits == 8) {
                const int8_t* w = static_cast<const int8_t*>(L.w) + static_cast<long>(o) * L.inPad + blk * kBlock;
                w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
                w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16)));
            } else {
                const uint8_t* p = static_cast<const uint8_t*>(L.w) + static_cast<long>(o) * (L.inPad / 2) + blk * (kBlock / 2);
                const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
                w0 = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_and_si128(bytes, low4), eight));
                w1 = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(bytes, 4), low4), eight));
            }
            const __m256i a = _mm256_add_epi32(_mm256_madd_epi16(w0, x0), _mm256_madd_epi16(w1, x1));
            const int32_t part = kernel::hsum128(_mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)));
            s += L.bits == 4 ? int32_t(L.d[static_cast<long>(o) * blocks + blk]) * part : part;
        }
        acc[o] = s;
    }
}
#endif  // FUSION_AVX2
#endif  // FUSION_X86

#if FUSION_NEON
inline void rowsNeon(const Layer& L, const int16_t* x, int32_t* acc) {
    const int blocks = L.inPad / kBlock;
    const uint8x16_t low4 = vdupq_n_u8(15);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int o = 0; o < L.out; o++) {
        int32_t s = L.b[o];
        for (int blk = 0; blk < blocks; blk++) {
            const int16_t* xb = x + blk * kBlock;
            int16x8_t w[4];
            if (L.bits == 16) {
                const int16_t* p = static_cast<const int16_t*>(L.w) + static_cast<long>(o) * L.inPad + blk * kBlock;
                for (int k = 0; k < 4; k++) w[k] = vld1q_s16(p + 8 * k);
            } else {
                int8x16_t v0, v1;
                if (L.bits == 8) {
                    const int8_t* p = static_cast<const int8_t*>(L.w) + static_cast<long>(o) * L.inPad + blk * kBlock;
                    v0 = vld1q_s8(p);
                    v1 = vld1q_s8(p + 16);
                } else {
                    const uint8_t* p = static_cast<const uint8_t*>(L.w) + static_cast<long>(o) * (L.inPad / 2) + blk * (kBlock / 2);
                    const uint8x16_t bytes = vld1q_u8(p);
                    v0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, low4)), eight);
                    v1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), eight);
                }
                w[0] = vmovl_s8(vget_low_s8(v0));
                w[1] = vmovl_high_s8(v0);
                w[2] = vmovl_s8(vget_low_s8(v1));
                w[3] = vmovl_high_s8(v1);
            }
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            for (int k = 0; k < 4; k++) {
                const int16x8_t v = vld1q_s16(xb + 8 * k);
                a0 = vmlal_s16(a0, vget_low_s16(w[k]), vget_low_s16(v));
                a1 = vmlal_high_s16(a1, w[k], v);
            }
            const int32_t part = vaddvq_s32(vaddq_s32(a0, a1));
            s += L.bits == 4 ? int32_t(L.d[static_cast<long>(o) * blocks + blk]) * part : part;
        }
        acc[o] = s;
    }
}
#endif  // FUSION_NEON

struct Kernels {
    RowsFn rows;
    const char* name;
};

inline Kernels choose() {
#if FUSION_X86
#if FUSION_AVX2
    if (kernel::cpuHasAvx2()) return {rowsAvx2, "avx2"};
#endif
    return {rowsSse2, "sse2"};
#elif FUSION_NEON
    return {rowsNeon, "neon"};
#else
    return {rowsScalar, "scalar"};
#endif
}

inline const Kernels& active() {
    static const Kernels k = choose();
    return k;
}

inline const char* name() { return active().name; }

inline int32_t requantize(const Layer& L, int o, int32_t acc) {
    const int64_t v = int64_t(acc) * L.mult[o] + (L.rshift > 0 ? (int64_t(1) << (L.rshift - 1)) : 0);
    return static_cast<int32_t>(std::clamp<int64_t>(v >> L.rshift, 0, kActOne));
}

// Every layer on `x` (net.in() raw integers); acc receives the last layer's accumulators. `rows` picks a
// kernel path explicitly (tests), else the active one. Fixed stack buffers; padding columns are zero.
template <int Width = kMaxWidth>
inline void forwardAcc(const Net& net, const int32_t* x, int32_t* acc, RowsFn rows = nullptr) {
    static_assert(Width % kBlock == 0, "buffers hold whole blocks");
    alignas(64) int16_t bufA[Width], bufB[Width];
    alignas(64) int32_t sum[Width];
    if (!rows) rows = active().rows;
    const Layer& first = net.layers[0];
    for (int i = 0; i < first.inPad; i++) bufA[i] = i < first.in ? static_cast<int16_t>(x[i]) : int16_t(0);
    const int16_t* src = bufA;
    for (int l = 0; l < net.count; l++) {
        const Layer& L = net.layers[l];
        if (l + 1 == net.count) {
            rows(L, src, acc);
            return;
        }
        int16_t* dst = src == bufA ? bufB : bufA;
        rows(L, src, sum);
        const int nextPad = net.layers[l + 1].inPad;
        for (int o = 0; o < nextPad; o++) dst[o] = o < L.out ? static_cast<int16_t>(requantize(L, o, sum[o])) : int16_t(0);
        src = dst;
    }
}

// The product is rounded on its own (a volatile store): a compiler that fuses it into the caller's next addition
// (an FMA, allowed by -ffp-contract) would round once instead of twice and differ from the emulation in the last bit.
inline double toReal(const Net& net, int o, int32_t acc) {
    volatile double real = static_cast<double>(acc) * net.layers[net.count - 1].outScale[o];
    return real;
}

}  // namespace fusion::lowbit
#endif  // FUSION_POLICY_PRECISION_LOWBIT_HPP
