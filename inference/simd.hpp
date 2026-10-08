// Integer multiply-add kernels for the network layers, bit-exact with the
// scalar loop on every path.
//
// A dense layer is y[o] = b[o] + sum_i W[o][i] * x[i] with int16 weights,
// int16 inputs (features are clipped small integers, hidden activations are
// 0..kActOne) and int32 sums. The exporter proves that no sum can leave
// [-2^30, 2^30] for any input inside the feature spec, and every partial sum
// is bounded by the same total, so integer addition in any order gives the
// identical result: vector lanes, pair-wise multiply-add and different
// summation orders cannot change a single bit.
//
// Paths, chosen once at startup:
//   avx2    x86-64 with AVX2 (runtime CPU check; compiled through a target
//           attribute, so the bot needs no extra compiler flags)
//   sse2    every x86-64 CPU (pmaddwd: eight 16-bit products, pair-summed)
//   neon    every AArch64 CPU (widening multiply-accumulate)
//   scalar  anything else, or -DFUSION_SIMD=0 to force it (A/B timing)
//
// The intrinsics are the vendors' standard ones (Intel <immintrin.h>, Arm
// <arm_neon.h>); nothing here allocates.
#ifndef FUSION_SIMD_HPP
#define FUSION_SIMD_HPP
#include <cstdint>

#ifndef FUSION_SIMD
#define FUSION_SIMD 1
#endif
#if FUSION_SIMD && (defined(__x86_64__) || defined(_M_X64))
#define FUSION_X86 1
#include <immintrin.h>
#if defined(__GNUC__) && !defined(FUSION_NO_AVX2)
#define FUSION_AVX2 1
#endif
#elif FUSION_SIMD && (defined(__aarch64__) || defined(_M_ARM64))
#define FUSION_NEON 1
#include <arm_neon.h>
#endif

namespace fusion::kernel {

// y[o] = b[o] + W[o] . x for o < out; W row-major [out][in].
using LayerFn = void (*)(const int16_t* W, const int32_t* b, const int16_t* x, int in, int out, int32_t* y);
// acc[o] += col[o] * scale for o < n (a column of the transposed first layer times a feature change).
using AxpyFn = void (*)(int32_t* acc, const int16_t* col, int32_t scale, int n);

inline void layerScalar(const int16_t* W, const int32_t* b, const int16_t* x, int in, int out, int32_t* y) {
    for (int o = 0; o < out; o++) {
        const int16_t* row = W + static_cast<long>(o) * in;
        int32_t s = b[o];
        for (int i = 0; i < in; i++) s += int32_t(row[i]) * x[i];
        y[o] = s;
    }
}

inline void axpyScalar(int32_t* acc, const int16_t* col, int32_t scale, int n) {
    for (int o = 0; o < n; o++) acc[o] += int32_t(col[o]) * scale;
}

#if FUSION_X86
inline int32_t hsum128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}

inline void layerSse2(const int16_t* W, const int32_t* b, const int16_t* x, int in, int out, int32_t* y) {
    for (int o = 0; o < out; o++) {
        const int16_t* row = W + static_cast<long>(o) * in;
        __m128i a0 = _mm_setzero_si128(), a1 = _mm_setzero_si128();
        int i = 0;
        for (; i + 16 <= in; i += 16) {
            a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i)),
                                                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i))));
            a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i + 8)),
                                                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + 8))));
        }
        for (; i + 8 <= in; i += 8)
            a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i)),
                                                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i))));
        int32_t s = b[o] + hsum128(_mm_add_epi32(a0, a1));
        for (; i < in; i++) s += int32_t(row[i]) * x[i];
        y[o] = s;
    }
}

inline void axpySse2(int32_t* acc, const int16_t* col, int32_t scale, int n) {
    // scale is a feature change, |scale| <= 32767: exact 32-bit products from the low and high halves.
    const __m128i d = _mm_set1_epi16(static_cast<int16_t>(scale));
    int o = 0;
    for (; o + 8 <= n; o += 8) {
        const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(col + o));
        const __m128i lo = _mm_mullo_epi16(c, d), hi = _mm_mulhi_epi16(c, d);
        __m128i* p = reinterpret_cast<__m128i*>(acc + o);
        _mm_storeu_si128(p, _mm_add_epi32(_mm_loadu_si128(p), _mm_unpacklo_epi16(lo, hi)));
        _mm_storeu_si128(p + 1, _mm_add_epi32(_mm_loadu_si128(p + 1), _mm_unpackhi_epi16(lo, hi)));
    }
    for (; o < n; o++) acc[o] += int32_t(col[o]) * scale;
}

#if FUSION_AVX2
__attribute__((target("avx2"))) inline void layerAvx2(const int16_t* W, const int32_t* b, const int16_t* x, int in,
                                                      int out, int32_t* y) {
    for (int o = 0; o < out; o++) {
        const int16_t* row = W + static_cast<long>(o) * in;
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        int i = 0;
        for (; i + 32 <= in; i += 32) {
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i)),
                                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i))));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i + 16)),
                                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + 16))));
        }
        for (; i + 16 <= in; i += 16)
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i)),
                                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i))));
        __m256i a = _mm256_add_epi32(a0, a1);
        __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        for (; i + 8 <= in; i += 8)
            h = _mm_add_epi32(h, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + i)),
                                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i))));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, _MM_SHUFFLE(1, 0, 3, 2)));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, _MM_SHUFFLE(2, 3, 0, 1)));
        int32_t s = b[o] + _mm_cvtsi128_si32(h);
        for (; i < in; i++) s += int32_t(row[i]) * x[i];
        y[o] = s;
    }
}

__attribute__((target("avx2"))) inline void axpyAvx2(int32_t* acc, const int16_t* col, int32_t scale, int n) {
    const __m256i d = _mm256_set1_epi32(scale);
    int o = 0;
    for (; o + 8 <= n; o += 8) {
        const __m256i c = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(col + o)));
        __m256i* p = reinterpret_cast<__m256i*>(acc + o);
        _mm256_storeu_si256(p, _mm256_add_epi32(_mm256_loadu_si256(p), _mm256_mullo_epi32(c, d)));
    }
    for (; o < n; o++) acc[o] += int32_t(col[o]) * scale;
}

inline bool cpuHasAvx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}
#endif  // FUSION_AVX2
#endif  // FUSION_X86

#if FUSION_NEON
inline void layerNeon(const int16_t* W, const int32_t* b, const int16_t* x, int in, int out, int32_t* y) {
    for (int o = 0; o < out; o++) {
        const int16_t* row = W + static_cast<long>(o) * in;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int i = 0;
        for (; i + 8 <= in; i += 8) {
            const int16x8_t w = vld1q_s16(row + i), v = vld1q_s16(x + i);
            a0 = vmlal_s16(a0, vget_low_s16(w), vget_low_s16(v));
            a1 = vmlal_high_s16(a1, w, v);
        }
        int32_t s = b[o] + vaddvq_s32(vaddq_s32(a0, a1));
        for (; i < in; i++) s += int32_t(row[i]) * x[i];
        y[o] = s;
    }
}

inline void axpyNeon(int32_t* acc, const int16_t* col, int32_t scale, int n) {
    const int16_t d = static_cast<int16_t>(scale);  // a feature change, |scale| <= 32767
    int o = 0;
    for (; o + 8 <= n; o += 8) {
        const int16x8_t c = vld1q_s16(col + o);
        vst1q_s32(acc + o, vmlal_n_s16(vld1q_s32(acc + o), vget_low_s16(c), d));
        vst1q_s32(acc + o + 4, vmlal_high_n_s16(vld1q_s32(acc + o + 4), c, d));
    }
    for (; o < n; o++) acc[o] += int32_t(col[o]) * scale;
}
#endif

struct Kernels {
    LayerFn layer;
    AxpyFn axpy;
    const char* name;
};

inline Kernels choose() {
#if FUSION_X86
#if FUSION_AVX2
    if (cpuHasAvx2()) return {layerAvx2, axpyAvx2, "avx2"};
#endif
    return {layerSse2, axpySse2, "sse2"};
#elif FUSION_NEON
    return {layerNeon, axpyNeon, "neon"};
#else
    return {layerScalar, axpyScalar, "scalar"};
#endif
}

// Chosen once, on first use (safe even from static initializers).
inline const Kernels& active() {
    static const Kernels k = choose();
    return k;
}

inline void layer(const int16_t* W, const int32_t* b, const int16_t* x, int in, int out, int32_t* y) {
    active().layer(W, b, x, in, out, y);
}
inline void axpy(int32_t* acc, const int16_t* col, int32_t scale, int n) { active().axpy(acc, col, scale, n); }
inline const char* name() { return active().name; }

}  // namespace fusion::kernel
#endif  // FUSION_SIMD_HPP
