// Bit-exact parity of the SIMD multiply-add kernels (inference/simd.hpp)
// against the scalar loop they replaced. Run by tests/test_contracts.py, and
// on the host by tools/env_check.py and tools/capacity_bench.py --verify.
//
//   kernel_check [cases]      exit 0 when every available path matches
//
// Every path the CPU can run (scalar, sse2, avx2, neon) is called directly on
// random layers: input widths around every vector boundary (1..1024), weights
// up to the int16 limit with sums kept inside the exporter's proven
// [-2^30, 2^30] bound, inputs in the feature range (first layers) or 0..128
// (hidden layers). Then whole networks through forwardRaw (the active path)
// against the original scalar forwardFrom, kept verbatim below, and value
// accumulator updates (transposed column kernel) against a full refresh.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../inference/mlp.hpp"
#include "../inference/value.hpp"

using namespace fusion;

namespace {

uint64_t g_state = 0x9E3779B97F4A7C15ull;

uint64_t next() {
    uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int32_t uniform(int32_t lo, int32_t hi) { return lo + static_cast<int32_t>(next() % static_cast<uint64_t>(hi - lo + 1)); }

// The loop forwardFrom used before the SIMD kernels, verbatim (int32 activations).
void forwardFromReference(const Net& net, int first, const int32_t* x, int32_t* acc) {
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

struct Path {
    const char* name;
    kernel::LayerFn layer;
    kernel::AxpyFn axpy;
};

std::vector<Path> paths() {
    std::vector<Path> p{{"scalar", kernel::layerScalar, kernel::axpyScalar}};
#if FUSION_X86
    p.push_back({"sse2", kernel::layerSse2, kernel::axpySse2});
#if FUSION_AVX2
    if (kernel::cpuHasAvx2()) p.push_back({"avx2", kernel::layerAvx2, kernel::axpyAvx2});
#endif
#endif
#if FUSION_NEON
    p.push_back({"neon", kernel::layerNeon, kernel::axpyNeon});
#endif
    return p;
}

// Weights as large as the int16 limit allows while every sum stays inside 2^30.
int16_t weightLimit(int in, int xmax) {
    const int64_t lim = (int64_t(1) << 30) / (int64_t(in) * xmax + 1);
    return static_cast<int16_t>(lim > 32767 ? 32767 : lim);
}

struct OwnedLayer {
    std::vector<int16_t> w, wT;
    std::vector<int32_t> b;
    Layer layer;
};

OwnedLayer randomLayer(int in, int out, int xmax, bool hidden) {
    OwnedLayer L;
    const int16_t wl = weightLimit(in, xmax);
    L.w.resize(static_cast<size_t>(in) * out);
    for (auto& v : L.w) v = static_cast<int16_t>(uniform(-wl, wl));
    L.b.resize(out);
    for (auto& v : L.b) v = uniform(-(1 << 20), 1 << 20);
    L.wT.resize(L.w.size());
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++) L.wT[static_cast<size_t>(i) * out + o] = L.w[static_cast<size_t>(o) * in + i];
    L.layer = Layer{in, out, L.w.data(), L.b.data(), uniform(3, 14), hidden, L.wT.data()};
    return L;
}

}  // namespace

int main(int argc, char** argv) {
    const int cases = argc > 1 ? std::atoi(argv[1]) : 200;
    const int widths[] = {1, 2, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 48, 63, 64, 65, 100, 127, 128,
                          129, 191, 222, 255, 256, 257, 383, 500, 511, 512, 513, 1000, 1023, 1024};
    long mismatches = 0, checks = 0;
    const std::vector<Path> ps = paths();
    std::vector<int16_t> x16(kMaxWidth);
    std::vector<int32_t> ref(kMaxWidth), got(kMaxWidth);

    // 1. One layer, every path, every width.
    for (int w : widths) {
        if (w > kMaxWidth) continue;
        for (int c = 0; c < 4; c++) {
            const int out = uniform(1, 40);
            const int xmax = c % 2 ? 128 : 400;
            OwnedLayer L = randomLayer(w, out, xmax, true);
            for (int i = 0; i < w; i++) x16[i] = static_cast<int16_t>(c % 2 ? uniform(0, 128) : uniform(-400, 400));
            kernel::layerScalar(L.w.data(), L.b.data(), x16.data(), w, out, ref.data());
            for (const Path& p : ps) {
                p.layer(L.w.data(), L.b.data(), x16.data(), w, out, got.data());
                for (int o = 0; o < out; o++) mismatches += got[o] != ref[o];
                checks++;
            }
        }
    }

    // 2. Column updates (axpy), every path.
    for (int n : widths) {
        if (n > kMaxWidth) continue;
        std::vector<int16_t> col(n);
        for (auto& v : col) v = static_cast<int16_t>(uniform(-32767, 32767));
        for (int c = 0; c < 4; c++) {
            const int32_t scale = uniform(-800, 800);
            std::vector<int32_t> base(n);
            for (auto& v : base) v = uniform(-(1 << 29), 1 << 29);
            std::vector<int32_t> expect = base;
            kernel::axpyScalar(expect.data(), col.data(), scale, n);
            for (const Path& p : ps) {
                std::vector<int32_t> a = base;
                p.axpy(a.data(), col.data(), scale, n);
                for (int o = 0; o < n; o++) mismatches += a[o] != expect[o];
                checks++;
            }
        }
    }

    // 3. Whole networks through the active path against the original loop.
    std::vector<int32_t> x32(kMaxWidth);
    for (int c = 0; c < cases; c++) {
        const int depth = uniform(1, 4);
        std::vector<int> dims{widths[uniform(0, 33)]};
        for (int d = 0; d < depth; d++) dims.push_back(widths[uniform(0, 30)]);
        bool fits = true;
        for (int d : dims) fits &= d <= kMaxWidth;
        if (!fits) continue;
        std::vector<OwnedLayer> owned;
        for (int d = 0; d < depth; d++) owned.push_back(randomLayer(dims[d], dims[d + 1], d == 0 ? 400 : 128, d + 1 < depth));
        std::vector<Layer> layers;
        for (auto& o : owned) layers.push_back(o.layer);
        const Net net{layers.data(), depth, 14};
        for (int i = 0; i < dims[0]; i++) x32[i] = uniform(-400, 400);
        forwardFromReference(net, 0, x32.data(), ref.data());
        forwardRaw(net, x32.data(), got.data());
        for (int o = 0; o < dims.back(); o++) mismatches += got[o] != ref[o];
        checks++;
        if (depth > 1) {  // the value tail: forwardFrom(net, 1, ...) on activations 0..128
            for (int i = 0; i < dims[1]; i++) x32[i] = uniform(0, 128);
            forwardFromReference(net, 1, x32.data(), ref.data());
            forwardFrom(net, 1, x32.data(), got.data());
            for (int o = 0; o < dims.back(); o++) mismatches += got[o] != ref[o];
            checks++;
        }
    }

    // 4. Value accumulator: incremental updates (contiguous columns) against a full refresh, bit for bit.
    for (int c = 0; c < cases; c++) {
        OwnedLayer first = randomLayer(kV, std::min(kMaxWidth, widths[uniform(10, 33)]), 400, true);
        OwnedLayer tail = randomLayer(first.layer.out, 1, 128, false);
        const Layer layers[2] = {first.layer, tail.layer};
        const ValueModel model{Net{layers, 2, 14}};
        ValueAccumulator inc, full;
        ValueFeatures f{};
        for (int step = 0; step < 6; step++) {
            for (int k = 0; k < uniform(1, 8); k++) f[uniform(0, kV - 1)] = static_cast<int16_t>(uniform(-20, 400));
            update(model, f, inc);
            refresh(model, f, full);
            for (int o = 0; o < first.layer.out; o++) mismatches += inc.pre[o] != full.pre[o];
            mismatches += valueRaw(model, inc) != valueRaw(model, full);
            checks++;
        }
    }

    std::printf("kernel_check active %s paths", kernel::name());
    for (const Path& p : ps) std::printf(" %s", p.name);
    std::printf(" checks %ld mismatches %ld\n", checks, mismatches);
    return mismatches ? 1 : 0;
}
