// Parity driver for the experimental low-bit policy kernels (task T0030; tests/test_precision.py builds it).
//
// Compiled against a generated "precision_case.hpp" (namespace pcase) that holds lowbit::Net values
// (pcase::nets, pcase::kCount) and, optionally, one fusion::PolicyModelLP (pcase::policy) with its family.
// Reads commands from stdin, one per line:
//
//   net <index> <n> x0 .. x(n-1)        every kernel path this CPU can run, each on its own output line:
//                                       "<path> acc0 acc1 .." (last-layer accumulators); "reference" is the
//                                       per-weight decode loop (lowbit::rowReference), the oracle
//   policy <values..>                   PolicyFeatures (p2) or PolicyFeaturesP3 (p3) from raw arrays in this
//                                       order: g, u, m, legal, actLegal, cand, candF[, combat]; prints the joint
//                                       logits (hex floats; illegal = -inf marker "x"), then act and target
//                                       probabilities for every (unit, legal move). Needs PCASE_POLICY_FAMILY.
//
// The first output line names the active path and every path tested.
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "policy_p3.hpp"
#include "policy_precision/policy_lp.hpp"
#include "precision_case.hpp"

using namespace fusion;

namespace {

struct Path {
    const char* name;
    lowbit::RowsFn rows;
};

std::vector<Path> paths() {
    std::vector<Path> out{{"reference", nullptr}, {"scalar", lowbit::rowsScalar}};
#if FUSION_X86
    out.push_back({"sse2", lowbit::rowsSse2});
#if FUSION_AVX2
    if (kernel::cpuHasAvx2()) out.push_back({"avx2", lowbit::rowsAvx2});
#endif
#endif
#if FUSION_NEON
    out.push_back({"neon", lowbit::rowsNeon});
#endif
    out.push_back({"active", lowbit::active().rows});
    return out;
}

// The oracle: every layer through rowReference, requantized with the same rule.
void referenceForward(const lowbit::Net& net, const int32_t* x, int32_t* acc) {
    std::vector<int16_t> src(net.layers[0].inPad, 0), dst;
    for (int i = 0; i < net.layers[0].in; i++) src[i] = static_cast<int16_t>(x[i]);
    for (int l = 0; l < net.count; l++) {
        const lowbit::Layer& L = net.layers[l];
        if (l + 1 == net.count) {
            for (int o = 0; o < L.out; o++) acc[o] = lowbit::rowReference(L, o, src.data());
            return;
        }
        dst.assign(net.layers[l + 1].inPad, 0);
        for (int o = 0; o < L.out; o++) dst[o] = static_cast<int16_t>(lowbit::requantize(L, o, lowbit::rowReference(L, o, src.data())));
        src.swap(dst);
    }
}

#ifdef PCASE_POLICY_FAMILY
void readFeatures(std::istream& in, PolicyFeatures& f) {
    auto get = [&]() { long v; in >> v; return v; };
    for (auto& v : f.g) v = static_cast<int16_t>(get());
    for (auto& r : f.u) for (auto& v : r) v = static_cast<int16_t>(get());
    for (auto& r : f.m) for (auto& mm : r) for (auto& v : mm) v = static_cast<int16_t>(get());
    for (auto& r : f.legal) for (auto& v : r) v = static_cast<unsigned char>(get());
    for (auto& r : f.actLegal) for (auto& mm : r) for (auto& v : mm) v = static_cast<unsigned char>(get());
    for (auto& r : f.cand) for (auto& v : r) v = static_cast<int>(get());
    for (auto& r : f.candF) for (auto& c : r) for (auto& v : c) v = static_cast<int16_t>(get());
}

void readFeatures(std::istream& in, PolicyFeaturesP3& f) {
    readFeatures(in, static_cast<PolicyFeatures&>(f));
    for (auto& r : f.c) for (auto& mm : r) for (auto& v : mm) { long x; in >> x; v = static_cast<int16_t>(x); }
}

void printDouble(double v) {
    if (v <= kNegInf) std::printf(" x");
    else std::printf(" %a", v);
}

template <class F>
void runPolicy(std::istream& in, F& f) {
    readFeatures(in, f);
    PolicyOutput out;
    jointPolicyLP(*pcase::policy, f, out);
    std::printf("joint");
    for (double v : out.logit) printDouble(v);
    std::printf("\n");
    for (int k = 0; k < 3; k++)
        for (int m = 0; m < kMoves; m++) {
            if (!f.legal[k][m]) continue;
            double a[kActs], t[kCand];
            actPolicyLP(*pcase::policy, f, k, m, a);
            targetPolicyLP(*pcase::policy, f, k, m, t);
            std::printf("act %d %d", k, m);
            for (double v : a) std::printf(" %a", v);
            std::printf("\ntarget %d %d", k, m);
            for (double v : t) std::printf(" %a", v);
            std::printf("\n");
        }
}

#endif  // PCASE_POLICY_FAMILY

}  // namespace

int main() {
    const auto all = paths();
    std::printf("active %s tested", lowbit::name());
    for (const auto& p : all) std::printf(" %s", p.name);
    std::printf("\n");
    std::string cmd;
    std::vector<int32_t> x, acc(kMaxWidth);
    while (std::cin >> cmd) {
        if (cmd == "net") {
            int idx, n;
            std::cin >> idx >> n;
            x.assign(n, 0);
            for (auto& v : x) std::cin >> v;
            const lowbit::Net& net = *pcase::nets[idx];
            for (const auto& p : all) {
                if (p.rows) lowbit::forwardAcc(net, x.data(), acc.data(), p.rows);
                else referenceForward(net, x.data(), acc.data());
                std::printf("%s", p.name);
                for (int o = 0; o < net.out(); o++) std::printf(" %d", acc[o]);
                std::printf("\n");
            }
        } else if (cmd == "policy") {
#ifdef PCASE_POLICY_FAMILY
#if PCASE_POLICY_FAMILY == 3
            PolicyFeaturesP3 f{};
            runPolicy(std::cin, f);
#else
            PolicyFeatures f{};
            runPolicy(std::cin, f);
#endif
#else
            std::fprintf(stderr, "no policy in this case\n");
            return 2;
#endif
        } else {
            std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
            return 2;
        }
        std::fflush(stdout);
    }
    return 0;
}
