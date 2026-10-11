// Benchmark driver for an exported low-bit policy (task T0030): prepared for a later host task, not run by T0030.
//
//   precision_bench <iterations> < features.txt
//
// Compiled against one header from training/export_precision.py (-include <header>, -DPBENCH_NS=<namespace>),
// with the host's compiler and flags (`python -m training.export_precision bench-plan` records them). For each
// head it reports the cold first call, then the warm mean over <iterations> calls of the full forward, with the
// weight bytes each call reads (packed arrays, scales, biases) and the activation bytes it moves. Inputs are the
// raw feature vectors of recorded rows, one per line (head index, then values). A development machine's numbers
// are not evidence (AGENTS.md); timing is measured only on the host, under its stated concurrency.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "policy_precision/lowbit.hpp"

#ifndef PBENCH_NS
#error "build with -DPBENCH_NS=<namespace of the exported header>"
#endif

using namespace fusion;

namespace {

long long weightBytes(const lowbit::Net& net) {
    long long n = 0;
    for (int l = 0; l < net.count; l++) {
        const auto& L = net.layers[l];
        n += static_cast<long long>(L.out) * L.inPad * L.bits / 8 + 4LL * L.out;   // weights, bias
        if (L.bits == 4) n += static_cast<long long>(L.out) * (L.inPad / lowbit::kBlock);
        n += L.hidden ? 4LL * L.out : 8LL * L.out;                                  // multipliers or scales
    }
    return n;
}

long long activationBytes(const lowbit::Net& net) {
    long long n = 0;
    for (int l = 0; l < net.count; l++) n += 2LL * net.layers[l].inPad + 4LL * net.layers[l].out;
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 1000;
    const lowbit::Net* heads[4] = {&PBENCH_NS::model.mover, &PBENCH_NS::model.squad, &PBENCH_NS::model.act,
                                   &PBENCH_NS::model.target};
    const char* names[4] = {"mover", "squad", "act", "target"};
    std::vector<std::vector<int32_t>> inputs[4];
    int head, n;
    while (std::cin >> head >> n) {
        std::vector<int32_t> x(n);
        for (auto& v : x) std::cin >> v;
        if (head >= 0 && head < 4) inputs[head].push_back(std::move(x));
    }
    std::printf("kernel %s iterations %d\n", lowbit::name(), iterations);
    int32_t acc[kMaxWidth];
    volatile int32_t sink = 0;
    for (int h = 0; h < 4; h++) {
        if (inputs[h].empty()) continue;
        const auto t0 = std::chrono::steady_clock::now();
        lowbit::forwardAcc(*heads[h], inputs[h][0].data(), acc);
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            lowbit::forwardAcc(*heads[h], inputs[h][static_cast<size_t>(i) % inputs[h].size()].data(), acc);
            sink = sink + acc[0];
        }
        const auto t2 = std::chrono::steady_clock::now();
        std::printf("%s first_ns %lld warm_ns %.1f weight_bytes %lld activation_bytes %lld\n", names[h],
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                    std::chrono::duration<double, std::nano>(t2 - t1).count() / iterations, weightBytes(*heads[h]),
                    activationBytes(*heads[h]));
    }
    return sink == 12345 ? 1 : 0;
}
