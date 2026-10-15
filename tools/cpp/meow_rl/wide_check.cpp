// Native parity check of the generated Meowmind RL headers of task T0061 (tests/test_meow_rl_aligned.py and
// bots/meow_rl/aligned/export.py run it on every export).
//
// Critic mode:  c++ -std=c++20 -O2 -I inference -DCRITIC_HEADER='"path/critic_<format>.hpp"' [-DFUSION_SIMD=0] wide_check.cpp
//   stdin: rows of 416 little-endian float32 value features; stdout: one float64 per row, meow_rl_critic::value (the
//   function the search calls).
// Actor mode:   c++ -std=c++20 -O2 -I inference -I tools/cpp/meow_rl -DACTOR_HEADER='"path/actor.hpp"' [-DFUSION_SIMD=0] wide_check.cpp
//   stdin: rows of 3 x 5 x 112 float32 features, 125 mask bytes and one float64 uniform; stdout per row: the 15 int32
//   output accumulators (meow_rl_int::score), the int32 joint sample.hpp draws with that uniform (the deployed
//   choice) and the greedy joint.
// stderr names the mode, the format and the kernel the dispatch actually ran ("kernel avx2|neon|scalar"; FUSION_SIMD=0
// forces the scalar loops). Standard streams are switched to binary mode on Windows: the payloads are raw bytes.
#include <cstdint>
#include <cstdio>
#include <vector>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

static void binary_stdio() {
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
}

#ifndef FUSION_MAX_WIDTH
#define FUSION_MAX_WIDTH 1024
#endif

#if defined(CRITIC_HEADER)
#include CRITIC_HEADER

int main() {
    binary_stdio();
    constexpr int kVF = 416;
    std::vector<float> row(kVF);
    long n = 0;
    while (std::fread(row.data(), sizeof(float), kVF, stdin) == size_t(kVF)) {
        double v = meow_rl_critic::value(row.data());
        std::fwrite(&v, sizeof v, 1, stdout);
        n++;
    }
    std::fprintf(stderr, "critic format %s simd %d kernel %s rows %ld\n", meow_rl_critic::kFormat, FUSION_SIMD,
                 meow_rl_critic::kernel_name(), n);
    return 0;
}

#elif defined(ACTOR_HEADER)
#include <string>
#include <utility>
#include <vector>
struct Game {   // the fields sample.hpp's INIT helpers name; actor mode never calls them
    int teams = 0, me = 0, size = 0, turns = 0;
    std::vector<std::string> map;
    std::vector<std::pair<int, int>> bases, spots;
};
#include ACTOR_HEADER
#include "actor_int.hpp"
#include "../../../bots/meow_rl/sample.hpp"

int main() {
    binary_stdio();
    float feats[3][5][112];
    unsigned char mask[125];
    double u;
    long n = 0;
    while (std::fread(feats, sizeof feats, 1, stdin) == 1 && std::fread(mask, 1, 125, stdin) == 125 &&
           std::fread(&u, sizeof u, 1, stdin) == 1) {
        int64_t sc[3][5];
        int32_t out[17];
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < 5; m++) out[k * 5 + m] = int32_t(sc[k][m] = meow_rl_int::score(feats[k][m]));
        out[15] = meow_rl_sample::draw(sc, mask, meow_rl_model::kActor.outExp, meow_rl_model::kTemperature, u);
        out[16] = meow_rl_int::greedy(feats, mask);
        std::fwrite(out, sizeof out, 1, stdout);
        n++;
    }
    std::fprintf(stderr, "actor %s simd %d kernel %s rows %ld\n", meow_rl_model::kName, FUSION_SIMD, fusion::kernel::name(), n);
    return 0;
}
#else
#error "define CRITIC_HEADER or ACTOR_HEADER"
#endif
