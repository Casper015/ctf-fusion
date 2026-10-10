// Contract check for inference/adapter.hpp (tests/test_math.py):
//
//   adapter_check < case.txt
//
// case.txt: "adapter alpha draws seed", then the legacy unit marginals (3 x 5
// numbers), then the network joint (125 numbers). Prints three lines: the
// exact probability of each of the 125 joint moves (mixedJointProb), how often
// drawTeam drew each one in `draws` draws, and the 15 mixed unit marginals.
// The test compares the first line with fusion/mixture.py adapter_table and
// the counts with the exact probabilities.
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "../../inference/adapter.hpp"

using namespace fusion;

namespace {

uint64_t g_state = 0;

double uniform() {  // splitmix64 -> [0, 1)
    uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

}  // namespace

int main() {
    int adapter = 0;
    long draws = 0;
    double alpha = 0;
    std::cin >> adapter >> alpha >> draws >> g_state;
    LegacyForecast qh;
    for (auto& row : qh.move)
        for (double& v : row) std::cin >> v;
    PolicyOutput nn;
    for (double& v : nn.prob) std::cin >> v;
    if (!std::cin) return 2;
    for (int j = 0; j < kJoint; j++) {
        nn.marginal[0][j / 25] += nn.prob[j];
        nn.marginal[1][j / 5 % 5] += nn.prob[j];
        nn.marginal[2][j % 5] += nn.prob[j];
    }
    for (int j = 0; j < kJoint; j++) std::printf("%s%.17g", j ? " " : "", mixedJointProb(adapter, alpha, qh, nn, j));
    std::printf("\n");
    long counts[kJoint] = {};
    for (long i = 0; i < draws; i++) {
        const double u[4] = {uniform(), uniform(), uniform(), uniform()};
        const TeamDraw d = drawTeam(adapter, alpha, qh, nn, u);
        counts[d.move[0] * 25 + d.move[1] * 5 + d.move[2]]++;
    }
    for (int j = 0; j < kJoint; j++) std::printf("%s%ld", j ? " " : "", counts[j]);
    std::printf("\n");
    for (int k = 0; k < 3; k++)
        for (int m = 0; m < kMoves; m++) std::printf("%s%.17g", k || m ? " " : "", mixedMarginal(alpha, qh, nn, k, m));
    std::printf("\n");
    return 0;
}
