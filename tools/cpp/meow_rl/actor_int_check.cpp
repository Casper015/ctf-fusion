// Parity tool for the integer actor (tests/test_meow_rl_bridge.py).
//   g++ -std=c++20 -O2 -Iinference -Itools/cpp/meow_rl -include <generated header> tools/cpp/meow_rl/actor_int_check.cpp -o check
// stdin: N, then N rows of 112 float32 bit patterns (hex); stdout: one output accumulator per row,
// then for every group of 15 rows (3 units x 5 moves) with a 125-character 0/1 mask line, the greedy joint.
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "actor_int.hpp"

int main() {
    int n = 0, groups = 0;
    std::cin >> n >> groups;
    std::vector<float> rows(size_t(n) * 112);
    for (auto& v : rows) {
        uint32_t bits;
        std::cin >> std::hex >> bits;
        std::memcpy(&v, &bits, sizeof v);
    }
    std::cin >> std::dec;
    for (int r = 0; r < n; r++) std::cout << meow_rl_int::score(&rows[size_t(r) * 112]) << '\n';
    for (int g = 0; g < groups; g++) {
        std::string mask;
        std::cin >> mask;
        unsigned char m[125];
        for (int j = 0; j < 125; j++) m[j] = mask[j] == '1';
        auto feats = reinterpret_cast<const float (*)[5][112]>(&rows[size_t(g) * 15 * 112]);
        std::cout << "G " << meow_rl_int::greedy(feats, m) << '\n';
    }
    return 0;
}
