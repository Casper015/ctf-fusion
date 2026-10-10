#include <iostream>
#include <vector>
#include "../../inference/state.hpp"
#include "../../inference/features.hpp"
#include "../../inference/mlp.hpp"
#include "../../inference/policy.hpp"
#include "../../inference/value.hpp"
#include "../../inference/sim.hpp"

using namespace fusion;

int main() {
    std::cout << "--- Core Data Structures ---" << std::endl;
    std::cout << "sizeof(State):            " << sizeof(State) << " bytes" << std::endl;
    std::cout << "sizeof(Board):            " << sizeof(Board) << " bytes" << std::endl;
    std::cout << "sizeof(TurnView):         " << sizeof(TurnView) << " bytes" << std::endl;
    std::cout << "sizeof(PolicyFeatures):   " << sizeof(PolicyFeatures) << " bytes" << std::endl;
    std::cout << "sizeof(ValueFeatures):    " << sizeof(ValueFeatures) << " bytes" << std::endl;
    std::cout << "sizeof(PolicyOutput):     " << sizeof(PolicyOutput) << " bytes" << std::endl;
    std::cout << "sizeof(ValueAccumulator): " << sizeof(ValueAccumulator) << " bytes" << std::endl;
    std::cout << "sizeof(Workspace):        " << sizeof(Workspace) << " bytes" << std::endl;
    std::cout << "sizeof(Net):              " << sizeof(Net) << " bytes" << std::endl;
    std::cout << "kMaxWidth:                " << kMaxWidth << std::endl;
    std::cout << "Inference stack buffers:  " << 2 * kMaxWidth * sizeof(int16_t) + kMaxWidth * sizeof(int32_t) << " bytes" << std::endl;
    return 0;
}
