// Raw full/incremental parity at the actual exported width; no timing evidence.
#include <cstdio>
#include "stream.hpp"
#include FUSION_MODEL_HEADER

int main() {
    fusion::binaryStdout();
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1) return 5;
#endif
    const auto& model = FUSION_MODEL_NS::kValue;
    auto noTranspose = model;
    fusion::Layer layers[3];
    if (model.net.count != 3 || model.net.in() != fusion::kV3) return 2;
    std::copy(model.net.layers, model.net.layers + 3, layers);
    layers[0].wT = nullptr;
    noTranspose.net.layers = layers;
    fusion::ValueFeaturesV3 f;
    fusion::ValueAccumulatorV3 transposed, rowMajor;
    int32_t input[fusion::kV3], full[1];
    while (std::fread(f.data(), sizeof(int16_t), f.size(), stdin) == f.size()) {
        for (int i = 0; i < fusion::kV3; ++i) input[i] = f[i];
        fusion::forwardRaw(model.net, input, full);
        fusion::update(model, f, transposed);
        fusion::update(noTranspose, f, rowMajor);
        int32_t result[3] = {full[0], fusion::valueRaw(model, transposed), fusion::valueRaw(noTranspose, rowMajor)};
        if (result[0] != result[1] || result[0] != result[2]) return 3;
        std::fwrite(result, sizeof(int32_t), 3, stdout);
    }
    return std::ferror(stdin) ? 4 : 0;
}
