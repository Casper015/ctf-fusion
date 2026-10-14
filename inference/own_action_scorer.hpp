// Own-action scorer Q(s, a) at run time: the float reference route (task T0044, docs/RL_PLAN.md section 4).
//
// An exported scorer (training/export_decision_scorer.py writes scorer_model.hpp) is a ScorerModel: the layers of
// training.train_decision_ranker.Scorer and every identity it was bound to. It has its own kind, own_action_scorer,
// and its own types: it is never an opponent P (PolicyModel*) or a board value V (ValueModel*), and those routes
// cannot read it.
//
//   linear  Q = w . phi(a)                 (the candidate features only; a root-shared term cancels in g)
//   mlp     Q = MLP([psi(s), phi(a)])      (clipped ReLU on [0, 1] between layers, linear output; fusion/nn.MLP)
//
// psi and phi are inference/own_action.hpp rootFeatures and candidateFeatures. The weights are the trained float32
// values, written exactly (hexadecimal literals); sums are accumulated in double. Python's float64 forward of the
// same weights agrees within 1e-9 and the trainer's float32 forward within 1e-4 (tests/test_own_action_scorer.py).
// There is no quantized route here.
//
// validate() refuses (the caller keeps its incumbent) a model of another kind, of another feature identity than
// this build's own_action.hpp, with inconsistent dimensions, a layer wider than kScorerMaxWidth, or a non-finite
// weight.
#ifndef FUSION_OWN_ACTION_SCORER_HPP
#define FUSION_OWN_ACTION_SCORER_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "own_action.hpp"

namespace fusion::own_action {

inline constexpr const char* kScorerKind = "own_action_scorer";
inline constexpr int kScorerMaxWidth = 1024;

enum class ScorerInput { kCandidate = 0, kRootAndCandidate = 1 };

struct ScorerLayer {
    int in = 0, out = 0;
    const float* w = nullptr;  // row-major [out][in], as fusion/nn.MLP stores W
    const float* b = nullptr;
    bool hidden = true;        // false only for the last layer
};

struct ScorerModel {
    const char* kind = nullptr;              // must be kScorerKind
    const char* model = nullptr;             // "linear" or "mlp"
    const char* featureVersion = nullptr;    // own-action-1
    const char* featureDigest = nullptr;     // decision_features.feature_identity digest
    int rootDim = 0, candDim = 0;
    ScorerInput input = ScorerInput::kCandidate;
    const ScorerLayer* layers = nullptr;
    int count = 0;
    // Bindings recorded by the export (strings; the carrier hook compares them, nothing here does).
    const char* runConfigSha256 = nullptr;
    const char* modelSha256 = nullptr;
    const char* carrierConfigSha256 = nullptr;  // "unbound" without a validated carrier configuration
    const char* carrierVariant = nullptr;       // the carrier that collected the training data
    const char* carrierStatus = nullptr;        // admitted, experimental or unbound
    const char* carrierPFamily = nullptr;       // its opponent P family (p2, p3, ...)
    const char* carrierPModels = nullptr;       // its opponent P model directory (inference/models*)
    int carrierValue = -1;                      // its V: 1 on, 0 off, -1 unbound
    double lambda = 0.0;                        // the frozen rule's weight (training.evaluate_decisions select)
    const char* norm = nullptr;                 // zscore or incumbent_override
    const char* selectionSha256 = nullptr;      // "unselected" for a test export
};

inline bool sameText(const char* a, const char* b) { return a && b && std::strcmp(a, b) == 0; }

// True when the model can be used; otherwise false and *why says what is wrong.
inline bool validate(const ScorerModel& m, const char** why = nullptr) {
    auto fail = [&](const char* w) { if (why) *why = w; return false; };
    if (!sameText(m.kind, kScorerKind)) return fail("not an own_action_scorer");
    if (!sameText(m.featureVersion, kVersion) || !sameText(m.featureDigest, kIdentityDigest))
        return fail("feature identity differs from this build's own_action.hpp");
    if (m.rootDim != kRootDim || m.candDim != kCandDim) return fail("feature dimensions differ");
    if (!m.layers || m.count < 1) return fail("no layers");
    const bool linear = sameText(m.model, "linear");
    if (!linear && !sameText(m.model, "mlp")) return fail("unknown model");
    if ((m.input == ScorerInput::kCandidate) != linear) return fail("input mode does not match the model");
    const int in = m.input == ScorerInput::kCandidate ? m.candDim : m.rootDim + m.candDim;
    if (m.layers[0].in != in) return fail("first layer width differs from the input");
    for (int i = 0; i < m.count; i++) {
        const ScorerLayer& l = m.layers[i];
        if (l.in < 1 || l.out < 1 || l.in > kScorerMaxWidth || l.out > kScorerMaxWidth) return fail("layer width");
        if (i && l.in != m.layers[i - 1].out) return fail("layers do not chain");
        if (l.hidden != (i + 1 < m.count)) return fail("hidden flags");
        for (int j = 0; j < l.in * l.out; j++) if (!std::isfinite(l.w[j])) return fail("non-finite weight");
        for (int j = 0; j < l.out; j++) if (!std::isfinite(l.b[j])) return fail("non-finite bias");
    }
    if (m.layers[m.count - 1].out != 1) return fail("output is not one value");
    if (!std::isfinite(m.lambda) || m.lambda < 0) return fail("lambda");
    if (!sameText(m.norm, "zscore") && !sameText(m.norm, "incumbent_override")) return fail("normalization");
    return true;
}

// Q(s, a) of a validated model. `root` holds kRootDim floats (rootFeatures), `cand` kCandDim (candidateFeatures).
inline double q(const ScorerModel& m, const float* root, const float* cand) {
    std::array<double, kScorerMaxWidth> a{}, b{};
    int n = 0;
    if (m.input == ScorerInput::kRootAndCandidate)
        for (int i = 0; i < m.rootDim; i++) a[n++] = root[i];
    for (int i = 0; i < m.candDim; i++) a[n++] = cand[i];
    for (int li = 0; li < m.count; li++) {
        const ScorerLayer& l = m.layers[li];
        for (int o = 0; o < l.out; o++) {
            double s = 0.0;
            const float* w = l.w + static_cast<size_t>(o) * l.in;
            for (int i = 0; i < l.in; i++) s += static_cast<double>(w[i]) * a[i];
            s += l.b[o];
            b[o] = l.hidden ? std::clamp(s, 0.0, 1.0) : s;
        }
        std::copy(b.begin(), b.begin() + l.out, a.begin());
    }
    return a[0];
}

}  // namespace fusion::own_action
#endif  // FUSION_OWN_ACTION_SCORER_HPP
