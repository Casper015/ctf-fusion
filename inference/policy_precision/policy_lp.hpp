// The opponent policy heads on low-bit nets (task T0030): the same joint / act / target semantics, masks and
// summation order as inference/policy.hpp, with lowbit::Net in place of the INT16 Net. Works for both feature
// families through the existing input builders (policy.hpp for p2, policy_p3.hpp for p3); the caller includes
// whichever family header it uses before this one. Experimental: nothing in a production bot includes it.
#ifndef FUSION_POLICY_PRECISION_POLICY_LP_HPP
#define FUSION_POLICY_PRECISION_POLICY_LP_HPP
#include <array>
#include <cstdint>

#include "../policy.hpp"
#include "lowbit.hpp"

namespace fusion {

struct PolicyModelLP {
    lowbit::Net mover, squad, act, target;
    int family = 2;            // 2 or 3: checked against the features the carrier encodes
    uint64_t identity = 0;     // low 64 bits of the manifest's weights digest (cache and receipt identity)
    uint64_t profile = 0;      // low 64 bits of the profile identity
};

template <class F>
inline void jointPolicyLP(const PolicyModelLP& model, const F& f, PolicyOutput& out) {
    std::array<std::array<double, kMoves>, 3> mover{};
    int32_t x[kMaxWidth], acc[kMaxWidth];
    for (int k = 0; k < 3; k++)
        for (int m = 0; m < kMoves; m++) {
            if (!f.legal[k][m]) { mover[k][m] = kNegInf; continue; }
            moverInput(f, k, m, x);
            lowbit::forwardAcc(model.mover, x, acc);
            mover[k][m] = lowbit::toReal(model.mover, 0, acc[0]);
        }
    squadInput(f, x);
    lowbit::forwardAcc(model.squad, x, acc);
    for (int j = 0; j < kJoint; j++) {
        const int m0 = j / 25, m1 = j / 5 % 5, m2 = j % 5;
        const bool legal = f.legal[0][m0] && f.legal[1][m1] && f.legal[2][m2];
        out.logit[j] = legal ? mover[0][m0] + mover[1][m1] + mover[2][m2] + lowbit::toReal(model.squad, j, acc[j]) : kNegInf;
    }
    softmaxMasked(out.logit.data(), kJoint, out.prob.data());
    for (auto& row : out.marginal) row.fill(0);
    for (int j = 0; j < kJoint; j++) {
        out.marginal[0][j / 25] += out.prob[j];
        out.marginal[1][j / 5 % 5] += out.prob[j];
        out.marginal[2][j % 5] += out.prob[j];
    }
}

template <class F>
inline void actPolicyLP(const PolicyModelLP& model, const F& f, int k, int move, double prob[kActs]) {
    double logit[kActs];
    int32_t x[kMaxWidth], acc[kMaxWidth];
    moverInput(f, k, move, x);
    lowbit::forwardAcc(model.act, x, acc);
    for (int a = 0; a < kActs; a++) logit[a] = f.actLegal[k][move][a] ? lowbit::toReal(model.act, a, acc[a]) : kNegInf;
    softmaxMasked(logit, kActs, prob);
}

template <class F>
inline void targetPolicyLP(const PolicyModelLP& model, const F& f, int k, int move, double prob[kCand]) {
    double logit[kCand];
    int32_t x[kMaxWidth], acc[kMaxWidth];
    for (int c = 0; c < kCand; c++) {
        if (f.cand[k][c] < 0) { logit[c] = kNegInf; continue; }
        targetInput(f, k, c, move, x);
        lowbit::forwardAcc(model.target, x, acc);
        logit[c] = lowbit::toReal(model.target, 0, acc[0]);
    }
    softmaxMasked(logit, kCand, prob);
}

}  // namespace fusion
#endif  // FUSION_POLICY_PRECISION_POLICY_LP_HPP
