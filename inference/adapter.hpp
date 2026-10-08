// Deployment adapter: how the bot combines the opponent policy q_theta with
// the legacy forecast q_H. The offline gate (training/evaluate_offline.py)
// fits alpha for exactly one adapter and the exporter records it in the
// model header (kAdapter), so the distribution the bot samples is the one the
// gate scored (fusion/mixture.py documents the same formulas).
//
//   kAdapterJoint (v2, default): one mixture choice per opponent team and
//     scenario. With probability alpha the team's move comes from the 125-way
//     network joint (its coordination intact), and its pickup / drop / attack
//     actions from the network's act and target heads conditioned on the
//     sampled moves. Otherwise every unit draws from q_H and fires as the
//     legacy planner says.
//       q(m) = (1 - a) prod_k qH_k(m_k) + a qNN(m)
//   kAdapterMarginal (v1): every unit draws from its own mixed marginal,
//     independently of its teammates; the network's joint structure is lost.
//       q(m) = prod_k [(1 - a) qH_k(m_k) + a qNN_k(m_k)]
//
// Both adapters give each unit the same marginal (1 - a) qH_k + a qNN_k, so
// search code that needs unit marginals analytically (expected incoming fire)
// uses mixedMarginal() under either. Nothing here allocates.
#ifndef FUSION_ADAPTER_HPP
#define FUSION_ADAPTER_HPP
#include <array>

#include "policy.hpp"

namespace fusion {

enum Adapter { kAdapterMarginal = 1, kAdapterJoint = 2 };

// The legacy forecast for one opponent team: per unit, a distribution over
// moves S, U, D, L, R (a unit off the board puts all its mass on S).
struct LegacyForecast {
    std::array<std::array<double, kMoves>, 3> move{};
};

inline double mixedMarginal(double alpha, const LegacyForecast& qh, const PolicyOutput& nn, int k, int m) {
    return (1 - alpha) * qh.move[k][m] + alpha * nn.marginal[k][m];
}

// Probability of joint move j = m0 * 25 + m1 * 5 + m2 under the adapter.
inline double mixedJointProb(int adapter, double alpha, const LegacyForecast& qh, const PolicyOutput& nn, int j) {
    const int m[3] = {j / 25, j / 5 % 5, j % 5};
    if (adapter == kAdapterJoint)
        return (1 - alpha) * qh.move[0][m[0]] * qh.move[1][m[1]] * qh.move[2][m[2]] + alpha * nn.prob[j];
    double p = 1;
    for (int k = 0; k < 3; k++) p *= mixedMarginal(alpha, qh, nn, k, m[k]);
    return p;
}

// Inverse-CDF draw from p[0..n) with u in [0, 1). Rounding can leave u past the
// last cumulative sum; the draw then falls on the last entry with mass.
inline int sampleIndex(const double* p, int n, double u) {
    double acc = 0;
    int last = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] <= 0) continue;
        last = i;
        acc += p[i];
        if (u < acc) return i;
    }
    return last;
}

struct TeamDraw {
    int move[3] = {0, 0, 0};
    bool learned = false;  // joint adapter: the network drew this move, so it also draws the actions
};

// One team move. u[0] chooses the mixture component (joint adapter), u[1]
// draws the network joint, u[1 + k] draws unit k otherwise. Uniforms in [0, 1).
inline TeamDraw drawTeam(int adapter, double alpha, const LegacyForecast& qh, const PolicyOutput& nn, const double u[4]) {
    TeamDraw d;
    if (adapter == kAdapterJoint && u[0] < alpha) {
        const int j = sampleIndex(nn.prob.data(), kJoint, u[1]);
        d.move[0] = j / 25;
        d.move[1] = j / 5 % 5;
        d.move[2] = j % 5;
        d.learned = true;
        return d;
    }
    for (int k = 0; k < 3; k++) {
        double p[kMoves];
        for (int m = 0; m < kMoves; m++) p[m] = adapter == kAdapterJoint ? qh.move[k][m] : mixedMarginal(alpha, qh, nn, k, m);
        d.move[k] = sampleIndex(p, kMoves, u[1 + k]);
    }
    return d;
}

// Network action of unit k after a learned draw, conditioned on its sampled
// move: returns the act (kWait, kPickup, kDrop, kAttack) and, for an attack,
// the target's candidate slot in `slot` (an index into f.cand[k]).
inline int drawAct(const PolicyModel& model, const PolicyFeatures& f, int k, int move, double uAct, double uTarget, int& slot) {
    double pa[kActs], pt[kCand];
    actPolicy(model, f, k, move, pa);
    const int act = sampleIndex(pa, kActs, uAct);
    slot = -1;
    if (act == kAttack) {
        targetPolicy(model, f, k, move, pt);
        slot = sampleIndex(pt, kCand, uTarget);
    }
    return act;
}

}  // namespace fusion
#endif  // FUSION_ADAPTER_HPP
