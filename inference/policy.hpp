// Opponent squad policy q(b|s): joint move distribution over 5^3 = 125 joint
// moves, plus the action and attack-target heads conditioned on a move.
//
//   joint logit(m0,m1,m2) = mover(0,m0) + mover(1,m1) + mover(2,m2) + squad[m0*25+m1*5+m2]
//
// mover scores each unit's move from [global, unit, move] features; squad is a
// joint residual over the whole squad (coordination: focus fire, spreading,
// escorting). Illegal joint moves (a wall, a unit off the board) get zero
// probability. The head inputs below must match fusion/models.py exactly.
#ifndef FUSION_POLICY_HPP
#define FUSION_POLICY_HPP
#include <array>
#include <cmath>
#include <cstdint>

#include "features.hpp"
#include "mlp.hpp"

namespace fusion {

inline constexpr int kJoint = kMoves * kMoves * kMoves;
inline constexpr int kMoverIn = kG + kU + kM;
inline constexpr int kSquadIn = kG + 3 * kU + 3 * kMoves * kM;
inline constexpr int kTargetIn = 7;
inline constexpr double kNegInf = -1e300;

struct PolicyModel { Net mover, squad, act, target; };

struct PolicyOutput {
    std::array<double, kJoint> logit{};       // kNegInf when illegal
    std::array<double, kJoint> prob{};
    std::array<std::array<double, kMoves>, 3> marginal{};
};

inline void moverInput(const PolicyFeatures& f, int k, int move, int32_t* x) {
    int i = 0;
    for (int v : f.g) x[i++] = v;
    for (int v : f.u[k]) x[i++] = v;
    for (int v : f.m[k][move]) x[i++] = v;
}

inline void squadInput(const PolicyFeatures& f, int32_t* x) {
    int i = 0;
    for (int v : f.g) x[i++] = v;
    for (int k = 0; k < 3; k++)
        for (int v : f.u[k]) x[i++] = v;
    for (int k = 0; k < 3; k++)
        for (int m = 0; m < kMoves; m++)
            for (int v : f.m[k][m]) x[i++] = v;
}

inline void targetInput(const PolicyFeatures& f, int k, int c, int move, int32_t* x) {
    const auto& t = f.candF[k][c];
    const int32_t vals[kTargetIn] = {t[0], t[1], t[2], t[3], t[4 + move], t[9], t[10]};
    for (int i = 0; i < kTargetIn; i++) x[i] = vals[i];
}

inline void softmaxMasked(const double* logit, int n, double* prob) {
    double mx = kNegInf;
    for (int i = 0; i < n; i++) mx = std::max(mx, logit[i]);
    double sum = 0;
    for (int i = 0; i < n; i++) {
        prob[i] = logit[i] <= kNegInf ? 0.0 : std::exp(logit[i] - mx);
        sum += prob[i];
    }
    for (int i = 0; i < n; i++) prob[i] = sum > 0 ? prob[i] / sum : 0.0;
}

inline void jointPolicy(const PolicyModel& model, const PolicyFeatures& f, PolicyOutput& out) {
    std::array<std::array<double, kMoves>, 3> mover{};
    int32_t x[kMaxWidth], acc[kMaxWidth];
    for (int k = 0; k < 3; k++)
        for (int m = 0; m < kMoves; m++) {
            if (!f.legal[k][m]) { mover[k][m] = kNegInf; continue; }
            moverInput(f, k, m, x);
            forwardRaw(model.mover, x, acc);
            mover[k][m] = toReal(model.mover, acc[0]);
        }
    squadInput(f, x);
    forwardRaw(model.squad, x, acc);
    for (int j = 0; j < kJoint; j++) {
        const int m0 = j / 25, m1 = j / 5 % 5, m2 = j % 5;
        const bool legal = f.legal[0][m0] && f.legal[1][m1] && f.legal[2][m2];
        out.logit[j] = legal ? mover[0][m0] + mover[1][m1] + mover[2][m2] + toReal(model.squad, acc[j]) : kNegInf;
    }
    softmaxMasked(out.logit.data(), kJoint, out.prob.data());
    for (auto& row : out.marginal) row.fill(0);
    for (int j = 0; j < kJoint; j++) {
        out.marginal[0][j / 25] += out.prob[j];
        out.marginal[1][j / 5 % 5] += out.prob[j];
        out.marginal[2][j % 5] += out.prob[j];
    }
}

// P(act | unit k makes `move`): wait, pickup, drop, attack (zero when not effective).
inline void actPolicy(const PolicyModel& model, const PolicyFeatures& f, int k, int move, double prob[kActs]) {
    double logit[kActs];
    int32_t x[kMaxWidth], acc[kMaxWidth];
    moverInput(f, k, move, x);
    forwardRaw(model.act, x, acc);
    for (int a = 0; a < kActs; a++) logit[a] = f.actLegal[k][move][a] ? toReal(model.act, acc[a]) : kNegInf;
    softmaxMasked(logit, kActs, prob);
}

// P(target slot c | unit k attacks after `move`), over f.cand[k].
inline void targetPolicy(const PolicyModel& model, const PolicyFeatures& f, int k, int move, double prob[kCand]) {
    double logit[kCand];
    int32_t x[kMaxWidth], acc[kMaxWidth];
    for (int c = 0; c < kCand; c++) {
        if (f.cand[k][c] < 0) { logit[c] = kNegInf; continue; }
        targetInput(f, k, c, move, x);
        forwardRaw(model.target, x, acc);
        logit[c] = toReal(model.target, acc[0]);
    }
    softmaxMasked(logit, kCand, prob);
}

}  // namespace fusion
#endif  // FUSION_POLICY_HPP
