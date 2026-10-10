// Policy feature family p3: the legacy opponent policy (policy.hpp, family
// p2) with the combat-context block (combat.hpp) appended per unit and
// destination.
//
//   mover / act  [global, unit k, move (k, m), combat (k, m)]      kMoverInP3
//   squad        [global, units 0..2, 15 moves, 15 combat blocks]  kSquadInP3
//   target       unchanged (policy.hpp targetInput, 7 inputs)
//
// The legacy encoder runs unchanged inside encodePolicyP3, so the p2 part of
// the input is bit-identical to what a p2 model sees. The types keep the
// families apart: a p3 model only accepts PolicyFeaturesP3, and a p2 model
// given PolicyFeaturesP3 reads exactly its p2 inputs (the base object). The
// exported header of a p3 model static_asserts its input widths against the
// constants below (training/export_weights.py), so a model trained on another
// combat block cannot be compiled in.
#ifndef FUSION_POLICY_P3_HPP
#define FUSION_POLICY_P3_HPP
#include <array>
#include <cstdint>

#include "adapter.hpp"
#include "combat.hpp"
#include "policy.hpp"

namespace fusion {

inline constexpr int kPolicyFamilyP3 = 3;
inline constexpr int kMoverInP3 = kMoverIn + kPC;
inline constexpr int kSquadInP3 = kSquadIn + 3 * kMoves * kPC;

struct PolicyFeaturesP3 : PolicyFeatures {
    CombatBlock c{};  // [unit k][move m][combat feature]
};

struct PolicyModelP3 { Net mover, squad, act, target; };

// `cs` must summarize the same (b, s, v); one summary serves every team.
inline void encodePolicyP3(const Board& b, const State& s, const TurnView& v, const CombatSummary& cs, int team,
                           PolicyFeaturesP3& out) {
    encodePolicy(b, s, v, team, out);
    encodeCombatPolicy(b, s, v, cs, team, out.c);
}

inline void moverInput(const PolicyFeaturesP3& f, int k, int move, int32_t* x) {
    moverInput(static_cast<const PolicyFeatures&>(f), k, move, x);
    int i = kMoverIn;
    for (int v : f.c[k][move]) x[i++] = v;
}

inline void squadInput(const PolicyFeaturesP3& f, int32_t* x) {
    squadInput(static_cast<const PolicyFeatures&>(f), x);
    int i = kSquadIn;
    for (int k = 0; k < 3; k++)
        for (int m = 0; m < kMoves; m++)
            for (int v : f.c[k][m]) x[i++] = v;
}

inline void jointPolicy(const PolicyModelP3& model, const PolicyFeaturesP3& f, PolicyOutput& out) {
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

inline void actPolicy(const PolicyModelP3& model, const PolicyFeaturesP3& f, int k, int move, double prob[kActs]) {
    double logit[kActs];
    int32_t x[kMaxWidth], acc[kMaxWidth];
    moverInput(f, k, move, x);
    forwardRaw(model.act, x, acc);
    for (int a = 0; a < kActs; a++) logit[a] = f.actLegal[k][move][a] ? toReal(model.act, acc[a]) : kNegInf;
    softmaxMasked(logit, kActs, prob);
}

// The target head keeps the legacy 7 inputs.
inline void targetPolicy(const PolicyModelP3& model, const PolicyFeaturesP3& f, int k, int move, double prob[kCand]) {
    const PolicyModel legacy{model.mover, model.squad, model.act, model.target};
    targetPolicy(legacy, static_cast<const PolicyFeatures&>(f), k, move, prob);
}

inline int drawAct(const PolicyModelP3& model, const PolicyFeaturesP3& f, int k, int move, double uAct, double uTarget, int& slot) {
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
#endif  // FUSION_POLICY_P3_HPP
