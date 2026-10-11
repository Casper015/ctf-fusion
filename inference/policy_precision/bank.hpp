// One opponent policy in several precisions, and the dispatch a carrier calls (task T0030).
//
// Reference16 is always the existing INT16 model (fusion::jointPolicy / actPolicy / targetPolicy), so a bank
// whose selector is disabled computes exactly what the carrier computed before. A low-bit profile is used
// only when its model is present and was exported for the same feature family as the features being
// evaluated; otherwise the request falls back to Reference16 and is counted.
#ifndef FUSION_POLICY_PRECISION_BANK_HPP
#define FUSION_POLICY_PRECISION_BANK_HPP
#include <cstdint>

#include "../policy.hpp"
#include "../policy_p3.hpp"
#include "policy_lp.hpp"
#include "routing.hpp"

namespace fusion::precision {

inline constexpr int familyOf(const PolicyFeatures&) { return 2; }
inline constexpr int familyOf(const PolicyFeaturesP3&) { return 3; }

struct BankCounters {
    long long byProfile[3] = {0, 0, 0};
    long long missing = 0;        // requested a low-bit profile that the bank does not hold
    long long familyRejects = 0;  // a low-bit model of another feature family: refused
};

// LegacyModel: fusion::PolicyModel (p2) or fusion::PolicyModelP3 (p3).
template <class LegacyModel>
struct PolicyBank {
    const LegacyModel* reference16 = nullptr;
    const PolicyModelLP* uniform8 = nullptr;
    const PolicyModelLP* mixed4_8 = nullptr;
    uint64_t referenceIdentity = 0;   // the INT16 manifest's weights digest (low 64 bits)
    mutable BankCounters counters;

    const PolicyModelLP* lowbit(Profile p) const {
        return p == Profile::Uniform8 ? uniform8 : p == Profile::Mixed4_8 ? mixed4_8 : nullptr;
    }

    // The profile that will actually run for features of `family`.
    Profile resolve(Profile p, int family) const {
        if (p == Profile::Reference16) return p;
        const PolicyModelLP* m = lowbit(p);
        if (!m) { counters.missing++; return Profile::Reference16; }
        if (m->family != family) { counters.familyRejects++; return Profile::Reference16; }
        return p;
    }

    // Identity of the numerics a prediction depends on (for PredictionKey.model / .quant).
    uint64_t modelIdentity(Profile p) const { const auto* m = lowbit(p); return m ? m->identity : referenceIdentity; }
    uint64_t quantIdentity(Profile p) const { const auto* m = lowbit(p); return m ? m->profile : 0; }
};

template <class Bank, class F>
inline Profile predictJoint(const Bank& bank, Profile requested, const F& f, PolicyOutput& out) {
    const Profile p = bank.resolve(requested, familyOf(f));
    bank.counters.byProfile[static_cast<int>(p)]++;
    if (p == Profile::Reference16) jointPolicy(*bank.reference16, f, out);
    else jointPolicyLP(*bank.lowbit(p), f, out);
    return p;
}

template <class Bank, class F>
inline Profile predictAct(const Bank& bank, Profile requested, const F& f, int k, int move, double prob[kActs]) {
    const Profile p = bank.resolve(requested, familyOf(f));
    bank.counters.byProfile[static_cast<int>(p)]++;
    if (p == Profile::Reference16) actPolicy(*bank.reference16, f, k, move, prob);
    else actPolicyLP(*bank.lowbit(p), f, k, move, prob);
    return p;
}

template <class Bank, class F>
inline Profile predictTarget(const Bank& bank, Profile requested, const F& f, int k, int move, double prob[kCand]) {
    const Profile p = bank.resolve(requested, familyOf(f));
    bank.counters.byProfile[static_cast<int>(p)]++;
    if (p == Profile::Reference16) targetPolicy(*bank.reference16, f, k, move, prob);
    else targetPolicyLP(*bank.lowbit(p), f, k, move, prob);
    return p;
}

}  // namespace fusion::precision
#endif  // FUSION_POLICY_PRECISION_BANK_HPP
