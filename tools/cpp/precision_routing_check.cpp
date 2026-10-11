// Contracts of precision routing (task T0030; tests/test_precision.py builds and runs it). Synthetic states only:
// the four-turn search is task T0005's, and nothing here measures time on a host.
//
//   precision_routing_check        prints "ok <check>" per passing check, "FAIL <check>: <why>"; exit 1 on a failure
//
// Checks: the default selector is Reference16 for every context; the enabled routing table; prediction cache
// isolation across profile, model, family, perspective, team coverage and history parent; root immutability and
// root predictions independent of branch order and count; shortlist always keeps the incumbent; verification
// rounds are aligned (one scenario per round for every finalist), incomplete rounds are discarded whole, and
// deadline cuts keep the completed estimates or the incumbent; the bank's Reference16 path is the INT16 path
// bit for bit, and missing or wrong-family low-bit models fall back to it.
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "policy_precision/bank.hpp"

using namespace fusion;
using namespace fusion::precision;

namespace {

int failures = 0;

void check(bool ok, const char* name, const std::string& why = "") {
    if (ok) std::printf("ok %s\n", name);
    else { std::printf("FAIL %s: %s\n", name, why.c_str()); failures++; }
}

// A synthetic public state: what the cache must compare in full.
struct SynState {
    int turn = 0;
    std::vector<int> cells;
    bool operator==(const SynState& o) const { return turn == o.turn && cells == o.cells; }
};

uint64_t hashOf(const SynState& s) {
    uint64_t h = 1469598103934665603ull ^ static_cast<uint64_t>(s.turn);
    for (int c : s.cells) h = (h ^ static_cast<uint64_t>(c)) * 1099511628211ull;
    return h;
}

// The prediction a branch would compute: depends only on (state, parent, profile) through `salt`.
double predict(const SynState& s, const SynState* parent, Profile p) {
    double v = s.turn * 0.25 + (parent ? parent->turn * 0.125 : 0.0) + static_cast<int>(p) * 1000.0;
    for (int c : s.cells) v = v * 1.0001 + c;
    return v;
}

PredictionKey keyOf(const SynState& s, const SynState* parent, Profile p, uint64_t teams = 0b1110, int me = 0) {
    PredictionKey k;
    k.model = 0xA11CEull;
    k.quant = static_cast<uint64_t>(p) + 77;
    k.profile = p;
    k.family = 2;
    k.perspective = me;
    k.teams = teams;
    k.state = hashOf(s);
    k.parent = parent ? hashOf(*parent) : 0;
    return k;
}

void selectorChecks() {
    Selector off;
    bool allRef = true;
    for (int depth = 0; depth < 6; depth++)
        for (int flags = 0; flags < 64; flags++) {
            NodeContext c;
            c.depth = depth;
            c.carrierEvent = flags & 1; c.deathEvent = flags & 2; c.deliveryEvent = flags & 4;
            c.dropEvent = flags & 8; c.closeComparison = flags & 16; c.screening = flags & 32;
            allRef &= off.choose(c) == Profile::Reference16;
        }
    check(allRef, "selector_default_reference16");

    Selector on;
    on.enabled = true;
    NodeContext root;
    NodeContext screen; screen.depth = 1; screen.screening = true;
    NodeContext quiet; quiet.depth = 2;
    NodeContext event = quiet; event.carrierEvent = true;
    NodeContext close = quiet; close.closeComparison = true;
    NodeContext verify = screen; verify.finalVerification = true;
    check(on.choose(root) == Profile::Reference16 && on.choose(screen) == Profile::Uniform8 &&
              on.choose(quiet) == Profile::Mixed4_8 && on.choose(event) == Profile::Reference16 &&
              on.choose(close) == Profile::Reference16 && on.choose(verify) == Profile::Reference16,
          "selector_routing_table");
}

void cacheChecks() {
    PredictionCache<SynState, double> cache(8);
    SynState root{10, {1, 2, 3}}, parent{9, {1, 2, 2}}, other{9, {0, 2, 2}};
    const double v8 = cache.store(keyOf(root, &parent, Profile::Uniform8), root, &parent, predict(root, &parent, Profile::Uniform8));
    const double* hit = cache.find(keyOf(root, &parent, Profile::Uniform8), root, &parent);
    check(hit && *hit == v8, "cache_hit_same_numerics");
    check(!cache.find(keyOf(root, &parent, Profile::Mixed4_8), root, &parent) && cache.counters.otherNumerics == 1,
          "cache_profile_isolation");
    check(!cache.find(keyOf(root, &parent, Profile::Uniform8, 0b0110), root, &parent), "cache_team_coverage_isolation");
    check(!cache.find(keyOf(root, &parent, Profile::Uniform8, 0b1110, 1), root, &parent), "cache_perspective_isolation");
    check(!cache.find(keyOf(root, &other, Profile::Uniform8), root, &other), "cache_history_isolation");
    PredictionKey otherModel = keyOf(root, &parent, Profile::Uniform8);
    otherModel.model ^= 1;
    check(!cache.find(otherModel, root, &parent), "cache_model_isolation");
    PredictionKey otherFamily = keyOf(root, &parent, Profile::Uniform8);
    otherFamily.family = 3;
    check(!cache.find(otherFamily, root, &parent), "cache_family_isolation");
    // A hash collision must not alias two different states: same key hashes, different cells.
    SynState twin = root;
    twin.cells[0] = 99;
    PredictionKey forged = keyOf(root, &parent, Profile::Uniform8);
    check(!cache.find(forged, twin, &parent), "cache_full_state_equality");
    for (int i = 0; i < 20; i++) {
        SynState s{20 + i, {i}};
        cache.store(keyOf(s, &root, Profile::Uniform8), s, &root, predict(s, &root, Profile::Uniform8));
    }
    check(cache.size() == 8 && cache.counters.evictions > 0, "cache_bounded");
}

void rootChecks() {
    SynState root{30, {4, 5, 6}}, history{29, {4, 5, 5}};
    RootGuard<SynState> guard;
    guard.freeze(root, &history);
    const double before = predict(root, &history, Profile::Reference16);
    std::vector<double> order1, order2;
    {   // branch predictions in one order, then the other, with different counts
        PredictionCache<SynState, double> c(64);
        for (int i = 0; i < 12; i++) {
            SynState b{31, {i, i + 1}};
            const Profile p = i % 3 == 0 ? Profile::Reference16 : Profile::Uniform8;
            c.store(keyOf(b, &root, p), b, &root, predict(b, &root, p));
        }
        order1.push_back(predict(root, &history, Profile::Reference16));
    }
    {
        PredictionCache<SynState, double> c(64);
        for (int i = 20; i >= 0; i--) {
            SynState b{31, {i, i + 1}};
            c.store(keyOf(b, &root, Profile::Mixed4_8), b, &root, predict(b, &root, Profile::Mixed4_8));
        }
        order2.push_back(predict(root, &history, Profile::Reference16));
    }
    check(order1[0] == before && order2[0] == before && guard.unchanged(root, &history), "root_invariant_branch_order_count");
    SynState moved = root;
    moved.turn++;
    check(!guard.unchanged(moved, &history) && !guard.unchanged(root, nullptr), "root_guard_detects_replacement");
}

void verifyChecks() {
    const std::vector<double> cheap{0.10, 0.90, 0.85, 0.20, 0.88, 0.05};
    VerifyConfig cfg;
    cfg.keep = 2;
    cfg.rounds = 6;
    const int incumbent = 3;  // a weak cheap score: the incumbent is still a finalist
    const auto fin = shortlist(cheap, incumbent, cfg);
    check(!fin.empty() && fin[0] == incumbent && fin.size() == 3, "shortlist_keeps_incumbent");
    cfg.closeGap = 0.06;
    check(shortlist(cheap, incumbent, cfg).size() == 4, "shortlist_close_gap");  // 0.85 within 0.06 of 0.90
    cfg.closeGap = 0;

    // Aligned scenarios: every finalist sees round r before any sees r + 1.
    std::vector<std::pair<int, int>> calls;
    long long clock = 0;
    auto now = [&] { return clock; };
    auto value = [&](int cand, int round) { calls.push_back({cand, round}); clock += 10; return cand == 1 ? 1.0 + round : 0.5; };
    VerifyCounters ctr;
    auto res = verify(cheap, incumbent, cfg, value, now, 1'000'000, ctr);
    bool aligned = calls.size() == fin.size() * cfg.rounds;
    for (size_t i = 0; i < calls.size(); i++) aligned &= calls[i].second == static_cast<int>(i / fin.size());
    check(aligned && res.completedRounds == cfg.rounds && res.choice == 1 && !res.fallback, "verify_aligned_rounds");

    // A deadline inside round 3: rounds 0..2 count, round 3 is discarded whole.
    clock = 0;
    calls.clear();
    VerifyCounters c2;
    const long long deadline = static_cast<long long>(10 * fin.size() * 3 + 15);
    auto cut = verify(cheap, incumbent, cfg, value, now, deadline, c2);
    check(cut.completedRounds == 3 && c2.roundsDiscarded == 1 && c2.deadlineCuts >= 1 && cut.choice == 1, "verify_deadline_keeps_completed");

    // No completed round: the incumbent, counted as a fallback.
    clock = 0;
    VerifyCounters c3;
    auto none = verify(cheap, incumbent, cfg, value, now, 5, c3);
    check(none.fallback && none.choice == incumbent && none.completedRounds == 0 && c3.fallbacks == 1, "verify_fallback_incumbent");

    // The cost reserve stops work before an atomic estimate that would not fit.
    clock = 0;
    VerifyConfig reserved = cfg;
    reserved.reserveNs = 1'000'000;
    VerifyCounters c4;
    auto r4 = verify(cheap, incumbent, reserved, value, now, 500'000, c4);
    check(r4.fallback && c4.predictions == 0, "verify_cost_reserve");
}

// Tiny INT16 and int8 nets with the policy heads' widths, built here (no generated header).
struct TinyModels {
    std::vector<int16_t> w16[4];
    std::vector<int32_t> b16[4];
    Layer l16[4];
    std::vector<int8_t> w8[4];
    std::vector<int32_t> b8[4], mult8[4];
    std::vector<double> scale8[4];
    lowbit::Layer l8[4];
    PolicyModel legacy;
    PolicyModelLP lp;

    TinyModels() {
        const int ins[4] = {kMoverIn, kSquadIn, kMoverIn, kTargetIn}, outs[4] = {1, kJoint, kActs, 1};
        uint32_t seed = 12345;
        auto rnd = [&](int lo, int hi) { seed = seed * 1664525u + 1013904223u; return lo + static_cast<int>(seed % static_cast<uint32_t>(hi - lo + 1)); };
        for (int h = 0; h < 4; h++) {
            const int in = ins[h], out = outs[h], pad = (in + 31) / 32 * 32;
            w16[h].resize(static_cast<size_t>(in) * out);
            b16[h].resize(out);
            w8[h].assign(static_cast<size_t>(pad) * out, 0);
            b8[h].resize(out);
            scale8[h].resize(out);
            for (auto& v : w16[h]) v = static_cast<int16_t>(rnd(-300, 300));
            for (auto& v : b16[h]) v = rnd(-1000, 1000);
            for (int o = 0; o < out; o++) {
                for (int i = 0; i < in; i++) w8[h][static_cast<size_t>(o) * pad + i] = static_cast<int8_t>(rnd(-127, 127));
                b8[h][o] = rnd(-500, 500);
                scale8[h][o] = 1.0 / 4096;
            }
            l16[h] = Layer{in, out, w16[h].data(), b16[h].data(), 0, false, nullptr};
            l8[h] = lowbit::Layer{8, in, out, pad, w8[h].data(), nullptr, b8[h].data(), false, nullptr, 0, scale8[h].data()};
        }
        legacy = PolicyModel{Net{&l16[0], 1, 12}, Net{&l16[1], 1, 12}, Net{&l16[2], 1, 12}, Net{&l16[3], 1, 12}};
        lp = PolicyModelLP{lowbit::Net{&l8[0], 1}, lowbit::Net{&l8[1], 1}, lowbit::Net{&l8[2], 1}, lowbit::Net{&l8[3], 1}, 2, 0xBEEF, 0x88};
    }
};

PolicyFeatures features(uint32_t seed) {
    PolicyFeatures f{};
    auto rnd = [&](int lo, int hi) { seed = seed * 1103515245u + 12345u; return lo + static_cast<int>((seed >> 8) % static_cast<uint32_t>(hi - lo + 1)); };
    for (auto& v : f.g) v = static_cast<int16_t>(rnd(-5, 40));
    for (auto& r : f.u) for (auto& v : r) v = static_cast<int16_t>(rnd(0, 30));
    for (auto& r : f.m) for (auto& mm : r) for (auto& v : mm) v = static_cast<int16_t>(rnd(0, 12));
    for (auto& r : f.legal) for (auto& v : r) v = static_cast<unsigned char>(rnd(0, 3) != 0);
    for (auto& r : f.legal) r[0] = 1;
    for (auto& r : f.actLegal) for (auto& mm : r) { for (auto& v : mm) v = static_cast<unsigned char>(rnd(0, 1)); mm[0] = 1; }
    for (auto& r : f.cand) for (auto& v : r) v = rnd(-1, 20);
    for (auto& r : f.candF) for (auto& c : r) for (auto& v : c) v = static_cast<int16_t>(rnd(0, 20));
    return f;
}

void bankChecks() {
    static TinyModels m;
    PolicyBank<PolicyModel> bank;
    bank.reference16 = &m.legacy;
    bank.uniform8 = &m.lp;
    const Selector off;
    bool identical = true, masked = true;
    for (uint32_t s = 1; s <= 20; s++) {
        const PolicyFeatures f = features(s);
        PolicyOutput a, b;
        jointPolicy(m.legacy, f, a);
        predictJoint(bank, off.choose(NodeContext{}), f, b);
        identical &= a.logit == b.logit && a.prob == b.prob;
        double pa[kActs], pb[kActs], ta[kCand], tb[kCand];
        actPolicy(m.legacy, f, 1, 0, pa);
        predictAct(bank, Profile::Reference16, f, 1, 0, pb);
        targetPolicy(m.legacy, f, 2, 0, ta);
        predictTarget(bank, Profile::Reference16, f, 2, 0, tb);
        for (int i = 0; i < kActs; i++) identical &= pa[i] == pb[i];
        for (int i = 0; i < kCand; i++) identical &= ta[i] == tb[i];
        PolicyOutput c;
        predictJoint(bank, Profile::Uniform8, f, c);
        double sum = 0;
        for (int j = 0; j < kJoint; j++) {
            const bool legal = f.legal[0][j / 25] && f.legal[1][j / 5 % 5] && f.legal[2][j % 5];
            masked &= legal || c.prob[j] == 0.0;
            sum += c.prob[j];
        }
        masked &= sum > 1 - 1e-12 && sum < 1 + 1e-12;
    }
    check(identical, "bank_reference16_is_int16_path");
    check(masked, "lowbit_joint_legal_mask_normalized");
    const long long before = bank.counters.missing;
    PolicyOutput o;
    const PolicyFeatures f = features(99);
    check(predictJoint(bank, Profile::Mixed4_8, f, o) == Profile::Reference16 && bank.counters.missing == before + 1,
          "bank_missing_profile_falls_back");
    PolicyModelLP wrong = m.lp;
    wrong.family = 3;
    bank.uniform8 = &wrong;
    check(predictJoint(bank, Profile::Uniform8, f, o) == Profile::Reference16 && bank.counters.familyRejects == 1,
          "bank_rejects_other_family");
}

}  // namespace

int main() {
    selectorChecks();
    cacheChecks();
    rootChecks();
    verifyChecks();
    bankChecks();
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
