// Precision routing for experimental mixed-precision P (task T0030, docs/P_MIXED_PRECISION.md). Disabled by
// default: a default Selector returns Reference16 for every node, which is the existing INT16 model.
//
//   Profile          which weights compute a prediction (tensor precision)
//   NodeContext      why a prediction is requested (search-node routing): depth below the real root, important
//                    events (carrier, death, delivery, drop), close comparisons, final verification, screening
//   Selector         NodeContext -> Profile; high precision wins for every important case
//   PredictionKey    everything a cached prediction depends on that the cache lifetime does not fix: model and
//                    profile identity, feature family, perspective, requested-team coverage, state and parent
//                    (history) hashes. A hit also requires full equality of state and parent (hashes only pick).
//   PredictionCache  bounded cache of predictions keyed by PredictionKey; never mutates or replaces the root
//   ShortlistVerify  optional cheap screening, then verification of the finalists at one precision on aligned
//                    scenarios, round by round; an incomplete round is discarded whole, and with no completed
//                    round the completed incumbent is returned
//
// Depth is a routing input only: nothing here expands the search (four-turn depth is task T0005's). Entropy,
// score gaps and calibration quantiles used by a carrier as routing heuristics are not error bounds.
#ifndef FUSION_POLICY_PRECISION_ROUTING_HPP
#define FUSION_POLICY_PRECISION_ROUTING_HPP
#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace fusion::precision {

enum class Profile : uint8_t { Reference16 = 0, Uniform8 = 1, Mixed4_8 = 2 };

inline const char* profileName(Profile p) {
    switch (p) {
        case Profile::Reference16: return "reference16";
        case Profile::Uniform8: return "uniform8";
        case Profile::Mixed4_8: return "mixed4_8";
    }
    return "?";
}

struct NodeContext {
    int depth = 0;                 // plies below the real root (0: the root itself)
    bool carrierEvent = false;     // a flag carrier is involved
    bool deathEvent = false;       // a unit can die
    bool deliveryEvent = false;    // a capture is possible
    bool dropEvent = false;        // a voluntary or forced drop
    bool closeComparison = false;  // candidates within the carrier's close-score gap
    bool finalVerification = false;
    bool screening = false;        // broad successor screening
    bool important() const { return carrierEvent || deathEvent || deliveryEvent || dropEvent; }
};

struct Selector {
    bool enabled = false;                       // false: Reference16 everywhere (the existing INT16 model)
    Profile high = Profile::Reference16;        // root, important events, close comparisons, final verification
    Profile screen = Profile::Uniform8;         // broad successor screening
    Profile quietDeep = Profile::Mixed4_8;      // quiet nodes at depth >= quietDepth
    int quietDepth = 2;

    Profile choose(const NodeContext& c) const {
        if (!enabled) return Profile::Reference16;
        if (c.depth == 0 || c.finalVerification || c.closeComparison || c.important()) return high;
        if (c.depth >= quietDepth) return quietDeep;
        if (c.screening) return screen;
        return high;
    }
};

struct PredictionKey {
    uint64_t model = 0;       // checkpoint / weights identity
    uint64_t quant = 0;       // profile and quantization identity
    Profile profile = Profile::Reference16;
    int family = 2;           // feature family (2 or 3)
    int perspective = -1;     // the encoding team's opponent view: the carrier's own team id
    uint64_t teams = 0;       // requested-team coverage (bit t: team t encoded)
    uint64_t state = 0;       // hash of the full public state (hits also compare the states)
    uint64_t parent = 0;      // hash of the history parent (0: none)
    int depth = 0;            // routing context does not change numerics, but a carrier may keep it apart

    bool operator==(const PredictionKey& o) const {
        return model == o.model && quant == o.quant && profile == o.profile && family == o.family &&
               perspective == o.perspective && teams == o.teams && state == o.state && parent == o.parent;
    }
};

struct CacheCounters {
    long long hits = 0, misses = 0, stores = 0, evictions = 0;
    long long otherNumerics = 0;  // same state and history, different model/profile/family/coverage: not reused
};

// State: any copyable type; Same(a, b): full equality. The cache stores its own copies.
template <class State, class Value, class Same = std::equal_to<State>>
class PredictionCache {
public:
    explicit PredictionCache(size_t capacity = 64, Same same = Same()) : capacity_(capacity), same_(same) {}

    const Value* find(const PredictionKey& key, const State& state, const State* parent) {
        bool sameInputs = false;
        for (const auto& e : entries_) {
            const bool inputs = e.key.state == key.state && e.key.parent == key.parent && same_(e.state, state) &&
                                e.hasParent == (parent != nullptr) && (!parent || same_(e.parentState, *parent));
            if (!inputs) continue;
            if (e.key == key) {
                counters.hits++;
                return &e.value;
            }
            sameInputs = true;
        }
        if (sameInputs) counters.otherNumerics++;
        counters.misses++;
        return nullptr;
    }

    const Value& store(const PredictionKey& key, const State& state, const State* parent, Value value) {
        Entry e{key, state, parent ? *parent : State{}, parent != nullptr, std::move(value)};
        counters.stores++;
        if (entries_.size() < capacity_) {
            entries_.push_back(std::move(e));
            return entries_.back().value;
        }
        counters.evictions++;
        Entry& slot = entries_[next_++ % entries_.size()];
        slot = std::move(e);
        return slot.value;
    }

    void clear() { entries_.clear(); next_ = 0; }
    size_t size() const { return entries_.size(); }
    CacheCounters counters;

private:
    struct Entry {
        PredictionKey key;
        State state, parentState;
        bool hasParent = false;
        Value value;
    };
    size_t capacity_;
    Same same_;
    std::vector<Entry> entries_;
    size_t next_ = 0;
};

// The real root and its history are fixed for one decision: a carrier records them once and checks, after
// any branch work, that nothing replaced them (branches predict from copies with the root as parent).
template <class State, class Same = std::equal_to<State>>
class RootGuard {
public:
    void freeze(const State& root, const State* parent) {
        root_ = root;
        parent_ = parent ? std::optional<State>(*parent) : std::nullopt;
    }
    bool unchanged(const State& root, const State* parent, Same same = Same()) const {
        if (!root_ || !same(*root_, root)) return false;
        if (parent_.has_value() != (parent != nullptr)) return false;
        return !parent || same(*parent_, *parent);
    }

private:
    std::optional<State> root_, parent_;
};

// ---------------------------------------------------------------------------------- shortlist / verification

struct VerifyConfig {
    int keep = 4;                 // finalists by cheap score, besides the incumbent
    int maxFinalists = 8;
    double closeGap = 0.0;        // also keep candidates within this cheap-score gap of the best (0: off)
    int rounds = 8;               // aligned verification scenarios
    long long reserveNs = 0;      // cost reserve before each atomic prediction: an interface, not a measured value
};

struct VerifyCounters {
    long long screened = 0, finalists = 0, roundsStarted = 0, roundsCompleted = 0, roundsDiscarded = 0;
    long long predictions = 0, deadlineCuts = 0, fallbacks = 0;
};

struct VerifyResult {
    int choice = -1;              // candidate index
    int completedRounds = 0;
    bool fallback = false;        // no completed round: the incumbent
    std::vector<int> finalists;
    std::vector<double> mean;     // per finalist, over completed rounds (same count for every finalist)
};

// The finalists: the incumbent, the `keep` best cheap scores, and anything within closeGap of the best.
inline std::vector<int> shortlist(const std::vector<double>& cheap, int incumbent, const VerifyConfig& cfg) {
    std::vector<int> order(cheap.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cheap[a] > cheap[b]; });
    std::vector<int> out;
    if (incumbent >= 0 && incumbent < static_cast<int>(cheap.size())) out.push_back(incumbent);
    const double best = cheap.empty() ? 0.0 : cheap[order[0]];
    for (size_t r = 0; r < order.size() && static_cast<int>(out.size()) < cfg.maxFinalists; r++) {
        const int c = order[r];
        if (c == incumbent) continue;
        const bool top = static_cast<int>(r) < cfg.keep;
        const bool close = cfg.closeGap > 0 && best - cheap[c] <= cfg.closeGap;
        if (top || close) out.push_back(c);
    }
    return out;
}

// Verify finalists round by round. evaluate(candidate, round) runs one atomic high-precision estimate of a
// candidate under scenario `round` (the same scenario for every finalist); now() returns nanoseconds.
// The deadline is checked before each atomic estimate (with the reserve) and after it; a round that cannot
// finish is discarded whole, so every reported mean covers the same completed rounds for all finalists.
template <class Evaluate, class Now>
inline VerifyResult verify(const std::vector<double>& cheap, int incumbent, const VerifyConfig& cfg, Evaluate evaluate,
                           Now now, long long deadlineNs, VerifyCounters& counters) {
    VerifyResult res;
    res.finalists = shortlist(cheap, incumbent, cfg);
    counters.screened += static_cast<long long>(cheap.size());
    counters.finalists += static_cast<long long>(res.finalists.size());
    const size_t F = res.finalists.size();
    std::vector<double> sum(F, 0.0), round(F, 0.0);
    for (int r = 0; r < cfg.rounds; r++) {
        if (now() + cfg.reserveNs >= deadlineNs) { counters.deadlineCuts++; break; }
        counters.roundsStarted++;
        bool complete = true;
        for (size_t f = 0; f < F; f++) {
            if (now() + cfg.reserveNs >= deadlineNs) { complete = false; counters.deadlineCuts++; break; }
            round[f] = evaluate(res.finalists[f], r);
            counters.predictions++;
            if (now() > deadlineNs) { complete = false; counters.deadlineCuts++; break; }
        }
        if (!complete) { counters.roundsDiscarded++; break; }
        for (size_t f = 0; f < F; f++) sum[f] += round[f];
        res.completedRounds++;
        counters.roundsCompleted++;
    }
    res.mean.assign(F, 0.0);
    if (res.completedRounds == 0 || F == 0) {
        res.fallback = true;
        res.choice = incumbent;
        counters.fallbacks++;
        return res;
    }
    size_t best = 0;
    for (size_t f = 0; f < F; f++) {
        res.mean[f] = sum[f] / res.completedRounds;
        if (res.mean[f] > res.mean[best]) best = f;  // ties keep the earlier finalist (the incumbent first)
    }
    res.choice = res.finalists[best];
    return res;
}

}  // namespace fusion::precision
#endif  // FUSION_POLICY_PRECISION_ROUTING_HPP
