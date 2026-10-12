// Decision shortlist hook of the family carriers (task T0036, docs/RL_PLAN.md section 5), off by default.
//
//   PONLY_DECISION 0  (default) every PONLY_DLOG statement compiles to nothing: no logging, no extra work, the
//                     carrier's reply is unchanged (bots/p_only/tests/test_decision_hook.py checks it).
//   PONLY_DECISION 1  log-only collection mode: the reply's orders are unchanged; the shortlist the carrier
//                     actually scored this turn is appended to the reply line after '#', which the engine's
//                     parser ignores and the match record keeps verbatim (fusion.protocol.parse_reply strips it).
//   PONLY_DECISION 2  deployed scoring with an own_action_scorer: not implemented here (no admitted model).
//
// Comment format (one line, ctf-fusion-shortlist-log-1, parsed by training/decision_branch_collect.py):
//
//   # dsl1 stage=<s> k=<K> inc=<i> root=<16 hex> surv=<n> checked=<0|1> | <order>;<beam>;<refined>;<source> | ...
//
//   order    the complete order exactly as the reply writes it ("S - U 4 L -"), deduplicated by that text;
//   beam     the carrier's own score of the order at the stage that produced the shortlist, or "na";
//   refined  the score after the carrier's refinement (v12's rollout value + beam value), or "na";
//   source   beam, choose, survivor (C08: finished the final search round), search, portfolio, incumbent;
//   stage    which scores decided: refine (v12 rollouts completed for every candidate), beam (no refinement ran),
//            cut (a deadline cut the refinement: the reply is the beam choice), v_cut (V was cut: the reply is
//            the choice without V), search (C08's sampled expectimax; only the first surv entries finished the
//            final round, so only they are comparable), choose (C08 without the search), duel (C08's two-team
//            planner: no shortlist, k=1);
//   inc      index of the reply's own order; it is always present (added with "na" scores when the carrier
//            did not score it at this stage);
//   checked  C08: the checked lookahead ran after the search and may have replaced the search's choice;
//   root     FNV-1a 64 of the turn, the scores and every unit (id, team, x, y, hp, flag, respawn_at) as read,
//            so a log can be bound to its recorded TURN block (decision_branch_collect.root_digest).
//
// Missing scores stay "na"; nothing is estimated. K is bounded by PONLY_DECISION_K (default 8): the reply's order
// and then the best-scored others are kept.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifndef PONLY_DECISION
#define PONLY_DECISION 0
#endif
#if PONLY_DECISION > 1
#error "PONLY_DECISION 2 (deployed own-action scoring) is not implemented; only 0 (off) and 1 (log-only) exist"
#endif
#ifndef PONLY_DECISION_K
#define PONLY_DECISION_K 8
#endif

#if PONLY_DECISION
#define PONLY_DLOG(...) do { __VA_ARGS__; } while (0)
#else
#define PONLY_DLOG(...) do { } while (0)
#endif

namespace ponly_decision {

struct Entry {
    std::string order;
    double beam = NAN, refined = NAN;
    std::string source;
};

inline std::string number(double v) {
    if (!std::isfinite(v)) return "na";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    return buf;
}

struct Log {
    std::vector<Entry> entries;
    std::string stage = "none";
    int survivors = -1;
    bool checked = false;

    void clear() { entries.clear(); stage = "none"; survivors = -1; checked = false; }
    int find(const std::string& order) const {
        for (size_t i = 0; i < entries.size(); i++) if (entries[i].order == order) return int(i);
        return -1;
    }
    // The first score of an order is kept: a later duplicate of the same text adds nothing.
    void add(const std::string& order, double beam, const char* source) {
        if (find(order) < 0) entries.push_back({order, beam, NAN, source});
    }
    void refined(const std::string& order, double value) {
        int i = find(order);
        if (i >= 0) entries[size_t(i)].refined = value;
    }
    // " # dsl1 ..." with the reply's order always present and at most PONLY_DECISION_K entries.
    std::string comment(const std::string& incumbent, uint64_t root) const {
        std::vector<Entry> e = entries;
        int inc = -1;
        for (size_t i = 0; i < e.size(); i++) if (e[i].order == incumbent) inc = int(i);
        if (inc < 0) { e.push_back({incumbent, NAN, NAN, "incumbent"}); inc = int(e.size()) - 1; }
        const bool useRefined = stage == "refine";
        auto key = [&](const Entry& x) { double v = useRefined ? x.refined : x.beam; return std::isfinite(v) ? v : -1e300; };
        std::vector<size_t> order(e.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if ((int)a == inc) return (int)b != inc;
            if ((int)b == inc) return false;
            return key(e[a]) > key(e[b]);
        });
        if (order.size() > size_t(PONLY_DECISION_K)) order.resize(size_t(PONLY_DECISION_K));
        char head[160];
        std::snprintf(head, sizeof head, " # dsl1 stage=%s k=%d inc=0 root=%016llx surv=%d checked=%d", stage.c_str(),
                      int(order.size()), (unsigned long long)root, survivors, checked ? 1 : 0);
        std::string out = head;
        for (size_t i : order) out += " | " + e[i].order + ";" + number(e[i].beam) + ";" + number(e[i].refined) + ";" + e[i].source;
        return out;
    }
};

inline void fnv(uint64_t& h, const std::string& s) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
}

// The root as read: turn, scores, units. Same text and order as decision_branch_collect.root_digest.
template <class G>
uint64_t rootDigest(const G& g) {
    uint64_t h = 1469598103934665603ull;
    std::string s = std::to_string(g.turn) + "|";
    for (size_t i = 0; i < g.score.size(); i++) s += (i ? "," : "") + std::to_string(g.score[i]);
    s += "|";
    for (const auto& u : g.units)
        s += std::to_string(u.id) + "," + std::to_string(u.team) + "," + std::to_string(u.x) + "," + std::to_string(u.y) + "," +
             std::to_string(u.hp) + "," + std::to_string(u.flag) + "," + std::to_string(u.respawn_at) + ";";
    fnv(h, s);
    return h;
}

}  // namespace ponly_decision
