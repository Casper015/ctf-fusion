// Deployed own-action scoring of the family carriers: PONLY_DECISION 2 (task T0045, docs/OWN_ACTION_RUNTIME.md).
//
// Only the scoring carriers (v12_ponly_fs.hpp, bots/p_only/tools/derive/make_scoring.py) include this header, in
// place of decision_hook.hpp, which stays byte-identical (T0037's collector identity hashes it):
//
//   PONLY_DECISION 0 or 1   exactly decision_hook.hpp: off, or log-only collection.
//   PONLY_DECISION 2        needs an exported scorer (training/export_decision_scorer.py) included first. With its
//                           lambda = 0 the whole hook compiles to nothing (PONLY_DECISION 0: no logging, no feature
//                           encoding, no scoring; the production reply). Otherwise the shortlist machinery of
//                           decision_hook.hpp collects the orders the carrier scored (nothing is appended to the
//                           reply) and Hook::decide may replace the reply's order by a shortlisted one.
//
// Hook::decide reproduces the offline pipeline on the turn's own shortlist:
//   1. the logged shortlist as decision_hook.hpp Log::comment orders it (the reply's order first, then the others
//      by the deciding stage's score, at most PONLY_DECISION_K);
//   2. deciding scores as training/decision_branch_collect.deciding_scores, as logged (9 significant digits):
//      refine -> refined, beam / choose -> beam;
//      any other stage (cut, v_cut, none) has none and keeps the reply; so does an unscored reply or K = 1;
//   3. candidates as decision_branch_collect.select_candidates: the reply's order, then the PONLY_DSCORE_ALTERNATIVES
//      best others by (-score, order text);
//   4. canonical orders, effective duplicates and support (inference/own_action.hpp, as fusion/decision_features);
//   5. the rule of training/evaluate_decisions.apply_rule for one root: gains g = Q(c) - Q(reply), NaN outside the
//      support and for every alternative of an unsupported reply; zscore (the carrier's scores z-scored over the
//      rule's candidates, population sd, the reply raised to their top) or incumbent_override (1 for the reply, 0
//      otherwise); argmax of S + lambda g; ties within 1e-12 go to the reply, then to the smallest canonical key.
// The reply stands on an invalid model (validate), a missing deciding score, K = 1, or the deadline:
// PONLY_DSCORE_DEADLINE_MS after the turn began (default 40 of the 50 ms reply limit; not an extra budget: the
// carrier's own budgets are unchanged, and the host measures the real cost before any screen).
//
// Identity: the scorer must have been trained on data of PONLY_DSCORE_DATA_CARRIER (the collection carrier whose
// shortlist this build reproduces, e.g. "v12_ponly_xxl_dlog"), with this build's opponent-P family and V setting;
// otherwise the build fails (Hook::check). An unbound test export builds only with PONLY_DSCORE_ALLOW_UNBOUND.
//
// Counters, one stderr line at the last turn: DSCORE roots scored switched no_scores k1 deadline invalid candidates
// unsupported. PONLY_DSCORE_TRACE appends the logged shortlist and " ## dsc1 ..." to every reply (tests only: the
// engine ignores everything after '#').
#pragma once
#ifndef PONLY_DECISION
#define PONLY_DECISION 0
#endif
#if PONLY_DECISION == 2
#ifndef FUSION_OWN_ACTION_SCORER_MODEL_HPP
#error "PONLY_DECISION 2 needs an exported own-action scorer: include its scorer_model.hpp before the carrier"
#endif
#undef PONLY_DECISION
#if OWN_ACTION_SCORER_LAMBDA_ZERO
#define PONLY_DECISION 0  // lambda = 0: the reply without any scoring, logging or encoding
#define PONLY_DSCORE 0
#else
#define PONLY_DECISION 1  // decision_hook.hpp's Log collects the scored shortlist; nothing is appended to the reply
#define PONLY_DSCORE 1
#endif
#else
#define PONLY_DSCORE 0
#endif
#include "decision_hook.hpp"

#if PONLY_DSCORE
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include "own_action_scorer.hpp"

#ifndef PONLY_DSCORE_DATA_CARRIER
#error "PONLY_DECISION 2: define PONLY_DSCORE_DATA_CARRIER, the collection carrier whose shortlist this build reproduces"
#endif
#ifndef PONLY_DSCORE_ALTERNATIVES
#define PONLY_DSCORE_ALTERNATIVES 3  // decision_branch_collect --alternatives
#endif
#ifndef PONLY_DSCORE_DEADLINE_MS
#define PONLY_DSCORE_DEADLINE_MS 40
#endif
#ifndef PONLY_DSCORE_TRACE
#define PONLY_DSCORE_TRACE 0
#endif

namespace ponly_dscore {

namespace oa = fusion::own_action;
inline constexpr double kTie = 1e-12;  // training/evaluate_decisions.TIE

// Whether the exported scorer may run in a build with this opponent-P family and V setting.
constexpr bool bound(const oa::ScorerModel& m, int pFamily, int value) {
#ifdef PONLY_DSCORE_ALLOW_UNBOUND
    if (std::string_view(m.carrierStatus) == "unbound") return true;
#endif
    const char family[3] = {'p', char('0' + pFamily), 0};
    return std::string_view(m.carrierStatus) != "unbound" && std::string_view(m.carrierVariant) == PONLY_DSCORE_DATA_CARRIER &&
           std::string_view(m.carrierPFamily) == std::string_view(family, 2) && m.carrierValue == value;
}

struct Counters {
    long long roots = 0, scored = 0, switched = 0, noScores = 0, k1 = 0, deadline = 0, invalid = 0, candidates = 0,
              unsupported = 0;
};

struct Entry { std::string order; double score; };

class Hook {
public:
    Counters counters;

    // The reply line `out` ("<turn> <order>"), possibly replaced by a shortlisted order. `session` is the carrier's
    // PSession (board, rules, the root view with its history); `log` the turn's shortlist; `start` the turn's start.
    template <class GameT, class Session>
    std::string decide(const GameT& g, const Session& session, const ponly_decision::Log& log, const std::string& out,
                       std::chrono::steady_clock::time_point start) {
        counters.roots++;
        std::string trace;
        std::string result = choose(g, session, log, out, start, trace);
        if (g.turn == g.turns)
            std::cerr << "DSCORE roots " << counters.roots << " scored " << counters.scored << " switched " << counters.switched
                      << " no_scores " << counters.noScores << " k1 " << counters.k1 << " deadline " << counters.deadline
                      << " invalid " << counters.invalid << " candidates " << counters.candidates << " unsupported "
                      << counters.unsupported << std::endl;
        // Tests only: the shortlist exactly as the log-only carrier would have logged it for the reply's own order,
        // then this decision.
        if (PONLY_DSCORE_TRACE)
            result += log.comment(out.substr(out.find(' ') + 1), ponly_decision::rootDigest(g)) + " ## dsc1 " + trace;
        return result;
    }

private:
    int valid_ = -1;  // validate() once

    static bool late(std::chrono::steady_clock::time_point start) {
        return std::chrono::steady_clock::now() - start > std::chrono::milliseconds(PONLY_DSCORE_DEADLINE_MS);
    }

    // decision_hook.hpp Log::comment's order: the reply first, then by the stage's score (stable), at most K.
    static std::vector<ponly_decision::Entry> logged(const ponly_decision::Log& log, const std::string& reply) {
        std::vector<ponly_decision::Entry> e = log.entries;
        int inc = -1;
        for (size_t i = 0; i < e.size(); i++) if (e[i].order == reply) inc = int(i);
        if (inc < 0) { e.push_back({reply, NAN, NAN, "incumbent"}); inc = int(e.size()) - 1; }
        const bool useRefined = log.stage == "refine";
        auto key = [&](const ponly_decision::Entry& x) { double v = useRefined ? x.refined : x.beam; return std::isfinite(v) ? v : -1e300; };
        std::vector<size_t> order(e.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if ((int)a == inc) return (int)b != inc;
            if ((int)b == inc) return false;
            return key(e[a]) > key(e[b]);
        });
        if (order.size() > size_t(PONLY_DECISION_K)) order.resize(size_t(PONLY_DECISION_K));
        std::vector<ponly_decision::Entry> out;
        for (size_t i : order) out.push_back(e[i]);
        return out;
    }

    template <class GameT, class Session>
    std::string choose(const GameT& g, const Session& session, const ponly_decision::Log& log, const std::string& out,
                       std::chrono::steady_clock::time_point start, std::string& trace) {
        const oa::ScorerModel& m = oa::exported::kScorer;
        if (valid_ < 0) valid_ = oa::validate(m) ? 1 : 0;
        if (!valid_) { counters.invalid++; trace = "invalid"; return out; }
        const std::string reply = out.substr(out.find(' ') + 1);
        // 2. deciding scores (decision_branch_collect.deciding_scores)
        const bool refine = log.stage == "refine", beam = log.stage == "beam" || log.stage == "choose";
        if (!refine && !beam) { counters.noScores++; trace = "no_scores stage=" + log.stage; return out; }
        std::vector<Entry> keep;
        for (const auto& e : logged(log, reply)) {
            const double v = refine ? e.refined : e.beam;
            // The collector parses the logged text (9 significant digits): the rule sees exactly those values.
            if (std::isfinite(v)) keep.push_back({e.order, std::strtod(ponly_decision::number(v).c_str(), nullptr)});
        }
        if (keep.empty() || keep[0].order != reply) { counters.noScores++; trace = "no_scores reply_unscored"; return out; }
        if (keep.size() < 2) { counters.k1++; trace = "k1"; return out; }
        // 3. the reply, then the best others by (-score, order)
        std::vector<Entry> cand{keep[0]};
        std::vector<Entry> rest(keep.begin() + 1, keep.end());
        std::sort(rest.begin(), rest.end(), [](const Entry& a, const Entry& b) {
            return a.score != b.score ? a.score > b.score : a.order < b.order;
        });
        for (size_t i = 0; i < rest.size() && i < size_t(PONLY_DSCORE_ALTERNATIVES); i++) cand.push_back(rest[i]);
        if (late(start)) { counters.deadline++; trace = "deadline"; return out; }
        // 4. features of the root and canonical orders
        fusion::CombatSummary cs;
        const auto& view = session.view;
        fusion::summarize(session.board, session.rules, view.state(), view.turnView(), cs);
        oa::Root root;
        oa::encodeRoot(session.board, view.state(), view.turnView(), cs, g.me, root);
        std::vector<oa::Canon> canon;
        for (const auto& c : cand) canon.push_back(oa::canonicalizeLine(root, std::to_string(g.turn) + " " + c.order));
        oa::markDuplicates(canon, 0);
        const int n = int(cand.size());
        // 5. gains of the supported candidates (mask_gains), then the rule
        std::vector<double> gain(n, NAN);
        gain[0] = 0.0;
        if (canon[0].supported()) {
            std::array<float, oa::kRootDim> rx{};
            std::array<float, oa::kCandDim> cx{};
            oa::rootFeatures(root, rx.data());
            oa::candidateFeatures(root, canon[0], cx.data());
            const double q0 = oa::q(m, rx.data(), cx.data());
            for (int i = 1; i < n; i++) {
                if (!canon[i].supported()) continue;
                if (late(start)) { counters.deadline++; trace = "deadline"; return out; }
                oa::candidateFeatures(root, canon[i], cx.data());
                gain[i] = oa::q(m, rx.data(), cx.data()) - q0;
            }
        }
        std::vector<char> ok(n, 0);
        int unsupported = 0;
        for (int i = 0; i < n; i++) {
            ok[i] = i == 0 || std::isfinite(gain[i]);
            unsupported += !canon[i].supported();
        }
        std::vector<double> s(n, 0.0);
        if (std::string_view(m.norm) == "incumbent_override") {
            s[0] = 1.0;
        } else {
            int count = 0;
            double mean = 0.0;
            for (int i = 0; i < n; i++) if (ok[i]) { mean += cand[i].score; count++; }
            if (count >= 2) {
                mean /= count;
                double var = 0.0;
                for (int i = 0; i < n; i++) if (ok[i]) var += (cand[i].score - mean) * (cand[i].score - mean);
                const double sd = std::sqrt(var / count);
                double top = -1e300;
                for (int i = 0; i < n; i++)
                    if (ok[i]) { s[i] = sd > 0 ? (cand[i].score - mean) / sd : 0.0; top = std::max(top, s[i]); }
                s[0] = top;
            }
        }
        double best = -1e300;
        std::vector<double> score(n, -1e300);
        for (int i = 0; i < n; i++)
            if (ok[i]) { score[i] = s[i] + m.lambda * (std::isfinite(gain[i]) ? gain[i] : 0.0); best = std::max(best, score[i]); }
        int chosen = 0;
        if (score[0] < best - kTie) {
            chosen = -1;
            for (int i = 1; i < n; i++)
                if (ok[i] && score[i] >= best - kTie && (chosen < 0 || canon[i].key < canon[chosen].key)) chosen = i;
        }
        counters.scored++;
        counters.candidates += n;
        counters.unsupported += unsupported;
        counters.switched += chosen != 0;
        if (PONLY_DSCORE_TRACE) {
            char buf[64];
            trace = "stage=" + log.stage + " chosen=" + std::to_string(chosen);
            for (int i = 0; i < n; i++) {
                std::snprintf(buf, sizeof buf, ";%.17g;", cand[i].score);
                trace += " | " + cand[i].order + buf;
                std::snprintf(buf, sizeof buf, "%.17g", gain[i]);
                trace += std::string(std::isfinite(gain[i]) ? buf : "na") + ";" + canon[i].key;
            }
        }
        return chosen == 0 ? out : std::to_string(g.turn) + " " + cand[chosen].order;
    }
};

}  // namespace ponly_dscore
#endif  // PONLY_DSCORE
