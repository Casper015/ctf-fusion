// Own-action scorer inputs at run time: the C++ twin of fusion/decision_features.py (feature version
// own-action-1, docs/RL_PLAN.md sections 3 and 4).
//
// A carrier that scores its own complete orders encodes its team's root exactly as the extractor writes the
// record row (tools/cpp/extract_features.cpp): the same encoders, the same history (TurnView of the root and the
// previous TURN block) and the same int16 values. Everything here reads the root state and its recorded history
// only; no function takes an outcome, a later block, an opponent's reply of this turn or a successor state.
//
//   Root            the record blocks one team's scorer reads (u, m, legal, act_legal, origin, cand, cand_f, v, pc, vx)
//   parseReply      mirror of fusion.protocol.parse_reply (and of the engine's Game.decode)
//   canonicalize    mirror of decision_features.canonicalize: the canonical complete order and its flags
//   markDuplicates  mirror of decision_features.mark_duplicates (the incumbent represents its key)
//   rootFeatures    mirror of decision_features.root_features (kRootDim floats)
//   candidateFeatures  mirror of decision_features.candidate_features (kCandDim floats)
//
// Normalization divides the int16 value by its spec's integer scale in double and rounds once to float, as numpy
// does (int16 / float64, then float32), so the two agree bit for bit (tests/test_own_action_parity.py).
// kIdentityDigest pins decision_features.feature_identity(layout())["digest"]: the version, every group spec
// (bounds and scales) and every block shape. A test fails when the extractor schema changes without it.
#ifndef FUSION_OWN_ACTION_HPP
#define FUSION_OWN_ACTION_HPP
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "combat.hpp"
#include "features.hpp"
#include "state.hpp"

namespace fusion::own_action {

inline constexpr const char* kVersion = "own-action-1";
inline constexpr const char* kIdentityDigest = "9ed719f238cd5d70";
inline constexpr int kLayoutVersion = 3;
inline constexpr int kPerUnit = kMoves + kActs + kM + kPC + kTF + 2;
inline constexpr int kCandDim = 3 * kPerUnit;
inline constexpr int kRootDim = 3 * kU + kV + kVX;
static_assert(kPerUnit == 48, "decision_features.PER_UNIT");
static_assert(kRootDim == 150, "decision_features.root_dim of layout 3");

// Candidate flags (decision_features cand_flags bits).
inline constexpr int kParseFail = 1;
inline constexpr int kReqMoveIllegal = 2;
inline constexpr int kActCanonicalized = 4;
inline constexpr int kTargetOutOfCap = 8;
inline constexpr int kTargetInvalid = 16;
inline constexpr int kDuplicateEffective = 32;
inline constexpr int kCanonChanged = 64;
inline constexpr int kUnsupported = kParseFail | kTargetOutOfCap | kDuplicateEffective;

inline constexpr int kNoAction = -1;  // a unit off the board
inline constexpr int kNoTarget = -1;
inline constexpr int kOutOfCapRange = 4;  // decision_features via labels._hittable_beyond_cap
inline constexpr char kActCode[kActs + 1] = "-PXA";

// One team's record row at the root (the blocks the scorer reads), as the extractor writes them.
struct Root {
    int turn = 0, team = -1, size = 0;
    PolicyFeatures pf;
    ValueFeatures vf{};
    CombatBlock pc{};
    std::array<int16_t, kVX> vx{};
    std::array<int16_t, 3> origin{};
    std::array<std::array<int16_t, kCand>, 3> cand{};
    std::vector<Unit> units;  // the root TURN block's units as read (not respawned): the out-of-cap test
};

// `v` and `cs` summarize (b, s) with the previous TURN block as history, exactly as the extractor's makeView and
// summarize; one summary serves every team.
inline void encodeRoot(const Board& b, const State& s, const TurnView& v, const CombatSummary& cs, int team, Root& out) {
    out.turn = s.turn;
    out.team = team;
    out.size = b.n;
    encodePolicy(b, s, v, team, out.pf);
    encodeValue(b, s, v, team, out.vf);
    encodeCombatPolicy(b, s, v, cs, team, out.pc);
    encodeValueExt(b, s, v, cs, team, out.vx.data());
    for (int k = 0; k < 3; k++) {
        out.origin[k] = static_cast<int16_t>(v.origin[3 * team + k]);
        for (int c = 0; c < kCand; c++) out.cand[k][c] = static_cast<int16_t>(out.pf.cand[k][c]);
    }
    out.units = s.units;
}

inline constexpr int kWait = 0, kPickup = 1, kDrop = 2, kAttack = 3;  // fusion.protocol WAIT, PICKUP, DROP, ATTACK

struct Request { int move = 0, act = kWait, target = -1; };

// fusion.protocol.parse_reply: "<turn> <m> <a> <m> <a> <m> <a>" before any '#'; false where the engine rejects it.
inline bool parseReply(const std::string& line, int turn, std::array<Request, 3>& out) {
    const std::string body = line.substr(0, line.find('#'));
    std::vector<std::string> tok;
    size_t i = 0;
    while (i < body.size()) {
        while (i < body.size() && std::isspace(static_cast<unsigned char>(body[i]))) i++;
        size_t j = i;
        while (j < body.size() && !std::isspace(static_cast<unsigned char>(body[j]))) j++;
        if (j > i) tok.push_back(body.substr(i, j - i));
        i = j;
    }
    if (tok.size() != 7 || tok[0] != std::to_string(turn)) return false;
    for (int k = 0; k < 3; k++) {
        const std::string& mv = tok[1 + 2 * k];
        const std::string& act = tok[2 + 2 * k];
        int m = -1;
        for (int x = 0; x < kMoves; x++)
            if (mv.size() == 1 && mv[0] == kMoveChar[x]) m = x;
        if (m < 0) return false;
        Request r;
        r.move = m;
        if (act == "-") r.act = kWait;
        else if (act == "P") r.act = kPickup;
        else if (act == "X") r.act = kDrop;
        else {
            for (char c : act)
                if (c < '0' || c > '9') return false;
            r.act = kAttack;
            // Python int(): any length. An id this long names no unit, so a cap keeps it out of every slot.
            long long t = 0;
            for (char c : act) t = std::min<long long>(t * 10 + (c - '0'), 1LL << 40);
            r.target = t > (1 << 30) ? (1 << 30) : static_cast<int>(t);
        }
        out[k] = r;
    }
    return true;
}

struct Canon {
    std::string key;
    std::array<int, 3> move{}, act{kNoAction, kNoAction, kNoAction}, target{kNoTarget, kNoTarget, kNoTarget},
        slot{kNoTarget, kNoTarget, kNoTarget};
    int flags = 0;
    bool supported() const { return (flags & kUnsupported) == 0; }
};

// labels._hittable_beyond_cap: an enemy on the board within Manhattan 4 of the origin, missing from the slots
// only because they were full.
inline bool hittableBeyondCap(const Root& r, int target, int origin) {
    if (target < 0 || target >= static_cast<int>(r.units.size())) return false;
    const Unit& u = r.units[target];
    if (u.team == r.team || u.x < 0) return false;
    return std::abs(u.x - origin % r.size) + std::abs(u.y - origin / r.size) <= kOutOfCapRange;
}

inline Canon canonicalize(const Root& r, const std::array<Request, 3>& reqs) {
    Canon c;
    for (int k = 0; k < 3; k++) {
        if (r.origin[k] < 0) {  // off the board: no action, whatever was written
            c.key += c.key.empty() ? "x" : "|x";
            continue;
        }
        const Request& q = reqs[k];
        const int m = r.pf.legal[k][q.move] ? q.move : 0;
        if (m != q.move) c.flags |= kReqMoveIllegal;
        int a = r.pf.actLegal[k][m][q.act] ? q.act : kWait;
        if (a != q.act) c.flags |= kActCanonicalized;
        int t = kNoTarget;
        if (a == kAttack) {
            int found = -1;
            for (int s = 0; s < kCand && found < 0; s++)
                if (r.cand[k][s] == q.target) found = s;
            if (found >= 0) {
                c.slot[k] = found;
                t = q.target;
            } else if (hittableBeyondCap(r, q.target, r.origin[k])) {
                c.flags |= kTargetOutOfCap;
                t = q.target;
            } else {
                c.flags |= kTargetInvalid;
                a = kWait;
            }
        }
        c.move[k] = m;
        c.act[k] = a;
        c.target[k] = t;
        std::string part(1, kMoveChar[m]);
        part += kActCode[a];
        if (a == kAttack) part += std::to_string(t);
        c.key += (c.key.empty() ? "" : "|") + part;
    }
    if (c.flags & (kReqMoveIllegal | kActCanonicalized | kTargetInvalid)) c.flags |= kCanonChanged;
    return c;
}

// The canonical order of a reply line; a line the engine would reject is parse_fail with key "invalid".
inline Canon canonicalizeLine(const Root& r, const std::string& line) {
    std::array<Request, 3> reqs{};
    if (!parseReply(line, r.turn, reqs)) {
        Canon c;
        c.key = "invalid";
        c.flags = kParseFail;
        return c;
    }
    return canonicalize(r, reqs);
}

// dup_of per candidate (-1 unique or first); the incumbent always represents its key. Sets kDuplicateEffective.
inline std::vector<int> markDuplicates(std::vector<Canon>& cands, int incumbent = 0) {
    std::vector<int> dup(cands.size(), -1), order;
    order.push_back(incumbent);
    for (int i = 0; i < static_cast<int>(cands.size()); i++)
        if (i != incumbent) order.push_back(i);
    std::vector<std::pair<std::string, int>> first;
    for (int i : order) {
        if (cands[i].flags & kParseFail) continue;
        int seen = -1;
        for (const auto& [key, idx] : first)
            if (key == cands[i].key) seen = idx;
        if (seen >= 0) {
            dup[i] = seen;
            cands[i].flags |= kDuplicateEffective;
        } else {
            first.emplace_back(cands[i].key, i);
        }
    }
    return dup;
}

inline float scaled(int16_t v, const FeatureSpec& s) { return static_cast<float>(static_cast<double>(v) / s.scale); }

// Our three units' unit features, then the v3 value inputs (value features, then the extension), scaled.
inline void rootFeatures(const Root& r, float* out) {
    int i = 0;
    for (int k = 0; k < 3; k++)
        for (int f = 0; f < kU; f++) out[i++] = scaled(r.pf.u[k][f], kUnitSpec[f]);
    for (int f = 0; f < kG; f++) out[i++] = scaled(r.vf[f], kGlobalSpec[f]);
    for (int k = 0; k < 3; k++)
        for (int f = 0; f < kVU; f++) out[i++] = scaled(r.vf[kG + k * kVU + f], kValueUnitSpec[f]);
    for (int f = 0; f < kVA; f++) out[i++] = scaled(r.vf[kG + 3 * kVU + f], kValueAggSpec[f]);
    for (int f = 0; f < kVX; f++) out[i++] = scaled(r.vx[f], kValueExtSpec[f]);
}

// Per unit: one-hot move and action, the move's features, its p3 combat block, the chosen target slot's features,
// then (target present, unit active). A unit off the board is all zeros.
inline void candidateFeatures(const Root& r, const Canon& c, float* out) {
    for (int i = 0; i < kCandDim; i++) out[i] = 0.0f;
    for (int k = 0; k < 3; k++) {
        if (c.act[k] == kNoAction) continue;
        float* row = out + k * kPerUnit;
        const int mv = c.move[k], a = c.act[k];
        row[mv] = 1.0f;
        row[kMoves + a] = 1.0f;
        for (int f = 0; f < kM; f++) row[kMoves + kActs + f] = scaled(r.pf.m[k][mv][f], kMoveSpec[f]);
        for (int f = 0; f < kPC; f++) row[kMoves + kActs + kM + f] = scaled(r.pc[k][mv][f], kCombatSpec[f]);
        if (c.slot[k] >= 0) {
            for (int f = 0; f < kTF; f++) row[kMoves + kActs + kM + kPC + f] = scaled(r.pf.candF[k][c.slot[k]][f], kCandSpec[f]);
            row[kPerUnit - 2] = 1.0f;
        }
        row[kPerUnit - 1] = 1.0f;
    }
}

}  // namespace fusion::own_action
#endif  // FUSION_OWN_ACTION_HPP
