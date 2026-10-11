"""The joint-action search experiment of task T0018 (docs/SEARCH_PORTFOLIO.md), applied by make_families.py to the
family carriers. Every change is behind its own build setting, and with every setting at its default the
carriers reply exactly as before (bots/p_only/tests/test_search_portfolio.py checks it on recorded games):

  PONLY_PORTFOLIO_K      0 (off) or K >= 2: up to K complete action assignments per shortlisted joint movement,
                         from a k-best version of the carrier's own 8-mask attack/pickup DP (alternate targets,
                         attack vs pickup, explicit wait vs pickup); a shooter that adds nothing to a target is
                         not assigned to it (no unintended overkill). C08 keeps its gated own-drop candidates
                         (appendOwnDrops) for the extended set; v12 has no drop alternatives (its evaluation has
                         no possession-loss accounting). Alternatives are added to the search, never displace a
                         base plan, and are deduplicated by complete order (movement and every action).
  PONLY_PORTFOLIO_MOVES  shortlisted movements, the incumbent's first (default 3).
  PONLY_PLY2_MODE        C08 sampled search: 0 the reference plan's ply-2 prediction shared by every plan of a
                         sample (current), 1 each plan's ply-2 prediction from its own settled successor,
                         2 diagnostic: play as 0, and also compute 1 to compare predictions and rankings.
  PONLY_PLY2_CACHE       1: in modes 1 and 2, P views of identical successor states (every field of the public
                         state, same root parent, same turn) are built once per search call.
  PONLY_TURN_MS / PONLY_REFINE_MS   the C08 turn deadline and the v12 refinement deadline (35 and 24 ms;
                         tests shrink them to check the fallback).

Corrections that leave default replies unchanged (the defaults never produce the affected codes):
  - v12 serialized every negative target of an unladen unit as a pickup; an explicit wait (-1) is now "-" and
    an own drop (-3) "X" (ponlyActionText);
  - C08's sampled rollout credited a pickup to any unladen unit with a negative target; only a pickup order
    (-2) picks up in the engine.
"""


def apply(text: str, edits, name: str) -> str:
    import sys  # noqa: PLC0415
    for old, new, count in edits:
        n = text.count(old)
        if n != count:
            sys.exit(f"{name}: expected {count} occurrence(s) of {old[:70]!r}, found {n}")
        text = text.replace(old, new)
    return text


SETTINGS = """#ifndef PONLY_PORTFOLIO_K
#define PONLY_PORTFOLIO_K 0  // 0: the current search; K >= 2: complete action assignments per shortlisted movement
#endif
#ifndef PONLY_PORTFOLIO_MOVES
#define PONLY_PORTFOLIO_MOVES 3  // shortlisted joint movements, the incumbent's first
#endif
#ifndef PONLY_PLY2_MODE
#define PONLY_PLY2_MODE 0  // C08: 0 reference-plan ply-2 predictions, 1 per branch, 2 diagnostic (plays as 0)
#endif
#ifndef PONLY_PLY2_CACHE
#define PONLY_PLY2_CACHE 1
#endif
#ifndef PONLY_PLY2_CACHE_SIZE
#define PONLY_PLY2_CACHE_SIZE 64
#endif
#ifndef PONLY_TURN_MS
#define PONLY_TURN_MS 35
#endif
#ifndef PONLY_REFINE_MS
#define PONLY_REFINE_MS 24
#endif
// Tests only: act as if v12's refinement deadline passed from candidate PONLY_TEST_CUT_AT on.
#ifdef PONLY_TEST_CUT_AT
#define PONLY_CUT_TEST(i) ((size_t)(i) >= (size_t)PONLY_TEST_CUT_AT)
#else
#define PONLY_CUT_TEST(i) false
#endif
#include <string>
// The protocol action of one unit for a search target code: an enemy id attacks, -2 picks up (a carrier
// cannot: wait), -3 drops (only a carrier: else wait), -1 and anything else wait. Never a pickup for -1.
inline std::string ponlyActionText(int target, bool carrying) {
    if (target >= 0) return std::to_string(target);
    if (target == -2) return carrying ? "-" : "P";
    if (target == -3) return carrying ? "X" : "-";
    return "-";
}
"""

PORTFOLIO_STATS = """
// Portfolio counters (PONLY_PORTFOLIO_K): movements re-evaluated, complete assignments kept, explicit waits,
// overkill subsets skipped, alternatives entering the sampled search / the refinement, refinement candidate
// sets with two orders of the same movement, deadline cuts inside the alternatives.
struct PortfolioStats { long long calls = 0, movements = 0, assignments = 0, waits = 0, overkill = 0, searched = 0, refined = 0, sameMove = 0, cut = 0; };

"""


def kbest_tail(position: bool) -> str:
    pv = ",positionValue" if position else ""
    return f"""        // k-best version of the attack/pickup DP above: per used-shooter mask, the K best distinct target
        // assignments. A subset with a shooter that adds nothing to the target's utility is skipped: that
        // shooter is left for other targets or a pickup (no unintended overkill).
        struct KCell {{ double value; std::array<int,3> target; }};
        std::array<std::vector<KCell>,8> kd;
        kd[0].push_back(KCell{{0.0,{{-2,-2,-2}}}});
        auto keep=[&](std::vector<KCell>& cell,const KCell& x){{
            for(const KCell& y:cell) if(y.target==x.target) return;
            cell.push_back(x);
            std::stable_sort(cell.begin(),cell.end(),[](const KCell& p,const KCell& q){{return p.value>q.value;}});
            if((int)cell.size()>K) cell.pop_back();
        }};
        for(int e=0;e<E;++e) if(alive(enemies[e].u) && !enemies[e].u.immune) {{
            auto next=kd;
            for(int used=0;used<8;++used) for(const KCell& from:kd[used]) {{
                int remaining=armedMask&~used;
                for(int sub=remaining;sub;sub=(sub-1)&remaining) {{
                    if(utilities[e][sub]<=1e-8) continue;
                    bool idle=false;
                    for(int i=0;i<3;++i) if((sub&(1<<i)) && (sub&~(1<<i)) && utilities[e][sub&~(1<<i)]>=utilities[e][sub]-1e-8) idle=true;
                    if(idle) {{ ps.overkill++; continue; }}
                    KCell x=from;
                    x.value+=utilities[e][sub];
                    for(int i=0;i<3;++i) if(sub&(1<<i)) x.target[i]=enemies[e].u.id;
                    keep(next[used|sub],x);
                }}
            }}
            kd=next;
        }}
        std::vector<Choice> local;
        for(int used=0;used<8;++used) for(const KCell& cell:kd[used]) {{
            double action=cell.value;
            std::array<int,3> t=cell.target;
            for(int i=0;i<3;++i) if(!(used&(1<<i)) && own[i].flag<0)
                action+=options[i][indices[i]].pickupValue*success[i]*survival[i];
            for(int i=0;i<3;++i) if(own[i].flag>=0) t[i]=-1;
            local.push_back(Choice{{indices,t,value+action{pv}}});
            // Wait instead of picking up where a pickup is possible: the unit stays unladen (and can fire next turn).
            for(int i=0;i<3;++i) if(!(used&(1<<i)) && own[i].flag<0 && alive(own[i]) && options[i][indices[i]].pickupValue>0) {{
                Choice w=local.back();
                w.target[i]=-1;
                w.value-=options[i][indices[i]].pickupValue*success[i]*survival[i];
                local.push_back(w);
                ps.waits++;
            }}
        }}
        std::stable_sort(local.begin(),local.end(),[](const Choice& p,const Choice& q){{return p.value>q.value;}});
        int kept=0;
        for(const Choice& c:local) {{
            if(kept>=K) break;
            bool dup=false;
            for(const Choice& o:alts) if(o.moveIndex==c.moveIndex && o.target==c.target) dup=true;
            if(dup) continue;
            alts.push_back(c);
            ++kept;
            ps.assignments++;
        }}
    }}
}}

"""


def portfolio_function(text: str, carrier: str) -> str:
    """choosePortfolio: the carrier's own choose() restricted to a movement shortlist, with the k-best DP tail."""
    start = text.index("inline Choice choose(")
    end = text.index("} // namespace ctfcombat")
    func = text[start:end]
    sig_old = ("const Params& params = Params{}, std::vector<Choice>* beam = nullptr) {" if carrier == "v12" else
               "const Params& params = Params{}, std::vector<Choice>* beam=nullptr, const std::chrono::steady_clock::time_point* deadline=nullptr,\n"
               "        std::vector<Choice>* all=nullptr) {")
    head_old = ("    Choice best;Choice bestVolley;\n    if(options[0].empty() || options[1].empty() || options[2].empty()) return best;\n"
                if carrier == "v12" else
                "    Choice best;\n    if(options[0].empty() || options[1].empty() || options[2].empty()) return best;\n")
    edits = [
        ("inline Choice choose(", "inline void choosePortfolio(", 1),
        (sig_old, "const Params& params, const std::vector<std::array<int,3>>& shortlist, int K,\n"
                  "        std::vector<Choice>& alts, PortfolioStats& ps) {", 1),
        (head_old, "    ps.calls++;\n    if(options[0].empty() || options[1].empty() || options[2].empty()) return;\n", 1),
        ("        std::array<int,3> indices{a,b,c};\n", "        std::array<int,3> indices{a,b,c};\n"
         "        if(std::find(shortlist.begin(),shortlist.end(),indices)==shortlist.end()) continue;\n        ps.movements++;\n", 1),
    ]
    if carrier == "c08":
        edits.append(("        if(deadline&&std::isfinite(best.value)&&((a*25+b*5+c)&15)==0&&std::chrono::steady_clock::now()>=*deadline)return best;\n", "", 1))
    func = apply(func, edits, f"{carrier} choose")
    cut = func.index("        std::array<detail::DpCell,8> dp;")
    return PORTFOLIO_STATS + "// Generated from choose() by bots/p_only/tools/derive/search_experiment.py (PONLY_PORTFOLIO_K).\n" + \
        func[:cut] + kbest_tail(carrier == "c08")


SHORTLIST = """            std::vector<std::array<int,3>> shortlist{{choice.moveIndex}};
            std::vector<ctfcombat::Choice> pool={pool};
            std::stable_sort(pool.begin(),pool.end(),[](const ctfcombat::Choice& p,const ctfcombat::Choice& q){{return p.value>q.value;}});
            for(const auto& c:pool){{if((int)shortlist.size()>=PONLY_PORTFOLIO_MOVES)break;
                if(std::find(shortlist.begin(),shortlist.end(),c.moveIndex)==shortlist.end())shortlist.push_back(c.moveIndex);}}
            portfolioAlts.clear();
            ctfcombat::choosePortfolio(g.map,own,options,foes,params,shortlist,PONLY_PORTFOLIO_K,portfolioAlts,portStats);
"""

PORTFOLIO_REPORT = """    // Portfolio counters (PONLY_STATS: one stderr line at the last turn).
    void portfolioReport(const Game& g)const{
#if PONLY_STATS && PONLY_PORTFOLIO_K
        if(g.turn==g.turns)std::cerr<<"PORTFOLIO k "<<PONLY_PORTFOLIO_K<<" moves "<<PONLY_PORTFOLIO_MOVES<<" calls "<<portStats.calls
            <<" movements "<<portStats.movements<<" assignments "<<portStats.assignments<<" waits "<<portStats.waits<<" overkill_skipped "<<portStats.overkill
            <<" searched "<<portStats.searched<<" refined "<<portStats.refined<<" same_move "<<portStats.sameMove<<" cut "<<portStats.cut<<std::endl;
#else
        (void)g;
#endif
    }
"""


def v12(text: str) -> str:
    text = apply(text, [
        ("#define PONLY_V_DEADLINE_MS 1e9\n#endif\n", "#define PONLY_V_DEADLINE_MS 1e9\n#endif\n" + SETTINGS, 1),
        ("} // namespace ctfcombat", portfolio_function(text, "v12") + "} // namespace ctfcombat", 1),
        ("    ponly::View branchView;  // the network's view of each simulated state in refine()\n",
         "    ponly::View branchView;  // the network's view of each simulated state in refine()\n"
         "    ctfcombat::PortfolioStats portStats;              // PONLY_PORTFOLIO_K\n"
         "    std::vector<ctfcombat::Choice> portfolioAlts;     // this turn's alternatives, distinct complete orders\n" + PORTFOLIO_REPORT, 1),
        ("        auto deadline=chrono::steady_clock::now()+chrono::milliseconds(24);Sim start=initialSim(g);",
         "#if PONLY_PORTFOLIO_K\n"
         "        // Alternatives join after the base candidates, deduplicated by complete order.\n"
         "        const size_t baseCount=beam.size();\n"
         "        for(const auto& a:portfolioAlts){bool dup=false;for(const auto& b:beam)dup=dup||(b.moveIndex==a.moveIndex&&b.target==a.target);\n"
         "            if(!dup){beam.push_back(a);portStats.refined++;}}\n"
         "        {int same=0;for(size_t i=0;i<beam.size();i++)for(size_t j=i+1;j<beam.size();j++)same+=beam[i].moveIndex==beam[j].moveIndex;portStats.sameMove+=same>0;}\n"
         "#endif\n"
         "        auto deadline=chrono::steady_clock::now()+chrono::milliseconds(PONLY_REFINE_MS);Sim start=initialSim(g);", 1),
        ("                if(chrono::steady_clock::now()>deadline)return fallback;",
         "                if(chrono::steady_clock::now()>deadline||PONLY_CUT_TEST(&candidate-beam.data())){\n"
         "#if PONLY_PORTFOLIO_K\n"
         "                    // Inside the alternatives every base candidate has been compared: keep the best so far.\n"
         "                    if(size_t(&candidate-beam.data())>=baseCount&&best>-1e99){portStats.cut++;return answer;}\n"
         "#endif\n"
         "                    return fallback;}", 1),
        ("        ctfcombat::Params params;vector<ctfcombat::Choice> beam;auto choice=ctfcombat::choose(g.map,own,options,foes,params,&beam);"
         "choice=refine(g,jobs,options,beam,choice);",
         "        ctfcombat::Params params;vector<ctfcombat::Choice> beam;auto choice=ctfcombat::choose(g.map,own,options,foes,params,&beam);\n"
         "#if PONLY_PORTFOLIO_K\n        {\n" + SHORTLIST.format(pool="beam") +
         "            // Orders already in the beam stay there once.\n"
         "            portfolioAlts.erase(std::remove_if(portfolioAlts.begin(),portfolioAlts.end(),[&](const ctfcombat::Choice& a){\n"
         "                for(const auto& b:beam)if(b.moveIndex==a.moveIndex&&b.target==a.target)return true;return false;}),portfolioAlts.end());\n"
         "        }\n#endif\n"
         "        choice=refine(g,jobs,options,beam,choice);", 1),
        ("if(target>=0)out+=to_string(target);else out+=(g.units[3*g.me+k].flag>=0?\"-\":\"P\");}",
         "out+=ponlyActionText(target,g.units[3*g.me+k].flag>=0);}", 1),
        ("        vTurnBegin();fSession.update(g);fSession.report(g);valueReport(g);vBudgetReport(g);",
         "        vTurnBegin();fSession.update(g);fSession.report(g);valueReport(g);vBudgetReport(g);portfolioReport(g);", 1),
    ], "v12_ponly_f.hpp")
    return text


PLY2_CACHE = """        // PONLY_PLY2_MODE 1/2: P views of successor states, built once per identical state within this search
        // call (setRoot clears the keys; the views' buffers are reused). Identity is every field of the public
        // state (turn, scores, units with HP, flags and respawn times, flags, last attacks); the parent is
        // always this turn's root and the model and perspective are fixed within a call.
        mutable std::vector<fusion::State> cacheKey;mutable std::vector<ponly::View> cacheView;mutable size_t cacheNext=0;
#if PONLY_PRECISION
        // Precision (task T0030): a successor view is reused only at the profile it was built with. The selector is
        // disabled by default (Reference16 everywhere); successors are depth-1 screening nodes for its routing.
        fusion::precision::Selector precisionSelector{PONLY_PRECISION_ROUTE!=0};mutable std::vector<fusion::precision::Profile> cacheProfile;
        fusion::precision::Profile successorProfile()const{fusion::precision::NodeContext c;c.depth=1;c.screening=true;return precisionSelector.choose(c);}
#endif
        mutable long long viewCalls=0,viewHits=0,viewBuilds=0;
        static bool sameState(const fusion::State& a,const fusion::State& b){
            if(a.turn!=b.turn||a.score!=b.score||a.units.size()!=b.units.size()||a.flags.size()!=b.flags.size()||a.attacks.size()!=b.attacks.size())return false;
            for(size_t i=0;i<a.units.size();i++){const auto& u=a.units[i];const auto& v=b.units[i];
                if(u.id!=v.id||u.team!=v.team||u.x!=v.x||u.y!=v.y||u.hp!=v.hp||u.flag!=v.flag||u.respawnAt!=v.respawnAt)return false;}
            for(size_t i=0;i<a.flags.size();i++){const auto& f=a.flags[i];const auto& h=b.flags[i];
                if(f.id!=h.id||f.status!=h.status||f.x!=h.x||f.y!=h.y||f.holder!=h.holder||f.returnAt!=h.returnAt)return false;}
            for(size_t i=0;i<a.attacks.size();i++)if(a.attacks[i].unit!=b.attacks[i].unit||a.attacks[i].target!=b.attacks[i].target)return false;
            return true;
        }
        ponly::View& viewOf(const fusion::State& st)const{
            viewCalls++;
#if PONLY_PLY2_MODE != 0
#if PONLY_PLY2_CACHE
#if PONLY_PRECISION
            const auto want=successorProfile();
            for(size_t i=0;i<cacheKey.size();i++)if(sameState(cacheKey[i],st)&&cacheProfile[i]==want){viewHits++;return cacheView[i];}
#else
            for(size_t i=0;i<cacheKey.size();i++)if(sameState(cacheKey[i],st)){viewHits++;return cacheView[i];}
#endif
#endif
            viewBuilds++;size_t slot;
            if(cacheKey.size()<PONLY_PLY2_CACHE_SIZE){slot=cacheKey.size();cacheKey.push_back(st);if(cacheView.size()<=slot)cacheView.emplace_back();}
            else{slot=cacheNext++%cacheKey.size();cacheKey[slot]=st;}
#if PONLY_PRECISION
            if(cacheProfile.size()<=slot)cacheProfile.resize(slot+1);
            cacheProfile[slot]=successorProfile();cacheView[slot].profile=cacheProfile[slot];
#endif
            fs->predictInto(st,root.me,cacheView[slot],&teams);return cacheView[slot];
#else
            viewBuilds++;
#if PONLY_PRECISION
            scratch.profile=successorProfile();
#endif
            fs->predictInto(st,root.me,scratch,&teams);return scratch;
#endif
        }
"""

RANK_START = "    std::vector<int> pick(E); std::vector<double> tmp(C); Ply1Outcome p1;\n"
RANK_END = "    return ranked;\n}\n"
CLOCK = """// Time source of search::rank: steady_clock, or a test's clock (bots/p_only/tests/rank_deadline_test.cpp).
inline std::chrono::steady_clock::time_point (*clockHook)() = nullptr;
inline std::chrono::steady_clock::time_point clockNow() { return clockHook ? clockHook() : std::chrono::steady_clock::now(); }

"""
# The body of search::rank after the sampler is built (task T0018, deadline handling task T0020). Mode 0 makes
# the same calls in the same order as the production carrier.
RANK_BODY = """    std::vector<int> pick(E); std::vector<double> tmp(C), tmpDiag(C); Ply1Outcome p1;
    PlanScores scores(C, cfg.riskTau), diagScores(C, cfg.riskTau);
    // PONLY_PLY2_MODE (search_experiment.py): 0 the reference plan's prediction per sample, shared by every
    // plan; 1 each plan's own; 2 plays as 0 and also scores every plan with its own prediction (diagnostic).
    //
    // A sample is one sweep over a plan set. Its values go to scratch (tmp, tmpDiag) and are committed only when
    // the sweep ends, so every committed plan of a call has the same samples. In modes 1 and 2 every prediction
    // is checked against the applicable deadline before it starts and after it ends; a sweep that reaches the
    // deadline is discarded whole (counted: aborted sweeps, their predictions) and no further work starts. A
    // prediction already running is not interrupted: these checks bound when work starts, they do not certify
    // the reply time, which needs a measured reserve for one prediction. Mode 0 keeps the current checks.
    long long p2calls = 0, p2discarded = 0, aborted = 0, diagPairs = 0, diagMoves = 0; double p2sec = 0, diagAim = 0;
    long long sweepCalls = 0, sweepPairs = 0, sweepMoves = 0; double sweepAim = 0;
    auto predictFor = [&](const Plan& pl, int smp) {
        const auto t = clockNow(); predictPly2(in, pl, smp, world, p1);
        p2sec += std::chrono::duration<double>(clockNow() - t).count(); p2calls++; sweepCalls++;
    };
    auto draw = [&](int smp, const Plan& ref) {
        sampler.draw(in, smp, pick); dynamics::setRivalIntents(in, pick, world); world.sample = smp;
#if PONLY_PLY2_MODE != 1
        predictFor(ref, smp);
#else
        (void)ref;
#endif
    };
    // One sample over `set` into scratch; false when `stop` passed before the sweep ended (nothing committed).
    auto sweep = [&](int smp, const Plan& ref, const std::vector<int>& set, std::chrono::steady_clock::time_point stop) {
        sweepCalls = sweepPairs = sweepMoves = 0; sweepAim = 0;
        draw(smp, ref);
#if PONLY_PLY2_MODE != 0
        if (clockNow() >= stop) return false;
#else
        (void)stop;
#endif
        for (int ci : set) {
#if PONLY_PLY2_MODE == 1
            if (clockNow() >= stop) return false;
            predictFor(plans[ci], smp);
            if (clockNow() >= stop) return false;
#endif
            tmp[ci] = rollout(in, cfg, plans[ci], stay, pick, world, ev);
#if PONLY_PLY2_MODE == 2
            const Ply2 shared = world.p2;
            if (clockNow() >= stop) return false;
            predictFor(plans[ci], smp);
            if (clockNow() >= stop) { world.p2 = shared; return false; }
            for (int e = 0; e < world.E; e++) { if (shared.move[e] < 0 && world.p2.move[e] < 0) continue; sweepPairs++;
                sweepMoves += shared.move[e] != world.p2.move[e];
                for (int k = 0; k < 3; k++) sweepAim += std::abs(shared.aim[e][k] - world.p2.aim[e][k]); }
            tmpDiag[ci] = rollout(in, cfg, plans[ci], stay, pick, world, ev);
            world.p2 = shared;
#endif
        }
        return true;
    };
    auto commit = [&](const std::vector<int>& set) {
        for (int ci : set) scores.add(ci, tmp[ci]);
#if PONLY_PLY2_MODE == 2
        for (int ci : set) diagScores.add(ci, tmpDiag[ci]);
        diagPairs += sweepPairs; diagMoves += sweepMoves; diagAim += sweepAim;
#endif
    };
    auto discard = [&]() { aborted++; p2discarded += sweepCalls; };
    // Diagnostic ranking comparison over `set` (plans with the same committed samples): does the best plan change,
    // and how many pairs swap order? `first`: every plan after the first round; else the final survivors.
    auto diagCompare = [&](const std::vector<int>& set, bool first) {
        if (!stats || PONLY_PLY2_MODE != 2 || set.size() < 2 || scores.count[set[0]] == 0 || diagScores.count[set[0]] == 0) return;
        auto ref = [&](int ci) { return scores.blended(ci, plans[ci].value, cfg); };
        auto own = [&](int ci) { return diagScores.blended(ci, plans[ci].value, cfg); };
        int bestRef = set[0], bestOwn = set[0], swaps = 0;
        for (int ci : set) { if (ref(ci) > ref(bestRef)) bestRef = ci; if (own(ci) > own(bestOwn)) bestOwn = ci; }
        for (size_t i = 0; i < set.size(); i++) for (size_t j = i + 1; j < set.size(); j++)
            swaps += (ref(set[i]) > ref(set[j])) != (own(set[i]) > own(set[j]));
        if (first) { stats->diagTop0 = bestRef != bestOwn; stats->diagSwaps0 = swaps; stats->diagCompared0 = int(set.size()); }
        else { stats->diagTopChanged = bestRef != bestOwn; stats->diagSwaps = swaps; stats->diagCompared = int(set.size()); }
    };
    auto finish = [&]() { if (stats) { stats->ply2Calls = p2calls; stats->ply2Seconds = p2sec; stats->memoHits = world.memoHit;
        stats->memoMisses = world.memoMiss; stats->diagPairs = diagPairs; stats->diagMoves = diagMoves; stats->diagAim = diagAim;
        stats->abortedSweeps = aborted; stats->discardedPredictions = p2discarded; } };
    int done = 0;
    if (!cfg.halving) {
        const int limit = fixedSamples > 0 ? fixedSamples : cfg.maxSamples;
        std::vector<int> every(C); for (int ci = 0; ci < C; ci++) every[ci] = ci;
        for (int smp = 0; smp < limit; smp++) {
            if (clockNow() >= deadline) break;  // binds in both modes
            if (!sweep(smp, plans[0], every, deadline)) { discard(); break; }
            commit(every);
            done++;
        }
        if (stats) { stats->samples = done; stats->plans = C; stats->requested = limit; stats->survivors = C;
                     stats->rollouts = (long long)done * C; stats->truncated = done < limit; }
        finish(); if (done) { diagCompare(every, true); diagCompare(every, false); }
        if (!done) return {};
        for (int ci = 0; ci < C; ci++) plans[ci].value = scores.blended(ci, plans[ci].value, cfg);
        std::sort(plans.begin(), plans.end(), [](const Plan& a, const Plan& b) { return a.value > b.value; });
        return plans;
    }
    // Sequential halving (Karnin, Koren and Somekh 2013; the root rule of Gumbel
    // AlphaZero): ceil(log2 C) rounds share the time budget equally; each round
    // sweeps common samples over the surviving plans, then keeps the better half
    // by blended score. Every survivor has seen exactly the same samples, and no
    // plan is sampled more than maxSamples times.
    const auto t0 = clockNow();
    const auto total = deadline - t0;
    int rounds = 1; while ((1 << rounds) < C) rounds++;
    std::vector<int> alive(C); for (int ci = 0; ci < C; ci++) alive[ci] = ci;
    std::vector<int> out;  // eliminated plans, best-first within each later round
    bool cut = false, expired = false; long long rolled = 0;
    auto blended = [&](int ci) { return scores.blended(ci, plans[ci].value, cfg); };
    for (int r = 0; r < rounds && !alive.empty(); r++) {
        const auto roundEnd = t0 + total * (r + 1) / rounds;
        const int quota = fixedSamples > 0 ? std::max(1, fixedSamples / rounds) : cfg.maxSamples - scores.count[alive[0]];
        // Fixed samples: only the hard deadline stops a round (counted); time mode: the round's share.
        const auto stop = fixedSamples > 0 ? deadline : roundEnd;
        for (int got = 0; got < quota && !expired; got++) {
            if (clockNow() >= stop) { cut = cut || fixedSamples > 0; break; }
            if (!sweep(done, plans[alive[0]], alive, stop)) {
                discard(); cut = cut || fixedSamples > 0;
                expired = clockNow() >= deadline;  // past the call's deadline: no further round starts work
                break;
            }
            commit(alive);
            rolled += (long long)alive.size(); done++;
        }
        if (r == 0) diagCompare(alive, true);
        if (scores.count[alive[0]] == 0) break;
        std::stable_sort(alive.begin(), alive.end(), [&](int a, int b) { return blended(a) > blended(b); });
        if (r + 1 == rounds || alive.size() <= 1 || expired) break;
        const size_t keep = (alive.size() + 1) / 2;
        out.insert(out.begin(), alive.begin() + keep, alive.end());
        alive.resize(keep);
    }
    if (stats) { stats->samples = done; stats->plans = C; stats->requested = fixedSamples > 0 ? fixedSamples : cfg.maxSamples;
                 stats->survivors = int(alive.size()); stats->rollouts = rolled;
                 stats->truncated = fixedSamples > 0 ? cut : done < cfg.maxSamples; }
    finish(); if (done) diagCompare(alive, false);
    if (!done) return {};
    std::vector<Plan> ranked;
    for (int ci : alive) { Plan pl = plans[ci]; pl.value = scores.count[ci] ? blended(ci) : -1e300; ranked.push_back(pl); }
    for (int ci : out) { Plan pl = plans[ci]; pl.value = blended(ci); ranked.push_back(pl); }
    return ranked;
}
"""


def c08(text: str) -> str:
    # search::rank: the clock hook before it, and its body after the sampler (one exact span of the source).
    start = text.index(RANK_START)
    end = text.index(RANK_END, start) + len(RANK_END)
    text = text[:start] + RANK_BODY + text[end:]
    text = apply(text, [
        ("template <typename Eval = eval::DefaultEvaluator>\ninline std::vector<Plan> rank(",
         CLOCK + "template <typename Eval = eval::DefaultEvaluator>\ninline std::vector<Plan> rank(", 1),
        ("#define PONLY_V_DEADLINE_MS 1e9\n#endif\n", "#define PONLY_V_DEADLINE_MS 1e9\n#endif\n" + SETTINGS, 1),
        # Search statistics: ply-2 predictions, the per-sample settlement memo, diagnostic comparisons.
        ("struct Stats { int samples = 0, plans = 0, requested = 0, survivors = 0; long long rollouts = 0; bool truncated = false; };",
         "struct Stats { int samples = 0, plans = 0, requested = 0, survivors = 0; long long rollouts = 0; bool truncated = false;\n"
         "    long long ply2Calls = 0, memoHits = 0, memoMisses = 0, diagPairs = 0, diagMoves = 0; double ply2Seconds = 0, diagAim = 0;\n"
         "    int diagTopChanged = 0, diagSwaps = 0, diagCompared = 0, diagTop0 = 0, diagSwaps0 = 0, diagCompared0 = 0;\n"
         "    long long abortedSweeps = 0, discardedPredictions = 0; };", 1),
        ("    std::array<std::array<double,3>,125> memoHits{}, memoDeath{};\n",
         "    std::array<std::array<double,3>,125> memoHits{}, memoDeath{};\n    long long memoHit = 0, memoMiss = 0;  // counters (search_experiment.py)\n", 1),
        ("    if (key >= 0 && w.stamp[key] == w.sample) {\n",
         "    if (key >= 0) (w.stamp[key] == w.sample ? w.memoHit : w.memoMiss)++;\n    if (key >= 0 && w.stamp[key] == w.sample) {\n", 1),
        # Engine semantics: only a pickup order picks up (an explicit wait does not).
        ("        if (!in.allies[k].f.carrying && c.target[k] < 0 && moved && mv.pickup > 0) {",
         "        if (!in.allies[k].f.carrying && c.target[k] == -2 && moved && mv.pickup > 0) {", 1),
        # The P-view cache of ply-2 successors.
        ("        void setRoot(const Game& g){root=g;hidden=false;",
         PLY2_CACHE + "        void setRoot(const Game& g){root=g;hidden=false;cacheKey.clear();cacheNext=0;", 1),
        ("            const fusion::State st=fusion_hook::makeState(next);\n            fs->predictInto(st,root.me,scratch,&teams);\n",
         "            const fusion::State st=fusion_hook::makeState(next);\n            ponly::View& view=viewOf(st);\n", 1),
        ("if(t<0||t>=16||!scratch.has(t)||scratch.origin(id)<0)continue;", "if(t<0||t>=16||!view.has(t)||view.origin(id)<0)continue;", 1),
        ("if(!drawn[t]){scratch.jointDraw(t,ponly::uniform(st.turn,t,sample,29),mv[t].data());drawn[t]=1;}",
         "if(!drawn[t]){view.jointDraw(t,ponly::uniform(st.turn,t,sample,29),mv[t].data());drawn[t]=1;}", 1),
        ("out.aim[e][k]=ally>=0?scratch.aim(id,m,ally):0.;", "out.aim[e][k]=ally>=0?view.aim(id,m,ally):0.;", 1),
        # Planner members, accumulation and the report.
        ("    fusion_hook::PSession fSession;\n    // Expectimax calls and joint samples (PONLY_STATS: one stderr line at the last turn).\n",
         "    fusion_hook::PSession fSession;\n"
         "    ctfcombat::PortfolioStats portStats;              // PONLY_PORTFOLIO_K\n"
         "    std::vector<ctfcombat::Choice> portfolioAlts;     // this turn's alternatives, distinct complete orders\n"
         "    int portfolioFrom=-1;                             // first alternative in the sampled search's candidates\n"
         "    long long statPly2Calls=0,statViewCalls=0,statViewHits=0,statViewBuilds=0,statMemoHits=0,statMemoMisses=0;\n"
         "    long long statDiagPairs=0,statDiagMoves=0,statDiagSearches=0,statDiagTopChanged=0,statDiagSwaps=0,statDiagCompared=0;\n"
         "    long long statDiagTop0=0,statDiagSwaps0=0,statDiagCompared0=0,statAbortedSweeps=0,statDiscardedPredictions=0;\n"
         "    double statPly2Seconds=0,statDiagAim=0;\n" + PORTFOLIO_REPORT +
         "    // Expectimax calls and joint samples (PONLY_STATS: one stderr line at the last turn).\n", 1),
        ("statPlans+=lastSearch.plans;statSurvivors+=lastSearch.survivors;\n",
         "statPlans+=lastSearch.plans;statSurvivors+=lastSearch.survivors;\n"
         "        statPly2Calls+=lastSearch.ply2Calls;statPly2Seconds+=lastSearch.ply2Seconds;statMemoHits+=lastSearch.memoHits;statMemoMisses+=lastSearch.memoMisses;\n"
         "        statViewCalls+=ply2Source.viewCalls;statViewHits+=ply2Source.viewHits;statViewBuilds+=ply2Source.viewBuilds;\n"
         "        ply2Source.viewCalls=ply2Source.viewHits=ply2Source.viewBuilds=0;\n"
         "        statAbortedSweeps+=lastSearch.abortedSweeps;statDiscardedPredictions+=lastSearch.discardedPredictions;\n"
         "        statDiagPairs+=lastSearch.diagPairs;statDiagMoves+=lastSearch.diagMoves;statDiagAim+=lastSearch.diagAim;\n"
         "        if(lastSearch.diagCompared){statDiagSearches++;statDiagTopChanged+=lastSearch.diagTopChanged;statDiagSwaps+=lastSearch.diagSwaps;statDiagCompared+=lastSearch.diagCompared;}\n"
         "        if(lastSearch.diagCompared0){statDiagTop0+=lastSearch.diagTop0;statDiagSwaps0+=lastSearch.diagSwaps0;statDiagCompared0+=lastSearch.diagCompared0;}\n", 1),
        ("                 <<\" refines \"<<statRefines<<\" refine_cuts \"<<statRefineCuts<<std::endl;\n",
         "                 <<\" refines \"<<statRefines<<\" refine_cuts \"<<statRefineCuts<<std::endl;\n"
         "        std::cerr<<\"PLY2 mode \"<<PONLY_PLY2_MODE<<\" cache \"<<PONLY_PLY2_CACHE<<\" predictions \"<<statPly2Calls<<\" seconds \"<<statPly2Seconds\n"
         "                 <<\" view_calls \"<<statViewCalls<<\" view_hits \"<<statViewHits<<\" view_builds \"<<statViewBuilds<<\" memo_hits \"<<statMemoHits\n"
         "                 <<\" memo_misses \"<<statMemoMisses<<\" diag_searches \"<<statDiagSearches<<\" diag_pairs \"<<statDiagPairs<<\" diag_move_diff \"<<statDiagMoves\n"
         "                 <<\" diag_aim_l1 \"<<statDiagAim<<\" diag_top_changed \"<<statDiagTopChanged<<\" diag_swaps \"<<statDiagSwaps<<\" diag_compared \"<<statDiagCompared\n"
         "                 <<\" diag_first_top_changed \"<<statDiagTop0<<\" diag_first_swaps \"<<statDiagSwaps0<<\" diag_first_compared \"<<statDiagCompared0\n"
         "                 <<\" aborted_sweeps \"<<statAbortedSweeps<<\" discarded_predictions \"<<statDiscardedPredictions<<std::endl;\n", 1),
        # Portfolio in the sampled search: alternatives join the asymmetric plan set, never displace it.
        ("            vector<int> keep=asymmetricPlans(g,own,cands,cfg.maxPlans);\n",
         "#if PONLY_PORTFOLIO_K\n"
         "            const int baseCount=portfolioFrom>=0?portfolioFrom:int(cands.size());\n"
         "            vector<int> keep=asymmetricPlans(g,own,vector<ctfcombat::Choice>(cands.begin(),cands.begin()+baseCount),cfg.maxPlans);\n"
         "            for(int i=baseCount;i<int(cands.size());i++){keep.push_back(i);portStats.searched++;}\n"
         "#else\n"
         "            vector<int> keep=asymmetricPlans(g,own,cands,cfg.maxPlans);\n"
         "#endif\n", 1),
        # Checked lookahead: complete-order deduplication and the incumbent movement's alternatives.
        ("        vector<ctfcombat::Choice> candidates{fallback};for(auto c:beam)if(c.value>=fallback.value-12&&c.moveIndex!=fallback.moveIndex)candidates.push_back(c);\n",
         "#if PONLY_PORTFOLIO_K\n"
         "        // Complete orders, not movements: the same movement with other actions is a different candidate.\n"
         "        vector<ctfcombat::Choice> candidates{fallback};for(auto c:beam)if(c.value>=fallback.value-12)addUniqueCandidate(candidates,c);\n"
         "        {int added=0;for(const auto& a:portfolioAlts)if(a.moveIndex==fallback.moveIndex&&added<PONLY_PORTFOLIO_K-1){size_t before=candidates.size();\n"
         "            addUniqueCandidate(candidates,a);if(candidates.size()>before){added++;portStats.refined++;}}}\n"
         "        {int same=0;for(size_t i=0;i<candidates.size();i++)for(size_t j=i+1;j<candidates.size();j++)same+=candidates[i].moveIndex==candidates[j].moveIndex;portStats.sameMove+=same>0;}\n"
         "#else\n"
         "        vector<ctfcombat::Choice> candidates{fallback};for(auto c:beam)if(c.value>=fallback.value-12&&c.moveIndex!=fallback.moveIndex)candidates.push_back(c);\n"
         "#endif\n", 1),
        ("        if(refineAllowed)decisionDeadline=chrono::steady_clock::now()+chrono::milliseconds(g.turn==1?1800:35);",
         "        if(refineAllowed)decisionDeadline=chrono::steady_clock::now()+chrono::milliseconds(g.turn==1?1800:PONLY_TURN_MS);", 1),
        ("        auto choice=ctfcombat::choose(g.map,own,options,foes,params,refine?&beam:nullptr,&decisionDeadline,deep?&all:nullptr);\n",
         "        auto choice=ctfcombat::choose(g.map,own,options,foes,params,refine?&beam:nullptr,&decisionDeadline,deep?&all:nullptr);\n"
         "#if PONLY_PORTFOLIO_K\n"
         "        if(refineAllowed){portfolioFrom=-1;portfolioAlts.clear();\n"
         "          if((deep||refine)&&std::isfinite(choice.value)&&chrono::steady_clock::now()<decisionDeadline){\n" +
         SHORTLIST.format(pool="deep?all:beam") +
         "            if(deep){portfolioFrom=int(all.size());for(const auto& a:portfolioAlts)addUniqueCandidate(all,a);}\n"
         "          }}\n"
         "#endif\n", 1),
        ("        vTurnBegin();auto result=planTurn(g,true);fSession.report(g);searchReport(g);vBudgetReport(g);",
         "        vTurnBegin();auto result=planTurn(g,true);fSession.report(g);searchReport(g);vBudgetReport(g);portfolioReport(g);", 1),
    ], "c08_ponly_f.hpp (search)")
    return apply(text, [("} // namespace ctfcombat", portfolio_function(text, "c08") + "} // namespace ctfcombat", 1)], "c08 portfolio")
