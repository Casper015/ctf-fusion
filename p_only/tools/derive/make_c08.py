"""Derive p_only/bots/c08_ponly.hpp from bots/hybrid/C08_Player5_Omega_LIT_Fusion.hpp.

Record of how the P-only carrier was made (source snapshot 99ecbe1). Every replaced
range is checked at both ends, so the script refuses a changed source instead of
splicing at the wrong place. Run: python p_only/tools/derive/make_c08.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
src = (REPO / "bots/hybrid/C08_Player5_Omega_LIT_Fusion.hpp").read_text(encoding="utf-8").split("\n")
edits = {}          # original line -> replacement lines ([] deletes)


def expect(n, text):
    if text not in src[n - 1]:
        sys.exit(f"line {n}: expected {text!r}, got {src[n - 1][:90]!r}")


def block(a, b, first, last, text):
    """Replace original lines a..b (checked at both ends) by `text` (None deletes)."""
    expect(a, first)
    expect(b, last)
    for n in range(a, b + 1):
        if n in edits:
            sys.exit(f"line {n} edited twice")
        edits[n] = []
    if text is not None:
        edits[a] = text.strip("\n").split("\n")


def inline(n, old, new):
    expect(n, old)
    if n in edits:
        sys.exit(f"line {n} edited twice")
    edits[n] = [src[n - 1].replace(old, new, 1)]


# ---------------------------------------------------------------- header, includes, session
block(27, 32, "#ifndef MOD_FUSION_P", "#endif", """
#ifndef MOD_FUSION_V
#define MOD_FUSION_V 0
#endif""")
block(51, 69, "#if MOD_FUSION_P || MOD_FUSION_V", "#endif", """
#include "features.hpp"
#include "policy.hpp"
#include "adapter.hpp"
#include "value.hpp"
#include "state.hpp"
#include "ponly.hpp"

#if defined(FUSION_TIER_M)
#include "models_m/registry.hpp"
#elif defined(FUSION_TIER_L)
#include "models_l/registry.hpp"
#elif defined(FUSION_TIER_XL)
#include "models_xl/registry.hpp"
#elif defined(FUSION_TIER_XXL)
#include "models_xxl/registry.hpp"
#else
#include "models/registry.hpp"
#endif""")
block(89, 89, "#if MOD_FUSION_P || MOD_FUSION_V", "#if MOD_FUSION_P || MOD_FUSION_V", None)
block(138, 187, "struct Session {", "};", """
    // `view` holds the network's predictions for the current real turn; its
    // history is the previous real observation. A branch (a simulated state
    // inside the checked lookahead) goes to `branchView` with the root state as
    // its previous observation; the real history never sees it. `cur` is the
    // view that forecasts and rival orders read; callers restore it.
    struct PSession {
        bool initialized = false;
        fusion::Board board;
        fusion::State lastReal;
        bool hasReal = false;
        ponly::View view, branchView;
        ponly::View* cur = &view;
        mutable ponly::Counters counters;
        const fusion::models::ModelSet* model = nullptr;

        PSession() = default;
        PSession(const PSession&) = delete;
        PSession& operator=(const PSession&) = delete;

        const fusion::PolicyModel* policy() const { return model ? model->policy : nullptr; }

        void update(const Game& g) {
            if (!initialized) { board.init(makeStatic(g)); initialized = true; }
            model = fusion::models::forTeams(g.teams);
            fusion::State s = makeState(g);
            view.counters = &counters;
            view.build(board, s, hasReal ? &lastReal : nullptr, g.me, policy());
            lastReal = std::move(s);
            hasReal = true;
            cur = &view;
        }

        // A simulated successor of the root: the branch view becomes current.
        void branch(const Game& next) {
            branchView.counters = &counters;
            counters.branches++;
            branchView.build(board, makeState(next), &view.state(), next.me, policy());
            cur = &branchView;
        }

        // A view of state `s` for the teams in `teams` (root as previous observation).
        void predictInto(const fusion::State& s, int me, ponly::View& out, const std::vector<char>* teams) const {
            out.counters = &counters;
            out.build(board, s, &view.state(), me, policy(), teams);
        }

        double evalGameValue(const Game& g) const {
            if (!model || !model->value) return 0.5;
            fusion::State st = makeState(g);
            fusion::TurnView vw;
            fusion::makeView(board, st, nullptr, vw);
            fusion::ValueFeatures vf;
            fusion::encodeValue(board, st, vw, g.me, vf);
            return fusion::value(*model->value, vf);
        }

#if PONLY_STATS
        // One line of counters on stderr at the last turn (screen builds only).
        void report(const Game& g) const {
            if (g.turn != g.turns) return;
            std::cerr << "PONLY builds " << counters.builds << " teams " << counters.teams << " act " << counters.actHeads
                      << " target " << counters.targetHeads << " fallback " << counters.fallback
                      << " inconsistent " << counters.inconsistent << " skipped " << counters.skipped << std::endl;
        }
#else
        void report(const Game&) const {}
#endif
    };""")
block(189, 189, "#endif", "#endif", None)

# ---------------------------------------------------------------- search: types
block(375, 376, "int plies = 2;", "double plyDiscount", None)
block(412, 417, "struct Rival {", "};", """
struct Rival {     // an opponent with the network's destinations
    Fighter f;
    std::vector<int> cell;
    std::vector<double> prob;                        // P(destination)
    std::vector<std::array<double,3>> aim;           // P(attack and target = our unit k | destination)
    double killValue = 11;
};""")
block(419, 430, "struct Input {", "};", """
// The rivals' ply-2 orders (P-only), per rival: the network's move index (-1:
// no prediction, the rival keeps its cell) and P(attack and target = our unit
// k) after that move.
struct Ply2 { std::vector<int> move; std::vector<std::array<double,3>> aim; };
struct Input;
struct Plan;
// What ply 1 of a joint sample does under the reference plan, per search unit
// (our units 0..2, then the rivals): the requested cell, the settled cell (-1 =
// not on the board) and, for rivals, whether our volley kills it.
struct Ply1Outcome { std::vector<int> want, at; std::vector<char> killed; };
// Predicts Ply2 for joint sample `sample` from the engine state that ply 1
// reaches under the reference plan.
struct Ply2Source {
    virtual void predict(const Input& in, const Plan& ref, const Ply1Outcome& p1, int sample, Ply2& out) const = 0;
    virtual ~Ply2Source() = default;
};
struct Input {
    Board board;
    int me = 0, turn = 0;
    Costs costs;
    std::array<Ally,3> allies;
    std::vector<Rival> rivals;
    const std::vector<double>* pickCell = nullptr;   // pickup value of reaching a flag cell
    std::vector<Plan> plans;                         // analytic joint plans (move = option index)
    const ponly::View* view = nullptr;               // the network at the root: joint enemy samples
    const Ply2Source* ply2 = nullptr;                // the network at ply 2: the rivals' next orders
};""")

# ---------------------------------------------------------------- search: dynamics
block(467, 470, "// Plies 2..H: rival cells", "ownPri;", """
    // Ply 2: rival cells (-1 = dead or killed) and the sample's predicted orders.
    std::vector<int> e2;
    Ply2 p2;""")
inline(490, "e2.assign(E, -1); prevE.assign(E, -1); cross.assign(E, 0); ownPri.assign(E, {0, 0, 0});", "e2.assign(E, -1); p2.move.assign(E, -1); p2.aim.assign(E, {0, 0, 0});")
block(514, 522, "// Target-priority rule", "", None)
block(571, 601, "// Incoming fire at the settled cells", "}", """
// Incoming fire at the settled cells (P-only): rival e attacks our unit k with
// the network's probability for its sampled destination, P(attack and target
// = k), and the order lands when the settled cells allow the shot. Per-ally hit
// counts are convolved.
inline Ply1 incomingFire(const Input& in, const std::vector<int>& pick, World& w) {
    const Board& B = in.board;
    Ply1 out;
    std::array<std::array<double,4>,3> hc{};
    for (int k = 0; k < 3; k++) hc[k] = {1, 0, 0, 0};
    for (int e = 0; e < w.E; e++) {
        const Rival& r = in.rivals[e]; const int j = 3 + e;
        if (!w.live[j] || r.f.carrying || r.f.immune || r.cell.empty()) continue;
        const auto& aim = r.aim[pick[e]];
        for (int k = 0; k < 3; k++) {
            if (!w.live[k] || in.allies[k].f.immune || aim[k] <= 0 || !B.shot(w.at[j], w.at[k])) continue;
            out.hits[k] += aim[k]; addShot(hc[k], aim[k]);
        }
    }
    for (int k = 0; k < 3; k++) {
        const Fighter& f = in.allies[k].f;
        if (!w.live[k] || f.immune) continue;
        int lethal = (f.hp + 33) / 34; for (int x = std::min(lethal, 3); x < 4; x++) out.death[k] += hc[k][x];
    }
    return out;
}""")
block(639, 661, "// Scripted enemy step", "}", """
// Ply-2 rival cells (P-only): each surviving rival takes the network's move
// drawn for this sample from its settled ply-1 cell (into a wall: it stays).
// Rivals without a prediction (teams too far from our units to reach them by
// ply 2) keep their cell. Killed rivals drop out.
inline void rivalsStep2(const Input& in, World& w) {
    const auto& nb = *in.board.nb;
    for (int e = 0; e < w.E; e++) {
        const int j = 3 + e; w.e2[e] = -1;
        if (!w.live[j] || w.killed[e]) continue;
        const int m = w.p2.move[e];
        w.e2[e] = m < 0 || nb[w.at[j]][m] < 0 ? w.at[j] : nb[w.at[j]][m];
    }
}""")
block(663, 664, "// Plies 2..H. beginPly", "// surviving rivals at e2", """
// Ply 2. beginPly indexes the positions of the ply (our units at `cur`,
// surviving rivals at e2) for range queries until endPly.""")
block(670, 688, "// Attention every armed rival", "", None)
block(696, 711, "// Exposure of our unit k", "}", """
// Exposure of our unit k if it steps to q at ply 2 with expected HP `hp`: the
// rivals in `close` follow the network's ply-2 orders, P(attack and target =
// our unit k), which land when the shot from their ply-2 cell reaches q.
struct Exposure { double hits = 0, death = 0; };
inline Exposure exposure(const Input& in, const World& w, int k, int q, double hp, const std::vector<int>& close) {
    const Board& B = in.board;
    const int lethal = std::max(1, (int)std::ceil(std::max(1.0, hp) / 34.0));
    std::array<double,4> h{1, 0, 0, 0}; Exposure out;
    for (int e : close) {
        const double pp = w.p2.aim[e][k];
        if (pp <= 0 || !B.shot(w.e2[e], q)) continue;
        out.hits += pp; addShot(h, pp);
    }
    for (int x = std::min(lethal, 3); x < 4; x++) out.death += h[x];
    return out;
}""")

# ---------------------------------------------------------------- search: rollout
block(798, 804, "//   ply 2..H our surviving units", "// With plies = 2 this is Player2 v005", """
//   ply 2    the rivals take the network's ply-2 orders (one draw per joint
//            sample); our surviving units step greedily on
//              potential - fireCost + best shotValue (unladen units only);
//            units lost at ply 1 keep the potential of the cell where they fell
//   result   v + continuationWeight * v2
// P-only: there are no scripted later plies.""")
block(864, 908, "// Plies 2..H.", "}", """
    // Ply 2 (P-only).
    dynamics::rivalsStep2(in, w);
    std::array<int,3> cur{}; std::array<double,3> hpNow{}, alive{}, dead{};
    for (int k = 0; k < 3; k++) {
        if (!w.live[k]) continue;
        cur[k] = w.at[k]; hpNow[k] = in.allies[k].f.hp - 34 * p1.hits[k]; alive[k] = 1 - p1.death[k];
        dead[k] = p1.death[k] * ev.lostValue(in, k, w.at[k]);
    }
    double v2 = 0;
    dynamics::beginPly(w, cur);
    for (int k = 0; k < 3; k++) {
        if (!w.live[k]) continue;
        const int p = cur[k]; const bool car = in.allies[k].f.carrying || picked[k];
        const std::vector<int>& close = dynamics::rivalsNear(w, p);
        double best = -1e300;
        for (int m = 0; m < 5; m++) {
            const int q = nb[p][m]; if (q < 0) continue;
            double val = ev.potential(in, k, q, picked[k], car);
            const dynamics::Exposure x = dynamics::exposure(in, w, k, q, hpNow[k], close);
            val -= ev.fireCost(in, car, x.hits, x.death);
            if (!car) {
                double off = 0;
                for (int e : close) if (dynamics::canShoot(in, w, q, e)) off = std::max(off, ev.shotValue(in, e, w.hpE[e]));
                val += off;
            }
            best = std::max(best, val);
        }
        v2 += alive[k] * best + dead[k];
    }
    dynamics::endPly(w);""")

# ---------------------------------------------------------------- search: sampling and rank
block(919, 922, "// JointSampler: sample s gives rival e", "// numbers), and the sequence is deterministic.", """
// JointSampler (P-only): in sample s every opponent team takes one draw from
// the network's joint at the root (uniform keyed by turn, team and s), so every
// plan meets the same joint sample (common random numbers) and the sequence is
// deterministic. A rival of a team without a prediction draws from its outcome
// probabilities (uniform; counted by the session).""")
block(936, 990, "struct JointSampler {", "};", """
struct JointSampler {
    const ponly::View* view = nullptr;
    explicit JointSampler(const Input& in) : view(in.view) {}
    // pick[e] = destination index of rival e in joint sample `smp`.
    void draw(const Input& in, int smp, std::vector<int>& pick) const {
        const auto& nb = *in.board.nb;
        std::array<std::array<int,3>,16> team{}; std::array<char,16> drawn{};
        for (size_t e = 0; e < in.rivals.size(); e++) {
            const Rival& r = in.rivals[e]; const int t = r.f.team, id = r.f.id;
            int ch = 0;
            if (view && view->has(t) && r.f.cell >= 0 && t >= 0 && t < 16) {
                if (!drawn[t]) { view->jointDraw(t, ponly::uniform(in.turn, t, smp, 7), team[t].data()); drawn[t] = 1; }
                const int want = nb[r.f.cell][team[t][id % 3]];
                for (int o = 0; o < (int)r.cell.size(); o++) if (r.cell[o] == want) { ch = o; break; }
            } else {
                const double u = ponly::uniform(in.turn, id, smp, 9); double acc = 0;
                for (int o = 0; o < (int)r.prob.size(); o++) { ch = o; acc += r.prob[o]; if (u < acc) break; }
            }
            pick[e] = ch;
        }
    }
};

// P-only ply 2: once per joint sample, the network predicts the rivals' next
// orders on the state that ply 1 of the sample reaches under a reference plan
// (the best plan so far), and every plan of the sample shares them. Requires
// the sample's rival intents (dynamics::setRivalIntents) already set.
inline void predictPly2(const Input& in, const Plan& ref, int smp, dynamics::World& w, Ply1Outcome& p1) {
    if (!in.ply2) { w.p2.move.assign(w.E, -1); w.p2.aim.assign(w.E, {0, 0, 0}); return; }
    dynamics::setOwnIntents(in, ref, w); p1.want = w.want;
    dynamics::settle(w); dynamics::volley(in, ref, w);
    p1.at.assign(w.U, -1); p1.killed.assign(w.E, 0);
    for (int a = 0; a < w.U; a++) if (w.live[a]) p1.at[a] = w.at[a];
    for (int e = 0; e < w.E; e++) p1.killed[e] = w.killed[e];
    in.ply2->predict(in, ref, p1, smp, w.p2);
}""")
block(1037, 1038, "std::vector<int> pick(E); std::vector<double> tmp(C);", "auto draw = [&](int smp)", """
    std::vector<int> pick(E); std::vector<double> tmp(C); Ply1Outcome p1;
    auto draw = [&](int smp, const Plan& ref) {
        sampler.draw(in, smp, pick); dynamics::setRivalIntents(in, pick, world); world.sample = smp;
        predictPly2(in, ref, smp, world, p1);
    };""")
inline(1045, "draw(smp);", "draw(smp, plans[0]);")
inline(1072, "draw(done);", "draw(done, plans[alive[0]]);")
inline(1102, "dynamics::setRivalIntents(in, pick, world); world.sample = sample;",
       "dynamics::setRivalIntents(in, pick, world); world.sample = sample;\n    Ply1Outcome p1; predictPly2(in, plan, sample, world, p1);")

# ---------------------------------------------------------------- ctfcombat
block(1134, 1134, "W_other_attention", "W_other_attention", None)
block(1155, 1155, "struct Outcome", "struct Outcome", """
// One destination of an opponent unit, from the policy network: the move's
// probability; after it, the probabilities of attacking, picking up and
// dropping; and of attacking each of our three units (attack and target).
struct Outcome { Pos p; double prob = 0; double fire = 0, pickup = 0, drop = 0; std::array<double,3> aim{}; };""")
block(1159, 1159, "double attention = 1.0;", "double attention = 1.0;", None)
block(1236, 1243, "inline double priority", "", None)
block(1271, 1277, "// Crossfire between opponents", "}", None)
block(1369, 1382, "// A zero conditional firing chance", "incoming[e][i]+=", """
                    // The network's order lands when the settled cells allow the shot:
                    // P(attack and target = our unit i | this move).
                    if(enemy.u.flag>=0 || enemy.u.immune || out.fire<=1e-9) continue;
                    for(int i=0;i<3;++i) if(alive(own[i]) && !own[i].immune && canAttack(map,pos[3],pos[i])) {
                        possibleShot[i]=true;
                        incoming[e][i]+=probability*out.aim[i];
                    }""")

# ---------------------------------------------------------------- multi::Planner
block(1534, 1550, "bool groundFlagAt", "    }", None)
block(1578, 1582, "for(const auto& o:e.forecast){r.cell.push_back", "#endif", """
            for(const auto& o:e.forecast){r.cell.push_back(cellOf(o.p));r.prob.push_back(o.prob);r.aim.push_back(o.aim);}
            in.rivals.push_back(std::move(r));}
        in.view=fSession.cur;
        // Ply-2 predictions only for teams that can reach our units by ply 2
        // (within Manhattan 6 of one of them now); the others cannot matter there.
        ply2Source.setRoot(g);ply2Source.fs=&fSession;ply2Source.pl=this;ply2Source.n=nav.n;ply2Source.teams.assign(g.teams,0);
        for(const auto& e:foes){const int p=cellOf(e.u.p);for(int k=0;k<3;k++){int a=cellOf(own[k].p);if(a>=0&&p>=0&&man(a,p)<=6)ply2Source.teams[e.u.team]=1;}}
        in.ply2=&ply2Source;""")
block(1597, 1597, "if(genes.playout)cfg.plies=4;", "if(genes.playout)cfg.plies=4;", """
        // P-only: no scripted playouts (MOD_PLAYOUT); the network predicts ply 2.""")
inline(1599, "auto ranked=search::rank(in,cfg,stop,&lastSearch,fixedSamples);", "auto ranked=search::rank(in,cfg,stop,&lastSearch,fixedSamples);statSearches++;statSamples+=lastSearch.samples;")
block(1625, 1694, "void forecast(const Game& g){", "    }", """
    // P-only forecast: every legal destination of an opponent unit with the
    // network's move probability, its attack, pickup and drop probabilities
    // after the move, and its aim at each of our units. A unit without a
    // prediction gets uniform moves (counted). The threat map weighs cells
    // within reach of each destination by the probability of moving there and
    // firing.
    void forecast(const Game& g){
        foes.clear();threat.assign(nav.N,0);ponly::View& v=*fSession.cur;
        for(auto& u:g.units)if(u.team!=g.me&&pos[u.id]>=0){
            int p=pos[u.id];
            ctfcombat::Enemy e;e.u={u.id,u.team,{p%nav.n,p/nav.n},immune[u.id]?100:u.hp,u.flag,immune[u.id]};
            e.killValue=11+(u.flag>=0?16:0);
            const bool known=v.has(u.team)&&v.origin(u.id)>=0;double total=0;
            if(!known)fSession.counters.fallback++;
            for(int m=0;m<5;m++)if(nav.nb[p][m]>=0){int q=nav.nb[p][m];ctfcombat::Outcome o;o.p={q%nav.n,q/nav.n};
                if(known){o.prob=v.move(u.id,m);const auto& a=v.act(u.id,m);o.fire=a[fusion::kAttack];o.pickup=a[fusion::kPickup];o.drop=a[fusion::kDrop];
                    for(int k=0;k<3;k++)o.aim[k]=v.aim(u.id,m,3*g.me+k);}
                else o.prob=1.;
                total+=o.prob;e.forecast.push_back(o);}
            for(auto& o:e.forecast)o.prob=total>0?o.prob/total:1./e.forecast.size();
            foes.push_back(e);
            for(auto& o:e.forecast)if(o.fire>0){int q=nav.cell(o.p.x,o.p.y);for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++)if(abs(dx)+abs(dy)<=2){int x=o.p.x+dx,y=o.p.y+dy;if(x<0||x>=nav.n||y<0||y>=nav.n)continue;int r=nav.cell(x,y);if(nav.open(r)&&shot(q,r,g))threat[r]+=o.prob*o.fire;}}
        }
    }""")
block(1805, 1807, "#if MOD_FUSION_P || MOD_FUSION_V", "#endif", """
    fusion_hook::PSession fSession;
    // Expectimax calls and joint samples (PONLY_STATS: one stderr line at the last turn).
    long long statSearches=0,statSamples=0;
    void searchReport(const Game& g)const{
#if PONLY_STATS
        if(g.turn==g.turns)std::cerr<<"SEARCH calls "<<statSearches<<" samples "<<statSamples<<std::endl;
#else
        (void)g;
#endif
    }
    // The network's ply-2 orders for the sampled search. Per joint sample, the
    // engine turn that ply 1 reaches under the reference plan: our plan's moves
    // and acts, each rival's sampled move with the network's act and target for
    // it, settled by the exact kernel (stepExact: HP, deaths, pickups, drops,
    // captures, attack events). The network predicts from that state, with the
    // root as its previous observation. A flag due to return this turn reappears
    // at a random spot nobody can know: it stays off the board in these states
    // (each such state counted as skipped). Every unit alive after the step must stand where the
    // search settled it, and every rival the search's volley kills must be dead:
    // otherwise the state is counted as inconsistent.
    struct Ply2FromNetwork : search::Ply2Source {
        fusion_hook::PSession* fs=nullptr;Planner* pl=nullptr;int n=0;
        Game root;bool hidden=false;vector<char> teams;mutable ponly::View scratch;
        void setRoot(const Game& g){root=g;hidden=false;
            for(auto& f:root.flags)if((f.status=="cooldown"||f.status=="dropped")&&f.return_at<=g.turn){
                f.status="cooldown";f.x=f.y=-1;f.holder=-1;f.return_at=g.turn+1;hidden=true;}}
        void predict(const search::Input& in,const search::Plan& ref,const search::Ply1Outcome& p1,int sample,search::Ply2& out)const override{
            const int E=int(in.rivals.size());out.move.assign(E,-1);out.aim.assign(E,{0,0,0});
            bool any=false;for(char c:teams)any|=c!=0;if(!any||!fs||!pl)return;
            const int U=int(root.units.size());ponly::View& rv=*fs->cur;if(hidden)fs->counters.skipped++;
            vector<int> want(U,-1),acts(U,-1);
            for(int a=0;a<3;a++){const int id=in.allies[a].f.id;if(id<0||id>=U)continue;want[id]=p1.want[a];acts[id]=ref.target[a];}
            for(int e=0;e<E;e++){const int id=in.rivals[e].f.id;if(id<0||id>=U)continue;want[id]=p1.want[3+e];
                const int m=ponly::moveIndex(in.rivals[e].f.cell,p1.want[3+e],n);
                if(m<0||!rv.has(id/3)||rv.origin(id)<0)continue;
                const auto o=rv.order(id,m,ponly::uniform(root.turn,id,sample,41),ponly::uniform(root.turn,id,sample,43));
                acts[id]=o.act==fusion::kAttack?o.target:o.act==fusion::kPickup?-2:o.act==fusion::kDrop?-3:-1;}
            Game next;if(!pl->stepExact(root,want,acts,next)){fs->counters.skipped++;return;}
            for(int a=0;a<3+E;a++){const int id=a<3?in.allies[a].f.id:in.rivals[a-3].f.id;if(id<0||id>=U)continue;const auto& u=next.units[id];
                if(u.x>=0&&p1.at[a]>=0&&u.y*n+u.x!=p1.at[a])fs->counters.inconsistent++;
                if(a>=3&&p1.killed[a-3]&&u.x>=0)fs->counters.inconsistent++;}
            const fusion::State st=fusion_hook::makeState(next);
            fs->predictInto(st,root.me,scratch,&teams);
            array<array<int,3>,16> mv{};array<char,16> drawn{};
            for(int e=0;e<E;e++){const int id=in.rivals[e].f.id,t=id/3;if(t<0||t>=16||!scratch.has(t)||scratch.origin(id)<0)continue;
                if(!drawn[t]){scratch.jointDraw(t,ponly::uniform(st.turn,t,sample,29),mv[t].data());drawn[t]=1;}
                const int m=mv[t][id%3];out.move[e]=m;
                for(int k=0;k<3;k++){const int ally=in.allies[k].f.id;out.aim[e][k]=ally>=0?scratch.aim(id,m,ally):0.;}}
        }
    } ply2Source;""")
block(1851, 1853, "struct Runtime {", "void restoreRuntime", """
    struct Runtime {vector<Unit> previous;array<int,3> type,id;vector<int> positions;vector<bool> immunity;vector<double> hazards;array<vector<double>,3> routes;vector<ctfcombat::Enemy> enemies;ponly::View* view;};
    Runtime saveRuntime()const{return {prev,oldType,oldId,pos,immune,threat,route,foes,fSession.cur};}
    void restoreRuntime(const Runtime& s){prev=s.previous;oldType=s.type;oldId=s.id;pos=s.positions;immune=s.immunity;threat=s.hazards;route=s.routes;foes=s.enemies;fSession.cur=s.view;}""")
block(1878, 1912, "// Beta(0.25,16) prior per context", "    }", None)
block(1940, 1969, "void rivalOrders(const Game& g,int model", "    }", """
    // The rivals' orders from the network's view of this state (the root, or a
    // branch inside refineChecked): model 0 is each team's most probable joint
    // move and act, model 1 one draw shared by every candidate. acts: target id
    // (attack), -2 pickup, -3 drop, -1 nothing.
    void rivalOrders(const Game& g,int model,vector<int>& want,vector<int>& acts){
        want=pos;acts.assign(g.units.size(),-1);ponly::View& v=*fSession.cur;
        for(int t=0;t<g.teams;t++)if(t!=g.me){
            if(!v.has(t)){for(int k=0;k<3;k++)if(pos[3*t+k]>=0)fSession.counters.fallback++;continue;}
            auto orders=v.teamOrders(t,model==0?-1:0,17);
            for(int k=0;k<3;k++){int id=3*t+k;if(pos[id]<0)continue;int q=nav.nb[pos[id]][orders[k].move];want[id]=q>=0?q:pos[id];
                acts[id]=orders[k].act==fusion::kAttack?orders[k].target:orders[k].act==fusion::kPickup?-2:orders[k].act==fusion::kDrop?-3:-1;}
        }
    }""")
inline(2018, "rivalAttacks(g,want,acts);Game next;", "Game next;")
inline(2023, "rivalAttacks(next,want2,acts2);Game end;", "Game end;")
block(2054, 2056, "#if MOD_FUSION_P || MOD_FUSION_V", "#endif", """
        // The real turn updates the root view and history; a branch only the branch view.
        if(refineAllowed)fSession.update(g);else fSession.branch(g);""")
block(2114, 2114, "rivalAttacks(g, want, acts);", "rivalAttacks(g, want, acts);", None)
block(2138, 2138, "if(active(g).drop)observeDrops(g);", "if(active(g).drop)observeDrops(g);", None)
inline(2139, "auto result=planTurn(g,true);", "auto result=planTurn(g,true);fSession.report(g);searchReport(g);")

# ---------------------------------------------------------------- duel::Player
block(2154, 2154, "array<double,5> other;", "array<double,5> other;", """
    array<double,5> fire;              // P(attack | move)
    array<array<double,3>,5> aim;      // P(attack and target = our unit k | move)""")
inline(2163, "vector<int> oldp, lasttarget, oldhp, occupied;", "vector<int> oldp, oldhp, occupied;")
block(2169, 2169, "vector<double> otherFocus;", "vector<double> otherFocus;", None)
block(2176, 2178, "#if MOD_FUSION_P || MOD_FUSION_V", "#endif", """
    fusion_hook::PSession fSession;""")
block(2197, 2199, "double tw(int id) const {", "    }", None)
inline(2229, "oldp.assign(g.units.size(),-1);lasttarget.assign(g.units.size(),-1);", "oldp.assign(g.units.size(),-1);")
block(2244, 2244, "lasttarget[e.args[0]]=e.args[1];", "lasttarget[e.args[0]]=e.args[1];", None)
block(2376, 2442, "void predict(const Game& g) {", "        }", """
    // P-only: each enemy unit's legal moves with the network's probabilities,
    // and after each move its attack probability and aim at our units. A unit
    // without a prediction gets uniform moves (counted).
    void predict(const Game& g) {
        (void)g;pred.clear();ponly::View& v=*fSession.cur;
        for(auto& e:u)if(e.team!=me&&pos[e.id]>=0) {
            Prediction pr;pr.id=e.id;pr.p=nb[pos[e.id]];pr.pr.fill(0);pr.fire.fill(0);for(auto& a:pr.aim)a.fill(0);
            const bool known=v.has(e.team)&&v.origin(e.id)>=0;double sum=0;int cnt=0;
            if(!known)fSession.counters.fallback++;
            for(int m=0;m<5;m++)if(pr.p[m]>=0){cnt++;
                if(known){pr.pr[m]=v.move(e.id,m);pr.fire[m]=v.fire(e.id,m);for(int k=0;k<3;k++)pr.aim[m][k]=v.aim(e.id,m,own[k]);}
                else pr.pr[m]=1;
                sum+=pr.pr[m];}
            for(int m=0;m<5;m++)if(pr.p[m]>=0)pr.pr[m]=sum>0?pr.pr[m]/sum:1./cnt;
            pred.push_back(pr);
        }""")
block(2460, 2513, "double incoming(const array<int,3>& move", "    }", """
    // Expected incoming fire under the network's predictions: enemy j moving h
    // attacks our unit k with P(attack and target = k | h), and the shot lands
    // when the cells allow it (hit table). Hit counts per unit are convolved.
    double incoming(const array<int,3>& move,array<double,3>& death,array<double,3>& expected) const {
        array<array<double,4>,3> dp{};array<double,3> eh{};
        for(int k=0;k<3;k++)dp[k][0]=1;
        for(int j=0;j<(int)pred.size();j++) {
            auto& e=pred[j];if(u[e.id].flag>=0||immune[e.id])continue;
            array<double,3> prob{};
            for(int h=0;h<5;h++)if(e.p[h]>=0)
                for(int k=0;k<3;k++)if(pos[own[k]]>=0&&!immune[own[k]]&&hit[k][move[k]][j][h]>0)prob[k]+=e.pr[h]*e.aim[h][k];
            for(int k=0;k<3;k++) {
                eh[k]+=prob[k];
                array<double,4> z{};
                for(int a=0;a<4;a++){z[a]+=dp[k][a]*(1-prob[k]);z[min(3,a+1)]+=dp[k][a]*prob[k];}dp[k]=z;
            }
        }
        double v=0;
        for(int k=0;k<3;k++)if(pos[own[k]]>=0) {
            int id=own[k],need=(u[id].hp+33)/34;death[k]=0;
            for(int a=need;a<4;a++)death[k]+=dp[k][a];
            double dmg=0;for(int a=0;a<4;a++)dmg+=dp[k][a]*min(a,need);
            double life= u[id].flag>=0 ? 24.0+max(0,12-bd[me][pos[id]])*.5 : 8.0;
            if(bd[me][pos[id]]<4&&u[id].flag<0)life=6;
            v-=dmg*(u[id].flag>=0?2.1:1.1)+life*death[k];
            if(u[id].flag>=0&&u[id].hp<=68)v-=eh[k]*1.2;
        }
        for(int k=0;k<3;k++)expected[k]=eh[k];
        // Survival gating keeps unscaled probabilities; only material costs decay.
        return material*v;
    }""")
block(2549, 2550, "double w=u[id].hp<=34?15", "ft[k][j][h][shots]=exposed*.85*w/(w+pred[j].other[h]);", """
                        ft[k][j][h][shots]=exposed*.85;""")
block(2596, 2671, "struct Candidate {", "    }", None)
block(2674, 2677, "const auto decideStart", "#endif", """
        fSession.update(g);fSession.report(g);""")
block(2722, 2722, "vector<Candidate> candidates;", "vector<Candidate> candidates;", None)
block(2732, 2732, "Candidate candidate{", "Candidate candidate{", None)
block(2738, 2738, "if(v>candidate.value)candidate=", "if(v>candidate.value)candidate=", None)
block(2740, 2740, "candidates.push_back(candidate);", "candidates.push_back(candidate);", None)
block(2742, 2816, "// Direction 6 in duels", "        }", None)

inline(2672, "public:", "public:\n    const fusion_hook::PSession& session() const { return fSession; }")
block(2826, 2826, "int main() {", "int main() {", """
#ifndef PONLY_NO_MAIN  // tests include the bot without its main()
int main() {""")
inline(2834, "}", "}\n#endif")

HEADER = """// C08 Player5_Omega_LIT carrier with the P-only opponent model.
//
// Every prediction about what opponents will do comes from the policy network
// through p_only/include/ponly.hpp: movement, firing, targets, pickups and
// drops at the root; the joint enemy samples of the expectimax search; the
// rivals' ply-2 orders in every sampled rollout (one network view per sample);
// and the rivals' orders and forecasts on the simulated states of the checked
// lookahead (each predicted from that state, with the root as its previous
// observation, never written into the real history).
//
// Removed from bots/hybrid/C08_Player5_Omega_LIT_Fusion.hpp: the hand-written
// movement forecasts (multi and duel) and their mixture weight, the opponent
// goal and target guesses, attention/priority/crossfire/focus weights and the
// last-target habit, the ground-flag pickup-versus-fire rule, the learned
// per-team drop rates and their contexts for rivals, "a blocked rival always
// fires", the scripted rival steps of the playouts (MOD_PLAYOUT) and the
// hand-written rival fire allocation of the lookahead. The duel's 24-sample
// refinement, never active at two teams in this build (Expectimax is frozen to
// multi-team counts), is deleted with its V term.
// Kept: game rules (legal moves, settlement, range and line of fire, respawns,
// flag returns), our own task assignment, candidate moves, scoring, search and
// evaluation; the duel's reachability-based carrier-route risk. MOD_FUSION_V
// adds the optional value terms of the multi-team planner. PONLY_STATS=1
// prints the network call counters on stderr at the last turn."""

for n in sorted(edits):
    if n < 1 or n > len(src):
        sys.exit(f"bad line {n}")
out = HEADER.split("\n")
for i, line in enumerate(src, 1):
    out += edits[i] if i in edits else [line]

# ---------------------------------------------------------------- experiment switches (applied to the derived text)
POST = [
("#ifndef MOD_FUSION_V\n#define MOD_FUSION_V 0\n#endif",
 """#ifndef MOD_FUSION_V
#define MOD_FUSION_V 0
#endif
// PONLY_FIXED_SAMPLES=N (diagnostic equal-work mode): every expectimax call asks
// for exactly N joint samples instead of filling the 30 ms search budget. The
// hard turn deadline (35 ms compute target) still binds: a call that reaches it
// stops and is counted as truncated, and the checked lookahead keeps only that
// hard deadline. 0 = the deployed fixed-time mode.
#ifndef PONLY_FIXED_SAMPLES
#define PONLY_FIXED_SAMPLES 0
#endif
// V coefficients (MOD_FUSION_V): heuristic points per unit of V - 0.5 in the
// checked lookahead (refine) and in the one-ply beam re-ranking. Tuning
// parameters, not validated constants.
#ifndef PONLY_V_REFINE
#define PONLY_V_REFINE 50.0
#endif
#ifndef PONLY_V_BEAM
#define PONLY_V_BEAM 35.0
#endif"""),
("#elif defined(FUSION_TIER_XXL)\n#include \"models_xxl/registry.hpp\"\n#else",
 "#elif defined(FUSION_TIER_XXL)\n#include \"models_xxl/registry.hpp\"\n#elif defined(FUSION_TIER_PXL_VM)\n#include \"models_pxl_vm/registry.hpp\"\n#elif defined(FUSION_TIER_PXXL_VM)\n#include \"models_pxxl_vm/registry.hpp\"\n#else"),
("struct Stats { int samples = 0, plans = 0; };",
 """// samples: joint samples completed; requested: the call's target (fixed
// samples, or cfg.maxSamples when time-bounded); rollouts: plan evaluations;
// survivors: plans still compared at the end (all, without halving);
// truncated: a deadline stopped the call before its target.
struct Stats { int samples = 0, plans = 0, requested = 0, survivors = 0; long long rollouts = 0; bool truncated = false; };"""),
("            if (fixedSamples <= 0 && std::chrono::steady_clock::now() >= deadline) break;",
 "            if (std::chrono::steady_clock::now() >= deadline) break;  // binds in both modes"),
("        if (stats) { stats->samples = done; stats->plans = C; }\n        if (!done) return {};",
 "        if (stats) { stats->samples = done; stats->plans = C; stats->requested = limit; stats->survivors = C;\n                     stats->rollouts = (long long)done * C; stats->truncated = done < limit; }\n        if (!done) return {};"),
("    std::vector<int> out;  // eliminated plans, best-first within each later round",
 "    std::vector<int> out;  // eliminated plans, best-first within each later round\n    bool cut = false; long long rolled = 0;"),
("            if (fixedSamples <= 0 && std::chrono::steady_clock::now() >= roundEnd) break;",
 "            // Fixed samples: only the hard deadline stops a round (counted); time mode: the round's share.\n            if (std::chrono::steady_clock::now() >= (fixedSamples > 0 ? deadline : roundEnd)) { cut = cut || fixedSamples > 0; break; }"),
("            for (int ci : alive) scores.add(ci, rollout(in, cfg, plans[ci], stay, pick, world, ev));\n            done++;",
 "            for (int ci : alive) scores.add(ci, rollout(in, cfg, plans[ci], stay, pick, world, ev));\n            rolled += (long long)alive.size(); done++;"),
("    if (stats) { stats->samples = done; stats->plans = C; }\n    if (!done) return {};\n    std::vector<Plan> ranked;",
 "    if (stats) { stats->samples = done; stats->plans = C; stats->requested = fixedSamples > 0 ? fixedSamples : cfg.maxSamples;\n                 stats->survivors = int(alive.size()); stats->rollouts = rolled;\n                 stats->truncated = fixedSamples > 0 ? cut : done < cfg.maxSamples; }\n    if (!done) return {};\n    std::vector<Plan> ranked;"),
("auto ranked=search::rank(in,cfg,stop,&lastSearch,fixedSamples);statSearches++;statSamples+=lastSearch.samples;",
 "auto ranked=search::rank(in,cfg,stop,&lastSearch,fixedSamples);statSearches++;statSamples+=lastSearch.samples;\n        statRequested+=lastSearch.requested;statTruncated+=lastSearch.truncated;statRollouts+=lastSearch.rollouts;statPlans+=lastSearch.plans;statSurvivors+=lastSearch.survivors;"),
("    long long statSearches=0,statSamples=0;",
 """    long long statSearches=0,statSamples=0,statRequested=0,statTruncated=0,statRollouts=0,statPlans=0,statSurvivors=0;
#if MOD_FUSION_V
    // V use per path (0: checked lookahead, 1: one-ply beam): network calls, exact
    // terminal utilities, decisions taken with V, decisions V changed against the
    // same path without V, and seconds spent encoding and evaluating.
    long long vCalls[2]={0,0},vExact[2]={0,0},vDecisions[2]={0,0},vChanged[2]={0,0};double vSeconds=0;
    double stateValue(const Game& s,int path){
        if(s.turn>s.turns){vExact[path]++;double u=0;int n=0;
            for(int t=0;t<s.teams;t++)if(t!=s.me){u+=s.score[s.me]>s.score[t]?1.:s.score[s.me]==s.score[t]?.5:0.;n++;}return n?u/n:.5;}
        const auto t0=chrono::steady_clock::now();vCalls[path]++;const double v=fSession.evalGameValue(s);
        vSeconds+=chrono::duration<double>(chrono::steady_clock::now()-t0).count();return v;}
#endif"""),
("        if(g.turn==g.turns)std::cerr<<\"SEARCH calls \"<<statSearches<<\" samples \"<<statSamples<<std::endl;",
 """        if(g.turn!=g.turns)return;
        std::cerr<<"SEARCH calls "<<statSearches<<" samples "<<statSamples<<" requested "<<statRequested<<" truncated "<<statTruncated
                 <<" rollouts "<<statRollouts<<" plans "<<statPlans<<" survivors "<<statSurvivors<<" fixed "<<PONLY_FIXED_SAMPLES<<std::endl;
#if MOD_FUSION_V
        std::cerr<<"VALUE refine_calls "<<vCalls[0]<<" refine_exact "<<vExact[0]<<" refine_decisions "<<vDecisions[0]<<" refine_changed "<<vChanged[0]
                 <<" beam_calls "<<vCalls[1]<<" beam_exact "<<vExact[1]<<" beam_decisions "<<vDecisions[1]<<" beam_changed "<<vChanged[1]
                 <<" seconds "<<vSeconds<<" coef_refine "<<PONLY_V_REFINE<<" coef_beam "<<PONLY_V_BEAM<<std::endl;
#endif"""),
("            auto ranked=deepen(g,own,options,params,all,cellValue,pickCell,min(decisionDeadline,budget-reserve));\n            decisionDeadline=min(decisionDeadline,budget);",
 """#if PONLY_FIXED_SAMPLES
            // Equal-work mode: exactly PONLY_FIXED_SAMPLES samples, bounded only by the hard turn deadline.
            (void)budget;(void)reserve;
            auto ranked=deepen(g,own,options,params,all,cellValue,pickCell,decisionDeadline,PONLY_FIXED_SAMPLES);
#else
            auto ranked=deepen(g,own,options,params,all,cellValue,pickCell,min(decisionDeadline,budget-reserve));
            decisionDeadline=min(decisionDeadline,budget);
#endif"""),
("auto saved=saveRuntime();auto outer=decisionDeadline;decisionDeadline=min(outer,chrono::steady_clock::now()+chrono::milliseconds(20));",
 "auto saved=saveRuntime();auto outer=decisionDeadline;\n#if !PONLY_FIXED_SAMPLES\n        decisionDeadline=min(outer,chrono::steady_clock::now()+chrono::milliseconds(20));\n#endif"),
("        double answerWorst=-1e100;\n        for(const auto&candidate:candidates){double mean=0,worst=1e100;bool complete=true;",
 "        double answerWorst=-1e100;\n#if MOD_FUSION_V\n        ctfcombat::Choice answerH=fallback;double bestH=-1e100,answerWorstH=-1e100;  // the same choice without V\n#endif\n        for(const auto&candidate:candidates){double mean=0,worst=1e100;bool complete=true;\n#if MOD_FUSION_V\n            double meanH=0,worstH=1e100;\n#endif"),
("""#if MOD_FUSION_V
                if (fSession.model && fSession.model->value) {
                    double vEnd = fSession.evalGameValue(end);
                    outcome += 50.0 * (vEnd - 0.5);
                }
#endif
                mean+=.5*outcome;worst=min(worst,outcome);""",
 """#if MOD_FUSION_V
                meanH+=.5*outcome;worstH=min(worstH,outcome);
                // V of this scenario's settled end state (the exact pairwise utility once
                // the game is over); the two scenarios combine through mean and worst.
                if (fSession.model && fSession.model->value) outcome += PONLY_V_REFINE * (stateValue(end,0) - 0.5);
#endif
                mean+=.5*outcome;worst=min(worst,outcome);"""),
("            bool dropping=false;for(int t:candidate.target)dropping|=t==-3;\n            if(dropping&&(value<best+5||worst<answerWorst))continue;",
 """            bool dropping=false;for(int t:candidate.target)dropping|=t==-3;
#if MOD_FUSION_V
            {const double valueH=candidate.positionValue+.8*meanH+.2*worstH;
             if(!(dropping&&(valueH<bestH+5||worstH<answerWorstH))&&valueH>bestH+1e-8){bestH=valueH;answerH=candidate;answerWorstH=worstH;}}
#endif
            if(dropping&&(value<best+5||worst<answerWorst))continue;"""),
("        restoreRuntime(saved);decisionDeadline=outer;return answer;",
 "#if MOD_FUSION_V\n        if(fSession.model&&fSession.model->value){vDecisions[0]++;if(answer.moveIndex!=answerH.moveIndex||answer.target!=answerH.target)vChanged[0]++;}\n#endif\n        restoreRuntime(saved);decisionDeadline=outer;return answer;"),
("""                    double bestVal = -1e100;
                    ctfcombat::Choice bestChoice = choice;
                    for (auto& c : beam) {
                        vector<int> want, acts;
                        rivalOrders(g, 0, want, acts);
                        for(int k=0;k<3;k++){int i=3*g.me+k;auto p=options[k][c.moveIndex[k]].p;if(pos[i]>=0)want[i]=nav.cell(p.x,p.y);acts[i]=c.target[k];}
                        Game nxt;
                        if (stepExact(g, want, acts, nxt)) {
                            double vVal = fSession.evalGameValue(nxt);
                            double score = c.value + 35.0 * (vVal - 0.5);
                            if (score > bestVal) {
                                bestVal = score;
                                bestChoice = c;
                            }
                        }
                    }
                    choice = bestChoice;""",
 """                    // One scenario, every rival on its most probable order (model 0), settled
                    // exactly; V of that successor (exact utility once the game is over). Only the
                    // plans that survived the search's final round compete: sequential halving
                    // values eliminated plans on fewer samples, so their values are not comparable
                    // (with every V coefficient 0 this hook then keeps the search's choice).
                    double bestVal = -1e100;
                    ctfcombat::Choice bestChoice = choice;
                    const size_t pool = std::min(beam.size(), size_t(std::max(1, lastSearch.survivors)));
                    for (size_t bi = 0; bi < pool; bi++) {
                        const auto& c = beam[bi];
                        vector<int> want, acts;
                        rivalOrders(g, 0, want, acts);
                        for(int k=0;k<3;k++){int i=3*g.me+k;auto p=options[k][c.moveIndex[k]].p;if(pos[i]>=0)want[i]=nav.cell(p.x,p.y);acts[i]=c.target[k];}
                        Game nxt;
                        if (stepExact(g, want, acts, nxt)) {
                            double score = c.value + PONLY_V_BEAM * (stateValue(nxt,1) - 0.5);
                            if (score > bestVal) {
                                bestVal = score;
                                bestChoice = c;
                            }
                        }
                    }
                    vDecisions[1]++;if(bestChoice.moveIndex!=choice.moveIndex||bestChoice.target!=choice.target)vChanged[1]++;
                    choice = bestChoice;"""),
]
POST += [
("if(chrono::steady_clock::now()>=decisionDeadline){complete=false;break;}",
 "if(chrono::steady_clock::now()>=decisionDeadline){complete=false;refineCut=true;break;}", 3),
("        auto saved=saveRuntime();auto outer=decisionDeadline;",
 "        auto saved=saveRuntime();auto outer=decisionDeadline;bool refineCut=false;  // a deadline stopped the lookahead", 1),
("        restoreRuntime(saved);decisionDeadline=outer;return answer;",
 "        statRefines++;statRefineCuts+=refineCut;\n        restoreRuntime(saved);decisionDeadline=outer;return answer;", 1),
("    long long statSearches=0,statSamples=0,statRequested=0,statTruncated=0,statRollouts=0,statPlans=0,statSurvivors=0;",
 "    long long statSearches=0,statSamples=0,statRequested=0,statTruncated=0,statRollouts=0,statPlans=0,statSurvivors=0;\n"
 "    long long statRefines=0,statRefineCuts=0;  // checked lookaheads run, and those a deadline stopped early", 1),
("<<\" survivors \"<<statSurvivors<<\" fixed \"<<PONLY_FIXED_SAMPLES<<std::endl;",
 "<<\" survivors \"<<statSurvivors<<\" fixed \"<<PONLY_FIXED_SAMPLES\n                 <<\" refines \"<<statRefines<<\" refine_cuts \"<<statRefineCuts<<std::endl;", 1),
]
text = "\n".join(out)
for item in POST:
    old, new, times = item if len(item) == 3 else (*item, 1)
    if text.count(old) != times:
        sys.exit(f"POST: expected {times} occurrence(s) of {old[:70]!r}, found {text.count(old)}")
    text = text.replace(old, new)

dst = Path(os.environ.get("PONLY_DERIVE_DIR", REPO / "p_only/bots")) / "c08_ponly.hpp"   # tests derive into a scratch dir
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(text, encoding="utf-8")
print(dst, text.count("\n") + 1, "lines")
