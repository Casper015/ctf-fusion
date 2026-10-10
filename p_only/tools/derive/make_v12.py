"""Derive p_only/bots/v12_ponly.hpp from candidate/Player5_v12_Fusion.hpp.

Record of how the P-only carrier was made (source snapshot 99ecbe1). Every replaced
range is checked, so the script refuses a changed source instead of splicing at
the wrong place. Run: python p_only/tools/derive/make_v12.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
src = (REPO / "candidate/Player5_v12_Fusion.hpp").read_text(encoding="utf-8").split("\n")
L = lambda a, b: src[a - 1:b]          # 1-based inclusive


def expect(n, text):
    if text not in src[n - 1]:
        sys.exit(f"line {n}: expected {text!r}, got {src[n - 1][:80]!r}")


for n, t in [(16, "#ifndef MOD_FUSION_P"), (41, "#endif"), (53, "#if MOD_FUSION_P || MOD_FUSION_V"), (102, "struct Session"),
             (166, "};"), (168, "#endif"), (203, "struct Outcome"), (207, "double attention"), (280, "inline double priority"),
             (286, "}"), (314, "Crossfire between"), (320, "}"), (408, "!out.canShoot"), (420, "incoming[e][i]+="),
             (524, "#if MOD_FUSION_P"), (526, "#endif"), (546, "int closestGoal"), (556, "void forecast"), (611, "}"),
             (703, "vector<int> rivalWants"), (724, "void simStep"), (734, "act[3*g.me+k]=ourActs[k]"), (735, "for(int team=0"),
             (749, "}"), (750, "for(auto& u:s.u){int target=act[u.id]"), (753, "act[u.id]==-2"), (754, "s.score[u.team]++"),
             (755, "}"), (771, "ctfcombat::Choice refine"), (817, "}"), (818, "string decide"), (821, "#endif")]:
    expect(n, t)

HEADER = """// Player5 v12 (rollout_volley) carrier with the P-only opponent model.
//
// Every prediction about what opponents will do comes from the policy network
// through p_only/include/ponly.hpp: movement, firing, targets, pickups and
// drops at the root, and the rivals' orders and forecasts on the simulated
// states of the two-turn rollout (each predicted from that state, with the
// root as its previous observation).
//
// Removed from candidate/Player5_v12_Fusion.hpp: the hand-written movement
// forecast and its mixture weight, the opponent goal and target guesses, the
// attention, priority and crossfire weights, the rules "a unit able to shoot
// always shoots" and "a unit on a flag picks it up instead of shooting", the
// scripted rival moves of the rollout and its hand-written rival fire
// allocation. Kept: game rules (legal moves, settlement, range and line of
// fire, respawns, flag returns), our own task assignment, candidate moves,
// scoring and rollout evaluation. MOD_FUSION_V adds the optional value term;
// PONLY_STATS=1 prints the network call counters on stderr at the last turn.
"""

INCLUDES = """#ifndef MOD_FUSION_V
#define MOD_FUSION_V 0
#endif

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
#endif"""

SESSION = """    template <typename SimType>
    inline fusion::State simState(const SimType& sim, int turn) {
        fusion::State st;
        st.turn = turn;
        st.score = sim.score;
        st.units.resize(sim.u.size());
        for (size_t i = 0; i < sim.u.size(); i++)
            st.units[i] = {sim.u[i].id, sim.u[i].team, sim.u[i].x, sim.u[i].y, sim.u[i].hp, sim.u[i].flag, sim.u[i].respawn_at};
        st.flags.resize(sim.f.size());
        for (size_t i = 0; i < sim.f.size(); i++) {
            int s = fusion::kHome;
            if (sim.f[i].status == "carried") s = fusion::kCarried;
            else if (sim.f[i].status == "dropped") s = fusion::kDropped;
            else if (sim.f[i].status == "cooldown") s = fusion::kCooldown;
            st.flags[i] = {sim.f[i].id, s, sim.f[i].x, sim.f[i].y, sim.f[i].holder, sim.f[i].return_at};
        }
        for (const auto& a : sim.attacks) st.attacks.push_back({a.first, a.second});
        return st;
    }

    // `view` holds the network's predictions for the current real turn; its
    // history is the previous real observation. branch() predicts a simulated
    // state with the root as its previous observation and never changes the
    // root view or the real history.
    struct PSession {
        bool initialized = false;
        fusion::Board board;
        fusion::State lastReal;
        bool hasReal = false;
        ponly::View view;
        mutable ponly::Counters counters;
        const fusion::models::ModelSet* model = nullptr;

        const fusion::PolicyModel* policy() const { return model ? model->policy : nullptr; }

        void update(const Game& g) {
            if (!initialized) { board.init(makeStatic(g)); initialized = true; }
            model = fusion::models::forTeams(g.teams);
            fusion::State s = makeState(g);
            view.counters = &counters;
            view.build(board, s, hasReal ? &lastReal : nullptr, g.me, policy());
            lastReal = std::move(s);
            hasReal = true;
        }

        void branch(const fusion::State& s, int me, ponly::View& out) const {
            out.counters = &counters;
            counters.branches++;
            out.build(board, s, &view.state(), me, policy());
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

        template <typename SimType>
        double evalSimValue(const SimType& sim, int turn, int me) const {
            if (!model || !model->value) return 0.5;
            fusion::State st = simState(sim, turn);
            fusion::TurnView vw;
            fusion::makeView(board, st, nullptr, vw);
            fusion::ValueFeatures vf;
            fusion::encodeValue(board, st, vw, me, vf);
            return fusion::value(*model->value, vf);
        }
    };"""

OUTCOME = """// One destination of an opponent unit, from the policy network: the move's
// probability, the probability of attacking after it, and of attacking each of
// our three units (attack and target together).
struct Outcome { Pos p; double prob = 0; double fire = 0; std::array<double,3> aim{}; };"""

INCOMING = """                    // The network's order lands when the settled cells allow the shot:
                    // P(attack and target = our unit i | this move).
                    if(enemy.u.flag>=0 || enemy.u.immune || out.fire<=0) continue;
                    for(int i=0;i<3;++i) if(alive(own[i]) && !own[i].immune && canAttack(map,pos[3],pos[i])) {
                        possibleShot[i]=true;
                        incoming[e][i]+=probability*out.aim[i];
                    }"""

MEMBERS = """    fusion_hook::PSession fSession;
    ponly::View branchView;  // the network's view of each simulated state in refine()"""

FORECAST = """    // Every legal destination of an opponent unit with the network's move
    // probability; with heads, also its attack probability and its aim at our
    // units. A unit without a prediction gets uniform moves (counted).
    vector<ctfcombat::Outcome> outcomes(ponly::View& v,int me,int uid,int p,bool heads)const{
        vector<ctfcombat::Outcome> out;const bool known=v.has(uid/3)&&v.origin(uid)>=0;double total=0;
        if(!known)fSession.counters.fallback++;
        for(int m=0;m<5;m++)if(nav.nb[p][m]>=0){int q=nav.nb[p][m];ctfcombat::Outcome o;o.p={q%nav.n,q/nav.n};
            o.prob=known?v.move(uid,m):1.;total+=o.prob;
            if(known&&heads){o.fire=v.fire(uid,m);for(int k=0;k<3;k++)o.aim[k]=v.aim(uid,m,3*me+k);}
            out.push_back(o);}
        for(auto& o:out)o.prob=total>0?o.prob/total:1./out.size();
        return out;
    }
    void forecast(const Game& g){
        foes.clear();threat.assign(nav.N,0);
        for(auto& u:g.units)if(u.team!=g.me&&pos[u.id]>=0){
            int p=pos[u.id];
            ctfcombat::Enemy e;e.u={u.id,u.team,{p%nav.n,p/nav.n},immune[u.id]?100:u.hp,u.flag,immune[u.id]};
            e.killValue=11+(u.flag>=0?16:0);e.forecast=outcomes(fSession.view,g.me,u.id,p,true);
            foes.push_back(e);
            // Threat: cells within reach of each destination, weighted by the
            // probability that the unit moves there and fires.
            for(auto& o:e.forecast)if(o.fire>0){int q=nav.cell(o.p.x,o.p.y);for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++)if(abs(dx)+abs(dy)<=2){int x=o.p.x+dx,y=o.p.y+dy;if(x<0||x>=nav.n||y<0||y>=nav.n)continue;int r=nav.cell(x,y);if(nav.open(r)&&shot(q,r,g))threat[r]+=o.prob*o.fire;}}
        }
    }"""

RIVALS = """    // The rivals' orders on the network's view of state s: model 0 is the most
    // probable joint move and act of each team, model 1 one draw (the same
    // draw for every candidate; `salt` separates the plies). want = destination
    // cell; act = target id (attack), -2 pickup, -3 drop, -1 nothing.
    void rivalOrders(const Game& g,ponly::View& v,const Sim& s,int model,uint32_t salt,vector<int>& want,vector<int>& act)const{
        want.assign(s.u.size(),-1);act.assign(s.u.size(),-1);
        for(int t=0;t<g.teams;t++)if(t!=g.me){
            if(!v.has(t)){for(int k=0;k<3;k++){int id=3*t+k;if(id<(int)s.u.size()&&sp(s.u[id])>=0){want[id]=sp(s.u[id]);fSession.counters.fallback++;}}continue;}
            auto orders=v.teamOrders(t,model==0?-1:0,salt);
            for(int k=0;k<3;k++){int id=3*t+k;int p=sp(s.u[id]);if(p<0)continue;
                int q=nav.nb[p][orders[k].move];want[id]=q>=0?q:p;
                act[id]=orders[k].act==fusion::kAttack?orders[k].target:orders[k].act==fusion::kPickup?-2:orders[k].act==fusion::kDrop?-3:-1;}
        }
    }
    // One engine turn on the simulation: movement, attacks, deaths, pickups,
    // voluntary drops, captures. act: target id (attack), -2 pickup, -3 drop,
    // anything else nothing; our three units use ourActs.
    void simStep(const Game& g,Sim& s,int t,vector<int> want,vector<int> act,const array<int,3>& ourActs)const{
        int N=s.u.size();vector<int> start(N),hpLoss(N,0);vector<bool> blocked(N,false);act.resize(N,-1);s.attacks.clear();"""

DROP = """        for(auto& u:s.u)if(sp(u)>=0&&u.flag>=0&&act[u.id]==-3){auto& f=s.f[u.flag];f.status="dropped";f.x=u.x;f.y=u.y;f.holder=-1;f.return_at=t+g.flag_return;u.flag=-1;}"""

REFINE = r"""    ctfcombat::Choice refine(const Game& g,const array<Task,3>& jobs,const array<vector<ctfcombat::Move>,3>& initialOptions,vector<ctfcombat::Choice> beam,ctfcombat::Choice fallback){
        int near=0;for(auto& e:foes){bool close=false;for(int k=0;k<3;k++)if(pos[3*g.me+k]>=0&&man(pos[3*g.me+k],nav.cell(e.u.p.x,e.u.p.y))<=5)close=true;near+=close;}
        if(near==0||near>9)return fallback;
        ctfcombat::Choice bestVolley;for(auto& c:beam)if(c.target[0]>=0&&c.target[1]>=0&&c.target[2]>=0&&c.value>bestVolley.value)bestVolley=c;
        sort(beam.begin(),beam.end(),[](auto& a,auto& b){return a.value>b.value;});if(beam.size()>4)beam.resize(4);
        bool hasVolley=false;for(auto& c:beam)if(c.target[0]>=0&&c.target[1]>=0&&c.target[2]>=0)hasVolley=true;
        if(!hasVolley&&std::isfinite(bestVolley.value))beam.push_back(bestVolley);
        auto deadline=chrono::steady_clock::now()+chrono::milliseconds(24);Sim start=initialSim(g);
        double best=-1e100;ctfcombat::Choice answer=fallback;
        for(auto& candidate:beam){double total=0;int scenarios=0;
            for(int model=0;model<2;model++){
                if(chrono::steady_clock::now()>deadline)return fallback;
                Sim one=start;array<int,3> acts=candidate.target;
                // Ply 1: the rivals' orders from the network at the root.
                vector<int> want,act;rivalOrders(g,fSession.view,one,model,11,want,act);
                for(int k=0;k<3;k++){auto& m=initialOptions[k][candidate.moveIndex[k]];want[3*g.me+k]=m.p.x<0?-1:nav.cell(m.p.x,m.p.y);}
                simStep(g,one,g.turn,want,act,acts);
                Sim two=one;advanceSim(two,g.turn+1);
                // Ply 2: the network predicts the simulated next turn (its previous observation: the root).
                fSession.branch(fusion_hook::simState(two,g.turn+1),g.me,branchView);
                array<ctfcombat::Fighter,3> futureOwn;array<vector<ctfcombat::Move>,3> opts;vector<ctfcombat::Enemy> futureFoes;
                for(auto& u:two.u)if(u.team!=g.me&&sp(u)>=0){ctfcombat::Enemy e;e.u={u.id,u.team,{u.x,u.y},u.hp,u.flag,two.immune[u.id]};e.killValue=11+(u.flag>=0?16:0);
                    bool relevant=false;for(int k=0;k<3;k++){int a=sp(two.u[3*g.me+k]);if(a>=0&&man(a,sp(u))<=4)relevant=true;}
                    e.forecast=outcomes(branchView,g.me,u.id,sp(u),relevant);futureFoes.push_back(e);}
"""

REFINE_TAIL = r"""                ctfcombat::Params params;auto follow=ctfcombat::choose(g.map,futureOwn,opts,futureFoes,params);
                vector<int> want2,act2;rivalOrders(g,branchView,two,model,23,want2,act2);
                for(int k=0;k<3;k++){auto& m=opts[k][follow.moveIndex[k]];want2[3*g.me+k]=m.p.x<0?-1:nav.cell(m.p.x,m.p.y);}
                simStep(g,two,g.turn+1,want2,act2,follow.target);
                double value=.2*simIncrement(g,start,one,jobs)+.7*simIncrement(g,one,two,jobs);
#if MOD_FUSION_V
                if (fSession.model && fSession.model->value) {
                    double vVal = fSession.evalSimValue(two, g.turn + 2, g.me);
                    value += 40.0 * (vVal - 0.5);
                }
#endif
                total+=value*(model==0?.7:.3);scenarios++;
            }
            if(scenarios==2&&candidate.value+total>best){best=candidate.value+total;answer=candidate;}
        }return answer;
    }"""

# our ply-2 option generation is kept verbatim (original lines 792-801)
for n, t in [(792, "for(int k=0;k<3;k++){auto& u=two.u[3*g.me+k]"), (803, "ctfcombat::Params params;auto follow")]:
    expect(n, t)

out = [HEADER.rstrip("\n")]
out += L(1, 15)
out += INCLUDES.split("\n")
out += L(42, 52)
out += L(54, 101)
out += SESSION.split("\n")
out += L(167, 167)
out += L(169, 202)
out += OUTCOME.split("\n")
out += L(204, 206)
out += L(208, 279)
out += L(288, 313)
out += L(321, 407)
out += INCOMING.split("\n")
out += L(421, 523)
out += MEMBERS.split("\n")
out += L(527, 545)
out += FORECAST.split("\n")
out += L(612, 702)
out += RIVALS.split("\n")
out += L(726, 733)            # settlement, unchanged
out += L(734, 734)            # our acts
out += L(750, 753)            # attacks, deaths, pickups
out += DROP.split("\n")
out += L(754, 755)            # captures, end
out += L(756, 770)            # simIncrement
out += REFINE.rstrip("\n").split("\n")
out += L(792, 802)            # our ply-2 options, unchanged
out += REFINE_TAIL.split("\n")
out += L(818, 818)
out += ["        fSession.update(g);fSession.report(g);"]
expect(850, "int main(){")
out += L(822, 849)
out += ["#ifndef PONLY_NO_MAIN  // tests include the bot without its main()", src[849], "#endif"]
out += L(851, len(src))

# Simulated turns also record their attack events (the network reads hits taken).
EDITS = [("    struct Sim {vector<Unit> u;vector<Flag> f;vector<bool> immune;vector<int> score;};",
          "    struct Sim {vector<Unit> u;vector<Flag> f;vector<bool> immune;vector<int> score;vector<pair<int,int>> attacks;};"),
         ("            auto& v=s.u[target];if(v.team!=u.team&&sp(v)>=0&&!s.immune[v.id]&&shot(sp(u),sp(v),g))hpLoss[target]+=34;}}",
          "            auto& v=s.u[target];if(v.team!=u.team&&sp(v)>=0&&!s.immune[v.id]&&shot(sp(u),sp(v),g)){hpLoss[target]+=34;s.attacks.push_back({u.id,target});}}}"),
         ("        Sim s{g.units,g.flags,immune,g.score};", "        Sim s{g.units,g.flags,immune,g.score,{}};"),
         # A dead unit has HP 0 in the engine (it went negative here).
         ("        for(auto& u:s.u)if(sp(u)>=0){u.hp-=hpLoss[u.id];if(u.hp<=0){if(u.flag>=0){auto& f=s.f[u.flag];f.status=\"dropped\";f.x=u.x;f.y=u.y;f.holder=-1;f.return_at=t+15;}u.x=u.y=-1;u.flag=-1;u.respawn_at=t+10;}}",
          "        for(auto& u:s.u)if(sp(u)>=0){u.hp-=hpLoss[u.id];if(u.hp<=0){if(u.flag>=0){auto& f=s.f[u.flag];f.status=\"dropped\";f.x=u.x;f.y=u.y;f.holder=-1;f.return_at=t+15;}u.x=u.y=-1;u.flag=-1;u.hp=0;u.respawn_at=t+10;}}")]
for old, new in EDITS:
    hits = [i for i, line in enumerate(out) if line == old]
    if len(hits) != 1:
        sys.exit(f"expected one line {old[:60]!r}, found {len(hits)}")
    out[hits[0]] = new

dst = Path(os.environ.get("PONLY_DERIVE_DIR", REPO / "p_only/bots")) / "v12_ponly.hpp"   # tests derive into a scratch dir
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(out), encoding="utf-8")
print(dst, len(out), "lines")
