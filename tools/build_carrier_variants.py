"""Script to build C08_Player5_Omega_LIT_Fusion and its tier variants.

Generates:
  candidate/C08_Player5_Omega_LIT_Fusion.hpp (single source with conditional MOD_FUSION_P hooks)
  candidate/C08_Player5_Omega_LIT_control.cpp (MOD_FUSION_P=0, identical to vanilla)
  candidate/C08_Player5_Omega_LIT_p_s.cpp (Tier S: MOD_FUSION_P=1, FUSION_TIER_S=1)
  candidate/C08_Player5_Omega_LIT_p_m.cpp (Tier M: MOD_FUSION_P=1, FUSION_TIER_M=1)
  candidate/C08_Player5_Omega_LIT_p_l.cpp (Tier L: MOD_FUSION_P=1, FUSION_TIER_L=1)
  candidate/C08_Player5_Omega_LIT_p_xl.cpp (Tier XL: MOD_FUSION_P=1, FUSION_TIER_XL=1)
  candidate/C08_Player5_Omega_LIT_p_xxl.cpp (Tier XXL: MOD_FUSION_P=1, FUSION_TIER_XXL=1)
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AI_TEST_ROOT = Path("c:/Users/caspe/OneDrive/Code/AI_Test").resolve()
C08_SRC = AI_TEST_ROOT / "ctf-benchmark-v3" / "track2_2k_c_series" / "bots" / "C08_Player5_Omega_LIT.cpp"
CANDIDATE_DIR = REPO / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)


def build_fusion_hpp():
    code = C08_SRC.read_text(encoding="utf-8")

    # 1. Macro configuration block: add MOD_FUSION_P and MOD_FUSION_V
    macro_anchor = "#define MOD_EXPECTIMAX_SIZES NG_SIZES_MULTI"
    macro_insert = """#define MOD_EXPECTIMAX_SIZES NG_SIZES_MULTI

#ifndef MOD_FUSION_P
#define MOD_FUSION_P 0
#endif
#ifndef MOD_FUSION_V
#define MOD_FUSION_V 0
#endif"""
    assert macro_anchor in code
    code = code.replace(macro_anchor, macro_insert, 1)

    # 2. Includes: include inference headers and appropriate model registry
    include_anchor = "using namespace std;"
    include_insert = """#if MOD_FUSION_P || MOD_FUSION_V
#include "features.hpp"
#include "policy.hpp"
#include "adapter.hpp"
#include "value.hpp"
#include "state.hpp"

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
#endif
#endif

using namespace std;"""
    assert include_anchor in code
    code = code.replace(include_anchor, include_insert, 1)

    # 3. Game struct conversion & session right after struct Game
    game_anchor = "struct Game {\n    int teams = 0, me = 0, size = 0, turns = 0, hp = 0, damage = 0, respawn = 0, flag_return = 0, flag_cooldown = 0;\n    vector<string> map;\n    vector<pair<int, int>> bases;\n    vector<pair<int, int>> spots;\n    int turn = 0;\n    vector<int> score;\n    vector<Unit> units;\n    vector<Flag> flags;\n    vector<Event> events;\n};"
    game_insert = """struct Game {
    int teams = 0, me = 0, size = 0, turns = 0, hp = 0, damage = 0, respawn = 0, flag_return = 0, flag_cooldown = 0;
    vector<string> map;
    vector<pair<int, int>> bases;
    vector<pair<int, int>> spots;
    int turn = 0;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;
    vector<Event> events;
};

#if MOD_FUSION_P || MOD_FUSION_V
namespace fusion_hook {
    inline fusion::Static makeStatic(const Game& g) {
        fusion::Static s;
        s.teams = g.teams;
        s.me = g.me;
        s.size = g.size;
        s.turns = g.turns;
        s.hp = g.hp;
        s.damage = g.damage;
        s.respawn = g.respawn;
        s.flagReturn = g.flag_return;
        s.flagCooldown = g.flag_cooldown;
        s.wall.assign(g.size * g.size, 0);
        for (int y = 0; y < g.size; y++)
            for (int x = 0; x < g.size; x++)
                s.wall[y * g.size + x] = (g.map[y][x] == '#');
        s.baseCenter.resize(g.bases.size());
        for (size_t i = 0; i < g.bases.size(); i++)
            s.baseCenter[i] = {g.bases[i].first, g.bases[i].second};
        s.spots.resize(g.spots.size());
        for (size_t i = 0; i < g.spots.size(); i++)
            s.spots[i] = {g.spots[i].first, g.spots[i].second};
        return s;
    }

    inline fusion::State makeState(const Game& g) {
        fusion::State s;
        s.turn = g.turn;
        s.score = g.score;
        s.units.resize(g.units.size());
        for (size_t i = 0; i < g.units.size(); i++) {
            s.units[i] = {g.units[i].id, g.units[i].team, g.units[i].x, g.units[i].y, g.units[i].hp, g.units[i].flag, g.units[i].respawn_at};
        }
        s.flags.resize(g.flags.size());
        for (size_t i = 0; i < g.flags.size(); i++) {
            int st = fusion::kHome;
            if (g.flags[i].status == "carried") st = fusion::kCarried;
            else if (g.flags[i].status == "dropped") st = fusion::kDropped;
            else if (g.flags[i].status == "cooldown") st = fusion::kCooldown;
            s.flags[i] = {g.flags[i].id, st, g.flags[i].x, g.flags[i].y, g.flags[i].holder, g.flags[i].return_at};
        }
        for (const auto& e : g.events) {
            if (e.type == "attack" && e.args.size() >= 2)
                s.attacks.push_back({e.args[0], e.args[1]});
        }
        return s;
    }

    struct Session {
        bool initialized = false;
        fusion::Board board;
        fusion::State state, prevState;
        bool hasPrev = false;
        fusion::TurnView view;
        std::vector<fusion::PolicyFeatures> feat;
        std::vector<fusion::PolicyOutput> policyOut;
        std::vector<fusion::LegacyForecast> qh;
        const fusion::models::ModelSet* model = nullptr;

        void update(const Game& g) {
            if (!initialized) {
                board.init(makeStatic(g));
                feat.resize(g.teams);
                policyOut.resize(g.teams);
                qh.resize(g.teams);
                initialized = true;
            }
            if (hasPrev) prevState = state;
            state = makeState(g);
            makeView(board, state, hasPrev ? &prevState : nullptr, view);
            hasPrev = true;
            model = fusion::models::forTeams(g.teams);
            if (model && model->policy && model->alpha > 0) {
                for (int t = 0; t < g.teams; t++) {
                    if (t == g.me) continue;
                    bool active = false;
                    for (int k = 0; k < 3; k++) {
                        int uid = 3 * t + k;
                        if (uid < (int)g.units.size() && view.origin[uid] >= 0) active = true;
                    }
                    if (active) {
                        fusion::encodePolicy(board, state, view, t, feat[t]);
                        fusion::jointPolicy(*model->policy, feat[t], policyOut[t]);
                    }
                }
            }
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
    };
}
#endif"""
    assert game_anchor in code
    code = code.replace(game_anchor, game_insert, 1)

    # 4. Input struct: add session pointer
    input_anchor = "struct Input {\n    Board board;\n    int me = 0, turn = 0;\n    Costs costs;\n    std::array<Ally,3> allies;\n    std::vector<Rival> rivals;\n    const std::vector<double>* pickCell = nullptr;   // pickup value of reaching a flag cell\n    std::vector<Plan> plans;                         // analytic joint plans (move = option index)\n};"
    input_insert = """struct Input {
    Board board;
    int me = 0, turn = 0;
    Costs costs;
    std::array<Ally,3> allies;
    std::vector<Rival> rivals;
    const std::vector<double>* pickCell = nullptr;   // pickup value of reaching a flag cell
    std::vector<Plan> plans;                         // analytic joint plans (move = option index)
#if MOD_FUSION_P
    const fusion_hook::Session* fusionSession = nullptr;
#endif
};"""
    assert input_anchor in code
    code = code.replace(input_anchor, input_insert, 1)

    # 5. JointSampler: hook in drawTeam
    sampler_anchor = """struct JointSampler {
    std::vector<double> off;
    explicit JointSampler(const Input& in) : off(in.rivals.size()) {
        for (size_t e = 0; e < in.rivals.size(); e++) {
            uint32_t h = (uint32_t)(in.turn * 101 + in.rivals[e].f.id * 7919 + 187);
            h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16; off[e] = (h % 65536) / 65536.0;
        }
    }
    // pick[e] = forecast index of rival e in joint sample `smp`.
    void draw(const Input& in, int smp, std::vector<int>& pick) const {
        for (size_t e = 0; e < in.rivals.size(); e++) {
            double r = std::fmod(off[e] + smp * 0.6180339887498949, 1.0), acc = 0; int ch = 0;
            for (int o = 0; o < (int)in.rivals[e].prob.size(); o++) { ch = o; acc += in.rivals[e].prob[o]; if (r <= acc) break; }
            pick[e] = ch;
        }
    }
};"""

    sampler_insert = """struct JointSampler {
    std::vector<double> off;
#if MOD_FUSION_P
    const fusion_hook::Session* fusion = nullptr;
#endif
    explicit JointSampler(const Input& in) : off(in.rivals.size()) {
#if MOD_FUSION_P
        fusion = in.fusionSession;
#endif
        for (size_t e = 0; e < in.rivals.size(); e++) {
            uint32_t h = (uint32_t)(in.turn * 101 + in.rivals[e].f.id * 7919 + 187);
            h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16; off[e] = (h % 65536) / 65536.0;
        }
    }
    // pick[e] = forecast index of rival e in joint sample `smp`.
    void draw(const Input& in, int smp, std::vector<int>& pick) const {
#if MOD_FUSION_P
        if (fusion && fusion->model && fusion->model->policy && fusion->model->alpha > 0 &&
            fusion->model->adapter == fusion::kAdapterJoint && in.board.nb) {
            const auto* mset = fusion->model;
            std::vector<bool> handled(in.rivals.size(), false);
            for (size_t e = 0; e < in.rivals.size(); e++) {
                if (handled[e]) continue;
                int t = in.rivals[e].f.team;
                uint32_t th = (uint32_t)(in.turn * 101 + t * 7919 + 187);
                th ^= th >> 16; th *= 0x7feb352dU; th ^= th >> 15; th *= 0x846ca68bU; th ^= th >> 16;
                double toff = (th % 65536) / 65536.0;
                double u[4];
                for (int i = 0; i < 4; i++) u[i] = std::fmod(toff + (smp * 4 + i) * 0.6180339887498949, 1.0);
                fusion::TeamDraw td = fusion::drawTeam(mset->adapter, mset->alpha, fusion->qh[t], fusion->policyOut[t], u);
                for (size_t rIdx = 0; rIdx < in.rivals.size(); rIdx++) {
                    if (in.rivals[rIdx].f.team == t) {
                        handled[rIdx] = true;
                        int k = in.rivals[rIdx].f.id % 3;
                        int wantMove = td.move[k];
                        int startCell = in.rivals[rIdx].f.cell;
                        int wantCell = (startCell >= 0 && startCell < (int)in.board.nb->size()) ? (*in.board.nb)[startCell][wantMove] : -1;
                        int ch = 0;
                        for (int o = 0; o < (int)in.rivals[rIdx].cell.size(); o++) {
                            if (in.rivals[rIdx].cell[o] == wantCell) { ch = o; break; }
                        }
                        pick[rIdx] = ch;
                    }
                }
            }
            return;
        }
#endif
        for (size_t e = 0; e < in.rivals.size(); e++) {
            double r = std::fmod(off[e] + smp * 0.6180339887498949, 1.0), acc = 0; int ch = 0;
            for (int o = 0; o < (int)in.rivals[e].prob.size(); o++) { ch = o; acc += in.rivals[e].prob[o]; if (r <= acc) break; }
            pick[e] = ch;
        }
    }
};"""
    assert sampler_anchor in code
    code = code.replace(sampler_anchor, sampler_insert, 1)

    # 6. multi::Planner::deepen: pass fusionSession
    deepen_anchor = """        for(const auto& e:foes){search::Rival r;r.f={e.u.id,e.u.team,cellOf(e.u.p),e.u.hp,e.u.flag>=0,e.u.immune};r.killValue=e.killValue;
            for(const auto& o:e.forecast){r.cell.push_back(cellOf(o.p));r.prob.push_back(o.prob);r.fire.push_back(o.canShoot?o.shootChance:0.);}
            in.rivals.push_back(move(r));}"""
    deepen_insert = """        for(const auto& e:foes){search::Rival r;r.f={e.u.id,e.u.team,cellOf(e.u.p),e.u.hp,e.u.flag>=0,e.u.immune};r.killValue=e.killValue;
            for(const auto& o:e.forecast){r.cell.push_back(cellOf(o.p));r.prob.push_back(o.prob);r.fire.push_back(o.canShoot?o.shootChance:0.);}
            in.rivals.push_back(move(r));}
#if MOD_FUSION_P
        in.fusionSession = &fSession;
#endif"""
    assert deepen_anchor in code
    code = code.replace(deepen_anchor, deepen_insert, 1)

    # 7. multi::Planner::forecast: mix marginal and firing
    forecast_anchor = """            double mx=*max_element(vals.begin(),vals.end()),sum=0;
            for(size_t i=0;i<vals.size();i++){double v=exp((vals[i]-mx)/.75);e.forecast[i].prob=v;sum+=v;}
            for(auto& o:e.forecast)o.prob=.88*o.prob/sum+.12/e.forecast.size();"""

    forecast_insert = """            double mx=*max_element(vals.begin(),vals.end()),sum=0;
            for(size_t i=0;i<vals.size();i++){double v=exp((vals[i]-mx)/.75);e.forecast[i].prob=v;sum+=v;}
            for(auto& o:e.forecast)o.prob=.88*o.prob/sum+.12/e.forecast.size();
#if MOD_FUSION_P
            int k = u.id % 3, t = u.team;
            if (fSession.model && fSession.model->policy && fSession.model->alpha > 0) {
                for (int m = 0; m < 5; m++) fSession.qh[t].move[k][m] = 0.0;
                for (auto& o : e.forecast) {
                    int dx = o.p.x - (p % nav.n), dy = o.p.y - (p / nav.n);
                    int m = (dx == 0 && dy == 0) ? 0 : (dx == 0 && dy == -1) ? 1 : (dx == 0 && dy == 1) ? 2 : (dx == -1 && dy == 0) ? 3 : 4;
                    fSession.qh[t].move[k][m] = o.prob;
                }
                for (auto& o : e.forecast) {
                    int dx = o.p.x - (p % nav.n), dy = o.p.y - (p / nav.n);
                    int m = (dx == 0 && dy == 0) ? 0 : (dx == 0 && dy == -1) ? 1 : (dx == 0 && dy == 1) ? 2 : (dx == -1 && dy == 0) ? 3 : 4;
                    o.prob = fusion::mixedMarginal(fSession.model->alpha, fSession.qh[t], fSession.policyOut[t], k, m);
                    double pa[fusion::kActs];
                    fusion::actPolicy(*fSession.model->policy, fSession.feat[t], k, m, pa);
                    double pAttack = pa[fusion::kAttack];
                    o.shootChance = (1.0 - fSession.model->alpha) * o.shootChance + fSession.model->alpha * pAttack;
                }
            }
#endif"""
    assert forecast_anchor in code
    code = code.replace(forecast_anchor, forecast_insert, 1)

    # 8. multi::Planner::planTurn: update fSession
    planturn_anchor = """        const auto planStart=chrono::steady_clock::now();
        if(refineAllowed)decisionDeadline=chrono::steady_clock::now()+chrono::milliseconds(g.turn==1?1800:35);
        if(!initialized){nav.init(g);initialized=true;}respawns(g);forecast(g);auto jobs=assign(g);"""

    planturn_insert = """        const auto planStart=chrono::steady_clock::now();
        if(refineAllowed)decisionDeadline=chrono::steady_clock::now()+chrono::milliseconds(g.turn==1?1800:35);
#if MOD_FUSION_P || MOD_FUSION_V
        fSession.update(g);
#endif
        if(!initialized){nav.init(g);initialized=true;}respawns(g);forecast(g);auto jobs=assign(g);"""
    assert planturn_anchor in code
    code = code.replace(planturn_anchor, planturn_insert, 1)

    # 9. multi::Planner refineChecked: hook V into outcome
    refine_outcome_anchor = """                double outcome=realized(g,next)+.75*realized(next,end)-ownershipLoss(g,next,end)+terminalFlags(end)-terminalFlags(g);
                mean+=.5*outcome;worst=min(worst,outcome);"""
    refine_outcome_insert = """                double outcome=realized(g,next)+.75*realized(next,end)-ownershipLoss(g,next,end)+terminalFlags(end)-terminalFlags(g);
#if MOD_FUSION_V
                if (fSession.model && fSession.model->value) {
                    double vEnd = fSession.evalGameValue(end);
                    outcome += 50.0 * (vEnd - 0.5);
                }
#endif
                mean+=.5*outcome;worst=min(worst,outcome);"""
    assert refine_outcome_anchor in code
    code = code.replace(refine_outcome_anchor, refine_outcome_insert, 1)

    # 10. multi::Planner planTurn beam rerank: hook V when refineGate is false
    beam_rerank_anchor = """            if(!ranked.empty()){choice=ranked[0];beam.assign(ranked.begin(),ranked.begin()+min<size_t>(8,ranked.size()));}"""
    beam_rerank_insert = """            if(!ranked.empty()){choice=ranked[0];beam.assign(ranked.begin(),ranked.begin()+min<size_t>(8,ranked.size()));
#if MOD_FUSION_V
                if (fSession.model && fSession.model->value && beam.size() > 1 && !refineGate(g)) {
                    double bestVal = -1e100;
                    ctfcombat::Choice bestChoice = choice;
                    for (auto& c : beam) {
                        vector<int> want, acts;
                        rivalOrders(g, 0, want, acts);
                        for(int k=0;k<3;k++){int i=3*g.me+k;auto p=options[k][c.moveIndex[k]].p;if(pos[i]>=0)want[i]=nav.cell(p.x,p.y);acts[i]=c.target[k];}
                        rivalAttacks(g, want, acts);
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
                    choice = bestChoice;
                }
#endif
            }"""
    assert beam_rerank_anchor in code
    code = code.replace(beam_rerank_anchor, beam_rerank_insert, 1)

    # 11. multi::Planner member: fSession
    planner_member_anchor = "    chrono::steady_clock::time_point decisionDeadline;"
    planner_member_insert = """#if MOD_FUSION_P || MOD_FUSION_V
    fusion_hook::Session fSession;
#endif
    chrono::steady_clock::time_point decisionDeadline;"""
    assert planner_member_anchor in code
    code = code.replace(planner_member_anchor, planner_member_insert, 1)

    # 12. duel::Player member: fSession
    duel_member_anchor = "    NgActive act;  // module activation for 2-team games"
    duel_member_insert = """#if MOD_FUSION_P || MOD_FUSION_V
    fusion_hook::Session fSession;
#endif
    NgActive act;  // module activation for 2-team games"""
    assert duel_member_anchor in code
    code = code.replace(duel_member_anchor, duel_member_insert, 1)

    # 13. duel::Player::decide: update fSession
    duel_decide_anchor = """        const auto decideStart=chrono::steady_clock::now();
        if(!n)init(g);update(g);act=ngActive(g.teams);"""
    duel_decide_insert = """        const auto decideStart=chrono::steady_clock::now();
#if MOD_FUSION_P || MOD_FUSION_V
        fSession.update(g);
#endif
        if(!n)init(g);update(g);act=ngActive(g.teams);"""
    assert duel_decide_anchor in code
    code = code.replace(duel_decide_anchor, duel_decide_insert, 1)

    # 14. duel::Player::predict: mix marginal
    duel_predict_anchor = """            for(int m=0;m<5;m++)if(pr.p[m]>=0)pr.pr[m]=.90*pr.pr[m]/sum+.10/cnt;
            pred.push_back(pr);"""
    duel_predict_insert = """            for(int m=0;m<5;m++)if(pr.p[m]>=0)pr.pr[m]=.90*pr.pr[m]/sum+.10/cnt;
#if MOD_FUSION_P
            if (fSession.model && fSession.model->policy && fSession.model->alpha > 0) {
                int t = u[pr.id].team, k = pr.id % 3;
                for (int m = 0; m < 5; m++) fSession.qh[t].move[k][m] = pr.p[m] >= 0 ? pr.pr[m] : 0.0;
                for (int m = 0; m < 5; m++) if (pr.p[m] >= 0) {
                    pr.pr[m] = fusion::mixedMarginal(fSession.model->alpha, fSession.qh[t], fSession.policyOut[t], k, m);
                }
            }
#endif
            pred.push_back(pr);"""
    assert duel_predict_anchor in code
    code = code.replace(duel_predict_anchor, duel_predict_insert, 1)

    # 15. duel::Player duel refinement: team draw
    duel_refine_anchor = """            constexpr int samples=24;vector<vector<int>> enemyMoves(samples,vector<int>(pred.size()));
            for(int s=0;s<samples;s++)for(int j=0;j<(int)pred.size();j++) {
                uint32_t h=(uint32_t)(g.turn*101+pred[j].id*7919+187);
                h^=h>>16;h*=0x7feb352dU;h^=h>>15;h*=0x846ca68bU;h^=h>>16;
                double r=fmod((s+.5)/samples+(h%65536)/65536.0,1.0),acc=0;
                int chosen=0;for(int m=0;m<5;m++)if(pred[j].p[m]>=0){chosen=m;acc+=pred[j].pr[m];if(r<=acc)break;}
                enemyMoves[s][j]=chosen;
            }"""
    duel_refine_insert = """            constexpr int samples=24;vector<vector<int>> enemyMoves(samples,vector<int>(pred.size()));
#if MOD_FUSION_P
            if (fSession.model && fSession.model->policy && fSession.model->alpha > 0 &&
                fSession.model->adapter == fusion::kAdapterJoint) {
                int t = 1 - me;
                uint32_t th = (uint32_t)(g.turn * 101 + t * 7919 + 187);
                th ^= th >> 16; th *= 0x7feb352dU; th ^= th >> 15; th *= 0x846ca68bU; th ^= th >> 16;
                double toff = (th % 65536) / 65536.0;
                for (int s = 0; s < samples; s++) {
                    double u[4];
                    for (int i = 0; i < 4; i++) u[i] = fmod(toff + (s * 4 + i) * 0.6180339887498949, 1.0);
                    fusion::TeamDraw td = fusion::drawTeam(fSession.model->adapter, fSession.model->alpha, fSession.qh[t], fSession.policyOut[t], u);
                    for (int j = 0; j < (int)pred.size(); j++) {
                        int k = pred[j].id % 3;
                        enemyMoves[s][j] = td.move[k];
                    }
                }
            } else
#endif
            for(int s=0;s<samples;s++)for(int j=0;j<(int)pred.size();j++) {
                uint32_t h=(uint32_t)(g.turn*101+pred[j].id*7919+187);
                h^=h>>16;h*=0x7feb352dU;h^=h>>15;h*=0x846ca68bU;h^=h>>16;
                double r=fmod((s+.5)/samples+(h%65536)/65536.0,1.0),acc=0;
                int chosen=0;for(int m=0;m<5;m++)if(pred[j].p[m]>=0){chosen=m;acc+=pred[j].pr[m];if(r<=acc)break;}
                enemyMoves[s][j]=chosen;
            }"""
    assert duel_refine_anchor in code
    code = code.replace(duel_refine_anchor, duel_refine_insert, 1)

    duel_v_anchor = """                double value=(1-COLLISION_WEIGHT)*ca.value+COLLISION_WEIGHT*actual;
                if(value>refinedBest){refinedBest=value;bm=ca.mv;ba=ca.ac;}"""
    duel_v_insert = """                double value=(1-COLLISION_WEIGHT)*ca.value+COLLISION_WEIGHT*actual;
#if MOD_FUSION_V
                if (fSession.model && fSession.model->value) {
                    auto p0 = settle(ca.mv, enemyMoves[0]);
                    Game nextGame = g;
                    for (int id = 0; id < (int)nextGame.units.size(); id++) {
                        if (p0[id] >= 0) {
                            nextGame.units[id].x = p0[id] % n;
                            nextGame.units[id].y = p0[id] / n;
                        } else {
                            nextGame.units[id].x = nextGame.units[id].y = -1;
                        }
                    }
                    double vVal = fSession.evalGameValue(nextGame);
                    value += 25.0 * (vVal - 0.5);
                }
#endif
                if(value>refinedBest){refinedBest=value;bm=ca.mv;ba=ca.ac;}"""
    assert duel_v_anchor in code
    code = code.replace(duel_v_anchor, duel_v_insert, 1)

    target_hpp = CANDIDATE_DIR / "C08_Player5_Omega_LIT_Fusion.hpp"
    target_hpp.write_text(code, encoding="utf-8")
    print(f"Generated {target_hpp} ({len(code)} bytes)")

    # Generate the variant bot source files
    variants = {
        "C08_Player5_Omega_LIT_control.cpp": """// Frozen C08 Control: MOD_FUSION_P=0, MOD_FUSION_V=0 (bit-identical to C08_Player5_Omega_LIT.cpp)
#define MOD_FUSION_P 0
#define MOD_FUSION_V 0
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
        "C08_Player5_Omega_LIT_p_s.cpp": """// C08 Carrier + Tier S Opponent Policy
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_S 1
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
        "C08_Player5_Omega_LIT_p_m.cpp": """// C08 Carrier + Tier M Opponent Policy (P-M without V / 卸掉V)
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_M 1
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
        "C08_Player5_Omega_LIT_pm_no_v.cpp": """// C08 Carrier + Tier M Opponent Policy (P-M without V / 卸掉V)
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_M 1
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
        "C08_Player5_Omega_LIT_pm_with_v.cpp": """// C08 Carrier + Tier M Policy + Value (P-M with V)
#define MOD_FUSION_P 1
#define MOD_FUSION_V 1
#define FUSION_TIER_M 1
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
        "C08_Player5_Omega_LIT_p_l.cpp": """// C08 Carrier + Tier L Opponent Policy
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_L 1
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
        "C08_Player5_Omega_LIT_p_xl.cpp": """// C08 Carrier + Tier XL Opponent Policy
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_XL 1
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
        "C08_Player5_Omega_LIT_p_xxl.cpp": """// C08 Carrier + Tier XXL Opponent Policy
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_XXL 1
#include "C08_Player5_Omega_LIT_Fusion.hpp"
""",
    }

    for name, content in variants.items():
        vp = CANDIDATE_DIR / name
        vp.write_text(content, encoding="utf-8")
        print(f"Generated {vp}")


def build_player5_v12_fusion():
    v12_src = REPO / "roster" / "new_bots" / "forks" / "ctf-arena_Player5_v12_rollout_volley.cpp"
    code = v12_src.read_text(encoding="utf-8")

    # 1. Macro and includes at top
    inc_anchor = "using namespace std;"
    inc_insert = """#ifndef MOD_FUSION_P
#define MOD_FUSION_P 0
#endif
#ifndef MOD_FUSION_V
#define MOD_FUSION_V 0
#endif

#if MOD_FUSION_P || MOD_FUSION_V
#include "features.hpp"
#include "policy.hpp"
#include "adapter.hpp"
#include "value.hpp"
#include "state.hpp"

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
#endif
#endif

using namespace std;"""
    assert inc_anchor in code
    code = code.replace(inc_anchor, inc_insert, 1)

    # 2. Namespace fusion_hook after Game struct
    game_anchor = "struct Game {\n    int teams=0,me=0,size=0,turns=0,hp=0,damage=0,respawn=0,flag_return=0,flag_cooldown=0;\n    vector<string> map; vector<pair<int,int>> bases,spots;\n    int turn=0; vector<int> score; vector<Unit> units; vector<Flag> flags; vector<Event> events;\n};"
    game_insert = """struct Game {
    int teams=0,me=0,size=0,turns=0,hp=0,damage=0,respawn=0,flag_return=0,flag_cooldown=0;
    vector<string> map; vector<pair<int,int>> bases,spots;
    int turn=0; vector<int> score; vector<Unit> units; vector<Flag> flags; vector<Event> events;
};

#if MOD_FUSION_P || MOD_FUSION_V
namespace fusion_hook {
    inline fusion::Static makeStatic(const Game& g) {
        fusion::Static s;
        s.teams = g.teams;
        s.me = g.me;
        s.size = g.size;
        s.turns = g.turns;
        s.hp = g.hp;
        s.damage = g.damage;
        s.respawn = g.respawn;
        s.flagReturn = g.flag_return;
        s.flagCooldown = g.flag_cooldown;
        s.wall.assign(g.size * g.size, 0);
        for (int y = 0; y < g.size; y++)
            for (int x = 0; x < g.size; x++)
                s.wall[y * g.size + x] = (g.map[y][x] == '#');
        s.baseCenter.resize(g.bases.size());
        for (size_t i = 0; i < g.bases.size(); i++)
            s.baseCenter[i] = {g.bases[i].first, g.bases[i].second};
        s.spots.resize(g.spots.size());
        for (size_t i = 0; i < g.spots.size(); i++)
            s.spots[i] = {g.spots[i].first, g.spots[i].second};
        return s;
    }

    inline fusion::State makeState(const Game& g) {
        fusion::State s;
        s.turn = g.turn;
        s.score = g.score;
        s.units.resize(g.units.size());
        for (size_t i = 0; i < g.units.size(); i++) {
            s.units[i] = {g.units[i].id, g.units[i].team, g.units[i].x, g.units[i].y, g.units[i].hp, g.units[i].flag, g.units[i].respawn_at};
        }
        s.flags.resize(g.flags.size());
        for (size_t i = 0; i < g.flags.size(); i++) {
            int st = fusion::kHome;
            if (g.flags[i].status == "carried") st = fusion::kCarried;
            else if (g.flags[i].status == "dropped") st = fusion::kDropped;
            else if (g.flags[i].status == "cooldown") st = fusion::kCooldown;
            s.flags[i] = {g.flags[i].id, st, g.flags[i].x, g.flags[i].y, g.flags[i].holder, g.flags[i].return_at};
        }
        for (const auto& e : g.events) {
            if (e.type == "attack" && e.args.size() >= 2)
                s.attacks.push_back({e.args[0], e.args[1]});
        }
        return s;
    }

    struct Session {
        bool initialized = false;
        fusion::Board board;
        fusion::State state, prevState;
        bool hasPrev = false;
        fusion::TurnView view;
        std::vector<fusion::PolicyFeatures> feat;
        std::vector<fusion::PolicyOutput> policyOut;
        std::vector<fusion::LegacyForecast> qh;
        const fusion::models::ModelSet* model = nullptr;

        void update(const Game& g) {
            if (!initialized) {
                board.init(makeStatic(g));
                feat.resize(g.teams);
                policyOut.resize(g.teams);
                qh.resize(g.teams);
                initialized = true;
            }
            if (hasPrev) prevState = state;
            state = makeState(g);
            makeView(board, state, hasPrev ? &prevState : nullptr, view);
            hasPrev = true;
            model = fusion::models::forTeams(g.teams);
            if (model && model->policy && model->alpha > 0) {
                for (int t = 0; t < g.teams; t++) {
                    if (t == g.me) continue;
                    bool active = false;
                    for (int k = 0; k < 3; k++) {
                        int uid = 3 * t + k;
                        if (uid < (int)g.units.size() && view.origin[uid] >= 0) active = true;
                    }
                    if (active) {
                        fusion::encodePolicy(board, state, view, t, feat[t]);
                        fusion::jointPolicy(*model->policy, feat[t], policyOut[t]);
                    }
                }
            }
        }

        template <typename SimType>
        double evalSimValue(const SimType& sim, int turn, int me) const {
            if (!model || !model->value) return 0.5;
            fusion::State st;
            st.turn = turn;
            st.score = sim.score;
            st.units.resize(sim.u.size());
            for (size_t i = 0; i < sim.u.size(); i++) {
                st.units[i] = {sim.u[i].id, sim.u[i].team, sim.u[i].x, sim.u[i].y, sim.u[i].hp, sim.u[i].flag, sim.u[i].respawn_at};
            }
            st.flags.resize(sim.f.size());
            for (size_t i = 0; i < sim.f.size(); i++) {
                int s = fusion::kHome;
                if (sim.f[i].status == "carried") s = fusion::kCarried;
                else if (sim.f[i].status == "dropped") s = fusion::kDropped;
                else if (sim.f[i].status == "cooldown") s = fusion::kCooldown;
                st.flags[i] = {sim.f[i].id, s, sim.f[i].x, sim.f[i].y, sim.f[i].holder, sim.f[i].return_at};
            }
            fusion::TurnView vw;
            fusion::makeView(board, st, nullptr, vw);
            fusion::ValueFeatures vf;
            fusion::encodeValue(board, st, vw, me, vf);
            return fusion::value(*model->value, vf);
        }
    };
}
#endif"""
    assert game_anchor in code
    code = code.replace(game_anchor, game_insert, 1)

    # 3. Session in struct Planner
    planner_anchor = "struct Planner {\n    Nav nav; bool initialized=false; vector<Unit> prev;"
    planner_insert = """struct Planner {
#if MOD_FUSION_P || MOD_FUSION_V
    fusion_hook::Session fSession;
#endif
    Nav nav; bool initialized=false; vector<Unit> prev;"""
    assert planner_anchor in code
    code = code.replace(planner_anchor, planner_insert, 1)

    # 4. In decide: fSession.update(g)
    decide_anchor = "    string decide(Game& g){\n        if(!initialized){nav.init(g);initialized=true;}respawns(g);forecast(g);auto jobs=assign(g);"
    decide_insert = """    string decide(Game& g){
#if MOD_FUSION_P || MOD_FUSION_V
        fSession.update(g);
#endif
        if(!initialized){nav.init(g);initialized=true;}respawns(g);forecast(g);auto jobs=assign(g);"""
    assert decide_anchor in code
    code = code.replace(decide_anchor, decide_insert, 1)

    # 5. In forecast: blend probabilities
    forecast_anchor = """            double mx=*max_element(vals.begin(),vals.end()),sum=0;
            for(size_t i=0;i<vals.size();i++){double v=exp((vals[i]-mx)/.75);e.forecast[i].prob=v;sum+=v;}
            for(auto& o:e.forecast)o.prob=.88*o.prob/sum+.12/e.forecast.size();"""
    forecast_insert = """            double mx=*max_element(vals.begin(),vals.end()),sum=0;
            for(size_t i=0;i<vals.size();i++){double v=exp((vals[i]-mx)/.75);e.forecast[i].prob=v;sum+=v;}
            for(auto& o:e.forecast)o.prob=.88*o.prob/sum+.12/e.forecast.size();
#if MOD_FUSION_P
            int k = u.id % 3, t = u.team;
            if (fSession.model && fSession.model->policy && fSession.model->alpha > 0) {
                for (int m = 0; m < 5; m++) fSession.qh[t].move[k][m] = 0.0;
                for (auto& o : e.forecast) {
                    int dx = o.p.x - (p % nav.n), dy = o.p.y - (p / nav.n);
                    int m = (dx == 0 && dy == 0) ? 0 : (dx == 0 && dy == -1) ? 1 : (dx == 0 && dy == 1) ? 2 : (dx == -1 && dy == 0) ? 3 : 4;
                    fSession.qh[t].move[k][m] = o.prob;
                }
                for (auto& o : e.forecast) {
                    int dx = o.p.x - (p % nav.n), dy = o.p.y - (p / nav.n);
                    int m = (dx == 0 && dy == 0) ? 0 : (dx == 0 && dy == -1) ? 1 : (dx == 0 && dy == 1) ? 2 : (dx == -1 && dy == 0) ? 3 : 4;
                    o.prob = fusion::mixedMarginal(fSession.model->alpha, fSession.qh[t], fSession.policyOut[t], k, m);
                }
            }
#endif"""
    assert forecast_anchor in code
    code = code.replace(forecast_anchor, forecast_insert, 1)

    # 6. In refine: evaluate future state two with V
    refine_anchor = """                double value=.2*simIncrement(g,start,one,jobs)+.7*simIncrement(g,one,two,jobs);
                total+=value*(model==0?.7:.3);scenarios++;"""
    refine_insert = """                double value=.2*simIncrement(g,start,one,jobs)+.7*simIncrement(g,one,two,jobs);
#if MOD_FUSION_V
                if (fSession.model && fSession.model->value) {
                    double vVal = fSession.evalSimValue(two, g.turn + 2, g.me);
                    value += 40.0 * (vVal - 0.5);
                }
#endif
                total+=value*(model==0?.7:.3);scenarios++;"""
    assert refine_anchor in code
    code = code.replace(refine_anchor, refine_insert, 1)

    target_hpp = CANDIDATE_DIR / "Player5_v12_Fusion.hpp"
    target_hpp.write_text(code, encoding="utf-8")
    print(f"Generated {target_hpp} ({len(code)} bytes)")

    v12_variants = {
        "Player5_v12_control.cpp": """// Frozen Player5_v12 Control: MOD_FUSION_P=0, MOD_FUSION_V=0
#define MOD_FUSION_P 0
#define MOD_FUSION_V 0
#include "Player5_v12_Fusion.hpp"
""",
        "Player5_v12_p_s.cpp": """// Player5_v12 Carrier + Tier S Opponent Policy
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_S 1
#include "Player5_v12_Fusion.hpp"
""",
        "Player5_v12_p_m.cpp": """// Player5_v12 Carrier + Tier M Opponent Policy (P-M without V / 卸掉V)
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_M 1
#include "Player5_v12_Fusion.hpp"
""",
        "Player5_v12_pm_no_v.cpp": """// Player5_v12 Carrier + Tier M Opponent Policy (P-M without V / 卸掉V)
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_M 1
#include "Player5_v12_Fusion.hpp"
""",
        "Player5_v12_pm_with_v.cpp": """// Player5_v12 Carrier + Tier M Policy + Value (P-M with V)
#define MOD_FUSION_P 1
#define MOD_FUSION_V 1
#define FUSION_TIER_M 1
#include "Player5_v12_Fusion.hpp"
""",
        "Player5_v12_p_l.cpp": """// Player5_v12 Carrier + Tier L Opponent Policy
#define MOD_FUSION_P 1
#define MOD_FUSION_V 0
#define FUSION_TIER_L 1
#include "Player5_v12_Fusion.hpp"
""",
    }

    for name, content in v12_variants.items():
        vp = CANDIDATE_DIR / name
        vp.write_text(content, encoding="utf-8")
        print(f"Generated {vp}")


if __name__ == "__main__":
    build_fusion_hpp()
    build_player5_v12_fusion()

