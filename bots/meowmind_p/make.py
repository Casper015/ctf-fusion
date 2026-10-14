"""Patch meowmind's search to use our P3 policy, then bundle it into one .cpp.

    python bots/meowmind_p/make.py MEOWMIND.cpp MODEL_DIR OUT.cpp [--own 0.5] [--pexp 1] [--prior 0.10]

MODEL_DIR is an inference/models*/ directory with a p3 registry.hpp (relative to inference/).
--own   weight of log P(joint) in our own candidate prior (0 keeps meowmind's prior).
--pexp  1 adds P's unit marginals as one more expert in meowmind's opponent mixture.
--prior that expert's prior weight (meowmind's experts sum to 1.0).
Every anchor must occur exactly once in the source; otherwise the script stops.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# The repository whose bots/p_only/tools/bundle.py and inference/ are used; FUSION_REPO points at another checkout
# (for example one holding a model set that is not committed).
REPO = Path(os.environ.get("FUSION_REPO", Path(__file__).resolve().parents[2]))

GLUE = r'''
// ---- P3 policy glue (ctf-fusion inference/): our P over the raw TURN block ----
namespace pglue {
struct PRoot {
    bool ready = false, hasPrev = false;
    fusion::Board b; fusion::Rules rules{}; const fusion::PolicyModelP3* model = nullptr;
    fusion::State prev, cur; fusion::TurnView v; fusion::CombatSummary cs;
    std::vector<fusion::PolicyFeaturesP3> feat; std::vector<fusion::PolicyOutput> out; std::vector<char> has;
    void init(const Game& g) {
        fusion::Static st; st.teams = g.teams; st.me = g.me; st.size = g.size; st.turns = g.turns; st.hp = g.hp;
        st.damage = g.damage; st.respawn = g.respawn; st.flagReturn = g.flag_return; st.flagCooldown = g.flag_cooldown;
        st.wall.assign(g.size * g.size, 0);
        for (int y = 0; y < g.size; y++) for (int x = 0; x < g.size; x++) st.wall[y * g.size + x] = g.map[y][x] == '#';
        for (auto& c : g.bases) st.baseCenter.push_back({c.first, c.second});
        for (auto& c : g.spots) st.spots.push_back({c.first, c.second});
        b.init(st);
        rules = {g.hp, g.damage, g.respawn, g.flag_return, g.flag_cooldown};
        auto* set = fusion::models::forTeams(g.teams);
        model = set ? set->policy : nullptr;
        feat.resize(g.teams); out.resize(g.teams); has.assign(g.teams, 0);
        ready = true;
    }
    static fusion::State convert(const Game& g) {
        fusion::State s; s.turn = g.turn; s.score = g.score;
        for (auto& u : g.units) s.units.push_back({u.id, u.team, u.x, u.y, u.hp, u.flag, u.respawn_at});
        for (auto& f : g.flags) {
            int st = f.status == "carried" ? fusion::kCarried : f.status == "dropped" ? fusion::kDropped
                   : f.status == "cooldown" ? fusion::kCooldown : fusion::kHome;
            s.flags.push_back({f.id, st, f.x, f.y, f.holder, f.return_at});
        }
        for (auto& e : g.events) if (e.type == "attack" && e.args.size() >= 2) s.attacks.push_back({e.args[0], e.args[1]});
        return s;
    }
    // One real turn: every team with an active unit gets its 125-way joint.
    void build(const Game& g) {
        cur = convert(g);
        fusion::makeView(b, cur, hasPrev ? &prev : nullptr, v);
        fusion::summarize(b, rules, cur, v, cs);
        has.assign(g.teams, 0);
        if (model) for (int t = 0; t < g.teams; t++) {
            bool active = false;
            for (int k = 0; k < 3; k++) if (v.origin[3 * t + k] >= 0) active = true;
            if (!active) continue;
            fusion::encodePolicyP3(b, cur, v, cs, t, feat[t]);
            fusion::jointPolicy(*model, feat[t], out[t]);
            has[t] = 1;
        }
        prev = cur; hasPrev = true;
    }
};
}  // namespace pglue
'''


def once(src: str, old: str, new: str) -> str:
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor found {n} times, expected 1: {old[:80]!r}")
    return src.replace(old, new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("model_dir"); ap.add_argument("out")
    ap.add_argument("--own", type=float, default=0.5)
    ap.add_argument("--pexp", type=int, default=1)
    ap.add_argument("--prior", type=float, default=0.10)
    a = ap.parse_args()
    s = Path(a.src).read_text(encoding="utf-8")

    head = (f"#define P_OWN {a.own}f\n#define PEXP {a.pexp}\n#define P_EXPERT_PRIOR {a.prior}f\n"
            f"#define FUSION_MAX_WIDTH 1024\n#include \"{a.model_dir}/registry.hpp\"\n")
    s = head + s
    s = once(s, "using State=Game;\n", "using State=Game;\n" + GLUE)

    # Expert count: meowmind's NN learned heads + 5 hand experts (+ P).
    s = once(s, "E=NN+5,", "E=NN+5+PEXP,")
    s = once(s, "    uint64_t rng=0xD1B54A32D192ED03ull;\n",
             "    uint64_t rng=0xD1B54A32D192ED03ull;\n    pglue::PRoot proot;bool rootP=false;\n")
    s = once(s, "for(int modelId=NN;modelId<E;modelId++){", "for(int modelId=NN;modelId<NN+5;modelId++){")
    s = once(s, "            for(float& q:model[modelId]){q=std::exp(q-maximum);sum+=q;}for(float& q:model[modelId])q/=sum;\n        }return model;",
             "            for(float& q:model[modelId]){q=std::exp(q-maximum);sum+=q;}for(float& q:model[modelId])q/=sum;\n        }\n"
             "        if constexpr(PEXP){\n"
             "            // P's marginal at the real root; elsewhere a copy of head 0 (evidence only uses root predictions).\n"
             "            int team=u.team,k=id-3*team;\n"
             "            if(rootP&&proot.has[team])for(int m=0;m<5;m++)model[NN+5][m]=float(proot.out[team].marginal[k][m]);\n"
             "            else model[NN+5]=model[0];\n"
             "        }\n        return model;")
    s = once(s, "        const float fallback[5]={.08f,.06f,.06f,.08f,.02f};for(int k=0;k<5;k++)prior[NN+k]=fallback[k];\n",
             "        const float fallback[5]={.08f,.06f,.06f,.08f,.02f};for(int k=0;k<5;k++)prior[NN+k]=fallback[k];\n"
             "        if constexpr(PEXP)prior[NN+5]=P_EXPERT_PRIOR;\n")
    s = once(s, "w[k]=likelihood+std::log(k<NN?.70f/NN:fallback[k-NN]);",
             "w[k]=likelihood+std::log(k<NN?.70f/NN:k<NN+5?fallback[k-NN]:P_EXPERT_PRIOR);")

    # Root: P for every team, then our own candidates under the blended prior.
    s = once(s, "        auto prob=predict(observation,true);auto options=moves(observation,g.me,prob);\n",
             "        if(!proot.ready)proot.init(g);proot.build(g);rootP=true;\n"
             "        auto prob=predict(observation,true);rootP=false;auto options=ownMoves(observation,prob);\n")
    s = once(s, "    std::array<int,3> shots(",
             "    // Our 125 joint moves: legal under either policy, no shared destination; prior blends\n"
             "    // meowmind's sum of unit log-probs with log P(joint).\n"
             "    std::vector<Candidate> ownMoves(const Game& g,const std::vector<Prob>& prob)const{\n"
             "        if(!(P_OWN>0)||!proot.has[g.me])return moves(g,g.me,prob);\n"
             "        const auto& P=proot.out[g.me];std::vector<Candidate> out;int start[3];\n"
             "        for(int k=0;k<3;k++)start[k]=geo.cell(g.units[g.me*3+k]);\n"
             "        for(int j=0;j<125;j++){\n"
             "            Candidate ca;ca.moves={j/25,j/5%5,j%5};int p[3];bool valid=true;float mm=0;\n"
             "            for(int k=0;k<3;k++){int id=g.me*3+k,m=ca.moves[k];\n"
             "                if(start[k]<0&&m){valid=false;break;}\n"
             "                if(start[k]>=0&&m&&geo.nb[start[k]][m]==start[k]){valid=false;break;}\n"
             "                p[k]=start[k]<0?-1:geo.nb[start[k]][m];mm+=std::log(std::max(.00001f,prob[id][m]));}\n"
             "            if(!valid)continue;\n"
             "            for(int k=0;k<3;k++)for(int l=k+1;l<3;l++)if(p[k]>=0&&p[k]==p[l])valid=false;\n"
             "            if(!valid)continue;\n"
             "            float lp=std::log(std::max(1e-6f,float(P.prob[j])));\n"
             "            ca.prior=(1-P_OWN)*mm+P_OWN*lp;out.push_back(ca);\n"
             "        }\n"
             "        if(out.empty())return moves(g,g.me,prob);\n"
             "        std::sort(out.begin(),out.end(),[](const auto& a,const auto& b){return a.prior>b.prior;});return out;\n"
             "    }\n"
             "    std::array<int,3> shots(")

    tmp = Path(a.out).with_suffix(".patched.cpp")
    tmp.write_text(s, encoding="utf-8")
    r = subprocess.run([sys.executable, str(REPO / "bots/p_only/tools/bundle.py"), str(tmp), "-o", a.out])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
