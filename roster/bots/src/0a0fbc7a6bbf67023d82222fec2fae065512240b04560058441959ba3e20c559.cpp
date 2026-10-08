// R3_FLAGS_joint: Direction 5a: flag tasks valued by securing x return probability with strict feasibility.
// Generated from nextgen/master.cpp 45b345f8003fe7b2 with modules carrier_dp.inc c917208fbadc, drop.inc 6ac0f3c5637b, expectimax.inc 99bb121422a6, habits.inc b582335e92ec, habits_duel.inc 644840cb5b5f, habits_multi.inc 62db5ef59afc, macro5.inc 5c6cfaacf57b.
// Lineage: Control_Hybrid 98a9568798ee, R3_CL_checked 294846930414, R3_CFL_conditional 1e9c6e0d0ebd.
// Enabled: MOD_CHECKED, MOD_FLAGSHOOT, MOD_FLAGS, MOD_HYBRID_MACRO. Per-team-count masks MOD_*_SIZES default to every applicable size.
#define MOD_CHECKED 1
#define MOD_FLAGSHOOT 1
#define MOD_CFL 0
#define MOD_DROP 0
#define MOD_HABITS 0
#define MOD_TERMINAL 0
#define MOD_CDP 0
#define MOD_FLAGS 1
#define MOD_ESCORT 0
#define MOD_HEALTHY 0
#define MOD_EXPECTIMAX 0
#define MOD_WEIGHTS_TUNED 0
#define MOD_HYBRID_MACRO 1
#define NG_TRACE 0
#define NG_UNCACHED_NEIGHBORS 0
#define CTF_RELAY 0
// Hybrid CTF bot: duel planner for 2-team games, multi-team combat planner for 3-15 teams.
// Built from two user-supplied reference bots (experiments/ref-demo.cpp and experiments/ref-player5.cpp).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

struct Unit { int id, team, x, y, hp, flag, respawn_at; };      // Dead units have x = y = -1
struct Flag { int id; string status; int x, y, holder, return_at; };
struct Event { string type; vector<int> args; };

struct Game {
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

bool read_turn(Game& g) {
    string w;
    while (cin >> w) {
        if (w == "INIT") {
            cin >> g.teams >> g.me >> g.size >> g.turns >> g.hp >> g.damage >> g.respawn >> g.flag_return >> g.flag_cooldown;
        } else if (w == "MAP") {
            g.map.assign(g.size, "");
            for (auto& row : g.map) cin >> row;
        } else if (w == "BASES") {
            int k; cin >> k; g.bases.resize(k);
            for (auto& b : g.bases) cin >> b.first >> b.second;
        } else if (w == "SPOTS") {
            int k; cin >> k; g.spots.resize(k);
            for (auto& s : g.spots) cin >> s.first >> s.second;
        } else if (w == "TURN") {
            cin >> g.turn;
        } else if (w == "SCORE") {
            g.score.resize(g.teams);
            for (auto& s : g.score) cin >> s;
        } else if (w == "UNITS") {
            int k; cin >> k; g.units.resize(k);
            for (auto& u : g.units) { string tag; cin >> tag >> u.id >> u.team >> u.x >> u.y >> u.hp >> u.flag >> u.respawn_at; }
        } else if (w == "FLAGS") {
            int k; cin >> k; g.flags.resize(k);
            for (auto& f : g.flags) { string tag; cin >> tag >> f.id >> f.status >> f.x >> f.y >> f.holder >> f.return_at; }
        } else if (w == "EVENTS") {
            int k; cin >> k; g.events.assign(k, {});
            string line; getline(cin, line);
            for (auto& e : g.events) {
                getline(cin, line);
                istringstream in(line);
                string tag; in >> tag >> e.type;
                for (int v; in >> v;) e.args.push_back(v);
            }
        } else if (w == "END") {
            return true;
        }
    }
    return false;
}

#ifndef C_MULTI
#define C_MULTI 1.0
#endif
#ifndef C_DUEL
#define C_DUEL 1.0
#endif
#ifndef F_MULTI
#define F_MULTI 0.0
#endif
#ifndef F_DUEL
#define F_DUEL 1.0
#endif
#ifndef PICK_BONUS
#define PICK_BONUS 8.5
#endif
#ifndef SHOT_MIX
#define SHOT_MIX 0.0
#endif
#ifndef COLLISION_WEIGHT
#define COLLISION_WEIGHT .7
#endif

// Next-generation module switches (numeric; 0 disables). The validated Player 5
// foundation is MOD_CHECKED (R3_CL_checked lookahead) plus MOD_FLAGSHOOT
// (R3_CFL_conditional fire and neighborhood cache). Exports resolve every switch.
// Control_Hybrid's macro layer (timed camp coverage, kept camp option, flag
// duplicate support rule). 0 = the plain Player5 macro layer used by the R3
// planner and by Player2's v005 (Hybrid_Expectimax).
// Per-team-count activation: bit t of a mask enables that module in t-team
// games, so one binary can be evaluated, and later composed, size by size.
#define NG_SIZES_ALL 0xFFFCu
#define NG_SIZES_MULTI 0xFFF8u
#ifndef MOD_CFL_SIZES
#define MOD_CFL_SIZES NG_SIZES_ALL
#endif
#ifndef MOD_DROP_SIZES
#define MOD_DROP_SIZES NG_SIZES_MULTI
#endif
#ifndef MOD_HABITS_SIZES
#define MOD_HABITS_SIZES NG_SIZES_ALL
#endif
#ifndef MOD_TERMINAL_SIZES
#define MOD_TERMINAL_SIZES NG_SIZES_ALL
#endif
#ifndef MOD_CDP_SIZES
#define MOD_CDP_SIZES NG_SIZES_MULTI
#endif
#ifndef MOD_FLAGS_SIZES
#define MOD_FLAGS_SIZES NG_SIZES_ALL
#endif
#ifndef MOD_ESCORT_SIZES
#define MOD_ESCORT_SIZES NG_SIZES_ALL
#endif
#ifndef MOD_HEALTHY_SIZES
#define MOD_HEALTHY_SIZES NG_SIZES_ALL
#endif
#ifndef MOD_EXPECTIMAX_SIZES
#define MOD_EXPECTIMAX_SIZES NG_SIZES_ALL
#endif
#ifndef MOD_HYBRID_MACRO_SIZES
#define MOD_HYBRID_MACRO_SIZES NG_SIZES_ALL
#endif
struct NgActive {bool cfl=false,drop=false,habits=false,terminal=false,cdp=false,flags=false,escort=false,healthy=false,expectimax=false,hybridMacro=false;};
inline NgActive ngActive(int teams){
    auto on=[&](bool compiled,unsigned mask){return compiled&&teams>=0&&teams<32&&((mask>>teams)&1u);};
    NgActive a;a.cfl=on(MOD_CFL,MOD_CFL_SIZES);a.drop=on(MOD_DROP,MOD_DROP_SIZES);a.habits=on(MOD_HABITS,MOD_HABITS_SIZES);
    a.terminal=on(MOD_TERMINAL,MOD_TERMINAL_SIZES);a.cdp=on(MOD_CDP,MOD_CDP_SIZES);a.flags=on(MOD_FLAGS,MOD_FLAGS_SIZES);
    a.escort=on(MOD_ESCORT,MOD_ESCORT_SIZES);a.healthy=on(MOD_HEALTHY,MOD_HEALTHY_SIZES);a.expectimax=on(MOD_EXPECTIMAX,MOD_EXPECTIMAX_SIZES);
    a.hybridMacro=on(MOD_HYBRID_MACRO,MOD_HYBRID_MACRO_SIZES);
    return a;
}


namespace multi {
#ifndef CTF_LOOKAHEAD
#define CTF_LOOKAHEAD 1
#endif

// Hand-set defaults: the pre-tuning literals (round2 autotune.py DEFAULT).
constexpr double W_ground=30;
constexpr double W_home_small=.48;
constexpr double W_home_large=.8;
constexpr double W_approach=.9;
constexpr double W_late=.9;
constexpr double W_camp=12;
constexpr double W_camp_home=.32;
constexpr double W_camp_approach=.6;
constexpr double W_camp_duplicate=8;
constexpr double W_flag_duplicate=14;
constexpr double W_contest_discount=3;
constexpr double W_escort=11;
constexpr double W_deny_small=19;
constexpr double W_deny_large=17;
constexpr double W_deny_remote=.32;
constexpr double W_cohesion=.8;
constexpr double W_other_attention=.35;
constexpr double W_pickup=22;
constexpr double W_route=1.7;
constexpr double W_death=12;
constexpr double W_hit=1.1;
constexpr double W_damage=.04;
constexpr double W_carrier_death=45;
constexpr double W_carrier_hit=3;
constexpr double W_worst=10;
namespace ctfcombat {

struct Pos { int x = -1, y = -1; };
inline bool operator==(Pos a, Pos b) { return a.x == b.x && a.y == b.y; }
inline bool operator!=(Pos a, Pos b) { return !(a == b); }
inline int distance(Pos a, Pos b) { return std::abs(a.x-b.x) + std::abs(a.y-b.y); }
struct Fighter {
    int id = -1, team = -1;
    Pos p;
    int hp = 0, flag = -1;
    bool immune = false;
};
struct Outcome { Pos p; double prob = 0; bool canShoot = true; double shootChance = 1.0; };
struct Enemy {
    Fighter u;
    std::vector<Outcome> forecast;
    double attention = 1.0;
    double killValue = 11.0;
};
struct Move {
    Pos p;
    char code = 'S';
    double score = 0;
    double pickupValue = 0;
    double captureValue = 0;
};
struct Params {
    double dealtHpValue = 0.04;
    double carrierKillExtra = 18.0;
    double incomingHitCost = 1.1;
    double ownDeathCost = 12.0;
    double carrierDeathExtra = 45.0;
    double carrierHitExtra = 3.0;
    double carrierWorstDeathCost = 10.0;
    double collisionCost = 1.5;
};
struct Choice {
    std::array<int,3> moveIndex{0,0,0};
    // -1 idle, -2 pickup (default for any unladen unit), otherwise enemy ID.
    std::array<int,3> target{-2,-2,-2};
    double value = -std::numeric_limits<double>::infinity();
    // Movement/task score before tactical and capture terms; the checked
    // lookahead replaces those immediate terms by realized branch outcomes.
    double positionValue = 0;
};

inline bool alive(const Fighter& u) { return u.hp > 0 && u.p.x >= 0; }
inline bool wall(const std::vector<std::string>& map, Pos p) {
    return p.x < 0 || p.y < 0 || p.y >= int(map.size()) ||
        p.x >= int(map[p.y].size()) || map[p.y][p.x] == '#';
}
inline bool canAttack(const std::vector<std::string>& map, Pos a, Pos b) {
    int dx = b.x-a.x, dy = b.y-a.y;
    if (std::abs(dx)+std::abs(dy) > 2) return false;
    if (std::abs(dx) == 2 && dy == 0) return !wall(map,{a.x+dx/2,a.y});
    if (std::abs(dy) == 2 && dx == 0) return !wall(map,{a.x,a.y+dy/2});
    if (std::abs(dx) == 1 && std::abs(dy) == 1)
        return !wall(map,{a.x+dx,a.y}) || !wall(map,{a.x,a.y+dy});
    return true;
}

namespace detail {

inline bool blocks(const Fighter& own, Pos want, const Fighter& e, Pos eWant) {
    if (!alive(own) || !alive(e) || want == own.p) return false;
    return want == eWant || (want == e.p && eWant == own.p);
}

// Exact movement for the three allies and one sampled enemy. Remaining enemies
// are represented by a sampled mask of initially blocked allied movements.
inline std::array<Pos,4> simulate(const std::array<Fighter,3>& own,
        const std::array<Pos,3>& want, const Fighter& enemy, Pos eWant,
        int externalBlocked) {
    std::array<Pos,4> start{own[0].p,own[1].p,own[2].p,enemy.p};
    std::array<Pos,4> dest{want[0],want[1],want[2],eWant};
    std::array<bool,4> live{alive(own[0]),alive(own[1]),alive(own[2]),alive(enemy)};
    int blocked = externalBlocked;
    for (int i=0;i<4;++i) if (live[i]) for(int j=i+1;j<4;++j) if(live[j]) {
        if (dest[i] == dest[j]) blocked |= (1<<i)|(1<<j);
        if (j == 3 && dest[i] == start[j] && dest[j] == start[i])
            blocked |= (1<<i)|(1<<j);
    }
    for (int pass=0;pass<4;++pass) {
        int before=blocked;
        for(int i=0;i<4;++i) if(live[i] && !(blocked&(1<<i)) && dest[i]!=start[i])
            for(int j=0;j<4;++j) if(i!=j && live[j] && dest[i]==start[j] &&
                    ((blocked&(1<<j)) || dest[j]==start[j])) blocked|=1<<i;
        if (blocked == before) break;
    }
    for(int i=0;i<4;++i) if(blocked&(1<<i)) dest[i]=start[i];
    return dest;
}

inline double priority(const Fighter& u, Pos from, Pos at) {
    double value = u.flag >= 0 ? 5.0 : 1.0;
    if (u.hp <= 34) value *= 2.4;
    else if (u.hp <= 68) value *= 1.35;
    if (distance(from,at) <= 1) value *= 1.12;
    return value;
}

struct DpCell {
    double value=-std::numeric_limits<double>::infinity();
    std::array<int,3> target{-2,-2,-2};
};

} // namespace detail

inline Choice choose(const std::vector<std::string>& map,
        const std::array<Fighter,3>& own,
        const std::array<std::vector<Move>,3>& options,
        const std::vector<Enemy>& enemies,
        const Params& params = Params{}, std::vector<Choice>* beam=nullptr, const std::chrono::steady_clock::time_point* deadline=nullptr) {
    Choice best;
    if(options[0].empty() || options[1].empty() || options[2].empty()) return best;
    std::array<double,3> stayScore{},stayCapture{};
    int armedMask=0;
    for(int i=0;i<3;++i) {
        stayScore[i]=options[i][0].score;
        for(const Move& m:options[i]) if(m.p==own[i].p) {stayScore[i]=m.score;stayCapture[i]=m.captureValue;}
        if(alive(own[i]) && !own[i].immune && own[i].flag<0) armedMask|=1<<i;
    }
    const int E=int(enemies.size());
    std::vector<bool> relevant(E,false);
    for(int e=0;e<E;++e) if(alive(enemies[e].u))
        for(int i=0;i<3;++i) if(alive(own[i]) && distance(enemies[e].u.p,own[i].p)<=4)
            relevant[e]=true;
    // Crossfire between opponents is independent of our move triple.
    std::vector<double> crossStay(E,0.);
    std::vector<std::vector<double>> crossMove(E);
    for(int e=0;e<E;++e)if(relevant[e]){
        auto cross=[&](Pos p){double value=0;for(int f=0;f<E;++f)if(f!=e&&alive(enemies[f].u)&&enemies[f].u.team!=enemies[e].u.team&&!enemies[f].u.immune&&canAttack(map,p,enemies[f].u.p))value+=detail::priority(enemies[f].u,p,enemies[f].u.p);return value;};
        crossStay[e]=cross(enemies[e].u.p);for(auto& out:enemies[e].forecast)crossMove[e].push_back(cross(out.p));
    }
    std::vector<std::array<double,3>> directBlock(E),incoming(E);
    std::vector<std::array<double,8>> utilities(E);
    for(int a=0;a<int(options[0].size());++a)
    for(int b=0;b<int(options[1].size());++b)
    for(int c=0;c<int(options[2].size());++c) {
        if(deadline&&std::isfinite(best.value)&&((a*25+b*5+c)&15)==0&&std::chrono::steady_clock::now()>=*deadline)return best;
        std::array<int,3> indices{a,b,c};
        std::array<Pos,3> want{options[0][a].p,options[1][b].p,options[2][c].p};
        bool valid=true;
        for(int i=0;i<3;++i) if(alive(own[i])) {
            if(wall(map,want[i])) valid=false;
            for(int j=i+1;j<3;++j) if(alive(own[j]) && want[i]==want[j]) valid=false;
        }
        if(!valid) continue;

        // Closure of movement dependence within our team; same-team cycles work.
        std::array<int,3> dependencies{1,2,4};
        for(int pass=0;pass<3;++pass) for(int i=0;i<3;++i) if(alive(own[i]))
            for(int j=0;j<3;++j) if(i!=j && alive(own[j]) && want[i]==own[j].p)
                dependencies[i]|=dependencies[j];

        for(auto& row:directBlock)row.fill(0.);
        std::array<double,3> success{1,1,1};
        for(int e=0;e<E;++e) {
            if(!relevant[e]) continue;
            for(const Outcome& out:enemies[e].forecast) {
                for(int i=0;i<3;++i) if(detail::blocks(own[i],want[i],enemies[e].u,out.p)) {
                    directBlock[e][i]+=out.prob;
                }
            }
            for(int i=0;i<3;++i) {
                double p=0;
                for(const Outcome& out:enemies[e].forecast) {
                    bool hit=false;
                    for(int j=0;j<3;++j) if((dependencies[i]&(1<<j)) &&
                        detail::blocks(own[j],want[j],enemies[e].u,out.p)) hit=true;
                    if(hit) p+=out.prob;
                }
                success[i]*=std::max(0.0,1.0-p);
            }
        }
        double value=0;
        for(int i=0;i<3;++i) if(alive(own[i])) {
            value+=success[i]*options[i][indices[i]].score+(1-success[i])*stayScore[i];
            if(want[i]!=own[i].p) value-=params.collisionCost*(1-success[i]);
        }

        // Threatened squads retain mutual support before the first volley.
        for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)
            if((armedMask&(1<<i)) && (armedMask&(1<<j)) && distance(own[i].p,own[j].p)<=5) {
                bool transitThreat=false;
                for(const Enemy& e:enemies) if(e.u.flag<0 && !e.u.immune &&
                    (distance(e.u.p,own[i].p)<=6 || distance(e.u.p,own[j].p)<=6)) {transitThreat=true;break;}
                if(transitThreat)value-=W_cohesion*std::max(0,distance(want[i],want[j])-3);
            }
        const double positionValue=value;

        for(auto& row:incoming)row.fill(0.);
        std::array<int,3> worstShots{};
        for(auto& row:utilities)row.fill(0.);

        for(int e=0;e<E;++e) {
            const Enemy& enemy=enemies[e];
            if(!relevant[e]) continue;
            std::array<double,3> pOther{};
            for(int i=0;i<3;++i) {
                double unblocked=1;
                for(int f=0;f<E;++f) if(f!=e) unblocked*=std::max(0.0,1-directBlock[f][i]);
                pOther[i]=1-unblocked;
            }
            std::array<double,8> blockProb{};
            for(int mask=0;mask<8;++mask) {
                blockProb[mask]=1;
                for(int i=0;i<3;++i) blockProb[mask]*=(mask&(1<<i)) ? pOther[i] : 1-pOther[i];
            }
            std::array<bool,3> possibleShot{};
            for(int oi=0;oi<(int)enemy.forecast.size();oi++) { const Outcome& out=enemy.forecast[oi];if(out.prob<=1e-9)continue;
                for(int block=0;block<8;++block) if(blockProb[block]>1e-9) {
                    const auto pos=detail::simulate(own,want,enemy.u,out.p,block);
                    double probability=out.prob*blockProb[block];
                    int hitMask=0;
                    if(!enemy.u.immune) for(int i=0;i<3;++i)
                        if((armedMask&(1<<i)) && canAttack(map,pos[i],pos[3])) hitMask|=1<<i;
                    for(int subset=1;subset<8;++subset) if(!(subset&~armedMask)) {
                        int hits=0;
                        for(int i=0;i<3;++i) if((subset&hitMask)&(1<<i)) ++hits;
                        double offensive=std::min(enemy.u.hp,34*hits)*params.dealtHpValue;
                        if(34*hits>=enemy.u.hp)
                            offensive+=enemy.killValue+(enemy.u.flag>=0 ? params.carrierKillExtra : 0);
                        utilities[e][subset]+=probability*offensive;
                    }
                    // A zero conditional firing chance cannot add a worst-case shot.
                    if(enemy.u.flag>=0 || enemy.u.immune || !out.canShoot || out.shootChance<=1e-9) continue;
                    std::array<double,3> weights{};
                    double denominator=0;
                    for(int i=0;i<3;++i) if(alive(own[i]) && !own[i].immune && canAttack(map,pos[3],pos[i])) {
                        weights[i]=detail::priority(own[i],pos[3],pos[i]);
                        denominator+=weights[i];
                        possibleShot[i]=true;
                    }
                    // Crossfire competition makes large multiplayer battles less
                    // suicidal than assuming every hostile attacks our team.
                    denominator+=pos[3]==enemy.u.p?crossStay[e]:crossMove[e][oi];
                    if(denominator>0) for(int i=0;i<3;++i)
                        incoming[e][i]+=probability*out.shootChance*std::clamp(enemy.attention,0.0,1.0)*weights[i]/denominator;
                }
            }
            for(int i=0;i<3;++i) if(possibleShot[i]) ++worstShots[i];
        }

        std::array<double,3> survival{1,1,1};
        for(int i=0;i<3;++i) if(alive(own[i]) && !own[i].immune) {
            const int lethal=(own[i].hp+33)/34;
            std::array<double,4> hitCount{1,0,0,0};
            double expectedHits=0;
            for(int e=0;e<E;++e) {
                double p=std::clamp(incoming[e][i],0.0,1.0);
                expectedHits+=p;
                auto next=hitCount;
                next.fill(0);
                for(int k=0;k<4;++k) {
                    next[k]+=hitCount[k]*(1-p);
                    next[std::min(3,k+1)]+=hitCount[k]*p;
                }
                hitCount=next;
            }
            double death=0;
            for(int k=lethal;k<4;++k) death+=hitCount[k];
            survival[i]=1-death;
            value-=expectedHits*(params.incomingHitCost+(own[i].flag>=0 ? params.carrierHitExtra : 0));
            value-=death*(params.ownDeathCost+(own[i].flag>=0 ? params.carrierDeathExtra : 0));
            if(own[i].flag>=0 && worstShots[i]>=lethal)
                value-=params.carrierWorstDeathCost*(1-death);
        }
        // Attacks and deaths resolve before delivery. A base entry only pays
        // when the carrier actually reaches it and survives the volley.
        for(int i=0;i<3;++i) if(alive(own[i]) && own[i].flag>=0)
            value+=survival[i]*(success[i]*options[i][indices[i]].captureValue+
                               (1-success[i])*stayCapture[i]);

        std::array<detail::DpCell,8> dp;
        dp[0].value=0;
        for(int e=0;e<E;++e) if(alive(enemies[e].u) && !enemies[e].u.immune) {
            auto next=dp;
            for(int used=0;used<8;++used) if(std::isfinite(dp[used].value)) {
                int remaining=armedMask&~used;
                for(int sub=remaining;sub;sub=(sub-1)&remaining) {
                    if(utilities[e][sub]<=1e-8) continue;
                    int combined=used|sub;
                    double candidate=dp[used].value+utilities[e][sub];
                    if(candidate>next[combined].value+1e-9) {
                        next[combined]=dp[used];
                        next[combined].value=candidate;
                        for(int i=0;i<3;++i) if(sub&(1<<i)) next[combined].target[i]=enemies[e].u.id;
                    }
                }
            }
            dp=next;
        }
        double actionValue=-std::numeric_limits<double>::infinity();
        std::array<int,3> targets{-2,-2,-2};
        for(int used=0;used<8;++used) if(std::isfinite(dp[used].value)) {
            double candidate=dp[used].value;
            for(int i=0;i<3;++i) if(!(used&(1<<i)) && own[i].flag<0)
                candidate+=options[i][indices[i]].pickupValue*success[i]*survival[i];
            if(candidate>actionValue+1e-9) { actionValue=candidate; targets=dp[used].target; }
        }
        for(int i=0;i<3;++i) if(own[i].flag>=0) targets[i]=-1;
        value+=actionValue;
        if(beam){Choice candidate{indices,targets,value,positionValue};
            if(beam->size()<8)beam->push_back(candidate);
            else{int w=0;for(int z=1;z<(int)beam->size();z++)if((*beam)[z].value<(*beam)[w].value)w=z;
                if(value>(*beam)[w].value)(*beam)[w]=candidate;}}
        if(value>best.value+1e-9) best={indices,targets,value,positionValue};
    }
    return best;
}

} // namespace ctfcombat

struct Nav {
    int n=0,N=0; vector<array<int,5>> nb; vector<uint16_t> all;
    vector<vector<int>> home; vector<vector<int>> spawn; vector<int> floors;
    int cell(int x,int y)const{return y*n+x;}
    int dist(int a,int b)const{return a<0||b<0?999:all[(size_t)a*N+b];}
    bool open(int p)const{return p>=0&&p<N&&nb[p][0]>=0;}
    void init(const Game& g) {
        n=g.size;N=n*n;nb.resize(N);all.assign((size_t)N*N,999);home.assign(g.teams,vector<int>(N,999));spawn.resize(g.teams);
        const int dx[]={0,0,0,-1,1},dy[]={0,-1,1,0,0};
        for(int p=0;p<N;p++) {
            nb[p].fill(-1);int x=p%n,y=p/n;if(g.map[y][x]=='#')continue;floors.push_back(p);
            for(int m=0;m<5;m++){int xx=x+dx[m],yy=y+dy[m];if(xx>=0&&xx<n&&yy>=0&&yy<n&&g.map[yy][xx]!='#')nb[p][m]=cell(xx,yy);}
        }
        vector<int> q(N);
        for(int s:floors){auto* d=&all[(size_t)s*N];int b=0,e=0;q[e++]=s;d[s]=0;while(b<e){int p=q[b++];for(int m=1;m<5;m++){int v=nb[p][m];if(v>=0&&d[v]==999){d[v]=d[p]+1;q[e++]=v;}}}}
        int c=n/2;
        for(int t=0;t<g.teams;t++){
            int bx=g.bases[t].first,by=g.bases[t].second;for(int yy=by-1;yy<=by+1;yy++)for(int xx=bx-1;xx<=bx+1;xx++)spawn[t].push_back(cell(xx,yy));
            stable_sort(spawn[t].begin(),spawn[t].end(),[&](int a,int b){auto key=[&](int p){return pair<int,int>{abs(p%n-c)+abs(p/n-c),abs(p%n-bx)+2*abs(p/n-by)};};return key(a)<key(b);});
            for(int p:floors)for(int b:spawn[t])home[t][p]=min(home[t][p],dist(p,b));
        }
    }
};
struct Task {int type=0,id=-1,p=-1;double reward=0,contest=0;};
struct Planner {
    Nav nav; bool initialized=false; vector<Unit> prev;
    // Module activation for this game's team count (cached; see ngActive).
    mutable NgActive act;mutable int actTeams=-1;
    const NgActive& active(const Game& g)const{if(actTeams!=g.teams){act=ngActive(g.teams);actTeams=g.teams;}return act;}
    array<int,3> oldType{{-1,-1,-1}},oldId{{-1,-1,-1}};
    vector<int> pos; vector<bool> immune; vector<double> threat; array<vector<double>,3> route;
    vector<ctfcombat::Enemy> foes;
    static constexpr const char* letters="SUDLR";
    int man(int a,int b)const {return a<0||b<0?999:abs(a%nav.n-b%nav.n)+abs(a/nav.n-b/nav.n);}
    bool shot(int a,int b,const Game& g)const{
        if(a<0||b<0||man(a,b)>2)return false;
        int x=a%nav.n,y=a/nav.n,dx=b%nav.n-x,dy=b/nav.n-y;
        if(abs(dx)==2&&dy==0)return g.map[y][x+dx/2]!='#';
        if(abs(dy)==2&&dx==0)return g.map[y+dy/2][x]!='#';
        if(abs(dx)==1&&abs(dy)==1)return g.map[y][x+dx]!='#'||g.map[y+dy][x]!='#';return true;
    }
    void respawns(const Game& g){
        pos.assign(g.units.size(),-1);immune.assign(g.units.size(),false);vector<bool> occupied(nav.N,false);
        for(auto& u:g.units)if(u.x>=0){pos[u.id]=nav.cell(u.x,u.y);occupied[pos[u.id]]=true;}
        for(auto& u:g.units)if(u.x<0&&u.respawn_at<=g.turn){for(int p:nav.spawn[u.team])if(!occupied[p]){pos[u.id]=p;occupied[p]=true;immune[u.id]=true;break;}}
    }
    bool available(const Flag& f,const Game& g)const{return f.x>=0&&(f.status=="home"||(f.status=="dropped"&&f.return_at>g.turn));}
    int flagEta(const Unit& u,int q,const Game& g)const {
        if(u.flag>=0)return 999;
        if(pos[u.id]>=0)return nav.dist(pos[u.id],q);
        return 999;
    }
    double timedCampValue(int from,int q,const Game& g)const {
        if(from<0||q<0)return 0;
        int homeFlags=0;bool excluded=false;
        for(const auto& f:g.flags)if(f.status=="home"){
            ++homeFlags;if(f.x>=0&&q==nav.cell(f.x,f.y))excluded=true;
        }
        if(excluded)return 0;
        const int empty=max(1,(int)g.spots.size()-homeFlags);
        const int approach=nav.dist(from,q),home=nav.home[g.me][q];
        double expected=0;
        for(const auto& f:g.flags)if(f.status=="cooldown"||f.status=="dropped"){
            const int delay=f.return_at-g.turn;
            if(delay<0||delay>4||approach>delay+1||f.return_at+home>g.turns)continue;
            if(delay>0&&f.status=="dropped"&&f.x>=0){
                int recovery=999;const int fp=nav.cell(f.x,f.y);
                for(const auto& v:g.units)if(v.flag<0&&pos[v.id]>=0)recovery=min(recovery,nav.dist(pos[v.id],fp));
                if(recovery<=delay)continue;
            }
            expected+=12.0/empty/(1.0+.3*delay)/(1.0+home/12.0);
        }
        return expected;
    }

    bool groundFlagAt(int q,const Game& g)const{for(const auto& f:g.flags)if(available(f,g)&&q==nav.cell(f.x,f.y))return true;return false;}
    // R3_CFL_conditional pickup-versus-fire model for an unladen actor at q.
    double groundPickupProbability(const Game& g,const Unit& u,int q)const{
        int pressure=0,support=0;
        for(const auto& v:g.units)if(pos[v.id]>=0&&v.flag<0&&!immune[v.id]&&man(q,pos[v.id])<=3){if(v.team==u.team)support++;else pressure++;}
        return pressure==0?.98:(u.hp<=34?.3:support>=3&&pressure>=2?.4:.65);
    }
    int closestGoal(const Unit& u,const Game& g)const{
        int p=pos[u.id],best=-1;double bv=1e9;if(u.flag>=0)return -2;
        for(auto& f:g.flags)if(available(f,g)){
            int q=nav.cell(f.x,f.y),d=nav.dist(p,q);if(f.status=="dropped"&&d>f.return_at-g.turn)continue;
            double v=d+0.28*nav.home[u.team][q];if(v<bv){bv=v;best=q;}
        }
        if(best>=0)return best;
        if(u.team<(int)g.spots.size())return nav.cell(g.spots[u.team].first,g.spots[u.team].second);
        return nav.cell(nav.n/2,nav.n/2);
    }
    // ---- Direction 5 (MOD_FLAGS, MOD_ESCORT, MOD_HEALTHY) inside multi::Planner.
    // Joint flag-spot value = mission reward x P(secure first) x P(return home).
    // P(return): Poisson survival of the expected shots met on the least
    // hazardous route from the spot to any base tile (public threat field).
    vector<double> hazardHome;
    void buildHazardHome(const Game& g){
        hazardHome.assign(nav.N,1e9);priority_queue<pair<double,int>,vector<pair<double,int>>,greater<pair<double,int>>> pq;
        for(int p:nav.spawn[g.me]){hazardHome[p]=0;pq.push({0,p});}
        while(!pq.empty()){auto [v,p]=pq.top();pq.pop();if(v>hazardHome[p]+1e-12)continue;
            for(int m=1;m<5;m++){int q=nav.nb[p][m];if(q<0)continue;const double c=v+threat[q]+1e-3;if(c<hazardHome[q]){hazardHome[q]=c;pq.push({c,q});}}}
    }
    static double poissonBelow(double mu,int lives){double term=exp(-mu),sum=0;for(int j=0;j<lives;j++){sum+=term;term*=mu/(j+1);}return min(1.,sum);}
    double returnProbability(int q,int lives)const{return q<0||q>=int(hazardHome.size())||hazardHome[q]>=1e8?1.:poissonBelow(hazardHome[q],max(1,lives));}
    // Arrival race against the nearest rival (Astra goal-selection table).
    static double secureProbability(int d,int rival){return d<rival?1.:d==rival?.73:d==rival+1?.55:d<=rival+3?.28:.07;}
    void forecast(const Game& g){
        foes.clear();threat.assign(nav.N,0);
        // Exact per-view cache: target support counts depend on victim/team, not actor.
        vector<array<int,15>> nearbyTeams(g.units.size());
        for(const auto& v:g.units)if(pos[v.id]>=0)for(const auto& w:g.units)
            if(pos[w.id]>=0&&man(pos[w.id],pos[v.id])<=4)nearbyTeams[v.id][w.team]++;
        for(auto& u:g.units)if(u.team!=g.me&&pos[u.id]>=0){
            int p=pos[u.id],goal=closestGoal(u,g),target=-1;double tv=-1e9;
            for(auto& v:g.units)if(v.team!=u.team&&pos[v.id]>=0&&!immune[v.id]){
                int d=nav.dist(p,pos[v.id]);if(d>7)continue;
                double val=8-d+(v.flag>=0?6:0)+(100-v.hp)*.025;
                int friends=nearbyTeams[v.id][u.team],hostiles=nearbyTeams[v.id][v.team];
                val+=.5*(friends-hostiles);if(val>tv){tv=val;target=v.id;}
            }
            ctfcombat::Enemy e;e.u={u.id,u.team,{p%nav.n,p/nav.n},immune[u.id]?100:u.hp,u.flag,immune[u.id]};
            double attention=1.;if(target>=0&&g.units[target].team!=g.me)attention=W_other_attention;
            e.attention=attention;e.killValue=11+(u.flag>=0?16:0);
            vector<double> vals;
            for(int m=0;m<5;m++)if(nav.nb[p][m]>=0){
                int q=nav.nb[p][m];double s=0;
                if(u.flag>=0){
                    s=2.2*(nav.home[u.team][p]-nav.home[u.team][q]);double risk=0;
                    for(auto& v:g.units)if(v.team!=u.team&&v.flag<0&&pos[v.id]>=0&&!immune[v.id]){int d=man(q,pos[v.id]);risk+=d<=2?1.3:d==3?.7:d==4?.15:0;}
                    s-=risk*(u.hp<=34?2.3:u.hp<=68?1.8:1.2);
                }else {
                    if(goal>=0)s+=1.25*(nav.dist(p,goal)-nav.dist(q,goal));
                    if(target>=0){int tp=pos[target],d=man(q,tp);double fight=(tv>8?1.2:.5);s+=fight*(shot(q,tp,g)?1.3:d<=3?.5:0);
                        if(u.hp<=34&&g.units[target].flag<0)s-=.45*(d<=2?1.:0.);
                    }
                    for(auto& f:g.flags)if(available(f,g)&&q==nav.cell(f.x,f.y))s+=2.5;
                }
                if(m==0)s-=.35;
                if(prev.size()==g.units.size()&&prev[u.id].x>=0){int pp=nav.cell(prev[u.id].x,prev[u.id].y);if(pp!=p&&p-pp==q-p)s+=.35;}
                for(auto& v:g.units)if(v.id!=u.id&&pos[v.id]==q)s-=.65;
                vals.push_back(s);e.forecast.push_back({{q%nav.n,q/nav.n},0.,u.flag<0&&!immune[u.id]});
            }
            double mx=*max_element(vals.begin(),vals.end()),sum=0;
            for(size_t i=0;i<vals.size();i++){double v=exp((vals[i]-mx)/.75);e.forecast[i].prob=v;sum+=v;}
            for(auto& o:e.forecast)o.prob=.88*o.prob/sum+.12/e.forecast.size();
            // Movement marginal and conditional firing are kept separate: an
            // enemy on a ground flag may still shoot before picking it up.
            if(u.flag<0&&!immune[u.id])for(auto& o:e.forecast){
                const int q=nav.cell(o.p.x,o.p.y);if(!groundFlagAt(q,g))continue;
                o.shootChance=1-groundPickupProbability(g,u,q);
            }
            foes.push_back(e);
            if(u.flag<0&&!immune[u.id])for(auto& o:e.forecast){int q=nav.cell(o.p.x,o.p.y);for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++)if(abs(dx)+abs(dy)<=2){int x=o.p.x+dx,y=o.p.y+dy;if(x<0||x>=nav.n||y<0||y>=nav.n)continue;int r=nav.cell(x,y);if(nav.open(r)&&shot(q,r,g))threat[r]+=o.prob*e.attention;}}
        }
        if(active(g).flags)buildHazardHome(g);
    }
    int intercept(const Unit& u,int from)const{
        int p=pos[u.id],h=nav.home[u.team][p],best=-1;double cost=1e9;if(h<=1)return p;
        for(int q:nav.floors){int dt=nav.dist(p,q);if(dt>min(h,14)||nav.home[u.team][q]+dt>h+1)continue;
            int d=nav.dist(from,q);if(d>dt+2)continue;double v=max(dt,d-2)+.18*d+.3*nav.home[u.team][q];if(v<cost){cost=v;best=q;}}
        return best;
    }
    vector<Task> tasks(const Game& g){
        vector<Task> ts;
        for(auto& f:g.flags)if(available(f,g)){
            int q=nav.cell(f.x,f.y),ct=0;vector<int> tdist(g.teams,999);
            for(auto& u:g.units)if(u.team!=g.me&&u.flag<0){int d=flagEta(u,q,g);tdist[u.team]=min(tdist[u.team],d);if(d<=6)ct++;}
            int nt=0;for(int d:tdist)if(d<=6)nt++;double cont=min(3.,ct*.75)*(nt>1?.7:1.);
            ts.push_back({1,f.id,q,W_ground-(g.teams>=6?W_home_large:W_home_small)*nav.home[g.me][q],cont});
        }
        for(auto& u:g.units)if(pos[u.id]>=0&&u.flag>=0){
            if(u.team==g.me)ts.push_back({3,u.id,pos[u.id],W_escort+min(7.,threat[pos[u.id]]*5),0});
            else ts.push_back({2,u.id,pos[u.id],(g.teams<=3?W_deny_small:W_deny_large)-W_deny_remote*max(0,nav.home[g.me][pos[u.id]]-7)+(abs(g.score[u.team]-g.score[g.me])<=1?3:0),0});
        }
        for(int j=0;j<(int)g.spots.size();j++){
            int q=nav.cell(g.spots[j].first,g.spots[j].second);bool occupied=false;
            for(auto& f:g.flags)if(available(f,g)&&q==nav.cell(f.x,f.y))occupied=true;if(occupied)continue;
            int home=nav.home[g.me][q];double crowd=0;
            for(auto& u:g.units)if(u.team!=g.me&&pos[u.id]>=0){int d=nav.dist(pos[u.id],q);if(d<=4)crowd+=.5*(5-d);}
            double reward=W_camp-W_camp_home*home-min(6.,crowd*.55);if(j==g.me)reward+=1.2;
            ts.push_back({4,j,q,reward,0});
        }
        if(ts.empty())ts.push_back({4,g.me,nav.cell(g.spots[g.me].first,g.spots[g.me].second),8,0});return ts;
    }
    double taskValue(int k,const Task& t,const Game& g,int& target){
        const Unit& u=g.units[3*g.me+k];int p=pos[u.id];target=t.p;if(p<0)return -1000;
        if(u.flag>=0){target=-2;return t.type==0?1000:-10000;}
        if(t.type==2){auto& v=g.units[t.id];target=intercept(v,p);if(target<0)return -1000;return t.reward-.9*nav.dist(p,target)-.55*max(0,nav.home[v.team][pos[v.id]]-10);}
        if(t.type==3){int cp=pos[t.id];double best=1e9;int bq=cp;
            for(int m=0;m<5;m++)if(nav.nb[cp][m]>=0){int q=nav.nb[cp][m];double cv=nav.home[g.me][q]+.4*nav.dist(p,q);if(cv<best){best=cv;bq=q;}}
            target=bq;return t.reward-.65*max(0,nav.dist(p,bq)-2);
        }
        int d=nav.dist(p,t.p);
        if(t.type==1){auto& f=g.flags[t.id];if(f.status=="dropped"&&d>f.return_at-g.turn)return -1000;
            int rival=999;for(auto& v:g.units)if(v.team!=g.me&&v.flag<0)rival=min(rival,flagEta(v,t.p,g));
            // Securing x return x feasibility, replacing the linear lateness and
            // hopeless-contest penalties; a flag that cannot score is not a task.
            if(active(g).flags){if(g.turn+max(0,d-1)+nav.home[g.me][t.p]>g.turns)return -1000;
                const int lives=((immune[u.id]?100:u.hp)+33)/34;
                return t.reward*secureProbability(d,rival)*returnProbability(t.p,lives)-W_approach*d;}
            double late=max(0,d-rival-1);double val=t.reward-W_approach*d-W_late*late;
            if(rival<=1&&d>4&&t.contest<1.5)val-=10;
            if(g.turn+max(0,d-1)+nav.home[g.me][t.p]>g.turns)val-=30;return val;
        }
        // Posts at flag spots: same feasibility and return-risk weighting.
        if(active(g).flags&&t.type==4){if(g.turn+max(0,d-1)+nav.home[g.me][t.p]>g.turns)return -1000;
            const double back=returnProbability(t.p,((immune[u.id]?100:u.hp)+33)/34);
            return t.reward*back-W_camp_approach*d+(hybridMacro(g)?back*timedCampValue(p,t.p,g):0.);}
        if(!hybridMacro(g))return t.reward-W_camp_approach*d;
        return t.reward-W_camp_approach*d+timedCampValue(p,t.p,g);
    }
    bool hybridMacro(const Game& g)const{
        return active(g).hybridMacro;
    }
    array<Task,3> assign(const Game& g){
        const bool hybrid=hybridMacro(g);
        vector<Task> ts=tasks(g);array<vector<pair<double,int>>,3> ranked;
        vector<array<int,3>> targets(ts.size());vector<array<double,3>> vals(ts.size()),timedVals(ts.size());
        for(int i=0;i<(int)ts.size();i++)for(int k=0;k<3;k++){
            int q;double v=taskValue(k,ts[i],g,q);targets[i][k]=q;if(ts[i].type==oldType[k]&&ts[i].id==oldId[k])v+=.9;
            vals[i][k]=v;ranked[k].push_back({v,i});
            timedVals[i][k]=hybrid&&ts[i].type==4?timedCampValue(pos[3*g.me+k],ts[i].p,g):0;
        }
        for(int k=0;k<3;k++){
            sort(ranked[k].rbegin(),ranked[k].rend());
            if(!hybrid){if(ranked[k].size()>8)ranked[k].resize(8);continue;}
            pair<double,int> camp{-1e20,-1};
            for(auto entry:ranked[k])if(ts[entry.second].type==4){camp=entry;break;}
            if(ranked[k].size()>8)ranked[k].resize(8);
            if(camp.second>=0&&none_of(ranked[k].begin(),ranked[k].end(),[&](auto entry){return entry.second==camp.second;}))ranked[k].back()=camp;
        }
        double best=-1e20;array<int,3> ids{};
        for(auto a:ranked[0])for(auto b:ranked[1])for(auto c:ranked[2]){
            array<int,3> id{{a.second,b.second,c.second}};double value=a.first+b.first+c.first;
            for(int i=0;i<3;i++)if(pos[3*g.me+i]<0)value-=vals[id[i]][i];
            for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)if(pos[3*g.me+i]>=0&&pos[3*g.me+j]>=0){
                if(id[i]==id[j]){Task& t=ts[id[i]];if(t.type==1&&!hybrid)value-=W_flag_duplicate-W_contest_discount*t.contest;
                    if(t.type==4)value-=W_camp_duplicate;if(t.type==2)value-=3;if(t.type==3)value-=3;}
                else {int d=nav.dist(targets[id[i]][i],targets[id[j]][j]);if(d<=4)value+=.5*(5-d);}
            }
            if(hybrid)for(int i=0;i<3;i++)if(pos[3*g.me+i]>=0&&g.units[3*g.me+i].flag<0){
                bool first=true;for(int j=0;j<i;j++)if(pos[3*g.me+j]>=0&&g.units[3*g.me+j].flag<0&&id[j]==id[i])first=false;
                if(!first)continue;
                const Task& task=ts[id[i]];int count=0;double timedSum=0,timedBest=0;
                for(int j=i;j<3;j++)if(pos[3*g.me+j]>=0&&g.units[3*g.me+j].flag<0&&id[j]==id[i]){
                    ++count;
                    const double timed=timedVals[id[i]][j];
                    timedSum+=timed;timedBest=max(timedBest,timed);
                }
                if(task.type==1&&count>1){
                    const double support=task.contest>0?min(task.reward,8.0+7.0*task.contest):0;
                    value-=(count-1)*(task.reward-support);
                }
                if(task.type==4&&count>1&&timedSum>0){
                    value-=(timedSum-timedBest);
                }
            }
            if(value>best){best=value;ids=id;}
        }
        array<Task,3> out;
        for(int k=0;k<3;k++){out[k]=ts[ids[k]];out[k].p=targets[ids[k]][k];oldType[k]=out[k].type;oldId[k]=out[k].id;
            if(g.units[3*g.me+k].flag>=0){out[k]={0,g.units[3*g.me+k].flag,-2,0,0};oldType[k]=0;}}
        return out;
    }
    vector<double> homeRoute(const Game& g,int uid){
        vector<double> d(nav.N,1e9);priority_queue<pair<double,int>,vector<pair<double,int>>,greater<pair<double,int>>> pq;
        for(int p:nav.spawn[g.me]){d[p]=0;pq.push({0,p});}const Unit& u=g.units[uid];double hp=immune[uid]?100:u.hp;
        while(!pq.empty()){
            auto [v,p]=pq.top();pq.pop();if(v>d[p]+1e-6)continue;
            double risk=threat[p],shield=0;for(int k=0;k<3;k++){int id=3*g.me+k;if(id!=uid&&pos[id]>=0&&g.units[id].flag<0&&man(pos[id],p)<=3)shield+=.3;}
            risk=max(0.,risk-shield);
            double cost=1+min(12.,risk*(hp<=34?7.:hp<=68?5.:3.));
            for(int m=1;m<5;m++){int q=nav.nb[p][m];if(q>=0&&v+cost<d[q]){d[q]=v+cost;pq.push({d[q],q});}}
        }return d;
    }
    chrono::steady_clock::time_point decisionDeadline;
    // Public-state-only deterministic turn kernel. Unknown random returns are
    // not guessed: return false and retain the unrefined policy instead.
    bool stepExact(const Game& g,const vector<int>& requested,const vector<int>& actions,Game& next){
        for(const auto& f:g.flags)if((f.status=="cooldown"||f.status=="dropped")&&f.return_at<=g.turn)return false;
        respawns(g);next=g;next.turn=g.turn+1;next.events.clear();
        const int U=(int)g.units.size();vector<int> start=pos,want(U,-1),owner(nav.N,-1),count(nav.N,0),damage(U,0);vector<bool> blocked(U,false);
        for(int i=0;i<U;i++)if(start[i]>=0){
            owner[start[i]]=i;int q=i<(int)requested.size()?requested[i]:start[i];
            if(!nav.open(q)||man(q,start[i])>1)q=start[i];want[i]=q;count[q]++;
            auto& u=next.units[i];u.x=start[i]%nav.n;u.y=start[i]/nav.n;u.hp=immune[i]?g.hp:u.hp;u.respawn_at=-1;
        }
        for(int i=0;i<U;i++)if(start[i]>=0){
            if(count[want[i]]>1)blocked[i]=true;int j=owner[want[i]];
            if(j>=0&&j!=i&&g.units[i].team!=g.units[j].team&&want[j]==start[i])blocked[i]=blocked[j]=true;
        }
        for(bool changed=true;changed;){changed=false;for(int i=0;i<U;i++)if(start[i]>=0&&!blocked[i]&&want[i]!=start[i]){
            int j=owner[want[i]];if(j>=0&&(blocked[j]||want[j]==start[j])){blocked[i]=true;changed=true;}}}
        for(int i=0;i<U;i++)if(start[i]>=0){if(blocked[i])want[i]=start[i];next.units[i].x=want[i]%nav.n;next.units[i].y=want[i]/nav.n;}
        auto action=[&](int i){return i<(int)actions.size()?actions[i]:-1;};
        // All attack orders use pre-damage eligibility and contribute together.
        for(int i=0;i<U;i++)if(start[i]>=0&&next.units[i].flag<0&&!immune[i]){
            int t=action(i);if(t>=0&&t<U&&start[t]>=0&&!immune[t]&&g.units[t].team!=g.units[i].team&&shot(want[i],want[t],g)){damage[t]+=g.damage;next.events.push_back({"attack",{i,t}});}
        }
        for(int i=0;i<U;i++)if(start[i]>=0){auto& u=next.units[i];u.hp-=damage[i];if(u.hp<=0){
            if(u.flag>=0){auto& f=next.flags[u.flag];f.status="dropped";f.holder=-1;f.x=u.x;f.y=u.y;f.return_at=g.turn+g.flag_return;}
            u.x=u.y=-1;u.flag=-1;u.hp=0;u.respawn_at=g.turn+g.respawn;
        }}
        // Pickups precede voluntary drops: a teammate cannot pick up a newly
        // dropped flag in this same turn. Dead carriers cannot execute X.
        for(auto& u:next.units)if(u.x>=0&&u.flag<0&&action(u.id)==-2){
            for(auto& f:next.flags)if((f.status=="home"||f.status=="dropped")&&f.x==u.x&&f.y==u.y){
                u.flag=f.id;f.status="carried";f.x=f.y=-1;f.holder=u.id;f.return_at=-1;break;}
        }
        for(auto& u:next.units)if(u.x>=0&&u.flag>=0&&action(u.id)==-3){auto& f=next.flags[u.flag];
            f.status="dropped";f.holder=-1;f.x=u.x;f.y=u.y;f.return_at=g.turn+g.flag_return;u.flag=-1;}
        for(auto& u:next.units)if(u.x>=0&&u.flag>=0&&nav.home[u.team][nav.cell(u.x,u.y)]==0){
            auto& f=next.flags[u.flag];f.status="cooldown";f.holder=-1;f.x=f.y=-1;f.return_at=g.turn+g.flag_cooldown;
            next.events.push_back({"capture",{u.id,u.team,u.flag}});
            u.flag=-1;next.score[u.team]++;
        }
        return true;
    }
    struct Runtime {vector<Unit> previous;array<int,3> type,id;vector<int> positions;vector<bool> immunity;vector<double> hazards;array<vector<double>,3> routes;vector<ctfcombat::Enemy> enemies;};
    Runtime saveRuntime()const{return {prev,oldType,oldId,pos,immune,threat,route,foes};}
    void restoreRuntime(const Runtime& s){prev=s.previous;oldType=s.type;oldId=s.id;pos=s.positions;immune=s.immunity;threat=s.hazards;route=s.routes;foes=s.enemies;}
    // Checked lookahead (R3_CL_checked lineage). Structured decisions are
    // simulated directly; future plans never serialize and reparse a command.
    enum class ActionKind {Idle,Pickup,Drop,Attack};
    struct Action {ActionKind kind=ActionKind::Idle;int target=-1;
        static Action fromTarget(int t){return t>=0?Action{ActionKind::Attack,t}:t==-2?Action{ActionKind::Pickup,-1}:t==-3?Action{ActionKind::Drop,-1}:Action{};}
        int code()const{return kind==ActionKind::Attack?target:kind==ActionKind::Pickup?-2:kind==ActionKind::Drop?-3:-1;}
        string text()const{return kind==ActionKind::Attack?to_string(target):kind==ActionKind::Pickup?"P":kind==ActionKind::Drop?"X":"-";}
    };
    struct Decision {array<int,3>wanted{{-1,-1,-1}};array<Action,3>actions{};array<char,3>move{{'S','S','S'}};};
    void rivalOrders(const Game& g,int model,vector<int>& want,vector<int>& acts){
        want=pos;acts.assign(g.units.size(),-1);
        for(const auto& e:foes){int oi=0;if(model==0){for(int j=1;j<(int)e.forecast.size();j++)if(e.forecast[j].prob>e.forecast[oi].prob)oi=j;}
            else{unsigned h=unsigned(g.turn*7919+e.u.id*104729+65537);h^=h>>13;h*=0x85ebca6bU;double r=(h%65536+.5)/65536.;for(int j=0;j<(int)e.forecast.size();j++){oi=j;r-=e.forecast[j].prob;if(r<=0)break;}}
            const auto& out=e.forecast[oi];want[e.u.id]=nav.cell(out.p.x,out.p.y);bool ground=false;
            for(const auto& f:g.flags)if(available(f,g)&&want[e.u.id]==nav.cell(f.x,f.y))ground=true;
            // Select movement from its marginal, then the conditional action.
            // This avoids a pickup/fire split diluting the destination mode.
            bool fire=out.canShoot;
            if(ground&&fire){
                unsigned h=unsigned(g.turn*3571+e.u.id*100003+model*65537);h^=h>>16;h*=0x85ebca6bU;h^=h>>13;
                const double r=(h%65536+.5)/65536.;
                fire=model==0?out.shootChance>=.5:r<out.shootChance;
            }
            if(e.u.flag<0&&ground&&!fire)acts[e.u.id]=-2;
            else if(e.u.flag<0&&!immune[e.u.id]&&fire)acts[e.u.id]=ground?-5:-4;
        }
    }
    void rivalAttacks(const Game& g,const vector<int>& want,vector<int>& acts){
        vector<vector<int>> allocated(g.teams,vector<int>(g.units.size(),0));
        for(const auto& e:foes){int i=e.u.id;if(acts[i]!=-4&&acts[i]!=-5)continue;const bool ground=acts[i]==-5;double best=0;int target=-1;
            for(const auto& v:g.units)if(v.team!=g.units[i].team&&pos[v.id]>=0&&!immune[v.id]&&shot(want[i],want[v.id],g)){
                double priority=v.flag>=0?5.:1.;if(v.hp<=g.damage)priority*=2.4;else if(v.hp<=2*g.damage)priority*=1.35;
                int need=(v.hp+g.damage-1)/g.damage;if(allocated[g.units[i].team][v.id]>=need)priority*=.15;else if(allocated[g.units[i].team][v.id]+1>=need)priority+=3;
                if(priority>best){best=priority;target=v.id;}}
            // A firing intent on a ground flag without a reachable target picks up instead.
            acts[i]=target>=0?target:ground?-2:-1;if(target>=0)allocated[g.units[i].team][target]++;
        }
    }
    unsigned ownedFlags(const Game& g)const{unsigned mask=0;for(const auto&u:g.units)if(u.team==g.me&&u.x>=0&&u.flag>=0)mask|=1u<<u.flag;return mask;}
    unsigned deliveredFlags(const Game& g)const{unsigned mask=0;for(const auto&e:g.events)if(e.type=="capture"&&e.args.size()==3&&e.args[1]==g.me)mask|=1u<<e.args[2];return mask;}
    // A root- or middle-owned flag lost at the endpoint costs once, whether the
    // holder died carrying it or dropped it first. Physical deaths cost separately.
    double ownershipLoss(const Game& root,const Game& middle,const Game& end)const{
        unsigned delivered=deliveredFlags(middle)|deliveredFlags(end),owned=ownedFlags(root)|ownedFlags(middle);
        unsigned lost=owned&~ownedFlags(end)&~delivered;int count=0;while(lost){count+=lost&1;lost>>=1;}
        return W_carrier_death*count;
    }
    double terminalFlags(const Game& g)const{double value=0;for(const auto& f:g.flags)if(f.status=="carried"&&f.holder>=0){const auto&u=g.units[f.holder];if(u.x<0)continue;int q=nav.cell(u.x,u.y);
            double v=30./(1+.10*nav.home[u.team][q]);
            value+=u.team==g.me?v:-.5*v;}
        return value;
    }
    double realized(const Game& before,const Game& after)const{
        double value=70.*(after.score[before.me]-before.score[before.me]);for(int t=0;t<before.teams;t++)if(t!=before.me)value-=35.*(after.score[t]-before.score[t]);
        vector<int> ownHits(before.units.size(),0),allHits(before.units.size(),0);
        for(const auto&e:after.events)if(e.type=="attack"&&e.args.size()==2){int a=e.args[0],t=e.args[1];allHits[t]++;if(before.units[a].team==before.me)ownHits[t]++;}
        for(const auto&u:before.units)if(u.x>=0){const auto&v=after.units[u.id];bool dead=v.x<0;double hit=(u.hp-max(0,v.hp))/double(before.damage);
            if(u.team==before.me)value-=W_hit*hit+W_death*dead;
            else if(ownHits[u.id]){double share=ownHits[u.id]/double(max(1,allHits[u.id]));value+=share*(W_damage*(u.hp-max(0,v.hp))+(dead?(11+(u.flag>=0?34:0)):0));}}
        return value;
    }
    ctfcombat::Choice refineChecked(const Game& g,const array<vector<ctfcombat::Move>,3>& options,vector<ctfcombat::Choice> beam,ctfcombat::Choice fallback){
        if(g.turn>=g.turns)return fallback;int nearby=0;for(const auto&e:foes){bool close=false;for(int k=0;k<3;k++)if(pos[3*g.me+k]>=0&&man(pos[3*g.me+k],nav.cell(e.u.p.x,e.u.p.y))<=5)close=true;nearby+=close;}
        if(nearby==0||nearby>9)return fallback;bool flagBattle=ownedFlags(g)!=0;
        for(const auto&f:g.flags)if(available(f,g)){int q=nav.cell(f.x,f.y);bool ownClose=false,rivalClose=false;for(int k=0;k<3;k++)if(pos[3*g.me+k]>=0&&man(pos[3*g.me+k],q)<=4)ownClose=true;for(const auto&e:foes)if(e.u.flag<0&&man(nav.cell(e.u.p.x,e.u.p.y),q)<=4)rivalClose=true;flagBattle|=ownClose&&rivalClose;}
        if(!flagBattle)return fallback;
        // The engine returns to a random free spot before movement. A bot does
        // not know the draw, so do not silently delete or predict that flag.
        for(const auto&f:g.flags)if((f.status=="cooldown"||f.status=="dropped")&&f.return_at<=g.turn+1)return fallback;
        sort(beam.begin(),beam.end(),[](const auto&a,const auto&b){return a.value>b.value;});int width=g.teams<=5?4:2;if(beam.size()>size_t(width))beam.resize(width);
        vector<ctfcombat::Choice> candidates{fallback};for(auto c:beam)if(c.value>=fallback.value-12&&c.moveIndex!=fallback.moveIndex)candidates.push_back(c);
        auto saved=saveRuntime();auto outer=decisionDeadline;decisionDeadline=min(outer,chrono::steady_clock::now()+chrono::milliseconds(20));
        ctfcombat::Choice answer=fallback;double best=-1e100;
        for(const auto&candidate:candidates){double mean=0,worst=1e100;bool complete=true;
            for(int model=0;model<2;model++){
                if(chrono::steady_clock::now()>=decisionDeadline){complete=false;break;}
                restoreRuntime(saved);vector<int>want,acts;rivalOrders(g,model,want,acts);
                for(int k=0;k<3;k++){int i=3*g.me+k;auto p=options[k][candidate.moveIndex[k]].p;if(pos[i]>=0)want[i]=nav.cell(p.x,p.y);acts[i]=candidate.target[k];}
                rivalAttacks(g,want,acts);Game next;if(!stepExact(g,want,acts,next)){complete=false;break;}
                prev=g.units;auto follow=planTurn(next,false);
                if(chrono::steady_clock::now()>=decisionDeadline){complete=false;break;}
                vector<int>want2,acts2;rivalOrders(next,model,want2,acts2);for(int k=0;k<3;k++){int i=3*g.me+k;if(pos[i]>=0)want2[i]=follow.wanted[k];acts2[i]=follow.actions[k].code();}
                rivalAttacks(next,want2,acts2);Game end;if(!stepExact(next,want2,acts2,end)){complete=false;break;}
                if(chrono::steady_clock::now()>=decisionDeadline){complete=false;break;}
                double outcome=realized(g,next)+.75*realized(next,end)-ownershipLoss(g,next,end)+terminalFlags(end)-terminalFlags(g);
                mean+=.5*outcome;worst=min(worst,outcome);
            }
            if(!complete)break;
            // Replace immediate tactical/capture prediction by branch outcomes;
            // keep the gold movement/task score once. No repeated capture bonus.
            double value=candidate.positionValue+.8*mean+.2*worst;
            if(value>best+1e-8){best=value;answer=candidate;
            }
        }
        restoreRuntime(saved);decisionDeadline=outer;return answer;
    }

    Decision planTurn(Game& g,bool refineAllowed){
        if(refineAllowed)decisionDeadline=chrono::steady_clock::now()+chrono::milliseconds(g.turn==1?1800:35);
        if(!initialized){nav.init(g);initialized=true;}respawns(g);forecast(g);auto jobs=assign(g);
        array<ctfcombat::Fighter,3> own;array<vector<ctfcombat::Move>,3> options;
        for(int k=0;k<3;k++){
            const Unit& u=g.units[3*g.me+k];int p=pos[u.id],hp=immune[u.id]?100:u.hp;
            own[k]={u.id,u.team,{p<0?-1:p%nav.n,p<0?-1:p/nav.n},hp,u.flag,immune[u.id]};
            if(p<0){options[k].push_back({{-1,-1},'S',0,0});continue;}if(u.flag>=0)route[k]=homeRoute(g,u.id);
            for(int m=0;m<5;m++)if(nav.nb[p][m]>=0){int q=nav.nb[p][m];double score=0,pickup=0,capture=0;
                if(u.flag>=0){score=-W_route*route[k][q];if(nav.home[g.me][q]==0)capture=70;}
                else {
                    if(jobs[k].type==3){int d=nav.dist(q,jobs[k].p);score=-.85*max(0,d-2);}
                    else if(jobs[k].p>=0)score=-1.05*nav.dist(q,jobs[k].p);
                    for(auto& f:g.flags)if(available(f,g)&&q==nav.cell(f.x,f.y))pickup=W_pickup-.2*nav.home[g.me][q];
                    bool returnNow=false;for(auto& f:g.flags)if((f.status=="cooldown"||f.status=="dropped")&&f.return_at<=g.turn)returnNow=true;
                    if(returnNow)for(auto& s:g.spots)if(q==nav.cell(s.first,s.second))pickup=max(pickup,4./max(1,(int)g.spots.size()-(int)g.flags.size()));
                    if(jobs[k].type==4&&nav.dist(q,jobs[k].p)<=1)score+=.08;
                }
                if(m==0&&q!=jobs[k].p&&u.flag<0)score-=.05;
                if(prev.size()==g.units.size()&&prev[u.id].x>=0){int pp=nav.cell(prev[u.id].x,prev[u.id].y);if(q==pp&&p!=pp)score-=.12;}
                options[k].push_back({{q%nav.n,q/nav.n},letters[m],score,pickup,capture});
            }
        }
        ctfcombat::Params params;params.ownDeathCost=W_death;params.incomingHitCost=W_hit;params.dealtHpValue=W_damage;params.carrierDeathExtra=W_carrier_death;params.carrierHitExtra=W_carrier_hit;params.carrierWorstDeathCost=W_worst;
        vector<ctfcombat::Choice> beam;const bool refine=CTF_LOOKAHEAD&&refineAllowed;
        auto choice=ctfcombat::choose(g.map,own,options,foes,params,refine?&beam:nullptr,&decisionDeadline);
        if(refine)choice=refineChecked(g,options,beam,choice);
        Decision result;
        for(int k=0;k<3;k++){int m=choice.moveIndex[k],t=choice.target[k];if(m<0||m>=(int)options[k].size())m=0;
            result.move[k]=options[k][m].code;auto p=options[k][m].p;result.wanted[k]=p.x<0?-1:nav.cell(p.x,p.y);result.actions[k]=Action::fromTarget(t);}
        return result;
    }
    string decide(Game& g){
        auto result=planTurn(g,true);
        string out=to_string(g.turn);for(int k=0;k<3;k++){out+=' ';out+=result.move[k];out+=' ';out+=result.actions[k].text();}prev=g.units;return out;}
};
}  // namespace multi


namespace duel {
struct Mission {
    int type, key, p, unit = -1;
    double danger = 0;
};
struct Prediction {
    int id;
    array<int,5> p;
    array<double,5> pr;
    array<double,5> other;
};
struct Choice { int target; double v; };

class Player {
    int n=0, cells=0, me=0;
    vector<array<int,5>> nb;
    vector<vector<uint16_t>> dist;
    vector<vector<int>> bd;
    vector<int> oldp, lasttarget, oldhp, occupied;
    array<int,3> lastmission{{-100,-100,-100}};
    vector<Unit> u;
    vector<int> pos;
    vector<bool> immune;
    vector<Prediction> pred;
    vector<double> otherFocus;
    vector<Mission> missions;
    vector<array<int,3>> goal;
    array<int,3> assigned, own;
    array<double,3> rolew;
    array<vector<double>,3> route;
    array<array<double,5>,3> block;
    NgActive act;  // module activation for 2-team games
    // Probability of being able to shoot a particular enemy from each move.
    array<array<vector<array<double,5>>,5>,3> hit;
    const array<char,5> mc{{'S','U','D','L','R'}};
    int D(int a,int b) const { return a<0||b<0 ? 999 : dist[a][b]; }
    int man(int a,int b) const { return abs(a%n-b%n)+abs(a/n-b/n); }
    bool shot(int a,int b) const {
        if(a<0||b<0) return false;
        int dx=b%n-a%n,dy=b/n-a/n;
        if(abs(dx)+abs(dy)>2) return false;
        if(abs(dx)==2) return nb[a][dx>0?4:3]>=0;
        if(abs(dy)==2) return nb[a][dy>0?2:1]>=0;
        if(abs(dx)==1 && abs(dy)==1) return nb[a][dx>0?4:3]>=0 || nb[a][dy>0?2:1]>=0;
        return true;
    }
    int cell(int x,int y) const { return x<0 ? -1 : y*n+x; }
    double tw(int id) const {
        return (u[id].flag>=0?5.0:1.0)*(u[id].hp<=34?3.1:u[id].hp<=68?1.65:1.0);
    }
    void init(const Game& g) {
        n=g.size;cells=n*n;me=g.me;
        nb.resize(cells);dist.resize(cells);
        for(int p=0;p<cells;p++) {
            nb[p].fill(-1);
            if(g.map[p/n][p%n]=='#') continue;
            nb[p][0]=p;
            const int dx[5]={0,0,0,-1,1},dy[5]={0,-1,1,0,0};
            for(int m=1;m<5;m++) {
                int x=p%n+dx[m],y=p/n+dy[m];
                if(x>=0&&y>=0&&x<n&&y<n&&g.map[y][x]!='#') nb[p][m]=y*n+x;
            }
        }
        vector<int> q(cells);
        for(int p=0;p<cells;p++) {
            dist[p].assign(cells,999);
            if(nb[p][0]<0) continue;
            int head=0,tail=0;q[tail++]=p;dist[p][p]=0;
            while(head<tail) {
                int a=q[head++];
                for(int m=1;m<5;m++) { int b=nb[a][m];
                    if(b>=0&&dist[p][b]==999) {dist[p][b]=dist[p][a]+1;q[tail++]=b;}
                }
            }
        }
        bd.assign(g.teams,vector<int>(cells,999));
        for(int t=0;t<g.teams;t++) for(int p=0;p<cells;p++)
            for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
                bd[t][p]=min(bd[t][p],D(p,cell(g.bases[t].first+dx,g.bases[t].second+dy)));
        oldp.assign(g.units.size(),-1);lasttarget.assign(g.units.size(),-1);oldhp.assign(g.units.size(),100);
    }
    void update(const Game& g) {
        u=g.units;pos.assign(u.size(),-1);immune.assign(u.size(),false);
        occupied.assign(cells,-1);
        for(auto& a:u) {pos[a.id]=cell(a.x,a.y);if(a.x>=0)occupied[pos[a.id]]=a.id;}
        for(auto& a:u) if(pos[a.id]<0&&a.respawn_at<=g.turn&&a.respawn_at>=0) {
            vector<int> spawn;
            for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) spawn.push_back(cell(g.bases[a.team].first+dx,g.bases[a.team].second+dy));
            int c=cell(n/2,n/2),bc=cell(g.bases[a.team].first,g.bases[a.team].second);
            stable_sort(spawn.begin(),spawn.end(),[&](int x,int y){
                return make_pair(man(x,c),abs(x%n-bc%n)+2*abs(x/n-bc/n))<make_pair(man(y,c),abs(y%n-bc%n)+2*abs(y/n-bc/n));
            });
            for(int p:spawn) if(occupied[p]<0){pos[a.id]=p;occupied[p]=a.id;a.hp=100;a.flag=-1;immune[a.id]=true;break;}
        }
        for(auto& e:g.events) if(e.type=="attack"&&e.args.size()>=2) lasttarget[e.args[0]]=e.args[1];
        for(int k=0;k<3;k++)own[k]=3*me+k;
    }
    vector<double> safeRoute(int uid,const Game& g) {
        vector<double> d(cells,1e9),cost(cells,1);
        for(int p=0;p<cells;p++) if(nb[p][0]>=0) {
            double risk=0;
            for(auto& e:u) if(e.team!=me&&pos[e.id]>=0&&e.flag<0&&!immune[e.id]) {
                int r=D(p,pos[e.id]);
                if(r>7)continue;
                double escort=0;
                for(int id:own) if(id!=uid&&pos[id]>=0&&u[id].flag<0) {
                    int dd=D(pos[id],pos[e.id]);
                    if(dd<=4)escort+= (u[id].hp>34?0.6:0.3);
                }
                double v=r<=2?3.5:r==3?2.8:r==4?1.5:r==5?.6:.15;
                risk+=v*(.5+.5*e.hp/100.0)/(1+escort*1.6);
            }
            cost[p]=1+min(6.0,risk)*(u[uid].hp<=34?.85:u[uid].hp<=68?.6:.45);
        }
        priority_queue<pair<double,int>,vector<pair<double,int>>,greater<pair<double,int>>> q;
        for(int p=0;p<cells;p++) if(bd[me][p]==0){d[p]=0;q.push({0,p});}
        while(!q.empty()) {
            auto [v,p]=q.top();q.pop();if(v!=d[p])continue;
            for(int m=1;m<5;m++){int b=nb[p][m];if(b>=0&&d[b]>v+(cost[b]+cost[p])*.5){d[b]=v+(cost[b]+cost[p])*.5;q.push({d[b],b});}}
        }
        return d;
    }
    int intercept(int uid,int enemy) const {
        int a=pos[uid],b=pos[enemy],t=u[enemy].team;
        if(a<0||b<0)return b;
        int p=b;vector<int> path{p};
        while(bd[t][p]>0&&path.size()<80) {
            int best=-1;
            for(int m=1;m<5;m++)if(nb[p][m]>=0&&bd[t][nb[p][m]]<bd[t][p]){
                int z=nb[p][m];if(best<0||D(a,z)<D(a,best))best=z;
            }
            if(best<0)break;p=best;path.push_back(p);
        }
        int best=b;double val=1e9;
        for(int j=0;j<(int)path.size();j++) {
            double v=max(0,D(a,path[j])-2-j)*3.0+j*.35+D(a,path[j])*.1;
            if(v<val){val=v;best=path[j];}
        }
        return best;
    }
    void plan(const Game& g) {
        missions.clear();goal.clear();
        auto add=[&](Mission m){missions.push_back(m);array<int,3> a; a.fill(m.p);goal.push_back(a);};
        for(auto& f:g.flags) if((f.status=="home"||f.status=="dropped")&&f.x>=0&&(f.return_at<0||f.return_at>g.turn))
            add({0,f.id,cell(f.x,f.y)});
        for(auto& e:u) if(e.team!=me&&pos[e.id]>=0&&e.flag>=0&&bd[e.team][pos[e.id]]>0) {
            bool can=false;for(int id:own)if(pos[id]>=0&&u[id].flag<0&&D(pos[id],pos[e.id])<=bd[e.team][pos[e.id]]+3)can=true;
            if(can) {add({1,100+e.id,pos[e.id],e.id});for(int k=0;k<3;k++)goal.back()[k]=intercept(own[k],e.id);}
        }
        for(int id:own)if(pos[id]>=0&&u[id].flag>=0) {
            double threat=0;int bad=-1;double bv=0;
            for(auto& e:u)if(e.team!=me&&pos[e.id]>=0&&e.flag<0&&!immune[e.id]) {
                int d=D(pos[e.id],pos[id]);double v=d<=3?2.0:d<=5?1.2:d<=7?.5:0;
                threat+=v;if(v>bv){bv=v;bad=e.id;}
            }
            if(threat>.1&&bd[me][pos[id]]>1){add({2,200+id,pos[id],id,min(4.0,threat)});
                if(bad>=0)goal.back().fill(pos[bad]);
            }
        }
        vector<pair<int,int>> ss;
        for(int j=0;j<(int)g.spots.size();j++)ss.push_back({bd[me][cell(g.spots[j].first,g.spots[j].second)],j});
        sort(ss.begin(),ss.end());
        for(int j=0;j<min(3,(int)ss.size());j++){int s=ss[j].second;add({3,300+s,cell(g.spots[s].first,g.spots[s].second)});}
        if(missions.empty())add({3,300,cell(n/2,n/2)});
        int M=missions.size();
        array<vector<int>,3> options;
        for(int k=0;k<3;k++) {
            int id=own[k];assigned[k]=-1;rolew[k]=.85;route[k].clear();
            if(pos[id]<0||u[id].flag>=0){options[k]={-1};if(pos[id]>=0)route[k]=safeRoute(id,g);continue;}
            vector<pair<double,int>> order;
            for(int j=0;j<M;j++) {
                auto& m=missions[j];int d=D(pos[id],goal[j][k]);double val=0;
                if(m.type==0)val=240.0/(8+d+bd[me][m.p]);
                if(m.type==1)val=100.0/(6+d+bd[u[m.unit].team][pos[m.unit]]);
                if(m.type==2)val=12*m.danger/(3+d);
                if(m.type==3)val=2.8/(1+.15*d);
                if(lastmission[k]==m.key)val+=.3;
                order.push_back({val,j});
            }
            sort(order.rbegin(),order.rend());
            for(int j=0;j<min(7,M);j++)options[k].push_back(order[j].second);
        }
        double best=-1e30;
        for(int a:options[0])for(int b:options[1])for(int c:options[2]) {
            array<int,3> as{{a,b,c}};double v=0;
            for(int j=0;j<M;j++) {
                int cnt=0,mind=999;double extra=0,health=0;
                for(int k=0;k<3;k++)if(as[k]==j){cnt++;int d=D(pos[own[k]],goal[j][k]);mind=min(mind,d);extra+=d;health+=(u[own[k]].hp<=34?.65:u[own[k]].hp<=68?.85:1.0);}
                if(!cnt)continue;
                auto& m=missions[j];
                if(m.type==0) {
                    vector<int> nearest(g.teams,999),force(g.teams,0);int eMin=999;
                    for(auto& e:u)if(e.team!=me&&pos[e.id]>=0&&e.flag<0){nearest[e.team]=min(nearest[e.team],D(pos[e.id],m.p));eMin=min(eMin,D(pos[e.id],m.p));}
                    for(auto& e:u)if(e.team!=me&&pos[e.id]>=0&&e.flag<0&&D(pos[e.id],m.p)<=nearest[e.team]+3&&nearest[e.team]<=mind+3)force[e.team]++;
                    double ef=0,others=0;for(int t=0;t<g.teams;t++)if(t!=me&&force[t]){others+=force[t];ef=max(ef,(double)force[t]);}
                    ef+=.15*(others-ef);
                    double pr=.97;
                    if(eMin<=mind+3) {double lead=exp(max(-1.0,min(1.5,.22*(mind-eMin))));pr=.08+.90*health*health/(health*health+ef*ef*lead);}
                    double vv=240.0/(8+mind+bd[me][m.p])*pr;
                    if(mind+bd[me][m.p]+g.turn>g.turns)vv*=.08;
                    if(cnt>1&&eMin<=mind+3)vv+=.7*(cnt-1);
                    v+=vv-.045*(extra-cnt*mind);
                } else if(m.type==1) {
                    int e=m.unit;double pr=.25+.75*health*health/(health*health+2.0);
                    double leader=1+min(.6,.12*max(0,g.score[u[e].team]-g.score[me]));
                    v+=100*pr*leader/(6+mind+bd[u[e].team][pos[e]])-.06*(extra-cnt*mind);
                } else if(m.type==2) {
                    int d=D(pos[m.unit],m.p);
                    double escortWeight=g.teams>=6?1.8:1.0;
                    v+=escortWeight*min(14.0,6*m.danger)*(1-exp(-.8*cnt))/(1+.13*mind)-.03*extra;
                } else v+=2.5/(1+.13*mind)+.12*(cnt-1)-.04*(extra-cnt*mind);
            }
            for(int k=0;k<3;k++)if(as[k]>=0&&missions[as[k]].key==lastmission[k])v+=.12;
            if(v>best){best=v;assigned=as;}
        }
        for(int k=0;k<3;k++)if(assigned[k]>=0) {
            auto& m=missions[assigned[k]];lastmission[k]=m.key;
            if(m.type==0) {
                int runner=k;
                for(int h=0;h<3;h++)if(assigned[h]==assigned[k]&&D(pos[own[h]],m.p)+(100-u[own[h]].hp)*.025<D(pos[own[runner]],m.p)+(100-u[own[runner]].hp)*.025)runner=h;
                rolew[k]=runner==k?1.0:.72;
            } else if(m.type==1)rolew[k]=.85;
            else if(m.type==2)rolew[k]=.65;
            else rolew[k]=.6;
        }
    }
    void predict(const Game& g) {
        pred.clear();
        for(auto& e:u)if(e.team!=me&&pos[e.id]>=0) {
            Prediction pr;pr.id=e.id;pr.p=nb[pos[e.id]];pr.pr.fill(0);pr.other.fill(0);
            int target=-1,dd=999;
            if(e.flag<0) {
                for(auto& f:g.flags)if(f.x>=0&&(f.status=="home"||f.status=="dropped")&&(f.return_at<0||f.return_at>g.turn)){
                    int p=cell(f.x,f.y),d=D(pos[e.id],p);if(d<dd){dd=d;target=p;}
                }
                int victim=-1;double bv=-1e9;
                for(auto& z:u)if(z.team!=e.team&&pos[z.id]>=0&&!immune[z.id]) {
                    int d=D(pos[e.id],pos[z.id]);double v=(z.flag>=0?8.0:0)+(100-z.hp)*.035-d*1.5;
                    if(d<=5&&v>bv){bv=v;victim=z.id;}
                }
                if(victim>=0&&(u[victim].flag>=0||dd>3)) { target=pos[victim];dd=D(pos[e.id],target); }
                if(target<0) {
                    int d0=999;for(auto s:g.spots){int p=cell(s.first,s.second),d=D(pos[e.id],p)+bd[e.team][p]/3;if(d<d0){d0=d;target=p;}}
                }
            }
            double sum=0;
            for(int m=0;m<5;m++)if(pr.p[m]>=0) {
                int p=pr.p[m];double v=0;
                if(e.flag>=0) {
                    v=-2.4*(bd[e.team][p]-bd[e.team][pos[e.id]]);
                    for(int id:own)if(pos[id]>=0&&u[id].flag<0&&!immune[id]&&D(pos[id],p)<=3)v-=.3*(4-D(pos[id],p));
                } else {
                    if(target>=0)v=-1.8*(D(p,target)-D(pos[e.id],target));
                    if(target>=0&&D(p,target)<=2&&dd<=5)v+=.4;
                }
                if(oldp[e.id]>=0&&pos[e.id]!=oldp[e.id]&&p-pos[e.id]==pos[e.id]-oldp[e.id])v+=.35;
                if(m==0)v-=.15;
                if(occupied[p]>=0&&occupied[p]!=e.id) v-=.65;
                pr.pr[m]=exp(max(-5.0,min(5.0,v)));sum+=pr.pr[m];
            }
            int cnt=0;for(int p:pr.p)if(p>=0)cnt++;
            for(int m=0;m<5;m++)if(pr.p[m]>=0)pr.pr[m]=.90*pr.pr[m]/sum+.10/cnt;
            pred.push_back(pr);
        }
        for(auto& e:pred)for(int m=0;m<5;m++)if(e.p[m]>=0) {
            for(auto& z:pred)if(u[z.id].team!=u[e.id].team&&!immune[z.id]) {
                double p=0;for(int j=0;j<5;j++)if(z.p[j]>=0&&shot(e.p[m],z.p[j]))p+=z.pr[j];
                e.other[m]+=tw(z.id)*p;
            }
        }
        otherFocus.assign(g.teams,0);
        for(int t=0;t<g.teams;t++)if(t!=me) for(auto& victim:pred)if(u[victim.id].team!=t&&!immune[victim.id]) {
            double val=0;int need=(u[victim.id].hp+33)/34;
            for(int m=0;m<5;m++)if(victim.p[m]>=0) {
                array<double,4> dp{{1,0,0,0}};
                for(auto& a:pred)if(u[a.id].team==t&&u[a.id].flag<0&&!immune[a.id]) {
                    double p=0;for(int h=0;h<5;h++)if(a.p[h]>=0&&shot(a.p[h],victim.p[m]))p+=a.pr[h];
                    array<double,4> z{};for(int h=0;h<4;h++){z[h]+=dp[h]*(1-p);z[min(3,h+1)]+=dp[h]*p;}dp=z;
                }
                double hits=0,kill=0;for(int h=0;h<4;h++){hits+=dp[h]*min(h,need);if(h>=need)kill+=dp[h];}
                val+=victim.pr[m]*(hits*1.75+kill*(u[victim.id].flag>=0?19.0:7.0));
            }
            otherFocus[t]+=val*val*val;
        }
        for(int k=0;k<3;k++)for(int m=0;m<5;m++) {
            hit[k][m].assign(pred.size(),{});block[k][m]=0;
            int a=pos[own[k]];if(a<0||nb[a][m]<0)continue;int p=nb[a][m];
            double unblocked=1;
            for(int j=0;j<(int)pred.size();j++) {
                auto& e=pred[j];double bp=0;int b=pos[e.id];
                for(int h=0;h<5;h++)if(e.p[h]>=0) {
                    int q=e.p[h],ap=p,bp0=q;
                    if(p==q||(p==b&&q==a)||(p==b&&q==b)) {ap=a;bp0=b;bp+=e.pr[h];}
                    if(q==a&&p==a)bp0=b;
                    hit[k][m][j][h]=shot(ap,bp0)?1.0:0.0;
                }
                unblocked*=1-bp;
            }
            block[k][m]=1-unblocked;
        }
    }
    double incoming(const array<int,3>& move,array<double,3>& death,array<double,3>& expected) const {
        array<array<double,4>,3> dp{};array<double,3> eh{};
        for(int k=0;k<3;k++)dp[k][0]=1;
        for(int j=0;j<(int)pred.size();j++) {
            auto& e=pred[j];if(u[e.id].flag>=0||immune[e.id])continue;
            array<double,3> prob{};
            for(int h=0;h<5;h++)if(e.p[h]>=0) {
                array<double,3> w{};double sum=e.other[h];
                for(int k=0;k<3;k++)if(pos[own[k]]>=0&&!immune[own[k]]) {
                    double s=hit[k][move[k]][j][h];
                    if(s>0) {w[k]=tw(own[k]);if(lasttarget[e.id]==own[k])w[k]*=1.25;sum+=w[k];}
                }
                if(sum>0)for(int k=0;k<3;k++)prob[k]+=e.pr[h]*w[k]/sum;
            }
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
        double focusloss=0;
        array<double,3> focusdeath{};
        for(int t=0;t<(int)otherFocus.size();t++)if(t!=me) {
            array<double,3> value{},loss{},dk{};double sum=otherFocus[t];
            for(int k=0;k<3;k++)if(pos[own[k]]>=0&&!immune[own[k]]) {
                int id=own[k],need=(u[id].hp+33)/34;array<double,4> z{{1,0,0,0}};
                for(int j=0;j<(int)pred.size();j++)if(u[pred[j].id].team==t&&u[pred[j].id].flag<0&&!immune[pred[j].id]) {
                    double p=0;for(int h=0;h<5;h++)p+=pred[j].pr[h]*hit[k][move[k]][j][h];
                    array<double,4> zz{};for(int h=0;h<4;h++){zz[h]+=z[h]*(1-p);zz[min(3,h+1)]+=z[h]*p;}z=zz;
                }
                double hits=0,kills=0;for(int h=0;h<4;h++){hits+=z[h]*min(h,need);if(h>=need)kills+=z[h];}
                double desirability=hits*1.75+kills*(u[id].flag>=0?22.0:7.0);
                value[k]=desirability*desirability*desirability;
                if(u[id].flag>=0)value[k]*=1.8;
                sum+=value[k];dk[k]=kills;
                double life=u[id].flag>=0?24.0+max(0,12-bd[me][pos[id]])*.5:8.0;
                loss[k]=hits*(u[id].flag>=0?2.1:1.1)+life*kills;
            }
            if(sum>0)for(int k=0;k<3;k++){focusloss+=value[k]*loss[k]/sum;focusdeath[k]+=value[k]*dk[k]/sum;}
        }
        for(int k=0;k<3;k++){death[k]=max(death[k],min(1.0,focusdeath[k])*.65);expected[k]=eh[k];}
        return .55*v-.45*max(-v,focusloss);
    }
    using FutureTable=array<vector<array<array<double,4>,5>>,3>;
    FutureTable futureTable(const array<int,3>& mv,const array<int,3>& pp,const Game& g) const {
        FutureTable ft;
        for(int k=0;k<3;k++) {
            ft[k].resize(pred.size());int id=own[k],p=pp[k];if(p<0)continue;
            bool carry=u[id].flag>=0;
            for(auto& f:g.flags)if(f.x>=0&&cell(f.x,f.y)==p&&(f.return_at<0||f.return_at>g.turn))carry=true;
            if(!carry||bd[me][p]==0)continue;
            vector<int> path{p};
            while(bd[me][path.back()]>0&&path.size()<9) {
                int a=path.back(),best=-1;double bv=1e9;
                for(int m=1;m<5;m++)if(nb[a][m]>=0&&bd[me][nb[a][m]]<bd[me][a]) {
                    int z=nb[a][m];double v=0;
                    for(auto& e:pred)if(u[e.id].flag<0&&D(z,pos[e.id])<=6)v+=1.0/(.4+D(z,pos[e.id]));
                    if(v<bv){bv=v;best=z;}
                }
                if(best<0)break;path.push_back(best);
            }
            for(int j=0;j<(int)pred.size();j++) {
                int enemy=pred[j].id;
                if(u[enemy].flag>=0||D(p,pos[enemy])>7)continue;
                int need=(u[enemy].hp+33)/34;
                for(int h=0;h<5;h++)if(pred[j].p[h]>=0) {
                    int q=pred[j].p[h],first=99;
                    for(int step=1;step<(int)path.size();step++)if(D(q,path[step])<=step+2){first=step;break;}
                    if(first==99)continue;
                    double support=0;
                    for(int s=0;s<3;s++)if(s!=k&&pp[s]>=0&&u[own[s]].flag<0) {
                        int d=D(pp[s],q);
                        support+=d<=2?1:d==3?.95:d==4?.60:d==5?.25:0;
                    }
                    for(int shots=0;shots<4;shots++) {
                        if(shots>=need)continue;
                        int kill=support>.15?(int)ceil((need-shots)/support):99;
                        int exposed=max(0,min({bd[me][p],kill,7})-first+1);
                        double w=u[id].hp<=34?15:u[id].hp<=68?8:5;
                        ft[k][j][h][shots]=exposed*.85*w/(w+pred[j].other[h]);
                    }
                }
            }
        }
        return ft;
    }
    double futureRisk(const array<int,3>& mv,const array<int,3>& targets,const FutureTable& ft,const array<double,3>& now,const array<int,3>& pp) const {
        array<double,3> future{};
        for(int j=0;j<(int)pred.size();j++)for(int h=0;h<5;h++)if(pred[j].p[h]>=0) {
            int cnt=0;
            for(int k=0;k<3;k++)if(targets[k]==j&&hit[k][mv[k]][j][h]>.5)cnt++;
            for(int k=0;k<3;k++)if(u[own[k]].flag>=0||targets[k]<0)future[k]+=pred[j].pr[h]*ft[k][j][h][cnt];
        }
        double value=0;
        for(int k=0;k<3;k++)if(pp[k]>=0&&future[k]>0&&(u[own[k]].flag>=0||targets[k]<0)) {
            if(n>23&&u[own[k]].flag<0)continue;
            double need=(u[own[k]].hp+33)/34;
            double danger=max(0.0,min(1.0,(now[k]+future[k]-need+.45)/1.3));
            value-=danger*(u[own[k]].flag>=0?15:18);
            value-=future[k]*.3;
        }
        return value;
    }
    double firing(const array<int,3>& move,const array<int,3>& targets) const {
        double v=0;
        for(int j=0;j<(int)pred.size();j++) {
            int id=pred[j].id;if(immune[id])continue;
            bool any=false;for(int k=0;k<3;k++)if(targets[k]==j)any=true;if(!any)continue;
            int need=(u[id].hp+33)/34;
            double kill=0,hits=0;
            int legal=0;for(int h=0;h<5;h++)if(pred[j].p[h]>=0)legal++;
            for(int h=0;h<5;h++)if(pred[j].p[h]>=0) {
                int cnt=0;for(int k=0;k<3;k++)if(targets[k]==j&&hit[k][move[k]][j][h]>.5)cnt++;
                double mix=u[id].flag>=0?.06:SHOT_MIX;
                double probability=(1-mix)*pred[j].pr[h]+mix/legal;
                hits+=probability*min(need,cnt);if(cnt>=need)kill+=probability;
            }
            double value=7;
            if(u[id].flag>=0)value+=14.0/(1+.05*bd[me][pos[id]]);
            for(int k=0;k<3;k++)if(pos[own[k]]>=0&&u[own[k]].flag>=0&&D(pos[id],pos[own[k]])<=5)value+=5;
            v+=hits*1.75+kill*value;
        }
        return v;
    }
    struct Candidate {array<int,3> mv,ac;double value,future;};
    vector<int> settle(const array<int,3>& mv,const vector<int>& em) const {
        int count=u.size();vector<int> want(count,-1),owner(cells,-1),frequency(cells,0);vector<bool> blocked(count,false);
        for(int id=0;id<count;id++)if(pos[id]>=0) {owner[pos[id]]=id;want[id]=pos[id];}
        for(int j=0;j<(int)pred.size();j++)want[pred[j].id]=pred[j].p[em[j]];
        for(int k=0;k<3;k++)if(pos[own[k]]>=0)want[own[k]]=nb[pos[own[k]]][mv[k]];
        for(int p:want)if(p>=0)frequency[p]++;
        for(int id=0;id<count;id++)if(want[id]>=0) {
            if(frequency[want[id]]>1)blocked[id]=true;
            int other=owner[want[id]];
            if(other>=0&&other!=id&&u[other].team!=u[id].team&&want[other]==pos[id])blocked[id]=blocked[other]=true;
        }
        for(bool change=true;change;) {
            change=false;
            for(int id=0;id<count;id++)if(want[id]>=0&&!blocked[id]&&want[id]!=pos[id]) {
                int other=owner[want[id]];
                if(other>=0&&(blocked[other]||want[other]==pos[other])){blocked[id]=true;change=true;}
            }
        }
        for(int id=0;id<count;id++)if(blocked[id])want[id]=pos[id];
        return want;
    }
    double actualIncoming(const vector<int>& p,array<double,3>& death) const {
        array<array<double,4>,3> dp{};for(int k=0;k<3;k++)dp[k][0]=1;
        for(auto& e:pred)if(u[e.id].flag<0&&!immune[e.id]) {
            double total=0;array<double,3> w{};
            for(auto& z:u)if(z.team!=u[e.id].team&&p[z.id]>=0&&!immune[z.id]&&shot(p[e.id],p[z.id])) {
                double v=tw(z.id);if(z.team==me&&lasttarget[e.id]==z.id)v*=1.25;
                total+=v;if(z.team==me)w[z.id-3*me]=v;
            }
            if(total>0)for(int k=0;k<3;k++) {
                double pr=w[k]/total;array<double,4> z{};
                for(int h=0;h<4;h++){z[h]+=dp[k][h]*(1-pr);z[min(3,h+1)]+=dp[k][h]*pr;}dp[k]=z;
            }
        }
        double loss=0,focus=0;
        for(int k=0;k<3;k++)if(p[own[k]]>=0) {
            int id=own[k],need=(u[id].hp+33)/34;double hits=0,kills=0;
            for(int h=0;h<4;h++){hits+=dp[k][h]*min(need,h);if(h>=need)kills+=dp[k][h];}
            death[k]=kills;
            double life=u[id].flag>=0?24.0+max(0,12-bd[me][pos[id]])*.5:(bd[me][pos[id]]<4?6:8);
            loss+=hits*(u[id].flag>=0?2.1:1.1)+life*kills;
            if(u[id].flag>=0&&u[id].hp<=68)loss+=hits*1.2;
        }
        for(int t=0;t<(int)otherFocus.size();t++)if(t!=me) {
            double denominator=0;array<double,3> weight{},ls{},dk{};
            for(auto& victim:u)if(victim.team!=t&&p[victim.id]>=0&&!immune[victim.id]) {
                int cnt=0;for(int j=t*3;j<t*3+3;j++)if(p[j]>=0&&!immune[j]&&u[j].flag<0&&shot(p[j],p[victim.id]))cnt++;
                int need=(victim.hp+33)/34;double kills=cnt>=need?1:0,hits=min(need,cnt);
                double desire=hits*1.75+kills*(victim.flag>=0?(victim.team==me?22:19):7);
                double w=desire*desire*desire;if(victim.team==me&&victim.flag>=0)w*=1.8;
                denominator+=w;
                if(victim.team==me) {
                    int k=victim.id-3*me;weight[k]=w;dk[k]=kills;
                    double life=victim.flag>=0?24+max(0,12-bd[me][pos[victim.id]])*.5:8;
                    ls[k]=hits*(victim.flag>=0?2.1:1.1)+life*kills;
                }
            }
            if(denominator>0)for(int k=0;k<3;k++){focus+=weight[k]*ls[k]/denominator;death[k]=max(death[k],weight[k]*dk[k]/denominator*.65);}
        }
        return -.55*loss-.45*max(loss,focus);
    }
    double actualFire(const vector<int>& p,const array<int,3>& ac) const {
        vector<int> hits(u.size(),0);
        for(int k=0;k<3;k++)if(ac[k]>=0&&p[own[k]]>=0&&!immune[own[k]]&&u[own[k]].flag<0) {
            int id=pred[ac[k]].id;if(!immune[id]&&shot(p[own[k]],p[id]))hits[id]++;
        }
        double value=0;
        for(auto& e:pred)if(hits[e.id]) {
            int need=(u[e.id].hp+33)/34;double kv=7;
            if(u[e.id].flag>=0)kv+=14/(1+.05*bd[me][pos[e.id]]);
            for(int k=0;k<3;k++)if(pos[own[k]]>=0&&u[own[k]].flag>=0&&D(pos[e.id],pos[own[k]])<=5)kv+=5;
            value+=1.75*min(hits[e.id],need)+(hits[e.id]>=need?kv:0);
        }
        return value;
    }
public:
    string decide(const Game& g) {
        if(!n)init(g);update(g);act=ngActive(g.teams);
        plan(g);predict(g);
        array<vector<int>,3> moves;
        array<array<double,5>,3> positional{};
        array<array<vector<int>,5>,3> actions;
        array<array<double,5>,3> pickup{};
        for(int k=0;k<3;k++) {
            int id=own[k],a=pos[id];
            if(a<0){moves[k]={0};actions[k][0]={-1};continue;}
            for(int m=0;m<5;m++)if(nb[a][m]>=0) {
                int p=nb[a][m];moves[k].push_back(m);
                double v=0;
                if(u[id].flag>=0) {
                    double rd=route[k][p];v=-1.5*(.7*rd+.3*bd[me][p]);
                    if(bd[me][p]==0)v+=30;
                    v-=.15*(m==0);
                } else if(assigned[k]>=0) {
                    auto& ms=missions[assigned[k]];int gp=goal[assigned[k]][k];int d=D(p,gp);
                    if(ms.type==0&&rolew[k]<.9&&d==0)d=1;
                    if(ms.type==1)d=max(0,d-1);
                    if(ms.type==2)d=max(0,d-2);
                    v=-rolew[k]*d;
                    if(ms.type==0)for(int h=0;h<3;h++)if(h!=k&&assigned[h]==assigned[k]&&pos[own[h]]>=0)v-=.1*max(0,D(p,pos[own[h]])-3);
                }
                v-=block[k][m]*1.7;
                if(m!=0)v+=(g.turn*7919+id*193+m*769)%103*.00015;
                positional[k][m]=v;
                vector<pair<double,int>> opts;
                if(u[id].flag<0&&!immune[id])for(int j=0;j<(int)pred.size();j++)if(!immune[pred[j].id]) {
                    double h=0;for(int z=0;z<5;z++)h+=pred[j].pr[z]*hit[k][m][j][z];
                    if(h>.025){int e=pred[j].id;double val=h*(1.75+(u[e].hp<=34?7:u[e].hp<=68?3:1)+(u[e].flag>=0?7:0));opts.push_back({val,j});}
                }
                sort(opts.rbegin(),opts.rend());
                for(int z=0;z<min(4,(int)opts.size());z++)actions[k][m].push_back(opts[z].second);
                bool flaghere=false;
                for(auto& f:g.flags)if(f.x>=0&&cell(f.x,f.y)==p&&(f.return_at<0||f.return_at>g.turn))flaghere=true;
                if(u[id].flag<0&&flaghere){pickup[k][m]=PICK_BONUS*(1-block[k][m]);actions[k][m].push_back(-1);}
                if(actions[k][m].empty())actions[k][m].push_back(-1);
                if(u[id].flag>=0){actions[k][m]={-1};pickup[k][m]=0;}
            }
        }
        double best=-1e50;array<int,3> bm{{0,0,0}},ba{{-1,-1,-1}};
        vector<Candidate> candidates;
        for(int a:moves[0])for(int b:moves[1])for(int c:moves[2]) {
            array<int,3> mv{{a,b,c}},pp{{-1,-1,-1}};bool good=true;
            for(int k=0;k<3;k++)if(pos[own[k]]>=0)pp[k]=nb[pos[own[k]]][mv[k]];
            for(int k=0;k<3;k++)for(int h=k+1;h<3;h++)if(pp[k]>=0&&pp[k]==pp[h])good=false;
            if(!good)continue;
            double combat=g.teams==2?C_DUEL:C_MULTI,futurescale=g.teams==2?F_DUEL:F_MULTI;
            array<double,3> death{},expected{};double value=combat*incoming(mv,death,expected);
            for(int k=0;k<3;k++)value+=positional[k][mv[k]];
            auto ft=futureTable(mv,pp,g);
            Candidate candidate{mv,{{-1,-1,-1}},-1e50,0};
            for(int x:actions[0][a])for(int y:actions[1][b])for(int z:actions[2][c]) {
                array<int,3> ac{{x,y,z}};double fr=futurescale*futureRisk(mv,ac,ft,expected,pp);
                double v=value+combat*firing(mv,ac)+fr;
                for(int k=0;k<3;k++)if(ac[k]<0)v+=pickup[k][mv[k]]*(1-death[k]);
                if(v>best){best=v;bm=mv;ba=ac;}
                if(v>candidate.value)candidate={mv,ac,v,fr};
            }
            candidates.push_back(candidate);
        }
        if(COLLISION_WEIGHT>0&&g.teams>=4&&!pred.empty()) {
            auto refinementStart=chrono::steady_clock::now();
            sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){return a.value>b.value;});
            if(candidates.size()>16)candidates.resize(16);
            constexpr int samples=24;vector<vector<int>> enemyMoves(samples,vector<int>(pred.size()));
            for(int s=0;s<samples;s++)for(int j=0;j<(int)pred.size();j++) {
                uint32_t h=(uint32_t)(g.turn*101+pred[j].id*7919+187);
                h^=h>>16;h*=0x7feb352dU;h^=h>>15;h*=0x846ca68bU;h^=h>>16;
                double r=fmod((s+.5)/samples+(h%65536)/65536.0,1.0),acc=0;
                int chosen=0;for(int m=0;m<5;m++)if(pred[j].p[m]>=0){chosen=m;acc+=pred[j].pr[m];if(r<=acc)break;}
                enemyMoves[s][j]=chosen;
            }
            double refinedBest=-1e50;
            for(auto& ca:candidates) {
                if(chrono::duration<double,milli>(chrono::steady_clock::now()-refinementStart).count()>12)break;
                double actual=0;
                for(int s=0;s<samples;s++) {
                    auto p=settle(ca.mv,enemyMoves[s]);array<double,3> death{};
                    double v=(g.teams==2?C_DUEL:C_MULTI)*(actualIncoming(p,death)+actualFire(p,ca.ac))+ca.future;
                    for(int k=0;k<3;k++)if(p[own[k]]>=0) {
                        int id=own[k],want=nb[pos[id]][ca.mv[k]],a=p[id];
                        v+=positional[k][ca.mv[k]]+block[k][ca.mv[k]]*1.7;
                        if(u[id].flag>=0) {
                            v-=1.5*(.7*(route[k][a]-route[k][want])+.3*(bd[me][a]-bd[me][want]));
                            if(bd[me][want]==0&&bd[me][a]>0)v-=30;
                        } else if(assigned[k]>=0) {
                            auto& ms=missions[assigned[k]];int gp=goal[assigned[k]][k],offset=ms.type==1?1:ms.type==2?2:0;
                            auto gd=[&](int at){int d=D(at,gp);if(ms.type==0&&rolew[k]<.9&&d==0)d=1;return max(0,d-offset);};
                            v-=rolew[k]*(gd(a)-gd(want));
                        }
                        if(ca.ac[k]<0&&u[id].flag<0)for(auto& f:g.flags)if(f.x>=0&&cell(f.x,f.y)==a&&(f.return_at<0||f.return_at>g.turn)) {v+=PICK_BONUS*(1-death[k]);break;}
                    }
                    actual+=v/samples;
                }
                double value=(1-COLLISION_WEIGHT)*ca.value+COLLISION_WEIGHT*actual;
                if(value>refinedBest){refinedBest=value;bm=ca.mv;ba=ca.ac;}
            }
        }
        string out=to_string(g.turn);
        for(int k=0;k<3;k++) {out+=' ';out+=mc[bm[k]];out+=' ';out+=ba[k]>=0?to_string(pred[ba[k]].id):"P";}
        for(int id=0;id<(int)u.size();id++){oldp[id]=pos[id];oldhp[id]=u[id].hp;}
        return out;
    }
};
}  // namespace duel


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Game g;
    multi::Planner many;
    duel::Player two;
    while (read_turn(g)) cout << (g.teams == 2 ? two.decide(g) : many.decide(g)) << endl;
    return 0;
}

