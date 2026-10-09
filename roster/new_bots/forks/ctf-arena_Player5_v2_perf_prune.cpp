
#ifndef CTF_PROFILE
#define CTF_PROFILE 1
#endif
#ifndef PERF_NO_CUTOFF
#define PERF_NO_CUTOFF 1
#endif
#include <ctime>
#include <cstdlib>
#include <new>
#include <cstddef>
namespace perf {
enum Stage {DECIDE,FORECAST,ASSIGN,HOME_ROUTE,CHOOSE,STEP_EXACT,GAME_COPY,RUNTIME_COPY,SERIALIZE,REFINE,NAV_INIT,COUNT};
static constexpr const char* names[]={"decide_misc","forecast","assign","home_route","choose","step_exact","game_copy","runtime_copy","serialization","refine_misc","nav_init"};
struct Measure{double ticks=0,rootTicks=0,futureTicks=0;unsigned long long calls=0,rootCalls=0,futureCalls=0,allocations=0,allocation_bytes=0;};
inline Measure measures[COUNT];inline bool enabled=false;inline int stack[64],depth=0,decisionDepth=0;inline clock_t last=0;
inline unsigned long long moveTriples=0,validTriples=0,collisionCases=0,shotDpSubsets=0,refineCandidates=0,scenarioPairs=0,rootDecides=0,futureDecides=0;
inline void flush(){if(CTF_PROFILE&&enabled){clock_t now=std::clock();if(depth){auto& m=measures[stack[depth-1]];double d=double(now-last);m.ticks+=d;if(decisionDepth>1)m.futureTicks+=d;else m.rootTicks+=d;}last=now;}}
struct Scope{bool live;Stage id;Scope(Stage s):live(CTF_PROFILE&&enabled),id(s){if(live){flush();if(s==DECIDE)decisionDepth++;stack[depth++]=s;auto& m=measures[s];m.calls++;if(decisionDepth>1)m.futureCalls++;else m.rootCalls++;}}~Scope(){if(live){flush();--depth;if(id==DECIDE)decisionDepth--;}}};
inline void allocated(std::size_t n){if(CTF_PROFILE&&enabled&&depth){auto& s=measures[stack[depth-1]];s.allocations++;s.allocation_bytes+=n;}}
inline void add(unsigned long long& x){if(CTF_PROFILE&&enabled)++x;}
}
void* operator new(std::size_t n){perf::allocated(n);if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
#ifndef CTF_LOOKAHEAD
#define CTF_LOOKAHEAD 1
#endif
#ifndef CTF_RELAY
#define CTF_RELAY 0
#endif
#include <chrono>
constexpr double W_ground=30.434542;
constexpr double W_home_small=0.15;
constexpr double W_home_large=0.9003127;
constexpr double W_approach=0.9;
constexpr double W_late=0.9;
constexpr double W_camp=8.1214587;
constexpr double W_camp_home=0.32;
constexpr double W_camp_approach=0.4375542;
constexpr double W_camp_duplicate=8;
constexpr double W_flag_duplicate=14;
constexpr double W_contest_discount=3;
constexpr double W_escort=11;
constexpr double W_deny_small=19;
constexpr double W_deny_large=10.792114;
constexpr double W_deny_remote=0.40298439;
constexpr double W_cohesion=0.29081377;
constexpr double W_other_attention=0.2;
constexpr double W_pickup=22;
constexpr double W_route=1.7;
constexpr double W_death=12;
constexpr double W_hit=1.1;
constexpr double W_damage=0.04;
constexpr double W_carrier_death=45;
constexpr double W_carrier_hit=3;
constexpr double W_worst=10;
#include <algorithm>
#include <array>
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
struct Unit { int id, team, x, y, hp, flag, respawn_at; };
struct Flag { int id; string status; int x, y, holder, return_at; };
struct Event { string type; vector<int> args; };
struct Game {
    int teams=0,me=0,size=0,turns=0,hp=0,damage=0,respawn=0,flag_return=0,flag_cooldown=0;
    vector<string> map; vector<pair<int,int>> bases,spots;
    int turn=0; vector<int> score; vector<Unit> units; vector<Flag> flags; vector<Event> events;
};
bool read_turn(Game& g) {
    string w;
    while(cin>>w) {
        if(w=="INIT") cin>>g.teams>>g.me>>g.size>>g.turns>>g.hp>>g.damage>>g.respawn>>g.flag_return>>g.flag_cooldown;
        else if(w=="MAP") {g.map.resize(g.size);for(auto& s:g.map)cin>>s;}
        else if(w=="BASES" || w=="SPOTS") {int n;cin>>n;auto& a=w=="BASES"?g.bases:g.spots;a.resize(n);for(auto& p:a)cin>>p.first>>p.second;}
        else if(w=="TURN")cin>>g.turn;
        else if(w=="SCORE"){g.score.resize(g.teams);for(int& s:g.score)cin>>s;}
        else if(w=="UNITS"){int n;cin>>n;g.units.resize(n);for(auto& u:g.units){cin>>w>>u.id>>u.team>>u.x>>u.y>>u.hp>>u.flag>>u.respawn_at;}}
        else if(w=="FLAGS"){int n;cin>>n;g.flags.resize(n);for(auto& f:g.flags)cin>>w>>f.id>>f.status>>f.x>>f.y>>f.holder>>f.return_at;}
        // Current public state is sufficient for the policy; consume the event log.
        else if(w=="EVENTS"){int n;cin>>n;string line;getline(cin,line);while(n--)getline(cin,line);}
        else if(w=="END")return true;
    } return false;
}
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// A small joint move/action planner. All values are in units of useful movement.
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
struct Outcome { Pos p; double prob = 0; bool canShoot = true; };
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
    perf::Scope stageChoose(perf::CHOOSE);
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
        perf::add(perf::moveTriples);
        if(!PERF_NO_CUTOFF&&deadline&&std::isfinite(best.value)&&((a*25+b*5+c)&15)==0&&std::chrono::steady_clock::now()>=*deadline)return best;
        std::array<int,3> indices{a,b,c};
        std::array<Pos,3> want{options[0][a].p,options[1][b].p,options[2][c].p};
        bool valid=true;
        for(int i=0;i<3;++i) if(alive(own[i])) {
            if(wall(map,want[i])) valid=false;
            for(int j=i+1;j<3;++j) if(alive(own[j]) && want[i]==want[j]) valid=false;
        }
        if(!valid) continue;
        perf::add(perf::validTriples);

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
                    perf::add(perf::collisionCases);
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
                    if(enemy.u.flag>=0 || enemy.u.immune || !out.canShoot) continue;
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
                        incoming[e][i]+=probability*std::clamp(enemy.attention,0.0,1.0)*weights[i]/denominator;
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
        for(int e=0;e<E;++e) if(relevant[e] && alive(enemies[e].u) && !enemies[e].u.immune) {
            auto next=dp;
            for(int used=0;used<8;++used) if(std::isfinite(dp[used].value)) {
                int remaining=armedMask&~used;
                for(int sub=remaining;sub;sub=(sub-1)&remaining) {
                    perf::add(perf::shotDpSubsets);
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
        if(beam){Choice candidate{indices,targets,value};
            if(beam->size()<8)beam->push_back(candidate);
            else{int w=0;for(int z=1;z<(int)beam->size();z++)if((*beam)[z].value<(*beam)[w].value)w=z;
                if(value>(*beam)[w].value)(*beam)[w]=candidate;}}
        if(value>best.value+1e-9) best={indices,targets,value};
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
    void init(const Game& g) {perf::Scope stageNav(perf::NAV_INIT);
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
            auto [bx,by]=g.bases[t];for(int yy=by-1;yy<=by+1;yy++)for(int xx=bx-1;xx<=bx+1;xx++)spawn[t].push_back(cell(xx,yy));
            stable_sort(spawn[t].begin(),spawn[t].end(),[&](int a,int b){auto key=[&](int p){return pair<int,int>{abs(p%n-c)+abs(p/n-c),abs(p%n-bx)+2*abs(p/n-by)};};return key(a)<key(b);});
            for(int p:floors)for(int b:spawn[t])home[t][p]=min(home[t][p],dist(p,b));
        }
    }
};
struct Task {int type=0,id=-1,p=-1;double reward=0,contest=0;};
struct Planner {
    Nav nav; bool initialized=false; vector<Unit> prev;
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
    void forecast(const Game& g){perf::Scope stageForecast(perf::FORECAST);
        foes.clear();threat.assign(nav.N,0);
        for(auto& u:g.units)if(u.team!=g.me&&pos[u.id]>=0){
            int p=pos[u.id],goal=closestGoal(u,g),target=-1;double tv=-1e9;
            for(auto& v:g.units)if(v.team!=u.team&&pos[v.id]>=0&&!immune[v.id]){
                int d=nav.dist(p,pos[v.id]);if(d>7)continue;
                double val=8-d+(v.flag>=0?6:0)+(100-v.hp)*.025;
                int friends=0,hostiles=0;for(auto& w:g.units)if(pos[w.id]>=0&&man(pos[w.id],pos[v.id])<=4){if(w.team==u.team)friends++;else if(w.team==v.team)hostiles++;}
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
            if(u.flag<0)for(auto& o:e.forecast){int q=nav.cell(o.p.x,o.p.y);for(auto& f:g.flags)if(available(f,g)&&q==nav.cell(f.x,f.y))o.canShoot=false;}
            foes.push_back(e);
            if(u.flag<0&&!immune[u.id])for(auto& o:e.forecast){int q=nav.cell(o.p.x,o.p.y);for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++)if(abs(dx)+abs(dy)<=2){int x=o.p.x+dx,y=o.p.y+dy;if(x<0||x>=nav.n||y<0||y>=nav.n)continue;int r=nav.cell(x,y);if(nav.open(r)&&shot(q,r,g))threat[r]+=o.prob*e.attention;}}
        }
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
            for(auto& u:g.units)if(u.team!=g.me&&pos[u.id]>=0&&u.flag<0){int d=nav.dist(pos[u.id],q);tdist[u.team]=min(tdist[u.team],d);if(d<=6)ct++;}
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
            int rival=999;for(auto& v:g.units)if(v.team!=g.me&&pos[v.id]>=0&&v.flag<0)rival=min(rival,nav.dist(pos[v.id],t.p));
            double late=max(0,d-rival-1);double val=t.reward-W_approach*d-W_late*late;
            if(rival<=1&&d>4&&t.contest<1.5)val-=10;
            if(g.turn+max(0,d-1)+nav.home[g.me][t.p]>g.turns)val-=30;return val;
        }return t.reward-W_camp_approach*d;
    }
    array<Task,3> assign(const Game& g){perf::Scope stageAssign(perf::ASSIGN);
        vector<Task> ts=tasks(g);array<vector<pair<double,int>>,3> ranked;
        vector<array<int,3>> targets(ts.size());vector<array<double,3>> vals(ts.size());
        for(int i=0;i<(int)ts.size();i++)for(int k=0;k<3;k++){
            int q;double v=taskValue(k,ts[i],g,q);targets[i][k]=q;if(ts[i].type==oldType[k]&&ts[i].id==oldId[k])v+=.9;
            vals[i][k]=v;ranked[k].push_back({v,i});
        }
        for(int k=0;k<3;k++){sort(ranked[k].rbegin(),ranked[k].rend());if(ranked[k].size()>8)ranked[k].resize(8);}
        double best=-1e20;array<int,3> ids{};
        for(auto a:ranked[0])for(auto b:ranked[1])for(auto c:ranked[2]){
            array<int,3> id{{a.second,b.second,c.second}};double value=a.first+b.first+c.first;
            for(int i=0;i<3;i++)if(pos[3*g.me+i]<0)value-=vals[id[i]][i];
            for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)if(pos[3*g.me+i]>=0&&pos[3*g.me+j]>=0){
                if(id[i]==id[j]){Task& t=ts[id[i]];if(t.type==1)value-=W_flag_duplicate-W_contest_discount*t.contest;if(t.type==4)value-=W_camp_duplicate;if(t.type==2)value-=3;if(t.type==3)value-=3;}
                else {int d=nav.dist(targets[id[i]][i],targets[id[j]][j]);if(d<=4)value+=.5*(5-d);}
            }
            if(value>best){best=value;ids=id;}
        }
        array<Task,3> out;
        for(int k=0;k<3;k++){out[k]=ts[ids[k]];out[k].p=targets[ids[k]][k];oldType[k]=out[k].type;oldId[k]=out[k].id;
            if(g.units[3*g.me+k].flag>=0){out[k]={0,g.units[3*g.me+k].flag,-2,0,0};oldType[k]=0;}}
        return out;
    }
    vector<double> homeRoute(const Game& g,int uid){perf::Scope stageHome(perf::HOME_ROUTE);
        vector<double> d(nav.N,1e9);priority_queue<pair<double,int>,vector<pair<double,int>>,greater<pair<double,int>>> pq;
        for(int p:nav.spawn[g.me]){d[p]=0;pq.push({0,p});}const Unit& u=g.units[uid];double hp=immune[uid]?100:u.hp;
        while(!pq.empty()){
            auto [v,p]=pq.top();pq.pop();if(v>d[p]+1e-6)continue;
            double risk=threat[p],shield=0;for(int k=0;k<3;k++){int id=3*g.me+k;if(id!=uid&&pos[id]>=0&&g.units[id].flag<0&&man(pos[id],p)<=3)shield+=.3;}
            risk=max(0.,risk-shield);double cost=1+min(12.,risk*(hp<=34?7.:hp<=68?5.:3.));
            for(int m=1;m<5;m++){int q=nav.nb[p][m];if(q>=0&&v+cost<d[q]){d[q]=v+cost;pq.push({d[q],q});}}
        }return d;
    }
    bool refining=false;chrono::steady_clock::time_point decisionDeadline;
    // Public-state-only deterministic turn kernel. Unknown random returns are
    // not guessed: return false and retain the unrefined policy instead.
    bool stepExact(const Game& g,const vector<int>& requested,const vector<int>& actions,Game& next){perf::Scope stageStep(perf::STEP_EXACT);
        for(const auto& f:g.flags)if((f.status=="cooldown"||f.status=="dropped")&&f.return_at<=g.turn)return false;
        respawns(g);{perf::Scope gameCopy(perf::GAME_COPY);next=g;}next.turn=g.turn+1;next.events.clear();
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
            auto& f=next.flags[u.flag];f.status="cooldown";f.holder=-1;f.x=f.y=-1;f.return_at=g.turn+g.flag_cooldown;u.flag=-1;next.score[u.team]++;
        }
        return true;
    }
    struct Runtime {vector<Unit> previous;array<int,3> type,id;vector<int> positions;vector<bool> immunity;vector<double> hazards;array<vector<double>,3> routes;vector<ctfcombat::Enemy> enemies;};
    Runtime saveRuntime()const{perf::Scope stageCopy(perf::RUNTIME_COPY);return {prev,oldType,oldId,pos,immune,threat,route,foes};}
    void restoreRuntime(const Runtime& s){perf::Scope stageCopy(perf::RUNTIME_COPY);prev=s.previous;oldType=s.type;oldId=s.id;pos=s.positions;immune=s.immunity;threat=s.hazards;route=s.routes;foes=s.enemies;}
    void modeledOrders(const Game& g,int scenario,vector<int>& want,vector<int>& acts){
        want=pos;acts.assign(g.units.size(),-1);
        for(const auto& e:foes){int oi=0;
            if(scenario==0){for(int j=1;j<(int)e.forecast.size();j++)if(e.forecast[j].prob>e.forecast[oi].prob)oi=j;}
            else{unsigned h=unsigned(g.turn*7919+e.u.id*104729+scenario*65537);h^=h>>13;h*=0x85ebca6bU;
                double r=(h%65536+.5)/65536.;for(int j=0;j<(int)e.forecast.size();j++){oi=j;r-=e.forecast[j].prob;if(r<=0)break;}}
            auto p=e.forecast[oi].p;want[e.u.id]=nav.cell(p.x,p.y);
        }
    }
    void modeledAttacks(const Game& g,int scenario,const vector<int>& want,vector<int>& acts){
        vector<vector<int>> allocated(g.teams,vector<int>(g.units.size(),0));
        for(const auto& e:foes){int i=e.u.id;if(g.units[i].flag>=0||immune[i])continue;
            bool onFlag=false;for(const auto& f:g.flags)if(available(f,g)&&want[i]==nav.cell(f.x,f.y))onFlag=true;
            if(onFlag&&scenario==0){acts[i]=-2;continue;}
            double best=0;int target=-1;
            for(const auto& v:g.units)if(v.team!=g.units[i].team&&pos[v.id]>=0&&!immune[v.id]&&shot(want[i],want[v.id],g)){
                double priority=v.flag>=0?5.:1.;if(v.hp<=g.damage)priority*=2.4;else if(v.hp<=2*g.damage)priority*=1.35;
                // Each opposing squad avoids redundant overkill, while keeping
                // attack orders independent of later casualties this turn.
                int need=(v.hp+g.damage-1)/g.damage;if(allocated[g.units[i].team][v.id]>=need)priority*=.15;else if(allocated[g.units[i].team][v.id]+1>=need)priority+=3;
                if(priority>best){best=priority;target=v.id;}}
            acts[i]=target;if(target>=0)allocated[g.units[i].team][target]++;else if(onFlag)acts[i]=-2;
        }
    }
    double terminalFlags(const Game& g)const{
        double value=0;for(const auto& f:g.flags){
            if(f.status=="carried"&&f.holder>=0){const auto& u=g.units[f.holder];if(u.x<0)continue;
                int p=nav.cell(u.x,u.y);double v=30./(1+.10*nav.home[u.team][p]);value+=u.team==g.me?v:-.5*v;
            }else if(f.status=="dropped"&&f.x>=0){int p=nav.cell(f.x,f.y);bool friendly=false,enemy=false;
                for(const auto& u:g.units)if(u.x>=0&&nav.cell(u.x,u.y)==p){if(u.team==g.me)friendly=true;else enemy=true;}
                if(friendly&&!enemy)value+=.55*30./(1+.10*nav.home[g.me][p]);
            }}return value;
    }
    double realized(const Game& before,const Game& after)const{
        double value=70.*(after.score[before.me]-before.score[before.me]);
        for(int t=0;t<before.teams;t++)if(t!=before.me)value-=35.*(after.score[t]-before.score[t]);
        vector<int> ownHits(before.units.size(),0),allHits(before.units.size(),0);
        for(const auto& e:after.events)if(e.type=="attack"&&e.args.size()==2){int a=e.args[0],t=e.args[1];allHits[t]++;if(before.units[a].team==before.me)ownHits[t]++;}
        for(const auto& u:before.units)if(u.x>=0){const auto& v=after.units[u.id];bool dead=v.x<0;
            double hit=(u.hp-max(0,v.hp))/double(before.damage);if(u.team==before.me)value-=hit*(1.1+(u.flag>=0?3.:0))+(dead?(12+(u.flag>=0?45:0)):0);
            else if(ownHits[u.id]){double share=ownHits[u.id]/double(max(1,allHits[u.id]));
                value+=share*(.04*(u.hp-max(0,v.hp))+(dead?(11+(u.flag>=0?34:0)):0));}}
        return value;
    }
    ctfcombat::Choice refineLocal(const Game& g,const array<vector<ctfcombat::Move>,3>& options,vector<ctfcombat::Choice> beam,ctfcombat::Choice fallback){perf::Scope stageRefine(perf::REFINE);
        if(g.turn>=g.turns)return fallback;
        bool contact=false;for(int k=0;k<3;k++){const auto& u=g.units[3*g.me+k];if(u.flag<0||pos[u.id]<0)continue;
            for(const auto& e:foes)if(e.u.flag<0&&man(pos[u.id],nav.cell(e.u.p.x,e.u.p.y))<=6)contact=true;}
        if(!contact)return fallback;
        // Refinement never predicts a concealed random flag respawn.
        for(const auto& f:g.flags)if((f.status=="cooldown"||f.status=="dropped")&&f.return_at<=g.turn+1)return fallback;
        sort(beam.begin(),beam.end(),[](const auto& a,const auto& b){return a.value>b.value;});int width=g.teams<=5?4:2;if(beam.size()>size_t(width))beam.resize(width);
        vector<ctfcombat::Choice> candidates{fallback};
        for(auto c:beam)if(c.value>=fallback.value-12&&c.moveIndex!=fallback.moveIndex)candidates.push_back(c);
        if(CTF_RELAY){for(auto c:beam)for(int k=0;k<3;k++){const auto& u=g.units[3*g.me+k];auto p=options[k][c.moveIndex[k]].p;
            if(u.flag>=0&&u.hp>g.damage&&!immune[u.id]&&nav.home[g.me][nav.cell(p.x,p.y)]>=4&&g.turn+5<g.turns){c.target[k]=-3;candidates.push_back(c);break;}}}
        auto saved=saveRuntime();auto deadline=min(decisionDeadline,chrono::steady_clock::now()+chrono::milliseconds(20));auto outerDeadline=decisionDeadline;decisionDeadline=deadline;refining=true;
        ctfcombat::Choice best=fallback;double bestScore=-1e100;
        for(const auto& candidate:candidates){perf::add(perf::refineCandidates);double avg=0,worst=1e100;bool complete=true,dropSafe=true;
            for(int scenario=0;scenario<2;scenario++){
                if(!PERF_NO_CUTOFF&&chrono::steady_clock::now()>=deadline){complete=false;break;}
                restoreRuntime(saved);vector<int> want,acts;modeledOrders(g,scenario,want,acts);
                for(int k=0;k<3;k++){int i=3*g.me+k;auto p=options[k][candidate.moveIndex[k]].p;if(pos[i]>=0)want[i]=nav.cell(p.x,p.y);acts[i]=candidate.target[k];}
                modeledAttacks(g,scenario,want,acts);Game next;if(!stepExact(g,want,acts,next)){complete=false;break;}
                for(int k=0;k<3;k++)if(candidate.target[k]==-3&&next.units[3*g.me+k].x<0)dropSafe=false;
                double v=realized(g,next);{perf::Scope previousCopy(perf::RUNTIME_COPY);prev=g.units;}string reply=decide(next);
                if(!PERF_NO_CUTOFF&&chrono::steady_clock::now()>=deadline){complete=false;break;}
                vector<int> want2,acts2;modeledOrders(next,scenario,want2,acts2);istringstream input(reply);int turn;input>>turn;
                for(int k=0;k<3;k++){char m;string a;input>>m>>a;int i=3*g.me+k,mi=0;for(int j=0;j<5;j++)if(letters[j]==m)mi=j;
                    if(pos[i]>=0)want2[i]=nav.nb[pos[i]][mi];acts2[i]=a=="P"?-2:a=="X"?-3:a=="-"?-1:stoi(a);}
                modeledAttacks(next,scenario,want2,acts2);Game third;if(!stepExact(next,want2,acts2,third)){complete=false;break;}
                if(!PERF_NO_CUTOFF&&chrono::steady_clock::now()>=deadline){complete=false;break;}
                v+=.75*realized(next,third)+terminalFlags(third)-terminalFlags(g);
                perf::add(perf::scenarioPairs);
                avg+=v*.5;worst=min(worst,v);
            }
            if(!complete)break;if(!dropSafe)continue;
            double score=candidate.value+.55*(.75*avg+.25*worst);
            // Dropping loses one combat turn and releases exclusive ownership.
            // Require a tangible advantage instead of tie-level simulator noise.
            bool dropping=false;for(int a:candidate.target)if(a==-3)dropping=true;if(dropping)score-=5;
            if(score>bestScore+1e-8){bestScore=score;best=candidate;}
        }
        restoreRuntime(saved);decisionDeadline=outerDeadline;refining=false;return best;
    }

    string decide(Game& g){perf::Scope stageDecide(perf::DECIDE);if(refining)perf::add(perf::futureDecides);else perf::add(perf::rootDecides);
        if(!refining)decisionDeadline=chrono::steady_clock::now()+chrono::milliseconds(g.turn==1?1800:35);
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
        ctfcombat::Params params;params.ownDeathCost=W_death;params.incomingHitCost=W_hit;params.dealtHpValue=W_damage;params.carrierDeathExtra=W_carrier_death;params.carrierHitExtra=W_carrier_hit;params.carrierWorstDeathCost=W_worst;vector<ctfcombat::Choice> beam;
        auto choice=ctfcombat::choose(g.map,own,options,foes,params,(CTF_LOOKAHEAD&&!refining)?&beam:nullptr,CTF_LOOKAHEAD?&decisionDeadline:nullptr);
        if(CTF_LOOKAHEAD&&!refining)choice=refineLocal(g,options,beam,choice);
        perf::Scope stageSerialize(perf::SERIALIZE);
        string out=to_string(g.turn);
        for(int k=0;k<3;k++){int m=choice.moveIndex[k],target=choice.target[k];if(m<0||m>=(int)options[k].size())m=0;
            out+=' ';out+=options[k][m].code;out+=' ';if(target>=0)out+=to_string(target);else if(target==-3)out+="X";else out+=(g.units[3*g.me+k].flag>=0?"-":"P");}
        {perf::Scope previousCopy(perf::RUNTIME_COPY);prev=g.units;}return out;
    }
};
int original_main(){ios::sync_with_stdio(false);cin.tie(nullptr);Game g;Planner p;while(read_turn(g))cout<<p.decide(g)<<endl;return 0;}

#include <iomanip>
int main(int argc,char**argv){
 int repetitions=argc>1?std::stoi(argv[1]):500;Game g;if(!read_turn(g))return 2;
 int previousCount;cin>>previousCount;vector<Unit> previous(previousCount);for(auto&u:previous)cin>>u.id>>u.team>>u.x>>u.y>>u.hp>>u.flag>>u.respawn_at;
 Planner planner;planner.nav.init(g);planner.initialized=true;planner.prev=previous;planner.respawns(g);planner.forecast(g);planner.assign(g);auto initial=planner.saveRuntime();
 string last;unsigned long long checksum=0;int variations=0;string first;
 double elapsedTicks=0;double maximumTicks=0;
 for(int r=0;r<repetitions;r++){
  perf::enabled=false;planner.restoreRuntime(initial);
  clock_t began=std::clock();perf::last=began;perf::enabled=true;last=planner.decide(g);perf::flush();perf::enabled=false;
  double ticks=std::clock()-began;elapsedTicks+=ticks;maximumTicks=std::max(maximumTicks,ticks);
  if(r==0)first=last;else if(last!=first)variations++;
  for(unsigned char c:last)checksum=checksum*131+c;
 }
 std::cerr<<std::fixed<<std::setprecision(9)<<"{\"profile_enabled\":"<<(CTF_PROFILE?"true":"false")<<",\"cutoffs_disabled\":"<<(PERF_NO_CUTOFF?"true":"false")<<",\"lookahead_enabled\":"<<(CTF_LOOKAHEAD?"true":"false")<<",\"iterations\":"<<repetitions<<",\"cpu_seconds\":"<<elapsedTicks/CLOCKS_PER_SEC<<",\"max_iteration_cpu_seconds\":"<<maximumTicks/CLOCKS_PER_SEC<<",\"reply_variations\":"<<variations<<",\"checksum\":"<<checksum<<",\"nodes\":{\"root_decides\":"<<perf::rootDecides<<",\"future_decides\":"<<perf::futureDecides<<",\"move_triples\":"<<perf::moveTriples<<",\"valid_move_triples\":"<<perf::validTriples<<",\"collision_cases\":"<<perf::collisionCases<<",\"shot_dp_subsets\":"<<perf::shotDpSubsets<<",\"refine_candidates\":"<<perf::refineCandidates<<",\"two_turn_scenarios\":"<<perf::scenarioPairs<<"},\"stages\":{";
 for(int i=0;i<perf::COUNT;i++){if(i)std::cerr<<',';auto&m=perf::measures[i];std::cerr<<'"'<<perf::names[i]<<"\":{\"cpu_seconds\":"<<m.ticks/CLOCKS_PER_SEC<<",\"calls\":"<<m.calls<<",\"root_cpu_seconds\":"<<m.rootTicks/CLOCKS_PER_SEC<<",\"future_cpu_seconds\":"<<m.futureTicks/CLOCKS_PER_SEC<<",\"root_calls\":"<<m.rootCalls<<",\"future_calls\":"<<m.futureCalls<<",\"allocations\":"<<m.allocations<<",\"requested_allocation_bytes\":"<<m.allocation_bytes<<'}';}
 std::cerr<<"}}\n";std::cout<<last<<'\n';return 0;
}
