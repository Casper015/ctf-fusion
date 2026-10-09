#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
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
    int teams=0, me=0, size=0, turns=0, hp=0, damage=0, respawn=0, flag_return=0, flag_cooldown=0;
    vector<string> map; vector<pair<int,int>> bases, spots; int turn=0;
    vector<int> score; vector<Unit> units; vector<Flag> flags; vector<Event> events;
};

bool read_turn(Game& g) {
    string w;
    while (cin >> w) {
        if (w=="INIT") cin>>g.teams>>g.me>>g.size>>g.turns>>g.hp>>g.damage>>g.respawn>>g.flag_return>>g.flag_cooldown;
        else if (w=="MAP") { g.map.assign(g.size,""); for(auto& r:g.map) cin>>r; }
        else if (w=="BASES") { int n;cin>>n;g.bases.resize(n);for(auto& p:g.bases)cin>>p.first>>p.second; }
        else if (w=="SPOTS") { int n;cin>>n;g.spots.resize(n);for(auto& p:g.spots)cin>>p.first>>p.second; }
        else if (w=="TURN") cin>>g.turn;
        else if (w=="SCORE") { g.score.resize(g.teams);for(int& x:g.score)cin>>x; }
        else if (w=="UNITS") { int n;cin>>n;g.units.resize(n);for(auto& u:g.units){string tag;cin>>tag>>u.id>>u.team>>u.x>>u.y>>u.hp>>u.flag>>u.respawn_at;} }
        else if (w=="FLAGS") { int n;cin>>n;g.flags.resize(n);for(auto& f:g.flags){string tag;cin>>tag>>f.id>>f.status>>f.x>>f.y>>f.holder>>f.return_at;} }
        else if (w=="EVENTS") { int n;cin>>n;g.events.assign(n,{});string line;getline(cin,line);for(auto& e:g.events){getline(cin,line);istringstream in(line);string tag;in>>tag>>e.type;for(int v;in>>v;)e.args.push_back(v);} }
        else if (w=="END") return true;
    }
    return false;
}

struct P { int x,y; bool operator==(const P& o)const{return x==o.x&&y==o.y;} };
static const int dx[5]={0,0,0,-1,1}, dy[5]={0,-1,1,0,0};
static const char* mv[5]={"S","U","D","L","R"};
int md(P a,P b){return abs(a.x-b.x)+abs(a.y-b.y);}
bool inside(const Game& g,P p){return p.x>=0&&p.y>=0&&p.x<g.size&&p.y<g.size;}
bool wall(const Game& g,P p){return !inside(g,p)||g.map[p.y][p.x]=='#';}
int ix(const Game& g,P p){return p.y*g.size+p.x;}
P pt(const Game& g,int i){return {i%g.size,i/g.size};}
bool can_attack(const Game& g,P a,P b){
    int x=b.x-a.x,y=b.y-a.y;if(abs(x)+abs(y)>2)return false;
    if(abs(x)==2&&y==0&&wall(g,{a.x+x/2,a.y}))return false;
    if(abs(y)==2&&x==0&&wall(g,{a.x,a.y+y/2}))return false;
    if(abs(x)==1&&abs(y)==1&&wall(g,{a.x+x,a.y})&&wall(g,{a.x,a.y+y}))return false;
    return true;
}
vector<int> bfs(const Game& g,const vector<P>& goals){
    int n=g.size*g.size;vector<int>d(n,1e8);queue<int>q;
    for(P p:goals)if(!wall(g,p)&&d[ix(g,p)]>0){d[ix(g,p)]=0;q.push(ix(g,p));}
    while(!q.empty()){int z=q.front();q.pop();P p=pt(g,z);for(int k=1;k<=4;k++){P a{p.x+dx[k],p.y+dy[k]};if(wall(g,a))continue;int j=ix(g,a);if(d[j]>d[z]+1){d[j]=d[z]+1;q.push(j);}}}
    return d;
}
struct Goal { int kind=0, id=-1; vector<P> cells; double utility=0; vector<int> d; };

P predicted_spawn(const Game& g,int team,const vector<P>& occupied){
    auto [bx,by]=g.bases[team];P c{g.size/2,g.size/2};vector<P> cells;
    for(int y=by-1;y<=by+1;y++)for(int x=bx-1;x<=bx+1;x++)cells.push_back({x,y});
    stable_sort(cells.begin(),cells.end(),[&](P a,P b){return tuple(md(a,c),abs(a.x-bx)+2*abs(a.y-by))<tuple(md(b,c),abs(b.x-bx)+2*abs(b.y-by));});
    for(P p:cells){bool used=false;for(P q:occupied)if(p==q)used=true;if(!used)return p;}
    return cells.front();
}

string decide(const Game& g){
    int n=g.teams*3,N=g.size*g.size;vector<P> pos(n,{-1,-1});vector<int> active(n,0), newborn(n,0);
    vector<P> occupied;for(const auto& u:g.units)if(u.x>=0)occupied.push_back({u.x,u.y});
    // Respawning units can move and pick up this turn; predict the engine's first free spawn square.
    for(int i=0;i<n;i++){const auto&u=g.units[i];if(u.x>=0){pos[i]={u.x,u.y};active[i]=1;}else if(u.team==g.me&&u.respawn_at==g.turn){P p=predicted_spawn(g,g.me,occupied);pos[i]=p;active[i]=1;newborn[i]=1;occupied.push_back(p);}}
    vector<int> ours={3*g.me,3*g.me+1,3*g.me+2};
    vector<P> baseCells;auto [bx,by]=g.bases[g.me];for(int y=by-1;y<=by+1;y++)for(int x=bx-1;x<=bx+1;x++)baseCells.push_back({x,y});
    vector<Goal> goals(3);vector<int> hunters;
    for(int k=0;k<3;k++){
        int id=ours[k];const Unit&u=g.units[id];
        if(!active[id])continue;
        if(u.flag>=0){goals[k].kind=1;goals[k].cells=baseCells;goals[k].d=bfs(g,baseCells);}
        else hunters.push_back(k);
    }
    vector<Goal> tasks;
    for(const auto&f:g.flags)if((f.status=="home"||f.status=="dropped")&&f.x>=0){
        Goal z;z.kind=2;z.id=f.id;z.cells={{f.x,f.y}};z.d=bfs(g,z.cells);
        tasks.push_back(std::move(z));
    }
    // Intercept enemy flag carriers that are approaching a scoring base. Goal cells
    // are all legal firing positions, so the hunter arrives ready to shoot.
    for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&e.flag>=0){
        Goal z;z.kind=5;z.id=e.id;P ep{e.x,e.y};
        for(int y=max(1,ep.y-2);y<=min(g.size-2,ep.y+2);y++)for(int x=max(1,ep.x-2);x<=min(g.size-2,ep.x+2);x++)if(!wall(g,{x,y})&&can_attack(g,{x,y},ep))z.cells.push_back({x,y});
        if(!z.cells.empty()){z.d=bfs(g,z.cells);tasks.push_back(std::move(z));}
    }
    // A carrier scores only after reaching its own base. Include the return trip in flag value.
    vector<int> baseDist=bfs(g,baseCells);
    // Recompute task value per hunter, since distance from each spawn differs.
    vector<vector<double>> util(hunters.size(),vector<double>(tasks.size()+g.spots.size(),-1e6));
    vector<vector<Goal>> patrol(hunters.size());
    for(size_t h=0;h<hunters.size();h++){
        int k=hunters[h],id=ours[k];P p=pos[id];
        for(size_t j=0;j<tasks.size();j++){
            Goal&z=tasks[j];double v;
            if(z.kind==2){
                v=61.0-0.95*z.d[ix(g,p)]-0.85*baseDist[ix(g,z.cells[0])];
                const Flag&fl=g.flags[z.id];if(fl.status=="dropped")v+=1.8;
                if(fl.status=="dropped"&&fl.return_at>0&&fl.return_at-g.turn<z.d[ix(g,p)])v-=25;
                int contest=0;for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&md({e.x,e.y},z.cells[0])<=2)contest++;
                v-=min(5,contest)*1.1;
            }else{
                const Unit&e=g.units[z.id];P ep{e.x,e.y};auto [ex,ey]=g.bases[e.team];
                int home=max(0,abs(ep.x-ex)-1)+max(0,abs(ep.y-ey)-1);int leader=*max_element(g.score.begin(),g.score.end());
                v=26.0+1.7*max(0,14-home)+(g.score[e.team]>=leader-1?8.0:0.0)-0.95*z.d[ix(g,p)];
            }
            util[h][j]=v;
        }
        for(size_t s=0;s<g.spots.size();s++){
            Goal z;z.kind=3;z.id=(int)s;z.cells={{g.spots[s].first,g.spots[s].second}};z.d=bfs(g,z.cells);patrol[h].push_back(z);
            util[h][tasks.size()+s]=3.0-0.22*z.d[ix(g,p)]+(s+1==g.spots.size()?1.0:0.0);
        }
    }
    // Exact tiny assignment search: hunters take distinct live flags; spare units cover distinct respawn spots.
    double best=-1e100;vector<int> choice(hunters.size(),-1),bestChoice=choice;vector<int>used(tasks.size()+g.spots.size(),0);
    function<void(int,double)> assign=[&](int h,double val){if(h==(int)hunters.size()){if(val>best){best=val;bestChoice=choice;}return;}
        for(int j=0;j<(int)used.size();j++)if(!used[j]&&util[h][j]>-100){used[j]=1;choice[h]=j;assign(h+1,val+util[h][j]);used[j]=0;}
    };
    if(!hunters.empty())assign(0,0);
    for(size_t h=0;h<hunters.size();h++)if(bestChoice[h]>=0){int j=bestChoice[h],k=hunters[h];if(j<(int)tasks.size())goals[k]=tasks[j];else goals[k]=patrol[h][j-(int)tasks.size()];}

    // When a carrier is still far from home or has entered enemy range, send its
    // nearest free teammate alongside it. The escort can attack while the carrier cannot.
    int carrierK=-1,escortK=-1;double carrierThreat=-1;
    for(int k=0;k<3;k++)if(active[ours[k]]&&g.units[ours[k]].flag>=0){
        P cp=pos[ours[k]];int close=0;for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&md({e.x,e.y},cp)<=4)close++;
        double threat=close*20.0+(g.units[ours[k]].hp<70?10.0:0.0)+0.25*baseDist[ix(g,cp)];
        if(threat>carrierThreat){carrierThreat=threat;carrierK=k;}
    }
    if(carrierK>=0&&hunters.size()>0){
        int cid=ours[carrierK];bool threatened=baseDist[ix(g,pos[cid])]>8;
        for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&md({e.x,e.y},pos[cid])<=4)threatened=true;
        if(threatened){int bestD=1e9;for(int k:hunters)if(active[ours[k]]){int d=md(pos[ours[k]],pos[cid]);if(d<bestD){bestD=d;escortK=k;}}
            if(escortK>=0){Goal z;z.kind=4;z.id=cid;for(int y=max(1,pos[cid].y-2);y<=min(g.size-2,pos[cid].y+2);y++)for(int x=max(1,pos[cid].x-2);x<=min(g.size-2,pos[cid].x+2);x++)if(abs(x-pos[cid].x)+abs(y-pos[cid].y)<=2&&!wall(g,{x,y}))z.cells.push_back({x,y});z.d=bfs(g,z.cells);goals[escortK]=move(z);}
        }
    }

    // Estimate enemy fire zones. A tile attackable now or after one enemy step is dangerous;
    // carrying a flag and low health sharply increase the cost.
    vector<vector<double>> risk(3,vector<double>(N,0));
    for(int k=0;k<3;k++)if(active[ours[k]]){
        const Unit&u=g.units[ours[k]];double mult=(u.flag>=0?2.2:1.0)*(u.hp<=34?1.7:(u.hp<=68?1.3:1.0));
        for(int c=0;c<N;c++){P q=pt(g,c);if(wall(g,q))continue;double r=0;
            for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&e.hp>0){P ep{e.x,e.y};int d=md(ep,q);if(d>3)continue;int hits=0,total=0;
                for(int m=0;m<5;m++){P v{ep.x+dx[m],ep.y+dy[m]};if(wall(g,v))v=ep;total++;if(can_attack(g,v,q))hits++;}
                if(hits)r+=(double)hits/total*(d<=2?1.2:0.55);
            }
            risk[k][c]=r*mult;
        }
    }
    // Danger-weighted shortest paths to each unit's strategic destination.
    vector<vector<double>> path(3,vector<double>(N,1e9));
    for(int k=0;k<3;k++)if(active[ours[k]]&&!goals[k].cells.empty()){
        using Q=pair<double,int>;priority_queue<Q,vector<Q>,greater<Q>>pq;
        for(P p:goals[k].cells)if(!wall(g,p)){int z=ix(g,p);path[k][z]=0;pq.push({0,z});}
        while(!pq.empty()){auto [d,z]=pq.top();pq.pop();if(d!=path[k][z])continue;P p=pt(g,z);
            for(int m=1;m<=4;m++){P a{p.x+dx[m],p.y+dy[m]};if(wall(g,a))continue;int j=ix(g,a);double nd=d+1.0+0.12*risk[k][j];if(nd<path[k][j]){path[k][j]=nd;pq.push({nd,j});}}
        }
    }
    // Jointly pick three moves to prevent friendly collisions and let carriers reach base squares.
    int bestm[3]={0,0,0};double bestMove=1e100;P dest[3][5];double val[3][5];
    for(int k=0;k<3;k++)for(int m=0;m<5;m++){
        int id=ours[k];P q=pos[id];if(active[id]){P t{q.x+dx[m],q.y+dy[m]};if(!wall(g,t))q=t;else m==0?void():void();}
        dest[k][m]=q;
        if(!active[id])val[k][m]=0;
        else {double d=path[k][ix(g,q)];val[k][m]=d+0.65*risk[k][ix(g,q)];
            for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&e.x==q.x&&e.y==q.y)val[k][m]+=1.2;
        }
    }
    for(int a=0;a<5;a++)for(int b=0;b<5;b++)for(int c=0;c<5;c++){
        int mm[3]={a,b,c};double v=val[0][a]+val[1][b]+val[2][c];
        for(int i=0;i<3;i++)if(active[ours[i]])for(int j=i+1;j<3;j++)if(active[ours[j]]){
            if(dest[i][mm[i]]==dest[j][mm[j]])v+=80;
            if(dest[i][mm[i]]==pos[ours[j]]&&dest[j][mm[j]]==pos[ours[j]])v+=12;
            if(dest[j][mm[j]]==pos[ours[i]]&&dest[i][mm[i]]==pos[ours[i]])v+=12;
        }
        if(v<bestMove){bestMove=v;copy(mm,mm+3,bestm);}
    }

    // Attack selection models uncertain enemy movement, then searches all focus-fire assignments.
    vector<int> shooters;vector<vector<double>> prob(3,vector<double>(n,0));
    for(int k=0;k<3;k++){int id=ours[k];if(!active[id]||g.units[id].flag>=0||newborn[id])continue;
        bool pickup=false;for(const auto&f:g.flags)if((f.status=="home"||f.status=="dropped")&&f.x==dest[k][bestm[k]].x&&f.y==dest[k][bestm[k]].y)pickup=true;
        if(pickup)continue;shooters.push_back(k);
        P from=dest[k][bestm[k]];
        for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&e.hp>0&&e.respawn_at!=g.turn){
            P ep{e.x,e.y};int hits=0,total=0;
            for(int m=0;m<5;m++){P v{ep.x+dx[m],ep.y+dy[m]};if(wall(g,v))v=ep;total++;if(can_attack(g,from,v))hits++;}
            prob[k][e.id]=(double)hits/total;
        }
    }
    // Precompute each enemy's value for every subset of our three attackers. The exact
    // assignment search then updates only the target receiving an added shot.
    vector<array<double,8>> attackValue(n);vector<vector<int>> attackTargets(3);
    for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&e.hp>0){
        for(int mask=1;mask<8;mask++){
            double dp[4]={1,0,0,0};int cnt=0;
            for(int k=0;k<3;k++)if((mask&(1<<k))&&prob[k][e.id]>0.01){double p=prob[k][e.id];for(int j=cnt+1;j>=0;j--){double old=(j<=cnt?dp[j]:0);dp[j]=old*(1-p)+(j?dp[j-1]*p:0);}cnt++;}
            int need=(e.hp+33)/34;double kill=0,ed=0;for(int j=0;j<=cnt;j++){ed+=dp[j]*min(e.hp,34*j);if(j>=need)kill+=dp[j];}
            attackValue[e.id][mask]=0.25*ed+kill*(25+(e.flag>=0?65:0));
        }
    }
    for(int k:shooters)for(const auto&e:g.units)if(e.team!=g.me&&e.x>=0&&e.hp>0&&prob[k][e.id]>0.01)attackTargets[k].push_back(e.id);
    int assigned[3]={-1,-1,-1},bestAssigned[3]={-1,-1,-1};vector<int> hitMask(n,0);double bestAttack=-1;
    function<void(int,double)> dfs=[&](int z,double value){if(z==(int)shooters.size()){
            if(value>bestAttack){bestAttack=value;copy(assigned,assigned+3,bestAssigned);}return;}
        int k=shooters[z];assigned[k]=-1;dfs(z+1,value);
        for(int target:attackTargets[k]){int old=hitMask[target],now=old|(1<<k);assigned[k]=target;hitMask[target]=now;
            dfs(z+1,value+attackValue[target][now]-attackValue[target][old]);hitMask[target]=old;}
        assigned[k]=-1;
    };
    dfs(0,0);
    string out=to_string(g.turn);
    for(int k=0;k<3;k++){
        int id=ours[k];int m=bestm[k];bool pickup=false;
        if(active[id]&&g.units[id].flag<0)for(const auto&f:g.flags)if((f.status=="home"||f.status=="dropped")&&f.x==dest[k][m].x&&f.y==dest[k][m].y)pickup=true;
        out+=" ";out+=mv[m];out+=" ";
        if(pickup)out+="P";else if(bestAssigned[k]>=0)out+=to_string(bestAssigned[k]);else out+="-";
    }
    return out;
}
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);Game g;while(read_turn(g))cout<<decide(g)<<endl;}
