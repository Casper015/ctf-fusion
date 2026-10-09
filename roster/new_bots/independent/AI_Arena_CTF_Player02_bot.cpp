#include <bits/stdc++.h>
using namespace std;

struct Unit { int id, team, x, y, hp, flag, respawn_at; };
struct Flag { int id; string status; int x, y, holder, return_at; };
struct Event { string type; vector<int> args; };
struct Game {
    int teams=0, me=0, size=0, turns=0, hp=0, damage=0, respawn=0, flag_return=0, flag_cooldown=0, turn=0;
    vector<string> map;
    vector<pair<int,int>> bases, spots;
    vector<int> score;
    vector<Unit> units;
    vector<Flag> flags;
    vector<Event> events;
};
bool read_turn(Game& g) {
    string w;
    while (cin >> w) {
        if (w=="INIT") cin >> g.teams >> g.me >> g.size >> g.turns >> g.hp >> g.damage >> g.respawn >> g.flag_return >> g.flag_cooldown;
        else if (w=="MAP") { g.map.assign(g.size,""); for (auto& r:g.map) cin >> r; }
        else if (w=="BASES") { int n; cin>>n; g.bases.resize(n); for(auto& p:g.bases)cin>>p.first>>p.second; }
        else if (w=="SPOTS") { int n; cin>>n; g.spots.resize(n); for(auto& p:g.spots)cin>>p.first>>p.second; }
        else if (w=="TURN") cin>>g.turn;
        else if (w=="SCORE") { g.score.resize(g.teams); for(int& s:g.score)cin>>s; }
        else if (w=="UNITS") { int n;cin>>n;g.units.resize(n);for(auto& u:g.units){string tag;cin>>tag>>u.id>>u.team>>u.x>>u.y>>u.hp>>u.flag>>u.respawn_at;} }
        else if (w=="FLAGS") { int n;cin>>n;g.flags.resize(n);for(auto& f:g.flags){string tag;cin>>tag>>f.id>>f.status>>f.x>>f.y>>f.holder>>f.return_at;} }
        else if (w=="EVENTS") { int n;cin>>n;g.events.assign(n,{});string line;getline(cin,line);for(auto& e:g.events){getline(cin,line);istringstream in(line);string tag;in>>tag>>e.type;for(int v;in>>v;)e.args.push_back(v);} }
        else if (w=="END") return true;
    }
    return false;
}

struct Planner {
    const Game& g; int n, me, sz;
    vector<int> baseD;
    vector<vector<int>> flagD;
    vector<pair<int,int>> pos;
    vector<char> active, immune;
    vector<int> hp, carrying;
    vector<pair<int,int>> goals;
    vector<int> role; // 0 idle/fighter, 1 flag runner, 2 carrier
    vector<int> goalD;
    vector<int> flagTargets;
    vector<int> spawnIndex;
    vector<int> outMoves;
    vector<int> outTargets;
    vector<string> outAct;
    vector<pair<int,int>> mv={{0,0},{0,-1},{0,1},{-1,0},{1,0}}; // S U D L R
    bool inside(int x,int y) const { return x>=0&&y>=0&&x<sz&&y<sz; }
    bool wall(int x,int y) const { return !inside(x,y)||g.map[y][x]=='#'; }
    int idx(int x,int y) const { return y*sz+x; }
    bool canAttack(pair<int,int> a,pair<int,int> b) const {
        int dx=b.first-a.first,dy=b.second-a.second;
        if(abs(dx)+abs(dy)>2)return false;
        if(abs(dx)==2&&dy==0)return !wall(a.first+dx/2,a.second);
        if(abs(dy)==2&&dx==0)return !wall(a.first,a.second+dy/2);
        if(abs(dx)==1&&abs(dy)==1)return !wall(a.first+dx,a.second)||!wall(a.first,a.second+dy);
        return true;
    }
    vector<int> bfs(const vector<pair<int,int>>& src) const {
        vector<int>d(sz*sz,1e6);queue<int>q;
        for(auto [x,y]:src)if(!wall(x,y)&&d[idx(x,y)]>0){d[idx(x,y)]=0;q.push(idx(x,y));}
        const int dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};
        while(!q.empty()){int a=q.front();q.pop();int x=a%sz,y=a/sz;for(int k=0;k<4;k++){int X=x+dx[k],Y=y+dy[k],b=idx(X,Y);if(!wall(X,Y)&&d[b]>d[a]+1){d[b]=d[a]+1;q.push(b);}}}
        return d;
    }
    vector<pair<int,int>> baseCells(int t) const { vector<pair<int,int>> r;auto [x,y]=g.bases[t];for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)r.push_back({x+dx,y+dy});return r; }
    pair<int,int> estimateSpawn(int uid, vector<char>& occ) const {
        int t=uid/3; auto cells=baseCells(t); auto c=g.bases[t];
        stable_sort(cells.begin(),cells.end(),[&](auto a,auto b){int da=abs(a.first-sz/2)+abs(a.second-sz/2),db=abs(b.first-sz/2)+abs(b.second-sz/2);if(da!=db)return da<db;int ea=abs(a.first-c.first)+2*abs(a.second-c.second),eb=abs(b.first-c.first)+2*abs(b.second-c.second);return ea<eb;});
        for(auto p:cells)if(!occ[idx(p.first,p.second)]&&!wall(p.first,p.second)){occ[idx(p.first,p.second)]=1;return p;}
        return c;
    }
    Planner(const Game& gg):g(gg),n(gg.teams),me(gg.me),sz(gg.size){}

    string decide() {
        const int total=(int)g.units.size();
        pos.assign(total,{-1,-1});active.assign(total,0);immune.assign(total,0);hp.assign(total,0);carrying.assign(total,-1);
        vector<char> occupied(sz*sz,0);
        for(auto& u:g.units)if(u.x>=0&&u.y>=0){pos[u.id]={u.x,u.y};active[u.id]=1;hp[u.id]=u.hp;carrying[u.id]=u.flag;occupied[idx(u.x,u.y)]=1;}
        // Recreate the engine's deterministic spawn slots for units returning this turn.
        for(int id=0;id<total;id++){auto&u=g.units[id];if(u.team==me&&!active[id]&&u.respawn_at==g.turn){pos[id]=estimateSpawn(id,occupied);active[id]=1;immune[id]=1;hp[id]=100;}}
        vector<pair<int,int>> base=baseCells(me);baseD=bfs(base);
        goals.assign(total,{-1,-1});role.assign(total,0);goalD.assign(total,-1);flagTargets.assign(total,-1);
        // Available ground flags are grouped by tile (one pickup command can take one per turn).
        vector<pair<int,int>> targets; vector<int> targetFlag;
        for(auto& f:g.flags)if((f.status=="home"||f.status=="dropped")&&f.x>=0&&f.y>=0&&!(f.status=="dropped"&&f.return_at<=g.turn)){
            pair<int,int> p={f.x,f.y};auto it=find(targets.begin(),targets.end(),p);
            if(it==targets.end()){targets.push_back(p);targetFlag.push_back(f.id);}
            else targetFlag[it-targets.begin()]=min(targetFlag[it-targets.begin()],f.id);
        }
        flagD.clear();for(auto p:targets)flagD.push_back(bfs({p}));
        vector<int> ours;for(int k=0;k<3;k++){int id=3*me+k;if(active[id])ours.push_back(id);}
        // Globally assign distinct flags to runners, minimizing pickup + return travel.
        double best=1e100; vector<int> bestAssign(3,-1), cur(3,-1); vector<char> used(targets.size(),0);
        function<void(int,double)> rec=[&](int k,double cost){
            if(k==3){if(cost<best){best=cost;bestAssign=cur;}return;}
            int id=3*me+k;
            if(!active[id]||carrying[id]>=0){cur[k]=-1;rec(k+1,cost);return;}
            // Leave a runner idle only when there are fewer distinct flag tiles than runners.
            cur[k]=-1;rec(k+1,cost+100.0);
            for(int j=0;j<(int)targets.size();j++)if(!used[j]){int d=flagD[j][idx(pos[id].first,pos[id].second)],back=baseD[idx(targets[j].first,targets[j].second)];if(d>=1e6||back>=1e6)continue;used[j]=1;cur[k]=j;double urgency=0;for(auto&f:g.flags)if(f.x==targets[j].first&&f.y==targets[j].second&&f.status=="dropped")urgency-=1.5;rec(k+1,cost+d+0.72*back+urgency);used[j]=0;}
        };
        rec(0,0);
        for(int k=0;k<3;k++){int id=3*me+k;if(!active[id])continue;if(carrying[id]>=0){role[id]=2;goals[id]={-2,-2};goalD[id]=baseD[idx(pos[id].first,pos[id].second)];}else if(bestAssign[k]>=0){int j=bestAssign[k];role[id]=1;flagTargets[id]=j;goals[id]=targets[j];goalD[id]=flagD[j][idx(pos[id].first,pos[id].second)];}}
        // Unassigned agents hunt enemy carriers or screen our carrier; otherwise pressure nearby opponents.
        for(int k=0;k<3;k++){int id=3*me+k;if(!active[id]||role[id])continue;
            int chosen=-1;double val=-1e9;
            for(auto&u:g.units)if(u.team!=me&&u.x>=0&&u.y>=0&&u.respawn_at<0){double d=abs(pos[id].first-u.x)+abs(pos[id].second-u.y);double v=(u.flag>=0?28.0:0.0)+(100-u.hp)*0.045-d*0.8;
                // Prefer foes threatening our runners or own base.
                for(int q=0;q<3;q++){int aid=3*me+q;if(active[aid]&&(role[aid]==1||role[aid]==2)&&abs(pos[aid].first-u.x)+abs(pos[aid].second-u.y)<=5)v+=3.0;}
                if(v>val){val=v;chosen=u.id;}}
            if(chosen>=0){auto&u=g.units[chosen];goals[id]={u.x,u.y};goalD[id]=bfs({goals[id]})[idx(pos[id].first,pos[id].second)];}
        }
        // BFS fields for fighters' selected targets.
        vector<vector<int>> fighterD(total);
        for(int id:ours)if(role[id]==0&&goals[id].first>=0)fighterD[id]=bfs({goals[id]});
        outMoves.assign(total,0);outTargets.assign(total,-1);outAct.assign(total,"-");
        // Enumerate all friendly move combinations, resolving allied blocking exactly and treating enemy cells conservatively.
        double bestScore=-1e100; vector<int> bestFinal(3),bestMove(3),bestAttack(3,-1);
        int ids[3]={3*me,3*me+1,3*me+2};
        vector<pair<int,int>> enemies;for(auto&u:g.units)if(u.team!=me&&u.x>=0&&u.y>=0)enemies.push_back({u.x,u.y});
        vector<pair<int,int>> enemyPred(total,{-1,-1});
        vector<vector<int>> enemyBaseD(n);
        for(auto&u:g.units)if(u.team!=me&&u.x>=0&&u.y>=0){
            pair<int,int> ep={u.x,u.y};enemyPred[u.id]=ep;
            if(u.flag>=0){if(enemyBaseD[u.team].empty())enemyBaseD[u.team]=bfs(baseCells(u.team));int curd=enemyBaseD[u.team][idx(ep.first,ep.second)];int dx[5]={0,0,0,-1,1},dy[5]={0,-1,1,0,0};for(int m=0;m<5;m++){int x=ep.first+dx[m],y=ep.second+dy[m];if(!wall(x,y)&&enemyBaseD[u.team][idx(x,y)]<curd){curd=enemyBaseD[u.team][idx(x,y)];enemyPred[u.id]={x,y};}}}
        }
        for(int a=0;a<5;a++)for(int b=0;b<5;b++)for(int c=0;c<5;c++){
            int choice[3]={a,b,c};pair<int,int> want[3],start[3],fin[3];bool blocked[3]={};
            for(int k=0;k<3;k++){int id=ids[k];if(!active[id]){start[k]=want[k]={-100,-100};blocked[k]=true;continue;}start[k]=pos[id];int x=start[k].first+mv[choice[k]].first,y=start[k].second+mv[choice[k]].second;if(wall(x,y)){x=start[k].first;y=start[k].second;}want[k]={x,y};
                bool ext=false;for(auto e:enemies)if(e==want[k])ext=true;if(ext)blocked[k]=true;}
            for(int i=0;i<3;i++)if(active[ids[i]])for(int j=i+1;j<3;j++)if(active[ids[j]]&&want[i]==want[j])blocked[i]=blocked[j]=true;
            bool changed=true;while(changed){changed=false;for(int i=0;i<3;i++)if(active[ids[i]]&&!blocked[i]&&want[i]!=start[i])for(int j=0;j<3;j++)if(i!=j&&active[ids[j]]&&want[i]==start[j]&&(blocked[j]||want[j]==start[j])){blocked[i]=true;changed=true;break;}}
            for(int k=0;k<3;k++)fin[k]=active[ids[k]]?(blocked[k]?start[k]:want[k]):pair<int,int>{-100,-100};
            double sc=0;
            for(int k=0;k<3;k++){int id=ids[k];if(!active[id])continue;auto p=fin[k];int cell=idx(p.first,p.second);
                if(role[id]==2)sc-=6.0*baseD[cell];
                else if(role[id]==1){int j=flagTargets[id];sc-=5.2*flagD[j][cell];if(p==goals[id])sc+=18.0;}
                else if(!fighterD[id].empty())sc-=1.35*fighterD[id][cell];
                // Stay close enough to fight, but avoid being surrounded. Carrying a flag is especially costly.
                for(auto&u:g.units)if(u.team!=me&&u.x>=0&&u.y>=0&&u.respawn_at!=g.turn){int d=abs(p.first-u.x)+abs(p.second-u.y);bool clear=canAttack({u.x,u.y},p);if(d<=2&&clear)sc-=(carrying[id]>=0?7.0:2.5)*(hp[id]<=34?1.7:1.0);else if(d==3)sc-=(carrying[id]>=0?1.7:0.45);}
                if(role[id]==2&&baseD[cell]==0)sc+=35.0;
            }
            // Greedy coordinated focus fire using estimated enemy next-step locations.
            double attackScore=0;int attack[3]={-1,-1,-1};vector<int> remHp(total,0);for(auto&u:g.units)remHp[u.id]=u.hp;
            for(int it=0;it<3;it++){double bs=0;int bk=-1,bt=-1;for(int k=0;k<3;k++){int id=ids[k];if(!active[id]||immune[id]||carrying[id]>=0||attack[k]>=0)continue;
                    for(auto&u:g.units)if(u.team!=me&&u.x>=0&&u.y>=0&&u.respawn_at!=g.turn&&remHp[u.id]>0){
                        pair<int,int> ep={u.x,u.y};double prob=0;
                        if(canAttack(fin[k],ep))prob=max(prob,0.42);
                        // Predict a carrier's one-step retreat toward its own base.
                        pair<int,int> pred=enemyPred[u.id];
                        if(pred!=ep&&canAttack(fin[k],pred))prob=max(prob,0.58);
                        if(prob<=0)continue;
                        bool kill=remHp[u.id]<=34;double reward=2.1+(kill?(u.flag>=0?18.0:8.0):0.0);reward*=prob;
                        if(reward>bs){bs=reward;bk=k;bt=u.id;}
                    }
                }if(bk<0)break;attack[bk]=bt;remHp[bt]-=34;attackScore+=bs;}
            sc+=attackScore;
            if(sc>bestScore){bestScore=sc;for(int k=0;k<3;k++){bestMove[k]=choice[k];bestFinal[k]=active[ids[k]]?idx(fin[k].first,fin[k].second):-1;bestAttack[k]=attack[k];}}
        }
        for(int k=0;k<3;k++){int id=ids[k];if(!active[id])continue;outMoves[id]=bestMove[k];if(carrying[id]<0){for(auto&f:g.flags)if((f.status=="home"||f.status=="dropped")&&f.x==bestFinal[k]%sz&&f.y==bestFinal[k]/sz&&!(f.status=="dropped"&&f.return_at<=g.turn)){outAct[id]="P";break;}if(outAct[id]!="P"&&bestAttack[k]>=0&&!immune[id]){outTargets[id]=bestAttack[k];outAct[id]=to_string(bestAttack[k]);}}
        }
        string out=to_string(g.turn);
        const string movec[5]={"S","U","D","L","R"};
        for(int k=0;k<3;k++){int id=ids[k];out+=" "+movec[outMoves[id]]+" "+outAct[id];}
        return out;
    }
};
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);Game g;while(read_turn(g)){Planner p(g);cout<<p.decide()<<'\n';cout.flush();}}
