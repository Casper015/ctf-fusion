#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
using namespace std;

struct U { int id,team,x,y,hp,flag,respawn; };
struct F { int id; string state; int x,y,holder,ret; };
struct State {
 int n=0,me=0,s=0,t=0,total=400; vector<string> map; vector<pair<int,int>> bases,spots; vector<U> u; vector<F> f;
 bool read(){string w; while(cin>>w){
  if(w=="INIT"){int h,d,r,fr,fc;cin>>n>>me>>s>>total>>h>>d>>r>>fr>>fc;}
  else if(w=="MAP"){map.resize(s);for(auto &r:map)cin>>r;}
  else if(w=="BASES"){int k;cin>>k;bases.resize(k);for(auto &p:bases)cin>>p.first>>p.second;}
  else if(w=="SPOTS"){int k;cin>>k;spots.resize(k);for(auto &p:spots)cin>>p.first>>p.second;}
  else if(w=="TURN")cin>>t;
  else if(w=="SCORE"){int x;for(int k=0;k<n;k++)cin>>x;}
  else if(w=="UNITS"){int k;cin>>k;u.resize(k);for(auto &v:u){string q;cin>>q>>v.id>>v.team>>v.x>>v.y>>v.hp>>v.flag>>v.respawn;}}
  else if(w=="FLAGS"){int k;cin>>k;f.resize(k);for(auto &v:f){string q;cin>>q>>v.id>>v.state>>v.x>>v.y>>v.holder>>v.ret;}}
  else if(w=="EVENTS"){int k;cin>>k;string line;getline(cin,line);while(k--)getline(cin,line);}
  else if(w=="END")return true;
 }return false;}
};

struct Bot {
 const int dx[5]={0,0,0,-1,1},dy[5]={0,-1,1,0,0};
 const char mc[5]={'S','U','D','L','R'};
 int s=0,V=0; vector<short> dist; vector<array<int,5>> neigh; vector<int> oldpos;
 bool floor(const State &g,int x,int y){return x>=0&&y>=0&&x<g.s&&y<g.s&&g.map[y][x]!='#';}
 int cell(const U &v){return v.x<0?-1:v.y*s+v.x;}
 int md(int a,int b){return abs(a%s-b%s)+abs(a/s-b/s);}
 int dd(int a,int b){return a<0||b<0?30000:dist[a*V+b];}
 void init(const State &g){s=g.s;V=s*s;neigh.resize(V);dist.assign(V*V,30000);oldpos.assign(g.u.size(),-1);
  for(int p=0;p<V;p++)for(int m=0;m<5;m++){int x=p%s+dx[m],y=p/s+dy[m];neigh[p][m]=floor(g,x,y)?y*s+x:p;}
  vector<int> q(V);for(int src=0;src<V;src++){if(!floor(g,src%s,src/s))continue;int h=0,z=0;q[z++]=src;dist[src*V+src]=0;while(h<z){int p=q[h++];for(int m=1;m<5;m++){int v=neigh[p][m];if(dist[src*V+v]==30000){dist[src*V+v]=dist[src*V+p]+1;q[z++]=v;}}}}
 }
 bool hit(const State &g,int a,int b){if(a<0||b<0||md(a,b)>2)return false;int ax=a%s,ay=a/s,bx=b%s,by=b/s;int x=bx-ax,y=by-ay;
  if(abs(x)==2&&!y)return floor(g,ax+x/2,ay);if(abs(y)==2&&!x)return floor(g,ax,ay+y/2);if(abs(x)==1&&abs(y)==1)return floor(g,bx,ay)||floor(g,ax,by);return true;
 }
 int home(const State &g,int p,int team){auto [x,y]=g.bases[team];int best=p,d=30000;for(int yy=y-1;yy<=y+1;yy++)for(int xx=x-1;xx<=x+1;xx++){int q=yy*s+xx;if(dd(p,q)<d){d=dd(p,q);best=q;}}return best;}
 int spawn(const State &g,int team,vector<int>&taken){auto [x,y]=g.bases[team];vector<int> cells;for(int yy=y-1;yy<=y+1;yy++)for(int xx=x-1;xx<=x+1;xx++)cells.push_back(yy*s+xx);auto key=[&](int p){return pair<int,int>(md(p,(s/2)*s+s/2),abs(p%s-x)+2*abs(p/s-y));};stable_sort(cells.begin(),cells.end(),[&](int a,int b){return key(a)<key(b);});for(int p:cells)if(find(taken.begin(),taken.end(),p)==taken.end()){taken.push_back(p);return p;}return -1;}
 string decide(const State &g){if(!s)init(g);const int nu=g.u.size();vector<int> pos(nu),taken;for(int i=0;i<nu;i++){pos[i]=cell(g.u[i]);if(pos[i]>=0)taken.push_back(pos[i]);}for(int i=0;i<nu;i++)if(pos[i]<0&&g.u[i].respawn<=g.t)pos[i]=spawn(g,g.u[i].team,taken);
  array<int,3> id={3*g.me,3*g.me+1,3*g.me+2};vector<int> enemies;for(int j=0;j<nu;j++)if(g.u[j].team!=g.me&&pos[j]>=0)enemies.push_back(j);
  int active=0,center=0;for(int k:id)if(pos[k]>=0){active++;center+=pos[k];}int leader=id[0];for(int k:id)if(pos[k]>=0){leader=k;break;}
  // One squad chooses the closest useful flag; carrier protection overrides raids.
  int squadgoal=(s/2)*s+s/2;double bestgoal=-1e30;int carrier=-1;
  for(int k:id)if(pos[k]>=0&&g.u[k].flag>=0)carrier=k;
  if(carrier>=0)squadgoal=pos[carrier];else{
   for(const auto &fl:g.f)if(fl.x>=0&&(fl.ret<0||fl.ret>g.t)){int p=fl.y*s+fl.x;double d=0;for(int k:id)if(pos[k]>=0)d+=dd(pos[k],p);double val=35.-d/max(1,active)-0.35*dd(p,home(g,p,g.me));int near=0;for(int j:enemies)if(dd(pos[j],p)<5)near++;val-=1.5*near;if(val>bestgoal){bestgoal=val;squadgoal=p;}}
   for(int j:enemies)if(g.u[j].flag>=0){int p=pos[j];double val=23.-dd(pos[leader],p);if(val>bestgoal){bestgoal=val;squadgoal=p;}}
  }
  // Predict each enemy's most likely movement, retaining a stay alternative.
  vector<int> predicted=pos;vector<int> enemygoal(nu,-1);for(int j:enemies){int goal=pos[j];double val=-1e30;if(g.u[j].flag>=0)goal=home(g,pos[j],g.u[j].team);else{
    for(const auto &fl:g.f)if(fl.x>=0&&(fl.ret<0||fl.ret>g.t)){int p=fl.y*s+fl.x;double v=25.-dd(pos[j],p);if(v>val){val=v;goal=p;}}
    for(int k:id)if(pos[k]>=0){double v=(g.u[k].flag>=0?29.:17.)-dd(pos[j],pos[k]);if(v>val){val=v;goal=pos[k];}}
   }enemygoal[j]=goal;int best=pos[j];double bv=-1e30;for(int m=0;m<5;m++){int p=neigh[pos[j]][m];double v=-dd(p,goal);if(p==goal)v+=0.2;if(p==oldpos[j])v-=0.2;for(int k:id)if(pos[k]>=0&&hit(g,p,pos[k]))v+=g.u[k].flag>=0?0.3:0.12;if(v>bv){bv=v;best=p;}}predicted[j]=best;
  }
  vector<array<double,5>> prob(nu);
  for(int j:enemies){double total=0;array<double,5> values{};double best=-1e30;
   for(int m=0;m<5;m++){int p=neigh[pos[j]][m];double v=1.45*(dd(pos[j],enemygoal[j])-dd(p,enemygoal[j]));
    if(g.u[j].flag<0){double shotv=0,risk=0;for(int k:id)if(pos[k]>=0){if(hit(g,p,pos[k]))shotv=max(shotv,g.u[k].flag>=0?4.5:g.u[k].hp<=34?5.0:g.u[k].hp<=68?2.5:1.4);if(g.u[k].flag<0&&hit(g,p,pos[k]))risk+=1.;}
     v+=shotv-risk*(g.u[j].hp<=34?1.2:0.45);
    }
    if(oldpos[j]>=0&&p-pos[j]==pos[j]-oldpos[j]&&p!=pos[j])v+=0.35;
    if(p==pos[j])v-=0.25;values[m]=v;best=max(best,v);
   }
   for(int m=0;m<5;m++){prob[j][m]=exp((values[m]-best)/0.7);total+=prob[j][m];}
   for(int m=0;m<5;m++)prob[j][m]=0.92*prob[j][m]/total+0.08/5.;
  }
  auto reach=[&](int p,int j){double chance=0;for(int m=0;m<5;m++)chance+=prob[j][m]*hit(g,p,neigh[pos[j]][m]);return chance;};
  array<int,3> goal,chosen,acts;for(int a=0;a<3;a++){int k=id[a];goal[a]=pos[k]<0?squadgoal:(g.u[k].flag>=0?home(g,pos[k],g.me):squadgoal);chosen[a]=0;acts[a]=-1;}
  double top=-1e30;array<int,3> bestm={0,0,0},besta={-1,-1,-1};
  for(int ma=0;ma<5;ma++)for(int mb=0;mb<5;mb++)for(int mc0=0;mc0<5;mc0++){
   array<int,3> moves={ma,mb,mc0},p;bool legal=true;for(int a=0;a<3;a++)p[a]=pos[id[a]]<0?-1:neigh[pos[id[a]]][moves[a]];
   for(int a=0;a<3;a++)for(int b=0;b<a;b++)if(p[a]>=0&&p[a]==p[b])legal=false;if(!legal)continue;
   double val=0;for(int a=0;a<3;a++){int k=id[a];if(p[a]<0)continue;int actualhp=g.u[k].hp>0?g.u[k].hp:100;bool flag=g.u[k].flag>=0;val-=dd(p[a],goal[a])*(flag?2.4:0.7);if(p[a]==pos[k]&&dd(p[a],goal[a])>0)val-=0.16;
    for(int j:enemies){double incoming=0.;if(g.u[j].flag<0&&g.u[j].x>=0){incoming=reach(p[a],j);if(incoming>0){int nearest=30000;for(int b=0;b<3;b++)if(p[b]>=0)nearest=min(nearest,dd(predicted[j],p[b]));if(dd(predicted[j],p[a])>nearest&&g.u[k].flag<0)incoming*=0.55;}}val-=incoming*(flag?8.0:actualhp<=34?7.0:actualhp<=68?4.2:2.7);if(p[a]==predicted[j])val-=8.;if(p[a]==pos[j])val-=2.5;}
    if(flag&&dd(p[a],goal[a])==0)val+=60.;
    if(!flag){for(const auto &fl:g.f)if(fl.x>=0&&(fl.ret<0||fl.ret>g.t)&&fl.x+fl.y*s==p[a]){double danger=0;for(int j:enemies)if(g.u[j].flag<0)danger+=reach(p[a],j);val+=18.-3.*danger;}}
   }
   // Estimate combined expected outgoing damage, then value focus fire and kills.
   array<int,3> action={-1,-1,-1};array<double,3> aw={0,0,0};vector<double> damage(nu,0);
   for(int iter=0;iter<3;iter++){double av=-1e30;int ba=-1,bj=-1;double prob=0;for(int a=0;a<3;a++)if(action[a]<0&&p[a]>=0&&g.u[id[a]].flag<0&&g.u[id[a]].x>=0){for(int j:enemies)if(g.u[j].x>=0){double pr=reach(p[a],j);if(pr<=0)continue;double hp=max(0.,g.u[j].hp-damage[j]);double v=pr*(3.4+(g.u[j].flag>=0?4.0:0.0));if(hp<=34)v+=pr*9.;else if(hp<=68)v+=pr*2.;if(damage[j]>=g.u[j].hp)v-=6.;if(v>av){av=v;ba=a;bj=j;prob=pr;}}}if(ba>=0){action[ba]=bj;damage[bj]+=34*prob;aw[ba]=av;}}
   for(double v:aw)val+=v;
   // Stay close enough to mutually defend, without forcing escorts to stand in the way.
   for(int a=0;a<3;a++)for(int b=0;b<a;b++)if(p[a]>=0&&p[b]>=0&&g.u[id[a]].flag<0&&g.u[id[b]].flag<0)val-=0.23*max(0,dd(p[a],p[b])-3);
   if(carrier>=0)for(int a=0;a<3;a++)if(p[a]>=0&&id[a]!=carrier&&g.u[id[a]].flag<0){int pc=p[carrier-3*g.me];val-=0.38*max(0,dd(p[a],pc)-2);}
   if(val>top){top=val;bestm=moves;besta=action;}
  }
  string out=to_string(g.t);for(int a=0;a<3;a++){int k=id[a];string action=besta[a]>=0?to_string(besta[a]):"P";int p=pos[k]<0?-1:neigh[pos[k]][bestm[a]];for(const auto &fl:g.f)if(p>=0&&fl.x+fl.y*s==p&&fl.x>=0&&g.u[k].flag<0)action="P";out+=" ";out+=mc[bestm[a]];out+=" "+action;}oldpos=pos;return out;
 }
};

int main(){ios::sync_with_stdio(false);cin.tie(nullptr);State g;Bot bot;while(g.read())cout<<bot.decide(g)<<endl;}
