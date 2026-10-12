// T0004 reference-team geometry on retained observations. No engine step.
// One int32 record per (turn,team): turn,team,d_reference,d_flag,d_carrier,
// potential_two_turn_contact,opponent_carrier,reference_carrier,
// opponent_active_units,reference_active_units.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include "stream.hpp"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
int main() {
#ifdef _WIN32
    _setmode(_fileno(stdout),_O_BINARY);
#endif
    fusion::Static init;if(!fusion::readInit(std::cin,init)) return 1;
    fusion::Board board;board.init(init);const int cells=board.n*board.n;
    std::vector<std::vector<int>> dist(cells),reach(cells);
    for(int c=0;c<cells;c++) if(board.open(c)) {
        board.bfs(&c,1,dist[c]);
        for(int p=0;p<cells;p++) if(dist[c][p]<=2) reach[c].push_back(p);
    }
    auto closest=[&](const std::vector<int>& a,const std::vector<int>& b) {
        int d=fusion::kUnreachable;
        for(int x:a) for(int y:b) d=std::min(d,dist[x][y]);
        return d;
    };
    fusion::State state;
    while(fusion::readTurn(std::cin,state)) {
        std::vector<std::vector<int>> teams(init.teams);std::vector<char> carrying(init.teams,0);
        std::vector<int> flags,carriers;
        for(const auto& u:state.units) if(u.x>=0) {
            int c=board.cell(u.x,u.y);teams[u.team].push_back(c);
            if(u.flag>=0) {carrying[u.team]=1;carriers.push_back(c);}
        }
        for(const auto& f:state.flags) {
            if(f.status==fusion::kCooldown) continue;
            if(f.status==fusion::kCarried) {
                if(f.holder>=0 && f.holder<(int)state.units.size()) {
                    const auto& u=state.units[f.holder];if(u.x>=0) flags.push_back(board.cell(u.x,u.y));
                }
            } else if(f.x>=0) flags.push_back(board.cell(f.x,f.y));
        }
        for(int t=0;t<init.teams;t++) {
            const auto& me=teams[init.me];const auto& opp=teams[t];int d=closest(opp,me);bool contact=false;
            if(t!=init.me && d<=6) {
                for(int a:opp) {for(int b:me) {
                    if(dist[a][b]>6) continue;
                    for(int x:reach[a]) {for(int y:reach[b]) if(board.canAttack(x,y)) {contact=true;break;}if(contact) break;}
                    if(contact) break;
                }if(contact) break;}
            }
            int32_t row[10]={state.turn,t,d,closest(opp,flags),closest(opp,carriers),contact,
                carrying[t],carrying[init.me],(int)opp.size(),(int)me.size()};
            if(std::fwrite(row,4,10,stdout)!=10) return 2;
        }
    }
    return 0;
}
