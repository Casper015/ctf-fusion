// T0004 read-only batch scorer for one unchanged committed P2 registry.
// Compile three separate executables: T0004_TIER=0(S),1(L),2(XXL).
// Stdin: uint32 record width, then int16 rows [extractor record, move3,
// act3, target3, joint1, team_count1]. Stdout: eight float64 values per row.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "policy.hpp"
#ifndef T0004_TIER
#error T0004_TIER must explicitly identify the frozen registry
#endif
#if T0004_TIER == 0
#include "models/registry.hpp"
#elif T0004_TIER == 1
#include "models_l/registry.hpp"
#elif T0004_TIER == 2
#include "models_xxl/registry.hpp"
#else
#error Invalid T0004_TIER
#endif
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

double nll(const double* z, int n, int y) {
    if (y<0 || y>=n || z[y]<=fusion::kNegInf) throw std::runtime_error("invalid label or mask");
    double mx=*std::max_element(z,z+n),sum=0;
    for(int i=0;i<n;i++) if(z[i]>fusion::kNegInf) sum+=std::exp(z[i]-mx);
    return mx+std::log(sum)-z[y];
}
int main(int argc, char** argv) try {
    const bool raw=argc==2 && std::string(argv[1])=="--raw";
    if(argc>1 && !raw) throw std::runtime_error("unknown mode");
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY); _setmode(_fileno(stdout),_O_BINARY);
#endif
    uint32_t width=0;
    if(std::fread(&width,4,1,stdin)!=1 || width<644 || width>10000) throw std::runtime_error("record framing");
    std::vector<int16_t> row(width+11);int32_t x[fusion::kMaxWidth],acc[fusion::kMaxWidth];
    while(true) {
        size_t got=std::fread(row.data(),2,row.size(),stdin);
        if(!got) break;
        if(got!=row.size()) throw std::runtime_error("partial record");
        const auto* p=row.data()+2;fusion::PolicyFeatures f;
        for(auto& a:f.g) a=*p++;
        for(auto& u:f.u) for(auto& a:u) a=*p++;
        for(auto& u:f.m) for(auto& m:u) for(auto& a:m) a=*p++;
        for(auto& u:f.legal) for(auto& a:u) a=*p++;
        for(auto& u:f.actLegal) for(auto& m:u) for(auto& a:m) a=*p++;
        p+=3; // origin is a record field, not part of PolicyFeatures.
        for(auto& u:f.cand) for(auto& a:u) a=*p++;
        for(auto& u:f.candF) for(auto& c:u) for(auto& a:c) a=*p++;
        if(p>row.data()+width) throw std::runtime_error("feature prefix exceeds record");
        const int16_t* y=row.data()+width;
        const auto* set=fusion::models::forTeams(y[10]);
        if(!set || !set->policy) throw std::runtime_error("missing registry model");
        const auto& model=*set->policy;
        if(raw) {
            std::vector<int32_t> values;
            for(int k=0;k<3;k++) for(int m=0;m<fusion::kMoves;m++) {
                acc[0]=0;
                if(f.legal[k][m]) {fusion::moverInput(f,k,m,x);fusion::forwardRaw(model.mover,x,acc);}
                values.push_back(acc[0]);
            }
            fusion::squadInput(f,x);fusion::forwardRaw(model.squad,x,acc);
            values.insert(values.end(),acc,acc+fusion::kJoint);
            for(int k=0;k<3;k++) for(int m=0;m<fusion::kMoves;m++) {
                std::fill(acc,acc+fusion::kActs,0);
                if(f.legal[k][m]) {fusion::moverInput(f,k,m,x);fusion::forwardRaw(model.act,x,acc);}
                values.insert(values.end(),acc,acc+fusion::kActs);
            }
            for(int k=0;k<3;k++) for(int c=0;c<fusion::kCand;c++) for(int m=0;m<fusion::kMoves;m++) {
                acc[0]=0;
                if(f.cand[k][c]>=0) {fusion::targetInput(f,k,c,m,x);fusion::forwardRaw(model.target,x,acc);}
                values.push_back(acc[0]);
            }
            if(values.size()!=320 || std::fwrite(values.data(),4,320,stdout)!=320) throw std::runtime_error("raw output failure");
            continue;
        }
        fusion::PolicyOutput joint;fusion::jointPolicy(model,f,joint);
        double out[8]={nll(joint.logit.data(),fusion::kJoint,y[9]),0,0,0,0,0,0,0};
        out[5]=std::max_element(joint.logit.begin(),joint.logit.end())-joint.logit.begin()==y[9];
        bool complete=true;
        for(int k=0;k<3;k++) {
            int move=y[k],act=y[3+k],target=y[6+k];
            if(move<0 || move>=fusion::kMoves) throw std::runtime_error("invalid move");
            if(act>=0) {
                double z[fusion::kActs];fusion::moverInput(f,k,move,x);fusion::forwardRaw(model.act,x,acc);
                for(int a=0;a<fusion::kActs;a++) z[a]=f.actLegal[k][move][a]?fusion::toReal(model.act,acc[a]):fusion::kNegInf;
                out[1]+=nll(z,fusion::kActs,act);out[2]++;
                if(act==fusion::kAttack && target<0) complete=false;
            }
            if(target>=0) {
                double z[fusion::kCand];
                for(int c=0;c<fusion::kCand;c++) {
                    z[c]=fusion::kNegInf;
                    if(f.cand[k][c]>=0) {fusion::targetInput(f,k,c,move,x);fusion::forwardRaw(model.target,x,acc);z[c]=fusion::toReal(model.target,acc[0]);}
                }
                out[3]+=nll(z,fusion::kCand,target);out[4]++;
            }
        }
        out[6]=complete?out[0]+out[1]+out[3]:std::nan("");
        out[7]=set->id[0]=='n'?std::atoi(set->id+1):0; // actual serving ID;0 is general.
        if(std::fwrite(out,sizeof(double),8,stdout)!=8) throw std::runtime_error("output failure");
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
