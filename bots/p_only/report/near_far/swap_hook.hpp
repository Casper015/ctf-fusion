// Report-only routing of immutable unchanged registry pointers in branch views.
// All modules use the same compiler and FUSION_MAX_WIDTH=2048 ABI.
#pragma once
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef near
#undef far
#include "policy.hpp"
#include "state.hpp"
namespace ponly {
inline bool t4_in_branch=false;
inline long long t4_teams=0,t4_near=0,t4_far=0,t4_middle=0,t4_inactive_reference=0,t4_selected_l=0;
inline int t4_mode() {static int mode=[](){const char* s=std::getenv("T0004_SWAP_MODE");return s?std::atoi(s):0;}();return mode;}
using T4Lookup=const fusion::PolicyModel* (*)(int);
inline const fusion::PolicyModel* t4_model(int tier,int teams) {
    static T4Lookup lookup[2]={nullptr,nullptr};
    if(!lookup[tier]) {
        const char* path=std::getenv(tier?"T0004_DLL_L":"T0004_DLL_S");
        if(!path) throw std::runtime_error("missing frozen model module");
        HMODULE handle=LoadLibraryA(path);
        if(!handle) throw std::runtime_error("cannot load frozen model module");
        lookup[tier]=reinterpret_cast<T4Lookup>(GetProcAddress(handle,"t0004_model"));
        if(!lookup[tier]) throw std::runtime_error("model module ABI export missing");
    }
    const auto* model=lookup[tier](teams);
    if(!model) throw std::runtime_error("missing served registry model");
    return model;
}
inline const fusion::PolicyModel* t4_route(const fusion::Board& b,const fusion::State& s,int me,int team,
                                          const fusion::PolicyModel* original) {
    if(!t4_in_branch) return original; // root predictions and real history unchanged.
    t4_teams++;
    std::vector<int> sources,dist;
    for(const auto& u:s.units) if(u.team==me&&u.x>=0) sources.push_back(b.cell(u.x,u.y));
    b.bfs(sources.data(),int(sources.size()),dist);
    int distance=fusion::kUnreachable;
    for(const auto& u:s.units) if(u.team==team&&u.x>=0) distance=std::min(distance,dist[b.cell(u.x,u.y)]);
    bool near_team=distance<=3,far_team=distance>=7&&distance<fusion::kUnreachable;
    t4_near+=near_team;t4_far+=far_team;t4_middle+=distance>=4&&distance<=6;t4_inactive_reference+=sources.empty();
    const int mode=t4_mode();
    if(mode==0) return original;
    if(mode<0||mode>4) throw std::runtime_error("invalid swap mode");
    const bool large=mode==4||(mode==2&&far_team)||(mode==3&&near_team);
    t4_selected_l+=large;
    return t4_model(large?1:0,b.teams);
}
inline void t4_report() {
    std::cerr<<"T0004 mode "<<t4_mode()<<" branch_teams "<<t4_teams<<" near "<<t4_near<<" far "<<t4_far
             <<" middle "<<t4_middle<<" inactive_reference "<<t4_inactive_reference<<" selected_l "<<t4_selected_l<<'\n';
}
}
