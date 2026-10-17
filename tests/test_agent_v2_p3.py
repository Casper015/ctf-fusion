"""P3 is a complete opponent-prediction backend in direct play and search. No host timing or games."""
from __future__ import annotations
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from fusion import runner

from tests.test_agent_v2 import (REPO, CXX, FLAGS, ACTORS, GAMES, stream, bundle_and_build,
                                assert_valid_replies, build_check)

VARIANTS = REPO / 'bots/agent_v2/p3_search/variants'

@unittest.skipUnless(CXX, 'no C++ compiler')
class P3Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='agent-v2-p3-'))
        cls.exes = {}
        for family, mode in [('p2','direct'),('p3','direct'),('p3','gw8x4_fp32')]:
            name = f'agent_v2_p3exp_{family}_{mode}'
            cls.exes[family,mode] = bundle_and_build(VARIANTS / f'{name}.cpp', cls.tmp / name)
        cls.check = cls.tmp / f'check_p3{runner.EXE}'
        build_check(cls.check, ('-DFUSION_TIER_P3XXL_VM=1',)).check_returncode()

    def test_actual_general_variants_replay_with_distinct_identities(self):
        for game in GAMES:
            text = stream(game)
            # Prefixes keep unit/flag states from the real recordings and avoid treating this as a strength run.
            text = 'END\n'.join(text.split('END\n')[:8]) + 'END\n'
            p2 = subprocess.run([str(self.exes['p2','direct'])],input=text,text=True,capture_output=True,check=True)
            self.assertEqual(len(p2.stdout.splitlines()),8)
            self.assertIn('predictor P model p2xxl_general ',p2.stderr)
            self.assertIn('refused 0 inconsistent 0',p2.stderr)
            for mode in ['direct','gw8x4_fp32']:
                r = subprocess.run([str(self.exes['p3',mode])],input=text,text=True,capture_output=True,check=True,timeout=180)
                self.assertEqual(len(r.stdout.splitlines()),8)
                self.assertTrue(all(len(x.split())==7 for x in r.stdout.splitlines()))
                self.assertIn('predictor P3 model p3xxl_general ',r.stderr)
                self.assertIn('refused 0 inconsistent 0',r.stderr)
                if mode!='direct':self.assertIn('search_used',r.stderr)

    def test_general_p2_routes_one_model_at_every_count(self):
        exe=self.tmp/f'check_p2_general{runner.EXE}'
        build_check(exe,('-DFUSION_TIER_P2GENERAL_VM=1',)).check_returncode()
        r=subprocess.run([str(exe),'bank'],text=True,capture_output=True,check=True)
        self.assertIn('bank models_pf_p2xxl_vm',r.stdout)
        self.assertIn('distinct 1',r.stdout)
        for n in range(2,16):self.assertIn(f'{n}:p2xxl_general',r.stdout)

    def test_p3_bank_and_full_contract_streams(self):
        r=subprocess.run([str(self.check),'bank'],text=True,capture_output=True,check=True)
        self.assertIn('bank models_pf_p3xxl_vm',r.stdout)
        self.assertIn('distinct 1',r.stdout)
        for game in GAMES:
            f=self.tmp/f'{game}.txt';f.write_text(stream(game))
            r=subprocess.run([str(self.check),'p',str(f)],text=True,capture_output=True,check=True,timeout=180)
            self.assertIn('turns 400',r.stdout)
            self.assertIn('model p3xxl_general',r.stdout)

    def test_p3_search_isolation_and_fixed_tree_work(self):
        text = stream('n04')
        text = 'END\n'.join(text.split('END\n')[:3]) + 'END\n'
        f = self.tmp/'isolation_prefix.txt'; f.write_text(text)
        r = subprocess.run([str(self.check),'isolation',str(f)],text=True,capture_output=True,check=True,timeout=180)
        self.assertIn('ok isolation turns 3',r.stdout)
        import os
        env = dict(os.environ, AGENT_V2_TREE_LEAVES=str(self.tmp/'leaves.f32'))
        r = subprocess.run([str(self.check),'tree',str(f),'2'],text=True,capture_output=True,check=True,env=env,timeout=180)
        tree = json.loads(r.stdout)
        self.assertEqual(tree['predictor'],'p3xxl_general')
        self.assertEqual(len(tree['leaves']),len(tree['roots'])*tree['samples'])
        self.assertTrue(any(not x['terminal'] for x in tree['leaves']))

    def test_root_and_branch_equal_existing_p3_with_init_rules(self):
        # Include the actual variant so type selection and runtime calls cannot be replaced by a test-only imitation.
        src=self.tmp/'parity.cpp'
        src.write_text(r'''
#define main ignored_bot_main
#include "bots/agent_v2/p3_search/variants/agent_v2_p3exp_p3_direct.cpp"
#undef main
#include <fstream>
#include <cassert>
using RT = agent_v2::Runtime<agent_v2::NoSearch>;
void check(RT& rt,agent_v2::Prediction& got,const fusion::State& state,const fusion::State* parent){
    ponly::View expected;
    expected.rules=rt.cache.rules;
    expected.build(rt.cache.board,state,parent,rt.cache.st.me,rt.pred.model());
    for(int uid=0;uid<int(state.units.size());uid++){
        if(!expected.has(uid/3))continue;
        for(int m=0;m<5;m++){
            assert(expected.move(uid,m)==got.view.move(uid,m));
            assert(expected.act(uid,m)==got.view.act(uid,m));
            assert(expected.target(uid,m)==got.view.target(uid,m));
        }
        for(int c=0;c<fusion::kCand;c++)assert(expected.candidate(uid,c)==got.view.candidate(uid,c));
    }
    assert(got.view.rules.damage==rt.cache.rules.damage);
}
int main(int argc,char**argv){
    RT rt;std::ifstream in(argv[1]);std::string line,block;int count=0;
    while(std::getline(in,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        block+=line+"\n";
        if(line!="END")continue;
        assert(rt.observe_block(block).empty());block.clear();
        check(rt,rt.pred.root,rt.s,rt.hist.parent());
        auto changed=rt.s;
        for(auto&u:changed.units)if(u.x>=0&&u.team!=rt.cache.st.me){u.hp=std::max(1,u.hp-5);break;}
        auto root_key=rt.pred.root.key;
        agent_v2::Prediction branch;rt.pred.branch(rt.cache,changed,branch);
        check(rt,branch,changed,&rt.s);
        assert(rt.pred.root.key==root_key);
        assert(rt.pred.counters.fallback==0);
        rt.answer(0);
        if(++count==6)break;
    }
    assert(count==6);std::cout<<"ok root branch P3 parity\n";
}
''')
        exe=self.tmp/f'parity{runner.EXE}'
        subprocess.run([CXX,*FLAGS,*(['-static'] if runner.WINDOWS else []),'-I',str(REPO),'-I',str(REPO/'inference'),str(src),'-o',str(exe)],capture_output=True,text=True,check=True)
        for game in ['n02','n15']:
            text=stream(game).replace(' 100 34 10 15 8',' 100 17 10 15 8',1)
            self.assertNotEqual(text,stream(game))
            f=self.tmp/f'custom_{game}.txt';f.write_text(text)
            r=subprocess.run([str(exe),str(f)],capture_output=True,text=True,check=True,timeout=180)
            self.assertIn('ok root branch P3 parity',r.stdout)

    def test_mixed_p2_p3_tiers_are_rejected(self):
        r=build_check(self.tmp/'bad',('-DFUSION_TIER_PXXL_VM=1','-DFUSION_TIER_P3XXL_VM=1'))
        self.assertNotEqual(r.returncode,0)
        self.assertIn('define exactly one',r.stderr)

class Manifest(unittest.TestCase):
    def test_paired_sources_only_change_the_p_bank(self):
        cfg=json.loads((VARIANTS.parent/'variants.json').read_text())
        self.assertEqual(len(cfg['variants']),4)
        for mode in ['direct','gw8x4_fp32']:
            a=(VARIANTS/f'agent_v2_p3exp_p2_{mode}.cpp').read_text()
            b=(VARIANTS/f'agent_v2_p3exp_p3_{mode}.cpp').read_text()
            self.assertEqual(a.replace('FUSION_TIER_P2GENERAL_VM','FUSION_TIER_P3XXL_VM'),b)
        self.assertNotIn('p3exp', (REPO/'profiles/agent_v2.json').read_text())

if __name__=='__main__':unittest.main()
