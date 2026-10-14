"""Offline contract regressions; synthetic fixtures are not pilot evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest

import host_collect as H
import host_preflight as P


def fixture():
    init='INIT 2 0 7 400 100 34 10 20 5\nMAP\n'+('.......\n'*7)+'BASES 2\n0 0\n6 6\nSPOTS 1\n2 2\n'
    blocks=[];replies=[]
    for t in range(1,12):
        block=f'TURN {t}\nSCORE 0 0\nUNITS 6\n'+''.join(f'U {i} {i//3} {1 if i<3 else 5} {i%3+1} 100 -1 -1\n' for i in range(6))+'FLAGS 1\nF 0 ground 2 2 -1 -1\nEVENTS 0\nEND\n'
        log=f'{t} S - S - S - # dsl1 stage=beam k=2 root={H.B.root_digest(block)} inc=0 checked=0 | S - S - S -;1;na;inc | U - S - S -;2;na;beam'
        blocks.append(block);replies.append([log,f'{t} S - S - S -'])
    return {'match_id':'synthetic-contract-only','teams':2,'map_seed':11,'init':init,'blocks':blocks,'replies':replies,'seats':[{'name':'@candidate','family':'candidate'},{'name':'toy','family':'toy'}],'result':{'score':[0,1]}}


class Contracts(unittest.TestCase):
    def test_final_scores_do_not_select_a_root(self):
        match=fixture();before=H.select_root(match,'flag')
        self.assertIsNotNone(before[0])
        changed=copy.deepcopy(match);changed['result']['score']=[999,-123]
        self.assertEqual(before,H.select_root(changed,'flag'))

    def test_scored_singleton_is_preserved(self):
        log=H.B.parse_log(fixture()['replies'][5][0],6)
        log['entries']=log['entries'][:1];log['k']=1
        scored,reason=H.scored_with_k1(log)
        self.assertIsNone(reason)
        self.assertEqual(len(scored['entries']),1)
        self.assertEqual(scored['entries'][0]['score'],1)

    def test_mutated_successful_probe_is_refused(self):
        with tempfile.TemporaryDirectory(dir=H.REPO/'build') as name:
            path=Path(name)/'probe.json.gz'
            P.write_gz(path,{'preflight_identity':'fixture-only','replies':['known']})
            self.assertEqual(P.read_probe(path,'fixture-only')['replies'],['known'])
            path.write_bytes(path.read_bytes()+b'mutation')
            with self.assertRaises(AssertionError):P.read_probe(path,'fixture-only')

    def test_foreign_probe_identity_is_refused(self):
        with tempfile.TemporaryDirectory(dir=H.REPO/'build') as name:
            path=Path(name)/'probe.json.gz'
            P.write_gz(path,{'preflight_identity':'fixture-only','replies':['known']})
            with self.assertRaises(AssertionError):P.read_probe(path,'another-identity')


if __name__=='__main__':unittest.main()
