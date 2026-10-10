"""Derive p_only/bots/c08_control.hpp: the unchanged C08 carrier plus a search statistics line.

Record of how the control was made (source snapshot 0c388ff). The only change to
candidate/C08_Player5_Omega_LIT_Fusion.hpp is code compiled under PONLY_STATS
(screen builds): count expectimax calls and joint samples, and print them on
stderr at the last turn, as the P-only carrier does. Without PONLY_STATS the
program is the original. The wrappers c08_control.cpp (4-ply scripted playouts,
as frozen) and c08_control_2ply.cpp (MOD_PLAYOUT_SIZES 0: 2-ply playouts, nothing
else changed) include it. Run: python p_only/tools/derive/make_c08_control.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
src = (REPO / "candidate/C08_Player5_Omega_LIT_Fusion.hpp").read_text(encoding="utf-8").split("\n")


def expect(n, text):
    if text not in src[n - 1]:
        sys.exit(f"line {n}: expected {text!r}, got {src[n - 1][:90]!r}")


expect(1599, "auto ranked=search::rank(in,cfg,stop,&lastSearch,fixedSamples);")
expect(1807, "#endif")
expect(2139, "auto result=planTurn(g,true);")
out = ["// C08 Player5_Omega_LIT, unchanged, plus a search statistics line under PONLY_STATS",
       "// (p_only/tools/derive/make_c08_control.py). The network is off (MOD_FUSION_P/V 0)."]
for i, line in enumerate(src, 1):
    if i == 1599:
        line = line.replace("fixedSamples);", "fixedSamples);\n#if PONLY_STATS\n        statSearches++;statSamples+=lastSearch.samples;\n#endif", 1)
    if i == 2139:
        line = line.replace("auto result=planTurn(g,true);", "auto result=planTurn(g,true);searchReport(g);", 1)
    out.append(line)
    if i == 1807:
        out += ["    // Expectimax calls and joint samples (PONLY_STATS: one stderr line at the last turn).",
                "#if PONLY_STATS",
                "    long long statSearches=0,statSamples=0;",
                "    void searchReport(const Game& g)const{",
                "        if(g.turn==g.turns)std::cerr<<\"SEARCH calls \"<<statSearches<<\" samples \"<<statSamples<<std::endl;",
                "    }",
                "#else",
                "    void searchReport(const Game&)const{}",
                "#endif"]
dst = Path(os.environ.get("PONLY_DERIVE_DIR", REPO / "p_only/bots")) / "c08_control.hpp"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(out), encoding="utf-8")
print(dst, len(out), "lines")
