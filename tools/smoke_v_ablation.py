import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import runner
from roster.strength import schedule, play_one

root = Path("c:/Users/caspe/OneDrive/Code/AI_Test").resolve()
build = Path("build/test_carrier").resolve()
roster_build = Path("build/roster").resolve()

sched = schedule(1, [2, 5], ["Astra", "Sol", "Fable", "Sonnet", "Player5"], 20261020, root)
pool = {e["name"]: e for e in json.loads(Path("profiles/pool.json").read_text())["bots"]}
ref_cmds = {r["name"]: runner.command_for(root / pool[r["name"]]["path"], roster_build) for r in sched["reference"]}
info = {r["name"]: {"family": pool[r["name"]]["family"], "sha256": r["source_sha256"]} for r in sched["reference"]}
info["@candidate"] = {"family": "candidate", "sha256": "test"}
from fusion.paths import sha256, git_commit
info["__engine__"] = sha256(root / "ctf-arena/Player7/workspace/engine.py")
info["__commit__"] = git_commit(root)
limits = (2.0, 0.05)

test_bins = [
    ("C08_ctrl", build / "C08_Player5_Omega_LIT_control-2e7c5307b352aa01.exe"),
    ("C08_pm_no_v", build / "C08_Player5_Omega_LIT_pm_no_v-0369cb4d84b75119.exe"),
    ("C08_pm_with_v", build / "C08_Player5_Omega_LIT_pm_with_v-d33cc06e01eb69d0.exe"),
    ("v12_ctrl", build / "Player5_v12_control-8a35602373bf8c34.exe"),
    ("v12_pm_no_v", build / "Player5_v12_pm_no_v-207c49734a432f45.exe"),
    ("v12_pm_with_v", build / "Player5_v12_pm_with_v-a1e634768e970a33.exe"),
]

print("=== SMOKE TEST: ALL 6 CANDIDATES ===")
for tag, exe in test_bins:
    for fx in sched["fixtures"]:
        cmds = dict(ref_cmds)
        cmds["@candidate"] = [str(exe)]
        job = (fx, {"id": tag, "key": tag, "source_sha256": tag, "schedule": "smoke"}, cmds, info, root, limits, None)
        t0 = time.time()
        res = play_one(job)
        dt = time.time() - t0
        print(f"[{tag:14s}] {fx['teams']} teams: score={res['score']}, util={res['utility']:.3f}, p99={res['p99_ms']:.1f}ms, faults={res['faults']} in {dt:.1f}s")

print("All smoke tests completed successfully!")
