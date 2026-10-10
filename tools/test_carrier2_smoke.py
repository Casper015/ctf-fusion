"""Smoke test for Carrier 2 (Player5_v12): prove null on control vs vanilla, and test p_s / p_m."""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fusion import runner
from roster.strength import schedule, play_one

root = Path("c:/Users/caspe/OneDrive/Code/AI_Test").resolve()
build = REPO / "build" / "test_carrier"
roster_build = REPO / "build" / "roster"

ref_names = ["Astra", "Sol", "Fable", "Sonnet", "Opus", "DeepSeek", "Qwen", "MiMo", "Kimi", "GLM", "Player5", "Baseline"]
sched = schedule(1, [2, 5], ref_names, 20261020, root)

cand_orig = {"id": "orig", "key": "orig", "source_sha256": "orig", "schedule": "smoke"}
cand_ctrl = {"id": "ctrl", "key": "ctrl", "source_sha256": "ctrl", "schedule": "smoke"}
cand_ps = {"id": "ps", "key": "ps", "source_sha256": "ps", "schedule": "smoke"}
cand_pm = {"id": "pm", "key": "pm", "source_sha256": "pm", "schedule": "smoke"}

# Build commands for reference bots
pool = {e["name"]: e for e in json.loads((REPO / "profiles" / "pool.json").read_text())["bots"]}
commands = {}
info = {}
for r in sched["reference"]:
    bpath = root / pool[r["name"]]["path"]
    commands[r["name"]] = runner.command_for(bpath, roster_build)
    info[r["name"]] = {"family": pool[r["name"]]["family"], "sha256": r["source_sha256"]}
info["@candidate"] = {"family": "candidate", "sha256": "test"}
from fusion.paths import sha256, git_commit
info["__engine__"] = sha256(root / "ctf-arena" / "Player7" / "workspace" / "engine.py")
info["__commit__"] = git_commit(root)

limits = (2.0, 0.05)
fx5 = [f for f in sched["fixtures"] if f["teams"] == 5][0]
fx2 = [f for f in sched["fixtures"] if f["teams"] == 2][0]

def run_fixture(fx, cand, bot_exe):
    cmds = dict(commands)
    cmds["@candidate"] = [str(bot_exe)]
    job = (fx, cand, cmds, info, root, limits, None)
    t0 = time.time()
    res = play_one(job)
    dt = time.time() - t0
    return res, dt

exe_orig = list((REPO / "build" / "roster").glob("ctf-arena_Player5_v12_rollout_volley*.exe"))[0]
exe_ctrl = list(build.glob("Player5_v12_control*.exe"))[0]
exe_ps = list(build.glob("Player5_v12_p_s*.exe"))[0]
exe_pm = list(build.glob("Player5_v12_p_m*.exe"))[0]

print("--- 5 TEAMS TEST (Player5_v12 Carrier) ---")
print("Playing orig (5 teams)...")
res_orig, dt_orig = run_fixture(fx5, cand_orig, exe_orig)
print(f"Orig: score={res_orig['score']}, util={res_orig['utility']:.3f}, p99={res_orig['p99_ms']:.1f}ms in {dt_orig:.1f}s")

print("Playing ctrl (5 teams)...")
res_ctrl, dt_ctrl = run_fixture(fx5, cand_ctrl, exe_ctrl)
print(f"Ctrl: score={res_ctrl['score']}, util={res_ctrl['utility']:.3f}, p99={res_ctrl['p99_ms']:.1f}ms in {dt_ctrl:.1f}s")
print("Scores identical (null proven)?", res_orig["score"] == res_ctrl["score"])
assert res_orig["score"] == res_ctrl["score"], "Scores must match between orig and ctrl!"

print("\nPlaying P-S (5 teams)...")
res_ps, dt_ps = run_fixture(fx5, cand_ps, exe_ps)
print(f"P-S: score={res_ps['score']}, util={res_ps['utility']:.3f}, p99={res_ps['p99_ms']:.1f}ms in {dt_ps:.1f}s")

print("\nPlaying P-M (5 teams)...")
res_pm, dt_pm = run_fixture(fx5, cand_pm, exe_pm)
print(f"P-M: score={res_pm['score']}, util={res_pm['utility']:.3f}, p99={res_pm['p99_ms']:.1f}ms in {dt_pm:.1f}s")

print("\n--- 2 TEAMS (DUEL) TEST (Player5_v12 Carrier) ---")
print("Playing orig (duel)...")
d_orig, _ = run_fixture(fx2, cand_orig, exe_orig)
print(f"Orig: score={d_orig['score']}, util={d_orig['utility']:.3f}")

print("Playing ctrl (duel)...")
d_ctrl, _ = run_fixture(fx2, cand_ctrl, exe_ctrl)
print(f"Ctrl: score={d_ctrl['score']}, util={d_ctrl['utility']:.3f}")
print("Duel scores identical (null proven)?", d_orig["score"] == d_ctrl["score"])
assert d_orig["score"] == d_ctrl["score"]

print("\nPlaying P-S (duel)...")
d_ps, _ = run_fixture(fx2, cand_ps, exe_ps)
print(f"P-S: score={d_ps['score']}, util={d_ps['utility']:.3f}")

print("\nPlaying P-M (duel)...")
d_pm, _ = run_fixture(fx2, cand_pm, exe_pm)
print(f"P-M: score={d_pm['score']}, util={d_pm['utility']:.3f}")

print("\nAll Player5_v12 smoke tests passed successfully!")
