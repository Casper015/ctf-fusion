"""Export every arm of the aligned Meowmind RL experiment against the one frozen corpus (task T0061, host).

    python tools/meow_rl_aligned/export.py --corpus data/meow_rl_aligned/corpus

Reads bots/meow_rl/aligned.json (the frozen actors, pins, configuration and variant names) and the corpus written by
tools/meow_rl_aligned/corpus.py. For each actor (the warm start, T0055 large_v, T0059 large_v, T0059 small_v):
  * the checkpoint's bytes must equal its committed pin (the warm start is the reference's own actor);
  * the actor is quantized to reference16 with the same margin, range rounding and input scale, calibrated on all
    calibration.npz decisions; quantizing twice must give the same digest;
  * agreement with FP32 on diagnostic.npz (disjoint maps): greedy agreement, total variation and KL of the joint
    distribution over the feasible mask at T = 1.05, log-probability error, the sampled joint's agreement under common
    uniforms, saturation per hidden layer;
  * with a compiler, native parity: the generated header's 15 scores, sampled joint and greedy joint equal the Python
    emulation on every diagnostic row, on the scalar and the SIMD path.
The warm start is exported through two routes (the reference weights from the bridge, and the same weights saved and
reloaded as arrays); their integer tensors, digests and probabilities must be identical.
For the search source (T0059 large_v) its critic is built as fp32, reference16 and wideact16 (bots/meow_rl/critics.py)
from the same calibration value features, each checked natively, with its error against the float64 forward pass.
Writes inference/meow_rl/aligned/<actor>/{actor.hpp,MANIFEST.json}, the critics next to the search source's actor,
bots/meow_rl/variants/aligned/<variant>.cpp (every search variant includes the same actor.hpp as D) and
inference/meow_rl/aligned/EXPORT.json (all identities and diagnostics). --actor-npz NAME=PATH replaces a checkpoint by
arrays for plumbing tests (the manifest then says so; never evidence). Loading a .pt checkpoint needs PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "meow_rl"))

import critics as C  # noqa: E402
import sampling as SM  # noqa: E402
from fusion import runner  # noqa: E402
from training.meow_rl import bridge as B  # noqa: E402
from training.meow_rl import identity as I  # noqa: E402
from training.meow_rl import policy as P  # noqa: E402
from training.meow_rl import quant as Q  # noqa: E402

E = C.E
CONFIG = REPO / "bots" / "meow_rl" / "aligned.json"
SCHEMA = "ctf-meow-rl-aligned-export-1"
UNIFORMS = 4                                   # common uniforms per diagnostic decision for the sampled agreement


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_corpus(directory: Path) -> tuple[dict, dict, dict]:
    d = Path(directory)
    man = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    out = []
    for split in ("calibration", "diagnostic"):
        path = d / f"{split}.npz"
        if sha256(path) != man["splits"][split]["file_sha256"]:
            raise SystemExit(f"{path} differs from the corpus manifest: the corpus is frozen")
        with np.load(path) as z:
            out.append({k: z[k] for k in ("feats", "mask", "value_feats", "episode", "turn", "teams", "map_seed")})
    return out[0], out[1], {"manifest_sha256": sha256(d / "MANIFEST.json"), "schema": man["schema"], "seed": man["seed"],
                            "rows": {s: man["splits"][s]["rows"] for s in man["splits"]}}


def uniforms(n: int, k: int = UNIFORMS) -> np.ndarray:
    """Deterministic common uniforms [n, k] (splitmix64 of the row and draw index; the same for every actor)."""
    return np.array([[(SM.splitmix64(i * 0x9E3779B97F4A7C15 + j) >> 11) * 2.0 ** -53 for j in range(k)] for i in range(n)])


def inverse_cdf(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The joint drawn by sample.hpp's rule (first index whose running sum exceeds u x total) from probabilities p."""
    c = np.cumsum(p, axis=-1)
    return np.minimum((c[:, None, :] <= (u[:, :, None] * c[:, -1:, None])).sum(-1), p.shape[-1] - 1)


def actor_agreement(actor: dict, q: dict, feats: np.ndarray, masks: np.ndarray) -> dict:
    lp32 = P.joint_log_probs(P.check_actor(actor), feats, masks).astype(np.float64)
    lpq = Q.log_probs(q, feats, masks)
    p32, pq = np.where(masks, np.exp(lp32), 0.0), np.where(masks, np.exp(lpq), 0.0)
    tv = 0.5 * np.abs(p32 - pq).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(masks, p32 * (lp32 - lpq), 0.0).sum(-1)
    ok = np.isfinite(lp32) & masks
    greedy_q = np.array([Q.greedy(q, f, m) for f, m in zip(feats, masks)])
    u = uniforms(len(feats))
    same_draw = inverse_cdf(p32, u) == inverse_cdf(pq, u)
    return {"observations": int(len(feats)), "greedy_agreement": float(np.mean(greedy_q == np.argmax(np.where(masks, lp32, -np.inf), -1))),
            "tv_mean": float(tv.mean()), "tv_p99": float(np.quantile(tv, 0.99)), "tv_max": float(tv.max()),
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "max_abs_logp_error": float(np.abs(lp32[ok] - lpq[ok]).max()), "mean_abs_logp_error": float(np.abs(lp32[ok] - lpq[ok]).mean()),
            "sampled_agreement": float(same_draw.mean()), "uniforms_per_decision": UNIFORMS,
            "saturated": Q.saturated(q, feats.reshape(-1, q["net"]["in"])), "rows_per_layer": int(feats.size // q["net"]["in"])}


def critic_error(backend: dict, ref: np.ndarray, value_feats: np.ndarray) -> dict:
    v = C.value(backend, value_feats)
    e = v - ref
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "max_abs": float(np.abs(e).max()),
            "bias": float(e.mean()), "observations": int(len(ref)), "saturated": C.saturation(backend, value_feats)}


# ---- native parity ---------------------------------------------------------------------------------------------

def compile_check(cxx: str, define: str, tmp: Path, simd: int, extra=()) -> Path:
    exe = tmp / f"check_{hashlib.sha256((define + str(simd)).encode()).hexdigest()[:12]}{runner.EXE}"
    subprocess.run([cxx, "-std=c++20", "-O2", "-w", "-I", str(REPO / "inference"), "-I", str(REPO / "tools" / "cpp" / "meow_rl"),
                    define, f"-DFUSION_SIMD={simd}", *extra, str(REPO / "tools" / "cpp" / "meow_rl" / "wide_check.cpp"),
                    "-o", str(exe)], check=True, capture_output=True)
    return exe


def native_actor(cxx: str, header: Path, q: dict, feats: np.ndarray, masks: np.ndarray, tmp: Path) -> dict:
    u = uniforms(len(feats), 1)[:, 0]
    payload = b"".join(np.ascontiguousarray(f, np.float32).tobytes() + np.asarray(m, np.uint8).tobytes() + np.float64(x).tobytes()
                       for f, m, x in zip(feats, masks, u))
    want_s = Q.forward_int(q, feats).reshape(len(feats), 15)
    want_d = np.array([SM.draw(q, f, m, x) for f, m, x in zip(feats, masks, u)])
    want_g = np.array([Q.greedy(q, f, m) for f, m in zip(feats, masks)])
    out = {}
    for simd in (0, 1):
        exe = compile_check(cxx, f'-DACTOR_HEADER="{header.as_posix()}"', tmp, simd)
        r = subprocess.run([str(exe)], input=payload, capture_output=True, check=True)
        got = np.frombuffer(r.stdout, np.int32).reshape(-1, 17)
        out[f"simd{simd}"] = {
            "scores_equal": bool(np.array_equal(got[:, :15], want_s)), "draws_equal": bool(np.array_equal(got[:, 15], want_d)),
            "greedy_equal": bool(np.array_equal(got[:, 16], want_g)), "rows": int(len(got)), "path": r.stderr.decode().strip()}
    return out


def native_critic(cxx: str, header: Path, backend: dict, value_feats: np.ndarray, tmp: Path) -> dict:
    want = C.value(backend, value_feats)
    out = {}
    for simd in (0, 1):
        exe = compile_check(cxx, f'-DCRITIC_HEADER="{header.as_posix()}"', tmp, simd)
        r = subprocess.run([str(exe)], input=np.ascontiguousarray(value_feats, np.float32).tobytes(), capture_output=True, check=True)
        got = np.frombuffer(r.stdout, np.float64)
        out[f"simd{simd}"] = {"bit_equal": bool(np.array_equal(got, want)), "max_abs_diff": float(np.abs(got - want).max()),
                              "rows": int(len(got)), "path": r.stderr.decode().strip()}
    return out


# ---- variants ---------------------------------------------------------------------------------------------------

VARIANT = """// Generated by tools/meow_rl_aligned/export.py: {name}. Do not edit.
// {role}
{defines}#include "{actor}"
{critic}#include "{bot}"
"""


def rel(p: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(p, base)).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def variant_source(name: str, spec: dict, cfg: dict, inf: Path, vdir: Path) -> str:
    defines = ["#define FUSION_MAX_WIDTH 1024"]
    critic = ""
    if spec.get("critic"):
        fmt, s = spec["critic"], cfg["search"]
        defines += ["#define MEOW_RL_SEARCH 2", f"#define MEOW_RL_SEARCH_CANDIDATES {s['candidates']}",
                    f"#define MEOW_RL_SEARCH_SAMPLES {s['samples']}", f"#define MEOW_RL_SEARCH_DEADLINE_MS {s['deadline_ms']}",
                    f"#define MEOW_RL_SEARCH_FIRST_MS {s['first_ms']}", f"#define MEOW_RL_SEARCH_SAFETY_PCT {s['safety_pct']}",
                    f"#define MEOW_RL_SEARCH_TRAJ_US {s['prior_us'][fmt]['trajectory']}",
                    f"#define MEOW_RL_SEARCH_CRITIC_US {s['prior_us'][fmt]['critic']}"]
        critic = f'#include "{rel(inf / spec["actor"] / f"critic_{fmt}.hpp", vdir)}"\n'
    return VARIANT.format(name=name, role=spec["role"], defines="".join(d + "\n" for d in defines),
                          actor=rel(inf / spec["actor"] / "actor.hpp", vdir), critic=critic,
                          bot=rel(REPO / "bots" / "meow_rl" / "bot.hpp", vdir))


# ---- main ---------------------------------------------------------------------------------------------------------

def load_actor(key: str, spec: dict, npz: dict, lib) -> tuple[dict, dict | None, dict, dict]:
    """(actor arrays, critic state or None, identity, source record) of one configured actor."""
    if spec["kind"] == "warm_start":
        actor = B.reference_weights(lib, "policy")
        identity = I.identity("large_v") | {"critic": None, "actor_weights": "reference_warm_start"}
        return actor, None, identity, {"kind": "warm_start", "reference_sha256": I.REFERENCE_SHA256}
    if key in npz:                                                     # plumbing: arrays instead of the checkpoint
        z = dict(np.load(npz[key]))
        actor = {k: z[k] for k in ("w0", "b0", "w1", "b1", "w2", "b2")}
        crit = {k[len("critic."):]: v for k, v in z.items() if k.startswith("critic.")} or None
        identity = I.identity(spec["arm"]) | {"actor_weights": "trained"}
        return actor, crit, identity, {"kind": "npz_plumbing", "path": E.portable(npz[key]), "sha256": sha256(Path(npz[key])),
                                       "pinned_checkpoint_sha256": spec["sha256"], "evidence": False}
    path = REPO / spec["path"]
    got = sha256(path)
    if got != spec["sha256"]:
        raise SystemExit(f"{key}: {spec['path']} has sha256 {got}, the committed pin is {spec['sha256']}: refusing")
    committed = json.loads((REPO / spec["manifest"]).read_text(encoding="utf-8"))
    if spec["sha256"] not in json.dumps(committed):
        raise SystemExit(f"{key}: the pin is not in {spec['manifest']}")
    actor, crit, identity = E.load_checkpoint(path)
    arm = E.check_identity(identity)
    if arm != spec["arm"]:
        raise SystemExit(f"{key}: checkpoint arm {arm}, configured {spec['arm']}")
    return actor, crit, identity | {"actor_weights": "trained"}, {"kind": "checkpoint", "path": spec["path"], "sha256": got,
                                                                     "manifest": spec["manifest"]}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="the frozen corpus directory (tools/meow_rl_aligned/corpus.py)")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--actor-npz", nargs="*", default=[], help="NAME=PATH arrays (w0..b2, critic.layers.<i>.weight/bias) "
                                                                 "instead of a checkpoint: plumbing only, never evidence")
    ap.add_argument("--out", default=str(REPO / "inference" / "meow_rl" / "aligned"))
    ap.add_argument("--variants-dir", default=str(REPO / "bots" / "meow_rl" / "variants" / "aligned"))
    ap.add_argument("--no-native", action="store_true", help="skip the native parity checks (no compiler)")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    npz = dict(x.split("=", 1) for x in a.actor_npz)
    cal, diag, corpus = load_corpus(Path(a.corpus))
    cxx = None if a.no_native else runner.find_compiler()
    if not a.no_native and not cxx:
        raise SystemExit("no C++ compiler for the native parity checks (pass --no-native to skip them)")
    lib = B.load()
    sources = I.sources()
    out, vdir = Path(a.out), Path(a.variants_dir)
    report = {"schema": SCHEMA, "generator": "tools/meow_rl_aligned/export.py", "config_sha256": sha256(Path(a.config)),
              "corpus": corpus, "code_digest": sources["code_digest"], "reference_sha256": I.REFERENCE_SHA256,
              "export": cfg["export"], "actors": {}, "critics": {}, "variants": {}}
    cal_f = cal["feats"].reshape(-1, 112)
    tmp = Path(tempfile.mkdtemp(prefix="meow-rl-aligned-"))
    critic_state = None
    for key, spec in cfg["actors"].items():
        actor, crit, identity, source = load_actor(key, spec, npz, lib)
        actor = P.check_actor(actor)
        q = Q.quantize(actor, cal_f, cfg["export"]["margin"])
        if Q.quantize(actor, cal_f, cfg["export"]["margin"])["digest"] != q["digest"]:
            raise SystemExit(f"{key}: quantizing the same weights twice gave different integers")
        entry = {"source": source, "identity": identity, "identity_digest": I.digest(identity), "actor_digest": q["digest"],
                 "agreement_diagnostic": actor_agreement(actor, q, diag["feats"], diag["mask"]),
                 "calibration": {"rows": int(len(cal["feats"])), "unit_move_rows": int(len(cal_f)), "margin": q["calibration"]["margin"]},
                 "layers": [{"exp": L["exp"], "shift": L["shift"], "bound": L["bound"]} for L in q["net"]["layers"]]}
        if spec["kind"] == "warm_start":                               # the same weights through the array route
            p = tmp / "warm.npz"
            np.savez(p, **actor)
            again = P.check_actor(dict(np.load(p)))
            q2 = Q.quantize(again, cal_f, cfg["export"]["margin"])
            same = all(np.array_equal(x["w"], y["w"]) and np.array_equal(x["b"], y["b"]) and x["shift"] == y["shift"]
                       for x, y in zip(q["net"]["layers"], q2["net"]["layers"]))
            lp1, lp2 = Q.log_probs(q, diag["feats"], diag["mask"]), Q.log_probs(q2, diag["feats"], diag["mask"])
            entry["array_route"] = {"tensors_equal": bool(same), "digest_equal": q2["digest"] == q["digest"],
                                    "log_probs_equal": bool(np.array_equal(lp1, lp2))}
            if not (same and q2["digest"] == q["digest"] and np.array_equal(lp1, lp2)):
                raise SystemExit("warm start: the reference and array routes differ")
        d = out / key
        d.mkdir(parents=True, exist_ok=True)
        (d / "actor.hpp").write_text(Q.header(q, f"aligned_{key}", identity), encoding="utf-8", newline="\n")
        entry["files"] = {"actor.hpp": sha256(d / "actor.hpp")}
        if cxx:
            entry["native"] = native_actor(cxx, d / "actor.hpp", q, diag["feats"], diag["mask"], tmp)
            bad = [k for k, v in entry["native"].items() if not (v["scores_equal"] and v["draws_equal"] and v["greedy_equal"])]
            if bad:
                raise SystemExit(f"{key}: native actor differs from the Python emulation on {bad}")
        if key == cfg["search_source"]:
            if crit is None:
                raise SystemExit(f"{key}: the search source needs its critic")
            critic_state, critic_identity = crit, identity
        (d / "MANIFEST.json").write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        report["actors"][key] = entry
    W, b = E.critic_layers(critic_state)
    E.check_critic(W, b, cfg["actors"][cfg["search_source"]]["arm"])
    ref = E.critic_forward(W, b, diag["value_feats"])
    d = out / cfg["search_source"]
    for fmt in C.DIAGNOSTIC_FORMATS:                                   # offline only: never a header or a variant
        be = C.build(fmt, W, b, cal["value_feats"], cfg["export"]["margin"])
        report["critics"][fmt] = {"arm": None, "deployed": False, "digest": be["digest"], "describe": be["describe"],
                                  "calibration": be.get("calibration"),
                                  "error_vs_float64": critic_error(be, ref, diag["value_feats"])}
    for fmt, arm in cfg["critics"].items():
        be = C.build(fmt, W, b, cal["value_feats"], cfg["export"]["margin"])
        path = d / f"critic_{fmt}.hpp"
        path.write_text(C.header(be, f"aligned_{cfg['search_source']}", critic_identity), encoding="utf-8", newline="\n")
        entry = {"arm": arm, "deployed": True, "digest": be["digest"], "describe": be["describe"], "file_sha256": sha256(path),
                 "calibration": be.get("calibration"), "error_vs_float64": critic_error(be, ref, diag["value_feats"])}
        if cxx:
            entry["native"] = native_critic(cxx, path, be, diag["value_feats"], tmp)
            if not all(v["bit_equal"] for v in entry["native"].values()):
                raise SystemExit(f"critic {fmt}: native differs from the Python emulation: {entry['native']}")
        report["critics"][fmt] = entry
    vdir.mkdir(parents=True, exist_ok=True)
    actor_files = {k: report["actors"][k]["files"]["actor.hpp"] for k in report["actors"]}
    for name, spec in cfg["variants"].items():
        src = vdir / f"{name}.cpp"
        src.write_text(variant_source(name, spec, cfg, out, vdir), encoding="utf-8", newline="\n")
        report["variants"][name] = {"source": E.portable(src), "sha256": sha256(src), "actor": spec["actor"],
                                    "actor_hpp_sha256": actor_files[spec["actor"]], "critic": spec.get("critic"),
                                    "search": bool(spec.get("critic"))}
    shared = {v["actor_hpp_sha256"] for v in report["variants"].values() if v["actor"] == cfg["search_source"]}
    if len(shared) != 1:
        raise SystemExit("the search variants and D do not include one identical actor header")
    report["evidence"] = not npz
    (out / "EXPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps({"actors": {k: v["agreement_diagnostic"] for k, v in r["actors"].items()},
                      "critics": {k: v["error_vs_float64"] for k, v in r["critics"].items()}}, indent=1))
