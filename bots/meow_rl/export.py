"""Export a Meowmind RL actor (and optionally its critic) into a CPU bot variant (task T0056).

    python bots/meow_rl/export.py --warm-start --name warm_start                       # the matched control
    python bots/meow_rl/export.py --checkpoint runs/meow_rl/large_v_s101/checkpoints/final.pt --name rl_large_v_s101
    python bots/meow_rl/export.py --checkpoint .../final.pt --name rl_large_v_s101_search --critic --search

Writes inference/meow_rl/<name>/actor.hpp (and critic.hpp with --critic), inference/meow_rl/<name>/MANIFEST.json and
bots/meow_rl/variants/<name>.cpp, the file bots/p_only/tools/build.py bundles (list it in bots/meow_rl/variants.json).
Both networks use the repository's INT16 format (reference16: training/meow_rl/quant.py, inference/mlp.hpp).

Identity. A checkpoint (training.meow_rl.train's state.pt / checkpoints/*.pt) carries the run's identity: actor and
critic architecture, feature schemas, temperature, reward units, resolver and the reference's sha256. The export
refuses one that differs from this code's identity (training.meow_rl.identity). The warm start is the reference
actor itself, exported through the same path: control and candidate differ only in their actor weights.

Calibration and agreement. The hidden activation ranges of the integer format come from recorded observations. By
default (plumbing, the warm start) the bridge replays the recorded 4-team seat stream (bots/p_only/tests/fixtures/
stream_n04) with the FP32 actor's greedy choices. For a trained export pass --calibration-batches: the run's
batches/update_*.npz written with training.meow_rl.train --keep-features (every team count, early and late turns,
combat, carriers, respawns), subsampled to --calibration-rows. Either way the agreement with FP32 is measured on the
held-out 2- and 15-team recorded streams and recorded in the manifest (greedy-joint agreement, log-probability
error, saturations; the critic's value error). Never calibrate on screen fixtures. Loading a .pt
checkpoint needs PyTorch; --actor-npz takes the actor (and --critic-npz the critic) as NumPy arrays instead.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fusion import quant as FQ  # noqa: E402
from training.meow_rl import bridge as B  # noqa: E402
from training.meow_rl import identity as I  # noqa: E402
from training.meow_rl import policy as P  # noqa: E402
from training.meow_rl import quant as Q  # noqa: E402

FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"
CALIBRATION = ("n04",)
HELD_OUT = ("n02", "n15")
CRITIC_WIDTH = 1024                     # FUSION_MAX_WIDTH a critic variant is built with (inference/mlp.hpp)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observations(lib, actor: dict, games) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(feats [N,3,5,112], masks [N,125], value feats [N,416]) of recorded seat streams, the FP32 actor choosing."""
    F, M, V = [], [], []
    for game in games:
        text = gzip.decompress((FIXTURES / f"stream_{game}.txt.gz").read_bytes()).decode()
        with B.Session(lib) as s:
            for block in B.split_stream(text):
                o = s.observe(block)
                F.append(o.feats), M.append(o.joint_mask), V.append(o.value_feats)
                s.resolve(P.greedy(P.joint_log_probs(actor, o.feats, o.joint_mask)))
    return np.array(F), np.array(M), np.array(V)


def batch_observations(directory: Path, rows: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Actor and value features from a run's saved batches, an evenly spaced subsample of `rows` decisions."""
    files = sorted(directory.glob("update_*.npz"))
    feats, values = [], []
    for f in files:
        with np.load(f) as z:
            if "feats" not in z or "value_feats" not in z:
                raise SystemExit(f"{f}: no features (the run needs --keep-features for calibration)")
            feats.append(z["feats"]), values.append(z["value_feats"])
    if not feats:
        raise SystemExit(f"no update_*.npz in {directory}")
    F, V = np.concatenate(feats), np.concatenate(values)
    pick = np.linspace(0, len(F) - 1, min(rows, len(F))).astype(np.int64)
    return F[pick], V[pick], {"batches": portable(directory), "files": len(files),
                              "files_sha256": hashlib.sha256("".join(sha256(f) for f in files).encode()).hexdigest()}


# ---- critic: the same per-neuron folding as the actor, for any depth ------------------------------------------

def critic_layers(state: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """[out, in] weights and biases of a training.meow_rl.nets.Critic state dict (tensors or arrays)."""
    as_np = lambda t: np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t, np.float32)  # noqa: E731
    n = len([k for k in state if k.endswith(".weight")])
    W = [as_np(state[f"layers.{i}.weight"]) for i in range(n)]
    b = [as_np(state[f"layers.{i}.bias"]) for i in range(n)]
    return W, b


def critic_forward(W, b, x: np.ndarray, hidden: bool = False):
    h, acts = np.asarray(x, np.float64), []
    for i, (w, c) in enumerate(zip(W, b)):
        h = h @ w.T.astype(np.float64) + c
        if i < len(W) - 1:
            h = np.maximum(h, 0)
            acts.append(h)
    return acts if hidden else h[:, 0]


def check_critic(W, b, arm: str) -> None:
    """The critic must be exactly the declared arm's architecture (416 -> hidden -> 1) with finite weights."""
    hidden = I.load_profile()["critic" if arm == "large_v" else "small_critic"]["hidden"]
    sizes = [I.VF, *hidden, 1]
    want = [((o, i), (o,)) for i, o in zip(sizes, sizes[1:])]
    got = [(w.shape, c.shape) for w, c in zip(W, b)]
    if got != want:
        raise SystemExit(f"critic shapes {got} are not the {arm} architecture {want}: refusing to export")
    if not all(np.isfinite(w).all() and np.isfinite(c).all() for w, c in zip(W, b)):
        raise SystemExit("critic has non-finite weights: refusing to export")


def quantize_critic(W, b, calibration: np.ndarray, margin: float = Q.MARGIN) -> dict:
    ranges = [np.maximum(a.max(axis=0) * margin, Q.FLOOR) for a in critic_forward(W, b, calibration, hidden=True)]
    Wf, bf, prev = [], [], None
    for i, (w, c) in enumerate(zip(W, b)):
        w, c = w.astype(np.float64), c.astype(np.float64)
        if prev is not None:
            w = w * prev[None, :]
        if i < len(W) - 1:
            w, c = w / ranges[i][:, None], c / ranges[i]
            prev = ranges[i]
        Wf.append(w), bf.append(c)
    n_in = W[0].shape[1]
    net = FQ.quantize(types.SimpleNamespace(W=Wf, b=bf), np.full(n_in, float(1 << Q.IN_BITS)),
                      np.full(n_in, -Q.IN_MAX), np.full(n_in, Q.IN_MAX))
    FQ.check_widths(net, "meow_rl critic", CRITIC_WIDTH)
    return {"scheme": Q.SCHEME, "format": "reference16", "net": net, "digest": FQ.digest(FQ.flatten(net, "critic")),
            "calibration": {"margin": margin, "observations": int(len(calibration))}}


def critic_value_int(q: dict, value_feats: np.ndarray) -> np.ndarray:
    return np.ldexp(FQ.forward_raw(q["net"], Q.quantize_input(value_feats))[:, 0].astype(np.float64), -q["net"]["out_exp"])


def critic_header(q: dict, name: str, identity: dict) -> str:
    from training.export_weights import _array, _emit_nets  # noqa: PLC0415
    body: list[str] = []
    _emit_nets(body, {"critic": q["net"]}, {}, False, _array)
    layers = q["net"]["layers"]
    return "\n".join([f"// Generated by bots/meow_rl/export.py: {name} critic. Do not edit.",
                      f"// scheme {q['scheme']} (reference16 INT16), critic {identity['critic']}, units {identity['reward_units']}",
                      "#pragma once", "#include <cstdint>", '#include "mlp.hpp"', "namespace meow_rl_critic {",
                      f'inline constexpr const char* kDigest = "{q["digest"]}";',
                      f'inline constexpr const char* kIdentity = "{I.digest(identity)}";', *body,
                      f"inline constexpr fusion::Net kCritic{{critic_layers, {len(layers)}, {q['net']['out_exp']}}};",
                      f'static_assert({layers[0]["w"].shape[1]} == 416 && {layers[-1]["w"].shape[0]} == 1, '
                      '"the critic maps the 416 value features to one value");',
                      "}  // namespace meow_rl_critic", ""])


# ---- inputs ----------------------------------------------------------------------------------------------------

def arm_of(identity: dict) -> str:
    for arm, arch in I.CRITIC_ARCH.items():
        if identity.get("critic") == arch:
            return arm
    raise SystemExit(f"checkpoint critic {identity.get('critic')!r} is not a known arm")


def load_checkpoint(path: Path) -> tuple[dict, dict, dict]:
    """(actor numpy dict, critic state dict, identity) of a training.meow_rl checkpoint."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        raise SystemExit("loading a .pt checkpoint needs PyTorch (or pass --actor-npz/--critic-npz)") from None
    st = torch.load(path, map_location="cpu", weights_only=False)
    actor, learned = st["actor_numpy"], st["learner"]["actor"]
    pairs = {"w0": "l0.weight", "b0": "l0.bias", "w1": "l1.weight", "b1": "l1.bias", "w2": "l2.weight", "b2": "l2.bias"}
    for k, t in pairs.items():                         # the NumPy snapshot must be the learner's actor exactly
        if not np.array_equal(np.asarray(actor[k], np.float32), learned[t].cpu().numpy().astype(np.float32)):
            raise SystemExit(f"{path}: actor_numpy[{k}] differs from learner.actor[{t}]: refusing to export")
    return actor, st["learner"]["critic"], st["identity"]


def portable(path) -> str:
    """A path as recorded in a committed manifest: repository-relative when inside it, else its file name only."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.name


def check_identity(identity: dict) -> str:
    arm = arm_of(identity)
    want = I.identity(arm)
    diff = sorted(k for k in set(want) | set(identity) if want.get(k) != identity.get(k))
    if diff:
        raise SystemExit(f"checkpoint identity differs from this code in {diff}: refusing to export")
    return arm


VARIANT = """// Generated by bots/meow_rl/export.py: {name}. Do not edit.
// {what}
{defines}#include "{actor}"
{critic}#include "{bot}"
"""


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--warm-start", action="store_true", help="the reference actor (the matched control)")
    src.add_argument("--checkpoint", help="a training.meow_rl checkpoint (.pt, needs PyTorch)")
    src.add_argument("--actor-npz", help="actor arrays w0..b2 (NumPy) with --identity-json")
    ap.add_argument("--identity-json", help="the run identity for --actor-npz (config.json's 'identity')")
    ap.add_argument("--critic-npz", help="critic arrays layers.<i>.weight/bias (NumPy) for --critic with --actor-npz")
    ap.add_argument("--name", required=True)
    ap.add_argument("--critic", action="store_true", help="also export the critic (needed by --search)")
    ap.add_argument("--search", action="store_true", help="the variant enables the bounded value search")
    ap.add_argument("--calibration-batches", help="a run's batches/ directory (update_*.npz with feats, --keep-features)")
    ap.add_argument("--calibration-rows", type=int, default=50_000, help="decisions sampled from the batches")
    ap.add_argument("--out", default=str(REPO / "inference" / "meow_rl"))
    ap.add_argument("--variants-dir", default=str(REPO / "bots" / "meow_rl" / "variants"))
    a = ap.parse_args(argv)
    if a.search and not a.critic:
        ap.error("--search needs --critic")
    sources = I.sources()                      # refuses a reference whose bytes differ from the frozen sha256
    lib = B.load()
    critic_state, source = None, {}
    if a.warm_start:
        if a.critic:
            ap.error("the warm start has no trained critic (the reference V has other units)")
        actor = B.reference_weights(lib, "policy")
        identity = I.identity("large_v") | {"critic": None, "actor_weights": "reference_warm_start"}
        source = {"kind": "warm_start", "reference_sha256": I.REFERENCE_SHA256}
    else:
        if a.checkpoint:
            actor, critic_state, identity = load_checkpoint(Path(a.checkpoint))
            source = {"kind": "checkpoint", "path": portable(a.checkpoint), "sha256": sha256(Path(a.checkpoint))}
        else:
            if not a.identity_json:
                ap.error("--actor-npz needs --identity-json")
            actor = dict(np.load(a.actor_npz))
            identity = json.loads(Path(a.identity_json).read_text(encoding="utf-8"))
            critic_state = dict(np.load(a.critic_npz)) if a.critic_npz else None
            source = {"kind": "npz", "path": portable(a.actor_npz), "sha256": sha256(Path(a.actor_npz))}
        arm = check_identity(identity)
        identity = identity | {"actor_weights": "trained"}
        if a.critic and critic_state is None:
            ap.error("--critic needs the critic (a .pt checkpoint or --critic-npz)")
    actor = P.check_actor(actor)
    if a.calibration_batches:
        cal_f, cal_v, cal_src = batch_observations(Path(a.calibration_batches), a.calibration_rows)
    else:
        cal_f, _, cal_v = observations(lib, actor, CALIBRATION)
        cal_src = {"games": list(CALIBRATION)}
    held_f, held_m, held_v = observations(lib, actor, HELD_OUT)
    q = Q.quantize(actor, cal_f)
    report = {"actor": Q.agreement(actor, q, held_f, held_m)}
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "actor.hpp").write_text(Q.header(q, a.name, identity), encoding="utf-8", newline="\n")
    files = {"actor.hpp": sha256(out / "actor.hpp")}
    if a.critic:
        W, b = critic_layers(critic_state)
        check_critic(W, b, arm)
        qc = quantize_critic(W, b, cal_v)
        err = np.abs(critic_value_int(qc, held_v) - critic_forward(W, b, held_v))
        report["critic"] = {"max_abs_value_error": float(err.max()), "mean_abs_value_error": float(err.mean()),
                            "observations": int(len(held_v)), "parameters": int(sum(w.size + c.size for w, c in zip(W, b)))}
        (out / "critic.hpp").write_text(critic_header(qc, a.name, identity), encoding="utf-8", newline="\n")
        files["critic.hpp"] = sha256(out / "critic.hpp")
    manifest = {"schema": "ctf-meow-rl-export-1", "name": a.name, "generator": "bots/meow_rl/export.py",
                "identity": identity, "identity_digest": I.digest(identity), "code_digest": sources["code_digest"],
                "source": source, "format": {"actor": q["scheme"], "critic": Q.SCHEME if a.critic else None},
                "actor_digest": q["digest"], "calibration": cal_src | {"margin": Q.MARGIN, "rows": int(len(cal_f))},
                "agreement_held_out": {"games": list(HELD_OUT), **report}, "files": files, "search": a.search,
                "deployment": {"choice": "sampled actor-direct: one draw from the integer actor's softmax over feasible "
                                         "joints at T = 1.05, common per-turn uniforms (bots/meow_rl/sample.hpp); "
                                         "MEOW_RL_GREEDY=1 builds the greedy diagnostic",
                               "search": "bots/meow_rl/search.hpp (MEOW_RL_SEARCH=1)" if a.search else None}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    defines = "#define MEOW_RL_SEARCH 1\n" if a.search else ""
    if a.critic:
        defines = f"#define FUSION_MAX_WIDTH {CRITIC_WIDTH}\n" + defines
    variant = Path(a.variants_dir) / f"{a.name}.cpp"
    variant.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: Path(os.path.relpath(p, variant.parent)).as_posix()  # noqa: E731
    critic = f'#include "{rel(out / "critic.hpp")}"\n' if a.critic else ""
    what = ("matched control: the reference actor (warm start), actor-direct" if a.warm_start
            else f"trained actor{' + bounded value search' if a.search else ', actor-direct'}")
    variant.write_text(VARIANT.format(name=a.name, what=what, defines=defines, critic=critic, actor=rel(out / "actor.hpp"),
                                      bot=rel(REPO / "bots" / "meow_rl" / "bot.hpp")), encoding="utf-8", newline="\n")
    return manifest | {"_out": str(out), "_variant": str(variant)}


if __name__ == "__main__":
    m = main()
    print(json.dumps({"out": m["_out"], "variant": m["_variant"], "agreement": m["agreement_held_out"]}, indent=1))
