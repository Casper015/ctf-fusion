# T0032 reproduction

Everything ran on Google Colab Pro from the Drive notebook `T0032_p3_distill_colab_pro.ipynb`, with code from this repository at `95e173b9ac87c3f769134669e488630d1f6b35bc` (main after PR #24). Driver v3 checks that `training/` and `fusion/` there are the same trees as `39eb4e5` (the T0031 code the first runtime used). Data: T0011's `data/pf_policy_cache/{train,val}` copied from Drive `cft-fusion/pf_policy_cache`, matched to `reports/models/p_features/POLICY_CACHE.json` by row count and every array's data SHA-256, and rehashed with `python -m training.distill_policy verify-cache` (identities train `350264d6881ddf31`, val `d899cf9eac2d2bf5`, seven arrays each including `teams.npy`). Weights, the teacher-target shard and logs stay on Drive under `cft-fusion/t0032/`; `RUNS.json` and `eval.json` here are byte-identical copies (SHA-256 `7aa67f8bdb74e4fcd37883913f2fa0b5173eabf07ae2c48e17e60476a70ec451` and `b1b67e0f55dae132a68e286a168f7eb347b6857dfdaf8f3349d49441fbf76623`).

## Runtimes

| Runtime | Machine | What happened |
|---|---|---|
| 1 | A100 40 GB, High-RAM, driver v2 | 6XL teacher stopped after one epoch (switched to 8XL); 8XL teacher reached epoch 6, then the runtime ended with nothing finished. Its mirror is Drive `t0032/lost_runtime_1`. Not a result. |
| 2 | A100 80 GB, High-RAM, driver v3 | Teacher (23:36-02:47Z), teacher targets, warmup_s101 alone (2 epochs, stopped for the parallel students). Runtime type changed at ~03:10Z, 3 minutes into the parallel students. |
| 3 | A100 40 GB, standard RAM (83 GB) | Teacher restored from Drive; warmup_s101-103 side by side and backed up. Distill stalled (train cache 42 GB + teacher shard 19 GB thrashed the page cache); stored to Drive, switched back. |
| 4 | A100 80 GB, High-RAM | Teacher and warmups restored from Drive; distill_s101-103 and qat_s101-103 side by side; evaluation and bundle. |

Stopped or lost runs are kept on Drive only and are not results. Results do not depend on which runtime produced a stage; every stage reran from its inputs with the same seed and command. Timings are development-machine numbers and not evidence.

## Order

1. Notebook cell 1 mounts Drive; cells 2-3 write `/content/t0032_train_cache.py` and `/content/t0032_colab_pro.sh`; the students cell writes `/content/t0032_students.sh`; the (re)start cell writes and starts `/content/t0032_restart.sh`.
2. Teacher: `run teacher_8xl_p3` in driver v3 (8XL widths, seed 100, T0011's arguments).
3. Teacher targets: `python -m training.distill_policy teacher-targets --teacher teacher_8xl_p3 --cache train --split train --tau 1.0 --batch 8192 --device cuda`.
4. Students: `t0032_students.sh`, seeds 101-103 side by side: warmup (alpha 1), distill (alpha 0.5, tau 1), qat mixed4_8 from distill (2 epochs, lr 1e-4 -> 1e-5).
5. `REPO=/content/t31 python t0032_eval.py --out /content/t0032 --val /content/pf_policy_cache/val warmup_s101 distill_s101 ...`, then `python t0032_bundle.py /content/t0032 > RUNS.json`.
6. On the Mac: `python reports/models/p_distill/scripts/t0032_summarize.py` writes `RESULTS.md`.

## t0032_train_cache.py

Teacher launcher: the unchanged policy trainer on T0011's cache, bound to POLICY_CACHE.json. SHA-256 `46cc43dbdc6935cb70df76141bded63514b0a9d66adc1197719fb50109811375`.

```python
"""Run the unchanged policy trainer (training/train_policy.py) on T0011's policy-row cache, on Colab.

T0011's launcher (reports/models/p_features/REPRODUCE.md, t0011_train_mmap.py) checked the cache against
data/shards_l3/index.json. Colab has the cache but not the shards, so this launcher binds the cache to the
committed receipt instead: every array's shape, dtype and data SHA-256 must equal
reports/models/p_features/POLICY_CACHE.json (the hashes themselves are checked once by
`python -m training.distill_policy verify-cache`). Everything else is T0011's launcher: read-only mmaps, the
chunked label check, and the unchanged trainer with its arguments from the command line.

    CACHE=/content/pf_policy_cache python t0032_train_cache.py --model general --shards data/shards_l3 ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("REPO", "/content/t31"))
CACHE = Path(os.environ.get("CACHE", "/content/pf_policy_cache"))
sys.path.insert(0, str(REPO))
from training import train_policy  # noqa: E402

COMMITTED = json.loads((REPO / "reports/models/p_features/POLICY_CACHE.json").read_text())
original_check = train_policy.check_labels


def prepare(shards, counts, split, max_rows, rng, keep_unlabeled=False, family="p2"):
    assert list(counts) == list(range(2, 16)) and not max_rows and not keep_unlabeled
    meta = json.loads((CACHE / split / "CACHE.json").read_text())
    want = COMMITTED["splits"][split]
    assert meta["complete"] and meta["row_order_and_values_equal"] and meta["rows"] == want["rows"]
    assert meta["identity"] == want["identity"], f"{split}: cache identity differs from POLICY_CACHE.json"
    out = {}
    for key, spec in want["arrays"].items():
        assert meta["arrays"][key]["data_sha256"] == spec["data_sha256"], (split, key)
        out[key] = np.load(CACHE / split / f"{key}.npy", mmap_mode="r")
        assert list(out[key].shape) == spec["shape"] and out[key].dtype.str == spec["dtype"], (split, key)
    print(f"[T0032 mmap] {split}: {meta['rows']} rows bound to POLICY_CACHE.json, no RAM concatenation", flush=True)
    return out


def check_labels(data, layout):
    for start in range(0, len(data["rec"]), 8192):
        original_check({k: v[start:start + 8192] for k, v in data.items()}, layout)


def data_manifest(shards, counts):
    return {"team_counts": list(counts), "source": "T0011 data/pf_policy_cache, bound to POLICY_CACHE.json",
            "rows": {s: COMMITTED["splits"][s]["rows"] for s in ("train", "val")},
            "data_sha256": {s: {k: v["data_sha256"] for k, v in COMMITTED["splits"][s]["arrays"].items()}
                            for s in ("train", "val")}}


def main():
    train_policy.prepare = prepare
    train_policy.check_labels = check_labels
    train_policy.data_manifest = data_manifest
    train_policy.main()


if __name__ == "__main__":
    main()
```

## t0032_colab_pro.sh

Driver v3: code check, restore finished stages from Drive, copy and verify the cache, 5-minute Drive mirror, teacher, teacher targets, then the students in sequence (handed over to t0032_students.sh in practice). SHA-256 `682b061b2d5f9b51ba73a719eaf374fa7af86d1bfa47d6989e9a92193326f032`.

```bash
#!/usr/bin/env bash
# T0032 driver v3 (Colab Pro, A100), Drive first. Replaces t0032_run.sh (v2, sha 1cea884a), whose runtime ended
# with nothing saved but finished stages. Order:
#   0. code at COMMIT (training/ and fusion/ must equal 39eb4e5's, the code of v2); stages finished on an earlier
#      runtime are restored from Drive; T0011's cache is copied from Drive and rehashed (verify-cache);
#      /content/t0032 is mirrored to Drive cft-fusion/t0032/live every 5 minutes from here on.
#   1. 8XL p3 teacher -> 2. teacher targets -> 3. seeds 101-103 x (warmup, distill, qat).
# Resumable: a stage whose model.npz and config.json exist (targets: IDENTITY.json) is skipped. Each finished stage
# is copied to Drive cft-fusion/t0032/<stage>/ with its log (the teacher shard arrays stay on the runtime).
set -euo pipefail
COMMIT=95e173b9ac87c3f769134669e488630d1f6b35bc
REPO=/content/t31
CACHE=/content/pf_policy_cache
OUT=/content/t0032
DRIVE_ROOT=/content/drive/MyDrive/cft-fusion
DRIVE=$DRIVE_ROOT/t0032
log() { echo "[t0032] $(date -u +%FT%TZ) $*"; }

[ -d "$DRIVE_ROOT" ] || { log "Drive is not mounted ($DRIVE_ROOT missing)"; exit 1; }
mkdir -p "$OUT" "$DRIVE/live" "$CACHE"

# 0a. code
[ -d "$REPO/.git" ] || git clone -q https://github.com/Casper015/ctf-fusion "$REPO"
git -C "$REPO" fetch -q origin
git -C "$REPO" checkout -q --detach "$COMMIT"
for p in training fusion; do
  [ "$(git -C "$REPO" rev-parse "$COMMIT:$p")" = "$(git -C "$REPO" rev-parse "39eb4e5:$p")" ] \
    || { log "$p/ differs from 39eb4e5"; exit 1; }
done
cd "$REPO"
log "code $(git rev-parse HEAD) torch $(python -c 'import torch; print(torch.__version__)') gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 0b. stages finished on an earlier runtime
for d in "$DRIVE"/*/; do
  name=$(basename "$d")
  if [ "$name" = live ] || [ -f "$OUT/$name/model.npz" ] || [ ! -f "$d/model.npz" ] || [ ! -f "$d/config.json" ]; then
    continue
  fi
  if python -c "import json, sys, numpy as np; json.load(open(sys.argv[1] + 'config.json')); dict(np.load(sys.argv[1] + 'model.npz'))" "$d"; then
    mkdir -p "$OUT/$name" && cp -r "$d." "$OUT/$name/" && log "restored $name from Drive"
  else
    log "Drive copy of $name is unreadable; it will be rerun"
  fi
done

# 0c. mirror to Drive every 5 minutes from now on, and once more on exit
command -v rsync > /dev/null || apt-get -qq install -y rsync > /dev/null
mirror() { rsync -a --exclude 'targets_train/*.npy' "$OUT/" "$DRIVE/live/" 2>/dev/null || true; }
( while sleep 300; do mirror; done ) &
MIRROR=$!
trap 'mirror; kill $MIRROR 2>/dev/null || true' EXIT

# 0d. data: T0011's data/pf_policy_cache/<split> from Drive, found by the split's row count and array digests in
# POLICY_CACHE.json (its "identity" block is the same for train and val, so it cannot tell the splits apart)
for split in train val; do
  if [ -f "$CACHE/$split.verified" ]; then log "cache $split already verified"; continue; fi
  src=$(python - "$split" "$DRIVE_ROOT" <<'PY'
import json, os, sys
split, root = sys.argv[1], sys.argv[2]
want = json.load(open("reports/models/p_features/POLICY_CACHE.json"))["splits"][split]
for top, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d != "t0032"] if top.count(os.sep) - root.count(os.sep) < 3 else []
    if "CACHE.json" not in files:
        continue
    meta = json.load(open(os.path.join(top, "CACHE.json")))
    if meta.get("rows") == want["rows"] and all(meta.get("arrays", {}).get(k, {}).get("data_sha256") == v["data_sha256"]
                                                for k, v in want["arrays"].items()):
        print(top)
        break
PY
)
  [ -n "$src" ] || { log "no $split cache with the committed identity under $DRIVE_ROOT"; exit 1; }
  log "copy $split cache from $src"
  rm -rf "$CACHE/$split" && mkdir -p "$CACHE/$split"
  cp -r "$src/." "$CACHE/$split/"
  python -m training.distill_policy verify-cache --cache "$CACHE/$split"
  touch "$CACHE/$split.verified"
done

COMMON=(--epochs 8 --batch 1024 --lr 1e-3 --min-lr 1e-4 --weight-decay 1e-5 --patience 2 --plateau-decay 1
        --smoothing 0.01 --act-weight 0.5 --target-weight 0.5 --count-weighting inv_teams)

backup() { mkdir -p "$DRIVE/$1" && cp -r "$OUT/$1/." "$DRIVE/$1/" && cp "$OUT/$1.log" "$DRIVE/$1/" && log "backed up $1"; }

run() {  # run NAME CMD...: skip when done, else run, then back up
  local name=$1; shift
  if [ -f "$OUT/$name/model.npz" ] && [ -f "$OUT/$name/config.json" ]; then log "skip $name (done)"; return; fi
  log "start $name"
  "$@" > "$OUT/$name.log" 2>&1
  log "done $name"
  backup "$name"
}

# 1. teacher: the unchanged trainer, T0011's arguments, 8XL widths
run teacher_8xl_p3 python /content/t0032_train_cache.py --model general --shards data/shards_l3 --out "$OUT/teacher_8xl_p3" \
    --policy-family p3 --hidden-mover 2048 --hidden-squad 8192,2048 --hidden-act 2048 --hidden-target 256 --max-width 8192 \
    --seed 100 --backend torch --device cuda --gpu-resident-gb 6 --gpu-window-rows 1000000 --lr-schedule cosine "${COMMON[@]}"

# 2. teacher targets on the train split
if [ ! -f "$OUT/targets_train/IDENTITY.json" ]; then
  log "start targets_train"
  rm -rf "$OUT/targets_train"
  python -m training.distill_policy teacher-targets --teacher "$OUT/teacher_8xl_p3" --cache "$CACHE/train" --split train \
      --out "$OUT/targets_train" --tau 1.0 --batch 8192 --device cuda > "$OUT/targets_train.log" 2>&1
  log "done targets_train"
  mkdir -p "$DRIVE/targets_train" && cp "$OUT/targets_train/IDENTITY.json" "$OUT/targets_train.log" "$DRIVE/targets_train/"
fi

# 3. students
STUDENT=(--cache "$CACHE/train" --val-cache "$CACHE/val" --student xxl --family p3 --backend torch --device cuda)
for S in 101 102 103; do
  run "warmup_s$S" python -m training.distill_policy train --stage warmup "${STUDENT[@]}" --seed $S "${COMMON[@]}" \
      --out "$OUT/warmup_s$S"
  run "distill_s$S" python -m training.distill_policy train --stage distill "${STUDENT[@]}" --seed $S "${COMMON[@]}" \
      --teacher-shard "$OUT/targets_train" --alpha 0.5 --tau 1.0 --out "$OUT/distill_s$S"
  run "qat_s$S" python -m training.distill_policy train --stage qat --profile mixed4_8 "${STUDENT[@]}" --seed $S "${COMMON[@]}" \
      --teacher-shard "$OUT/targets_train" --alpha 0.5 --tau 1.0 --init "$OUT/distill_s$S" --epochs 2 --lr 1e-4 --min-lr 1e-5 \
      --out "$OUT/qat_s$S"
done
log "ALL_DONE"
```

## t0032_students.sh

The three student seeds side by side, each warmup -> distill -> qat with the driver's commands. SHA-256 `75a5065cbe5bb45b535c3cb98894c82d36adada950476c93433a0d54c32ace34`.

```bash
#!/usr/bin/env bash
# T0032 students, the three seeds side by side (Colab Pro, A100). Driver v3 (t0032_colab_pro.sh) made the teacher and
# the teacher targets, then ran the students one after another; one XXL student kept the A100 at about 26 % (1 GB), so
# it was stopped during warmup_s101 epoch 3 and this script runs seeds 101, 102 and 103 at once. Each seed is the
# driver's sequence with the driver's commands (warmup -> distill -> qat). Results do not depend on the overlap;
# timings are not evidence. BLAS/OpenMP threads are 4 per seed (12 cores).
set -euo pipefail
COMMIT=95e173b9ac87c3f769134669e488630d1f6b35bc
REPO=/content/t31
CACHE=/content/pf_policy_cache
OUT=/content/t0032
DRIVE=/content/drive/MyDrive/cft-fusion/t0032
log() { echo "[t0032] $(date -u +%FT%TZ) $*"; }

[ -f "$OUT/teacher_8xl_p3/model.npz" ] && [ -f "$OUT/targets_train/IDENTITY.json" ] || { log "teacher or targets missing"; exit 1; }
[ -f "$CACHE/train.verified" ] && [ -f "$CACHE/val.verified" ] || { log "caches not verified"; exit 1; }
cd "$REPO"
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || { log "code is not $COMMIT"; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

COMMON=(--epochs 8 --batch 1024 --lr 1e-3 --min-lr 1e-4 --weight-decay 1e-5 --patience 2 --plateau-decay 1
        --smoothing 0.01 --act-weight 0.5 --target-weight 0.5 --count-weighting inv_teams)
STUDENT=(--cache "$CACHE/train" --val-cache "$CACHE/val" --student xxl --family p3 --backend torch --device cuda)

backup() { mkdir -p "$DRIVE/$1" && cp -r "$OUT/$1/." "$DRIVE/$1/" && cp "$OUT/$1.log" "$DRIVE/$1/" && log "backed up $1"; }

run() {  # run NAME CMD...: skip when done, else run, then back up
  local name=$1; shift
  if [ -f "$OUT/$name/model.npz" ] && [ -f "$OUT/$name/config.json" ]; then log "skip $name (done)"; return; fi
  log "start $name"
  "$@" > "$OUT/$name.log" 2>&1
  log "done $name"
  backup "$name"
}

seed() {
  local S=$1
  run "warmup_s$S" python -m training.distill_policy train --stage warmup "${STUDENT[@]}" --seed "$S" "${COMMON[@]}" \
      --out "$OUT/warmup_s$S"
  run "distill_s$S" python -m training.distill_policy train --stage distill "${STUDENT[@]}" --seed "$S" "${COMMON[@]}" \
      --teacher-shard "$OUT/targets_train" --alpha 0.5 --tau 1.0 --out "$OUT/distill_s$S"
  run "qat_s$S" python -m training.distill_policy train --stage qat --profile mixed4_8 "${STUDENT[@]}" --seed "$S" "${COMMON[@]}" \
      --teacher-shard "$OUT/targets_train" --alpha 0.5 --tau 1.0 --init "$OUT/distill_s$S" --epochs 2 --lr 1e-4 --min-lr 1e-5 \
      --out "$OUT/qat_s$S"
}

mirror() { rsync -a --exclude 'targets_train/*.npy' "$OUT/" "$DRIVE/live/" 2>/dev/null || true; }
( while sleep 300; do mirror; done ) &
MIRROR=$!
trap 'mirror; kill $MIRROR 2>/dev/null || true' EXIT

log "students: seeds 101 102 103 side by side, code $(git rev-parse HEAD)"
pids=()
for S in 101 102 103; do
  seed "$S" > "$OUT/lane_s$S.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 0 ]; then log "ALL_DONE"; else log "a seed failed: see $OUT/lane_s*.log"; exit 1; fi
```

## t0032_restart.sh

Written by the notebook's (re)start cell: runs driver v3 and hands over to t0032_students.sh at the first student stage it starts. SHA-256 `82b5577d79378ba5150af6b2bc65e7e75cd804b0ce92118506f6b93856aa45ad`.

```bash
#!/usr/bin/env bash
L=/content/t0032/driver.log
bash /content/t0032_colab_pro.sh >> $L 2>&1 &
D=$!
until grep -qE 'start (warmup|distill|qat)_s' $L || ! kill -0 $D 2>/dev/null; do sleep 5; done
grep -qE 'start (warmup|distill|qat)_s' $L || { echo "[t0032] $(date -u +%FT%TZ) driver v3 ended before the students" >> $L; exit 1; }
pkill -f t0032_colab_pro.sh; pkill -f 'distill_policy train'; sleep 3
for d in /content/t0032/warmup_s* /content/t0032/distill_s* /content/t0032/qat_s*; do [ -d "$d" ] && [ ! -f "$d/model.npz" ] && rm -rf "$d" "$d.log"; done
echo "[t0032] $(date -u +%FT%TZ) driver v3 handed over to t0032_students.sh at $(grep -oE 'start (warmup|distill|qat)_s[0-9]+' $L | tail -n 1 | cut -d' ' -f2)" >> $L
exec bash /content/t0032_students.sh >> $L 2>&1
```

## t0032_eval.py

Calibrated per-head validation loss of the warmup and distill students (T0011's evaluate). SHA-256 `8681ec514781157bae1fd84d47d46d9e6ad693a7e88ba16abf39b935c5691a4a`.

```python
"""T0032: calibrated per-head validation losses of finished warmup/distill students, as T0011 reports them.

training/train_policy.py logs, after calibration, evaluate(model, val, smoothing 0) with the fitted temperatures
applied and inv_teams row weights ("Calibrated joint / act / target (trainer weights)" in T0011's RESULTS.md).
training/distill_policy.py calibrates its students the same way but does not log that evaluation, so this script
does it from each run's model.npz (weights and beta) with the unchanged evaluate. QAT runs are not evaluated here:
their receipt's best_val is already the deployed integer forward (fusion/torch_qat.py), and the export fits
their temperatures. Results are appended to <out>/eval.json, keyed by run and bound to the model's SHA-256.

    python t0032_eval.py --out /content/t0032 --val /content/pf_policy_cache/val warmup_s101 distill_s101 ...
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("REPO", "/content/t31"))
from fusion.models import PolicyNet  # noqa: E402
from fusion.schema import layout as feature_layout  # noqa: E402
from training.distill_policy import LABELS, RowCache  # noqa: E402
from training.train_policy import evaluate  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="directory holding the run directories")
    ap.add_argument("--val", type=Path, required=True, help="validation cache")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    layout = feature_layout()
    val = RowCache(a.val)
    d = {k: val.arr[k] for k in ("rec", *LABELS)}
    path = a.out / "eval.json"
    results = json.loads(path.read_text()) if path.is_file() else {}
    for name in a.runs:
        run = a.out / name
        cfg = json.loads((run / "config.json").read_text())
        if cfg.get("stage") not in ("warmup", "distill"):
            raise SystemExit(f"{name}: stage {cfg.get('stage')} is not evaluated here (warmup and distill only)")
        if cfg["val_cache"] != val.identity() or cfg["val_rows"] != val.rows:
            raise SystemExit(f"{name}: validated on {cfg['val_cache']}, not on this cache {val.identity()}")
        digest = sha256(run / "model.npz")
        if results.get(name, {}).get("model_sha256") == digest:
            print(f"[t0032 eval] {name}: already evaluated", flush=True)
            continue
        w = np.asarray(val.arr["weight"], np.float32)
        if cfg["count_weighting"] == "inv_teams":
            w = (w / np.asarray(val.arr["teams"]).astype(np.float32)).astype(np.float32)
        model = PolicyNet(layout, np.random.default_rng(0), cfg["hidden"], cfg["family"])
        model.load(dict(np.load(run / "model.npz")))
        model.calibrated = True
        loss = cfg["loss"]
        t0 = time.time()
        total, parts = evaluate(model, layout, d | {"weight": w}, {"smoothing": 0.0, "act_weight": loss["act_weight"],
                                                                  "target_weight": loss["target_weight"]})
        results[name] = {"model_sha256": digest, "stage": cfg["stage"], "seed": cfg["seed"],
                         "val_cache": cfg["val_cache"], "val_rows": cfg["val_rows"], "count_weighting": cfg["count_weighting"],
                         "beta": dict(model.beta), "calibrated_total": float(total),
                         **{f"val_{k}": float(v) for k, v in parts.items()}, "seconds": round(time.time() - t0, 1)}
        path.write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
        print(f"[t0032 eval] {name}: " + json.dumps(results[name], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## t0032_bundle.py

Gathers every run's config, receipt and log into RUNS.json. SHA-256 `863539438a59b90c9cbbedc0052b8837800a2c675fc8705b11f0156079373dc3`.

```python
"""T0032: gather every run's receipts into one JSON (reports/models/p_distill/RUNS.json), printed to stdout.

Per run directory: config.json, receipt.json (distill_policy stages), every log.jsonl line, the model's SHA-256;
the teacher-target shard's IDENTITY.json; eval.json (t0032_eval.py); the driver log. Weights and the shard arrays
stay on the runtime and Drive.

    python t0032_bundle.py /content/t0032 > RUNS.json
"""
import hashlib
import json
import sys
from pathlib import Path

RUNS = ("teacher_8xl_p3", *(f"{s}_s{n}" for n in (101, 102, 103) for s in ("warmup", "distill", "qat")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    out = Path(sys.argv[1])
    bundle = {"runs": {}, "missing": []}
    for name in RUNS:
        run = out / name
        if not (run / "model.npz").is_file():
            bundle["missing"].append(name)
            continue
        r = {"model_sha256": sha256(run / "model.npz"),
             "config": json.loads((run / "config.json").read_text()),
             "log": [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines() if x.strip()]}
        if (run / "receipt.json").is_file():
            r["receipt"] = json.loads((run / "receipt.json").read_text())
        bundle["runs"][name] = r
    ident = out / "targets_train" / "IDENTITY.json"
    bundle["teacher_targets"] = json.loads(ident.read_text()) if ident.is_file() else None
    ev = out / "eval.json"
    bundle["eval"] = json.loads(ev.read_text()) if ev.is_file() else {}
    bundle["driver_log"] = (out / "driver.log").read_text().splitlines()
    print(json.dumps(bundle, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## t0032_summarize.py

Writes RESULTS.md from RUNS.json and T0011's SUMMARY.json (run on the Mac from the repository root). SHA-256 `3f283d9f85a5cef791b66d34a37934a66fcfe85eddd0b059de262c3fd599c94f`.

```python
"""T0032: write reports/models/p_distill/RESULTS.md from RUNS.json (t0032_bundle.py) and T0011's SUMMARY.json.

    python reports/models/p_distill/scripts/t0032_summarize.py   # from the repository root
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "reports/models/p_distill"
SEEDS = (101, 102, 103)


def f(x, n=5):
    return "–" if x is None else f"{x:.{n}f}"


def d(x, n=4):
    return "–" if x is None else f"{x:+.{n}f}"


def main() -> int:
    b = json.loads((OUT / "RUNS.json").read_text())
    ref = json.loads((ROOT / "reports/models/p_features/SUMMARY.json").read_text())
    runs, ev = b["runs"], b["eval"]
    p3 = sorted((r for r in ref["runs"] if r["family"] == "p3"), key=lambda r: r["seed"])
    t = runs["teacher_8xl_p3"]
    t_epochs = [x for x in t["log"] if "epoch" in x]
    t_cal = next(x for x in t["log"] if x.get("stage") == "calibrated")
    epochs = {name: [x for x in r["log"] if "epoch" in x] for name, r in runs.items()}
    best = {name: r["receipt"]["best_val"] for name, r in runs.items() if "receipt" in r}
    ref_best = [min(e["val"] for e in r["epochs"]) for r in p3]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    L = ["# T0032: P3 distillation from an 8XL teacher into XXL students, with QAT", "",
         "Generated by `scripts/t0032_summarize.py` from `RUNS.json` (every run's config, receipt and log, gathered on "
         "Colab by `scripts/t0032_bundle.py`) and T0011's `reports/models/p_features/SUMMARY.json`. "
         "Offline training and validation only: no test split, no export, no timing, no games.", "",
         "Losses are T0011's: the supervised validation objective (joint cross-entropy + 0.5 act + 0.5 target, "
         "label smoothing 0.01, inv_teams row weights, raw logits) for every stage, and the calibrated per-head "
         "losses (smoothing 0, fitted temperatures applied) for the float models. Lower is better. "
         "Three student seeds give per-seed results; no interval is claimed.", "",
         "## Teacher: 8XL p3, seed 100", "",
         "Unchanged trainer (training/train_policy.py) through `scripts/t0032_train_cache.py`, T0011's arguments, widths "
         "mover 2048, squad 8192-2048, act 2048, target 256. Offline only (too slow for the 50 ms budget).", "",
         "| Epoch | 8XL val | T0011 P3 XXL seeds 1-3 mean | Difference | XXL seed range |", "|---:|---:|---:|---:|---:|"]
    for i, e in enumerate(t_epochs):
        xs = [r["epochs"][i]["val"] for r in p3]
        L.append(f"| {e['epoch']} | {f(e['val'])} | {f(mean(xs))} | {d(e['val'] - mean(xs))} | {f(max(xs) - min(xs))} |")
    L += ["", "| Calibrated head | 8XL | T0011 P3 XXL mean | Difference | XXL seed range |", "|---|---:|---:|---:|---:|"]
    for h in ("joint", "act", "target"):
        xs = [r["calibrated"][f"val_{h}"] for r in p3]
        L.append(f"| {h} | {f(t_cal['val_' + h])} | {f(mean(xs))} | {d(t_cal['val_' + h] - mean(xs), 5)} | {f(max(xs) - min(xs))} |")
    tt = b["teacher_targets"] or {}
    L += ["", f"Teacher targets: tau {tt.get('tau')}, {tt.get('rows')} train rows, row digest `{tt.get('row_digest')}`, "
          f"mask digest `{tt.get('mask_digest')}`.", "",
          "## Students: XXL p3, seeds 101-103", "",
          "warmup is the supervised control (alpha 1) through the same code path as distill (alpha 0.5, tau 1). "
          "qat is mixed4_8 quantization-aware training from each distill model (2 epochs, lr 1e-4 to 1e-5); its loss "
          "is the deployed integer forward (fusion/torch_qat.py) on raw logits.", "",
          "| Seed | warmup best val | distill best val | distill - warmup | qat best val (integer) | qat - distill |",
          "|---:|---:|---:|---:|---:|---:|"]
    diffs, qd = [], []
    for s in SEEDS:
        w, di, q = best.get(f"warmup_s{s}"), best.get(f"distill_s{s}"), best.get(f"qat_s{s}")
        diffs.append(di - w)
        qd.append(q - di)
        L.append(f"| {s} | {f(w)} | {f(di)} | {d(di - w)} | {f(q)} | {d(q - di)} |")
    L.append(f"| mean | {f(mean([best[f'warmup_s{s}'] for s in SEEDS]))} | {f(mean([best[f'distill_s{s}'] for s in SEEDS]))} | "
             f"{d(mean(diffs))} | {f(mean([best[f'qat_s{s}'] for s in SEEDS]))} | {d(mean(qd))} |")
    L += ["", f"T0011 P3 XXL best val, seeds 1-3: {', '.join(f(x) for x in ref_best)} (mean {f(mean(ref_best))}).", "",
          "Distill minus warmup per epoch:", "", "| Epoch | " + " | ".join(f"s{s}" for s in SEEDS) + " |",
          "|---:|" + "---:|" * len(SEEDS)]
    for i in range(len(epochs["warmup_s101"])):
        L.append(f"| {i + 1} | " + " | ".join(d(epochs[f'distill_s{s}'][i]['val'] - epochs[f'warmup_s{s}'][i]['val'])
                                             for s in SEEDS) + " |")
    if ev:
        L += ["", "Calibrated per-head validation loss (scripts/t0032_eval.py: T0011's evaluate with smoothing 0 and the "
              "fitted temperatures):", "",
              "| Seed | Arm | joint | act | target | beta joint / act / target |", "|---:|---|---:|---:|---:|---|"]
        for s in SEEDS:
            for arm in ("warmup", "distill"):
                e = ev.get(f"{arm}_s{s}")
                if e:
                    bt = e["beta"]
                    L.append(f"| {s} | {arm} | {f(e['val_joint'])} | {f(e['val_act'])} | {f(e['val_target'])} | "
                             f"{bt['joint']:.4f} / {bt['act']:.4f} / {bt['target']:.4f} |")
        for h in ("joint", "act", "target"):
            dd = [ev[f"distill_s{s}"][f"val_{h}"] - ev[f"warmup_s{s}"][f"val_{h}"] for s in SEEDS
                  if f"distill_s{s}" in ev and f"warmup_s{s}" in ev]
            if dd:
                L.append(f"\ndistill - warmup, calibrated {h}: {', '.join(d(x, 5) for x in dd)} (mean {d(mean(dd), 5)}).")
        L.append("\nT0011 P3 XXL calibrated (trainer weights), seeds 1-3: " + "; ".join(
            f"{r['calibrated']['val_joint']:.5f} / {r['calibrated']['val_act']:.5f} / {r['calibrated']['val_target']:.5f}"
            for r in p3) + ".")
    L += ["", "## Reading", "",
          f"- Distillation from the 8XL teacher lowered the XXL student's supervised validation loss on every seed "
          f"({', '.join(d(x) for x in diffs)}; mean {d(mean(diffs))}). The teacher itself is "
          f"{d(t_epochs[-1]['val'] - mean([best[f'warmup_s{s}'] for s in SEEDS]))} from the warmup mean, so the student "
          f"recovers about {abs(mean(diffs)) / abs(t_epochs[-1]['val'] - mean([best[f'warmup_s{s}'] for s in SEEDS])):.0%} of it.",
          "- The gap is largest early (about -0.04 after epoch 1) and narrows as the schedule anneals; part of the effect "
          "is faster learning.",
          "- Calibrated per head, the whole gain is in the joint (movement) head; the act head is slightly worse on every "
          "seed and the target head does not move. This mirrors where the teacher itself beats XXL (joint -0.028, act "
          "-0.0008, target 0). The warmup arm reproduces T0011's P3 XXL per head.",
          f"- mixed4_8 QAT from the distilled models costs {', '.join(d(x) for x in qd)} against their float loss, "
          "more than distillation gained. No supervised model was quantized, so whether distillation still helps after "
          "QAT is not measured here.",
          "- These are prediction losses on T0011's validation split. They do not establish playing strength, the "
          "50 ms budget, or out-of-distribution behaviour; those need an export and host games.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## t0032_run.sh

Driver v2, used only by runtime 1 (lost); kept for the record. SHA-256 `1cea884a8f228e3d35d9694fc30d8a406642da5bfaef2e728d4d7e11c2875576`.

```bash
#!/usr/bin/env bash
# T0032 driver (Colab A100): 8XL p3 teacher -> teacher targets -> 3 seeds x (warmup, distill, qat).
# Resumable: a stage whose model.npz (targets: IDENTITY.json) exists is skipped. Each finished run directory is
# copied to Drive (without the teacher shard). Code: /content/t31 at the commit printed first.
set -euo pipefail
REPO=/content/t31
CACHE=/content/pf_policy_cache
OUT=/content/t0032
DRIVE=/content/drive/MyDrive/cft-fusion/t0032
mkdir -p "$OUT" "$DRIVE"
cd "$REPO"
echo "[t0032] $(date -u +%FT%TZ) code $(git rev-parse HEAD) torch $(python -c 'import torch; print(torch.__version__)') gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"

COMMON=(--epochs 8 --batch 1024 --lr 1e-3 --min-lr 1e-4 --weight-decay 1e-5 --patience 2 --plateau-decay 1
        --smoothing 0.01 --act-weight 0.5 --target-weight 0.5 --count-weighting inv_teams)

backup() { mkdir -p "$DRIVE/$1" && cp -r "$OUT/$1/." "$DRIVE/$1/" && echo "[t0032] $(date -u +%T) backed up $1"; }

run() {  # run NAME CMD...: skip when done, else run, then back up
  local name=$1; shift
  if [ -f "$OUT/$name/model.npz" ] && [ -f "$OUT/$name/config.json" ]; then echo "[t0032] skip $name (done)"; return; fi
  echo "[t0032] $(date -u +%T) start $name"
  "$@" > "$OUT/$name.log" 2>&1
  echo "[t0032] $(date -u +%T) done $name"
  backup "$name"
}

# 1. teacher: the unchanged trainer, T0011's arguments, 8XL widths (6XL was stopped after one epoch for this, 2026-10-11)
run teacher_8xl_p3 python /content/t0032_train_cache.py --model general --shards data/shards_l3 --out "$OUT/teacher_8xl_p3" \
    --policy-family p3 --hidden-mover 2048 --hidden-squad 8192,2048 --hidden-act 2048 --hidden-target 256 --max-width 8192 \
    --seed 100 --backend torch --device cuda --gpu-resident-gb 6 --gpu-window-rows 1000000 --lr-schedule cosine "${COMMON[@]}"

# 2. teacher targets on the train split
if [ ! -f "$OUT/targets_train/IDENTITY.json" ]; then
  echo "[t0032] $(date -u +%T) start targets_train"
  rm -rf "$OUT/targets_train"
  python -m training.distill_policy teacher-targets --teacher "$OUT/teacher_8xl_p3" --cache "$CACHE/train" --split train \
      --out "$OUT/targets_train" --tau 1.0 --batch 8192 --device cuda > "$OUT/targets_train.log" 2>&1
  echo "[t0032] $(date -u +%T) done targets_train"
  mkdir -p "$DRIVE/targets_train" && cp "$OUT/targets_train/IDENTITY.json" "$OUT/targets_train.log" "$DRIVE/targets_train/"
fi

# 3. students
STUDENT=(--cache "$CACHE/train" --val-cache "$CACHE/val" --student xxl --family p3 --backend torch --device cuda)
for S in 101 102 103; do
  run "warmup_s$S" python -m training.distill_policy train --stage warmup "${STUDENT[@]}" --seed $S "${COMMON[@]}" \
      --out "$OUT/warmup_s$S"
  run "distill_s$S" python -m training.distill_policy train --stage distill "${STUDENT[@]}" --seed $S "${COMMON[@]}" \
      --teacher-shard "$OUT/targets_train" --alpha 0.5 --tau 1.0 --out "$OUT/distill_s$S"
  run "qat_s$S" python -m training.distill_policy train --stage qat --profile mixed4_8 "${STUDENT[@]}" --seed $S "${COMMON[@]}" \
      --teacher-shard "$OUT/targets_train" --alpha 0.5 --tau 1.0 --init "$OUT/distill_s$S" --epochs 2 --lr 1e-4 --min-lr 1e-5 \
      --out "$OUT/qat_s$S"
done
echo "[t0032] $(date -u +%FT%TZ) ALL_DONE"
```
