---
id: T0012
title: "V pilot: four arms (v2/v3 x M/4XL), batch sweeps, ranking and distillation, offline report"
status: open
owner: ""
needs: [cuda]
writes: [reports/models/v_pilot/, inference/models_pxxl_vpA/, inference/models_pxxl_vpB/, inference/models_pxxl_vpC/, inference/models_pxxl_vpD/, inference/models_pxxl_vpE/]
depends: [T0010]
land: pr
branch: ""
updated: 2026-10-10
---

## Goal

The controlled V pilot of docs/FEATURE_FAMILIES.md section 6 and profiles/v_pilot.json, on T0010's frozen-carrier data, RTX 4080 Laptop 12 GB, FP32 reference. Every run under python tools/repo/task.py lock gpu --task T0012 --. (1) Rows: python -m training.dataset --raw data/p_only/v_data/matches --out data/shards_v12xxl --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --rows designated --workers 8; caches: python -m training.value_cache --shards data/shards_v12xxl --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --family v2 --out data/vcache/v12xxl_v2 and the same with --family v3 --out data/vcache/v12xxl_v3. (2) Batch sweep for C and D: python -m training.value_sweep --cache data/vcache/v12xxl_v3 --value-family v3 --hidden 8192,512 --batches 2048 4096 8192 --steps 300 --device cuda --out reports/models/v_pilot/sweep_v3_4xl_DESKTOP-G60OJUD.json (and v2); pick one batch for all arms (largest that fits with margin; record why). (3) Pilot, seed 1, identical arguments except family and widths: python -m training.train_value --model general --cache data/vcache/v12xxl_<fam> --value-family <fam> --hidden <w> --max-width 8192 --backend torch --device cuda --epochs 40 --patience 3 --plateau-decay 2 --lr 1e-3 --min-lr 1e-5 --batch <B> --seed 1 --checkpoint --out runs/v_pilot_<arm>_s1 for A=v2 256,32; B=v3 256,32; C=v2 8192,512; D=v3 8192,512. (4) Ranking arm: the best shape again with --rank data/rank/v12_ponly_xxl --rank-weight 0.5 (out runs/v_pilot_<arm>R_s1). (5) Distillation (Route B): python -m training.distill_value --teacher runs/v_pilot_D_s1 --cache data/vcache/v12xxl_v3 --soft-weight 0.5 --hidden 256,32 --rank data/rank/v12_ponly_xxl --rank-weight 0.5 --backend torch --device cuda --batch <B> --epochs 40 --checkpoint --out runs/v_pilot_E_s1 (optionally a wide-first student 4096,16). (6) Report: python -m training.evaluate_value --cache data/vcache/v12xxl_v3 --cache-for v2=data/vcache/v12xxl_v2 --split test --run A=runs/v_pilot_A_s1 --run B=... --run C=... --run D=... --run E=runs/v_pilot_E_s1 --rank data/rank/v12_ponly_xxl --compare B:A --compare D:C --compare C:A --compare D:B --compare E:D --compare E:B --out reports/models/v_pilot/test_s1; copy each run's config.json and log.jsonl into reports/models/v_pilot/runs/<arm>_s1/. (7) Finalists (decided on the val split, not test): train seeds 2 and 3 identically, re-run (6) per seed. (8) Export for the host: python -m training.export_mixed --policy-tier xxl --value-run runs/v_pilot_<arm>_s1 --out inference/models_pxxl_vp<arm> for A-E (4XL sets are about 35 MB of header: commit only sets the screen needs). Never claim a training time or speedup that was not measured; lower loss alone is not stronger decisions.

## Done when

PR merged with reports/models/v_pilot/ (sweeps with peak VRAM and throughput, run configs and learning curves, the evaluation report with paired intervals, ranking and regret) and the exported sets inference/models_pxxl_vpA..E needed by T0014.

## Handoff

Not started.

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
