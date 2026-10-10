# Size Tier XXL Export Summary

> **Training record (2026-10-10; replaces an incorrect note of 2026-10-09):** the `general` model's P and V were trained with `inv_teams` count weighting, as the message of commit 89ee436 says; the single-count models (`n02`…`n15`) record `none`, which is expected because each sees one team count. The earlier note claimed every manifest records `none`; that was wrong. All sizes S–XXL used seed 1 and 8 epochs; M used the NumPy trainer and L–XXL the PyTorch trainer, so compare sizes from L up (`bots/p_only/TASK.md`, evidence backfill).

- Models exported: 11 / 11
- Backend: torch (cuda)
- Required FUSION_MAX_WIDTH: 2048

| Model | Params | Mover | Squad | Act | Tgt | Val | Per-Team MACs | Gated Alphas |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `general` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 11: 0.983, 13: 0.981, 14: 0.981, 9: 0.977 |
| `n02` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 2: 0.992 |
| `n03` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 3: 0.983 |
| `n04` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 4: 0.986 |
| `n05` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 5: 0.964 |
| `n06` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 6: 0.981 |
| `n07` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 7: 0.977 |
| `n08` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 8: 0.981 |
| `n10` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 10: 0.977 |
| `n12` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 12: 0.983 |
| `n15` | 920,484 | 10753 | 522877 | 11524 | 289 | 375041 | 678,912 | 15: 0.984 |
