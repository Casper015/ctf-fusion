# Size Tier XL Export Summary

> **Training record (2026-10-10; replaces an incorrect note of 2026-10-09):** the `general` model's P and V were trained with `inv_teams` count weighting, as the message of commit 89ee436 says; the single-count models (`n02`…`n15`) record `none`, which is expected because each sees one team count. The earlier note claimed every manifest records `none`; that was wrong. All sizes S–XXL used seed 1 and 8 epochs; M used the NumPy trainer and L–XXL the PyTorch trainer, so compare sizes from L up (`bots/p_only/TASK.md`, evidence backfill).

- Models exported: 11 / 11
- Backend: torch (cuda)
- Required FUSION_MAX_WIDTH: 1024

| Model | Params | Mover | Squad | Act | Tgt | Val | Per-Team MACs | Gated Alphas |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `general` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 11: 0.979, 13: 0.977, 14: 0.978, 9: 0.971 |
| `n02` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 2: 0.992 |
| `n03` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 3: 0.981 |
| `n04` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 4: 0.983 |
| `n05` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 5: 0.960 |
| `n06` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 6: 0.978 |
| `n07` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 7: 0.976 |
| `n08` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 8: 0.977 |
| `n10` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 10: 0.974 |
| `n12` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 12: 0.981 |
| `n15` | 329,380 | 5377 | 195965 | 5764 | 289 | 121985 | 273,920 | 15: 0.981 |
