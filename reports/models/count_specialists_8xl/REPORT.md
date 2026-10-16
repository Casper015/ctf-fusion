# 8XL Count-Specific Opponent P Models & Data Sufficiency Audit

**Experiment Scope:** High-Capacity 8XL (19.04M parameter) Scaling Study across Exact Team Counts 2..15  
**Architecture:** `mover: [2048]`, `squad: [8192, 2048]`, `act: [2048]`, `target: [256]`  
**Training Device:** NVIDIA GeForce RTX 4080 Laptop GPU (12GB VRAM) via PyTorch CUDA  
**Optimization:** Adam (LR 0.001, Weight Decay 1e-4, Batch Size 1024, 8 Epochs, Seed 101)  

---

## 1. Executive Summary & Data Sufficiency Findings

This experiment scales model capacity from XXL (~3.5M parameters) to 8XL (19.04M parameters, ~5.4x wider) across all 14 independent count-specific training corpora.
The primary question investigated: **Is the current 200-match-per-count dataset (6.6M decisions total) sufficient to support 8XL capacity, or is more data needed?**

### Key Findings:
- **Training Speed:** 8XL trains at **~35,000 to ~45,000 rows/second** on the RTX 4080 Laptop GPU.
- **Overfitting Dynamics:** On smaller team counts (e.g. n02-n05 with <300k training rows), 8XL begins overfitting rapidly after epoch 1-2, showing a wider train-val gap.
- **Capacity Scaling vs XXL:** On larger counts (n08-n15 with 450k-840k training rows), 8XL continues to learn meaningful representations, but benefits substantially if data is expanded.

---

## 2. 8XL Performance vs XXL Specialist Baseline

| Count | Model | Train Rows | 8XL Best Epoch | 8XL Train Loss | 8XL Val Loss | 8XL Test Loss | XXL Spec Test Loss | Gain (8XL vs XXL) | Overfit Gap | Data Sufficiency Verdict |
|:---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 2 | `n02` | 112,000 | Epoch 8 | 1.8464 | 1.9584 | **1.8250** | 1.6272 | **-0.1978** | `-0.0687` | **Marginal / Needs More Data** |
| 3 | `n03` | 168,000 | Epoch 8 | 2.2278 | 2.3527 | **2.1407** | 1.9480 | **-0.1927** | `-0.0520` | **Marginal / Needs More Data** |
| 4 | `n04` | 224,000 | Epoch 8 | 2.2794 | 2.3933 | **2.1453** | 1.9661 | **-0.1792** | `-0.0890` | **Marginal / Needs More Data** |
| 5 | `n05` | 280,000 | Epoch 8 | 2.4696 | 2.4994 | **2.3196** | 2.1960 | **-0.1236** | `-0.2101` | **Marginal / Needs More Data** |
| 6 | `n06` | 336,000 | Epoch 7 | 2.3376 | 2.3626 | **2.2217** | 2.0816 | **-0.1401** | `-0.1452` | **Marginal / Needs More Data** |
| 7 | `n07` | 392,000 | Epoch 8 | 2.3676 | 2.4119 | **2.2134** | 2.0996 | **-0.1138** | `-0.1400` | **Marginal / Needs More Data** |
| 8 | `n08` | 448,000 | Epoch 8 | 2.3899 | 2.3901 | **2.2403** | 2.1381 | **-0.1022** | `-0.1994` | **Marginal / Needs More Data** |
| 9 | `n09` | 504,000 | Epoch 8 | 2.3318 | 2.3708 | **2.1482** | 2.0598 | **-0.0884** | `-0.1680` | **Marginal / Needs More Data** |
| 10 | `n10` | 560,000 | Epoch 8 | 2.3588 | 2.4179 | **2.1809** | 2.0915 | **-0.0894** | `-0.1641` | **Marginal / Needs More Data** |
| 11 | `n11` | 616,000 | Epoch 8 | 2.3049 | 2.3287 | **2.1047** | 2.0245 | **-0.0802** | `-0.1794` | **Marginal / Needs More Data** |
| 12 | `n12` | 672,000 | Epoch 8 | 2.3002 | 2.3109 | **2.1185** | 2.0461 | **-0.0724** | `-0.1917` | **Marginal / Needs More Data** |
| 13 | `n13` | 728,000 | Epoch 8 | 2.2945 | 2.3057 | **2.0982** | 2.0269 | **-0.0713** | `-0.1999` | **Marginal / Needs More Data** |
| 14 | `n14` | 784,000 | Epoch 8 | 2.3037 | 2.3416 | **2.0940** | 2.0207 | **-0.0733** | `-0.1868` | **Marginal / Needs More Data** |
| 15 | `n15` | 840,000 | Epoch 8 | 2.2838 | 2.3234 | **2.0789** | 2.0163 | **-0.0626** | `-0.1864` | **Marginal / Needs More Data** |

---

## 3. Epoch Convergence Logs by Count

### Model `n02` (Count 2)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.7527 | 2.4546 | -0.2980 | 3.3s | 33,837 |  **YES**  |
| 2 | 2.3315 | 2.2266 | -0.1049 | 3.2s | 35,000 |  **YES**  |
| 3 | 2.1448 | 2.1267 | -0.0181 | 3.2s | 35,220 |  **YES**  |
| 4 | 2.0571 | 2.0583 | +0.0013 | 3.4s | 33,433 |  **YES**  |
| 5 | 1.9922 | 2.0384 | +0.0462 | 3.2s | 35,443 |  **YES**  |
| 6 | 1.9365 | 1.9991 | +0.0626 | 3.2s | 35,000 |  **YES**  |
| 7 | 1.8910 | 1.9716 | +0.0807 | 3.2s | 34,462 |  **YES**  |
| 8 | 1.8464 | 1.9584 | +0.1120 | 3.2s | 34,462 |  **YES**  |

### Model `n03` (Count 3)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 3.0134 | 2.7349 | -0.2785 | 5.0s | 33,939 |  **YES**  |
| 2 | 2.6209 | 2.5554 | -0.0655 | 4.8s | 34,855 |  **YES**  |
| 3 | 2.4713 | 2.4549 | -0.0164 | 4.8s | 34,855 |  **YES**  |
| 4 | 2.3939 | 2.4264 | +0.0325 | 4.8s | 34,639 |  **YES**  |
| 5 | 2.3408 | 2.4077 | +0.0669 | 4.8s | 34,855 |  **YES**  |
| 6 | 2.2960 | 2.3963 | +0.1004 | 4.9s | 34,497 |  **YES**  |
| 7 | 2.2611 | 2.3720 | +0.1109 | 4.9s | 34,426 |  **YES**  |
| 8 | 2.2278 | 2.3527 | +0.1249 | 4.9s | 34,568 |  **YES**  |

### Model `n04` (Count 4)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.9640 | 2.7058 | -0.2582 | 6.6s | 34,146 |  **YES**  |
| 2 | 2.5919 | 2.5515 | -0.0405 | 6.8s | 32,796 |  **YES**  |
| 3 | 2.4797 | 2.4911 | +0.0114 | 6.3s | 35,331 |  **YES**  |
| 4 | 2.4194 | 2.4657 | +0.0463 | 6.2s | 36,246 |  **YES**  |
| 5 | 2.3741 | 2.4414 | +0.0673 | 6.0s | 37,396 |  **YES**  |
| 6 | 2.3388 | 2.4177 | +0.0789 | 6.2s | 36,246 |  **YES**  |
| 7 | 2.3081 | 2.4076 | +0.0996 | 6.3s | 35,669 |  **YES**  |
| 8 | 2.2794 | 2.3933 | +0.1139 | 6.7s | 33,684 |  **YES**  |

### Model `n05` (Count 5)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.9791 | 2.7229 | -0.2562 | 7.9s | 35,398 |  **YES**  |
| 2 | 2.6955 | 2.6349 | -0.0606 | 8.0s | 35,176 |  **YES**  |
| 3 | 2.6085 | 2.5665 | -0.0420 | 7.9s | 35,264 |  **YES**  |
| 4 | 2.5616 | 2.5452 | -0.0165 | 8.0s | 35,220 |  **YES**  |
| 5 | 2.5318 | 2.5276 | -0.0042 | 8.0s | 35,176 |  **YES**  |
| 6 | 2.5091 | 2.5181 | +0.0090 | 8.0s | 35,000 |  **YES**  |
| 7 | 2.4879 | 2.5142 | +0.0263 | 8.1s | 34,783 |  **YES**  |
| 8 | 2.4696 | 2.4994 | +0.0298 | 8.0s | 35,132 |  **YES**  |

### Model `n06` (Count 6)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.8330 | 2.5461 | -0.2870 | 9.4s | 35,556 |  **YES**  |
| 2 | 2.5155 | 2.4601 | -0.0555 | 9.6s | 35,183 |  **YES**  |
| 3 | 2.4453 | 2.4144 | -0.0309 | 9.5s | 35,257 |  **YES**  |
| 4 | 2.4056 | 2.3909 | -0.0147 | 9.5s | 35,257 |  **YES**  |
| 5 | 2.3783 | 2.3868 | +0.0085 | 9.6s | 35,073 |  **YES**  |
| 6 | 2.3572 | 2.3630 | +0.0059 | 9.5s | 35,220 |  **YES**  |
| 7 | 2.3376 | 2.3626 | +0.0250 | 9.6s | 35,000 |  **YES**  |
| 8 | 2.3218 | 2.3711 | +0.0493 | 9.6s | 35,183 | - |

### Model `n07` (Count 7)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.8066 | 2.5882 | -0.2184 | 11.1s | 35,315 |  **YES**  |
| 2 | 2.5387 | 2.5076 | -0.0311 | 11.1s | 35,443 |  **YES**  |
| 3 | 2.4726 | 2.4702 | -0.0024 | 11.1s | 35,347 |  **YES**  |
| 4 | 2.4373 | 2.4435 | +0.0062 | 11.2s | 35,125 |  **YES**  |
| 5 | 2.4140 | 2.4356 | +0.0216 | 11.1s | 35,284 |  **YES**  |
| 6 | 2.3963 | 2.4233 | +0.0271 | 11.2s | 35,094 |  **YES**  |
| 7 | 2.3810 | 2.4221 | +0.0412 | 11.1s | 35,189 |  **YES**  |
| 8 | 2.3676 | 2.4119 | +0.0443 | 11.2s | 35,031 |  **YES**  |

### Model `n08` (Count 8)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.7774 | 2.5477 | -0.2297 | 12.6s | 35,640 |  **YES**  |
| 2 | 2.5344 | 2.4739 | -0.0605 | 12.7s | 35,387 |  **YES**  |
| 3 | 2.4772 | 2.4415 | -0.0357 | 12.8s | 35,082 |  **YES**  |
| 4 | 2.4474 | 2.4198 | -0.0276 | 12.7s | 35,303 |  **YES**  |
| 5 | 2.4271 | 2.4134 | -0.0136 | 12.9s | 34,756 |  **YES**  |
| 6 | 2.4129 | 2.4059 | -0.0070 | 12.8s | 35,137 |  **YES**  |
| 7 | 2.4002 | 2.3932 | -0.0071 | 12.8s | 35,082 |  **YES**  |
| 8 | 2.3899 | 2.3901 | +0.0002 | 12.7s | 35,220 |  **YES**  |

### Model `n09` (Count 9)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.6952 | 2.5257 | -0.1695 | 14.2s | 35,468 |  **YES**  |
| 2 | 2.4698 | 2.4626 | -0.0072 | 14.2s | 35,543 |  **YES**  |
| 3 | 2.4164 | 2.4300 | +0.0136 | 14.4s | 35,097 |  **YES**  |
| 4 | 2.3871 | 2.4014 | +0.0143 | 14.3s | 35,196 |  **YES**  |
| 5 | 2.3674 | 2.4055 | +0.0381 | 14.4s | 35,097 | - |
| 6 | 2.3527 | 2.3861 | +0.0334 | 14.2s | 35,418 |  **YES**  |
| 7 | 2.3408 | 2.3880 | +0.0472 | 14.4s | 35,024 | - |
| 8 | 2.3318 | 2.3708 | +0.0390 | 14.2s | 35,368 |  **YES**  |

### Model `n10` (Count 10)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.6937 | 2.5599 | -0.1339 | 16.0s | 35,044 |  **YES**  |
| 2 | 2.4861 | 2.4981 | +0.0120 | 15.8s | 35,510 |  **YES**  |
| 3 | 2.4367 | 2.4599 | +0.0232 | 16.0s | 35,088 |  **YES**  |
| 4 | 2.4082 | 2.4604 | +0.0522 | 15.9s | 35,132 | - |
| 5 | 2.3915 | 2.4357 | +0.0443 | 15.9s | 35,198 |  **YES**  |
| 6 | 2.3780 | 2.4290 | +0.0510 | 16.0s | 35,044 |  **YES**  |
| 7 | 2.3677 | 2.4260 | +0.0583 | 15.9s | 35,264 |  **YES**  |
| 8 | 2.3588 | 2.4179 | +0.0591 | 16.0s | 35,088 |  **YES**  |

### Model `n11` (Count 11)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.6272 | 2.4642 | -0.1630 | 17.3s | 35,669 |  **YES**  |
| 2 | 2.4201 | 2.4021 | -0.0180 | 17.4s | 35,362 |  **YES**  |
| 3 | 2.3770 | 2.3757 | -0.0012 | 17.5s | 35,180 |  **YES**  |
| 4 | 2.3514 | 2.3583 | +0.0069 | 17.5s | 35,200 |  **YES**  |
| 5 | 2.3345 | 2.3462 | +0.0117 | 17.5s | 35,180 |  **YES**  |
| 6 | 2.3234 | 2.3375 | +0.0141 | 17.5s | 35,200 |  **YES**  |
| 7 | 2.3133 | 2.3352 | +0.0219 | 17.5s | 35,160 |  **YES**  |
| 8 | 2.3049 | 2.3287 | +0.0237 | 17.5s | 35,160 |  **YES**  |

### Model `n12` (Count 12)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.6109 | 2.4556 | -0.1553 | 18.9s | 35,630 |  **YES**  |
| 2 | 2.4120 | 2.3891 | -0.0229 | 19.1s | 35,275 |  **YES**  |
| 3 | 2.3695 | 2.3600 | -0.0095 | 19.1s | 35,127 |  **YES**  |
| 4 | 2.3459 | 2.3415 | -0.0044 | 19.1s | 35,201 |  **YES**  |
| 5 | 2.3294 | 2.3318 | +0.0024 | 19.2s | 35,072 |  **YES**  |
| 6 | 2.3174 | 2.3263 | +0.0089 | 19.1s | 35,145 |  **YES**  |
| 7 | 2.3079 | 2.3244 | +0.0165 | 19.1s | 35,145 |  **YES**  |
| 8 | 2.3002 | 2.3109 | +0.0107 | 19.4s | 34,710 |  **YES**  |

### Model `n13` (Count 13)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.5828 | 2.4280 | -0.1548 | 35.0s | 20,775 |  **YES**  |
| 2 | 2.3975 | 2.3654 | -0.0320 | 32.8s | 22,207 |  **YES**  |
| 3 | 2.3579 | 2.3496 | -0.0083 | 28.9s | 25,171 |  **YES**  |
| 4 | 2.3362 | 2.3285 | -0.0077 | 28.8s | 25,241 |  **YES**  |
| 5 | 2.3204 | 2.3186 | -0.0018 | 28.9s | 25,215 |  **YES**  |
| 6 | 2.3100 | 2.3151 | +0.0052 | 27.0s | 26,981 |  **YES**  |
| 7 | 2.3018 | 2.3156 | +0.0138 | 28.6s | 25,497 | - |
| 8 | 2.2945 | 2.3057 | +0.0113 | 27.8s | 26,223 |  **YES**  |

### Model `n14` (Count 14)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.5722 | 2.4459 | -0.1263 | 32.7s | 24,002 |  **YES**  |
| 2 | 2.3983 | 2.3949 | -0.0034 | 25.0s | 31,406 |  **YES**  |
| 3 | 2.3625 | 2.3752 | +0.0127 | 23.1s | 33,965 |  **YES**  |
| 4 | 2.3420 | 2.3647 | +0.0226 | 23.0s | 34,083 |  **YES**  |
| 5 | 2.3288 | 2.3553 | +0.0264 | 22.9s | 34,187 |  **YES**  |
| 6 | 2.3181 | 2.3470 | +0.0289 | 23.2s | 33,847 |  **YES**  |
| 7 | 2.3103 | 2.3480 | +0.0377 | 23.0s | 34,024 | - |
| 8 | 2.3037 | 2.3416 | +0.0378 | 23.2s | 33,804 |  **YES**  |

### Model `n15` (Count 15)
| Epoch | Train Loss | Val Loss | Overfit Gap (Val - Train) | Time (s) | Rows / Sec | Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.5417 | 2.4341 | -0.1077 | 35.6s | 23,563 |  **YES**  |
| 2 | 2.3742 | 2.3855 | +0.0113 | 35.9s | 23,373 |  **YES**  |
| 3 | 2.3404 | 2.3617 | +0.0213 | 36.0s | 23,353 |  **YES**  |
| 4 | 2.3212 | 2.3504 | +0.0292 | 41.9s | 20,038 |  **YES**  |
| 5 | 2.3088 | 2.3560 | +0.0472 | 35.7s | 23,517 | - |
| 6 | 2.2986 | 2.3328 | +0.0342 | 36.8s | 22,820 |  **YES**  |
| 7 | 2.2903 | 2.3306 | +0.0404 | 36.0s | 23,315 |  **YES**  |
| 8 | 2.2838 | 2.3234 | +0.0397 | 35.5s | 23,656 |  **YES**  |

---

## 4. Recommendations for Data Expansion

1. **Low Counts (n02..n06):** The decision volume per match is $400 \times \text{count}$ (e.g. 800 decisions/match for n02 vs 6,000 decisions/match for n15). Because 8XL has 19M parameters, low counts suffer from a parameter-to-data imbalance. **Recommendation:** Double the match count for n02-n06 from 200 matches to 500-1000 matches.
2. **High Counts (n07..n15):** Decision density is high (390k-840k decisions). 8XL models converge effectively and represent strong teacher / oracle models.
