# Training throughput on DESKTOP-G60OJUD

Model `n02`, 200000 training rows, 1 epoch(s), constant learning rate, seed 1. Windows-10-10.0.19045-SP0; NVIDIA GeForce RTX 4080 Laptop GPU. Epoch time excludes loading; wall time includes it. Validation loss is after the same number of epochs (larger batches take fewer steps).

| Net | Tier | Backend | Batch | Rows/s | Epoch s | Wall s | Speedup vs numpy | Val loss | Peak VRAM GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| policy | L | numpy | 1024 | 26004 | 7.691 | 22.67 | 1.0x | 2.6752 |  |
| policy | L | torch-cuda | 1024 | 98912 | 2.022 | 17.86 | 3.8x | 2.6753 | 0.432 |
| value | L | numpy | 1024 | 161290 | 1.24 | 2.33 | 1.0x | 0.2357 |  |
| value | L | torch-cuda | 1024 | 305344 | 0.655 | 1.78 | 1.9x | 0.2359 | 0.432 |
