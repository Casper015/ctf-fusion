# Exported models

`training/export_weights.py` writes one `<tier>.hpp` per tier here (int16 weights as constexpr arrays), its `<tier>.npz`
twin for the Python emulation, and `MANIFEST.json` (hashes, sources, alpha per team count, sizes). Nothing is exported
yet: models are trained on cluster data first (docs/REVIEW.md, E1).
