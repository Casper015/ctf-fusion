# T0025 delivery checks

The required core suite discovered 224 tests. Its first run completed 223 tests with zero failures/errors and one platform skip; only the report study registration method was deferred while T0030 reserved tests/. The current remote board marks T0030 done. After adding p_ood to MODEL_STUDIES, the one deferred method passed. Combined: 223 passed, one platform skip, zero failures/errors across all 224 tests.

No successful measurement, model prediction, independent audit or completed core test was repeated. Native-versus-actual-NPZ validation covered 28 count/size cases; the independent metric audit checked 168 size/cohort entries. Their receipts and limitations are retained in NATIVE_VALIDATION.json and INDEPENDENT_AUDIT.json.

A delivery preparation command had a Python syntax error before any edits; the first registration check consequently failed because p_ood was absent. Preparation was corrected and the final check passed. This did not affect any measured data or score.

Only reports/models/p_ood/ and the one-line study registration are included in this PR. No production/model/default change. Exact-head PR CI is recorded separately in the user delivery copy.
