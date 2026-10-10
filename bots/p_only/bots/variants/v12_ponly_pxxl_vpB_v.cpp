// v12 P-only carrier, frozen policy XXL, value pilot arm B (v3, M 256/32), V on (experimental; value_approved false).
// Model set inference/models_pxxl_vpB/: python -m training.export_mixed --policy-tier xxl --value-run runs/v_pilot_B_s1 --out inference/models_pxxl_vpB
// (bots/p_only/profiles/variants_families.json, docs/FEATURE_FAMILIES.md)
#define PONLY_POLICY_FAMILY 2
#define PONLY_VALUE_FAMILY 3
#define MOD_FUSION_V 1
#include "models_pxxl_vpB/registry.hpp"
#include "v12_ponly_f.hpp"
