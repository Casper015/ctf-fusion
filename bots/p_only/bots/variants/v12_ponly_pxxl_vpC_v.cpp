// v12 P-only carrier, frozen policy XXL, value pilot arm C (v2, 4XL 8192/512), V on (experimental; value_approved false).
// Model set inference/models_pxxl_vpC/: python -m training.export_mixed --policy-tier xxl --value-run runs/v_pilot_C_s1 --out inference/models_pxxl_vpC
// (bots/p_only/profiles/variants_families.json, docs/FEATURE_FAMILIES.md)
#define PONLY_POLICY_FAMILY 2
#define PONLY_VALUE_FAMILY 2
#define MOD_FUSION_V 1
#define PONLY_V_DEADLINE_MS 30  // Route A bound (tuning parameter): no network V after 30 ms of the turn
#include "models_pxxl_vpC/registry.hpp"
#include "v12_ponly_f.hpp"
