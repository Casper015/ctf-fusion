// C08 P-only carrier, policy family p3 retrained at XXL (P feature ablation, run pf_p3_xxl_s1), V off.
// Model set inference/models_pf_p3xxl_vm/: python -m training.export_mixed --policy-run runs/pf_p3_xxl_s1 --value-tier m --out inference/models_pf_p3xxl_vm
// (bots/p_only/profiles/variants_families.json, docs/FEATURE_FAMILIES.md)
#define PONLY_POLICY_FAMILY 3
#define PONLY_VALUE_FAMILY 2
#define MOD_FUSION_V 0
#include "models_pf_p3xxl_vm/registry.hpp"
#include "c08_ponly_f.hpp"
