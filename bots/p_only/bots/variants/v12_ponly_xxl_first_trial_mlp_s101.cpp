// v12 P-only carrier, policy XXL (p2), V off, own-action decision scorer (task T0051)
// first_trial_mlp_s101: MLP [64, 16] on own-action-1 features, lambda=100.0, zscore rule.
#define FUSION_TIER_XXL 1
#define PONLY_DECISION 2
#define PONLY_DSCORE_DATA_CARRIER "v12_ponly_xxl_dlog"
#include "own_action_scorers/first_trial_mlp_s101/scorer_model.hpp"
#include "v12_ponly_fs.hpp"
