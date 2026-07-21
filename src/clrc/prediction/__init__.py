"""clrc.prediction — XGBoost LOBO training, HPO, SHAP, feature interpretation."""

from clrc.prediction.bias_validation import (
    draw_random_lr_subsets_importance_matched,
    draw_random_lr_subsets_uniform,
    feature_mask_for_lr_subset,
)
from clrc.prediction.bootstrap import (
    compute_feature_level_ci,
    compute_group_level_ci,
)

__all__ = [
    "draw_random_lr_subsets_uniform",
    "draw_random_lr_subsets_importance_matched",
    "feature_mask_for_lr_subset",
    "compute_feature_level_ci",
    "compute_group_level_ci",
]
