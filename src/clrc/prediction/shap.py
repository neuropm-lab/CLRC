"""SHAP analysis for XGBoost LOBO models."""

from __future__ import annotations

import logging
import pickle
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import shap

    HAS_SHAP = True
except ImportError:
    shap = None  # ty: ignore[invalid-assignment]
    HAS_SHAP = False

logger = logging.getLogger(__name__)


class _FoldUnpickler(pickle.Unpickler):
    """Redirect old FoldArtifact pickle references to clrc.core.types."""

    def find_class(self, module: str, name: str):
        if name == "FoldArtifact":
            from clrc.core.types import FoldArtifact
            return FoldArtifact
        return super().find_class(module, name)


def load_fold(path: Path) -> Optional[object]:
    """Load a single fold pickle, returning None on failure."""
    try:
        with path.open("rb") as f:
            return _FoldUnpickler(f).load()
    except FileNotFoundError:
        warnings.warn(f"Fold file not found and skipped: {path}")
    except Exception as exc:
        warnings.warn(f"Failed to load fold file {path}: {exc}")
    return None


def booster_from_fold(fold_obj: object) -> xgb.Booster:
    """Reconstruct XGBoost Booster from a FoldArtifact's model_raw bytes."""
    model_raw = getattr(fold_obj, "model_raw", None)
    if model_raw is None:
        raise ValueError("Fold artifact missing model_raw.")
    booster = xgb.Booster()
    try:
        booster.load_model(bytearray(model_raw))
    except TypeError:
        # Older XGBoost versions accept a file-like object but not bytearray.
        booster.load_model(BytesIO(model_raw))  # ty: ignore[invalid-argument-type]
    return booster


def run_shap_analysis(
    model: xgb.Booster,
    X: np.ndarray,
    feature_names: List[str],
    sample_size: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, object, np.ndarray]:
    """Run TreeSHAP on an XGBoost Booster.

    Returns (shap_values, explainer, X_sample).
    """
    if not HAS_SHAP:
        raise RuntimeError("shap package is required but not installed.")

    rng = np.random.RandomState(seed)

    if X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    dmat = xgb.DMatrix(X_sample, feature_names=feature_names)
    explainer = shap.TreeExplainer(model)  # ty: ignore[unresolved-attribute]
    shap_values = explainer.shap_values(dmat)

    return shap_values, explainer, X_sample


def analyze_feature_directionality(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 50,
) -> pd.DataFrame:
    """Analyse directionality of each feature's SHAP contribution.

    For every feature computes: mean |SHAP|, Pearson r(feature, SHAP),
    fraction of positive SHAP when feature > median, and direction label.
    Returns DataFrame sorted by descending mean |SHAP|.
    """
    results: List[Dict] = []

    for i, fname in enumerate(feature_names):
        feat_vals = X[:, i]
        shap_vals = shap_values[:, i]

        mean_abs_shap = float(np.mean(np.abs(shap_vals)))

        if np.std(feat_vals) > 0 and np.std(shap_vals) > 0:
            corr = float(np.corrcoef(feat_vals, shap_vals)[0, 1])
        else:
            corr = 0.0

        median_feat = np.median(feat_vals)
        high_feat_mask = feat_vals > median_feat
        if high_feat_mask.sum() > 0:
            frac_positive_when_high = float(np.mean(shap_vals[high_feat_mask] > 0))
        else:
            frac_positive_when_high = float("nan")

        if high_feat_mask.sum() > 0 and (~high_feat_mask).sum() > 0:
            mean_shap_high = float(np.mean(shap_vals[high_feat_mask]))
            mean_shap_low = float(np.mean(shap_vals[~high_feat_mask]))
        else:
            mean_shap_high = float("nan")
            mean_shap_low = float("nan")

        direction = (
            "positive" if corr > 0.1 else ("negative" if corr < -0.1 else "mixed")
        )

        results.append(
            {
                "feature": fname,
                "mean_abs_shap": mean_abs_shap,
                "correlation": corr,
                "frac_positive_when_high": frac_positive_when_high,
                "mean_shap_high_feat": mean_shap_high,
                "mean_shap_low_feat": mean_shap_low,
                "direction": direction,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df
