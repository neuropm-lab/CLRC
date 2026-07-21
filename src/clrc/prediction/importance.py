"""Feature importance aggregation across LOBO folds.

Takes a collection of trained XGBoost fold boosters (stored as raw bytes
inside ``FoldArtifact`` objects) and produces:

    * Per-fold feature gain vectors  (``compute_fold_importances``).
    * Weighted per-feature importance, absolute and L1-normalized
      (``aggregate_importances``).
    * A feature-level DataFrame annotated with ``lr_name``, ``ct_L``, ``ct_R``
      from the alignment metadata (``build_feature_df``).
    * Group-level aggregated importance by ``lr_name`` / ``ct_L`` / ``ct_R``
      (``aggregate_by_group``).
    * An LR × fold matrix for clustermap visualization
      (``build_lr_fold_matrix``).

The weighting scheme: each fold's gain vector contributes in proportion to
the number of unique edges assigned to that fold (see
:func:`clrc.prediction.evaluation.assign_unique_edges`). This gives an
unbiased edge-weighted mean gain rather than a flat across-fold average.

Contract assumptions
--------------------
* Boosters in ``FoldArtifact.model_raw`` were trained by
  :func:`clrc.prediction.xgboost.train_predict_xgb`, which uses the default
  XGBoost ``f<int>`` feature naming. ``gain_vector`` validates this and
  raises on any violation.
* ``meta`` inputs conform to :class:`clrc.core.types.FeatureMeta` — every
  entry is a dict with all required keys. Validation happens at load time in
  :func:`clrc.core.io.load_alignment_data`; downstream code uses direct
  subscript access without ``.get()`` fallbacks.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from clrc.biology.classification import CELL_CLASS_MAP
from clrc.core.types import FeatureMeta, FoldArtifact

logger = logging.getLogger(__name__)

# Matches XGBoost's default feature naming convention ("f0", "f1", ...).
# Boosters in this project are trained without explicit feature_names, so
# get_score() always returns keys of this form.
_DEFAULT_FEATURE_KEY_RE = re.compile(r"^f(\d+)$")


# ---------------------------------------------------------------------------
#  Per-fold importance extraction
# ---------------------------------------------------------------------------

def _booster_from_bytes(model_raw: bytes) -> xgb.Booster:
    booster = xgb.Booster()
    try:
        booster.load_model(bytearray(model_raw))
    except TypeError:
        # Older XGBoost versions accept a file-like object but not bytearray.
        booster.load_model(BytesIO(model_raw))  # ty: ignore[invalid-argument-type]
    return booster


def gain_vector(model_raw: bytes, n_features: int) -> np.ndarray:
    """Extract an ``(n_features,)`` gain vector from a serialized booster.

    Contract
    --------
    The booster must have been trained via
    :func:`clrc.prediction.xgboost.train_predict_xgb` (or an equivalent
    path that passes a raw ndarray to XGBoost without setting
    ``feature_names``). This guarantees ``booster.get_score()`` returns
    keys of the form ``"f<int>"`` where ``<int>`` is the column index in
    the training X matrix. Both the feature count and the naming convention
    are validated at call time — any violation raises immediately.

    Features that were never used for a split are absent from
    ``get_score`` and get 0 in the output vector (this is the legitimate
    default, not a contract violation).

    Raises
    ------
    ValueError
        * If ``booster.num_features()`` disagrees with ``n_features``
          (indicates a training/aggregation misalignment — possibly a stale
          fold pickle from a different feature set).
        * If any key returned by ``get_score`` does not match the default
          ``f<int>`` naming convention (indicates a future regression in the
          upstream training code — caller must retrain without explicit
          ``feature_names`` or extend ``gain_vector`` to accept a name map).
    """
    booster = _booster_from_bytes(model_raw)

    actual_n = booster.num_features()
    if actual_n != n_features:
        raise ValueError(
            f"gain_vector: booster has {actual_n} features, caller expects "
            f"{n_features}. Likely a mismatch between the alignment pickle "
            f"used for training and the one used for aggregation — verify "
            f"both point at the same data/expanded_ABC/aligned_*.pkl."
        )

    score = booster.get_score(importance_type="gain")
    vec = np.zeros(n_features, dtype=float)
    for key, val in score.items():
        m = _DEFAULT_FEATURE_KEY_RE.fullmatch(key)
        if m is None:
            raise ValueError(
                f"gain_vector: booster key {key!r} does not match the "
                f"default f<int> naming convention. Upstream training code "
                f"has started passing feature_names to DMatrix — either "
                f"revert that change, or extend gain_vector to accept a "
                f"name-to-index mapping."
            )
        idx = int(m.group(1))
        if not (0 <= idx < n_features):
            raise ValueError(
                f"gain_vector: booster key {key!r} maps to index {idx}, "
                f"out of range for n_features={n_features}."
            )
        vec[idx] = float(val)
    return vec


def compute_fold_importances(
    fold_artifacts: Mapping[str, FoldArtifact],
    *,
    n_features: int,
) -> Dict[str, np.ndarray]:
    """Compute per-fold gain vectors keyed by ``holdout_region``.

    Folds missing ``model_raw`` are skipped with a warning.
    """
    out: Dict[str, np.ndarray] = {}
    for holdout, fold in fold_artifacts.items():
        model_raw = getattr(fold, "model_raw", None)
        if not model_raw:
            logger.warning("Fold %s missing model_raw; skipping importance.", holdout)
            continue
        out[holdout] = gain_vector(model_raw, n_features)
    return out


# ---------------------------------------------------------------------------
#  Weighted aggregation across folds
# ---------------------------------------------------------------------------

def aggregate_importances(
    fold_importances: Mapping[str, np.ndarray],
    fold_weights: Mapping[str, int],
    *,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted mean of per-fold gain vectors.

    Computes two variants:

    * ``imp_abs``: raw weighted mean gain.
    * ``imp_rel``: each fold's gain vector is L1-normalized before averaging,
      so each fold contributes equally in terms of total relative attention.
      Folds with zero total gain contribute nothing.

    Uses stacked-matrix matmul for a vectorized O(n_folds × n_features)
    implementation. No Python-level fold loop for the arithmetic.
    """
    holdouts = list(fold_importances.keys())
    weights = np.asarray(
        [float(fold_weights.get(h, 0)) for h in holdouts], dtype=float
    )
    total_weight = float(weights.sum())
    if total_weight <= 0:
        logger.warning("Total fold weight is zero; returning zero importance vectors.")
        return np.zeros(n_features), np.zeros(n_features)

    # (n_folds, n_features) matrix, each row = fold's gain vector
    if not holdouts:
        return np.zeros(n_features), np.zeros(n_features)
    G = np.vstack([np.asarray(fold_importances[h], dtype=float) for h in holdouts])

    # Absolute: weighted mean of rows
    imp_abs = (weights @ G) / total_weight

    # Relative: L1-normalize each row, then weighted mean. Rows with zero total
    # gain (degenerate folds) get rescaled to zeros, not NaN.
    row_sums = G.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        G_rel = np.where(row_sums[:, None] > 0, G / row_sums[:, None], 0.0)
    imp_rel = (weights @ G_rel) / total_weight

    return imp_abs, imp_rel


# ---------------------------------------------------------------------------
#  Feature-level DataFrame
# ---------------------------------------------------------------------------

def build_feature_df(
    imp_abs: np.ndarray,
    imp_rel: np.ndarray,
    feature_names: Sequence[str],
    meta: Sequence[FeatureMeta],
) -> pd.DataFrame:
    """Build a per-feature importance DataFrame annotated with LR/CT metadata.

    ``meta`` is required and must conform to :class:`FeatureMeta` — every
    entry has all required keys. No ``.get()`` fallbacks: if the meta
    contract is violated, we raise rather than silently producing a row of
    None values. The contract is established at load time in
    :func:`clrc.core.io.load_alignment_data` and by
    :func:`clrc.prediction.lobo.select_features` (which aligns meta to the
    returned ``feature_names``).
    """
    n = int(imp_abs.shape[0])
    if imp_rel.shape[0] != n:
        raise ValueError(
            f"imp_abs/imp_rel length mismatch: {imp_abs.shape[0]} vs {imp_rel.shape[0]}"
        )
    if len(feature_names) != n:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != n_features ({n})"
        )
    if len(meta) != n:
        raise ValueError(
            f"meta length ({len(meta)}) != n_features ({n}). "
            f"Use clrc.prediction.lobo.select_features which returns an "
            f"aligned (X, feature_names, meta) triple."
        )

    # Direct subscript access — meta conforms to FeatureMeta schema, so every
    # key is guaranteed present (validated at load time in load_alignment_data).
    return pd.DataFrame(
        {
            "feature_index": np.arange(n),
            "feature_name": list(feature_names),
            "weighted_mean_gain_abs": imp_abs.astype(float),
            "weighted_mean_gain_rel": imp_rel.astype(float),
            "ct_L": [m["ct_L"] for m in meta],
            "ct_R": [m["ct_R"] for m in meta],
            "ligand_genes": [m.get("ligand_genes") for m in meta],
            "receptor_genes": [m.get("receptor_genes") for m in meta],
            "lr_name": [m["lr_name"] for m in meta],
            "lr_index": [m["lr_index"] for m in meta],
        }
    )


# ---------------------------------------------------------------------------
#  Group-level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_group(
    feat_df: pd.DataFrame,
    group_col: str,
    *,
    sort_by: str = "aggregated_importance_abs",
) -> pd.DataFrame:
    """Sum weighted_mean_gain per group, returning a sorted DataFrame."""
    if group_col not in feat_df.columns:
        raise KeyError(f"Column {group_col!r} not found in feature DataFrame")
    grouped = (
        feat_df.groupby(group_col, dropna=True)[
            ["weighted_mean_gain_abs", "weighted_mean_gain_rel"]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                group_col: "group_name",
                "weighted_mean_gain_abs": "aggregated_importance_abs",
                "weighted_mean_gain_rel": "aggregated_importance_rel",
            }
        )
    )
    return grouped.sort_values(sort_by, ascending=False).reset_index(drop=True)


def add_cell_class_column(group_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate a cell-type-grouped DataFrame with its broader cell class."""
    return group_df.assign(
        cell_class=group_df["group_name"].map(CELL_CLASS_MAP).fillna("Other")
    )


# ---------------------------------------------------------------------------
#  LR × fold matrix (for clustermap)
# ---------------------------------------------------------------------------

def build_lr_fold_matrix(
    fold_importances: Mapping[str, np.ndarray],
    meta: Sequence[FeatureMeta],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Sum per-feature gain into per-LR scores for every fold.

    Parameters
    ----------
    fold_importances
        Mapping of holdout_region → gain vector.
    meta
        :class:`FeatureMeta` list aligned row-for-row with the gain vectors.
        Features with ``lr_name=None`` (e.g. fiber_distance) are excluded
        from the LR matrix — they have no LR to aggregate against.
    normalize
        If True, L1-normalize each fold's gain vector before aggregation so
        folds are comparable regardless of absolute gain magnitude.

    Returns
    -------
    DataFrame indexed by lr_name, columns = sorted holdout_region names.
    """
    holdouts = sorted(fold_importances.keys())
    if not holdouts:
        return pd.DataFrame()

    first_vec = np.asarray(fold_importances[holdouts[0]], dtype=float)
    n_features = first_vec.shape[0]
    if len(meta) != n_features:
        raise ValueError(
            f"meta length ({len(meta)}) != gain vector length ({n_features})"
        )

    # Direct subscript — meta conforms to FeatureMeta, every entry has lr_name.
    # Features with lr_name=None (e.g. fiber_distance synthetic entry) are
    # excluded via the groupby dropna=True below.
    lr_names = [m["lr_name"] for m in meta]

    # Build (n_features, n_folds) gain matrix in one column_stack — no
    # per-fold Python loop overhead beyond the unavoidable normalization.
    gain_cols = []
    for holdout in holdouts:
        vec = np.asarray(fold_importances[holdout], dtype=float)
        if normalize:
            denom = float(vec.sum())
            if denom > 0:
                vec = vec / denom
            else:
                logger.warning("Fold %s has zero total gain; using zeros.", holdout)
                vec = np.zeros_like(vec)
        gain_cols.append(vec)
    gain_mat = np.column_stack(gain_cols)

    gain_df = pd.DataFrame(gain_mat, columns=holdouts)
    gain_df["lr_name"] = lr_names
    return gain_df.groupby("lr_name", sort=True, dropna=True).sum()


# ---------------------------------------------------------------------------
#  End-to-end convenience
# ---------------------------------------------------------------------------

def aggregate_full_importance_pipeline(
    fold_artifacts: Mapping[str, FoldArtifact],
    fold_weights: Mapping[str, int],
    feature_names: Sequence[str],
    meta: Sequence[FeatureMeta],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete compute → aggregate → annotate pipeline.

    Returns ``(feat_df, lr_fold_matrix)``. ``meta`` is required — use
    :func:`clrc.prediction.lobo.select_features` which returns the aligned
    ``(X, feature_names, meta)`` triple.
    """
    n_features = len(feature_names)
    fold_importances = compute_fold_importances(fold_artifacts, n_features=n_features)
    imp_abs, imp_rel = aggregate_importances(
        fold_importances, fold_weights, n_features=n_features
    )
    feat_df = build_feature_df(imp_abs, imp_rel, feature_names, meta)
    lr_fold_matrix = build_lr_fold_matrix(fold_importances, meta, normalize=True)
    return feat_df, lr_fold_matrix
