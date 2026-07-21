"""Fold-level percentile CIs for XGBoost feature importance.

For each LOBO fold, record each feature's per-fold relative importance;
compute the 2.5th and 97.5th percentiles across folds; report these as
95% fold-level CIs.

This is **not** a classical bootstrap. No resampling is performed. CIs are
direct empirical percentiles of the already-available per-fold distribution
per feature -- the LOBO folds are themselves the independent units we need
uncertainty over.

Functions
---------
compute_feature_level_ci
    Per-feature percentile CIs — one row per input feature.
compute_group_level_ci
    Group-aggregated percentile CIs. CRITICAL: features are summed within
    each group PER FOLD first, producing a per-fold group-sum distribution,
    and percentiles are taken across folds on that distribution. This is
    not the same as taking percentiles of each feature and summing them —
    anti-correlated features within a group would produce misleadingly
    wide CIs under the latter.

Output schema
-------------
Feature-level columns:
    feature_name, lr_name, ct_L, ct_R, mean_imp_rel, median_imp_rel,
    ci_lo_2p5, ci_hi_97p5, ci_width
Group-level columns:
    group_name, aggregated_importance_rel, mean_imp_rel, median_imp_rel,
    ci_lo, ci_hi, ci_width

Notes
-----
Per-feature percentiles happen to match the marginal column-wise percentiles
of the input matrix (same operation on the same axis), but the group-level
path explicitly constructs per-fold group sums first — the two paths are
mathematically distinct for groups with more than one feature.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from clrc.core.types import FeatureMeta

logger = logging.getLogger(__name__)

_VALID_GROUP_KEYS = ("lr_name", "ct_L", "ct_R")


def _validate_inputs(
    fold_importance_matrix: np.ndarray,
    feature_names: Sequence[str],
    meta: Sequence[FeatureMeta],
) -> np.ndarray:
    """Validate shapes and return the matrix as a 2-D float ndarray."""
    arr = np.asarray(fold_importance_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"fold_importance_matrix must be 2-D (n_folds, n_features); "
            f"got shape {arr.shape}."
        )
    n_folds, n_features = arr.shape
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"fold_importance_matrix n_features ({n_features})."
        )
    if len(meta) != n_features:
        raise ValueError(
            f"meta length ({len(meta)}) does not match "
            f"fold_importance_matrix n_features ({n_features})."
        )
    if n_folds < 2:
        logger.warning(
            "compute_*_ci called with n_folds=%d; percentiles are degenerate.",
            n_folds,
        )
    return arr


def _alpha_to_q(alpha: float) -> tuple[float, float]:
    """Convert alpha to (lo_q, hi_q) percentile values (e.g. 0.05 -> (2.5, 97.5))."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return lo_q, hi_q


# ---------------------------------------------------------------------------
#  Per-feature CI
# ---------------------------------------------------------------------------


def compute_feature_level_ci(
    fold_importance_matrix: np.ndarray,
    feature_names: Sequence[str],
    meta: Sequence[FeatureMeta],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-feature fold-level percentile CIs.

    Parameters
    ----------
    fold_importance_matrix
        2-D ndarray of shape ``(n_folds, n_features)``. Each row is one
        fold's per-feature importance vector. Typically these are the
        L1-normalized (relative) per-fold gain vectors produced by
        :func:`clrc.prediction.importance.compute_fold_importances` and
        subsequently normalized row-wise by the caller — but this function
        does not enforce normalization; it percentiles columns directly.
    feature_names
        Length ``n_features`` identifiers for the matrix columns.
    meta
        Length ``n_features`` :class:`FeatureMeta` list, row-for-row aligned
        with ``feature_names``.
    alpha
        Two-sided significance level. ``alpha=0.05`` (default) gives the
        2.5/97.5 percentiles (95% CI).

    Returns
    -------
    pd.DataFrame
        One row per feature, sorted by ``mean_imp_rel`` descending.
        Columns: ``feature_name, lr_name, ct_L, ct_R, mean_imp_rel,
        median_imp_rel, ci_lo_2p5, ci_hi_97p5, ci_width``.

    Notes
    -----
    The column names ``ci_lo_2p5`` and ``ci_hi_97p5`` reflect the default
    alpha; when alpha is non-default, the same columns still hold the
    (alpha/2, 1-alpha/2) percentiles.
    """
    arr = _validate_inputs(fold_importance_matrix, feature_names, meta)
    lo_q, hi_q = _alpha_to_q(alpha)

    # Column-wise statistics across folds. axis=0 reduces over folds.
    mean_imp = arr.mean(axis=0)
    median_imp = np.median(arr, axis=0)
    ci_lo = np.percentile(arr, lo_q, axis=0)
    ci_hi = np.percentile(arr, hi_q, axis=0)

    df = pd.DataFrame(
        {
            "feature_name": list(feature_names),
            "lr_name": [m["lr_name"] for m in meta],
            "ct_L": [m["ct_L"] for m in meta],
            "ct_R": [m["ct_R"] for m in meta],
            "mean_imp_rel": mean_imp,
            "median_imp_rel": median_imp,
            "ci_lo_2p5": ci_lo,
            "ci_hi_97p5": ci_hi,
            "ci_width": ci_hi - ci_lo,
        }
    )
    return df.sort_values("mean_imp_rel", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Group-level CI (sum-then-percentile)
# ---------------------------------------------------------------------------


def compute_group_level_ci(
    fold_importance_matrix: np.ndarray,
    feature_names: Sequence[str],
    meta: Sequence[FeatureMeta],
    *,
    group_key: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Group-level fold-level percentile CIs.

    For each group (defined by ``meta[i][group_key]``), sum the per-feature
    importance values PER FOLD, producing an ``(n_folds,)`` fold-distribution
    of group-total importance. Then take percentiles across folds.

    Rationale: summing feature
    importances at each fold reflects how much predictive attention the
    model placed on the group *in that fold*. Taking percentiles of that
    per-fold distribution then gives a CI on the group-aggregated
    importance. Taking per-feature percentiles first and summing them
    would overstate uncertainty for groups of anti-correlated features.

    Parameters
    ----------
    fold_importance_matrix
        Shape ``(n_folds, n_features)``; see
        :func:`compute_feature_level_ci`.
    feature_names, meta
        Row-aligned with ``fold_importance_matrix``.
    group_key
        One of ``"lr_name"``, ``"ct_L"``, ``"ct_R"`` — the
        :class:`FeatureMeta` field to group by. Features with ``None``
        under this key (e.g. the synthetic ``fiber_distance`` feature with
        ``lr_name=None``) are dropped, not aggregated into a ``None`` group.
    alpha
        Two-sided significance level, default 0.05 (95% CI).

    Returns
    -------
    pd.DataFrame
        Columns: ``group_name, aggregated_importance_rel, mean_imp_rel,
        median_imp_rel, ci_lo, ci_hi, ci_width``. ``aggregated_importance_rel``
        equals ``mean_imp_rel`` (kept under both names for schema
        compatibility with downstream consumers). Sorted by
        ``aggregated_importance_rel`` descending.

    Raises
    ------
    ValueError
        If ``group_key`` is not one of the valid FeatureMeta keys, or if
        shapes/lengths mismatch.
    """
    if group_key not in _VALID_GROUP_KEYS:
        raise ValueError(
            f"group_key must be one of {_VALID_GROUP_KEYS}; got {group_key!r}."
        )
    arr = _validate_inputs(fold_importance_matrix, feature_names, meta)
    lo_q, hi_q = _alpha_to_q(alpha)
    n_folds = arr.shape[0]

    # Map each feature column to its group label (may be None).
    group_labels = np.array([m[group_key] for m in meta], dtype=object)

    # Collect non-None unique groups, preserving a deterministic sort order
    # for reproducibility.
    valid_mask = np.array([lbl is not None for lbl in group_labels], dtype=bool)
    unique_groups = sorted({lbl for lbl in group_labels[valid_mask]})

    rows = []
    for group in unique_groups:
        col_mask = group_labels == group
        # Per-fold group sum: sum features belonging to this group within each fold.
        # Shape: (n_folds,)
        group_fold_sums = arr[:, col_mask].sum(axis=1)
        assert group_fold_sums.shape == (n_folds,)

        rows.append(
            {
                "group_name": group,
                "aggregated_importance_rel": float(group_fold_sums.mean()),
                "mean_imp_rel": float(group_fold_sums.mean()),
                "median_imp_rel": float(np.median(group_fold_sums)),
                "ci_lo": float(np.percentile(group_fold_sums, lo_q)),
                "ci_hi": float(np.percentile(group_fold_sums, hi_q)),
                "ci_width": float(
                    np.percentile(group_fold_sums, hi_q)
                    - np.percentile(group_fold_sums, lo_q)
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        # Preserve schema even if no features had a non-None group_key.
        df = pd.DataFrame(
            columns=[
                "group_name",
                "aggregated_importance_rel",
                "mean_imp_rel",
                "median_imp_rel",
                "ci_lo",
                "ci_hi",
                "ci_width",
            ]
        )
    return df.sort_values(
        "aggregated_importance_rel", ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Cross-modality SC vs FC CIs (true bootstrap with resampling)
# ---------------------------------------------------------------------------


def compute_diff_pct_celltype_ci(
    sc_fold_matrix: np.ndarray,
    fc_fold_matrix: np.ndarray,
    feature_names: Sequence[str],
    meta: Sequence[FeatureMeta],
    *,
    group_key: str = "ct_L",
    n_boot: int = 5000,
    alpha: float = 0.05,
    eps: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-cell-type bootstrap CIs on SC%, FC%, log2(SC/FC), and (SC%-FC%).

    SC and FC LOBO folds hold out different brain regions, so the bootstrap
    resamples the two fold sets INDEPENDENTLY then differences (or log-ratios)
    each pair of bootstrap samples to build a distribution on the contrast.

    Parameters
    ----------
    sc_fold_matrix : (n_sc_folds, n_features)
        L1-normalized SC fold importance matrix. Rows are folds; columns are
        features (same order as ``feature_names`` / ``meta``).
    fc_fold_matrix : (n_fc_folds, n_features)
        L1-normalized FC fold importance matrix.
    feature_names, meta
        Row-aligned with both matrices' columns.
    group_key
        ``"ct_L"`` for sender role, ``"ct_R"`` for receiver role,
        ``"lr_name"`` for LR-pair role.
    n_boot
        Number of bootstrap iterations. Default 5000.
    alpha
        Two-sided significance level. Default 0.05 (95% CI).
    eps
        Pseudocount added to both SC% and FC% before log2 to avoid div-by-zero.
    rng
        Optional numpy Generator. Default seeds with 42 for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per cell type (or LR), columns:
        ``group_name``, ``sc_pct_mean``, ``sc_ci_lo``, ``sc_ci_hi``,
        ``fc_pct_mean``, ``fc_ci_lo``, ``fc_ci_hi``,
        ``diff_pct_mean``, ``diff_ci_lo``, ``diff_ci_hi``,
        ``diff_p_two_sided``, ``diff_fdr``,
        ``log2_ratio_mean``, ``log2_ci_lo``, ``log2_ci_hi``.
        Sorted by ``diff_pct_mean`` ascending.
    """
    if group_key not in _VALID_GROUP_KEYS:
        raise ValueError(
            f"group_key must be one of {_VALID_GROUP_KEYS}; got {group_key!r}."
        )
    sc_arr = _validate_inputs(sc_fold_matrix, feature_names, meta)
    fc_arr = _validate_inputs(fc_fold_matrix, feature_names, meta)
    n_sc = sc_arr.shape[0]
    n_fc = fc_arr.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)
    lo_q, hi_q = _alpha_to_q(alpha)

    group_labels = np.array([m[group_key] for m in meta], dtype=object)
    valid_mask = np.array([lbl is not None for lbl in group_labels], dtype=bool)
    unique_groups = sorted({lbl for lbl in group_labels[valid_mask]})
    if not unique_groups:
        return pd.DataFrame(
            columns=[
                "group_name",
                "sc_pct_mean", "sc_ci_lo", "sc_ci_hi",
                "fc_pct_mean", "fc_ci_lo", "fc_ci_hi",
                "diff_pct_mean", "diff_ci_lo", "diff_ci_hi",
                "log2_ratio_mean", "log2_ci_lo", "log2_ci_hi",
            ]
        )

    n_groups = len(unique_groups)
    sc_per_fold = np.zeros((n_sc, n_groups))
    fc_per_fold = np.zeros((n_fc, n_groups))
    for j, group in enumerate(unique_groups):
        col_mask = group_labels == group
        sc_per_fold[:, j] = sc_arr[:, col_mask].sum(axis=1) * 100.0
        fc_per_fold[:, j] = fc_arr[:, col_mask].sum(axis=1) * 100.0

    sc_boot = np.zeros((n_boot, n_groups))
    fc_boot = np.zeros((n_boot, n_groups))
    for b in range(n_boot):
        sc_boot[b] = sc_per_fold[rng.integers(0, n_sc, size=n_sc)].mean(axis=0)
        fc_boot[b] = fc_per_fold[rng.integers(0, n_fc, size=n_fc)].mean(axis=0)
    diff_boot = sc_boot - fc_boot
    log2_boot = np.log2((sc_boot + eps) / (fc_boot + eps))

    p_two_sided = 2.0 * np.minimum(
        (diff_boot < 0).mean(axis=0),
        (diff_boot > 0).mean(axis=0),
    )
    p_two_sided = np.clip(p_two_sided, 1.0 / n_boot, 1.0)
    fdr_diff = _bh_correct(p_two_sided)

    rows = []
    for j, group in enumerate(unique_groups):
        sc_mean = float(sc_per_fold[:, j].mean())
        fc_mean = float(fc_per_fold[:, j].mean())
        rows.append({
            "group_name": group,
            "sc_pct_mean": sc_mean,
            "sc_ci_lo": float(np.percentile(sc_boot[:, j], lo_q)),
            "sc_ci_hi": float(np.percentile(sc_boot[:, j], hi_q)),
            "fc_pct_mean": fc_mean,
            "fc_ci_lo": float(np.percentile(fc_boot[:, j], lo_q)),
            "fc_ci_hi": float(np.percentile(fc_boot[:, j], hi_q)),
            "diff_pct_mean": sc_mean - fc_mean,
            "diff_ci_lo": float(np.percentile(diff_boot[:, j], lo_q)),
            "diff_ci_hi": float(np.percentile(diff_boot[:, j], hi_q)),
            "diff_p_two_sided": float(p_two_sided[j]),
            "diff_fdr": float(fdr_diff[j]),
            "log2_ratio_mean": float(np.log2((sc_mean + eps) / (fc_mean + eps))),
            "log2_ci_lo": float(np.percentile(log2_boot[:, j], lo_q)),
            "log2_ci_hi": float(np.percentile(log2_boot[:, j], hi_q)),
        })
    return pd.DataFrame(rows).sort_values("diff_pct_mean", ascending=True).reset_index(drop=True)


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out
