"""Partial Spearman correlation: basic (per-feature) and batch (Numba-accelerated).

Also hosts region-expression covariate helpers for the caller-side
(region-mean-expression as a control variable in the AD partial Spearman
analyses).

Convention: proper Spearman partial -- x, y, AND covariates are all
rank-transformed before the linear projection. This differs from a common
shortcut that ranks only x and y while projecting out raw covariates;
that shortcut over-estimates residual correlation for monotonic
non-linear covariates because part of the covariate's monotonic
structure survives a linear fit and re-enters as spurious signal.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numba
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Single-feature partial Spearman
# ---------------------------------------------------------------------------

def partial_spearman(
    x: np.ndarray, y: np.ndarray, covariates: np.ndarray
) -> Tuple[float, float]:
    """Partial Spearman correlation between x and y, controlling for covariates.

    Returns (correlation, p-value).
    """
    from sklearn.linear_model import LinearRegression

    mask = ~(np.isnan(x) | np.isnan(y) | np.any(np.isnan(covariates), axis=1))
    if mask.sum() < 10:
        return np.nan, np.nan

    x_clean = x[mask]
    y_clean = y[mask]
    cov_clean = covariates[mask]

    x_ranked = stats.rankdata(x_clean)
    y_ranked = stats.rankdata(y_clean)
    cov_ranked = np.column_stack([
        stats.rankdata(cov_clean[:, j]) for j in range(cov_clean.shape[1])
    ])

    reg_x = LinearRegression().fit(cov_ranked, x_ranked)
    reg_y = LinearRegression().fit(cov_ranked, y_ranked)

    x_resid = x_ranked - reg_x.predict(cov_ranked)
    y_resid = y_ranked - reg_y.predict(cov_ranked)

    corr, pval = stats.pearsonr(x_resid, y_resid)
    return corr, pval


# ---------------------------------------------------------------------------
#  Numba-accelerated rank function
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _rankdata_columns(X):
    """Rank each column of X with tie averaging (matches scipy.stats.rankdata).

    Uses numba parallel prange for ~100x speedup over Python loop + scipy.
    """
    n, m = X.shape
    result = np.empty((n, m), dtype=np.float64)
    for col_idx in numba.prange(m):
        col = X[:, col_idx]
        order = np.argsort(col)
        i = 0
        while i < n:
            end = i + 1
            while end < n and col[order[end]] == col[order[i]]:
                end += 1
            avg_rank = (i + 1 + end) / 2.0
            for k in range(i, end):
                result[order[k], col_idx] = avg_rank
            i = end
    return result


# ---------------------------------------------------------------------------
#  Vectorized batch correlations
# ---------------------------------------------------------------------------

def _batch_correlations(
    X_batch: np.ndarray, y_batch: np.ndarray, cov_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized partial Spearman for a batch of features sharing the same
    valid-subject mask.

    1. Rank each column and y (via numba JIT with tie averaging)
    2. Project out covariates via P = I - C(C^TC)^{-1}C^T
    3. Pearson r on residuals
    4. p-value via t-distribution with df = n - 2
    """
    n_valid, n_features = X_batch.shape

    X_ranked = _rankdata_columns(
        np.ascontiguousarray(X_batch, dtype=np.float64)
    )
    y_ranked = stats.rankdata(y_batch)
    # Rank-transform the covariates too (proper Spearman partial
    # semantics). See module docstring.
    cov_ranked = _rankdata_columns(
        np.ascontiguousarray(cov_batch, dtype=np.float64)
    )

    # Projection matrix: P = I - C(C^TC)^{-1}C^T
    C = np.column_stack([np.ones(n_valid), cov_ranked])
    try:
        CTC_inv = np.linalg.inv(C.T @ C)
    except np.linalg.LinAlgError:
        CTC_inv = np.linalg.pinv(C.T @ C)
    P = np.eye(n_valid) - C @ CTC_inv @ C.T

    X_resid = P @ X_ranked
    y_resid = P @ y_ranked

    y_norm = np.sqrt(y_resid @ y_resid)
    X_norms = np.sqrt(np.sum(X_resid**2, axis=0))
    dots = X_resid.T @ y_resid

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(X_norms * y_norm > 0, dots / (X_norms * y_norm), np.nan)

    df = n_valid - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = r * np.sqrt(df) / np.sqrt(np.maximum(1 - r**2, 1e-300))
    p_vals = 2 * stats.t.sf(np.abs(t_stat), df)

    return r, p_vals


# ---------------------------------------------------------------------------
#  Batch partial Spearman (groups features by NaN mask pattern)
# ---------------------------------------------------------------------------

def partial_spearman_batch(
    X: np.ndarray, y: np.ndarray, covariates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized partial Spearman correlations for all columns of X against y.

    Groups features by their NaN mask pattern and processes each group with
    a single matrix projection.

    Parameters
    ----------
    X : (n_subjects, n_features)
    y : (n_subjects,)
    covariates : (n_subjects, n_covariates)

    Returns
    -------
    (correlations, pvalues) each of shape (n_features,)
    """
    n_subjects, n_features = X.shape
    correlations = np.full(n_features, np.nan)
    pvalues = np.full(n_features, np.nan)

    global_mask = ~(np.isnan(y) | np.any(np.isnan(covariates), axis=1))

    X_global = X[global_mask]
    feature_nan = np.isnan(X_global)
    any_nan = np.any(feature_nan, axis=0)

    # Batch 1: features with NO NaN (fully vectorized)
    no_nan_idx = np.where(~any_nan)[0]
    if len(no_nan_idx) > 0:
        n_valid = global_mask.sum()
        if n_valid >= 10:
            r, p = _batch_correlations(
                X_global[:, no_nan_idx], y[global_mask], covariates[global_mask]
            )
            correlations[no_nan_idx] = r
            pvalues[no_nan_idx] = p

    # Batch 2: features WITH NaN (group by mask pattern)
    nan_idx = np.where(any_nan)[0]
    if len(nan_idx) > 0:
        nan_cols = feature_nan[:, nan_idx].T
        mask_bytes = np.packbits(nan_cols, axis=1)
        unique_masks, inverse = np.unique(mask_bytes, axis=0, return_inverse=True)

        for group_id in range(len(unique_masks)):
            group_local = np.where(inverse == group_id)[0]
            group_features = nan_idx[group_local]

            feat_valid = ~feature_nan[:, group_features[0]]
            full_mask = global_mask.copy()
            full_mask[global_mask] &= feat_valid

            n_valid = full_mask.sum()
            if n_valid < 10:
                continue

            r, p = _batch_correlations(
                X[full_mask][:, group_features],
                y[full_mask],
                covariates[full_mask],
            )
            correlations[group_features] = r
            pvalues[group_features] = p

    return correlations, pvalues


# ---------------------------------------------------------------------------
#  Region-expression covariates
# ---------------------------------------------------------------------------
#
# Two covariate families:
#
# * **LR-pair-specific covariate**: mean(log1p expression) of that LR
#   pair's ligand AND receptor genes, per region. Gene-subset logic matches
#   ``clrc.features.coexpression.build_lr_expression_product`` — genes not
#   present in the expression matrix are silently dropped, and if neither a
#   ligand nor a receptor gene is present the column is all-NaN (same
#   contract as the LR-expression-product baseline).
#
# * **Global gradient covariate**: first principal component of the
#   per-region × per-gene expression matrix. One scalar per region,
#   capturing large-scale expression gradients.
#
# Both are **region-indexed** artifacts by design. Upstream pipeline
# broadcasts them to per-subject covariate columns at the analysis site
# (region-specific / region-collapsed). See
# ``src/pipeline/pathology_correlation/region_expression_covariate.py``.
# ---------------------------------------------------------------------------


def _parse_gene_list(v) -> List[str]:
    """Parse a ligand_genes / receptor_genes entry.

    Mirrors the parsing convention in
    :func:`clrc.features.coexpression.build_lr_expression_product` — accepts
    either a list/tuple/ndarray of strings or a ``"+"``-separated string
    (the convention used in the importance-CSV metadata). Empty / NaN
    inputs yield ``[]``.
    """
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return [str(x) for x in v]
    if v is None:
        return []
    try:
        if pd.isna(v):
            return []
    except (TypeError, ValueError):
        pass
    if v == "":
        return []
    return [g.strip() for g in str(v).split("+") if g.strip()]


def compute_lr_pair_expression_covariate(
    expression_regions_x_genes: pd.DataFrame,
    lr_gene_lookup: Dict[str, Tuple[Sequence[str], Sequence[str]]],
) -> pd.DataFrame:
    """Build the per-LR-pair, per-region expression covariate matrix.

    For each LR pair, the covariate value in region ``r`` is the arithmetic
    mean of ``expression_regions_x_genes.loc[r, G]`` where ``G`` is the
    intersection of the pair's ligand + receptor genes and
    ``expression_regions_x_genes.columns``. The gene-subset logic is the
    one used by :func:`clrc.features.coexpression.build_lr_expression_product`
    — genes absent from the expression panel are silently
    dropped; if **both** ligand and receptor gene lists produce zero
    overlap with ``expression_regions_x_genes.columns``, the column is
    entirely NaN (caller drops such columns downstream if needed).

    Parameters
    ----------
    expression_regions_x_genes
        Per-region × per-gene DataFrame. Rows indexed by region label,
        columns by gene symbol. **Caller** is responsible for the log1p
        transform (log1p of raw ABC counts).
    lr_gene_lookup
        Mapping ``lr_name -> (ligand_genes, receptor_genes)`` where each
        gene sequence is either a list[str] or a ``"+"``-separated string.

    Returns
    -------
    covariate : pd.DataFrame
        Shape ``(n_regions, n_lr_pairs)``. Index = regions (identical to
        ``expression_regions_x_genes.index``), columns = LR pair names (in
        ``lr_gene_lookup`` iteration order).
    """
    regions = expression_regions_x_genes.index
    cols = expression_regions_x_genes.columns
    col_set = set(cols)

    data = np.full((len(regions), len(lr_gene_lookup)), np.nan, dtype=np.float64)
    lr_names = list(lr_gene_lookup.keys())
    n_all_nan = 0
    for j, lr_name in enumerate(lr_names):
        lig_raw, rec_raw = lr_gene_lookup[lr_name]
        genes = _parse_gene_list(lig_raw) + _parse_gene_list(rec_raw)
        # Union (ligand + receptor), intersected with the panel
        present = [g for g in genes if g in col_set]
        # Deduplicate while preserving order (if a gene appears as both
        # ligand and receptor, counting it twice would double-weight it).
        seen = set()
        unique_present = []
        for g in present:
            if g not in seen:
                seen.add(g)
                unique_present.append(g)
        if not unique_present:
            n_all_nan += 1
            continue
        data[:, j] = (
            expression_regions_x_genes.loc[:, unique_present]
            .mean(axis=1)
            .to_numpy(dtype=np.float64)
        )

    logger.info(
        "LR-pair expression covariate: shape=(%d, %d), all-NaN columns=%d",
        data.shape[0], data.shape[1], n_all_nan,
    )
    return pd.DataFrame(data, index=regions.copy(), columns=lr_names)


def compute_pc1_covariate(
    expression_regions_x_genes: pd.DataFrame,
    *,
    standardize: bool = True,
) -> pd.Series:
    """Build the global-gradient covariate: PC1 of regional gene expression.

    Fits ``sklearn.decomposition.PCA(n_components=1)`` on the
    ``(n_regions, n_genes)`` matrix and returns the first principal-component
    score for each region (i.e. the first column of ``PCA.fit_transform(X)``).

    Standardization
    ---------------
    With ``standardize=True`` (default), genes are z-scored column-wise
    *before* fitting PCA — so each gene contributes equally regardless of
    its expression magnitude. This matches the typical convention in
    gene-expression PCA (Yeung 2001, PNAS; Alter 2000 PNAS).

    Passing ``standardize=False`` runs PCA directly on log1p counts; the
    first PC will then be dominated by the highest-variance genes.

    Parameters
    ----------
    expression_regions_x_genes
        Per-region × per-gene DataFrame (log1p'ed upstream).
    standardize
        Whether to z-score each gene column before PCA. Default True.

    Returns
    -------
    pc1 : pd.Series
        Length ``n_regions``. Index = ``expression_regions_x_genes.index``.
        Name = ``"pc1"``. Sign is not canonicalised — downstream partial
        Spearman is invariant to a sign flip, so we don't force a
        convention here.
    """
    from sklearn.decomposition import PCA

    X = expression_regions_x_genes.to_numpy(dtype=np.float64)
    if standardize:
        col_mean = X.mean(axis=0, keepdims=True)
        col_std = X.std(axis=0, keepdims=True, ddof=0)
        # Drop zero-variance genes — they contribute nothing and blow up the
        # z-score. Keep the remaining columns.
        nonzero = (col_std.ravel() > 0)
        if not nonzero.any():
            raise ValueError(
                "All gene columns have zero variance; cannot compute PC1."
            )
        X = (X[:, nonzero] - col_mean[:, nonzero]) / col_std[:, nonzero]

    pca = PCA(n_components=1)
    scores = pca.fit_transform(X)[:, 0]  # (n_regions,)
    logger.info(
        "PC1 covariate: n_regions=%d, explained_variance_ratio=%.4f, "
        "standardize=%s",
        len(scores), float(pca.explained_variance_ratio_[0]), standardize,
    )
    return pd.Series(
        scores, index=expression_regions_x_genes.index.copy(), name="pc1"
    )


def partial_spearman_with_extra_covariates(
    X: np.ndarray,
    y: np.ndarray,
    base_covariates: np.ndarray,
    extra_covariates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partial Spearman with extra covariates stacked onto the existing ones.

    Thin wrapper around :func:`partial_spearman_batch`: horizontally
    concatenates ``extra_covariates`` onto ``base_covariates`` and passes
    the resulting matrix through. Preserves the existing contract
    (returns ``(correlations, pvalues)``), so the caller can treat this
    exactly like the unadjusted call.

    Parameters
    ----------
    X
        ``(n_subjects, n_features)`` feature matrix (same as
        :func:`partial_spearman_batch`).
    y
        ``(n_subjects,)`` clinical variable.
    base_covariates
        ``(n_subjects, k_base)`` — the existing covariate array (e.g.
        age_death, educ, msex in the AD pipeline).
    extra_covariates
        ``(n_subjects, k_extra)`` — the **additional** conditioning
        variables (e.g. per-LR-pair expression mean and/or PC1).

    Returns
    -------
    (correlations, pvalues) : tuple of np.ndarray
        Each of shape ``(n_features,)``. Identical contract to
        :func:`partial_spearman_batch`.
    """
    if base_covariates.ndim != 2:
        raise ValueError(
            f"base_covariates must be 2-D, got shape {base_covariates.shape}"
        )
    if extra_covariates.ndim != 2:
        raise ValueError(
            f"extra_covariates must be 2-D, got shape {extra_covariates.shape}"
        )
    if base_covariates.shape[0] != extra_covariates.shape[0]:
        raise ValueError(
            f"base_covariates n_subjects={base_covariates.shape[0]} "
            f"!= extra_covariates n_subjects={extra_covariates.shape[0]}"
        )
    stacked = np.column_stack([base_covariates, extra_covariates])
    return partial_spearman_batch(X, y, stacked)


def correlate_one_lr_group(
    lr_idx: int,
    cols: np.ndarray,
    X_cols: np.ndarray,
    y: np.ndarray,
    base_cov_long: np.ndarray,
    cov_6a_col: "np.ndarray | None",
    cov_6c: "np.ndarray | None",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Run partial Spearman for one LR-group slice of feature columns.

    Lives here (rather than inside the pipeline driver) so joblib's loky
    workers can import it by its stable module path; pipeline drivers
    loaded via ``importlib.util.spec_from_file_location`` do not produce
    module names that worker processes can resolve.

    Returns
    -------
    (cols, r, p)
        ``cols`` is returned unchanged so the parallel dispatcher can
        re-scatter each worker's output into the full-feature result
        arrays without a shared mutable target.
    """
    extra_cols: List[np.ndarray] = []
    if cov_6a_col is not None:
        extra_cols.append(cov_6a_col)
    if cov_6c is not None:
        extra_cols.append(cov_6c)
    extra = (
        np.column_stack(extra_cols) if extra_cols
        else np.empty((X_cols.shape[0], 0), dtype=np.float64)
    )
    r, p = partial_spearman_with_extra_covariates(X_cols, y, base_cov_long, extra)
    return cols, r, p
