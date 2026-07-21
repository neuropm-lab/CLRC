"""Variogram-matched null test on a scalar feature-target alignment statistic.

Tests whether the observed feature-target spatial alignment exceeds what is
expected under a variogram-preserving spatial null. The alignment statistic
is aggregated across all features so the null test reduces to a single
real value vs N_surrogates null values -- no per-fit model training.

Default alignment statistic: the **mean absolute distance-partialled
Spearman correlation** between each feature column and the target,
partialling out a scalar distance covariate (fiber distance per edge).
With distance as a covariate, the null tests whether feature-target
alignment persists after stringent control for spatial proximity, once
the obvious proximity confound is removed.

Why mean absolute (not mean signed): each CCI feature has an unsigned
"strength-of-spatial-alignment" interpretation; signed means of
correlations across thousands of features are dominated by cancellation
rather than by the alignment magnitude. Mean absolute matches how
feature-level alignment is aggregated in the connectomics null-test
literature (e.g., Markello 2021, Hansen 2022).

Fast path
---------
For repeated application against a fixed ``(y, covariates)`` pair (e.g.
one real X plus N surrogate X matrices in the variogram-null test), build
a :class:`PartialSpearmanContext` once and call
:func:`mean_abs_partial_spearman_cached` per surrogate. The context
caches the ranked-and-centered ``y``, the ranked covariate column, and
the zero-order ``r_yc``. Per-surrogate compute is then one rank of X,
two gemv-shaped dot products, and a scalar partial-correlation formula:

    r_{xy|c} = (r_xy - r_xc * r_yc) / sqrt((1 - r_xc^2)(1 - r_yc^2))

No O(n^2) projection matrix is materialized. Supports a single covariate
column only; falls back to :func:`clrc.ad.correlation.partial_spearman_batch`
when X contains NaN entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from clrc.ad.correlation import _rankdata_columns, partial_spearman_batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Cached context for fixed (y, covariate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartialSpearmanContext:
    """Precomputed ``(y, covariate)`` state for repeated partial-Spearman calls.

    Built once per target; reused across all surrogate X matrices. Supports
    at most one covariate column (the variogram-null use case). Rank, center,
    and norm of ``y`` and ``c`` — plus the scalar ``r_yc`` zero-order
    correlation — are all constant across surrogates, so caching them avoids
    redoing ~820 MB of matrix algebra per call.
    """

    n: int
    y_centered: np.ndarray  # shape (n,), ranks of y minus their mean
    y_std: float            # sqrt(sum(y_centered ** 2))
    c_centered: Optional[np.ndarray]  # None if no covariate
    c_std: float            # 0.0 when no covariate
    r_yc: float             # zero-order r(y, c); 0.0 when no covariate


def build_partial_spearman_context(
    y: np.ndarray, covariates: Optional[np.ndarray] = None
) -> PartialSpearmanContext:
    """Construct a :class:`PartialSpearmanContext` for reuse across X matrices.

    Parameters
    ----------
    y : (n,) array
        Target vector.
    covariates : (n,) or (n, 1) array, optional
        Single-column covariate to partial out (typically fiber distance).
        Passing a multi-column covariate matrix raises ``ValueError`` —
        use :func:`clrc.ad.correlation.partial_spearman_batch` for the
        multi-covariate case.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n < 3:
        raise ValueError(f"Need at least 3 samples for correlation; got n={n}.")

    y_ranked = stats.rankdata(y).astype(np.float64)
    y_centered = y_ranked - y_ranked.mean()
    y_std = float(np.sqrt(float(y_centered @ y_centered)))

    if covariates is None:
        return PartialSpearmanContext(
            n=n, y_centered=y_centered, y_std=y_std,
            c_centered=None, c_std=0.0, r_yc=0.0,
        )

    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    if cov.shape[0] != n:
        raise ValueError(
            f"covariates has {cov.shape[0]} rows but y has {n}."
        )
    if cov.shape[1] != 1:
        raise ValueError(
            f"PartialSpearmanContext supports 1 covariate column; got "
            f"{cov.shape[1]}. Use partial_spearman_batch for multi-covariate."
        )

    c_ranked = stats.rankdata(cov[:, 0]).astype(np.float64)
    c_centered = c_ranked - c_ranked.mean()
    c_std = float(np.sqrt(float(c_centered @ c_centered)))
    if y_std > 0.0 and c_std > 0.0:
        r_yc = float((y_centered @ c_centered) / (y_std * c_std))
    else:
        r_yc = 0.0
    # Numerical clip into [-1, 1] — sqrt(1 - r_yc^2) becomes complex at 1+eps.
    r_yc = max(-1.0, min(1.0, r_yc))

    return PartialSpearmanContext(
        n=n, y_centered=y_centered, y_std=y_std,
        c_centered=c_centered, c_std=c_std, r_yc=r_yc,
    )


# ---------------------------------------------------------------------------
#  Fast-path rank + apply
# ---------------------------------------------------------------------------


def _rank_center_norms(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rank each column of X and return (X_centered, X_std_per_col).

    Uses the numba-parallel rank kernel from ``clrc.ad.correlation``. The
    returned ``X_centered`` has each column mean-shifted to zero; ``X_std``
    is the per-column L2 norm of ``X_centered``.
    """
    X_ranked = _rankdata_columns(np.ascontiguousarray(X, dtype=np.float64))
    X_ranked -= X_ranked.mean(axis=0)
    X_std = np.sqrt((X_ranked * X_ranked).sum(axis=0))
    return X_ranked, X_std


def _partial_r_from_centered(
    X_centered: np.ndarray, X_std: np.ndarray, ctx: PartialSpearmanContext,
) -> np.ndarray:
    """Compute per-feature partial Spearman r given a centered ranked X."""
    dots_xy = X_centered.T @ ctx.y_centered  # (f,)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_xy = np.where(X_std > 0.0, dots_xy / (X_std * ctx.y_std), np.nan)

    if ctx.c_centered is None:
        return r_xy

    dots_xc = X_centered.T @ ctx.c_centered
    with np.errstate(divide="ignore", invalid="ignore"):
        r_xc = np.where(X_std > 0.0, dots_xc / (X_std * ctx.c_std), np.nan)
    # Clip into the valid correlation range to guard sqrt against tiny
    # negative radicands from fp rounding.
    r_xc = np.clip(r_xc, -1.0, 1.0)
    denom = np.sqrt(np.maximum(
        (1.0 - r_xc * r_xc) * (1.0 - ctx.r_yc * ctx.r_yc), 0.0,
    ))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_partial = np.where(
            denom > 0.0, (r_xy - r_xc * ctx.r_yc) / denom, np.nan,
        )
    return r_partial


def mean_abs_partial_spearman_cached(
    X: np.ndarray, ctx: PartialSpearmanContext,
) -> float:
    """Fast-path mean |partial-Spearman| using a precomputed context.

    Falls back to the general :func:`partial_spearman_batch` when ``X``
    contains NaN entries (rare for surrogate X; possible for real X).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}.")
    if X.shape[0] != ctx.n:
        raise ValueError(
            f"X has {X.shape[0]} rows but context was built with n={ctx.n}."
        )

    if np.isnan(X).any():
        # General NaN-aware path. Rebuild a two-column C on the fly by
        # reconstructing ranks from the context; cheaper than the dense P
        # projection but keeps NaN-group semantics exact.
        if ctx.c_centered is None:
            covariates = np.zeros((ctx.n, 1), dtype=np.float64)
        else:
            covariates = ctx.c_centered.reshape(-1, 1)
        # Reconstruct y from the centered ranks (add back the mean).
        y_ranked = ctx.y_centered + (ctx.n + 1.0) / 2.0
        r, _ = partial_spearman_batch(X, y_ranked, covariates)
        valid = np.isfinite(r)
        if not valid.any():
            return float("nan")
        return float(np.abs(r[valid]).mean())

    X_centered, X_std = _rank_center_norms(X)
    r = _partial_r_from_centered(X_centered, X_std, ctx)
    valid = np.isfinite(r)
    if not valid.any():
        return float("nan")
    return float(np.abs(r[valid]).mean())


# ---------------------------------------------------------------------------
#  Scalar alignment statistic over a single (X, y, covariates) triple
# ---------------------------------------------------------------------------


def mean_abs_partial_spearman(
    X: np.ndarray,
    y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> float:
    """Compute the mean |distance-partialled Spearman(X[:, f], y)| across features.

    When ``covariates`` is ``None``, falls back to plain Spearman.
    Internally dispatches to the cached fast path; results match the
    classical projection-based partial-Spearman up to floating-point
    rounding for the single-covariate case.

    Parameters
    ----------
    X : (n_samples, n_features) array
        Feature matrix. May contain NaN.
    y : (n_samples,) array
        Target vector.
    covariates : (n_samples, n_covariates) array, optional
        Covariates to partial out. For ``n_covariates > 1`` this function
        delegates to :func:`partial_spearman_batch` directly, because the
        closed-form zero-order identity only applies to a single covariate.

    Returns
    -------
    scalar : mean |rho| across features with a valid correlation. ``nan``
    if no feature has a valid correlation.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]}."
        )

    # Multi-covariate case: defer to the general batch implementation.
    if covariates is not None:
        cov = np.asarray(covariates, dtype=np.float64)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        if cov.shape[1] > 1:
            r, _ = partial_spearman_batch(X, y, cov)
            valid = np.isfinite(r)
            if not valid.any():
                return float("nan")
            return float(np.abs(r[valid]).mean())
        covariates = cov

    ctx = build_partial_spearman_context(y, covariates)
    return mean_abs_partial_spearman_cached(X, ctx)


# ---------------------------------------------------------------------------
#  Null distribution: apply the alignment statistic over an iterable of X's
# ---------------------------------------------------------------------------


def null_alignment_distribution(
    surrogate_X_iter,
    y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    *,
    progress_every: int = 10,
) -> np.ndarray:
    """Apply :func:`mean_abs_partial_spearman` to each surrogate X matrix.

    Builds the cached context once and reuses it across every surrogate.

    Parameters
    ----------
    surrogate_X_iter : iterable of (n_samples, n_features) arrays
        Yields one null X matrix per surrogate. Use an iterator rather
        than a full tensor so callers can stream from disk if needed.
    y, covariates : see :func:`mean_abs_partial_spearman`.
    progress_every : int, default 10
        Emit a log line every ``progress_every`` surrogates processed.

    Returns
    -------
    np.ndarray, shape (n_surrogates,), dtype float64.
    """
    # Multi-covariate: fall back to per-call (no cache optimization available).
    cov_mat = None
    if covariates is not None:
        cov_mat = np.asarray(covariates, dtype=np.float64)
        if cov_mat.ndim == 1:
            cov_mat = cov_mat.reshape(-1, 1)

    if cov_mat is not None and cov_mat.shape[1] > 1:
        values = []
        for i, X_s in enumerate(surrogate_X_iter):
            v = mean_abs_partial_spearman(X_s, y, cov_mat)
            values.append(v)
            if (i + 1) % progress_every == 0:
                logger.info(
                    "null_alignment_distribution: %d surrogates done, latest=%.6f",
                    i + 1, v,
                )
        return np.array(values, dtype=np.float64)

    ctx = build_partial_spearman_context(y, cov_mat)
    values = []
    for i, X_s in enumerate(surrogate_X_iter):
        v = mean_abs_partial_spearman_cached(X_s, ctx)
        values.append(v)
        if (i + 1) % progress_every == 0:
            logger.info(
                "null_alignment_distribution: %d surrogates done, latest=%.6f",
                i + 1, v,
            )
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
#  One-sided empirical p with add-one smoothing
# ---------------------------------------------------------------------------


def one_sided_null_p(real_value: float, null_values: np.ndarray) -> float:
    """One-sided empirical p-value for H1: real > null.

    Uses add-one smoothing: p = (1 + #{null >= real}) / (1 + N_surrogates).
    Floor is 1/(1+N); never zero. Assumes "larger real is better";
    flip signs beforehand for lower-is-better statistics.
    """
    null_values = np.asarray(null_values).ravel()
    null_values = null_values[np.isfinite(null_values)]
    n = len(null_values)
    if n == 0:
        return float("nan")
    n_as_good = int(np.sum(null_values >= real_value))
    return (1.0 + n_as_good) / (1.0 + n)


def summarize_null_test(
    real_value: float, null_values: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """Return (p, z, null_mean, null_std, null_median) for a one-sided test.

    ``z`` is the standard-score of the real value in the null distribution,
    reported as a companion to the empirical p; finite-sample p is the
    primary test.
    """
    null_values = np.asarray(null_values, dtype=np.float64).ravel()
    null_values = null_values[np.isfinite(null_values)]
    mu = float(null_values.mean()) if len(null_values) else float("nan")
    sd = float(null_values.std(ddof=1)) if len(null_values) > 1 else float("nan")
    med = float(np.median(null_values)) if len(null_values) else float("nan")
    z = (real_value - mu) / sd if (sd and np.isfinite(sd)) else float("nan")
    p = one_sided_null_p(real_value, null_values)
    return p, z, mu, sd, med


__all__ = [
    "PartialSpearmanContext",
    "build_partial_spearman_context",
    "mean_abs_partial_spearman",
    "mean_abs_partial_spearman_cached",
    "null_alignment_distribution",
    "one_sided_null_p",
    "summarize_null_test",
]
