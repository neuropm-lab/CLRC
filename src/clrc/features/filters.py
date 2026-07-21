"""Feature filters: NaN/zero pre-selection and variance post-filter."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def apply_variance_filter(
    X: np.ndarray,
    feature_names: List[str],
    meta: List[Dict[str, Any]],
    var_thresh: float,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """Drop features whose variance falls below *var_thresh*.

    Parameters
    ----------
    X : (n_edges, n_features) array
    feature_names : list[str]
    meta : list[dict]
    var_thresh : float
        Minimum variance to keep (e.g. 1e-6).

    Returns
    -------
    X_out, names_out, meta_out — filtered versions.
    """
    variances = np.var(X, axis=0)
    keep = variances >= var_thresh
    n_before = X.shape[1]
    X_out = X[:, keep]
    names_out = [n for n, k in zip(feature_names, keep) if k]
    meta_out = [m for m, k in zip(meta, keep) if k]

    logger.info(
        "Variance filter: %d -> %d features (dropped %d)",
        n_before,
        X_out.shape[1],
        n_before - X_out.shape[1],
    )
    return X_out, names_out, meta_out
