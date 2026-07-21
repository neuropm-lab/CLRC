"""Leave-One-Brain-Region-Out (LOBO) cross-validation utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np
import pandas as pd

from clrc.core.transforms import ECDFTransform, fit_ecdf
from clrc.core.types import AlignmentData, FeatureMeta

logger = logging.getLogger(__name__)


class LoboFoldSplit(TypedDict):
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    y_train_t: np.ndarray
    y_test_t: np.ndarray
    ecdf: Optional[ECDFTransform]


def infer_regions(edge_table: pd.DataFrame) -> List[str]:
    """Get sorted unique regions from an edge table."""
    src_u = pd.unique(edge_table["src_region"])
    tgt_u = pd.unique(edge_table["tgt_region"])
    return sorted(set(src_u).union(set(tgt_u)))


def precompute_fold_masks(
    edge_table: pd.DataFrame,
    regions: List[str],
) -> Dict[str, np.ndarray]:
    """Precompute boolean test masks for each holdout region."""
    src = edge_table["src_region"].to_numpy()
    tgt = edge_table["tgt_region"].to_numpy()
    masks: Dict[str, np.ndarray] = {}
    for region in regions:
        is_test = (src == region) | (tgt == region)
        masks[region] = is_test
    return masks


# Synthetic meta entry for the fiber-distance feature. Conforms to FeatureMeta
# schema with all non-CCI fields set to None. Used for distance_only and
# cci_distance modes so the returned (X, feature_names, meta) triple stays
# aligned row-for-row.
_DISTANCE_META: FeatureMeta = {
    "feature_name": "fiber_distance",
    "lr_name": None,
    "ct_L": None,
    "ct_R": None,
    "lr_index": None,
    "ligand_genes": None,
    "receptor_genes": None,
}


def select_features(
    data: AlignmentData,
    *,
    feature_mode: Literal["cci_only", "distance_only", "cci_distance"],
) -> Tuple[np.ndarray, List[str], List[FeatureMeta]]:
    """Select features based on feature_mode.

    Returns an aligned triple ``(X, feature_names, meta)`` where all three
    have the same length along the feature axis. Callers can trust that
    ``feature_names[i]`` and ``meta[i]`` correspond to column ``X[:, i]``
    for every ``i`` — no manual realignment needed.

    Raises
    ------
    ValueError
        If ``data.meta`` is None (required for all modes), or if the
        requested mode needs distance data that isn't available.
    """
    if data.feature_names is None:
        raise ValueError(
            "select_features requires data.feature_names to be set. "
            "Load the alignment pickle via load_alignment_data with a version "
            "that writes feature_names_kept."
        )
    if data.meta is None:
        raise ValueError(
            "select_features requires data.meta to be set. "
            "Load the alignment pickle via load_alignment_data with a version "
            "that writes meta_ABC_kept (feature metadata)."
        )

    cci_names = list(data.feature_names)
    cci_meta = list(data.meta)

    if feature_mode == "cci_only":
        return data.X, cci_names, cci_meta

    if feature_mode == "distance_only":
        if data.distance_vec is None:
            raise ValueError("distance_only mode requires distance data in pickle.")
        X_dist = data.distance_vec.reshape(-1, 1)
        return X_dist, ["fiber_distance"], [cast(FeatureMeta, dict(_DISTANCE_META))]

    if feature_mode == "cci_distance":
        if data.distance_vec is None:
            logger.warning("distance_vec not available, falling back to cci_only.")
            return data.X, cci_names, cci_meta
        X_combined = np.column_stack([data.X, data.distance_vec])
        feature_names = cci_names + ["fiber_distance"]
        meta = cci_meta + [cast(FeatureMeta, dict(_DISTANCE_META))]
        return X_combined, feature_names, meta

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def compute_lobo_fold_split(
    *,
    holdout_region: str,
    y_all: np.ndarray,
    fold_masks: Dict[str, np.ndarray],
    eps: float = 0.0,
    y_transform: Literal["none", "log1p", "ecdf"] = "none",
    data_type: Literal["SC", "FC"] = "FC",
) -> Optional[LoboFoldSplit]:
    """Compute one LOBO fold's index-and-transform split for a holdout region.

    Returns a dict with ``train_idx``, ``test_idx``, ``y_train_raw``,
    ``y_test_raw``, ``y_train_t``, ``y_test_t``, and ``ecdf`` (or ``None``
    if the transform does not produce one). Pickle-friendly: holds only
    numpy arrays plus an optional :class:`ECDFTransform`. ``X_train`` /
    ``X_test`` must be sliced separately from ``X`` using the returned
    indices (this keeps the cached payload small).

    Returns ``None`` if the fold has zero train or test rows after
    ``eps`` filtering (caller should skip the region and log).
    """
    y_all = np.asarray(y_all, dtype=float).ravel()
    test_mask = fold_masks[holdout_region]
    train_idx_all = np.flatnonzero(~test_mask)
    test_idx_all = np.flatnonzero(test_mask)

    y_train_all = y_all[train_idx_all]
    y_test_all = y_all[test_idx_all]

    if data_type == "SC":
        train_keep = np.isfinite(y_train_all) & (y_train_all > eps)
        test_keep = np.isfinite(y_test_all) & (y_test_all > eps)
    else:
        train_keep = np.isfinite(y_train_all) & (y_train_all != eps)
        test_keep = np.isfinite(y_test_all) & (y_test_all != eps)

    train_idx = train_idx_all[train_keep]
    test_idx = test_idx_all[test_keep]
    if train_idx.size == 0 or test_idx.size == 0:
        logger.warning(
            "Skipping fold '%s': %d train / %d test rows after eps=%s filter. "
            "Region likely has zero target values (e.g. SC tractography limit).",
            holdout_region, int(train_idx.size), int(test_idx.size), eps,
        )
        return None

    y_train_raw = y_all[train_idx]
    y_test_raw = y_all[test_idx]

    ecdf: Optional[ECDFTransform] = None
    if y_transform == "none":
        y_train_t, y_test_t = y_train_raw, y_test_raw
    elif y_transform == "log1p":
        if data_type == "FC":
            min_val = float(np.min([np.min(y_train_raw), np.min(y_test_raw)]))
            shift = -min_val if min_val < 0 else 0.0
            y_train_raw = y_train_raw + shift
            y_test_raw = y_test_raw + shift
        if (y_train_raw < 0).any() or (y_test_raw < 0).any():
            raise ValueError("log1p requires non-negative y values.")
        y_train_t = np.log1p(y_train_raw)
        y_test_t = np.log1p(y_test_raw)
    elif y_transform == "ecdf":
        ecdf = fit_ecdf(y_train_raw)
        y_train_t = ecdf.forward(y_train_raw)
        y_test_t = ecdf.forward(y_test_raw)
    else:
        raise ValueError(f"Unknown y_transform: {y_transform}")

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "y_train_t": y_train_t,
        "y_test_t": y_test_t,
        "ecdf": ecdf,
    }


def iter_lobo_folds(
    X: np.ndarray,
    edge_table: pd.DataFrame,
    y_all: np.ndarray,
    fold_masks: Dict[str, np.ndarray],
    *,
    eps: float = 0.0,
    y_transform: Literal["none", "log1p", "ecdf"] = "none",
    data_type: Literal["SC", "FC"] = "FC",
    include_edge_tables: bool = False,
    regions: Optional[Sequence[str]] = None,
) -> Iterator[
    Tuple[
        str,                     # holdout_region
        np.ndarray,              # X_train
        np.ndarray,              # y_train_t
        np.ndarray,              # X_test
        np.ndarray,              # y_test_t
        np.ndarray,              # y_train_raw
        np.ndarray,              # y_test_raw
        Optional[ECDFTransform], # ecdf
        Optional[pd.DataFrame],  # edge_table_test
        np.ndarray,              # test_idx
    ]
]:
    """Iterate LOBO folds using precomputed masks.

    Yields one tuple per holdout region with train/test splits and
    transforms. Delegates per-region index+transform derivation to
    :func:`compute_lobo_fold_split` so cached-fold callers can reuse the
    same primitive.
    """
    y_all = np.asarray(y_all, dtype=float).ravel()

    if regions is None:
        regions_use = sorted(fold_masks.keys())
    else:
        regions_use = list(regions)

    base_edges = edge_table.loc[:, ["edge_idx", "src_region", "tgt_region"]]

    for region in regions_use:
        split = compute_lobo_fold_split(
            holdout_region=region,
            y_all=y_all,
            fold_masks=fold_masks,
            eps=eps,
            y_transform=y_transform,
            data_type=data_type,
        )
        if split is None:
            # Skip-warning already emitted inside compute_lobo_fold_split.
            continue

        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]

        if include_edge_tables:
            et_test = base_edges.iloc[test_idx].copy().reset_index(drop=True)
            et_test["y_raw"] = split["y_test_raw"]
        else:
            et_test = None

        yield (
            region,
            X_train,
            split["y_train_t"],
            X_test,
            split["y_test_t"],
            split["y_train_raw"],
            split["y_test_raw"],
            split["ecdf"],
            et_test,
            test_idx,
        )
