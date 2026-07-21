"""Shared dataclass definitions (no loaders — those live in io.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from clrc.core.transforms import ECDFTransform


class FeatureMeta(TypedDict):
    """Per-feature metadata for a CCI / distance feature.

    Every entry in ``AlignmentData.meta`` conforms to this schema after
    ``load_alignment_data`` validates it. Downstream code (importance
    aggregation, interpretation, plotting) relies on this contract and
    uses direct subscript access — no ``.get()`` fallbacks.

    Fields
    ------
    feature_name
        Canonical identifier for the feature. For CCI features this is the
        LR-pair + sender-CT + receiver-CT concatenation. For the synthetic
        fiber-distance feature it is ``"fiber_distance"``.
    lr_name
        Ligand-receptor pair name (e.g. ``"EFNA5_EPHA4"``), or ``None`` for
        non-CCI features (e.g. fiber_distance).
    ct_L
        Sender cell type (ligand side), or ``None`` for non-CCI features.
    ct_R
        Receiver cell type (receptor side), or ``None`` for non-CCI features.
    lr_index
        Integer index of the LR pair in the upstream interaction list, or
        ``None`` for non-CCI features.
    ligand_genes
        Semicolon-separated ligand gene symbols, or ``None``.
    receptor_genes
        Semicolon-separated receptor gene symbols, or ``None``.
    """

    feature_name: str
    lr_name: Optional[str]
    ct_L: Optional[str]
    ct_R: Optional[str]
    lr_index: Optional[int]
    ligand_genes: Optional[str]
    receptor_genes: Optional[str]


@dataclass
class AlignmentData:
    """Pre-aligned CCI features + structural/functional connectivity targets.

    Attributes
    ----------
    edge_table
        DataFrame with ``edge_idx``, ``src_region``, ``tgt_region`` columns
        describing each region-region edge in the prediction space.
    X
        Feature matrix, shape ``(n_edges, n_features)``.
    metric_names
        Labels for the columns of ``SC_naive`` and ``SC_voxel``.
    SC_naive, SC_voxel
        Target matrices (naive / voxel-weighted versions), shape
        ``(n_edges, n_metrics)``.
    feature_names
        Optional list of length ``n_features`` — real LR + CT labels, e.g.
        ``"EFNA5_EPHA4_Deep-layer.intratelencephalic_LAMP5-LHX6.and.Chandelier"``.
        Required when meta is also provided (the two must be aligned).
    meta
        Optional list of length ``n_features`` of :class:`FeatureMeta` dicts.
        Validated at load time by ``clrc.core.io.load_alignment_data``.
        Every entry has all required keys; downstream code uses direct
        subscript access.
    distance_vec
        Optional per-edge fiber-length distance vector, shape ``(n_edges,)``.
    """

    edge_table: pd.DataFrame
    X: np.ndarray
    metric_names: List[str]
    SC_naive: np.ndarray
    SC_voxel: np.ndarray
    feature_names: Optional[List[str]] = None
    meta: Optional[List[FeatureMeta]] = None
    distance_vec: Optional[np.ndarray] = None


@dataclass
class FoldArtifact:
    """Per-fold results from LOBO cross-validation.

    ``n_features`` records the width of the feature matrix at training time.
    It is written at artifact construction so downstream importance
    aggregation does not have to re-derive the feature count from the
    booster dump.
    """

    holdout_region: str
    metric: str
    version: str
    n_train: int
    n_test: int
    n_features: int
    eps: float
    y_transform: str
    params: Dict
    best_iteration: int
    model_raw: bytes
    ecdf: Optional[ECDFTransform]
    test_idx: np.ndarray
    y_test_raw: np.ndarray
    y_test_t: np.ndarray
    y_pred: np.ndarray
    fold_rmse: float
    fold_mae: float
    edge_table_test: Optional[pd.DataFrame]
    eval_metrics: List[str]
