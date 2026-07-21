"""Stream-build CCI features from expanded NeuronChat H5 (dict-of-2D format)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from clrc.features.construction import build_ct_region_index

logger = logging.getLogger(__name__)


def build_features_streaming(
    nc_h5_path: Path,
    interaction_names: List[str],
    regions_109: List[str],
    celltypes: List[str],
    node_lookup: Dict[Tuple[str, str], int],
    idx_abc_in_109: np.ndarray,
    flat_idx_map: np.ndarray,
    nan_thresh: float,
    zero_thresh: float,
) -> Tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
    """Stream NeuronChat H5, one interaction at a time.

    For each interaction x celltype-pair, builds the 101-region feature
    vector and applies pre-selection filters (NaN fraction + zero fraction)
    on-the-fly so the full feature matrix never exists in memory.

    Parameters
    ----------
    nc_h5_path : Path
        Expanded NeuronChat H5 (dict-of-2D format under ``net/``).
    interaction_names : list[str]
        Names of LR interactions in the H5.
    regions_109 : list[str]
        All 109 CCI regions (sorted).
    celltypes : list[str]
        All cell types (sorted).
    node_lookup : dict[(region, ct) → int]
        Maps (region, celltype) to node index in the 2133-dim space.
    idx_abc_in_109 : (101,) int array
        Position of each ABC region in regions_109.
    flat_idx_map : (10201,) int array
        Column-major 109x109 → 101x101 index mapping.
    nan_thresh : float
        Minimum fraction of non-NaN values to keep (e.g. 0.20).
    zero_thresh : float
        Maximum fraction of zeros to keep (e.g. 0.95).

    Returns
    -------
    kept_vectors : list of (10201,) arrays
    kept_names : list of feature name strings
    kept_meta : list of per-feature metadata dicts
    """
    n_regions_109 = len(regions_109)
    n_abc = len(idx_abc_in_109)
    n_edges = n_abc * n_abc

    # Precompute per-celltype region-index arrays
    ct_node_idx: Dict[str, np.ndarray] = {}
    for ct in celltypes:
        ct_node_idx[ct] = build_ct_region_index(ct, regions_109, node_lookup)

    kept_vectors: List[np.ndarray] = []
    kept_names: List[str] = []
    kept_meta: List[Dict[str, Any]] = []

    n_total_candidates = 0
    n_passed_nan = 0

    with h5py.File(nc_h5_path, "r") as f:
        net_grp = f["net"]
        for lr_idx, lr_name in enumerate(
            tqdm(interaction_names, desc="Streaming interactions")
        ):
            mat_lr = net_grp[lr_name][:]  # (2133, 2133): senders x receivers

            for ctL in celltypes:
                ligand_idx = ct_node_idx[ctL]
                valid_rows_109 = np.where(ligand_idx >= 0)[0]
                if len(valid_rows_109) == 0:
                    continue
                src_nodes = ligand_idx[valid_rows_109]

                for ctR in celltypes:
                    receptor_idx = ct_node_idx[ctR]
                    valid_cols_109 = np.where(receptor_idx >= 0)[0]
                    if len(valid_cols_109) == 0:
                        continue
                    tgt_nodes = receptor_idx[valid_cols_109]

                    n_total_candidates += 1

                    # Build 109x109 matrix for this LR + celltype pair
                    feat_mat_109 = np.full(
                        (n_regions_109, n_regions_109), np.nan, dtype=np.float64
                    )
                    feat_mat_109[np.ix_(valid_rows_109, valid_cols_109)] = mat_lr[
                        np.ix_(src_nodes, tgt_nodes)
                    ]

                    # Flatten 109x109 column-major, project to 101x101
                    feat_109_flat = feat_mat_109.reshape(-1, order="F")
                    feat_101 = feat_109_flat[flat_idx_map]

                    # --- Pre-selection filters ---
                    n_valid = np.count_nonzero(~np.isnan(feat_101))
                    valid_frac = n_valid / n_edges
                    if valid_frac < nan_thresh:
                        continue

                    feat_101_filled = np.where(np.isnan(feat_101), 0.0, feat_101)
                    zero_frac = np.count_nonzero(feat_101_filled == 0.0) / n_edges
                    if zero_frac > zero_thresh:
                        continue

                    n_passed_nan += 1

                    feature_name = (
                        f"{lr_name}_{ctL.replace(' ', '.')}_{ctR.replace(' ', '.')}"
                    )
                    kept_vectors.append(feat_101_filled)
                    kept_names.append(feature_name)
                    kept_meta.append(
                        dict(
                            feature_name=feature_name,
                            lr_index=lr_idx,
                            lr_name=lr_name,
                            ct_L=ctL,
                            ct_R=ctR,
                        )
                    )

    logger.info("Total celltype-pair candidates evaluated: %d", n_total_candidates)
    logger.info("Passed NaN/zero pre-selection: %d", n_passed_nan)
    return kept_vectors, kept_names, kept_meta
