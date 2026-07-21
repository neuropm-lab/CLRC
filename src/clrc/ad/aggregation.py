"""AD CCI aggregation: region-collapsed and region-specific feature matrices.

Builds subject x feature matrices from heterogeneous per-subject NeuronChat H5 files.
Subjects may have different sets of region::celltype groups, so the code builds a
union of all labels and maps each subject's data into the shared space.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from clrc.ad.h5_loader import parse_region_celltype

logger = logging.getLogger(__name__)


def build_feature_name(lr: str, sender_ct: str, receiver_ct: str) -> str:
    """Build feature name: '{LR} | {sender_ct} -> {receiver_ct}'."""
    return f"{lr} | {sender_ct}\u2192{receiver_ct}"


def collect_global_labels(
    h5_paths: Sequence[Union[str, Path]],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """First pass: collect union of all region::celltype labels and LR names.

    Returns
    -------
    all_region_ct_labels : sorted list of unique 'Region::CellType' labels
    all_lr_labels : list of LR interaction names (from first file)
    unique_regions : sorted unique brain regions
    unique_celltypes : sorted unique cell types
    """
    all_region_ct = set()
    all_lr = None

    for fpath in h5_paths:
        import h5py

        with h5py.File(fpath, "r") as f:
            labels_region_ct = list(f.attrs["group_names"])
            labels_lr = list(f.attrs["interaction_names"])
        all_region_ct.update(labels_region_ct)
        if all_lr is None:
            all_lr = labels_lr

    all_region_ct_sorted = sorted(all_region_ct)
    regions = set()
    celltypes = set()
    for label in all_region_ct_sorted:
        r, ct = parse_region_celltype(label)
        regions.add(r)
        celltypes.add(ct)

    return (
        all_region_ct_sorted,
        all_lr if all_lr is not None else [],
        sorted(regions),
        sorted(celltypes),
    )


def aggregate_region_collapsed(
    subject_data: Dict[str, Dict],
    subject_ids: List[str],
    lr_labels: List[str],
    unique_celltypes: List[str],
    global_label_ct_idx: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """Build region-collapsed feature matrix: average across brain regions.

    For each subject, each LR interaction, and each sender/receiver celltype pair,
    we average the NeuronChat net values across all brain regions that have both
    the sender and receiver celltypes present.

    Parameters
    ----------
    subject_data : dict[subject_id -> dict with 'net', 'labels_region_ct']
    subject_ids : ordered list of subject IDs
    lr_labels : list of LR interaction names
    unique_celltypes : sorted unique cell types (union across subjects)
    global_label_ct_idx : mapping from 'Region::CellType' label to celltype index

    Returns
    -------
    X_collapsed : (n_subjects, n_lr * n_ct * n_ct) array
    feature_names : list of feature name strings
    """
    n_subjects = len(subject_ids)
    n_lr = len(lr_labels)
    n_ct = len(unique_celltypes)
    n_features = n_lr * n_ct * n_ct

    X_collapsed = np.full((n_subjects, n_features), np.nan)

    feature_names = []
    for lr_name in lr_labels:
        for sender_ct in unique_celltypes:
            for receiver_ct in unique_celltypes:
                feature_names.append(build_feature_name(lr_name, sender_ct, receiver_ct))

    for subj_idx, sid in enumerate(tqdm(subject_ids, desc="Region-collapsed")):
        data = subject_data[sid]
        net = data["net"]  # (n_lr, n_subj_labels, n_subj_labels)
        subj_labels = data["labels_region_ct"]
        n_subj_labels = len(subj_labels)

        subj_ct_idx = [global_label_ct_idx[lbl] for lbl in subj_labels]

        ct_aggregated = np.zeros((n_lr, n_ct, n_ct))
        ct_counts = np.zeros((n_lr, n_ct, n_ct))

        for i in range(n_subj_labels):
            for j in range(n_subj_labels):
                sender_ct = subj_ct_idx[i]
                receiver_ct = subj_ct_idx[j]
                ct_aggregated[:, sender_ct, receiver_ct] += net[:, i, j]
                ct_counts[:, sender_ct, receiver_ct] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            ct_aggregated = np.where(
                ct_counts > 0, ct_aggregated / ct_counts, np.nan
            )

        X_collapsed[subj_idx, :] = ct_aggregated.flatten()

    logger.info(
        "Region-collapsed: %d subjects x %d features", n_subjects, n_features
    )
    return X_collapsed, feature_names


def aggregate_region_specific(
    subject_data: Dict[str, Dict],
    subject_ids: List[str],
    lr_labels: List[str],
    unique_regions: List[str],
    unique_celltypes: List[str],
    global_label_ct_idx: Dict[str, int],
    global_label_region_idx: Dict[str, int],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build region-specific feature matrix: keep brain region dimension.

    For each subject, each brain region, each LR interaction, and each
    sender/receiver celltype pair, we use the within-region NeuronChat values.

    Parameters
    ----------
    subject_data : dict[subject_id -> dict with 'net', 'labels_region_ct']
    subject_ids : ordered list of subject IDs
    lr_labels : list of LR interaction names
    unique_regions : sorted unique brain regions
    unique_celltypes : sorted unique cell types
    global_label_ct_idx : mapping 'Region::CellType' -> celltype index
    global_label_region_idx : mapping 'Region::CellType' -> region index

    Returns
    -------
    X_region : (n_subjects, n_regions, n_features_per_region) array
    feature_names_per_region : list of per-region feature names
    region_names : list of region names
    """
    n_subjects = len(subject_ids)
    n_lr = len(lr_labels)
    n_regions = len(unique_regions)
    n_ct = len(unique_celltypes)
    n_features_per_region = n_lr * n_ct * n_ct

    X_region = np.full(
        (n_subjects, n_regions, n_features_per_region), np.nan
    )

    feature_names_per_region = []
    for lr_name in lr_labels:
        for sender_ct in unique_celltypes:
            for receiver_ct in unique_celltypes:
                feature_names_per_region.append(
                    build_feature_name(lr_name, sender_ct, receiver_ct)
                )

    for subj_idx, sid in enumerate(tqdm(subject_ids, desc="Region-specific")):
        data = subject_data[sid]
        net = data["net"]
        subj_labels = data["labels_region_ct"]
        n_subj_labels = len(subj_labels)

        subj_region_idx = [global_label_region_idx[lbl] for lbl in subj_labels]
        subj_ct_idx = [global_label_ct_idx[lbl] for lbl in subj_labels]

        for region_idx in range(n_regions):
            region_mask = [
                i for i in range(n_subj_labels) if subj_region_idx[i] == region_idx
            ]
            if len(region_mask) == 0:
                continue

            ct_aggregated = np.zeros((n_lr, n_ct, n_ct))
            ct_counts = np.zeros((n_lr, n_ct, n_ct))

            for i in region_mask:
                for j in region_mask:
                    sender_ct = subj_ct_idx[i]
                    receiver_ct = subj_ct_idx[j]
                    ct_aggregated[:, sender_ct, receiver_ct] += net[:, i, j]
                    ct_counts[:, sender_ct, receiver_ct] += 1

            with np.errstate(divide="ignore", invalid="ignore"):
                ct_aggregated = np.where(
                    ct_counts > 0, ct_aggregated / ct_counts, np.nan
                )

            X_region[subj_idx, region_idx, :] = ct_aggregated.flatten()

    logger.info(
        "Region-specific: %d subjects x %d regions x %d features",
        n_subjects, n_regions, n_features_per_region,
    )
    return X_region, feature_names_per_region, unique_regions


def build_label_index_maps(
    all_region_ct_labels: List[str],
    unique_celltypes: List[str],
    unique_regions: List[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build mappings from global 'Region::CellType' labels to celltype/region indices.

    Returns (global_label_ct_idx, global_label_region_idx).
    """
    ct_to_idx = {ct: i for i, ct in enumerate(unique_celltypes)}
    region_to_idx = {r: i for i, r in enumerate(unique_regions)}

    global_label_ct_idx = {}
    global_label_region_idx = {}
    for label in all_region_ct_labels:
        region, ct = parse_region_celltype(label)
        global_label_ct_idx[label] = ct_to_idx[ct]
        global_label_region_idx[label] = region_to_idx[region]

    return global_label_ct_idx, global_label_region_idx
