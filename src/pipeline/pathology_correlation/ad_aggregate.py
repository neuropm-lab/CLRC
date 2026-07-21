#!/usr/bin/env python3
"""Aggregate per-subject NeuronChat H5 files into feature matrices for AD analysis."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from clrc.core.io import load_yaml_config, save_pickle, find_repo_root
from clrc.core.logging import setup_logging
from clrc.ad.h5_loader import load_subject_h5
from clrc.ad.aggregation import (
    collect_global_labels,
    build_label_index_maps,
    aggregate_region_collapsed,
    aggregate_region_specific,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-subject NeuronChat H5 files")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("ad_aggregate", out_base)

    repo_root = find_repo_root()

    # --- Load clinical metadata ---
    clinical_csv_path = repo_root / cfg["data"]["clinical_csv"]
    clinical_xlsx_path = repo_root / cfg["data"]["clinical_xlsx"]

    clinical_csv = pd.read_csv(clinical_csv_path)
    id_mapping = clinical_csv[["projid", "individualID"]].drop_duplicates()
    logger.info("ID mapping: %d entries", len(id_mapping))

    clinical_xlsx = pd.read_excel(clinical_xlsx_path)
    clinical_vars = cfg["clinical"]["variables"]
    covariates = cfg["clinical"]["covariates"]
    clinical_cols = ["projid"] + clinical_vars + covariates
    clinical_data = clinical_xlsx[clinical_cols].copy()
    logger.info("Clinical data: %d subjects", len(clinical_data))

    merged_clinical = id_mapping.merge(clinical_data, on="projid", how="inner")
    merged_clinical = merged_clinical.set_index("individualID")
    logger.info("Merged: %d subjects with both IDs", len(merged_clinical))

    # --- Find H5 files, match to clinical ---
    h5_dir = repo_root / cfg["data"]["h5_dir"]
    h5_pattern = cfg["data"]["h5_pattern"]
    h5_files = sorted(h5_dir.glob(h5_pattern))
    logger.info("Found %d H5 files", len(h5_files))

    valid_paths: list[Path] = []
    valid_ids: list[str] = []
    for fpath in h5_files:
        match = re.match(r"nc_subj_(.+)_M\d+\.h5", fpath.name)
        if match:
            individual_id = match.group(1)
            if individual_id in merged_clinical.index:
                valid_paths.append(fpath)
                valid_ids.append(individual_id)

    logger.info("Valid subjects (H5 + clinical): %d", len(valid_ids))

    # --- First pass: collect global labels ---
    all_region_ct, lr_labels, unique_regions, unique_celltypes = collect_global_labels(
        valid_paths
    )
    logger.info(
        "Union labels: %d region::CT, %d LR, %d regions, %d celltypes",
        len(all_region_ct), len(lr_labels), len(unique_regions), len(unique_celltypes),
    )

    label_ct_idx, label_region_idx = build_label_index_maps(
        all_region_ct, unique_celltypes, unique_regions
    )

    # --- Second pass: load all subject data ---
    logger.info("Loading subject data...")
    subject_data = {}
    for sid, fpath in tqdm(
        zip(valid_ids, valid_paths), total=len(valid_ids), desc="Loading H5"
    ):
        subject_data[sid] = load_subject_h5(fpath)

    # --- Aggregate ---
    X_collapsed, feature_names_collapsed = aggregate_region_collapsed(
        subject_data, valid_ids, lr_labels, unique_celltypes, label_ct_idx
    )

    X_region, feature_names_region, region_names = aggregate_region_specific(
        subject_data, valid_ids, lr_labels, unique_regions, unique_celltypes,
        label_ct_idx, label_region_idx,
    )

    # --- Build clinical arrays ---
    n_subjects = len(valid_ids)
    Y_clinical = merged_clinical.loc[valid_ids, clinical_vars].to_numpy(dtype=float)
    covariates_arr = merged_clinical.loc[valid_ids, covariates].to_numpy(dtype=float)

    # --- Save ---
    agg_dir = out_base / "aggregation"
    agg_dir.mkdir(parents=True, exist_ok=True)

    save_pickle(
        {
            "X_collapsed": X_collapsed,
            "X_region": X_region,
            "feature_names_collapsed": feature_names_collapsed,
            "feature_names_region": feature_names_region,
            "region_names": region_names,
            "Y_clinical": Y_clinical,
            "covariates_arr": covariates_arr,
            "subject_ids": valid_ids,
            "clinical_vars": clinical_vars,
            "covariate_names": covariates,
            "lr_labels": lr_labels,
            "unique_celltypes": unique_celltypes,
            "unique_regions": unique_regions,
        },
        agg_dir / "ad_aggregation.pkl",
    )

    logger.info(
        "Saved aggregation: %d subjects, collapsed=%s, region=%s",
        n_subjects, X_collapsed.shape, X_region.shape,
    )


if __name__ == "__main__":
    main()
