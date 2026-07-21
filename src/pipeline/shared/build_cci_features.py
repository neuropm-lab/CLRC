#!/usr/bin/env python3
"""Build CCI features from expanded NeuronChat H5. Calls clrc.features.streaming."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from clrc.core.io import load_yaml_config, save_pickle
from clrc.core.logging import setup_logging
from clrc.features.alignment import load_structural_ABC, load_distance_ABC
from clrc.features.construction import (
    build_edge_table,
    parse_group_names,
    restrict_to_ABC,
    vectorize_sc_block,
)
from clrc.features.filters import apply_variance_filter
from clrc.features.streaming import build_features_streaming


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CCI features from NeuronChat H5")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("build_cci_features", out_base)

    # Load NeuronChat H5 metadata
    nc_h5 = Path(cfg["data"]["nc_h5"])
    with h5py.File(nc_h5, "r") as f:
        interaction_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f.attrs["interaction_names"]
        ]
        group_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f.attrs["group_names"]
        ]

    logger.info("NeuronChat H5: %d interactions, %d groups", len(interaction_names), len(group_names))
    _, _, node_lookup, regions_109, celltypes = parse_group_names(group_names)

    # Load structural targets
    struct_data, ABC_regions_struct = load_structural_ABC(cfg["data"]["sc_h5"])

    # Load distance
    dist_data = None
    dist_h5 = Path(cfg["data"].get("distance_h5", ""))
    if dist_h5.exists():
        dist_data, _ = load_distance_ABC(str(dist_h5))

    # Build 109 -> ABC mapping
    region_aliases = cfg.get("region_aliases", {"A24": "ACC"})
    idx_abc_in_109, flat_idx_map, ABC_regions_cci = restrict_to_ABC(
        regions_109, ABC_regions_struct, region_aliases=region_aliases
    )

    # Stream features
    feat_cfg = cfg["features"]
    kept_vectors, kept_names, kept_meta = build_features_streaming(
        nc_h5_path=nc_h5,
        interaction_names=interaction_names,
        regions_109=regions_109,
        celltypes=celltypes,
        node_lookup=node_lookup,
        idx_abc_in_109=idx_abc_in_109,
        flat_idx_map=flat_idx_map,
        nan_thresh=feat_cfg["non_nan_thresh"],
        zero_thresh=feat_cfg["zero_max_frac"],
    )

    if len(kept_vectors) == 0:
        logger.error("No features survived pre-selection.")
        return

    X_preselect = np.column_stack(kept_vectors)
    logger.info("Pre-selection: %s (%.1f MB)", X_preselect.shape, X_preselect.nbytes / 1e6)

    # Variance filter
    X_kept, names_kept, meta_kept = apply_variance_filter(
        X_preselect, kept_names, kept_meta, feat_cfg.get("var_thresh", 1e-6)
    )

    # Build aligned pickle
    metric_names = struct_data.get("metric_names") or struct_data.get("metric_name")
    assert metric_names is not None, (
        "struct_data has neither 'metric_names' nor 'metric_name' — "
        "build_connectivity_targets.py always writes one of the two."
    )
    SC_naive_mat, _ = vectorize_sc_block(
        np.asarray(struct_data["conn_ABC_naive"]), metric_names
    )
    SC_voxel_mat, _ = vectorize_sc_block(
        np.asarray(struct_data["conn_ABC_voxel_weighted"]), metric_names
    )

    distance_vec_naive = distance_vec_voxel = None
    if dist_data is not None:
        dn = dist_data.get("dist_ABC_naive")
        dv = dist_data.get("dist_ABC_voxel_weighted")
        if dn is not None:
            distance_vec_naive = np.asarray(dn).reshape(-1, order="F")
        if dv is not None:
            distance_vec_voxel = np.asarray(dv).reshape(-1, order="F")

    edge_table = build_edge_table(ABC_regions_cci)

    payload = dict(
        ABC_regions_struct=ABC_regions_struct,
        ABC_regions_cci=ABC_regions_cci,
        vectorization_order="F",
        edge_table=edge_table,
        X_kept_np=X_kept,
        feature_names_kept=names_kept,
        meta_ABC_kept=meta_kept,
        metric_names=metric_names,
        SC_naive_mat=SC_naive_mat,
        SC_voxel_mat=SC_voxel_mat,
        distance_vec_naive=distance_vec_naive,
        distance_vec_voxel=distance_vec_voxel,
    )

    output_path = args.output or Path(cfg["data"]["alignment_pkl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(payload, output_path)
    logger.info("Saved alignment pickle: %s (%d features)", output_path, X_kept.shape[1])


if __name__ == "__main__":
    main()
