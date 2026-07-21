#!/usr/bin/env python3
"""Build ABC-space connectivity targets (structural / functional / fiber distance).

Inputs (see ``connectivity_targets:`` in the YAML config):
  - ``abc_region_data``: shared ABC<->Allen mapping CSVs + Allen labels + voxel counts.
  - ``connectivity_targets.structural``: 5 raw DSI Studio ``.mat`` files.
  - ``connectivity_targets.functional``: a single FC ``.mat`` with ``Mean_agr``.
  - ``connectivity_targets.distance``: a single fiber-span distance ``.mat``.

Outputs (gitignored ``data/``):
  - ``data/structural_connectivity/sc_5metrics_ABC.{h5,pkl}``
  - ``data/functional_connectivity/fc_agr_ABC.{h5,pkl}``
  - ``data/structural_connectivity/fiber_distance_ABC.{h5,pkl}``

Usage:
    uv run python src/pipeline/shared/build_connectivity_targets.py \\
        --config configs/abc_expanded.yaml \\
        --target all

``--target`` accepts ``structural``, ``functional``, ``distance``, ``all``,
or a comma-separated list (e.g. ``structural,distance``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np

from clrc.core.io import find_repo_root, load_yaml_config
from clrc.core.logging import setup_logging
from clrc.spatial.atlas import (
    AbcAllenMapping,
    load_abc_allen_mapping,
    load_allen_voxel_counts,
)
from clrc.spatial.connectivity_targets import (
    load_dsi_studio_matrices,
    load_mat_matrix,
    project_allen_to_abc_naive_and_voxel,
    project_multi_allen_to_abc,
    save_connectivity_target_h5,
    save_connectivity_target_pkl,
)

logger = logging.getLogger("clrc.pipeline.build_connectivity_targets")

VALID_TARGETS: Tuple[str, ...] = ("structural", "functional", "distance")


def _parse_targets(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(VALID_TARGETS)
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    bad = [p for p in parts if p not in VALID_TARGETS]
    if bad:
        raise ValueError(
            f"Invalid --target value(s) {bad}. Must be one of "
            f"{sorted(VALID_TARGETS) + ['all']} or a comma-separated list."
        )
    return parts


def _resolve(repo_root: Path, rel: str) -> Path:
    # Local resolver: clrc.core.io only resolves flat top-level sections
    # (data, output, spatial_null, abc_region_data). Nested keys under
    # connectivity_targets.{structural,functional,distance} are resolved here
    # rather than teaching io.py a recursive resolver that could accidentally
    # rewrite non-path strings (e.g. var_name, metric_name).
    p = Path(rel)
    return p if p.is_absolute() else repo_root / p


def _load_mapping_and_voxel(
    cfg: Mapping, repo_root: Path
) -> Tuple[AbcAllenMapping, np.ndarray]:
    abc_data = cfg["abc_region_data"]
    abc_allen_csv = _resolve(repo_root, abc_data["abc_allen_csv"])
    allen_labels_txt = _resolve(repo_root, abc_data["allen_labels_txt"])
    voxel_counts_csv = _resolve(repo_root, abc_data["voxel_counts_csv"])

    logger.info(
        "Loading ABC<->Allen mapping from %s (+ %s)",
        abc_allen_csv, allen_labels_txt,
    )
    mapping = load_abc_allen_mapping(abc_allen_csv, allen_labels_txt)
    allen_labels = mapping["allen_labels"]
    logger.info(
        "Loaded %d Allen labels and %d ABC regions",
        len(allen_labels), len(mapping["abc_regions"]),
    )
    voxel_counts = load_allen_voxel_counts(voxel_counts_csv, allen_labels)
    return mapping, voxel_counts


def _zero_out_nans(arr: np.ndarray, label: str) -> np.ndarray:
    if np.isnan(arr).any():
        logger.warning("%s: NaNs detected; replacing with 0.0", label)
        return np.nan_to_num(arr, nan=0.0)
    return arr


def _build_structural(
    cfg: Mapping, repo_root: Path, mapping: AbcAllenMapping, voxel_counts: np.ndarray,
) -> None:
    ct = cfg["connectivity_targets"]["structural"]
    data_dir = _resolve(repo_root, ct["dsi_studio_dir"])
    metric_files: Mapping[str, str] = ct["dsi_metrics"]
    output_dir = _resolve(repo_root, ct["output_dir"])
    metric_set_name = ct.get("metric_set_name", "sc_5metrics")
    dataset_name = ct.get("dataset_name", "ABC")

    logger.info(
        "Structural: loading %d DSI Studio matrices from %s",
        len(metric_files), data_dir,
    )
    conn_all, metric_names = load_dsi_studio_matrices(data_dir, metric_files)
    logger.info(
        "Structural: conn_all shape=%s, metrics=%s", conn_all.shape, metric_names,
    )
    conn_all = _zero_out_nans(conn_all, "Structural")

    conn_abc_naive, conn_abc_voxel, abc_regions = project_multi_allen_to_abc(
        conn_all, mapping, voxel_counts,
    )
    logger.info(
        "Structural: projected to ABC-space, shape=%s", conn_abc_naive.shape,
    )

    stem = f"{metric_set_name}_{dataset_name}"
    h5_path = save_connectivity_target_h5(
        output_dir / f"{stem}.h5",
        allen_array=conn_all,
        abc_naive=conn_abc_naive,
        abc_voxel=conn_abc_voxel,
        metric_names=metric_names,
        regions_allen=mapping["allen_labels"],
        regions_abc=abc_regions,
    )
    pkl_path = save_connectivity_target_pkl(
        output_dir / f"{stem}.pkl",
        {
            "conn_allen": conn_all,
            "conn_ABC_naive": conn_abc_naive,
            "conn_ABC_voxel_weighted": conn_abc_voxel,
            "metric_names": list(metric_names),
            "regions_conn_all": list(mapping["allen_labels"]),
            "regions_all_ABC_translated": list(abc_regions),
        },
    )
    logger.info("Structural: wrote %s and %s", h5_path, pkl_path)


def _build_functional(
    cfg: Mapping, repo_root: Path, mapping: AbcAllenMapping, voxel_counts: np.ndarray,
) -> None:
    ct = cfg["connectivity_targets"]["functional"]
    fc_mat = _resolve(repo_root, ct["fc_mat"])
    var_name = ct.get("var_name", "Mean_agr")
    metric_name = ct.get("metric_name", "fc_agr")
    output_dir = _resolve(repo_root, ct["output_dir"])
    dataset_name = ct.get("dataset_name", "ABC")

    logger.info(
        "Functional: loading FC matrix from %s (var=%s)", fc_mat, var_name,
    )
    C_allen = load_mat_matrix(fc_mat, var_name)
    logger.info("Functional: C_allen shape=%s", C_allen.shape)
    C_allen = _zero_out_nans(C_allen, "Functional")

    C_abc_naive, C_abc_voxel, abc_regions = project_allen_to_abc_naive_and_voxel(
        C_allen, mapping, voxel_counts,
    )
    logger.info(
        "Functional: projected to ABC-space, shape=%s", C_abc_naive.shape,
    )

    stem = f"{metric_name}_{dataset_name}"
    h5_path = save_connectivity_target_h5(
        output_dir / f"{stem}.h5",
        allen_array=C_allen,
        abc_naive=C_abc_naive,
        abc_voxel=C_abc_voxel,
        metric_names=[metric_name],
        regions_allen=mapping["allen_labels"],
        regions_abc=abc_regions,
    )
    pkl_path = save_connectivity_target_pkl(
        output_dir / f"{stem}.pkl",
        {
            "conn_allen": C_allen,
            "conn_ABC_naive": C_abc_naive,
            "conn_ABC_voxel_weighted": C_abc_voxel,
            "metric_name": metric_name,
            "regions_conn_all": list(mapping["allen_labels"]),
            "regions_all_ABC_translated": list(abc_regions),
        },
    )
    logger.info("Functional: wrote %s and %s", h5_path, pkl_path)


def _build_distance(
    cfg: Mapping, repo_root: Path, mapping: AbcAllenMapping, voxel_counts: np.ndarray,
) -> None:
    ct = cfg["connectivity_targets"]["distance"]
    distance_mat = _resolve(repo_root, ct["distance_mat"])
    var_name = ct.get("var_name", "connectivity")
    metric_name = ct.get("metric_name", "fiber_distance")
    output_dir = _resolve(repo_root, ct["output_dir"])
    dataset_name = ct.get("dataset_name", "ABC")

    logger.info(
        "Distance: loading distance matrix from %s (var=%s)",
        distance_mat, var_name,
    )
    D_allen = load_mat_matrix(distance_mat, var_name)
    logger.info("Distance: D_allen shape=%s", D_allen.shape)
    D_allen = _zero_out_nans(D_allen, "Distance")

    D_abc_naive, D_abc_voxel, abc_regions = project_allen_to_abc_naive_and_voxel(
        D_allen, mapping, voxel_counts,
    )
    logger.info(
        "Distance: projected to ABC-space, shape=%s", D_abc_naive.shape,
    )

    stem = f"{metric_name}_{dataset_name}"
    h5_path = save_connectivity_target_h5(
        output_dir / f"{stem}.h5",
        allen_array=D_allen,
        abc_naive=D_abc_naive,
        abc_voxel=D_abc_voxel,
        metric_names=[metric_name],
        regions_allen=mapping["allen_labels"],
        regions_abc=abc_regions,
        allen_key="dist_allen",
        abc_naive_key="dist_ABC_naive",
        abc_voxel_key="dist_ABC_voxel_weighted",
    )
    pkl_path = save_connectivity_target_pkl(
        output_dir / f"{stem}.pkl",
        {
            "dist_allen": D_allen,
            "dist_ABC_naive": D_abc_naive,
            "dist_ABC_voxel_weighted": D_abc_voxel,
            "metric_name": metric_name,
            "regions_conn_all": list(mapping["allen_labels"]),
            "regions_all_ABC_translated": list(abc_regions),
        },
    )
    logger.info("Distance: wrote %s and %s", h5_path, pkl_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ABC-space connectivity targets (SC / FC / distance).",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g. configs/abc_expanded.yaml).",
    )
    parser.add_argument(
        "--target", default="all",
        help="'structural' | 'functional' | 'distance' | 'all' | comma-separated list.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml_config(config_path)
    repo_root = find_repo_root(config_path.parent)

    out_base = Path(cfg["output"]["base_dir"])
    if not out_base.is_absolute():
        out_base = repo_root / out_base
    stage_out_dir = out_base / "connectivity_targets"
    stage_out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("build_connectivity_targets", output_dir=stage_out_dir)

    targets = _parse_targets(args.target)
    logger.info("Building connectivity targets: %s", targets)
    logger.info("Config:    %s", config_path)
    logger.info("Repo root: %s", repo_root)

    mapping, voxel_counts = _load_mapping_and_voxel(cfg, repo_root)

    builders = {
        "structural": _build_structural,
        "functional": _build_functional,
        "distance": _build_distance,
    }
    for target in targets:
        logger.info("=== Target: %s ===", target)
        builders[target](cfg, repo_root, mapping, voxel_counts)

    logger.info("Done: %d target(s) built.", len(targets))


if __name__ == "__main__":
    main()
