"""Load ABC-space structural connectivity, functional connectivity, and distance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import h5py

logger = logging.getLogger(__name__)


def load_structural_ABC(
    struct_h5_path: Union[str, Path],
) -> Tuple[Dict[str, Any], List[str]]:
    """Load ABC-space structural (or functional) connectivity from H5."""
    struct_h5_path = Path(struct_h5_path)
    out: Dict[str, Any] = {}
    ABC_regions: List[str] = []

    with h5py.File(struct_h5_path, "r") as f:
        if "conn_allen" in f:
            out["conn_allen"] = f["conn_allen"][:]
        if "conn_ABC_naive" in f:
            out["conn_ABC_naive"] = f["conn_ABC_naive"][:]
        if "conn_ABC_voxel_weighted" in f:
            out["conn_ABC_voxel_weighted"] = f["conn_ABC_voxel_weighted"][:]
        if "metric_names" in f:
            out["metric_names"] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["metric_names"][:]
            ]
        if "metric_name" in f:  # FC compatibility
            out["metric_name"] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["metric_name"][:]
            ]
        if "regions_all_ABC_translated" in f:
            ABC_regions = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["regions_all_ABC_translated"][:]
            ]

    logger.info(
        "Loaded structural/FC ABC from %s: %d regions.",
        struct_h5_path,
        len(ABC_regions),
    )
    return out, ABC_regions


def load_distance_ABC(
    dist_h5_path: Union[str, Path],
) -> Tuple[Dict[str, Any], List[str]]:
    """Load ABC-space fiber-distance matrices from H5."""
    dist_h5_path = Path(dist_h5_path)
    out: Dict[str, Any] = {}
    ABC_regions: List[str] = []

    with h5py.File(dist_h5_path, "r") as f:
        if "dist_allen" in f:
            out["dist_allen"] = f["dist_allen"][:]
        if "dist_ABC_naive" in f:
            out["dist_ABC_naive"] = f["dist_ABC_naive"][:]
        if "dist_ABC_voxel_weighted" in f:
            out["dist_ABC_voxel_weighted"] = f["dist_ABC_voxel_weighted"][:]
        if "metric_name" in f:
            out["metric_name"] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["metric_name"][:]
            ]
        if "regions_all_ABC_translated" in f:
            ABC_regions = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["regions_all_ABC_translated"][:]
            ]

    logger.info(
        "Loaded distance ABC from %s: %d regions.", dist_h5_path, len(ABC_regions)
    )
    return out, ABC_regions
