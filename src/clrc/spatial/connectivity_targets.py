"""Build ABC-space connectivity targets from Allen-space matrices.

Used by ``src/pipeline/shared/build_connectivity_targets.py`` to produce the
structural, functional, and fiber-distance target H5/pickle artifacts
consumed by the XGBoost LOBO training driver.

Provides:
- :func:`load_mat_matrix` -- scipy.io.loadmat with v7.3 h5py fallback.
- :func:`load_dsi_studio_matrices` -- consolidate the 5 raw DSI Studio
  ``connectivity_*_1mm.mat`` files into a stacked ``(n_metrics, R, R)``
  Allen-space tensor.
- :func:`project_allen_to_abc_naive_and_voxel` -- ``C_ABC = P @ C_Allen @ P.T``
  with both equal-weight and voxel-weighted projection matrices.
- :func:`save_connectivity_target_h5` / :func:`save_connectivity_target_pkl`
  -- unified writers for the on-disk connectivity-target artifacts.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
from h5py import special_dtype
from scipy.io import loadmat

from clrc.spatial.atlas import AbcAllenMapping

logger = logging.getLogger(__name__)


def load_mat_matrix(mat_path: Path, var_name: str) -> np.ndarray:
    """Load a 2D numeric matrix from a MATLAB .mat file under ``var_name``.

    Tries ``scipy.io.loadmat`` (v7 and earlier) first, falls back to h5py
    for v7.3 MAT-files.
    """
    mat_path = Path(mat_path)

    try:
        mat = loadmat(mat_path)
        if var_name in mat:
            arr = mat[var_name]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return arr
    except Exception as e:
        logger.debug(
            "scipy.io.loadmat failed on %s: %r (trying h5py fallback)",
            mat_path, e,
        )

    if not h5py.is_hdf5(mat_path):
        raise RuntimeError(
            f"Could not load {mat_path} with scipy.io.loadmat and file is not "
            "HDF5 (v7.3). Check the MAT-file version and variable name."
        )

    with h5py.File(mat_path, "r") as f:
        if var_name not in f:
            raise KeyError(
                f"Variable {var_name!r} not found in HDF5 MAT-file {mat_path}"
            )
        arr = f[var_name][:]

    if arr.ndim != 2:
        raise ValueError(
            f"Loaded {var_name!r} from {mat_path}, got shape={arr.shape} "
            "(expected 2D)."
        )
    return arr


def load_dsi_studio_matrices(
    data_dir: Path,
    metric_files: Mapping[str, str],
) -> Tuple[np.ndarray, List[str]]:
    """Consolidate the raw DSI Studio ``connectivity_*_1mm.mat`` files.

    Parameters
    ----------
    data_dir :
        Directory containing the raw ``.mat`` files.
    metric_files :
        Mapping of ``metric_name -> filename``. Output axis-0 order matches
        the insertion order of this mapping.

    Returns
    -------
    conn_all :
        ``(n_metrics, R, R)`` ndarray stacked in insertion order.
    metric_names :
        List of metric names matching axis 0.
    """
    data_dir = Path(data_dir)
    metric_names: List[str] = list(metric_files.keys())
    mats: list = []
    R_ref: int | None = None

    for metric_name in metric_names:
        mat_path = data_dir / metric_files[metric_name]
        if not mat_path.is_file():
            raise FileNotFoundError(
                f"DSI Studio matrix not found for metric {metric_name!r}: {mat_path}"
            )
        # Use load_mat_matrix so v7.3 DSI Studio .mat files (HDF5-based) also
        # work via h5py fallback, not only the v7 files that scipy can open.
        C = load_mat_matrix(mat_path, "connectivity")
        if C.shape[0] != C.shape[1]:
            raise ValueError(
                f"{mat_path.name}/connectivity is not square, shape={C.shape}"
            )
        if R_ref is None:
            R_ref = C.shape[0]
        elif C.shape[0] != R_ref:
            raise ValueError(
                f"Inconsistent matrix size for {metric_name}: {C.shape}, "
                f"expected {R_ref}x{R_ref}"
            )
        mats.append(np.asarray(C, dtype=float))

    conn_all = np.stack(mats, axis=0)
    return conn_all, metric_names


def project_allen_to_abc_naive_and_voxel(
    C_allen: np.ndarray,
    mapping: AbcAllenMapping,
    voxel_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Project an Allen-space matrix to ABC space via ``P @ C @ P.T``.

    Returns both naive (equal-weight) and voxel-weighted ABC projections
    alongside the ABC region order they share. Regions without voxel counts
    fall back to equal weighting.
    """
    abc_to_allen_idx = mapping["abc_to_allen_idx"]
    allen_labels = mapping["allen_labels"]

    R1, R2 = C_allen.shape
    if R1 != R2:
        raise ValueError(f"C_allen must be square, got shape={C_allen.shape}")
    if R1 != len(allen_labels):
        raise ValueError(
            f"C_allen size {R1} does not match number of Allen labels "
            f"{len(allen_labels)}"
        )
    if voxel_counts.shape[0] != R1:
        raise ValueError(
            f"voxel_counts length {voxel_counts.shape[0]} does not match "
            f"C_allen size {R1}"
        )

    abc_regions: List[str] = list(abc_to_allen_idx.keys())
    n_abc = len(abc_regions)

    P_naive = np.zeros((n_abc, R1), dtype=float)
    P_vox = np.zeros((n_abc, R1), dtype=float)

    for a_idx, abc in enumerate(abc_regions):
        idxs = abc_to_allen_idx[abc]
        if not idxs:
            continue
        w_naive = 1.0 / len(idxs)
        P_naive[a_idx, idxs] = w_naive
        v = voxel_counts[idxs]
        total_v = v.sum()
        if total_v > 0:
            P_vox[a_idx, idxs] = v / total_v
        else:
            P_vox[a_idx, idxs] = w_naive

    C_abc_naive = (P_naive @ C_allen) @ P_naive.T
    C_abc_voxel = (P_vox @ C_allen) @ P_vox.T
    return C_abc_naive, C_abc_voxel, abc_regions


def project_multi_allen_to_abc(
    conn_all: np.ndarray,
    mapping: AbcAllenMapping,
    voxel_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Stacked-metric variant of :func:`project_allen_to_abc_naive_and_voxel`.

    Expects ``conn_all`` of shape ``(n_metrics, R, R)`` and returns two
    ``(n_metrics, N_ABC, N_ABC)`` tensors plus the shared ABC region order.
    """
    n_metrics, R1, R2 = conn_all.shape
    if R1 != R2:
        raise ValueError(
            f"conn_all must be (n_metrics, R, R), got shape={conn_all.shape}"
        )

    C_abc_naive0, C_abc_voxel0, abc_regions = project_allen_to_abc_naive_and_voxel(
        conn_all[0], mapping, voxel_counts
    )
    n_abc = C_abc_naive0.shape[0]
    conn_abc_naive = np.zeros((n_metrics, n_abc, n_abc), dtype=float)
    conn_abc_voxel = np.zeros_like(conn_abc_naive)
    conn_abc_naive[0] = C_abc_naive0
    conn_abc_voxel[0] = C_abc_voxel0

    for k in range(1, n_metrics):
        n_k, v_k, abc_k = project_allen_to_abc_naive_and_voxel(
            conn_all[k], mapping, voxel_counts
        )
        if abc_k != abc_regions:
            raise RuntimeError("ABC region order mismatch across metrics.")
        conn_abc_naive[k] = n_k
        conn_abc_voxel[k] = v_k

    return conn_abc_naive, conn_abc_voxel, abc_regions


def save_connectivity_target_h5(
    out_path: Path,
    *,
    allen_array: np.ndarray,
    abc_naive: np.ndarray,
    abc_voxel: np.ndarray,
    metric_names: Sequence[str],
    regions_allen: Sequence[str],
    regions_abc: Sequence[str],
    allen_key: str = "conn_allen",
    abc_naive_key: str = "conn_ABC_naive",
    abc_voxel_key: str = "conn_ABC_voxel_weighted",
) -> Path:
    """Write an HDF5 connectivity-target artifact.

    Datasets written:
        - ``{allen_key}``                      (Allen-space matrix/tensor)
        - ``{abc_naive_key}``                  (ABC-space equal-weight projection)
        - ``{abc_voxel_key}``                  (ABC-space voxel-weighted projection)
        - ``metric_names``                     (VLA of strings)
        - ``regions_conn_all``                 (VLA of Allen region labels)
        - ``regions_all_ABC_translated``       (VLA of ABC region labels)

    Use ``dist_*`` keys for the distance target to preserve the existing
    on-disk schema read by ``clrc/features/alignment.py``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    str_dt = special_dtype(vlen=str)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(allen_key, data=allen_array, compression="gzip")
        f.create_dataset(abc_naive_key, data=abc_naive, compression="gzip")
        f.create_dataset(abc_voxel_key, data=abc_voxel, compression="gzip")
        f.create_dataset(
            "metric_names",
            data=np.array(list(metric_names), dtype=str_dt),
            dtype=str_dt,
        )
        f.create_dataset(
            "regions_conn_all",
            data=np.array(list(regions_allen), dtype=str_dt),
            dtype=str_dt,
        )
        f.create_dataset(
            "regions_all_ABC_translated",
            data=np.array(list(regions_abc), dtype=str_dt),
            dtype=str_dt,
        )
    return out_path


def save_connectivity_target_pkl(
    out_path: Path,
    data: Dict[str, Any],
) -> Path:
    """Write a pickle connectivity-target artifact."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    return out_path
