"""Atlas-based region centroid extraction and ABC-space distance matrix.

Implements the centroid-first pathway used to build a 101x101 metric
Euclidean distance matrix over ABC regions for brainSMASH variogram
matching:

1. Load a labeled Allen reference NIfTI (integer region IDs 1..141).
   Compute the 3D centroid of each labeled region in millimetre (world)
   coordinates using ``img.affine``.
2. For each ABC region (101 total), aggregate the centroids of its
   constituent Allen subregions via a voxel-count-weighted mean. The
   ABC <-> Allen many-to-many mapping is encoded in
   ``abc_allenRef_averaged.csv`` as '+'-joined labels, which this module
   parses natively.
3. Build the 101x101 symmetric, zero-diagonal Euclidean distance matrix
   between ABC centroids, ordered by a caller-supplied list of ABC
   region names (so the matrix aligns row-for-row with the ABC ordering
   used everywhere else in the pipeline, e.g. ``ABC_regions_cci`` from
   the alignment pickle).

This centroid-then-metric pipeline differs from the ``P @ D @ P.T``
projection used by ``clrc.spatial.connectivity_targets`` -- that
projection is appropriate for *averaging a pairwise quantity* (like
fibre distance already defined between Allen regions) but is not a
true metric distance over ABC centroids, because Euclidean distance
is nonlinear under weighted averaging. brainSMASH's variogram
matching requires a point-in-space metric, so we build centroids
first and then compute distances.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ABC <-> Allen mapping helpers (native implementations).
# ---------------------------------------------------------------------------


class AbcAllenMapping(TypedDict):
    allen_labels: List[str]
    allen_index: Dict[str, int]
    abc_to_allen_idx: Dict[str, List[int]]
    abc_regions: List[str]


def _parse_plus_list(s) -> List[str]:
    """Split a '+'-joined label string, trimming whitespace and dropping empties."""
    if pd.isna(s):
        return []
    parts = [p.strip() for p in str(s).split("+")]
    return [p for p in parts if p]


def load_allen_labels(txt_path: Path) -> List[str]:
    """Load Allen reference region labels from a two-column text file
    (``index <whitespace> label``), in line order.
    """
    labels: List[str] = []
    txt_path = Path(txt_path)
    with txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Unexpected line in {txt_path.name}: {line!r}"
                )
            labels.append(parts[-1])
    return labels


def load_abc_allen_mapping(
    abc_allen_csv: Path, allen_labels_txt: Path
) -> AbcAllenMapping:
    """Parse ``abc_allenRef_averaged.csv`` plus Allen labels into an ABC<->Allen
    many-to-many mapping.

    Returns a dict with keys:
        - ``allen_labels``:     list[str] — Allen region labels in NIfTI order.
        - ``allen_index``:      dict[label -> int] — inverse of allen_labels.
        - ``abc_to_allen_idx``: dict[abc_region -> sorted list[int]].
        - ``abc_regions``:      list[str] — ABC region names in insertion order.
    """
    abc_allen_csv = Path(abc_allen_csv)
    allen_labels_txt = Path(allen_labels_txt)

    allen_labels = load_allen_labels(allen_labels_txt)
    allen_index = {lab: i for i, lab in enumerate(allen_labels)}

    df = pd.read_csv(abc_allen_csv)
    if "ABC region" not in df.columns or "Allen ref region" not in df.columns:
        raise ValueError(
            f"{abc_allen_csv.name} is missing required columns "
            "'ABC region' / 'Allen ref region'. Found: "
            f"{list(df.columns)}"
        )

    abc_to_allen: Dict[str, set] = {}
    missing_allen: set = set()

    for _, row in df.iterrows():
        abc_components = _parse_plus_list(row["ABC region"])
        allen_components = _parse_plus_list(row["Allen ref region"])
        if not abc_components or not allen_components:
            continue
        for abc in abc_components:
            s = abc_to_allen.setdefault(abc, set())
            for lab in allen_components:
                if lab in allen_index:
                    s.add(lab)
                else:
                    missing_allen.add(lab)

    if missing_allen:
        logger.warning(
            "Allen labels in %s not found in %s and ignored: %s",
            abc_allen_csv.name, allen_labels_txt.name, sorted(missing_allen),
        )

    abc_to_allen_idx: Dict[str, list] = {}
    for abc, labs in abc_to_allen.items():
        idxs = [allen_index[lab] for lab in labs if lab in allen_index]
        if idxs:
            abc_to_allen_idx[abc] = sorted(idxs)

    abc_regions = list(abc_to_allen_idx.keys())

    return {
        "allen_labels": allen_labels,
        "allen_index": allen_index,
        "abc_to_allen_idx": abc_to_allen_idx,
        "abc_regions": abc_regions,
    }


def load_allen_voxel_counts(
    voxel_counts_csv: Path, allen_labels: Sequence[str]
) -> np.ndarray:
    """Load Allen voxel counts and align them to ``allen_labels``.

    Returns a ``(len(allen_labels),)`` float array. Labels without a voxel
    count are assigned 0.0 (callers fall back to equal weighting).
    """
    voxel_counts_csv = Path(voxel_counts_csv)
    df = pd.read_csv(voxel_counts_csv)

    required_cols = {"RegionName", "VoxelCount"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{voxel_counts_csv.name} must contain columns {required_cols}; "
            f"found {df.columns.tolist()}"
        )

    name_to_vox = dict(
        zip(df["RegionName"].astype(str), df["VoxelCount"].astype(float))
    )

    voxel_counts = np.zeros(len(allen_labels), dtype=float)
    missing: list = []
    for i, lab in enumerate(allen_labels):
        if lab in name_to_vox:
            voxel_counts[i] = name_to_vox[lab]
        else:
            missing.append(lab)
            voxel_counts[i] = 0.0

    if missing:
        logger.warning(
            "Allen labels without voxel counts (fall back to equal weighting): %s",
            missing,
        )

    return voxel_counts


# ---------------------------------------------------------------------------
# Allen centroids from a labeled NIfTI
# ---------------------------------------------------------------------------


def load_allen_centroids(nii_path: Path) -> pd.DataFrame:
    """Compute per-region 3D centroids (mm) from a labeled Allen NIfTI.

    For each non-zero integer label ``r`` in the volume, takes the mean
    voxel coordinate across all voxels with that label, then transforms
    to world (millimetre) coordinates via ``img.affine``.

    Parameters
    ----------
    nii_path
        Path to an integer-labeled NIfTI whose voxel values are the Allen
        region IDs (e.g. 1..141 for the Allen reference).

    Returns
    -------
    DataFrame
        Columns ``region_id`` (int), ``x_mm``, ``y_mm``, ``z_mm`` (float).
        Sorted by ``region_id`` ascending. Excludes label 0 (background).
    """
    import nibabel as nib  # local import -- optional heavy dep

    nii_path = Path(nii_path)
    if not nii_path.is_file():
        raise FileNotFoundError(f"Atlas NIfTI not found: {nii_path}")

    img = nib.load(str(nii_path))
    affine = np.asarray(img.affine, dtype=np.float64)
    # Use asanyarray on dataobj to avoid materialising the full float array
    # when the on-disk dtype is small int; the unique-label scan is O(N) in
    # voxels regardless.
    data = np.asanyarray(img.dataobj)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D labeled volume; got shape {data.shape} for {nii_path}."
        )

    # Integer labels only; silently cast float volumes to int if the values
    # are integer-valued. If they aren't, error out -- we should not be
    # silently rounding a probabilistic map.
    if not np.issubdtype(data.dtype, np.integer):
        if not np.allclose(data, np.round(data)):
            raise ValueError(
                f"Atlas {nii_path} has non-integer labels; expected an "
                "integer-labeled parcellation."
            )
        data = data.astype(np.int64)

    labels = np.unique(data)
    labels = labels[labels != 0]
    if labels.size == 0:
        raise ValueError(f"Atlas {nii_path} has no non-zero labels.")

    # Flatten once; use np.where per-label. For the Allen atlas
    # (~14 M non-zero voxels, 141 labels) the loop is fast enough.
    idx_i, idx_j, idx_k = np.nonzero(data)
    flat_labels = data[idx_i, idx_j, idx_k]

    rows = []
    for region_id in labels:
        mask = flat_labels == region_id
        if not mask.any():
            continue
        ii = idx_i[mask]
        jj = idx_j[mask]
        kk = idx_k[mask]
        # Mean voxel coordinate -> world (mm) via affine
        voxel_mean = np.array(
            [ii.mean(), jj.mean(), kk.mean(), 1.0], dtype=np.float64
        )
        world = affine @ voxel_mean
        rows.append(
            {
                "region_id": int(region_id),
                "x_mm": float(world[0]),
                "y_mm": float(world[1]),
                "z_mm": float(world[2]),
            }
        )

    df = pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)
    logger.info(
        "load_allen_centroids: %s -> %d regions (labels %d..%d)",
        nii_path, len(df), int(df["region_id"].min()), int(df["region_id"].max()),
    )
    return df


# ---------------------------------------------------------------------------
# ABC centroids from Allen centroids + many-to-many mapping + voxel counts
# ---------------------------------------------------------------------------


def compute_abc_centroids(
    allen_centroids: pd.DataFrame,
    abc_allen_mapping: AbcAllenMapping,
    voxel_counts: np.ndarray,
) -> pd.DataFrame:
    """Voxel-count-weighted mean of Allen centroids -> ABC centroids.

    For each ABC region ``a`` with constituent Allen indices
    ``i in abc_to_allen_idx[a]`` and voxel counts ``v_i``, the centroid is

        centroid(a) = sum_i v_i * centroid_Allen(i) / sum_i v_i

    matching the ``P_vox`` weighting in
    :func:`clrc.spatial.connectivity_targets.project_allen_to_abc_naive_and_voxel`.
    If the sum of voxel counts across constituents is zero, we fall back
    to the equal-weight mean (matches the ``w_naive`` fallback in that
    function).

    Parameters
    ----------
    allen_centroids
        DataFrame from :func:`load_allen_centroids` -- must be 1-indexed
        by ``region_id`` matching the Allen reference label scheme so
        that ``allen_to_allen_idx`` from the mapping (0-indexed positions
        within ``allen_labels``) can be used to pull rows.
    abc_allen_mapping
        dict from :func:`load_abc_allen_mapping`.
    voxel_counts
        (R,) array of Allen voxel counts aligned to the mapping's
        ``allen_labels``.

    Returns
    -------
    DataFrame
        Columns ``abc_region`` (str), ``x_mm``, ``y_mm``, ``z_mm``.
        Row order matches ``abc_allen_mapping['abc_regions']``.
    """
    allen_labels = list(abc_allen_mapping["allen_labels"])
    abc_to_allen_idx: Mapping[str, Sequence[int]] = abc_allen_mapping["abc_to_allen_idx"]
    abc_regions: Sequence[str] = abc_allen_mapping["abc_regions"]

    R = len(allen_labels)
    if allen_centroids.shape[0] != R:
        raise ValueError(
            f"allen_centroids has {allen_centroids.shape[0]} rows but the "
            f"mapping expects {R} Allen labels. Check that the NIfTI "
            "parcellation matches allenRef_region_labels.txt."
        )
    voxel_counts = np.asarray(voxel_counts, dtype=np.float64)
    if voxel_counts.shape != (R,):
        raise ValueError(
            f"voxel_counts shape {voxel_counts.shape} does not match "
            f"#allen_labels={R}."
        )

    # Centroid rows aligned by allen_labels order (i.e. allen_index).
    # allen_centroids is sorted by region_id, which equals the Allen
    # reference ID scheme (labels file's first column). The mapping's
    # ``allen_index`` maps a region-name string to a 0-based position in
    # ``allen_labels``, and allen_labels itself is that same 0-based
    # listing. If allen_centroids has region_id == positional_index + 1
    # (1-based labels 1..R) we reindex by position; otherwise we require
    # an explicit 1:1 row ordering by region_id.
    expected_ids = np.arange(1, R + 1, dtype=np.int64)
    if np.array_equal(allen_centroids["region_id"].to_numpy(), expected_ids):
        centroid_xyz = allen_centroids[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    else:
        # Non-sequential region_ids -- reindex on region_id with a
        # position lookup. We expect region_ids to be exactly 1..R; if
        # they are not, something is wrong upstream.
        ids = allen_centroids["region_id"].to_numpy()
        missing = set(expected_ids.tolist()) - set(ids.tolist())
        if missing:
            raise ValueError(
                f"Allen centroids missing expected region_ids: "
                f"{sorted(missing)[:10]}... total missing={len(missing)}. "
                "Atlas NIfTI and allenRef_region_labels.txt must agree."
            )
        order = {rid: pos for pos, rid in enumerate(ids)}
        perm = np.array([order[rid] for rid in expected_ids], dtype=np.int64)
        centroid_xyz = allen_centroids[["x_mm", "y_mm", "z_mm"]].to_numpy(
            dtype=np.float64
        )[perm]

    rows = []
    for abc in abc_regions:
        idxs = list(abc_to_allen_idx[abc])
        if not idxs:
            raise ValueError(
                f"ABC region {abc!r} has no Allen constituents in mapping."
            )
        v = voxel_counts[idxs]
        total_v = float(v.sum())
        sub_xyz = centroid_xyz[idxs, :]
        if total_v > 0:
            centroid = (v[:, None] * sub_xyz).sum(axis=0) / total_v
        else:
            # Equal-weight fallback (matches P_naive fallback in pipeline).
            centroid = sub_xyz.mean(axis=0)
            logger.warning(
                "compute_abc_centroids: ABC region %r has zero total voxel "
                "count across constituents; using equal-weight fallback.",
                abc,
            )
        rows.append(
            {
                "abc_region": abc,
                "x_mm": float(centroid[0]),
                "y_mm": float(centroid[1]),
                "z_mm": float(centroid[2]),
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "compute_abc_centroids: %d ABC regions (voxel-weighted mean of "
        "Allen centroids)",
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Pairwise Euclidean distance matrix ordered by a caller-supplied list
# ---------------------------------------------------------------------------


def build_abc_distance_matrix(
    abc_centroids: pd.DataFrame,
    ordering: Sequence[str],
) -> np.ndarray:
    """Build a 101x101 symmetric Euclidean distance matrix, ordered.

    Parameters
    ----------
    abc_centroids
        DataFrame from :func:`compute_abc_centroids`.
    ordering
        List of ABC region names in the desired row/column order (e.g.
        the ``ABC_regions_cci`` list from the alignment pickle). Every
        entry must appear in ``abc_centroids['abc_region']``.

    Returns
    -------
    D : (N, N) ndarray
        Symmetric, zero-diagonal, non-negative Euclidean distance matrix
        between ABC centroids, ordered by ``ordering``.
    """
    lookup = {
        row.abc_region: np.array([row.x_mm, row.y_mm, row.z_mm], dtype=np.float64)
        for row in abc_centroids.itertuples(index=False)
    }
    missing = [r for r in ordering if r not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} region(s) in `ordering` not present in "
            f"abc_centroids: {missing[:5]}..."
        )
    coords = np.stack([lookup[r] for r in ordering], axis=0)

    # Pairwise Euclidean distance via broadcasting. N is small (~101).
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=-1))
    # Symmetrise to kill any FP asymmetry from the sqrt path
    D = 0.5 * (D + D.T)
    # Zero diagonal exactly (eliminate FP noise from i==i sqrt(0)).
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Collapse-and-redistribute pipeline for the 86 unique spatial nodes.
#
# The 141-region Allen reference parcellation cannot uniquely resolve all
# 101 ABC regions used downstream: 22 ABC regions collapse onto 7 shared
# Allen-constituent sets (each set being a frozenset of Allen indices).
# That degeneracy is a fundamental data-resolution limit of the available
# atlas, not a bug. To run brainSMASH on a strictly-positive off-diagonal
# distance matrix (its check_distmat contract), we reduce to 86 unique
# spatial nodes (one per unique Allen-constituent set), surrogate at the
# 86-node level, then redistribute each surrogate value back to all ABC
# subregions in the corresponding group.
#
# Conservativeness: within-group variance is forced to zero in surrogates
# because the 141-Allen parcellation cannot distinguish those ABC
# subregions. This is a conservative null -- it under-represents
# small-scale spatial structure that exists biologically but is invisible
# to the atlas.
# ---------------------------------------------------------------------------


def build_unique_spatial_nodes(mapping: AbcAllenMapping) -> pd.DataFrame:
    """Reduce ABC regions to unique spatial nodes by Allen-constituent set.

    Two ABC regions that share the exact same set of Allen constituents
    are spatially indistinguishable in our centroid pipeline -- they
    would yield identical centroids and a zero off-diagonal in the
    distance matrix. We collapse them into one "unique spatial node".

    Parameters
    ----------
    mapping
        dict from :func:`load_abc_allen_mapping`. Must contain
        ``abc_to_allen_idx`` and ``abc_regions``.

    Returns
    -------
    DataFrame
        One row per unique Allen-constituent set, ordered by ascending
        ``group_id``. Columns:

        * ``group_id`` (int): 0..G-1 in order of first appearance.
        * ``allen_idx_tuple`` (tuple[int, ...]): sorted tuple of Allen
          indices (i.e. positions within ``allen_labels``).
        * ``abc_regions`` (list[str]): all ABC region names that share
          this constituent set.
        * ``n_abc_regions`` (int): ``len(abc_regions)`` for convenience.

    Notes
    -----
    Group IDs are assigned in the order the unique sets are first
    encountered while iterating over ``mapping['abc_regions']``. This
    keeps the assignment deterministic for a given mapping object.
    """
    abc_to_allen_idx: Mapping[str, Sequence[int]] = mapping["abc_to_allen_idx"]
    abc_regions: Sequence[str] = mapping["abc_regions"]

    set_to_group: Dict[Tuple[int, ...], int] = {}
    rows: list = []
    for abc in abc_regions:
        idxs = tuple(sorted(int(i) for i in abc_to_allen_idx[abc]))
        if not idxs:
            raise ValueError(
                f"ABC region {abc!r} has empty Allen constituent set in mapping."
            )
        if idxs not in set_to_group:
            gid = len(set_to_group)
            set_to_group[idxs] = gid
            rows.append(
                {
                    "group_id": gid,
                    "allen_idx_tuple": idxs,
                    "abc_regions": [abc],
                }
            )
        else:
            gid = set_to_group[idxs]
            rows[gid]["abc_regions"].append(abc)

    df = pd.DataFrame(rows)
    df["n_abc_regions"] = df["abc_regions"].map(len)
    logger.info(
        "build_unique_spatial_nodes: %d ABC regions -> %d unique spatial groups "
        "(%d singletons, %d multi-member groups, max group size=%d)",
        len(abc_regions),
        len(df),
        int((df["n_abc_regions"] == 1).sum()),
        int((df["n_abc_regions"] > 1).sum()),
        int(df["n_abc_regions"].max()) if len(df) else 0,
    )
    return df


def compute_unique_centroids(
    unique_nodes_df: pd.DataFrame,
    allen_centroids: pd.DataFrame,
    voxel_counts: np.ndarray,
) -> pd.DataFrame:
    """Voxel-count-weighted centroid for each unique spatial group.

    For each unique group with Allen indices ``i in allen_idx_tuple`` and
    voxel counts ``v_i``,

        centroid(group) = sum_i v_i * centroid_Allen(i) / sum_i v_i

    falling back to equal-weight mean if ``sum_i v_i == 0`` (matches the
    P_naive fallback in :func:`compute_abc_centroids`).

    Parameters
    ----------
    unique_nodes_df
        DataFrame from :func:`build_unique_spatial_nodes`.
    allen_centroids
        DataFrame from :func:`load_allen_centroids` -- expected to have
        ``region_id`` equal to 1..R (1-based positional, matching Allen
        labels file).
    voxel_counts
        (R,) array of Allen voxel counts aligned to the mapping's
        ``allen_labels``.

    Returns
    -------
    DataFrame
        One row per group. Columns:
        ``group_id``, ``x_mm``, ``y_mm``, ``z_mm``, ``abc_regions``,
        ``n_abc_regions``.
    """
    R = len(voxel_counts)
    voxel_counts = np.asarray(voxel_counts, dtype=np.float64)
    if voxel_counts.shape != (R,):
        raise ValueError(
            f"voxel_counts shape {voxel_counts.shape} unexpected (1D required)."
        )

    # Reindex allen_centroids by 1..R region_id ordering, matching the
    # convention in compute_abc_centroids.
    expected_ids = np.arange(1, R + 1, dtype=np.int64)
    if np.array_equal(allen_centroids["region_id"].to_numpy(), expected_ids):
        centroid_xyz = allen_centroids[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    else:
        ids = allen_centroids["region_id"].to_numpy()
        missing = set(expected_ids.tolist()) - set(ids.tolist())
        if missing:
            raise ValueError(
                f"Allen centroids missing expected region_ids: "
                f"{sorted(missing)[:10]}... total missing={len(missing)}."
            )
        order = {rid: pos for pos, rid in enumerate(ids)}
        perm = np.array([order[rid] for rid in expected_ids], dtype=np.int64)
        centroid_xyz = allen_centroids[["x_mm", "y_mm", "z_mm"]].to_numpy(
            dtype=np.float64
        )[perm]

    rows: list = []
    for row in unique_nodes_df.itertuples(index=False):
        idxs = list(row.allen_idx_tuple)
        v = voxel_counts[idxs]
        total_v = float(v.sum())
        sub_xyz = centroid_xyz[idxs, :]
        if total_v > 0:
            centroid = (v[:, None] * sub_xyz).sum(axis=0) / total_v
        else:
            centroid = sub_xyz.mean(axis=0)
            logger.warning(
                "compute_unique_centroids: group %d (ABC %s) has zero total "
                "voxel count across Allen constituents; using equal-weight "
                "fallback.",
                int(row.group_id), list(row.abc_regions),
            )
        rows.append(
            {
                "group_id": int(row.group_id),
                "x_mm": float(centroid[0]),
                "y_mm": float(centroid[1]),
                "z_mm": float(centroid[2]),
                "abc_regions": list(row.abc_regions),
                "n_abc_regions": int(len(row.abc_regions)),
            }
        )

    df = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)
    logger.info(
        "compute_unique_centroids: %d unique spatial groups (voxel-weighted "
        "Allen centroids)",
        len(df),
    )
    return df


def build_unique_distance_matrix(unique_centroids_df: pd.DataFrame) -> np.ndarray:
    """Symmetric, zero-diagonal Euclidean distance matrix over unique groups.

    Validates that all off-diagonal entries are strictly positive (any
    zero off-diagonal would mean two groups share a centroid, which
    contradicts the construction by unique constituent set and would
    break brainSMASH's variogram).

    Parameters
    ----------
    unique_centroids_df
        DataFrame from :func:`compute_unique_centroids`. Must contain
        ``group_id``, ``x_mm``, ``y_mm``, ``z_mm`` columns. Rows must be
        sorted by ``group_id`` ascending and contiguous 0..G-1.

    Returns
    -------
    D : (G, G) ndarray
        Symmetric, zero-diagonal, strictly-positive-off-diagonal
        Euclidean distance matrix.
    """
    df = unique_centroids_df.sort_values("group_id").reset_index(drop=True)
    expected = np.arange(len(df), dtype=int)
    if not np.array_equal(df["group_id"].to_numpy(), expected):
        raise ValueError(
            "unique_centroids_df group_id column must be a contiguous "
            "0..G-1 range."
        )

    coords = df[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=-1))
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    if not np.all(np.isfinite(D)):
        raise ValueError("unique-group distance matrix contains non-finite values.")
    mask = ~np.eye(D.shape[0], dtype=bool)
    if D.shape[0] > 1:
        off_diag_min = float(D[mask].min())
        if off_diag_min <= 0.0:
            pairs = np.argwhere((D == 0.0) & mask)
            raise ValueError(
                f"unique-group distance matrix has {pairs.shape[0]} zero "
                f"off-diagonal entries (examples: {pairs[:5].tolist()}). "
                "Two unique groups share a centroid -- this contradicts "
                "construction by unique constituent set."
            )
    return D


def aggregate_abc_map_to_unique(
    abc_values: np.ndarray,
    unique_nodes_df: pd.DataFrame,
    voxel_counts: np.ndarray,
    abc_to_allen_idx: Mapping[str, Sequence[int]],
    abc_regions_ordering: Sequence[str],
) -> np.ndarray:
    """Aggregate a 101-length ABC-region map to an 86-length unique-group map.

    For each unique group, the aggregated value is the voxel-count-weighted
    mean of the ABC values for the ABC regions belonging to that group.
    The weight for ABC region ``a`` is ``sum_i voxel_counts[i] for i in
    abc_to_allen_idx[a]``. If all member ABC regions have zero total
    voxel count, fall back to the equal-weight mean.

    Parameters
    ----------
    abc_values : (n_abc,) array
        Scalar map indexed by ``abc_regions_ordering``.
    unique_nodes_df
        DataFrame from :func:`build_unique_spatial_nodes`.
    voxel_counts : (R,) array
        Allen voxel counts aligned to the mapping's ``allen_labels``.
    abc_to_allen_idx
        ``mapping['abc_to_allen_idx']`` (after any region_aliases
        renaming if used by the caller).
    abc_regions_ordering
        ABC region names in the order that ``abc_values`` is indexed by.

    Returns
    -------
    unique_values : (G,) array
        Aggregated map at the unique-group resolution, ordered by
        ``group_id``.
    """
    abc_values = np.asarray(abc_values, dtype=np.float64)
    if abc_values.ndim != 1:
        raise ValueError(
            f"abc_values must be 1D; got shape {abc_values.shape}."
        )
    abc_regions_ordering = list(abc_regions_ordering)
    if abc_values.shape[0] != len(abc_regions_ordering):
        raise ValueError(
            f"abc_values length {abc_values.shape[0]} != "
            f"len(abc_regions_ordering)={len(abc_regions_ordering)}."
        )
    voxel_counts = np.asarray(voxel_counts, dtype=np.float64)

    pos_in_abc = {name: i for i, name in enumerate(abc_regions_ordering)}

    G = len(unique_nodes_df)
    out = np.empty(G, dtype=np.float64)
    for row in unique_nodes_df.itertuples(index=False):
        members = list(row.abc_regions)
        weights = np.array(
            [
                float(voxel_counts[list(abc_to_allen_idx[a])].sum())
                for a in members
            ],
            dtype=np.float64,
        )
        idxs = np.array([pos_in_abc[a] for a in members], dtype=np.int64)
        vals = abc_values[idxs]
        total_w = float(weights.sum())
        if total_w > 0:
            out[int(row.group_id)] = float((weights * vals).sum() / total_w)
        else:
            out[int(row.group_id)] = float(vals.mean())
    return out


def redistribute_surrogate_to_abc(
    unique_surrogate: np.ndarray,
    unique_nodes_df: pd.DataFrame,
    abc_regions_ordering: Sequence[str],
) -> np.ndarray:
    """Broadcast each unique-group surrogate value to its member ABC regions.

    Parameters
    ----------
    unique_surrogate : (n_surrogates, G) or (G,) array
        Surrogate values at the unique-group resolution.
    unique_nodes_df
        DataFrame from :func:`build_unique_spatial_nodes`. Each row
        carries the list of ABC region names in that group.
    abc_regions_ordering
        ABC region names defining the order of columns in the output.

    Returns
    -------
    abc_surrogate : (n_surrogates, n_abc) or (n_abc,) array
        Each ABC region inherits the surrogate value of its parent
        unique group. Within-group ABC values are identical (this is the
        conservative null discussed in the module docstring of
        ``src/pipeline/connectivity_prediction/spatial_null.py``).
    """
    arr = np.asarray(unique_surrogate, dtype=np.float64)
    one_d = arr.ndim == 1
    if one_d:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            f"unique_surrogate must be 1D or 2D; got shape {unique_surrogate.shape}."
        )

    n_surr, G = arr.shape
    if G != len(unique_nodes_df):
        raise ValueError(
            f"unique_surrogate has {G} columns but unique_nodes_df has "
            f"{len(unique_nodes_df)} groups."
        )

    abc_regions_ordering = list(abc_regions_ordering)
    pos_in_abc = {name: i for i, name in enumerate(abc_regions_ordering)}

    out = np.empty((n_surr, len(abc_regions_ordering)), dtype=np.float64)
    for row in unique_nodes_df.itertuples(index=False):
        gid = int(row.group_id)
        col_idxs = []
        for a in row.abc_regions:
            if a not in pos_in_abc:
                raise KeyError(
                    f"ABC region {a!r} (member of group {gid}) is not in "
                    "abc_regions_ordering."
                )
            col_idxs.append(pos_in_abc[a])
        out[:, np.array(col_idxs, dtype=np.int64)] = arr[:, gid][:, None]

    if one_d:
        return out[0]
    return out


__all__ = [
    "load_abc_allen_mapping",
    "load_allen_voxel_counts",
    "load_allen_centroids",
    "compute_abc_centroids",
    "build_abc_distance_matrix",
    "build_unique_spatial_nodes",
    "compute_unique_centroids",
    "build_unique_distance_matrix",
    "aggregate_abc_map_to_unique",
    "redistribute_surrogate_to_abc",
]
