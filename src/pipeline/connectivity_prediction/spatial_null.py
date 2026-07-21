#!/usr/bin/env python3
"""Spatial-null driver: distance matrix + brainSMASH surrogates + NC rebuild.

Centroid-first distance pathway with collapse-and-redistribute for
ABC regions that share Allen-constituent sets; per-map node-level
surrogacy at the unique-spatial-node grid; per-surrogate NeuronChat
rebuild on a reconstructed pseudo-AnnData; then LOBO XGBoost with
HPO-best params on the resulting null X.

Conservative null framing
-------------------------
The 141-region Allen reference parcellation cannot uniquely resolve all
101 ABC regions: 22 ABC regions collapse onto 7 shared Allen-constituent
sets (frozensets of Allen indices), leaving 86 unique spatial nodes.
We surrogate at the 86-node resolution and broadcast each surrogate
value to the ABC subregions sharing that constituent set. Within-group
variance is forced to zero in surrogates because the 141-Allen
parcellation cannot distinguish these ABC subregions. This is a
conservative null.

Stages
------
--stage build_distances
    1. Load labeled Allen NIfTI (${ATLAS_NII_PATH}) and compute per-region
       (1..141) 3D centroids in mm (via ``img.affine``).
    2. Build the ABC<->Allen many-to-many mapping via
       :func:`clrc.spatial.atlas.load_abc_allen_mapping`.
    3. Reduce to 86 unique spatial nodes via
       :func:`clrc.spatial.atlas.build_unique_spatial_nodes` (one node
       per unique Allen-constituent set across the 101 ABC regions).
    4. Compute voxel-count-weighted Allen centroids per unique group.
    5. Build the 86x86 Euclidean distance matrix and validate it
       (symmetric, zero diagonal, strictly-positive off-diagonal).
       Save to ``<output>/spatial_null/unique_distance_matrix_86.npy``,
       with sidecar CSVs ``unique_centroids.csv`` (group_id, x_mm, y_mm,
       z_mm, abc_regions_in_group) and ``abc_to_group_id.csv``
       (101-row mapping ABC region -> group_id).

--stage generate_surrogates
    1. Load the ABC AnnData (``data/UMAP_ANNDATA/ABC.h5ad``). Aggregate to
       per-(region, supercluster_name) cell counts -> (109, 31) abundance
       matrix, then restrict rows to the 101 ABC regions used throughout
       the pipeline.
    2. Aggregate log1p(mean raw counts) per (region, gene) from the same
       AnnData -> (109, n_genes) expression matrix, restricted to 101 ABC
       regions.
    3. For each column (cell-type / gene), aggregate the 101-length ABC
       map to an 86-length unique-group map (voxel-weighted mean of
       member ABC values), surrogate via brainSMASH on the 86x86
       distance matrix, then redistribute each surrogate back to a
       101-length ABC vector.
    4. Save both 86-node and 101-ABC forms to
       ``<output>/spatial_null/surrogates_celltype.npz`` and
       ``.../surrogates_gene.npz`` with sidecar metadata (column labels,
       region ordering, group ordering, seed).

--stage all
    Runs ``build_distances`` then ``generate_surrogates`` then
    ``rebuild_features`` (with LOBO) sequentially — full end-to-end path.

--stage rebuild_features
    For each surrogate draw:
    1. Load per-surrogate 101-ABC x 31-celltype abundance map and 101-ABC
       x 285-gene expression map from the npz files.
    2. Build a pseudo-AnnData whose per-(region, celltype) aggregate
       expression matches ``surrogate_gene_expr[region] *
       surrogate_abundance[region, ct]`` (see
       :func:`clrc.spatial.nc_rebuild.reconstruct_pseudo_anndata`).
    3. Run NeuronChat on the pseudo-AnnData (``M=50``, ``device='cuda'``).
    4. Stream the resulting H5 into a (n_edges, n_features) null CCI
       feature matrix aligned column-for-column to the real alignment
       pickle's schema.
    5. Run LOBO XGBoost with HPO-best params on the null X, save per-fold
       metrics (R^2, spearman, RMSE, MAE), and append to a combined CSV.

    NC + X_null artifacts are target-agnostic (deterministic in
    ``nc_seed + s_idx + surrogate inputs + LR DB``); if
    ``nc_outputs/{s_idx}.h5`` and ``null_features/{s_idx}.npy`` both
    exist on disk they are reused and the expensive NC call is skipped.
    Pass ``--force-rebuild`` to override this cache. LOBO metrics are
    per-target and always run fresh (written to
    ``null_metrics/{target}/{s_idx}.csv`` and
    ``combined_null_metrics_{target}.csv``).

--stage train_nulls
    Same as ``rebuild_features`` but skips the XGBoost training step.
    Intended for users who want to pre-compute pseudo-AnnData + NC H5 +
    null X_null separately from training -- typically followed by two
    ``--stage lobo_only`` invocations (one per target).

--stage lobo_only
    Train LOBO XGBoost on existing ``null_features/{s_idx}.npy`` files
    for the specified ``--target``. Fast path for the second target once
    the NC / X_null cache has been populated; avoids re-running
    NeuronChat. Writes the same per-target metrics outputs as
    ``rebuild_features``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, cast

import anndata as ad
import numpy as np
import pandas as pd

from clrc.core.io import (
    find_repo_root,
    load_pickle,
    load_yaml_config,
)
from clrc.core.logging import setup_logging
from clrc.spatial.atlas import (
    AbcAllenMapping,
    aggregate_abc_map_to_unique,
    build_unique_distance_matrix,
    build_unique_spatial_nodes,
    compute_unique_centroids,
    load_abc_allen_mapping,
    load_allen_centroids,
    load_allen_voxel_counts,
    redistribute_surrogate_to_abc,
)
from clrc.spatial.nulls import (
    _validate_distance_matrix as _brainsmash_validate_distance_matrix,
    generate_brainsmash_surrogates,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGION_PREFIX = "Human "  # ABC.h5ad region labels are prefixed like "Human A13"


def _resolve_out_root(cfg: dict) -> Path:
    out_root = Path(cfg["output"]["base_dir"])
    if not out_root.is_absolute():
        out_root = find_repo_root() / out_root
    return out_root


def _load_abc_regions_cci(cfg: dict) -> List[str]:
    """Return the canonical 101-length ``ABC_regions_cci`` list from the
    alignment pickle, which every other ABC-space artefact aligns to."""
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    if not align_pkl.is_absolute():
        align_pkl = find_repo_root() / align_pkl
    align = load_pickle(align_pkl)
    abc_regions_cci = list(align["ABC_regions_cci"])
    if len(abc_regions_cci) != 101:
        logger.warning(
            "ABC_regions_cci has %d entries (expected 101); continuing.",
            len(abc_regions_cci),
        )
    return abc_regions_cci


def _validate_distance_matrix(D: np.ndarray, n_expected: int) -> None:
    """Stop-the-world validation of the computed distance matrix.

    Checks: square (n_expected, n_expected), symmetric, zero-diagonal,
    non-negative, no zero off-diagonals, no non-finite entries. Mirrors
    the brainSMASH ``check_distmat`` contract and catches degeneracies
    (two unique groups sharing a centroid, which would contradict the
    construction by unique constituent set).
    """
    if D.shape != (n_expected, n_expected):
        raise ValueError(
            f"distance matrix shape {D.shape} != expected "
            f"({n_expected}, {n_expected})."
        )
    # Defer the finiteness/symmetry/zero-diagonal/non-negativity contract
    # checks to the brainSMASH-aligned validator in clrc.spatial.nulls so
    # we have one source of truth.
    _brainsmash_validate_distance_matrix(D, n_regions=n_expected)
    # Plus the additional zero-off-diagonal check (brainSMASH does not
    # require positivity; we do, because zero off-diagonals collapse the
    # variogram).
    mask = ~np.eye(n_expected, dtype=bool)
    off_diag_min = float(D[mask].min())
    if off_diag_min <= 0.0:
        pairs = np.argwhere((D == 0.0) & mask)
        raise ValueError(
            f"distance matrix has {pairs.shape[0]} zero off-diagonal "
            f"entries (examples: {pairs[:5].tolist()}). Two unique "
            "groups share a centroid -- this contradicts construction "
            "by unique Allen-constituent set."
        )


# ---------------------------------------------------------------------------
# Stage 1: build distances
# ---------------------------------------------------------------------------

def _load_and_align_mapping(cfg: dict, abc_regions_cci: List[str]) -> Tuple[AbcAllenMapping, np.ndarray]:
    """Load the ABC<->Allen mapping, apply region_aliases, restrict to
    ``abc_regions_cci`` (in the canonical CCI ordering), and return the
    aligned mapping plus the Allen voxel-count vector.

    The returned ``mapping['abc_regions']`` is exactly ``abc_regions_cci``
    (so unique-group ordering is deterministic w.r.t. the CCI region list).
    """
    abc_data = cfg["abc_region_data"]
    abc_allen_csv = Path(abc_data["abc_allen_csv"])
    allen_labels_txt = Path(abc_data["allen_labels_txt"])
    voxel_counts_csv = Path(abc_data["voxel_counts_csv"])

    if not abc_allen_csv.is_absolute():
        abc_allen_csv = find_repo_root() / abc_allen_csv
    if not allen_labels_txt.is_absolute():
        allen_labels_txt = find_repo_root() / allen_labels_txt
    if not voxel_counts_csv.is_absolute():
        voxel_counts_csv = find_repo_root() / voxel_counts_csv

    logger.info(
        "Loading ABC<->Allen mapping from %s (+ %s)",
        abc_allen_csv, allen_labels_txt,
    )
    mapping = load_abc_allen_mapping(abc_allen_csv, allen_labels_txt)
    allen_labels = mapping["allen_labels"]
    voxel_counts = load_allen_voxel_counts(voxel_counts_csv, allen_labels)

    # The ABC<->Allen CSV uses the "struct" ABC names (e.g. A24), while
    # the alignment pickle's ABC_regions_cci uses the aliased names (e.g.
    # ACC). Apply cfg.region_aliases to remap mapping keys into CCI
    # namespace.
    region_aliases: Dict[str, str] = dict(cfg.get("region_aliases", {}))
    if region_aliases:
        renamed_abc_to_allen_idx = {
            region_aliases.get(k, k): v
            for k, v in mapping["abc_to_allen_idx"].items()
        }
        if len(renamed_abc_to_allen_idx) != len(mapping["abc_to_allen_idx"]):
            raise ValueError(
                f"region_aliases {region_aliases} collapses two distinct "
                "ABC regions in the CSV mapping to the same name; cannot "
                "disambiguate."
            )
        mapping = cast(AbcAllenMapping, dict(mapping))
        mapping["abc_to_allen_idx"] = renamed_abc_to_allen_idx
        mapping["abc_regions"] = [
            region_aliases.get(r, r) for r in mapping["abc_regions"]
        ]
        logger.info(
            "Applied region_aliases to ABC<->Allen mapping: %s",
            region_aliases,
        )

    missing = [r for r in abc_regions_cci if r not in mapping["abc_to_allen_idx"]]
    if missing:
        raise KeyError(
            f"ABC_regions_cci contains {len(missing)} region(s) not found "
            f"in ABC<->Allen mapping: {missing[:10]}..."
        )

    # Restrict to the canonical CCI ABC ordering. This makes the unique-group
    # iteration order deterministic with respect to abc_regions_cci.
    aligned_mapping = cast(AbcAllenMapping, dict(mapping))
    aligned_mapping["abc_regions"] = list(abc_regions_cci)
    aligned_mapping["abc_to_allen_idx"] = {
        r: list(mapping["abc_to_allen_idx"][r]) for r in abc_regions_cci
    }
    return aligned_mapping, voxel_counts


def run_build_distances(cfg: dict, stage_out_dir: Path) -> Path:
    """Compute the 86-node unique-group distance matrix and write to disk.

    Returns the path to the saved ``unique_distance_matrix_86.npy``.
    """
    logger.info(
        "[build_distances] Conservative null: 141-region Allen atlas cannot "
        "uniquely resolve all 101 ABC regions; collapsing to 86 unique "
        "spatial nodes (one per unique Allen-constituent set), surrogating "
        "at 86, then redistributing each surrogate to its member ABC regions."
    )
    sn = cfg["spatial_null"]
    atlas_nii = Path(sn["atlas_nii"])
    if not atlas_nii.is_absolute():
        atlas_nii = find_repo_root() / atlas_nii

    abc_regions_cci = _load_abc_regions_cci(cfg)

    logger.info("Loading Allen centroids from %s", atlas_nii)
    allen_centroids = load_allen_centroids(atlas_nii)

    mapping, voxel_counts = _load_and_align_mapping(cfg, abc_regions_cci)

    unique_nodes_df = build_unique_spatial_nodes(mapping)
    n_unique = len(unique_nodes_df)
    n_singletons = int((unique_nodes_df["n_abc_regions"] == 1).sum())
    n_collapsed_groups = int((unique_nodes_df["n_abc_regions"] > 1).sum())
    n_collapsed_abcs = int(
        unique_nodes_df.loc[unique_nodes_df["n_abc_regions"] > 1, "n_abc_regions"].sum()
    )
    logger.info(
        "[build_distances] %d ABC regions -> %d unique spatial groups "
        "(%d singletons, %d multi-ABC groups containing %d ABC regions)",
        len(abc_regions_cci), n_unique, n_singletons, n_collapsed_groups, n_collapsed_abcs,
    )

    unique_centroids = compute_unique_centroids(
        unique_nodes_df=unique_nodes_df,
        allen_centroids=allen_centroids,
        voxel_counts=voxel_counts,
    )
    D = build_unique_distance_matrix(unique_centroids)
    _validate_distance_matrix(D, n_expected=n_unique)

    # Diagnostic stats for the validation log line
    mask = ~np.eye(n_unique, dtype=bool)
    logger.info(
        "[build_distances] Distance matrix validated: shape=%s, "
        "off-diag min=%.4f mm, max=%.4f mm",
        D.shape, float(D[mask].min()), float(D[mask].max()),
    )

    stage_out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = stage_out_dir / "unique_distance_matrix_86.npy"
    np.save(out_npy, D)
    logger.info("Saved unique-group distance matrix -> %s (shape=%s)", out_npy, D.shape)

    # Sidecar 1: unique_centroids.csv with abc_regions_in_group
    centroid_csv_rows = []
    for row in unique_centroids.itertuples(index=False):
        centroid_csv_rows.append(
            {
                "group_id": int(row.group_id),
                "x_mm": float(row.x_mm),
                "y_mm": float(row.y_mm),
                "z_mm": float(row.z_mm),
                "n_abc_regions": int(row.n_abc_regions),
                "abc_regions_in_group": "+".join(row.abc_regions),
            }
        )
    pd.DataFrame(centroid_csv_rows).to_csv(
        stage_out_dir / "unique_centroids.csv", index=False
    )
    logger.info(
        "Saved unique centroids CSV -> %s",
        stage_out_dir / "unique_centroids.csv",
    )

    # Sidecar 2: abc_to_group_id.csv (101-row mapping)
    abc_to_group_rows = []
    for row in unique_nodes_df.itertuples(index=False):
        gid = int(row.group_id)
        for abc in row.abc_regions:
            abc_to_group_rows.append({"abc_region": abc, "group_id": gid})
    abc_to_group_df = pd.DataFrame(abc_to_group_rows).sort_values("abc_region").reset_index(drop=True)
    abc_to_group_df.to_csv(
        stage_out_dir / "abc_to_group_id.csv", index=False
    )
    logger.info(
        "Saved ABC<->group mapping CSV (%d rows) -> %s",
        len(abc_to_group_df), stage_out_dir / "abc_to_group_id.csv",
    )

    # Sidecar 3: ABC region ordering (canonical CCI order) for provenance.
    (stage_out_dir / "abc_regions_cci.txt").write_text(
        "\n".join(abc_regions_cci) + "\n", encoding="utf-8"
    )

    return out_npy


# ---------------------------------------------------------------------------
# Stage 2: generate surrogates
# ---------------------------------------------------------------------------

def _load_abc_anndata(cfg: dict) -> ad.AnnData:
    """Load the ABC.h5ad in backed mode for streaming aggregation."""
    repo_root = find_repo_root()
    h5ad_path = repo_root / "data/UMAP_ANNDATA/ABC.h5ad"
    if not h5ad_path.is_file():
        raise FileNotFoundError(
            f"ABC.h5ad not found at {h5ad_path}. This driver expects the "
            "same AnnData used by src/pipeline/connectivity_prediction/coexpression_baseline.py."
        )
    logger.info("Loading AnnData (backed) from %s", h5ad_path)
    return ad.read_h5ad(h5ad_path, backed="r")


def _strip_region_prefix(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.replace(f"^{_REGION_PREFIX}", "", regex=True)
    )


def build_celltype_abundance_matrix(
    adata: ad.AnnData,
    abc_regions: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Count cells per (region, supercluster_name) restricted to ABC.

    Returns
    -------
    matrix : (n_regions, n_celltypes) ndarray of float
        Cell counts per region per supercluster, ordered by ``abc_regions``
        on axis 0 and by alphabetical supercluster name on axis 1.
    celltypes : list[str]
        Supercluster names in column order.
    """
    obs_region = _strip_region_prefix(adata.obs["region_of_interest_label"])
    obs_ct = adata.obs["supercluster_name"].astype(str)

    # Build region x celltype count table.
    counts = (
        pd.DataFrame({"region": obs_region.to_numpy(), "celltype": obs_ct.to_numpy()})
        .groupby(["region", "celltype"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    celltypes = sorted(counts.columns.tolist())
    counts = counts[celltypes]

    missing = [r for r in abc_regions if r not in counts.index]
    if missing:
        raise KeyError(
            f"ABC regions missing from AnnData supercluster counts: "
            f"{missing[:10]}... total missing={len(missing)}."
        )

    matrix = counts.loc[list(abc_regions), :].to_numpy(dtype=np.float64)
    logger.info(
        "Cell-type abundance matrix: shape=%s, celltypes=%d",
        matrix.shape, len(celltypes),
    )
    return matrix, celltypes


def build_region_expression_matrix(
    adata: ad.AnnData,
    abc_regions: Sequence[str],
    *,
    log1p: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """log1p(mean raw counts) per (region, gene) restricted to ABC regions.

    Intentionally parallels ``build_abc_region_expression`` from
    ``src/pipeline/connectivity_prediction/coexpression_baseline.py`` but returns a dense numpy
    matrix + gene list so it drops straight into
    ``surrogate_expression_maps``.
    """
    obs_region = _strip_region_prefix(adata.obs["region_of_interest_label"])
    gene_symbols = adata.var["gene_symbol"].astype(str).tolist()

    unique_regions = sorted(obs_region.unique().tolist())
    missing = [r for r in abc_regions if r not in unique_regions]
    if missing:
        raise KeyError(
            f"ABC.h5ad missing regions required by alignment pickle: "
            f"{missing[:10]}... total missing={len(missing)}."
        )

    n_genes = adata.shape[1]
    region_idx = {r: np.flatnonzero((obs_region == r).to_numpy())
                  for r in abc_regions}

    expr = np.zeros((len(abc_regions), n_genes), dtype=np.float64)
    for i, r in enumerate(abc_regions):
        rows = region_idx[r]
        if rows.size == 0:
            raise ValueError(f"Region {r!r} has zero cells.")
        X_r = adata.X[rows, :]
        if hasattr(X_r, "toarray"):
            X_r = X_r.toarray()
        else:
            X_r = np.asarray(X_r)
        expr[i, :] = X_r.mean(axis=0)
        logger.debug(
            "region %s (%d/%d): %d cells aggregated",
            r, i + 1, len(abc_regions), rows.size,
        )

    if log1p:
        expr = np.log1p(expr)
    logger.info(
        "Region-expression matrix: shape=%s (log1p=%s)",
        expr.shape, log1p,
    )
    return expr, gene_symbols


def _surrogate_abc_matrix_via_unique(
    abc_matrix: np.ndarray,
    abc_regions: List[str],
    unique_nodes_df: pd.DataFrame,
    voxel_counts: np.ndarray,
    abc_to_allen_idx: Mapping[str, Sequence[int]],
    D_unique: np.ndarray,
    n_surrogates: int,
    seed: int,
    n_jobs: int,
    *,
    what: str,
    column_subset: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate -> brainSMASH (on 86 nodes) -> redistribute, per column.

    Parameters
    ----------
    abc_matrix : (n_abc, n_cols) array
        Real per-region per-column map (abundance or expression).
    abc_regions
        Length-``n_abc`` ordering matching rows of ``abc_matrix``.
    unique_nodes_df
        From :func:`build_unique_spatial_nodes`.
    voxel_counts
        Allen voxel-count vector aligned to mapping's ``allen_labels``.
    abc_to_allen_idx
        Mapping from ABC region name -> list of Allen indices.
    D_unique : (G, G)
        Validated 86x86 unique-group distance matrix.
    n_surrogates : int
    seed : int
        Per-column seeds are ``seed + col_index`` for reproducibility and
        independence across columns.
    n_jobs : int
        Forwarded to brainSMASH per-column.
    what : str
        Tag used in log lines ("celltype_abundance" / "expression").
    column_subset
        If provided, only surrogate this subset of column indices (used
        for dry-runs to limit the work). Returns full-width arrays where
        non-subset columns are zero. Pass ``None`` to surrogate all
        columns.

    Returns
    -------
    surrogates_unique : (n_surrogates, G, n_cols)
    surrogates_abc    : (n_surrogates, n_abc, n_cols)
    """
    if abc_matrix.ndim != 2:
        raise ValueError(
            f"{what} matrix must be 2D; got shape {abc_matrix.shape}."
        )
    n_abc, n_cols = abc_matrix.shape
    if n_abc != len(abc_regions):
        raise ValueError(
            f"{what} matrix has {n_abc} rows but abc_regions has "
            f"{len(abc_regions)} entries."
        )
    G = D_unique.shape[0]
    if G != len(unique_nodes_df):
        raise ValueError(
            f"D_unique has {G} rows but unique_nodes_df has "
            f"{len(unique_nodes_df)} groups."
        )

    surrogates_unique = np.zeros((int(n_surrogates), G, n_cols), dtype=np.float64)
    surrogates_abc = np.zeros((int(n_surrogates), n_abc, n_cols), dtype=np.float64)

    if column_subset is None:
        col_iter: Sequence[int] = range(n_cols)
    else:
        col_iter = list(column_subset)

    t0 = time.perf_counter()
    for j in col_iter:
        unique_map = aggregate_abc_map_to_unique(
            abc_values=abc_matrix[:, j],
            unique_nodes_df=unique_nodes_df,
            voxel_counts=voxel_counts,
            abc_to_allen_idx=abc_to_allen_idx,
            abc_regions_ordering=abc_regions,
        )
        col_seed = int(seed) + int(j)
        surr_u = generate_brainsmash_surrogates(
            map_values=unique_map,
            distance_matrix=D_unique,
            n_surrogates=int(n_surrogates),
            seed=col_seed,
            n_jobs=int(n_jobs),
        )  # (n_surrogates, G)
        surrogates_unique[:, :, j] = surr_u
        surrogates_abc[:, :, j] = redistribute_surrogate_to_abc(
            unique_surrogate=surr_u,
            unique_nodes_df=unique_nodes_df,
            abc_regions_ordering=abc_regions,
        )

    elapsed = time.perf_counter() - t0
    n_done = len(list(col_iter))
    logger.info(
        "_surrogate_abc_matrix_via_unique[%s]: %d/%d columns surrogated, "
        "elapsed=%.2fs (%.3fs/map)",
        what, n_done, n_cols, elapsed, elapsed / max(n_done, 1),
    )
    return surrogates_unique, surrogates_abc


def run_generate_surrogates(
    cfg: dict,
    stage_out_dir: Path,
    distance_matrix_path: Path,
    *,
    override_n_surrogates: int | None = None,
    seed: int = 0,
    n_jobs: int = 1,
    dry_run_n_maps: int | None = None,
) -> Tuple[Path, Path]:
    """Load abundance + expression maps and generate brainSMASH surrogates.

    The pipeline aggregates each 101-length ABC map to an 86-length
    unique-group map (voxel-weighted), surrogates on the 86x86 distance
    matrix, then redistributes each surrogate to a 101-length ABC vector.
    Within-group ABC values are identical in surrogates (conservative
    null; see module docstring).

    Returns (surrogates_celltype_npz, surrogates_gene_npz).
    """
    logger.info(
        "[generate_surrogates] Conservative null: surrogating at the 86-node "
        "unique-group level (one node per unique Allen-constituent set), then "
        "redistributing each surrogate value to all member ABC regions. "
        "Within-group surrogate variance is zero by construction."
    )
    sn = cfg["spatial_null"]
    n_surrogates = int(override_n_surrogates or sn["n_surrogates"])

    if not distance_matrix_path.is_file():
        raise FileNotFoundError(
            f"Distance matrix not found at {distance_matrix_path}. "
            "Run --stage build_distances first."
        )
    D_unique = np.load(distance_matrix_path)
    abc_regions = _load_abc_regions_cci(cfg)

    # Rebuild the unique-group structure from the same mapping; this lets
    # us aggregate/redistribute without re-walking the NIfTI.
    mapping, voxel_counts = _load_and_align_mapping(cfg, abc_regions)
    unique_nodes_df = build_unique_spatial_nodes(mapping)
    _validate_distance_matrix(D_unique, n_expected=len(unique_nodes_df))
    logger.info(
        "Loaded unique-group distance matrix shape=%s from %s",
        D_unique.shape, distance_matrix_path,
    )

    adata = _load_abc_anndata(cfg)

    # -- Cell-type abundance --
    abundance, celltypes = build_celltype_abundance_matrix(adata, abc_regions)

    n_ct_cols = abundance.shape[1]
    if dry_run_n_maps is not None:
        ct_subset = list(range(min(dry_run_n_maps, n_ct_cols)))
        logger.info(
            "Dry-run: surrogating only the first %d/%d cell-type columns",
            len(ct_subset), n_ct_cols,
        )
    else:
        ct_subset = None

    logger.info(
        "Generating %d surrogates for %d cell-type abundance maps "
        "(86 unique groups -> %d ABC regions, n_jobs=%d)",
        n_surrogates, n_ct_cols, len(abc_regions), n_jobs,
    )
    surr_ct_unique, surr_ct_abc = _surrogate_abc_matrix_via_unique(
        abc_matrix=abundance,
        abc_regions=abc_regions,
        unique_nodes_df=unique_nodes_df,
        voxel_counts=voxel_counts,
        abc_to_allen_idx=mapping["abc_to_allen_idx"],
        D_unique=D_unique,
        n_surrogates=n_surrogates,
        seed=seed,
        n_jobs=n_jobs,
        what="celltype_abundance",
        column_subset=ct_subset,
    )

    ct_npz = stage_out_dir / "surrogates_celltype.npz"
    np.savez(
        ct_npz,
        surrogates_unique=surr_ct_unique,
        surrogates_abc=surr_ct_abc,
        real_abundance=abundance,
        regions=np.array(abc_regions, dtype=object),
        celltypes=np.array(celltypes, dtype=object),
        group_ids=unique_nodes_df["group_id"].to_numpy(),
        seed=np.array(seed),
        n_surrogates=np.array(n_surrogates),
        column_subset=np.array(ct_subset if ct_subset is not None else list(range(n_ct_cols))),
    )
    logger.info("Saved cell-type surrogates -> %s (unique=%s, abc=%s)",
                ct_npz, surr_ct_unique.shape, surr_ct_abc.shape)

    # -- Gene expression --
    expr, gene_symbols = build_region_expression_matrix(
        adata, abc_regions, log1p=True
    )
    n_g_cols = expr.shape[1]
    if dry_run_n_maps is not None:
        g_subset = list(range(min(dry_run_n_maps, n_g_cols)))
        logger.info(
            "Dry-run: surrogating only the first %d/%d gene columns",
            len(g_subset), n_g_cols,
        )
    else:
        g_subset = None

    logger.info(
        "Generating %d surrogates for %d gene expression maps "
        "(86 unique groups -> %d ABC regions, n_jobs=%d)",
        n_surrogates, n_g_cols, len(abc_regions), n_jobs,
    )
    surr_ge_unique, surr_ge_abc = _surrogate_abc_matrix_via_unique(
        abc_matrix=expr,
        abc_regions=abc_regions,
        unique_nodes_df=unique_nodes_df,
        voxel_counts=voxel_counts,
        abc_to_allen_idx=mapping["abc_to_allen_idx"],
        D_unique=D_unique,
        n_surrogates=n_surrogates,
        seed=seed + 100_000,  # offset so gene seeds don't collide with CT seeds
        n_jobs=n_jobs,
        what="expression",
        column_subset=g_subset,
    )

    ge_npz = stage_out_dir / "surrogates_gene.npz"
    np.savez(
        ge_npz,
        surrogates_unique=surr_ge_unique,
        surrogates_abc=surr_ge_abc,
        real_expression=expr,
        regions=np.array(abc_regions, dtype=object),
        genes=np.array(gene_symbols, dtype=object),
        group_ids=unique_nodes_df["group_id"].to_numpy(),
        seed=np.array(seed + 100_000),
        n_surrogates=np.array(n_surrogates),
        column_subset=np.array(g_subset if g_subset is not None else list(range(n_g_cols))),
    )
    logger.info("Saved gene surrogates -> %s (unique=%s, abc=%s)",
                ge_npz, surr_ge_unique.shape, surr_ge_abc.shape)

    return ct_npz, ge_npz


# ---------------------------------------------------------------------------
# Stage 3: rebuild CCI features per surrogate + train XGBoost
#
# Per surrogate:
#   1. Load the 101-ABC abundance + expression surrogate draws.
#   2. Build a pseudo-AnnData whose per-(region, celltype) aggregate
#      expression equals surrogate_gene_expr[region] * surrogate_abundance[region, ct].
#   3. Run NeuronChat on the pseudo-AnnData (M=50, device=cuda) and save H5.
#   4. Stream H5 -> (n_edges, n_features) null X matrix matching the real
#      alignment-pickle schema exactly (same column order, same features).
#   5. Train XGBoost with HPO-best params (loaded from cfg.xgboost.<target>.params_json)
#      for each LOBO fold and record per-fold R^2, spearman_rho, RMSE, MAE.
#   6. Append per-surrogate metrics to a combined null-distribution CSV.
# ---------------------------------------------------------------------------


def _load_pair_counts(
    adata: ad.AnnData,
) -> dict:
    """Thin wrapper around clrc.spatial.nc_rebuild.build_real_pair_counts."""
    from clrc.spatial.nc_rebuild import build_real_pair_counts

    return build_real_pair_counts(adata)


def _load_hpo_best_params(cfg: dict, target: str) -> Tuple[dict, str]:
    """Load the HPO-best XGBoost params + loss tag for a target (sc|fc)."""
    import json

    tcfg = cfg["xgboost"][target]
    params_path = Path(tcfg["params_json"])
    if not params_path.is_absolute():
        params_path = find_repo_root() / params_path
    if not params_path.is_file():
        raise FileNotFoundError(
            f"HPO-best params JSON not found at {params_path}. "
            "Run `src/pipeline/connectivity_prediction/hpo.py` first or point "
            f"cfg.xgboost.{target}.params_json to a valid file."
        )
    with params_path.open("r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return blob["best_params_xgb"], blob.get("loss", cfg["xgboost"]["loss"])


def _train_null_lobo(
    X_null: np.ndarray,
    data,
    cfg: dict,
    target: str,
    *,
    regions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run LOBO XGBoost on a null feature matrix and return per-fold metrics.

    ``data`` is an :class:`clrc.core.types.AlignmentData` loaded from the
    real alignment pickle: we reuse its edge_table, SC_voxel/SC_naive,
    metric_names, distance_vec as the target side. Only ``X`` is
    replaced by ``X_null`` -- the null is about spatial-AC in inputs,
    the targets (SC/FC) are the real pipeline's targets.
    """
    from clrc.core.io import stable_hash_int
    from clrc.core.metrics import mae as mae_fn
    from clrc.core.metrics import rmse as rmse_fn
    from clrc.prediction.lobo import (
        infer_regions,
        iter_lobo_folds,
        precompute_fold_masks,
    )
    from clrc.prediction.xgboost import train_predict_xgb
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    tcfg = cfg["xgboost"][target]
    best_params_xgb, _loss = _load_hpo_best_params(cfg, target)

    metric_names = list(data.metric_names)
    j = metric_names.index(tcfg["metric"])
    SC = data.SC_voxel if tcfg["version"] == "voxel" else data.SC_naive
    y_all = SC[:, j].astype(float)

    regions_all = list(regions) if regions is not None else infer_regions(data.edge_table)
    fold_masks = precompute_fold_masks(data.edge_table, regions_all)

    rows: List[Dict[str, object]] = []
    for fold in iter_lobo_folds(
        X_null, data.edge_table, y_all, fold_masks,
        eps=0.0,
        y_transform=cfg["xgboost"]["y_transform"],
        data_type=tcfg["data_type"],
        include_edge_tables=False,
        regions=regions_all,
    ):
        (holdout_region, X_train, y_train_t, X_test, y_test_t,
         _y_train_raw, _y_test_raw, _ecdf, _et_test, _test_idx) = fold

        split_seed = int(stable_hash_int(f"{cfg['xgboost']['seed']}_{holdout_region}"))
        y_pred, best_iter, _model_raw = train_predict_xgb(
            X_train, y_train_t, X_test,
            params=best_params_xgb,
            num_boost_round=cfg["xgboost"]["max_boost_rounds"],
            split_seed=split_seed,
            booster_seed=cfg["xgboost"]["seed"],
            device=cfg["xgboost"]["device"],
            valid_fraction=cfg["xgboost"]["valid_fraction"],
            early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
        )
        rho, p = spearmanr(y_test_t, y_pred) if len(y_test_t) > 1 else (float("nan"), float("nan"))
        r2 = float(r2_score(y_test_t, y_pred)) if len(y_test_t) > 1 else float("nan")
        rows.append({
            "holdout_region": holdout_region,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "best_iteration": int(best_iter),
            "R2": r2,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "RMSE": rmse_fn(y_test_t, y_pred),
            "MAE": mae_fn(y_test_t, y_pred),
        })
    return pd.DataFrame(rows)


def _resolve_lr_db(cfg: dict) -> object:
    """Resolve the LR database argument for ``neuronchat.create_neuronchat``.

    Precedence:
    1. ``cfg.spatial_null.lr_db`` if set (path or 'mouse'/'human').
    2. ``cfg.data.lr_db`` if set.
    3. Bundled expanded 1092-LR DB at
       ``src/neuronchat/data/merged_interactionDB_human_1092LR.json`` if
       present (matches the real NC run which used this DB).
    4. Fall back to the built-in 'human' keyword (190 LRs).
    """
    sn = cfg.get("spatial_null", {})
    if "lr_db" in sn:
        v = sn["lr_db"]
        if isinstance(v, str) and v not in ("mouse", "human"):
            vp = Path(v)
            if not vp.is_absolute():
                vp = find_repo_root() / vp
            return vp
        return v
    data = cfg.get("data", {})
    if "lr_db" in data:
        v = data["lr_db"]
        if isinstance(v, str) and v not in ("mouse", "human"):
            vp = Path(v)
            if not vp.is_absolute():
                vp = find_repo_root() / vp
            return vp
        return v
    # Default: prefer the bundled merged 1092-LR DB used by the real run.
    default_path = (
        find_repo_root()
        / "src"
        / "neuronchat"
        / "data"
        / "merged_interactionDB_human_1092LR.json"
    )
    if default_path.is_file():
        return default_path
    return "human"


def run_rebuild_features(
    cfg: dict,
    stage_out_dir: Path,
    *,
    target: str,
    surrogate_indices: Sequence[int] | None = None,
    nc_M: int = 50,
    nc_device: str = "cuda",
    nc_n_jobs: int = 1,
    nc_seed: int = 42,
    train_nulls: bool = True,
    lobo_region_subset: Sequence[str] | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Per surrogate, build pseudo-AnnData -> run NC -> stream X_null -> LOBO.

    Parameters
    ----------
    cfg : dict
        Loaded YAML config.
    stage_out_dir : Path
        ``<output_base>/spatial_null/`` root. Inputs read from:
        ``surrogates_celltype.npz``, ``surrogates_gene.npz``. Outputs
        written under subdirs ``pseudo_adata/``, ``nc_outputs/``,
        ``null_features/``, ``null_metrics/``, and
        ``combined_null_metrics.csv``.
    target : str
        'sc' or 'fc' -- selects HPO-best params + SC/FC target column.
    surrogate_indices : sequence of int | None
        Which surrogate draws to process (0-indexed rows of the
        surrogate npz). ``None`` processes all.
    nc_M, nc_device, nc_n_jobs, nc_seed
        Forwarded to NeuronChat.
    train_nulls : bool, default True
        If False, only build pseudo-AnnData + run NC + stream X_null;
        skip XGBoost training. Useful to split compute into two tmux
        sessions.
    lobo_region_subset : sequence of str | None
        If set, restrict LOBO evaluation to this region list (e.g. 2
        regions for a fast dry-run end-to-end check). ``None`` runs
        all 101 regions.

    Returns
    -------
    combined_df : DataFrame
        All per-surrogate per-fold metric rows, with a leading
        ``surrogate_idx`` column. Also written to
        ``<stage_out_dir>/combined_null_metrics.csv``.
    """
    from clrc.core.io import load_alignment_data
    from clrc.spatial.nc_rebuild import (
        build_null_cci_features,
        reconstruct_pseudo_anndata,
        run_nc_on_pseudo,
    )

    ct_npz_path = stage_out_dir / "surrogates_celltype.npz"
    ge_npz_path = stage_out_dir / "surrogates_gene.npz"
    for p in (ct_npz_path, ge_npz_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Surrogate npz not found: {p}. Run --stage generate_surrogates first."
            )

    logger.info("Loading surrogate npz files...")
    ct_npz = np.load(ct_npz_path, allow_pickle=True)
    ge_npz = np.load(ge_npz_path, allow_pickle=True)

    surr_ct_abc: np.ndarray = ct_npz["surrogates_abc"]  # (n_surr, n_abc, n_ct)
    surr_ge_abc: np.ndarray = ge_npz["surrogates_abc"]  # (n_surr, n_abc, n_genes)
    abc_regions: List[str] = list(ct_npz["regions"].tolist())
    celltypes: List[str] = list(ct_npz["celltypes"].tolist())
    genes: List[str] = list(ge_npz["genes"].tolist())

    n_surr_ct = int(surr_ct_abc.shape[0])
    n_surr_ge = int(surr_ge_abc.shape[0])
    if n_surr_ct != n_surr_ge:
        raise ValueError(
            f"Surrogate-count mismatch: celltype={n_surr_ct} vs gene={n_surr_ge}."
        )
    n_surr = n_surr_ct

    if surrogate_indices is None:
        surrogate_indices = list(range(n_surr))
    else:
        surrogate_indices = list(surrogate_indices)
        bad = [i for i in surrogate_indices if i < 0 or i >= n_surr]
        if bad:
            raise ValueError(f"surrogate_indices {bad} out of range [0, {n_surr}).")

    # Load real AnnData (backed) for real-pair-counts -- required so the
    # pseudo NC H5 has the same group_names as the real NC H5.
    adata_real = _load_abc_anndata(cfg)
    pair_counts = _load_pair_counts(adata_real)
    logger.info(
        "Real AnnData: %d (region, celltype) pairs present (used as pseudo-AnnData skeleton).",
        len(pair_counts),
    )

    # Load real alignment pickle for X_null schema reference + SC/FC targets + edge_table.
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    if not align_pkl.is_absolute():
        align_pkl = find_repo_root() / align_pkl
    data_align = load_alignment_data(
        align_pkl,
        version=cfg["xgboost"][target]["version"],
        target_scale=cfg["xgboost"][target].get("target_scale", 1.0),
    )
    logger.info(
        "Real alignment pickle: X shape=%s (used only for feature_names schema + SC/FC targets).",
        data_align.X.shape,
    )

    db_arg = _resolve_lr_db(cfg)
    logger.info("LR DB argument for NC: %r", db_arg)

    # NC outputs, pseudo-AnnData, X_null are target-agnostic (same surrogate
    # + nc_seed -> byte-identical NC H5). Only LOBO metrics are per-target.
    pseudo_dir = stage_out_dir / "pseudo_adata"
    nc_dir = stage_out_dir / "nc_outputs"
    feat_dir = stage_out_dir / "null_features"
    metr_dir = stage_out_dir / "null_metrics" / target
    for d in (pseudo_dir, nc_dir, feat_dir, metr_dir):
        d.mkdir(parents=True, exist_ok=True)

    region_aliases: Dict[str, str] = dict(cfg.get("region_aliases", {"A24": "ACC"}))

    combined_rows: List[pd.DataFrame] = []
    for s_idx in surrogate_indices:
        t0 = time.perf_counter()
        logger.info("[rebuild_features] surrogate %d/%d starting...",
                    s_idx, max(surrogate_indices))

        nc_h5 = nc_dir / f"{s_idx}.h5"
        feat_path = feat_dir / f"{s_idx}.npy"

        # Cache-skip: NC output and X_null are target-agnostic (deterministic
        # in nc_seed + s_idx + surrogate inputs + LR DB). If both cached
        # artifacts exist, skip the expensive NC rebuild and load X_null
        # from disk. Use --force-rebuild to override.
        cache_hit = (
            not force_rebuild
            and nc_h5.is_file()
            and feat_path.is_file()
        )

        if cache_hit:
            X_null = np.load(feat_path)
            logger.info(
                "[rebuild_features] surrogate %d: NC + X_null cached (%s, %s); "
                "skipping NC rebuild. X_null shape=%s",
                s_idx, nc_h5.name, feat_path.name, X_null.shape,
            )
            if X_null.shape != data_align.X.shape:
                logger.error(
                    "[rebuild_features] surrogate %d: cached X_null shape %s "
                    "!= real %s. Re-run with --force-rebuild.",
                    s_idx, X_null.shape, data_align.X.shape,
                )
        else:
            surrogate_abundance = surr_ct_abc[s_idx]       # (n_abc, n_ct)
            surrogate_gene_expr = surr_ge_abc[s_idx]        # (n_abc, n_genes)

            # 1. Build pseudo-AnnData
            pseudo = reconstruct_pseudo_anndata(
                surrogate_celltype_abundance=surrogate_abundance,
                surrogate_gene_expression=surrogate_gene_expr,
                region_codes=abc_regions,
                celltype_codes=celltypes,
                gene_names=genes,
                real_pair_counts=pair_counts,
                cells_per_group=1,
                group_by_col="region_celltype",
            )
            pseudo_path = pseudo_dir / f"{s_idx}.h5ad"
            pseudo.write_h5ad(pseudo_path)
            logger.info("[rebuild_features] saved pseudo-AnnData -> %s (n_cells=%d, n_genes=%d)",
                        pseudo_path, pseudo.n_obs, pseudo.n_vars)

            # 2. Run NeuronChat
            run_nc_on_pseudo(
                pseudo_adata=pseudo,
                db=db_arg,
                out_h5=nc_h5,
                group_by="region_celltype",
                M=int(nc_M),
                device=nc_device,
                n_jobs=int(nc_n_jobs),
                seed=int(nc_seed) + int(s_idx),  # deterministic per-surrogate seed
                progress=False,
            )

            # 3. Stream H5 -> X_null matching real schema
            X_null, _fn, _meta = build_null_cci_features(
                nc_h5_path=nc_h5,
                alignment_pkl=align_pkl,
                region_aliases=region_aliases,
            )
            np.save(feat_path, X_null)
            logger.info(
                "[rebuild_features] X_null shape=%s (real=%s) -> %s",
                X_null.shape, data_align.X.shape, feat_path,
            )
            if X_null.shape != data_align.X.shape:
                logger.error(
                    "[rebuild_features] surrogate %d: X_null shape %s != real %s. "
                    "Subsequent LOBO training will fail.",
                    s_idx, X_null.shape, data_align.X.shape,
                )

        # 4. LOBO training with HPO-best params (optional)
        if train_nulls:
            metrics_df = _train_null_lobo(
                X_null=X_null,
                data=data_align,
                cfg=cfg,
                target=target,
                regions=lobo_region_subset,
            )
            metrics_df.insert(0, "surrogate_idx", s_idx)
            metrics_df.insert(1, "target", target)
            metrics_csv = metr_dir / f"{s_idx}.csv"
            metrics_df.to_csv(metrics_csv, index=False)
            combined_rows.append(metrics_df)
            logger.info(
                "[rebuild_features] surrogate %d LOBO complete: %d folds, "
                "mean R2=%.4f, mean spearman=%.4f, elapsed=%.1fs",
                s_idx, len(metrics_df),
                float(metrics_df["R2"].mean()),
                float(metrics_df["spearman_rho"].mean()),
                time.perf_counter() - t0,
            )
        else:
            logger.info(
                "[rebuild_features] surrogate %d: train_nulls=False, skipping LOBO.",
                s_idx,
            )

    if combined_rows:
        combined_df = pd.concat(combined_rows, axis=0, ignore_index=True)
        combined_csv = stage_out_dir / f"combined_null_metrics_{target}.csv"
        # Append rather than overwrite so multiple CLI invocations can
        # accumulate surrogate batches. If the file does not exist yet
        # we write a header; otherwise we append headerless.
        if combined_csv.is_file():
            prior = pd.read_csv(combined_csv)
            combined_df = pd.concat([prior, combined_df], axis=0, ignore_index=True)
        combined_df.to_csv(combined_csv, index=False)
        logger.info(
            "[rebuild_features] wrote combined null metrics -> %s (n_rows=%d)",
            combined_csv, len(combined_df),
        )
        return combined_df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Stage: LOBO-only (reuse cached null_features/{s_idx}.npy)
# ---------------------------------------------------------------------------


def run_lobo_only(
    cfg: dict,
    stage_out_dir: Path,
    *,
    target: str,
    surrogate_indices: Sequence[int] | None = None,
    lobo_region_subset: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Train LOBO XGBoost on cached null_features/{s_idx}.npy per target.

    Used when ``rebuild_features`` has already run for one target (or via a
    dedicated ``train_nulls`` pass) and the NC-derived X_null artifacts are
    cached on disk. This avoids re-running NeuronChat for the second target.

    Parameters
    ----------
    cfg, stage_out_dir, target, lobo_region_subset
        Same semantics as :func:`run_rebuild_features`.
    surrogate_indices
        Surrogate indices to train on. ``None`` = all surrogates with an
        existing ``null_features/{s_idx}.npy`` on disk.

    Returns
    -------
    combined_df : DataFrame
        Per-fold metrics for every processed surrogate, with ``surrogate_idx``
        and ``target`` columns. Also written to
        ``<stage_out_dir>/combined_null_metrics_{target}.csv``.
    """
    from clrc.core.io import load_alignment_data

    feat_dir = stage_out_dir / "null_features"
    if not feat_dir.is_dir():
        raise FileNotFoundError(
            f"No cached X_null features at {feat_dir}. Run "
            f"--stage rebuild_features or --stage train_nulls first."
        )

    # Discover available surrogate indices if not provided.
    if surrogate_indices is None:
        available = sorted(
            int(p.stem) for p in feat_dir.glob("*.npy") if p.stem.isdigit()
        )
        if not available:
            raise FileNotFoundError(
                f"No null_features/*.npy in {feat_dir}. Run a rebuild_features "
                f"or train_nulls pass first."
            )
        surrogate_indices = available
        logger.info(
            "[lobo_only] auto-discovered %d cached surrogate features.",
            len(surrogate_indices),
        )
    else:
        surrogate_indices = list(surrogate_indices)

    # Load real alignment data for SC/FC targets + edge_table + feature schema.
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    if not align_pkl.is_absolute():
        align_pkl = find_repo_root() / align_pkl
    data_align = load_alignment_data(
        align_pkl,
        version=cfg["xgboost"][target]["version"],
        target_scale=cfg["xgboost"][target].get("target_scale", 1.0),
    )

    metr_dir = stage_out_dir / "null_metrics" / target
    metr_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[pd.DataFrame] = []
    for s_idx in surrogate_indices:
        t0 = time.perf_counter()
        feat_path = feat_dir / f"{s_idx}.npy"
        if not feat_path.is_file():
            logger.warning(
                "[lobo_only] surrogate %d: %s missing, skipping.",
                s_idx, feat_path,
            )
            continue

        X_null = np.load(feat_path)
        if X_null.shape != data_align.X.shape:
            logger.error(
                "[lobo_only] surrogate %d: X_null shape %s != real %s. "
                "Skipping (re-run rebuild_features with --force-rebuild).",
                s_idx, X_null.shape, data_align.X.shape,
            )
            continue

        metrics_df = _train_null_lobo(
            X_null=X_null,
            data=data_align,
            cfg=cfg,
            target=target,
            regions=lobo_region_subset,
        )
        metrics_df.insert(0, "surrogate_idx", s_idx)
        metrics_df.insert(1, "target", target)
        metrics_csv = metr_dir / f"{s_idx}.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        combined_rows.append(metrics_df)
        logger.info(
            "[lobo_only] surrogate %d LOBO complete: %d folds, "
            "mean R2=%.4f, mean spearman=%.4f, elapsed=%.1fs",
            s_idx, len(metrics_df),
            float(metrics_df["R2"].mean()),
            float(metrics_df["spearman_rho"].mean()),
            time.perf_counter() - t0,
        )

    if combined_rows:
        combined_df = pd.concat(combined_rows, axis=0, ignore_index=True)
        combined_csv = stage_out_dir / f"combined_null_metrics_{target}.csv"
        if combined_csv.is_file():
            prior = pd.read_csv(combined_csv)
            combined_df = pd.concat([prior, combined_df], axis=0, ignore_index=True)
        combined_df.to_csv(combined_csv, index=False)
        logger.info(
            "[lobo_only] wrote combined null metrics -> %s (n_rows=%d)",
            combined_csv, len(combined_df),
        )
        return combined_df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config path (e.g. configs/abc_expanded_hpobest.yaml).")
    parser.add_argument("--target", choices=["sc", "fc"], required=True,
                        help="SC or FC model -- only affects output subdir naming.")
    parser.add_argument(
        "--stage",
        choices=[
            "build_distances",
            "generate_surrogates",
            "rebuild_features",
            "train_nulls",
            "lobo_only",
            "all",
        ],
        required=True,
        help=(
            "Pipeline stage. ``rebuild_features`` builds pseudo-AnnData, "
            "runs NC, streams X_null, AND trains XGBoost; it skips NC + "
            "X_null build for surrogates whose cached artifacts "
            "exist on disk (override with --force-rebuild). ``train_nulls`` "
            "builds pseudo-AnnData + NC + X_null but skips XGBoost "
            "training (use to pre-compute X_null once, then train per "
            "target via ``lobo_only``). ``lobo_only`` trains XGBoost on "
            "cached null_features/{s_idx}.npy -- fast path for the second "
            "target after the first target has populated the cache. "
            "``all`` runs build_distances + generate_surrogates only "
            "(node-level inputs, target-agnostic); rebuild_features must "
            "be invoked explicitly per --target."
        ),
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=None,
        help="Override cfg.spatial_null.n_surrogates (useful for dry-runs).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Override cfg.output.base_dir (useful for dry-runs to /tmp).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallelism passed to brainSMASH per-map.")
    parser.add_argument(
        "--dry-run-n-maps", type=int, default=None,
        help=(
            "If set, only surrogate the first N columns of each input matrix "
            "(cell-type abundance and gene expression). Useful for fast "
            "shape/sanity dry-runs to /tmp."
        ),
    )
    parser.add_argument(
        "--surrogate-indices", type=str, default=None,
        help=(
            "Comma-separated list of surrogate indices to process in "
            "--stage rebuild_features / train_nulls (e.g. '0,1,2'). "
            "Default: all surrogates in the npz."
        ),
    )
    parser.add_argument(
        "--nc-M", type=int, default=50,
        help="NeuronChat permutation iterations M (default 50).",
    )
    parser.add_argument(
        "--nc-device", type=str, default="cuda",
        help="NeuronChat device (default 'cuda'; use 'cpu' for tests).",
    )
    parser.add_argument(
        "--nc-n-jobs", type=int, default=1,
        help="NeuronChat joblib n_jobs (CPU backend only).",
    )
    parser.add_argument(
        "--nc-seed", type=int, default=42,
        help="NeuronChat base seed; per-surrogate seed = nc_seed + s_idx.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help=(
            "Ignore cached nc_outputs/{s_idx}.h5 and null_features/{s_idx}.npy; "
            "re-run NeuronChat and rebuild X_null even if cached artifacts "
            "exist on disk. Only affects --stage rebuild_features / train_nulls."
        ),
    )
    parser.add_argument(
        "--lobo-regions", type=str, default=None,
        help=(
            "Comma-separated region list to restrict LOBO to (dry-runs). "
            "Default: all regions from the alignment pickle edge_table."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = load_yaml_config(args.config)
    if args.output_root is not None:
        cfg["output"]["base_dir"] = str(args.output_root)

    out_root = _resolve_out_root(cfg)
    # Shared across SC and FC -- node-level inputs are target-agnostic.
    # We still accept --target to keep CLI shape consistent with other
    # pipeline drivers and for downstream XGBoost refits.
    stage_out_dir = out_root / "spatial_null"
    stage_out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging("spatial_null", output_dir=stage_out_dir)
    logger.info("Config: %s (target=%s, stage=%s)",
                args.config, args.target, args.stage)
    logger.info("Output dir: %s", stage_out_dir)

    if args.stage in ("build_distances", "all"):
        run_build_distances(cfg, stage_out_dir)

    if args.stage in ("generate_surrogates", "all"):
        dist_path = stage_out_dir / "unique_distance_matrix_86.npy"
        run_generate_surrogates(
            cfg,
            stage_out_dir,
            distance_matrix_path=dist_path,
            override_n_surrogates=args.n_surrogates,
            seed=args.seed,
            n_jobs=args.n_jobs,
            dry_run_n_maps=args.dry_run_n_maps,
        )

    surr_idx = None
    if args.surrogate_indices:
        surr_idx = [int(s) for s in args.surrogate_indices.split(",") if s.strip()]
    lobo_regions = None
    if args.lobo_regions:
        lobo_regions = [s.strip() for s in args.lobo_regions.split(",") if s.strip()]

    if args.stage in ("rebuild_features", "train_nulls", "all"):
        # Stage semantics:
        #   rebuild_features  -> pseudo AnnData + NC + X_null + LOBO.
        #                        Cached nc_outputs/{s_idx}.h5 and
        #                        null_features/{s_idx}.npy are reused unless
        #                        --force-rebuild is passed; only LOBO runs
        #                        fresh per target.
        #   train_nulls       -> same but skips LOBO. Useful for a single
        #                        upfront NC pass shared across SC + FC;
        #                        follow up with --stage lobo_only per target.
        #   all               -> runs the full rebuild_features path (with LOBO)
        #                        after build_distances + generate_surrogates.
        run_rebuild_features(
            cfg,
            stage_out_dir,
            target=args.target,
            surrogate_indices=surr_idx,
            nc_M=args.nc_M,
            nc_device=args.nc_device,
            nc_n_jobs=args.nc_n_jobs,
            nc_seed=args.nc_seed,
            train_nulls=(args.stage in ("rebuild_features", "all")),
            lobo_region_subset=lobo_regions,
            force_rebuild=args.force_rebuild,
        )

    if args.stage == "lobo_only":
        # Fast path: train LOBO XGBoost on cached null_features/{s_idx}.npy.
        # Use when another --target has already populated the NC / X_null
        # cache and this target only needs the (cheap) XGBoost step.
        run_lobo_only(
            cfg,
            stage_out_dir,
            target=args.target,
            surrogate_indices=surr_idx,
            lobo_region_subset=lobo_regions,
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
