"""Rebuild CCI features for one spatial surrogate: NC run + feature streaming.

For each spatial surrogate produced by
:func:`clrc.spatial.nulls.surrogate_celltype_abundance_maps` /
:func:`clrc.spatial.nulls.surrogate_expression_maps`, build a
pseudo-AnnData, run NeuronChat, and stream the resulting H5 into a
(n_edges, n_features) null CCI feature matrix matching the main
pipeline's schema exactly.

Pseudo-AnnData construction
----------------------------
The real AnnData has millions of cells grouped by (region,
supercluster_name) into ~2133 non-empty pairs. NeuronChat's
``create_neuronchat`` + grouped expression aggregation only cares about
per-group statistics, so we do NOT replicate the full per-cell layout.
Instead, for each real (region, celltype) group we synthesize
``cells_per_group`` cells (default 1) with the same expression vector
``v``. NC's ``cal_expr_by_group`` (quantile-weighted mean, or plain mean
if ``mean_method='mean'``) over a constant-vector group returns that
same vector, so the per-group aggregate of the pseudo-AnnData equals
``v`` by construction.

The per-cell expression vector for group (region, ct) is

    v = surrogate_gene_expression[region, :] * surrogate_abundance[region, ct]

which encodes spatial AC of gene expression via the first factor and
spatial AC of cell-type abundance via the second. NC's downstream
``max`` normalization to [0, 1] is a global scalar divide so it does
not disturb relative structure across groups.

Group-membership restriction: we only create pseudo-cells for (region,
ct) groups present in the real AnnData. Empty real groups stay empty in
the pseudo-AnnData, so the pseudo NC H5 has the same group_names (and
therefore the same node structure) as the real NC H5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pseudo-AnnData construction
# ---------------------------------------------------------------------------

def reconstruct_pseudo_anndata(
    surrogate_celltype_abundance: np.ndarray,
    surrogate_gene_expression: np.ndarray,
    region_codes: Sequence[str],
    celltype_codes: Sequence[str],
    gene_names: Sequence[str],
    real_pair_counts: Optional[Dict[Tuple[str, str], int]] = None,
    *,
    cells_per_group: int = 1,
    group_by_col: str = "region_celltype",
) -> anndata.AnnData:
    """Build a pseudo-AnnData whose per-(region, celltype) aggregates
    equal the surrogate-encoded values.

    Parameters
    ----------
    surrogate_celltype_abundance : (n_regions, n_celltypes) array
        Surrogate cell-type abundance per ABC region, column-aligned to
        ``celltype_codes``. Values must be non-negative. Entries of zero
        are allowed (they produce all-zero pseudo-expression for that
        group, matching NC's handling of absent signaling).
    surrogate_gene_expression : (n_regions, n_genes) array
        Surrogate gene expression per ABC region, column-aligned to
        ``gene_names``.
    region_codes : sequence of str, length n_regions
        Row labels (e.g. ``ABC_regions_cci`` from the alignment pickle).
    celltype_codes : sequence of str, length n_celltypes
    gene_names : sequence of str, length n_genes
    real_pair_counts : dict {(region, celltype): int} or None
        Which (region, celltype) pairs exist in the real AnnData. Only
        pairs present here will get pseudo-cells; this preserves the
        real ``group_names`` structure of the NC H5 so the pseudo NC run
        outputs matrices aligned to the real 2133-node ordering. If
        ``None``, every (region, celltype) combination is populated
        (intended for tests with no real-data constraints).
    cells_per_group : int, default 1
        Number of pseudo-cells to synthesize per (region, celltype)
        group. The per-group aggregate is invariant to this count
        (constant-vector group), so 1 is sufficient and most efficient.
    group_by_col : str, default ``"region_celltype"``
        Name of the obs column used by NC's ``group_by`` argument.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_cells, n_genes)`` where ``n_cells = n_pairs *
        cells_per_group``. ``obs`` contains: ``region`` (ABC region
        string), ``supercluster_name`` (celltype string), and
        ``group_by_col`` (``"region::ct"`` label matching NC's
        ``group_names`` format). ``var_names`` equals ``gene_names``.

    Notes
    -----
    The output AnnData is deliberately tiny (2133 cells × ~285 genes for
    the ABC setup with ``cells_per_group=1``), because NC's grouped
    aggregation makes larger per-cell replication a waste. NC's
    ``create_neuronchat`` normalizes ``data_signaling`` by its global
    max; we set expression to a non-negative float64 matrix so that
    normalization is well-defined.
    """
    A = np.asarray(surrogate_celltype_abundance, dtype=np.float64)
    E = np.asarray(surrogate_gene_expression, dtype=np.float64)
    region_codes = list(region_codes)
    celltype_codes = list(celltype_codes)
    gene_names = list(gene_names)

    if A.ndim != 2 or A.shape != (len(region_codes), len(celltype_codes)):
        raise ValueError(
            f"surrogate_celltype_abundance shape {A.shape} != "
            f"({len(region_codes)}, {len(celltype_codes)})."
        )
    if E.ndim != 2 or E.shape != (len(region_codes), len(gene_names)):
        raise ValueError(
            f"surrogate_gene_expression shape {E.shape} != "
            f"({len(region_codes)}, {len(gene_names)})."
        )
    if cells_per_group < 1:
        raise ValueError(f"cells_per_group must be >= 1; got {cells_per_group}.")
    if (A < 0).any():
        # Small negative noise from brainSMASH resampling is expected;
        # clip to zero rather than propagating negatives into NC (which
        # would produce negative "expression" after the abundance scaling
        # — not biologically meaningful). The same clip is applied below
        # for the product; here we only warn if the magnitude is large.
        logger.warning(
            "surrogate_celltype_abundance has %d negative entries "
            "(min=%.3e); these will be clipped to 0 in the pseudo-AnnData.",
            int((A < 0).sum()), float(A.min()),
        )
    if (E < 0).any():
        logger.warning(
            "surrogate_gene_expression has %d negative entries "
            "(min=%.3e); these will be clipped to 0 in the pseudo-AnnData.",
            int((E < 0).sum()), float(E.min()),
        )

    # Build the list of (region_idx, ct_idx) pairs to populate.
    if real_pair_counts is None:
        pair_iter: List[Tuple[int, int]] = [
            (i, j)
            for i in range(len(region_codes))
            for j in range(len(celltype_codes))
        ]
    else:
        pair_iter = []
        for i, r in enumerate(region_codes):
            for j, ct in enumerate(celltype_codes):
                if (r, ct) in real_pair_counts:
                    pair_iter.append((i, j))
    if not pair_iter:
        raise ValueError(
            "No (region, celltype) pairs to populate. Check real_pair_counts."
        )

    n_pairs = len(pair_iter)
    n_cells = n_pairs * cells_per_group
    n_genes = len(gene_names)

    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    regions_col: List[str] = []
    ct_col: List[str] = []
    group_col: List[str] = []
    cell_names: List[str] = []

    for k, (i, j) in enumerate(pair_iter):
        region = region_codes[i]
        ct = celltype_codes[j]
        abundance = max(0.0, float(A[i, j]))
        expr_vec = np.maximum(E[i, :], 0.0) * abundance  # (n_genes,)
        start = k * cells_per_group
        stop = start + cells_per_group
        X[start:stop, :] = expr_vec
        for c in range(cells_per_group):
            cell_names.append(f"pseudo_{region}__{ct}__{c}")
            regions_col.append(region)
            ct_col.append(ct)
            group_col.append(f"{region}::{ct}")

    obs = pd.DataFrame(
        {
            "region": pd.Categorical(regions_col, categories=list(region_codes)),
            "supercluster_name": pd.Categorical(
                ct_col, categories=list(celltype_codes)
            ),
            group_by_col: pd.Categorical(group_col),
        },
        index=pd.Index(cell_names, name="cell_id"),
    )
    var = pd.DataFrame(
        {"gene_symbol": list(gene_names)},
        index=pd.Index(gene_names, name="gene"),
    )
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    logger.info(
        "reconstruct_pseudo_anndata: n_cells=%d, n_genes=%d, n_groups=%d, "
        "cells_per_group=%d",
        n_cells, n_genes, n_pairs, cells_per_group,
    )
    return adata


def build_real_pair_counts(
    real_anndata: anndata.AnnData,
    *,
    region_col: str = "region_of_interest_label",
    celltype_col: str = "supercluster_name",
    region_prefix: str = "Human ",
) -> Dict[Tuple[str, str], int]:
    """Count real (region, celltype) pair occurrences.

    Strips the ``region_prefix`` from region labels (matching the
    convention in ``src/pipeline/connectivity_prediction/spatial_null.py``) so the returned
    keys use the ABC-short-name form (e.g. ``A13``, ``ACC``).

    Returns
    -------
    dict mapping (region, celltype) -> int cell count. Only (region, ct)
    pairs that actually appear in ``real_anndata.obs`` are keys.
    """
    obs_region = (
        real_anndata.obs[region_col]
        .astype(str)
        .str.replace(f"^{region_prefix}", "", regex=True)
    )
    obs_ct = real_anndata.obs[celltype_col].astype(str)
    df = pd.DataFrame({"region": obs_region.to_numpy(), "ct": obs_ct.to_numpy()})
    counts = df.groupby(["region", "ct"], observed=True).size()
    return {(r, c): int(n) for (r, c), n in counts.items()}


# ---------------------------------------------------------------------------
# NeuronChat driver (thin wrapper)
# ---------------------------------------------------------------------------

def run_nc_on_pseudo(
    pseudo_adata: anndata.AnnData,
    db: object,
    out_h5: Path,
    *,
    group_by: str = "region_celltype",
    M: int = 50,
    device: str | list[str] = "cuda",
    n_jobs: int = 1,
    seed: int = 42,
    progress: bool = False,
) -> Path:
    """Run NeuronChat on the pseudo-AnnData and save to H5.

    This is a thin wrapper over :func:`neuronchat.create_neuronchat`
    followed by :func:`neuronchat.run_neuronchat` and
    :func:`neuronchat.save_h5` — no custom short-circuiting of the NC
    permutation test (that would silently simplify the method; we want
    the real NC procedure on surrogate inputs).

    Parameters
    ----------
    pseudo_adata : AnnData
        From :func:`reconstruct_pseudo_anndata`.
    db : str | Path | dict
        Argument forwarded to ``neuronchat.create_neuronchat``. For
        production runs this is typically a path to the merged 1092-LR
        JSON; for tests a tiny custom dict.
    out_h5 : Path
        Output H5 path.
    group_by : str
        obs column used as NC group label. Must exist in ``pseudo_adata.obs``.
    M : int
        NC permutation iterations. Matches the real pipeline default.
    device, n_jobs, seed, progress
        Forwarded to ``neuronchat.run_neuronchat``.

    Returns
    -------
    out_h5 : Path
        The same path as input (for convenience chaining).
    """
    from neuronchat import create_neuronchat, load_db, run_neuronchat, save_h5

    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    if group_by not in pseudo_adata.obs.columns:
        raise KeyError(
            f"group_by column {group_by!r} not found in pseudo_adata.obs "
            f"(columns: {list(pseudo_adata.obs.columns)})."
        )

    # ``create_neuronchat`` accepts either a str ("mouse"/"human") or a
    # pre-loaded dict; a Path object is not iterable. Resolve a Path ->
    # dict up-front so callers can pass a path directly.
    if isinstance(db, Path):
        db = load_db(db)

    obj = create_neuronchat(
        pseudo_adata,
        db=db,
        group_by=group_by,
        layer=None,
        keep_data=False,
    )
    logger.info(
        "run_nc_on_pseudo: created NeuronChat with %d interactions, %d groups",
        len(obj.lr), len(set(obj.data_signaling["cell_subclass"])),
    )
    obj = run_neuronchat(
        obj,
        M=M,
        device=device,
        n_jobs=n_jobs,
        seed=seed,
        progress=progress,
    )
    save_h5(obj, out_h5)
    logger.info("run_nc_on_pseudo: wrote NC H5 -> %s", out_h5)
    return out_h5


# ---------------------------------------------------------------------------
# Null CCI feature matrix from NC H5 (streaming)
# ---------------------------------------------------------------------------

def build_null_cci_features(
    nc_h5_path: Path,
    alignment_pkl: Path,
    *,
    nan_thresh: float | None = None,
    zero_thresh: float | None = None,
    region_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str], List[Dict[str, object]]]:
    """Stream an NC H5 into the same (n_edges, n_features) null feature
    matrix schema as the real pipeline, using the real alignment pickle
    as the schema source of truth.

    The real pipeline's feature schema is determined jointly by the real
    NC H5 group ordering and the pre-selection NaN/zero filters applied
    during alignment. We reuse
    :func:`clrc.features.streaming.build_features_streaming` exactly as
    in ``src/pipeline/shared/build_cci_features.py`` so the null schema is
    constructed by the same code path -- no reimplementation.

    Parameters
    ----------
    nc_h5_path : Path
        Pseudo NC H5 produced by :func:`run_nc_on_pseudo`.
    alignment_pkl : Path
        Real alignment pickle (e.g. ``aligned_ABC_expanded.pkl``). Used
        to recover the ABC region ordering and, optionally, the list of
        feature_names the real pipeline retained -- we subset the
        streamed null matrix to exactly those features so the null X has
        the same (n_edges, n_features) shape as the real X.
    nan_thresh, zero_thresh : float | None
        Pre-selection filter thresholds. If ``None``, we apply a
        permissive default (0.0, 1.0) so streaming produces ALL
        candidates, and then subset by the real feature_names list from
        the alignment pickle. This guarantees the null X matches the
        real X in column space exactly. Pass explicit values only if you
        want independent filtering (not recommended).
    region_aliases : dict | None
        Region rename map (e.g. ``{"A24": "ACC"}``) used by
        ``restrict_to_ABC``. Defaults to ``{"A24": "ACC"}``.

    Returns
    -------
    X_null : (n_edges, n_features) ndarray
        Aligned column-for-column with the real alignment pickle's
        ``X_kept_np``.
    feature_names : list[str]
        Identical to the real alignment pickle's ``feature_names_kept``.
    meta : list[dict]
        Identical to the real alignment pickle's ``meta_ABC_kept``.
    """
    from clrc.core.io import load_pickle
    from clrc.features.construction import parse_group_names, restrict_to_ABC
    from clrc.features.streaming import build_features_streaming

    nc_h5_path = Path(nc_h5_path)
    alignment_pkl = Path(alignment_pkl)
    if not nc_h5_path.is_file():
        raise FileNotFoundError(f"NC H5 not found: {nc_h5_path}")
    if not alignment_pkl.is_file():
        raise FileNotFoundError(f"Alignment pickle not found: {alignment_pkl}")

    align = load_pickle(alignment_pkl)
    real_feature_names: List[str] = list(align["feature_names_kept"])
    real_meta: List[Dict[str, object]] = list(align["meta_ABC_kept"])
    ABC_regions_struct: List[str] = list(align["ABC_regions_struct"])

    # Stream the pseudo H5 and build candidate feature columns. We apply
    # permissive filters by default so that EVERY (LR, ct_L, ct_R) feature
    # the real schema kept is present in the streamed output -- we then
    # subset to the real schema by name. This exactly matches the real
    # pipeline's column space.
    if nan_thresh is None:
        nan_thresh = 0.0  # accept any fraction of non-NaN
    if zero_thresh is None:
        zero_thresh = 1.0  # accept any fraction of zeros

    with h5py.File(nc_h5_path, "r") as f:
        interaction_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f.attrs["interaction_names"]
        ]
        group_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f.attrs["group_names"]
        ]
    _, _, node_lookup, regions_109, celltypes = parse_group_names(group_names)

    if region_aliases is None:
        region_aliases = {"A24": "ACC"}
    idx_abc_in_109, flat_idx_map, _ABC_regions_cci = restrict_to_ABC(
        regions_109, ABC_regions_struct, region_aliases=region_aliases
    )

    kept_vectors, kept_names, kept_meta = build_features_streaming(
        nc_h5_path=nc_h5_path,
        interaction_names=interaction_names,
        regions_109=regions_109,
        celltypes=celltypes,
        node_lookup=node_lookup,
        idx_abc_in_109=idx_abc_in_109,
        flat_idx_map=flat_idx_map,
        nan_thresh=nan_thresh,
        zero_thresh=zero_thresh,
    )

    # Align to real feature names: build a name -> vector lookup and
    # assemble the column matrix in the exact real ordering. Missing
    # features (e.g. an LR×ct×ct pair that the permissive null run
    # dropped) are filled with zeros -- this is the biologically faithful
    # null behaviour (no interaction evidence) and matches NC's zero-net
    # convention when an interaction has no ligand or receptor coverage.
    name_to_vec: Dict[str, np.ndarray] = {
        n: v for n, v in zip(kept_names, kept_vectors)
    }
    n_edges = kept_vectors[0].shape[0] if kept_vectors else idx_abc_in_109.size ** 2
    n_features = len(real_feature_names)
    X_null = np.zeros((n_edges, n_features), dtype=np.float64)
    missing = 0
    for j, name in enumerate(real_feature_names):
        vec = name_to_vec.get(name)
        if vec is None:
            missing += 1
            continue
        X_null[:, j] = vec
    if missing:
        logger.warning(
            "build_null_cci_features: %d/%d real features absent from the null "
            "NC H5 (zero-filled). This is expected if the surrogate inputs lead "
            "NC to report zero-net for an LR/celltype combination; it is *not* "
            "expected at any meaningful rate on a full surrogate run.",
            missing, n_features,
        )
    logger.info(
        "build_null_cci_features: X_null shape=%s, aligned to real schema "
        "(%d features from alignment pickle).",
        X_null.shape, n_features,
    )
    return X_null, real_feature_names, real_meta


__all__ = [
    "reconstruct_pseudo_anndata",
    "build_real_pair_counts",
    "run_nc_on_pseudo",
    "build_null_cci_features",
]
