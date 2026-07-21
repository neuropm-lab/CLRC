"""Three co-expression baselines against the same edge space as the main
CLRC analysis. All feed into the same XGBoost LOBO pipeline.

- :func:`build_region_collapsed_nc`. Per LR pair, collapse the NeuronChat
  communication matrix across (sender CT, receiver CT) pairs by arithmetic
  mean. Tests "does cell-type resolution add predictive signal?"

- :func:`build_lr_expression_product`. log1p-transformed bulk expression
  per region x per gene, arithmetic mean across the ligand genes and across
  the receptor genes of an LR pair, then outer product of the per-side means.
  Under arithmetic means, mean-then-product is algebraically identical to
  product-then-mean, so the aggregation order has no effect on the values.

- :func:`build_spatial_gene_coexpression`. Pearson correlation of per-region
  gene-expression profiles across a specified gene subset (default: all
  LR-related genes present on the expression panel). Single scalar per edge.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from clrc.features.construction import parse_group_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Region-collapsed NeuronChat
# ---------------------------------------------------------------------------

def build_region_collapsed_nc(
    nc_h5_path: Union[str, Path],
    edge_table: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Collapse each LR's NC matrix across cell-type pairs to one scalar per edge.

    For every LR pair and every (src_region, tgt_region) edge in
    ``edge_table``, returns the arithmetic mean of NC values across all
    (sender cell-type, receiver cell-type) pairs that exist in those
    regions. Missing (region, cell-type) nodes are simply omitted from the
    average (NaN-aware mean) — we never fabricate zeros.

    Parameters
    ----------
    nc_h5_path
        Path to the expanded NeuronChat H5 (``net/`` group, one dataset per
        interaction; ``group_names`` attribute on root with
        ``"{region}::{cell_type}"`` labels).
    edge_table
        DataFrame with ``edge_idx``, ``src_region``, ``tgt_region`` columns
        (produced by :func:`clrc.features.construction.build_edge_table`).
        Its ``src_region`` / ``tgt_region`` values must be a subset of the
        regions present in the NC H5 group_names.

    Returns
    -------
    X : np.ndarray, shape (n_edges, n_lr_pairs)
        Arithmetic-mean NC score per (edge, LR) pair.
    feature_names : list[str]
        LR pair names in the order they appear in the H5
        ``interaction_names`` attribute.
    """
    nc_h5_path = Path(nc_h5_path)

    with h5py.File(nc_h5_path, "r") as f:
        group_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f.attrs["group_names"]
        ]
        interaction_names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in f.attrs["interaction_names"]
        ]

        _, _, node_lookup, _, celltypes = parse_group_names(group_names)

        regions_in_edges = sorted(
            set(edge_table["src_region"]).union(edge_table["tgt_region"])
        )
        missing = [r for r in regions_in_edges if not any(
            (r, ct) in node_lookup for ct in celltypes
        )]
        if missing:
            raise KeyError(
                f"Edge-table regions not present in NC H5 group_names: {missing}"
            )

        # Precompute per-region -> node indices list (across all cell types)
        region_to_nodes_lists: Dict[str, list] = {r: [] for r in regions_in_edges}
        for (reg, ct), nidx in node_lookup.items():
            if reg in region_to_nodes_lists:
                region_to_nodes_lists[reg].append(nidx)
        region_to_nodes: Dict[str, np.ndarray] = {
            r: np.asarray(nodes, dtype=np.intp)
            for r, nodes in region_to_nodes_lists.items()
        }

        # Precompute edge (i, j) node-index arrays once
        src_nodes_per_edge: List[np.ndarray] = [
            region_to_nodes[r] for r in edge_table["src_region"]
        ]
        tgt_nodes_per_edge: List[np.ndarray] = [
            region_to_nodes[r] for r in edge_table["tgt_region"]
        ]

        n_edges = len(edge_table)
        n_lr = len(interaction_names)
        X = np.full((n_edges, n_lr), np.nan, dtype=np.float64)

        net_grp = f["net"]
        for lr_idx, lr_name in enumerate(
            tqdm(interaction_names, desc="collapsing NC")
        ):
            mat = net_grp[lr_name][:]  # (n_nodes, n_nodes)
            for e_idx in range(n_edges):
                src_n = src_nodes_per_edge[e_idx]
                tgt_n = tgt_nodes_per_edge[e_idx]
                block = mat[np.ix_(src_n, tgt_n)]
                # NaN-aware arithmetic mean. If all entries NaN, leave NaN.
                if np.all(np.isnan(block)):
                    continue
                X[e_idx, lr_idx] = float(np.nanmean(block))

    logger.info(
        "Region-collapsed NC: shape=%s, n_edges=%d, n_lr=%d",
        X.shape, n_edges, n_lr,
    )
    return X, list(interaction_names)


# ---------------------------------------------------------------------------
#  Ligand x Receptor expression product
# ---------------------------------------------------------------------------

def _region_mean_across_genes(
    expression: pd.DataFrame,
    gene_list: Sequence[str],
) -> Optional[np.ndarray]:
    """Return per-region arithmetic mean across ``gene_list`` genes present
    in ``expression.columns``. Returns ``None`` if no listed gene is present."""
    present = [g for g in gene_list if g in expression.columns]
    if not present:
        return None
    return expression.loc[:, present].mean(axis=1).to_numpy(dtype=np.float64)


def build_lr_expression_product(
    expression: pd.DataFrame,
    lr_pair_table: pd.DataFrame,
    edge_table: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Outer product of per-region ligand-mean and receptor-mean per LR pair.

    Under arithmetic means this is algebraically identical to
    ``mean_{(g_L, g_R)} E[i, g_L] * E[j, g_R]`` (the product-then-mean form).

    Parameters
    ----------
    expression
        Per-region × per-gene DataFrame. Rows indexed by region label,
        columns by gene symbol. Caller is responsible for upstream
        preprocessing (expected: log1p of raw ABC counts).
    lr_pair_table
        DataFrame with columns ``lr_name``, ``ligand_genes``, ``receptor_genes``.
        Gene columns can be either list[str] or "+"-separated strings (the
        latter matches the convention used in the feature-importance CSVs).
    edge_table
        As produced by ``build_edge_table``. ``src_region`` / ``tgt_region``
        must be a subset of ``expression.index``.

    Returns
    -------
    X : np.ndarray, shape (n_edges, n_lr_pairs)
        Feature matrix. For an LR pair with no ligand genes or no receptor
        genes present in ``expression.columns``, that column is entirely
        NaN. Caller can drop such columns downstream (same contract as the
        NaN-threshold filter in the main pipeline).
    feature_names : list[str]
        LR pair names in the order of ``lr_pair_table.lr_name``.
    """
    regions_in_edges = sorted(
        set(edge_table["src_region"]).union(edge_table["tgt_region"])
    )
    missing = [r for r in regions_in_edges if r not in expression.index]
    if missing:
        raise KeyError(
            f"Edge-table regions not present in expression.index: {missing}"
        )

    def _parse_genes(v) -> List[str]:
        if isinstance(v, (list, tuple, set, np.ndarray)):
            return [str(x) for x in v]
        if pd.isna(v) or v == "":
            return []
        # "+"-separated convention used in importance CSV metadata
        return [g.strip() for g in str(v).split("+") if g.strip()]

    n_edges = len(edge_table)
    n_lr = len(lr_pair_table)
    X = np.full((n_edges, n_lr), np.nan, dtype=np.float64)

    src_regions = edge_table["src_region"].to_numpy()
    tgt_regions = edge_table["tgt_region"].to_numpy()

    n_missing_side = 0
    for lr_idx, (_, row) in enumerate(tqdm(
        list(lr_pair_table.iterrows()), desc="L*R expression product"
    )):
        lig = _parse_genes(row["ligand_genes"])
        rec = _parse_genes(row["receptor_genes"])
        lig_mean = _region_mean_across_genes(expression, lig)
        rec_mean = _region_mean_across_genes(expression, rec)
        if lig_mean is None or rec_mean is None:
            n_missing_side += 1
            continue
        # Map region names to positions once per LR pair
        # expression.index is a pandas Index; use get_indexer for vectorized lookup
        src_idx = expression.index.get_indexer(src_regions)
        tgt_idx = expression.index.get_indexer(tgt_regions)
        X[:, lr_idx] = lig_mean[src_idx] * rec_mean[tgt_idx]

    logger.info(
        "L*R expression product: shape=%s, dropped_lr_pairs=%d (NaN column)",
        X.shape, n_missing_side,
    )
    return X, list(lr_pair_table["lr_name"].astype(str))


# ---------------------------------------------------------------------------
#  Spatial gene co-expression (Pearson correlation across gene profiles)
# ---------------------------------------------------------------------------

def build_spatial_gene_coexpression(
    expression: pd.DataFrame,
    lr_gene_subset: Optional[Sequence[str]],
    edge_table: pd.DataFrame,
) -> np.ndarray:
    """Per-edge Pearson correlation of region-gene profiles.

    For each edge ``(i, j)`` this is the Pearson correlation between the
    vectors ``expression[i, G]`` and ``expression[j, G]`` across genes ``G``
    (the ``lr_gene_subset`` when provided, else all columns of
    ``expression``). This is the "spatial gene co-expression" baseline.

    Parameters
    ----------
    expression
        Per-region × per-gene DataFrame. Rows indexed by region label.
    lr_gene_subset
        Optional list of gene symbols to restrict correlation to. Genes not
        present in ``expression.columns`` are silently dropped (not
        fabricated). Pass ``None`` to use all columns.
    edge_table
        As produced by ``build_edge_table``. ``src_region`` / ``tgt_region``
        must be a subset of ``expression.index``.

    Returns
    -------
    X : np.ndarray, shape (n_edges, 1)
        Pearson correlation per edge. Returns NaN for self-edges with
        zero-variance gene vectors (degenerate case); finite values lie
        in ``[-1, 1]``.
    """
    regions_in_edges = sorted(
        set(edge_table["src_region"]).union(edge_table["tgt_region"])
    )
    missing = [r for r in regions_in_edges if r not in expression.index]
    if missing:
        raise KeyError(
            f"Edge-table regions not present in expression.index: {missing}"
        )

    if lr_gene_subset is not None:
        present = [g for g in lr_gene_subset if g in expression.columns]
        if not present:
            raise ValueError(
                "lr_gene_subset has zero overlap with expression.columns. "
                "Verify gene-symbol aliasing upstream."
            )
        expr_matrix = expression.loc[:, present]
    else:
        expr_matrix = expression

    # region -> row index in the restricted matrix
    region_to_row = {r: i for i, r in enumerate(expr_matrix.index)}
    E = expr_matrix.to_numpy(dtype=np.float64)  # (n_regions, n_genes)

    # Precompute normalised vectors for fast correlation: (E - row_mean) / row_std
    row_mean = E.mean(axis=1, keepdims=True)
    row_std = E.std(axis=1, keepdims=True, ddof=0)  # scipy.stats.pearsonr uses ddof=0 normalization internally
    # In-division zero-std rows produce NaN -> later NaN in result; accept that.
    with np.errstate(invalid="ignore", divide="ignore"):
        E_norm = (E - row_mean) / row_std  # (n_regions, n_genes)

    n_genes = E.shape[1]
    # Full (n_regions x n_regions) correlation matrix once
    with np.errstate(invalid="ignore"):
        corr_mat = (E_norm @ E_norm.T) / n_genes

    src_rows = np.asarray(
        [region_to_row[r] for r in edge_table["src_region"]], dtype=np.intp
    )
    tgt_rows = np.asarray(
        [region_to_row[r] for r in edge_table["tgt_region"]], dtype=np.intp
    )
    X = corr_mat[src_rows, tgt_rows].reshape(-1, 1)

    logger.info(
        "Spatial gene co-expression: shape=%s, n_genes_used=%d",
        X.shape, n_genes,
    )
    return X
