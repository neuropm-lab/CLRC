"""Allen Brain Cell Atlas expression-matrix preparation.

Turns the raw ABC 10Xv3 neuronal and non-neuronal ``.h5ad`` releases into the
annotated, ligand-receptor-scoped matrix that the connectome build consumes:

1. Subset each release to the genes named by the interaction database,
   mapping the ABC ENSG identifiers to gene symbols.
2. Concatenate the neuronal and non-neuronal subsets.
3. Join the ABC cell and gene metadata, resolve each nucleus to its
   supercluster by walking the ABC cluster-annotation hierarchy, and build the
   ``region_supercluster_celltype`` node label.

Cell types are read from the ABC taxonomy directly. The ROSMAP counterpart in
:mod:`clrc.preprocessing.rosmap` instead maps onto this same taxonomy with
Cell Type Mapper; the two arrive at the same output contract by different
procedures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

logger = logging.getLogger("clrc.preprocessing.abc")

SUPERCLUSTER_SUFFIX = "_SUPC"


def find_supercluster_name(
    term_label: str,
    parent_map: dict[str, str],
    set_map: dict[str, str],
    name_map: dict[str, str],
) -> str:
    """Walk up the ABC annotation hierarchy to the enclosing supercluster.

    Parameters
    ----------
    term_label : str
        Cluster annotation term to start from.
    parent_map : dict
        Term label to parent term label.
    set_map : dict
        Term label to its annotation-set label; supercluster sets end in
        ``_SUPC``.
    name_map : dict
        Term label to human-readable name.

    Returns
    -------
    str
        Supercluster name, or an empty string if the walk reaches the top
        without finding one.
    """
    current = term_label
    while current:
        parent = parent_map.get(current)
        if not parent or parent not in set_map:
            return ""
        if set_map[parent].endswith(SUPERCLUSTER_SUFFIX):
            return name_map.get(parent, "")
        current = parent
    return ""


def subset_to_db_genes(
    input_h5ad: Path,
    output_h5ad: Path,
    *,
    db_csv: Path,
    gene_map_csv: Path,
    block_size: int,
) -> Path:
    """Write a copy of ``input_h5ad`` holding only interaction-database genes.

    The CSR matrix is filtered in row blocks and reassembled rather than
    loaded whole, which keeps peak memory proportional to ``block_size``
    instead of to the full ABC release.

    Parameters
    ----------
    input_h5ad : Path
        ABC release (neuronal or non-neuronal).
    output_h5ad : Path
        Destination.
    db_csv : Path
        Interaction database CSV with ``lig_contributor`` and
        ``receptor_subunit`` columns, semicolon-separated gene symbols.
    gene_map_csv : Path
        ABC gene table mapping ``gene_identifier`` (ENSG) to ``gene_symbol``.
    block_size : int
        Rows per read block.

    Returns
    -------
    Path
        ``output_h5ad``.
    """
    db = pd.read_csv(db_csv)
    cci_genes = {
        gene.strip()
        for _, row in db.iterrows()
        for gene in (
            str(row["receptor_subunit"]).split(";")
            + str(row["lig_contributor"]).split(";")
        )
        if gene.strip()
    }

    gene_map = pd.read_csv(gene_map_csv)
    ensg_to_symbol = dict(
        zip(gene_map["gene_identifier"], gene_map["gene_symbol"])
    )

    with h5py.File(input_h5ad, "r") as f:
        ensg_ids = [x.decode() for x in f["var"]["gene_identifier"][:]]
    mapped_symbols = [ensg_to_symbol.get(e) for e in ensg_ids]
    keep_cols = {
        i for i, sym in enumerate(mapped_symbols) if sym in cci_genes
    }
    kept_sorted = sorted(keep_cols)
    idx_map = {old: new for new, old in enumerate(kept_sorted)}
    logger.info(
        "%s: keeping %d of %d genes", input_h5ad.name, len(keep_cols), len(ensg_ids)
    )

    with h5py.File(input_h5ad, "r") as f:
        dtype_data = f["X"]["data"][0:1].dtype
        dtype_indices = f["X"]["indices"][0:1].dtype
        indptr = f["X"]["indptr"][:]
    n_obs = len(indptr) - 1

    new_data: list[np.ndarray] = []
    new_indices: list[list[int]] = []
    new_indptr = np.zeros(n_obs + 1, dtype=indptr.dtype)
    ptr = 0

    with h5py.File(input_h5ad, "r") as f:
        data_ds = f["X"]["data"]
        idx_ds = f["X"]["indices"]
        indptr = f["X"]["indptr"][:]
        for start in range(0, n_obs, block_size):
            end = min(start + block_size, n_obs)
            off_s, off_e = indptr[start], indptr[end]
            length = off_e - off_s
            buf_d = np.empty(length, dtype=dtype_data)
            buf_i = np.empty(length, dtype=dtype_indices)
            data_ds.read_direct(buf_d, source_sel=np.s_[off_s:off_e])
            idx_ds.read_direct(buf_i, source_sel=np.s_[off_s:off_e])
            rowptr = indptr[start : end + 1] - off_s
            for i in range(end - start):
                rs, re = rowptr[i], rowptr[i + 1]
                cols = buf_i[rs:re]
                vals = buf_d[rs:re]
                mask = np.isin(cols, kept_sorted)
                kept = cols[mask]
                new_data.append(vals[mask])
                new_indices.append([idx_map[c] for c in kept])
                ptr += int(mask.sum())
                new_indptr[start + i + 1] = ptr

    X_sub = csr_matrix(
        (np.concatenate(new_data), np.concatenate(new_indices), new_indptr),
        shape=(n_obs, len(keep_cols)),
    )

    meta = sc.read_h5ad(str(input_h5ad), backed="r")
    obs_df = meta.obs.copy()
    var_df = meta.var.copy().reset_index(drop=True)
    var_sub = var_df.iloc[kept_sorted].copy()
    var_sub["gene_symbol"] = [mapped_symbols[i] for i in kept_sorted]
    var_sub.index = var_sub["gene_symbol"]
    var_sub = var_sub.drop(columns=["gene_identifier"], errors="ignore")

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    sc.AnnData(X=X_sub, obs=obs_df, var=var_sub).write(str(output_h5ad))
    logger.info("wrote %s", output_h5ad)
    return output_h5ad


def annotate(
    input_h5ad: Path,
    output_h5ad: Path,
    *,
    cell_meta_csv: Path,
    gene_meta_csv: Path,
    annotation_csv: Path,
    membership_csv: Path,
) -> Path:
    """Attach ABC metadata and build the region-by-cell-type node label.

    Parameters
    ----------
    input_h5ad : Path
        Concatenated, gene-subset matrix.
    output_h5ad : Path
        Destination for the annotated matrix.
    cell_meta_csv, gene_meta_csv : Path
        ABC cell and gene metadata tables.
    annotation_csv : Path
        ABC cluster annotation terms (label, parent, set label, name).
    membership_csv : Path
        ABC cluster-alias to annotation-term membership table.

    Returns
    -------
    Path
        ``output_h5ad``.
    """
    adata = sc.read_h5ad(input_h5ad)

    cell_meta = pd.read_csv(cell_meta_csv, index_col="cell_label", dtype=str)
    # ABC appends a per-release suffix to the barcode that the metadata table
    # does not carry, so it has to come off before the join.
    adata.obs.index = (
        adata.obs.index.str.removesuffix("-0").str.removesuffix("-1")
    )
    adata.obs = adata.obs.join(
        cell_meta[cell_meta.columns.difference(adata.obs.columns)], how="left"
    )

    gene_meta = pd.read_csv(gene_meta_csv, index_col="gene_symbol", dtype=str)
    adata.var = adata.var.join(gene_meta, how="left")

    annot_df = pd.read_csv(annotation_csv)
    member_df = (
        pd.read_csv(membership_csv)
        .query("cluster_annotation_term_set_name == 'subcluster'")[
            ["cluster_alias", "cluster_annotation_term_label"]
        ]
        .rename(columns={"cluster_annotation_term_label": "term_label"})
    )
    alias_map = dict(
        zip(member_df.cluster_alias.astype(str), member_df.term_label)
    )
    parent_map = dict(zip(annot_df.label, annot_df.parent_term_label))
    set_map = dict(zip(annot_df.label, annot_df.cluster_annotation_term_set_label))
    name_map = dict(zip(annot_df.label, annot_df.name))

    obs = adata.obs
    obs["annotation_label"] = (
        obs["cluster_alias"].astype(str).map(alias_map).fillna("")
    )
    obs["subcluster_name"] = obs["annotation_label"].map(name_map).fillna("")
    obs["cluster_label"] = obs["annotation_label"].map(parent_map)
    obs["cluster_name"] = obs["cluster_label"].map(name_map).fillna("")
    obs["supercluster_name"] = obs["annotation_label"].apply(
        lambda term: find_supercluster_name(term, parent_map, set_map, name_map)
        if term
        else ""
    )
    obs["region_of_interest_label"] = obs["region_of_interest_label"].str.replace(
        r"^Human ", "", regex=True
    )
    obs["region_supercluster_celltype"] = (
        obs["region_of_interest_label"] + "::" + obs["supercluster_name"]
    )
    adata.obs = obs

    adata.layers["data"] = adata.X.copy()

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(output_h5ad))
    logger.info(
        "wrote %s (%d cells, %d nodes)",
        output_h5ad,
        adata.shape[0],
        obs["region_supercluster_celltype"].nunique(),
    )
    return output_h5ad


def build_expression_matrix(cfg: dict) -> Path:
    """Run the full ABC preparation from a config block.

    Parameters
    ----------
    cfg : dict
        The ``expression_matrix.abc`` config block. See
        ``configs/abc_expanded.example.yaml`` for the expected keys.

    Returns
    -------
    Path
        The annotated matrix, ready for ``build_connectome.py``.
    """
    db_csv = Path(cfg["db_csv"])
    gene_map_csv = Path(cfg["gene_map_csv"])
    block_size = cfg.get("block_size", 50_000)

    neuron_out = Path(cfg["neuron_subset_h5ad"])
    nonneuron_out = Path(cfg["nonneuron_subset_h5ad"])

    subset_to_db_genes(
        Path(cfg["neuron_h5ad"]),
        neuron_out,
        db_csv=db_csv,
        gene_map_csv=gene_map_csv,
        block_size=block_size,
    )
    subset_to_db_genes(
        Path(cfg["nonneuron_h5ad"]),
        nonneuron_out,
        db_csv=db_csv,
        gene_map_csv=gene_map_csv,
        block_size=block_size,
    )

    merged_out = Path(cfg["merged_h5ad"])
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged = sc.read_h5ad(neuron_out).concatenate(sc.read_h5ad(nonneuron_out))
    merged.write(str(merged_out))
    logger.info("merged neuronal + non-neuronal -> %s", merged_out)

    return annotate(
        merged_out,
        Path(cfg["output_h5ad"]),
        cell_meta_csv=Path(cfg["cell_meta_csv"]),
        gene_meta_csv=Path(cfg["gene_meta_csv"]),
        annotation_csv=Path(cfg["annotation_csv"]),
        membership_csv=Path(cfg["membership_csv"]),
    )
