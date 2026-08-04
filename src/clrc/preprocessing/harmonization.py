"""Cell-type label harmonization onto a reference taxonomy.

Maps a dataset whose cell-type labels do not already follow the reference
taxonomy onto it, using the Allen Institute's Cell Type Mapper:

1. Convert ``var_names`` from gene symbols to ENSG identifiers, which is what
   Cell Type Mapper requires, retaining the symbols for later restoration.
2. Run Cell Type Mapper against the precomputed statistics and query markers
   for the target taxonomy.
3. Join the assignments back, build the region-by-cell-type node label, and
   restore gene symbols.

The counterpart in :mod:`clrc.preprocessing.abc` needs none of this, because
the ABC atlas already carries the taxonomy and its cell types can be read
straight out of the cluster-annotation hierarchy. Both reach the same output
contract.

No gene selection is applied here; the matrix is scoped to ligand-receptor
genes later, inside the connectome build.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata
import pandas as pd
import scanpy as sc
import yaml

logger = logging.getLogger("clrc.preprocessing.harmonization")

# Cell Type Mapper writes a provenance preamble above the CSV header.
MAPPER_HEADER_ROWS = 4


def to_ensg_ids(
    adata: anndata.AnnData,
    *,
    symbol_column: str = "varnames",
    ensg_column: str = "gene_ids",
) -> anndata.AnnData:
    """Switch ``var_names`` from gene symbols to ENSG identifiers.

    Parameters
    ----------
    adata : anndata.AnnData
        Matrix whose ``var`` carries ENSG identifiers in ``ensg_column``.
    symbol_column : str
        ``var`` column to write the original symbols into so they can be
        restored after mapping.
    ensg_column : str
        ``var`` column holding the ENSG identifiers.

    Returns
    -------
    anndata.AnnData
        The same object, modified in place.
    """
    if ensg_column not in adata.var.columns:
        raise KeyError(
            f"var column {ensg_column!r} not found "
            f"(columns: {list(adata.var.columns)})"
        )
    adata.var[symbol_column] = adata.var_names
    adata.var_names = adata.var[ensg_column]
    return adata


def run_cell_type_mapper(mapper_config: Path) -> None:
    """Run Cell Type Mapper locally with the given configuration.

    Parameters
    ----------
    mapper_config : Path
        YAML consumed by ``FromSpecifiedMarkersRunner``, naming the query
        h5ad, the precomputed statistics, the query markers, and the output
        CSV. See the Cell Type Mapper documentation for the schema.
    """
    from cell_type_mapper.cli.from_specified_markers import (
        FromSpecifiedMarkersRunner,
    )

    with mapper_config.open() as handle:
        cfg = yaml.safe_load(handle)
    logger.info("running Cell Type Mapper with %s", mapper_config)
    FromSpecifiedMarkersRunner(input_data=cfg, args=[]).run()


def attach_assignments(
    adata: anndata.AnnData,
    result_csv: Path,
    *,
    region_column: str = "BrainRegion",
    supercluster_column: str = "supercluster_name",
    symbol_column: str = "varnames",
) -> anndata.AnnData:
    """Join Cell Type Mapper output and build the node label.

    Parameters
    ----------
    adata : anndata.AnnData
        Matrix that was submitted to the mapper.
    result_csv : Path
        Mapper output CSV, keyed by ``cell_id``.
    region_column : str
        ``obs`` column holding the brain region.
    supercluster_column : str
        ``obs`` column that the mapper output supplies.
    symbol_column : str
        ``var`` column holding the original gene symbols.

    Returns
    -------
    anndata.AnnData
        Matrix carrying ``region_supercluster_celltype``, a ``data`` layer,
        and gene symbols restored as ``var_names``.
    """
    result = pd.read_csv(result_csv, skiprows=MAPPER_HEADER_ROWS)
    adata.obs = adata.obs.join(
        result.set_index("cell_id"), how="inner", rsuffix="_mmc"
    )
    logger.info("%d cells retained after mapper join", adata.shape[0])

    # anndataR reads expression from a named layer rather than from X.
    adata.layers["data"] = adata.X

    for col in (region_column, supercluster_column):
        if col not in adata.obs.columns:
            raise KeyError(
                f"obs column {col!r} not found after mapper join "
                f"(columns: {list(adata.obs.columns)})"
            )
    adata.obs["region_supercluster_celltype"] = (
        adata.obs[region_column].astype(str)
        + "::"
        + adata.obs[supercluster_column].astype(str)
    )

    adata.var["ensg_id"] = adata.var_names.to_series()
    adata.var_names = adata.var[symbol_column]
    return adata


def build_expression_matrix(cfg: dict) -> Path:
    """Run the full ROSMAP preparation from a config block.

    Parameters
    ----------
    cfg : dict
        The ``expression_matrix.rosmap`` config block. See
        ``configs/rosmap_expanded.example.yaml`` for the expected keys.

    Returns
    -------
    Path
        The harmonized matrix, ready for ``build_connectome.py``.
    """
    source = Path(cfg["source_h5ad"])
    logger.info("reading %s", source)
    adata = sc.read_h5ad(source)
    logger.info("loaded %d cells x %d genes", adata.shape[0], adata.shape[1])

    mapper_input = Path(cfg["mapper_input_h5ad"])
    if cfg.get("write_mapper_input", True):
        to_ensg_ids(
            adata,
            symbol_column=cfg.get("symbol_column", "varnames"),
            ensg_column=cfg.get("ensg_column", "gene_ids"),
        )
        mapper_input.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(mapper_input)
        logger.info("wrote mapper input -> %s", mapper_input)

    run_cell_type_mapper(Path(cfg["mapper_config"]))

    adata = attach_assignments(
        adata,
        Path(cfg["mapper_result_csv"]),
        region_column=cfg.get("region_column", "BrainRegion"),
        supercluster_column=cfg.get("supercluster_column", "supercluster_name"),
        symbol_column=cfg.get("symbol_column", "varnames"),
    )

    output = Path(cfg["output_h5ad"])
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    logger.info(
        "wrote %s (%d cells, %d nodes)",
        output,
        adata.shape[0],
        adata.obs["region_supercluster_celltype"].nunique(),
    )
    return output
