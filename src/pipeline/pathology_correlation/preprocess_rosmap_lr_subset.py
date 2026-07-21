#!/usr/bin/env python3
"""Preprocess the full ROSMAP snRNA-seq h5ad into an LR-gene subset h5ad.

The per-subject per-region per-gene expression covariate in the AD
partial-Spearman driver reads only the union of LR-database genes. The
full panel (``snRNA_Matrix.2263395_Cells_July7_2025.h5ad``) is ~2.26 M
cells x ~36,600 genes (17 GB), expensive to load on every driver
invocation. This script builds a one-off gene-subsetted h5ad that
keeps only the union of genes referenced by the old and expanded
interaction databases.

Output is gitignored (``data/`` is in ``.gitignore``) and idempotent:
rerunning without ``--overwrite`` is a no-op.

Usage
-----
    uv run python src/pipeline/pathology_correlation/preprocess_rosmap_lr_subset.py \
        --full-h5ad <path> --old-db src/neuronchat/data/interactionDB_human.json \
        --expanded-db src/neuronchat/data/merged_interactionDB_human_1092LR.json \
        --output data/AD_Multiomic_MultiRegion/rosmap_lr_subset_{n_genes}.h5ad

``{n_genes}`` in ``--output`` is a literal placeholder: the script
substitutes the actual union-in-panel count at write time.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Set

import anndata as ad

from clrc.core.logging import setup_logging

logger = logging.getLogger("clrc.pipeline.preprocess_rosmap_lr_subset")


def _union_db_genes(db_paths: List[Path]) -> Set[str]:
    """Return the union of ligand + receptor genes across the provided DBs."""
    genes: Set[str] = set()
    for p in db_paths:
        with p.open() as f:
            db = json.load(f)
        for entry in db.values():
            for g in entry.get("lig_contributor", []):
                genes.add(str(g))
            for g in entry.get("receptor_subunit", []):
                genes.add(str(g))
        logger.info("DB %s -> %d entries", p.name, len(db))
    return genes


def _resolve_output_path(output: Path, n_genes: int) -> Path:
    """Substitute ``{n_genes}`` in the output path template."""
    name = output.name.format(n_genes=n_genes)
    return output.with_name(name)


def preprocess(
    full_h5ad: Path,
    old_db: Path,
    expanded_db: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Build the gene-subsetted h5ad and return its final path."""
    if not full_h5ad.is_file():
        raise FileNotFoundError(f"Full h5ad missing: {full_h5ad}")
    for p in (old_db, expanded_db):
        if not p.is_file():
            raise FileNotFoundError(f"Interaction DB missing: {p}")

    union_genes = _union_db_genes([old_db, expanded_db])
    logger.info("Union LR gene set across both DBs: %d genes", len(union_genes))

    # Backed-mode open so we can subset var and write without loading X fully.
    adata = ad.read_h5ad(full_h5ad, backed="r")
    panel = set(adata.var_names)
    present_genes = sorted(union_genes & panel)
    missing = sorted(union_genes - panel)
    logger.info(
        "Genes present in panel: %d / %d (missing %d)",
        len(present_genes), len(union_genes), len(missing),
    )
    if missing:
        logger.info("First 10 missing: %s", missing[:10])

    if len(present_genes) < 500:
        raise ValueError(
            f"Only {len(present_genes)} union genes found in panel; "
            f"expected >=500. Aborting — check DB / panel alignment."
        )

    final_output = _resolve_output_path(output, len(present_genes))
    final_output.parent.mkdir(parents=True, exist_ok=True)
    if final_output.exists() and not overwrite:
        logger.info(
            "Output %s exists and --overwrite not set; skipping.",
            final_output,
        )
        return final_output

    logger.info(
        "Subsetting (%d cells, %d genes) -> (%d cells, %d genes)...",
        adata.n_obs, adata.n_vars, adata.n_obs, len(present_genes),
    )
    # Use var-name indexing on backed mode. anndata loads X for the requested
    # columns lazily; calling .to_memory() on the sliced view materialises
    # only the selected columns.
    sub = adata[:, present_genes].to_memory()
    logger.info("Subset materialised: shape=%s", sub.shape)

    logger.info("Writing -> %s", final_output)
    sub.write_h5ad(final_output, compression="gzip")
    logger.info("Done.")
    return final_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the LR-gene subset ROSMAP h5ad."
    )
    parser.add_argument("--full-h5ad", type=Path, required=True)
    parser.add_argument("--old-db", type=Path, required=True)
    parser.add_argument("--expanded-db", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path (may contain '{n_genes}' placeholder).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        help="Override log directory (default: output parent).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or args.output.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("preprocess_rosmap_lr_subset", log_dir)

    preprocess(
        full_h5ad=args.full_h5ad,
        old_db=args.old_db,
        expanded_db=args.expanded_db,
        output=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
