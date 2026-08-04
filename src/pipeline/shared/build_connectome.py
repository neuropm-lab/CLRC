#!/usr/bin/env python3
"""Build the ligand-receptor connectome by running NeuronChat on an annotated
expression matrix. Calls clrc.preprocessing.connectome.

Consumes the output of ``build_expression_matrix.py`` plus the merged
ligand-receptor interaction database, and writes the NeuronChat H5 that every
downstream stage reads as ``data.nc_h5``.

Two scopes, both running the identical NeuronChat procedure:

  --scope dataset   one run over all cells -> a single connectome H5
  --scope subject   one run per subject   -> one H5 per subject

Inputs (see ``neuronchat:`` in the YAML config):
  - ``expression_h5ad``: annotated, gene-scoped matrix carrying ``group_by``.
  - ``db_json``: merged CellChatDB + NeuronChatDB interaction database.

Outputs:
  - scope ``dataset``: ``neuronchat.output_h5``
  - scope ``subject``: one H5 per subject under ``neuronchat.output_dir``,
    named by ``neuronchat.filename``

Usage:
    uv run python src/pipeline/shared/build_connectome.py \\
        --config configs/abc_expanded.yaml --scope dataset

    uv run python src/pipeline/shared/build_connectome.py \\
        --config configs/rosmap_expanded.yaml --scope subject

Subject runs are resumable: subjects with an existing output are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata

from clrc.core.io import load_yaml_config
from clrc.core.logging import setup_logging
from clrc.preprocessing.connectome import (
    normalize_if_needed,
    run_connectome,
    run_connectome_by_subject,
    subset_to_db_genes,
)
from neuronchat import load_db

SCOPES = ("dataset", "subject")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the NeuronChat ligand-receptor connectome"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default=None,
        help="Override neuronchat.scope from the config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override neuronchat.device ('cpu', 'cuda', 'cuda:0', ...).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override neuronchat.n_jobs (CPU backend only).",
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=None,
        help="Scope 'subject' only: file with one subject ID per line.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    nc_cfg = cfg["neuronchat"]
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("build_connectome", out_base)

    scope = args.scope or nc_cfg.get("scope", "dataset")
    if scope not in SCOPES:
        raise ValueError(f"Invalid scope {scope!r}. Must be one of {SCOPES}.")

    device = args.device or nc_cfg.get("device", "cpu")
    n_jobs = args.n_jobs if args.n_jobs is not None else nc_cfg.get("n_jobs", 1)
    M = nc_cfg["M"]

    logger.info(
        "scope=%s M=%s seed=%s device=%s", scope, M, nc_cfg.get("seed", 42), device
    )

    db = load_db(Path(nc_cfg["db_json"]))
    logger.info("loaded %d interactions", len(db))

    h5ad_path = Path(nc_cfg["expression_h5ad"])
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"neuronchat.expression_h5ad does not exist: {h5ad_path}. "
            "Run build_expression_matrix.py first."
        )
    logger.info("reading %s", h5ad_path)
    adata = anndata.read_h5ad(h5ad_path)
    logger.info("loaded %d cells x %d genes", adata.shape[0], adata.shape[1])

    normalize_if_needed(adata, nc_cfg.get("normalize", "auto"))
    adata = subset_to_db_genes(adata, db)

    group_by = nc_cfg["group_by"]
    seed = nc_cfg.get("seed", 42)
    fdr = nc_cfg.get("fdr", 0.05)
    layer = nc_cfg.get("layer")

    if scope == "dataset":
        run_connectome(
            adata,
            db,
            Path(nc_cfg["output_h5"]),
            group_by=group_by,
            M=M,
            fdr=fdr,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
            layer=layer,
        )
    else:
        subjects = None
        if args.subjects_file is not None:
            subjects = [
                line.strip()
                for line in args.subjects_file.read_text().splitlines()
                if line.strip()
            ]
        kwargs = {}
        for key in ("min_cells", "min_groups", "filename"):
            if key in nc_cfg:
                kwargs[key] = nc_cfg[key]
        written = run_connectome_by_subject(
            adata,
            db,
            Path(nc_cfg["output_dir"]),
            subject_col=nc_cfg["subject_col"],
            group_by=group_by,
            M=M,
            fdr=fdr,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
            layer=layer,
            subjects=subjects,
            **kwargs,
        )
        logger.info("wrote %d subject connectomes", len(written))


if __name__ == "__main__":
    main()
