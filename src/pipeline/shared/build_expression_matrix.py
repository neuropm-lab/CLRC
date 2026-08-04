#!/usr/bin/env python3
"""Build the annotated expression matrix that the connectome build consumes.
Calls clrc.preprocessing.abc / clrc.preprocessing.harmonization.

Both sources terminate at the same contract: an ``.h5ad`` carrying
``obs['region_supercluster_celltype']`` as the region-by-cell-type node label
and a ``data`` layer. They reach it by different procedures, because only one
of the two datasets ships with the ABC cell-type taxonomy already applied:

  --source abc      reads cell types from the ABC cluster-annotation hierarchy
  --source rosmap   maps onto that taxonomy with Cell Type Mapper

Inputs (see ``expression_matrix:`` in the YAML config):
  - ``abc``: the two ABC 10Xv3 releases, the interaction database CSV, the ABC
    gene map, and the ABC cell/gene/annotation/membership metadata tables.
  - ``rosmap``: the ROSMAP matrix and a Cell Type Mapper configuration.

Output:
  - ``expression_matrix.<source>.output_h5ad``, which is what
    ``build_connectome.py`` reads as ``neuronchat.expression_h5ad``.

Usage:
    uv run python src/pipeline/shared/build_expression_matrix.py \\
        --config configs/abc_expanded.yaml --source abc

    uv run python src/pipeline/shared/build_expression_matrix.py \\
        --config configs/rosmap_expanded.yaml --source rosmap
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clrc.core.io import load_yaml_config
from clrc.core.logging import setup_logging
from clrc.preprocessing import abc as abc_prep
from clrc.preprocessing import harmonization as harmonization_prep

SOURCES = ("abc", "rosmap")

BUILDERS = {
    "abc": abc_prep.build_expression_matrix,
    "rosmap": harmonization_prep.build_expression_matrix,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the annotated expression matrix for connectome construction"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=None,
        help="Override expression_matrix.source from the config.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    em_cfg = cfg["expression_matrix"]
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("build_expression_matrix", out_base)

    source = args.source or em_cfg.get("source")
    if source not in SOURCES:
        raise ValueError(
            f"Invalid source {source!r}. Must be one of {SOURCES}. "
            "Set expression_matrix.source in the config or pass --source."
        )
    if source not in em_cfg:
        raise KeyError(
            f"Config has no 'expression_matrix.{source}' block."
        )

    logger.info("building expression matrix from source=%s", source)
    output = BUILDERS[source](em_cfg[source])
    logger.info("expression matrix ready: %s", output)


if __name__ == "__main__":
    main()
