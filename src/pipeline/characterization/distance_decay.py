#!/usr/bin/env python3
"""Quantify how cell-type-pair connectivity decays with fiber distance.

Two stages:

1. Build a directed edge table joining the summed network with structural fiber
   length. For every ``(source region, source cell type) -> (target region,
   target cell type)`` node pair whose regions have a positive tractography
   fiber length, record that length alongside the summed connectivity.
2. Summarize distance-decay per connection category. Cell types are collapsed to
   coarse groups (Neuron / Glia / ...); region pairs are categorized as
   Neuron->Neuron, Neuron->Glia, Glia->Neuron or Glia->Glia; connectivity is
   averaged per region pair, binned by fiber length, and correlated
   (Pearson) with fiber length within each category.

Outputs (under ``<out-dir>``):
  - ``CT_CT_edges_with_distance.csv``   -- one row per directed node pair with
    source/target region, source/target cell type, fiber length, connectivity.
  - ``distance_decay_binned.csv``       -- per-category binned mean/std/count.
  - ``distance_decay_correlations.csv`` -- per-category Pearson r, p, N, range.

Usage::

    uv run python src/pipeline/characterization/distance_decay.py \\
        out/characterization/summed_net.csv \\
        --fiber-length data/mean_fiber_length.csv \\
        --out-dir out/characterization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from clrc.core.logging import setup_logging

from _common import celltypes_of, load_summed_matrix, regions_of, resolve_output_dir

logger = logging.getLogger("clrc.pipeline.characterization.distance_decay")

# Cell type -> coarse group for the distance-decay categorization.
CT_TO_GROUP = {
    # Neurons
    "Hippocampal dentate gyrus": "Neuron",
    "Hippocampal CA4": "Neuron",
    "Hippocampal CA1-3": "Neuron",
    "Cerebellar inhibitory": "Neuron",
    "Amygdala excitatory": "Neuron",
    "Deep-layer corticothalamic and 6b": "Neuron",
    "Deep-layer near-projecting": "Neuron",
    "Thalamic excitatory": "Neuron",
    "Deep-layer intratelencephalic": "Neuron",
    "Upper-layer intratelencephalic": "Neuron",
    "Medium spiny neuron": "Neuron",
    "Eccentric medium spiny neuron": "Neuron",
    "Midbrain-derived inhibitory": "Neuron",
    "LAMP5-LHX6 and Chandelier": "Neuron",
    "MGE interneuron": "Neuron",
    "CGE interneuron": "Neuron",
    "Mammillary body": "Neuron",
    # Glia
    "Astrocyte": "Glia",
    "Oligodendrocyte": "Glia",
    "Oligodendrocyte precursor": "Glia",
    "Committed oligodendrocyte precursor": "Glia",
    "Microglia": "Glia",
    "Bergmann glia": "Glia",
    # Vascular / stromal
    "Vascular": "Vascular/Stromal",
    "Fibroblast": "Vascular/Stromal",
    # Barrier / CSF
    "Ependymal": "Barrier/CSF",
    "Choroid plexus": "Barrier/CSF",
    # Developmental
    "Upper rhombic lip": "Developmental",
    "Lower rhombic lip": "Developmental",
    # Other
    "Miscellaneous": "Other",
    "Splatter": "Other",
}

CATEGORIES = ("Neuron-Neuron", "Neuron-Glia", "Glia-Neuron", "Glia-Glia")


def build_edge_table(
    summed_csv: Path,
    fiber_csv: Path,
    *,
    sc_region_map: Path | None,
    sc_region_col: str,
) -> pd.DataFrame:
    """Join the summed network with fiber length into a directed edge table."""
    row_labels, col_labels, M = load_summed_matrix(summed_csv)
    row_regions = regions_of(row_labels)
    row_cts = celltypes_of(row_labels)
    col_regions = regions_of(col_labels)
    col_cts = celltypes_of(col_labels)

    sc_df = pd.read_csv(fiber_csv, index_col=0)
    if sc_df.shape[0] != sc_df.shape[1]:
        raise ValueError(f"Fiber-length matrix is not square: {sc_df.shape}")
    sc_regions = sc_df.index.astype(str)
    region_to_sc_idx = {r: i for i, r in enumerate(sc_regions)}
    sc_mat = sc_df.to_numpy(dtype=float)

    # Restrict to the official SC region list if a mapping file is given,
    # otherwise use the regions present in the fiber-length matrix index.
    if sc_region_map is not None:
        table = (
            pd.read_excel(sc_region_map)
            if sc_region_map.suffix.lower() in (".xlsx", ".xls")
            else pd.read_csv(sc_region_map)
        )
        sc_region_set = set(table[sc_region_col].astype(str))
    else:
        sc_region_set = set(region_to_sc_idx)

    def keep_mask(regions: np.ndarray) -> np.ndarray:
        return np.array([r in region_to_sc_idx and r in sc_region_set for r in regions])

    keep_row = keep_mask(row_regions)
    keep_col = keep_mask(col_regions)
    logger.info(
        "Keeping %d/%d sender and %d/%d receiver nodes with SC regions",
        keep_row.sum(), len(keep_row), keep_col.sum(), len(keep_col),
    )

    r_reg = row_regions[keep_row]
    r_ct = row_cts[keep_row]
    c_reg = col_regions[keep_col]
    c_ct = col_cts[keep_col]
    M_keep = M[np.ix_(keep_row, keep_col)]

    r_sc = np.array([region_to_sc_idx[r] for r in r_reg])
    c_sc = np.array([region_to_sc_idx[r] for r in c_reg])
    fiber = sc_mat[np.ix_(r_sc, c_sc)]

    valid = (fiber > 0) & np.isfinite(M_keep)
    ii, jj = np.where(valid)
    edge_df = pd.DataFrame({
        "source_region": r_reg[ii],
        "target_region": c_reg[jj],
        "CT_source": r_ct[ii],
        "CT_target": c_ct[jj],
        "fiber_length": fiber[ii, jj],
        "connectivity": M_keep[ii, jj],
    })
    logger.info("Edge table: %d directed edges (SC fiber length > 0)", len(edge_df))
    return edge_df


def decay_by_category(
    edge_df: pd.DataFrame,
    *,
    bin_size: float,
    min_fiber: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin connectivity vs fiber length and correlate, per connection category."""
    df = edge_df[edge_df["source_region"] != edge_df["target_region"]].copy()
    df["src_group"] = df["CT_source"].map(CT_TO_GROUP)
    df["tgt_group"] = df["CT_target"].map(CT_TO_GROUP)
    df = df.dropna(subset=["src_group", "tgt_group"])

    df["category"] = "Other"
    for src, tgt, cat in [
        ("Neuron", "Neuron", "Neuron-Neuron"),
        ("Neuron", "Glia", "Neuron-Glia"),
        ("Glia", "Neuron", "Glia-Neuron"),
        ("Glia", "Glia", "Glia-Glia"),
    ]:
        df.loc[(df["src_group"] == src) & (df["tgt_group"] == tgt), "category"] = cat

    # Average connectivity per region pair (keeping one fiber length per pair).
    agg = (
        df.groupby(["source_region", "target_region", "category"])
        .agg(fiber_length=("fiber_length", "first"), connectivity=("connectivity", "mean"))
        .reset_index()
    )

    binned_rows: List[dict] = []
    corr_rows: List[dict] = []
    for cat in CATEGORIES:
        data = agg[(agg["category"] == cat) & (agg["fiber_length"] >= min_fiber)]
        if len(data) <= 10:
            logger.info("Category %s: only %d pairs — skipped", cat, len(data))
            continue

        fiber_bin = (data["fiber_length"] // bin_size) * bin_size
        fiber_bin = fiber_bin.where(fiber_bin != 0, min_fiber)
        grouped = data.assign(fiber_bin=fiber_bin).groupby("fiber_bin")["connectivity"]
        for fb, mean_c, std_c, cnt in zip(
            grouped.mean().index, grouped.mean().to_numpy(),
            grouped.std().to_numpy(), grouped.count().to_numpy(),
        ):
            binned_rows.append({
                "category": cat, "fiber_bin": float(fb),
                "mean_conn": float(mean_c),
                "std_conn": float(std_c) if np.isfinite(std_c) else 0.0,
                "count": int(cnt),
            })

        r, p = stats.pearsonr(data["fiber_length"], data["connectivity"])
        corr_rows.append({
            "category": cat,
            "n_pairs": int(len(data)),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "mean_conn": float(data["connectivity"].mean()),
            "fiber_min": float(data["fiber_length"].min()),
            "fiber_max": float(data["fiber_length"].max()),
        })
        logger.info("Category %s: N=%d, r=%.4f, p=%.3e", cat, len(data), r, p)

    return pd.DataFrame(binned_rows), pd.DataFrame(corr_rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summed_csv", type=Path, help="Summed matrix CSV from build_summed_matrix.py.")
    p.add_argument("--fiber-length", type=Path, required=True,
                   help="Square fiber-length matrix CSV (regions x regions, first column = region index).")
    p.add_argument("--sc-region-map", type=Path, default=None,
                   help="Optional table (.csv/.xlsx) restricting to the official SC region list.")
    p.add_argument("--sc-region-col", default="ABC region",
                   help="Region column in --sc-region-map (default 'ABC region').")
    p.add_argument("--bin-size", type=float, default=5.0, help="Fiber-length bin width in mm (default 5).")
    p.add_argument("--min-fiber", type=float, default=2.5, help="Minimum fiber length in mm (default 2.5).")
    p.add_argument("--out-dir", type=Path, default=Path("out/characterization"),
                   help="Output directory (relative paths anchored at the repo root).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = resolve_output_dir(args.out_dir)
    setup_logging("distance_decay", output_dir=out_dir)
    logger.info("Summed matrix: %s", args.summed_csv)
    logger.info("Fiber length: %s", args.fiber_length)
    logger.info("Output dir: %s", out_dir)

    edge_df = build_edge_table(
        args.summed_csv, args.fiber_length,
        sc_region_map=args.sc_region_map, sc_region_col=args.sc_region_col,
    )
    binned_df, corr_df = decay_by_category(
        edge_df, bin_size=args.bin_size, min_fiber=args.min_fiber,
    )

    edges_path = out_dir / "CT_CT_edges_with_distance.csv"
    binned_path = out_dir / "distance_decay_binned.csv"
    corr_path = out_dir / "distance_decay_correlations.csv"
    edge_df.to_csv(edges_path, index=False)
    binned_df.to_csv(binned_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    logger.info("Wrote %s, %s, %s", edges_path, binned_path, corr_path)

    print("\n" + "=" * 80)
    print("Distance-decay of connectivity by cell-type category "
          "(connectivity averaged per region pair)")
    print("=" * 80)
    if corr_df.empty:
        print("No category had enough region pairs to summarize.")
    else:
        print(corr_df.to_string(index=False))
    print(f"\nWritten to: {edges_path}, {binned_path}, {corr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
