#!/usr/bin/env python3
"""Rank cell types (and coarse classes) by outgoing/incoming strength.

Reads the summed node-by-node network (``build_summed_matrix.py`` output) and,
collapsing over regions, summarizes each cell type by its mean outgoing
(sender) and incoming (receiver) connectivity over positive entries. Each cell
type and each coarse class (Excitatory / Inhibitory / Glia / Other) receives a
percentile bootstrap 95% CI and a label-permutation p-value (Benjamini-Hochberg
corrected across cell types / across classes). A ranking of the strongest
directed cell-type -> cell-type pairs is also produced.

Outputs (under ``<out-dir>``):
  - ``celltype_stats.csv``       -- per-cell-type out/in mean, CI, permutation p/q.
  - ``class_stats.csv``          -- per-class out/in mean, CI, p/q, stars.
  - ``top_celltype_pairs.csv``   -- strongest directed cell-type pairs.

Usage::

    uv run python src/pipeline/characterization/top_cells.py \\
        out/characterization/summed_net.csv --out-dir out/characterization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from clrc.core.logging import setup_logging

from _common import (
    bootstrap_ci,
    cell_class,
    celltypes_of,
    fdr_bh,
    load_summed_matrix,
    mean_positive,
    perm_pvalue,
    resolve_output_dir,
    sig_star,
)

logger = logging.getLogger("clrc.pipeline.characterization.top_cells")

CLASSES = ("Excitatory", "Inhibitory", "Glia", "Other")


def _label_classes(celltypes: np.ndarray) -> np.ndarray:
    return np.array([cell_class(ct) for ct in celltypes])


def compute_stats(
    M: np.ndarray,
    row_ct: np.ndarray,
    col_ct: np.ndarray,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap CIs + permutation stats at the cell-type and class level."""
    rng = np.random.default_rng(seed)

    celltypes = np.unique(np.concatenate([row_ct, col_ct]))
    row_grp = _label_classes(row_ct)
    col_grp = _label_classes(col_ct)
    classes = [c for c in CLASSES if c in set(row_grp) | set(col_grp)]
    logger.info("Cell types=%d, classes=%d", len(celltypes), len(classes))

    obs_out = {ct: mean_positive(M[row_ct == ct, :]) for ct in celltypes}
    obs_in = {ct: mean_positive(M[:, col_ct == ct]) for ct in celltypes}
    obs_grp_out = {g: mean_positive(M[row_grp == g, :]) for g in classes}
    obs_grp_in = {g: mean_positive(M[:, col_grp == g]) for g in classes}

    # Label-permutation null: shuffle cell-type labels on rows/columns; class
    # labels are re-derived from the permuted cell-type labels.
    logger.info("Running %d permutations (cell type + class)", n_perm)
    null_out = {ct: np.zeros(n_perm) for ct in celltypes}
    null_in = {ct: np.zeros(n_perm) for ct in celltypes}
    null_grp_out = {g: np.zeros(n_perm) for g in classes}
    null_grp_in = {g: np.zeros(n_perm) for g in classes}
    for p in range(n_perm):
        if (p + 1) % 200 == 0 or (p + 1) == n_perm:
            logger.info("  permutation %d/%d", p + 1, n_perm)
        perm_row = rng.permutation(row_ct)
        perm_col = rng.permutation(col_ct)
        perm_row_grp = _label_classes(perm_row)
        perm_col_grp = _label_classes(perm_col)
        for ct in celltypes:
            null_out[ct][p] = mean_positive(M[perm_row == ct, :])
            null_in[ct][p] = mean_positive(M[:, perm_col == ct])
        for g in classes:
            null_grp_out[g][p] = mean_positive(M[perm_row_grp == g, :])
            null_grp_in[g][p] = mean_positive(M[:, perm_col_grp == g])

    logger.info("Computing bootstrap CIs")
    ct_rows: List[dict] = []
    for ct in celltypes:
        om, ol, oh = bootstrap_ci(M[row_ct == ct, :], rng=rng, n_boot=n_boot)
        im, il, ih = bootstrap_ci(M[:, col_ct == ct], rng=rng, n_boot=n_boot)
        ct_rows.append({
            "cell_type": ct,
            "class": cell_class(ct),
            "out_mean": om, "out_lo": ol, "out_hi": oh,
            "p_out": perm_pvalue(null_out[ct], obs_out[ct], n_perm),
            "in_mean": im, "in_lo": il, "in_hi": ih,
            "p_in": perm_pvalue(null_in[ct], obs_in[ct], n_perm),
        })
    ct_df = pd.DataFrame(ct_rows)
    ct_df["q_out"] = fdr_bh(ct_df["p_out"].to_numpy())
    ct_df["q_in"] = fdr_bh(ct_df["p_in"].to_numpy())

    grp_rows: List[dict] = []
    for g in classes:
        om, ol, oh = bootstrap_ci(M[row_grp == g, :], rng=rng, n_boot=n_boot)
        im, il, ih = bootstrap_ci(M[:, col_grp == g], rng=rng, n_boot=n_boot)
        grp_rows.append({
            "class": g,
            "out_mean": om, "out_lo": ol, "out_hi": oh,
            "p_out": perm_pvalue(null_grp_out[g], obs_grp_out[g], n_perm),
            "in_mean": im, "in_lo": il, "in_hi": ih,
            "p_in": perm_pvalue(null_grp_in[g], obs_grp_in[g], n_perm),
        })
    grp_df = pd.DataFrame(grp_rows)
    grp_df["q_out"] = fdr_bh(grp_df["p_out"].to_numpy())
    grp_df["q_in"] = fdr_bh(grp_df["p_in"].to_numpy())
    grp_df["sig_out"] = grp_df["q_out"].apply(sig_star)
    grp_df["sig_in"] = grp_df["q_in"].apply(sig_star)

    n_sig_out = int((ct_df["q_out"] < 0.05).sum())
    n_sig_in = int((ct_df["q_in"] < 0.05).sum())
    logger.info(
        "Significant cell types (FDR<0.05): %d outgoing, %d incoming (of %d)",
        n_sig_out, n_sig_in, len(celltypes),
    )
    return ct_df, grp_df


def top_celltype_pairs(
    M: np.ndarray,
    row_ct: np.ndarray,
    col_ct: np.ndarray,
    *,
    topk: int,
) -> pd.DataFrame:
    """Strongest directed cell-type -> cell-type pairs (self-pairs excluded)."""
    celltypes = np.unique(np.concatenate([row_ct, col_ct]))
    rows: List[dict] = []
    for cti in celltypes:
        for ctj in celltypes:
            if cti == ctj:
                continue
            m = mean_positive(M[np.ix_(row_ct == cti, col_ct == ctj)])
            if m > 0:
                rows.append({
                    "source_cell_type": cti,
                    "target_cell_type": ctj,
                    "mean_conn": m,
                    "source_class": cell_class(cti),
                    "target_class": cell_class(ctj),
                })
    return (
        pd.DataFrame(rows)
        .sort_values("mean_conn", ascending=False)
        .head(topk)
        .reset_index(drop=True)
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summed_csv", type=Path, help="Summed matrix CSV from build_summed_matrix.py.")
    p.add_argument("--topk", type=int, default=20, help="Rows to display / pairs to keep (default 20).")
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations (default 1000).")
    p.add_argument("--n-perm", type=int, default=1000, help="Permutations (default 1000).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    p.add_argument("--out-dir", type=Path, default=Path("out/characterization"),
                   help="Output directory (relative paths anchored at the repo root).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = resolve_output_dir(args.out_dir)
    setup_logging("top_cells", output_dir=out_dir)
    logger.info("Summed matrix: %s", args.summed_csv)
    logger.info("Output dir: %s", out_dir)

    row_labels, col_labels, M = load_summed_matrix(args.summed_csv)
    row_ct = celltypes_of(row_labels)
    col_ct = celltypes_of(col_labels)

    ct_df, grp_df = compute_stats(
        M, row_ct, col_ct, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
    )
    pairs_df = top_celltype_pairs(M, row_ct, col_ct, topk=args.topk)

    ct_path = out_dir / "celltype_stats.csv"
    grp_path = out_dir / "class_stats.csv"
    pairs_path = out_dir / "top_celltype_pairs.csv"
    ct_df.to_csv(ct_path, index=False)
    grp_df.to_csv(grp_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    logger.info("Wrote %s, %s, %s", ct_path, grp_path, pairs_path)

    top_out = ct_df.sort_values("out_mean", ascending=False).head(args.topk)
    top_in = ct_df.sort_values("in_mean", ascending=False).head(args.topk)
    print("\n" + "=" * 80)
    print(f"Top {args.topk} sending cell types (mean outgoing strength)")
    print("=" * 80)
    print(top_out[["cell_type", "class", "out_mean", "out_lo", "out_hi", "q_out"]].to_string(index=False))
    print("\n" + "=" * 80)
    print(f"Top {args.topk} receiving cell types (mean incoming strength)")
    print("=" * 80)
    print(top_in[["cell_type", "class", "in_mean", "in_lo", "in_hi", "q_in"]].to_string(index=False))
    print("\n" + "=" * 80)
    print("Cell-class strength (bootstrap 95% CI, permutation FDR)")
    print("=" * 80)
    print(grp_df[["class", "out_mean", "q_out", "sig_out", "in_mean", "q_in", "sig_in"]].to_string(index=False))
    print(f"\nWritten to: {ct_path}, {grp_path}, {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
