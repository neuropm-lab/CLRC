#!/usr/bin/env python3
"""Rank brain regions by outgoing/incoming connectivity strength.

Reads the summed node-by-node network (``build_summed_matrix.py`` output) and,
collapsing over cell types, summarizes each ABC region by its mean outgoing
(sender) and incoming (receiver) connectivity over positive entries. Each region
receives a percentile bootstrap 95% CI and a label-permutation p-value
(Benjamini-Hochberg corrected across regions). A ranking of the strongest
directed region->region pairs is also produced.

Outputs (under ``<out-dir>``):
  - ``region_stats.csv``        -- per-region out/in mean, CI, permutation p/q.
  - ``top_region_pairs.csv``    -- strongest directed region->region pairs.

Usage::

    uv run python src/pipeline/characterization/top_regions.py \\
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
    fdr_bh,
    load_summed_matrix,
    mean_positive,
    perm_pvalue,
    regions_of,
    resolve_output_dir,
)

logger = logging.getLogger("clrc.pipeline.characterization.top_regions")


def compute_stats(
    M: np.ndarray,
    row_regions: np.ndarray,
    col_regions: np.ndarray,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap CIs + permutation stats at the region level."""
    rng = np.random.default_rng(seed)

    regions = np.unique(np.concatenate([row_regions, col_regions]))
    logger.info("Regions=%d", len(regions))

    obs_out = {r: mean_positive(M[row_regions == r, :]) for r in regions}
    obs_in = {r: mean_positive(M[:, col_regions == r]) for r in regions}

    # Label-permutation null: shuffle region labels on rows/columns and
    # recompute means under the H0 that region identity is irrelevant.
    logger.info("Running %d permutations", n_perm)
    null_out = {r: np.zeros(n_perm) for r in regions}
    null_in = {r: np.zeros(n_perm) for r in regions}
    for p in range(n_perm):
        if (p + 1) % 200 == 0 or (p + 1) == n_perm:
            logger.info("  permutation %d/%d", p + 1, n_perm)
        perm_row = rng.permutation(row_regions)
        perm_col = rng.permutation(col_regions)
        for r in regions:
            null_out[r][p] = mean_positive(M[perm_row == r, :])
            null_in[r][p] = mean_positive(M[:, perm_col == r])

    logger.info("Computing bootstrap CIs")
    rows: List[dict] = []
    for r in regions:
        om, ol, oh = bootstrap_ci(M[row_regions == r, :], rng=rng, n_boot=n_boot)
        im, il, ih = bootstrap_ci(M[:, col_regions == r], rng=rng, n_boot=n_boot)
        rows.append({
            "region": r,
            "out_mean": om, "out_lo": ol, "out_hi": oh,
            "p_out": perm_pvalue(null_out[r], obs_out[r], n_perm),
            "in_mean": im, "in_lo": il, "in_hi": ih,
            "p_in": perm_pvalue(null_in[r], obs_in[r], n_perm),
        })
    region_df = pd.DataFrame(rows)
    region_df["q_out"] = fdr_bh(region_df["p_out"].to_numpy())
    region_df["q_in"] = fdr_bh(region_df["p_in"].to_numpy())

    n_sig_out = int((region_df["q_out"] < 0.05).sum())
    n_sig_in = int((region_df["q_in"] < 0.05).sum())
    logger.info(
        "Significant regions (FDR<0.05): %d outgoing, %d incoming (of %d)",
        n_sig_out, n_sig_in, len(regions),
    )
    return region_df


def top_region_pairs(
    M: np.ndarray,
    row_regions: np.ndarray,
    col_regions: np.ndarray,
    *,
    topk: int,
) -> pd.DataFrame:
    """Strongest directed region->region pairs (self-pairs excluded)."""
    regions = np.unique(np.concatenate([row_regions, col_regions]))
    rows: List[dict] = []
    for ri in regions:
        for rj in regions:
            if ri == rj:
                continue
            m = mean_positive(M[np.ix_(row_regions == ri, col_regions == rj)])
            if m > 0:
                rows.append({
                    "source_region": ri,
                    "target_region": rj,
                    "mean_conn": m,
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
    setup_logging("top_regions", output_dir=out_dir)
    logger.info("Summed matrix: %s", args.summed_csv)
    logger.info("Output dir: %s", out_dir)

    row_labels, col_labels, M = load_summed_matrix(args.summed_csv)
    row_regions = regions_of(row_labels)
    col_regions = regions_of(col_labels)

    region_df = compute_stats(
        M, row_regions, col_regions,
        n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
    )
    pairs_df = top_region_pairs(M, row_regions, col_regions, topk=args.topk)

    region_path = out_dir / "region_stats.csv"
    pairs_path = out_dir / "top_region_pairs.csv"
    region_df.to_csv(region_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    logger.info("Wrote %s, %s", region_path, pairs_path)

    top_out = region_df.sort_values("out_mean", ascending=False).head(args.topk)
    top_in = region_df.sort_values("in_mean", ascending=False).head(args.topk)
    print("\n" + "=" * 80)
    print(f"Top {args.topk} sending regions (mean outgoing strength)")
    print("=" * 80)
    print(top_out[["region", "out_mean", "out_lo", "out_hi", "q_out"]].to_string(index=False))
    print("\n" + "=" * 80)
    print(f"Top {args.topk} receiving regions (mean incoming strength)")
    print("=" * 80)
    print(top_in[["region", "in_mean", "in_lo", "in_hi", "q_in"]].to_string(index=False))
    print(f"\nWritten to: {region_path}, {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
