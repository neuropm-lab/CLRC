#!/usr/bin/env python3
"""Rank ligand-receptor interactions by mean realized connectivity.

Operates directly on the per-interaction ``(n_node, n_node)`` matrices stored in
the NeuronChat result HDF5 (not the summed matrix). For every interaction the
mean over positive entries is the observed statistic; the top-K interactions
then receive a percentile bootstrap 95% CI and a label-permutation p-value
(Benjamini-Hochberg corrected), computed only when at least 10 positive entries
exist. Interactions with fewer than 10 positive entries are ranked but left
untested (``sig = "n<10"``).

Outputs:
  - ``<out-dir>/top_lr_interactions.csv`` -- one row per top-K interaction with
    rank, n_nonzero, observed mean, bootstrap CI, permutation p and q, and a
    significance annotation.

Usage::

    uv run python src/pipeline/characterization/top_lr_interactions.py \\
        result_gpu.h5 --group net --topk 20 --out-dir out/characterization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import pandas as pd

from clrc.core.logging import setup_logging

from _common import (
    bootstrap_ci,
    count_positive,
    fdr_bh,
    mean_positive,
    perm_pvalue,
    resolve_output_dir,
    sig_star,
)

logger = logging.getLogger("clrc.pipeline.characterization.top_lr_interactions")

MIN_TESTABLE_N = 10        # need >= this many positive entries to test
MAX_BOOT_N = 10_000        # subsample cap to bound bootstrap memory


def _decode(x: object) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def rank_interactions(
    h5_path: Path,
    *,
    group: str,
    topk: int,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    """Rank interactions by mean connectivity; test the top-K."""
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        interaction_names = [_decode(s) for s in attrs["interaction_names"]]
        n_inter = len(interaction_names)
        logger.info("Interactions=%d; ranking group '%s'", n_inter, group)
        if group not in f:
            raise KeyError(
                f"Group '{group}' not found in {h5_path}. Available: {sorted(f.keys())}"
            )
        grp = f[group]

        # Pass 1: observed mean + positive count per interaction. Retain the
        # matrices in memory so the top-K bootstrap/permutation avoids reload.
        logger.info("Pass 1/3 — observed means for all interactions")
        obs_mean = np.zeros(n_inter)
        obs_n = np.zeros(n_inter, dtype=int)
        mats: Dict[str, np.ndarray] = {}
        for i, name in enumerate(interaction_names):
            mat = grp[name][:]
            mats[name] = mat
            obs_mean[i] = mean_positive(mat)
            obs_n[i] = count_positive(mat)
            if (i + 1) % 200 == 0 or (i + 1) == n_inter:
                logger.info("  processed %d/%d", i + 1, n_inter)

    top_idx = np.argsort(obs_mean)[::-1][:topk]
    top_names = [interaction_names[i] for i in top_idx]
    top_mean = obs_mean[top_idx]
    top_n = obs_n[top_idx]
    logger.info("Selected top %d interactions by observed mean", topk)

    # Pass 2: bootstrap CIs (top-K, only when testable).
    logger.info("Pass 2/3 — bootstrap CIs")
    boot_mean = np.zeros(topk)
    boot_lo = np.zeros(topk)
    boot_hi = np.zeros(topk)
    for j, name in enumerate(top_names):
        boot_mean[j], boot_lo[j], boot_hi[j] = bootstrap_ci(
            mats[name].ravel(), rng=rng, n_boot=n_boot,
            min_n=MIN_TESTABLE_N, max_n=MAX_BOOT_N,
        )

    # Pass 3: label-permutation null (top-K, only when testable). Shuffling the
    # flattened matrix scrambles which node-pair holds which value under the H0
    # that the interaction's node structure is irrelevant.
    logger.info("Pass 3/3 — %d permutations for top %d interactions", n_perm, topk)
    null_dist = {name: np.zeros(n_perm) for name in top_names}
    for p in range(n_perm):
        if (p + 1) % 200 == 0 or (p + 1) == n_perm:
            logger.info("  permutation %d/%d", p + 1, n_perm)
        for j, name in enumerate(top_names):
            if top_n[j] < MIN_TESTABLE_N:
                continue
            flat = mats[name].ravel()
            null_dist[name][p] = mean_positive(rng.permutation(flat))

    pvals: List[float] = []
    for j, name in enumerate(top_names):
        if top_n[j] < MIN_TESTABLE_N:
            pvals.append(np.nan)
        else:
            pvals.append(perm_pvalue(null_dist[name], top_mean[j], n_perm))

    pvals_arr = np.asarray(pvals, dtype=float)
    tested = ~np.isnan(pvals_arr)
    qvals = np.full(topk, np.nan)
    if tested.any():
        qvals[tested] = fdr_bh(pvals_arr[tested])

    rows = []
    for j, name in enumerate(top_names):
        q = qvals[j]
        rows.append({
            "rank": j + 1,
            "interaction": name,
            "n_nonzero": int(top_n[j]),
            "obs_mean": top_mean[j],
            "boot_mean": boot_mean[j],
            "boot_lo": boot_lo[j],
            "boot_hi": boot_hi[j],
            "pval": pvals[j],
            "qval": q,
            "sig": sig_star(q) if not np.isnan(q) else "n<10",
        })
    return pd.DataFrame(rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_h5", type=Path, help="Path to NeuronChat result HDF5 (e.g. result_gpu.h5).")
    p.add_argument("--group", default="net", choices=["net", "net0"],
                   help="HDF5 group: 'net' = FDR-filtered (default), 'net0' = raw.")
    p.add_argument("--topk", type=int, default=20, help="Number of top interactions (default 20).")
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations (default 1000).")
    p.add_argument("--n-perm", type=int, default=1000, help="Permutations (default 1000).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    p.add_argument("--out-dir", type=Path, default=Path("out/characterization"),
                   help="Output directory (relative paths anchored at the repo root).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = resolve_output_dir(args.out_dir)
    setup_logging("top_lr_interactions", output_dir=out_dir)
    logger.info("Input HDF5: %s", args.input_h5)
    logger.info("Output dir: %s", out_dir)

    result = rank_interactions(
        args.input_h5, group=args.group, topk=args.topk,
        n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
    )

    out_path = out_dir / "top_lr_interactions.csv"
    result.to_csv(out_path, index=False)
    logger.info("Wrote ranking -> %s", out_path)

    n_sig = int((result["sig"].isin(["*", "**", "***"])).sum())
    print("\n" + "=" * 80)
    print(f"Top {args.topk} ligand-receptor interactions by mean connectivity "
          f"({n_sig} significant at q<0.05)")
    print("=" * 80)
    print(result[["rank", "interaction", "n_nonzero", "obs_mean",
                  "boot_lo", "boot_hi", "qval", "sig"]].to_string(index=False))
    print(f"\nWritten to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
