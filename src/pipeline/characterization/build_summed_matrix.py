#!/usr/bin/env python3
"""Sum the per-ligand-receptor NeuronChat matrices into one node-by-node matrix.

NeuronChat writes one communication matrix per ligand-receptor interaction
(1092 for the expanded DB) into an HDF5 file, each of shape ``(n_node, n_node)``
over ``"<region>::<cell type>"`` nodes. This driver sums every interaction
matrix in the requested group into a single ``(n_node, n_node)`` matrix -- the
"summed network" consumed by ``top_regions.py``, ``top_cells.py`` and
``distance_decay.py`` -- and writes it as a labelled CSV (sender rows, receiver
columns).

Groups (HDF5 top-level group holding one dataset per interaction):
  - ``net``    FDR-filtered communication probabilities (default)
  - ``net0``   raw (pre-FDR) probabilities
  - ``pvalue`` per-interaction permutation p-values

Outputs:
  - ``<out-dir>/summed_net.csv`` -- ``(n_node, n_node)`` matrix indexed and
    headed by node label.

Usage::

    uv run python src/pipeline/characterization/build_summed_matrix.py \\
        result_gpu.h5 --group net --out-dir out/characterization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import h5py
import numpy as np
import pandas as pd

from clrc.core.logging import setup_logging

from _common import resolve_output_dir

logger = logging.getLogger("clrc.pipeline.characterization.build_summed_matrix")


def _decode(x: object) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def sum_interaction_matrices(h5_path: Path, group: str) -> pd.DataFrame:
    """Sum every interaction matrix in *group* into one labelled DataFrame."""
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        sender_names = [_decode(s) for s in attrs["sender_names"]]
        receiver_names = [_decode(s) for s in attrs["receiver_names"]]
        interaction_names = [_decode(s) for s in attrs["interaction_names"]]

        n_sender = len(sender_names)
        n_receiver = len(receiver_names)
        n_inter = len(interaction_names)
        logger.info(
            "Senders=%d, Receivers=%d, Interactions=%d; summing group '%s'",
            n_sender, n_receiver, n_inter, group,
        )

        if group not in f:
            raise KeyError(
                f"Group '{group}' not found in {h5_path}. "
                f"Available: {sorted(f.keys())}"
            )
        grp = f[group]

        summed = np.zeros((n_sender, n_receiver), dtype=np.float64)
        for i, name in enumerate(interaction_names):
            summed += grp[name][:]
            if (i + 1) % 100 == 0 or (i + 1) == n_inter:
                logger.info("  summed %d/%d interactions", i + 1, n_inter)

    n_pos = int(np.sum(np.isfinite(summed) & (summed > 0)))
    logger.info(
        "Summed matrix: %d positive entries (%.2f%% density)",
        n_pos, 100.0 * n_pos / summed.size,
    )
    return pd.DataFrame(summed, index=sender_names, columns=receiver_names)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_h5", type=Path, help="Path to NeuronChat result HDF5 (e.g. result_gpu.h5).")
    p.add_argument(
        "--group", default="net", choices=["net", "net0", "pvalue"],
        help="HDF5 group to sum (default 'net' = FDR-filtered).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("out/characterization"),
        help="Output directory (relative paths anchored at the repo root).",
    )
    p.add_argument(
        "--out-name", default="summed_net.csv",
        help="Output CSV filename (default 'summed_net.csv').",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = resolve_output_dir(args.out_dir)
    setup_logging("build_summed_matrix", output_dir=out_dir)
    logger.info("Input HDF5: %s", args.input_h5)
    logger.info("Output dir: %s", out_dir)

    df = sum_interaction_matrices(args.input_h5, args.group)

    out_path = out_dir / args.out_name
    df.to_csv(out_path)
    logger.info("Wrote summed matrix -> %s", out_path)

    print(f"\nSummed {df.shape[0]}x{df.shape[1]} network from group '{args.group}'")
    print(f"Written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
