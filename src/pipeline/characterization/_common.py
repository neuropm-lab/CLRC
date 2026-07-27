"""Shared helpers for the characterization drivers.

Factored out of the per-driver copies in ``top_regions.py``, ``top_cells.py``
and ``top_lr_interactions.py`` so all three share one implementation of the
statistics (positive-value means, percentile bootstrap CIs, Benjamini-Hochberg
FDR) and the summed-matrix loading / label parsing.

All statistics operate on the *positive, finite* entries of the supplied
arrays: the summed NeuronChat matrices are sparse and dominated by exact zeros
(no inferred communication), which are excluded from every mean so the
summaries describe realized edges only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from clrc.core.io import find_repo_root

logger = logging.getLogger("clrc.pipeline.characterization.common")

LABEL_SEP = "::"


# ---------------------------------------------------------------------------
#  Output directory resolution
# ---------------------------------------------------------------------------

def resolve_output_dir(out_dir: Path) -> Path:
    """Resolve *out_dir* (relative paths are anchored at the repo root) and mkdir.

    Mirrors how the config-driven drivers anchor ``output.base_dir`` at the
    repo root, but without requiring a YAML config. Falls back to resolving
    against the current working directory if the repo root cannot be found.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        try:
            out_dir = find_repo_root() / out_dir
        except FileNotFoundError:
            out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
#  Positive-value statistics
# ---------------------------------------------------------------------------

def positive_values(arr: np.ndarray) -> np.ndarray:
    """Flatten *arr* and keep only finite, strictly-positive entries."""
    v = np.asarray(arr).ravel()
    return v[np.isfinite(v) & (v > 0)]


def mean_positive(arr: np.ndarray) -> float:
    """Mean over the positive, finite entries of *arr* (0.0 if none)."""
    v = positive_values(arr)
    return float(v.mean()) if v.size else 0.0


def count_positive(arr: np.ndarray) -> int:
    """Number of positive, finite entries in *arr*."""
    return int(positive_values(arr).size)


def bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 1000,
    min_n: int = 1,
    max_n: int | None = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap 95% CI of the mean over positive values.

    Returns ``(mean, lo, hi)``. With no positive values it returns
    ``(0.0, 0.0, 0.0)``; with fewer than ``min_n`` the CI collapses to the
    point estimate (``lo == hi == mean``). When more than ``max_n`` values are
    present a random subsample of that size is drawn first so memory stays
    bounded — the mean estimate stays unbiased.
    """
    v = positive_values(values)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    if v.size < min_n:
        m = float(v.mean())
        return m, m, m
    if max_n is not None and v.size > max_n:
        v = rng.choice(v, max_n, replace=False)
    boots = rng.choice(v, (n_boot, v.size), replace=True).mean(axis=1)
    return (
        float(boots.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def fdr_bh(pvals: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (monotone, clipped to [0, 1])."""
    p = np.asarray(list(pvals), dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def perm_pvalue(null: np.ndarray, observed: float, n_perm: int) -> float:
    """One-sided permutation p (fraction of null >= observed) with add-one smoothing."""
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def sig_star(q: float) -> str:
    """Significance annotation for a q-value."""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
#  Summed-matrix loading and label parsing
# ---------------------------------------------------------------------------

def load_summed_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the summed matrix CSV -> ``(row_labels, col_labels, M)``.

    The first column holds sender node labels; the remaining columns are the
    receiver nodes. Node labels are ``"<region>::<cell type>"``. ``M`` is a
    float ``(n_sender, n_receiver)`` ndarray.
    """
    path = Path(path)
    df = pd.read_csv(path)
    row_labels = df.iloc[:, 0].astype(str).to_numpy()
    col_labels = df.columns[1:].astype(str).to_numpy()
    M = df.iloc[:, 1:].to_numpy(dtype=float)
    logger.info(
        "Loaded summed matrix %s: %d senders x %d receivers",
        path, M.shape[0], M.shape[1],
    )
    return row_labels, col_labels, M


def parse_region(label: str) -> str:
    return label.split(LABEL_SEP, 1)[0].strip()


def parse_celltype(label: str) -> str:
    parts = label.split(LABEL_SEP, 1)
    return parts[1].strip() if len(parts) == 2 else ""


def regions_of(labels: Iterable[str]) -> np.ndarray:
    return np.array([parse_region(x) for x in labels])


def celltypes_of(labels: Iterable[str]) -> np.ndarray:
    return np.array([parse_celltype(x) for x in labels])


_EXCITATORY = (
    "excitatory", "intratelencephalic", "corticothalamic",
    "near-projecting", "l2", "l3", "l4", "l5", "l6",
)
_INHIBITORY = ("inhibitory", "interneuron", "mge", "cge", "lamp5", "lhx6")
_GLIA = ("astro", "oligo", "opc", "micro", "ependymal", "bergmann", "choroid")


def cell_class(ct: str) -> str:
    """Coarse cell class: Excitatory / Inhibitory / Glia / Other."""
    c = ct.lower()
    if any(k in c for k in _EXCITATORY):
        return "Excitatory"
    if any(k in c for k in _INHIBITORY):
        return "Inhibitory"
    if any(k in c for k in _GLIA):
        return "Glia"
    return "Other"
