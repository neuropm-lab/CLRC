"""Post-training fold evaluation: candidate collection, unique-edge assignment, metrics.

Each LOBO fold produces predictions for edges that touch the held-out region.
Most edges appear in exactly two folds (once for the src-region holdout, once
for the tgt-region holdout). To build an edge-unique prediction table suitable
for parity plots, R² computation, and fold-level importance weighting, we
assign each edge to a single fold using a deterministic rule:

    1. Prefer the fold whose holdout region equals the edge's ``src_region``.
    2. Otherwise prefer the fold whose holdout equals the ``tgt_region``.
    3. Otherwise fall back to the first candidate in iteration order.

The resulting ``fold_weights`` (the number of edges assigned to each fold)
are used downstream by ``clrc.prediction.importance`` to compute weighted
averages of per-fold feature gain.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from clrc.core.types import FoldArtifact

logger = logging.getLogger(__name__)

_REQUIRED_EDGE_COLS = ("edge_idx", "src_region", "tgt_region")


def collect_candidates(fold: FoldArtifact) -> pd.DataFrame:
    """Convert a single fold's predictions into a candidate-edge table.

    Returns a DataFrame with columns:
    ``edge_idx, src_region, tgt_region, holdout_region, y_true_ecdf, y_pred_ecdf``.
    """
    edge_table = fold.edge_table_test
    if edge_table is None:
        raise ValueError(
            f"Fold {fold.holdout_region!r} missing edge_table_test; "
            "re-run training with include_edge_tables=True."
        )
    missing = set(_REQUIRED_EDGE_COLS) - set(edge_table.columns)
    if missing:
        raise ValueError(
            f"edge_table_test for fold {fold.holdout_region!r} missing columns: {missing}"
        )

    y_true = np.asarray(fold.y_test_t, dtype=float).ravel()
    y_pred = np.asarray(fold.y_pred, dtype=float).ravel()
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != len(edge_table):
        raise ValueError(
            f"Fold {fold.holdout_region!r}: length mismatch "
            f"(y_true={y_true.shape[0]}, y_pred={y_pred.shape[0]}, "
            f"edges={len(edge_table)})"
        )

    return pd.DataFrame(
        {
            "edge_idx": edge_table["edge_idx"].to_numpy(),
            "src_region": edge_table["src_region"].astype(str).to_numpy(),
            "tgt_region": edge_table["tgt_region"].astype(str).to_numpy(),
            "holdout_region": str(fold.holdout_region),
            "y_true_ecdf": y_true,
            "y_pred_ecdf": y_pred,
        }
    )


def collect_all_candidates(folds: Iterable[FoldArtifact]) -> pd.DataFrame:
    """Concatenate candidate tables across all folds."""
    return pd.concat(
        [collect_candidates(f) for f in folds], ignore_index=True, copy=False
    )


def assign_unique_edges(
    candidates: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Assign each edge to a single fold and compute per-fold edge weights.

    Vectorized implementation: no per-edge Python loops. For each edge index,
    we pick the best candidate by priority
    (src_match > tgt_match > first-seen) via groupby + idxmax on a priority key.

    Returns
    -------
    unique_df
        DataFrame with one row per edge, sorted by ``edge_idx``, with an
        ``assigned_holdout_region`` column.
    fold_weights
        Mapping ``holdout_region -> number of edges assigned to that fold``.
    """
    if candidates.empty:
        return candidates.copy(), {}

    df = candidates.copy()
    df["_seen_idx"] = np.arange(len(df))
    src_match = (df["holdout_region"] == df["src_region"]).to_numpy()
    tgt_match = (df["holdout_region"] == df["tgt_region"]).to_numpy()

    # Priority: higher is better. src_match=2, tgt_match=1, neither=0.
    # Break ties with smallest _seen_idx (first-seen).
    priority = np.where(src_match, 2, np.where(tgt_match, 1, 0))
    df["_priority"] = priority
    df["_tiebreak"] = -df["_seen_idx"].to_numpy()  # prefer smaller index → larger -idx

    # pick argmax of (priority, tiebreak) per edge_idx
    sort_order = np.lexsort(
        (df["_tiebreak"].to_numpy(), df["_priority"].to_numpy())
    )
    df_sorted = df.iloc[sort_order]
    # Last row per edge_idx (after stable sort) is the best one
    best = df_sorted.drop_duplicates(subset="edge_idx", keep="last")
    best = best.sort_values("edge_idx").reset_index(drop=True)
    best = best.rename(columns={"holdout_region": "assigned_holdout_region"})
    best = best.drop(columns=["_seen_idx", "_priority", "_tiebreak"])
    best["holdout_region"] = best["assigned_holdout_region"]

    fold_weights = (
        best["assigned_holdout_region"].value_counts().astype(int).to_dict()
    )
    return best, fold_weights


def summarize_candidates(candidates: pd.DataFrame) -> Dict[str, int]:
    """Report how many edges appeared in one / two / more folds."""
    counts = candidates.groupby("edge_idx").size()
    return {
        "n_candidates_total": int(len(candidates)),
        "n_edges_with_one_candidate": int((counts == 1).sum()),
        "n_edges_with_two_candidates": int((counts == 2).sum()),
        "n_edges_with_more_candidates": int((counts > 2).sum()),
        "n_unique_edges": int(counts.shape[0]),
    }


def compute_metrics(unique_df: pd.DataFrame) -> Dict[str, float]:
    """Compute R², Spearman ρ, RMSE, MAE on the edge-unique prediction table."""
    y_true = unique_df["y_true_ecdf"].to_numpy()
    y_pred = unique_df["y_pred_ecdf"].to_numpy()
    rho, p = spearmanr(y_true, y_pred)
    return {
        "R2_ecdf": float(r2_score(y_true, y_pred)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": int(len(unique_df)),
    }
